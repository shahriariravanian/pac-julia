# RustiCol's recurrence export (`pyamplicol export-recurrence LC_DIR -o FILE`, pyAmpliCol 1.0.x;
# schema 2, execution semantics rusticol-batched-f64-v1, pyAmpliCol's
# docs/reference/recurrence-export.md) read into Lowered programs, row for row: no Program and no
# pac2json in between. `load_recurrence_export(path)` gives one Lowered per program entry, in the
# file's order (a process-group member or alias: its representative's rows with its own crossing,
# group factor and helicity labels); `recurrence_references(path)` the reference values the file
# holds (`--reference-points`).
#
# The mapping (export → UIR v0.8):
#   currents       the producer's layout is kept (fixed_layout): a Lowered current is a range of
#                  the export's component offsets, (offset, components, type), made for the rows
#                  that address it; ranges of different currents may overlap, as the producer's
#                  storage reuse does. Its type is the one its row reads it as (vertex, propagator,
#                  plain source), else UNTYPED.
#   edges          (current, form); a current's own edge has the first form a row reads it with.
#   values         couplings, masses and kernel literals interned into `scales`; every other
#                  complex value (row factors, closure component factors, embedding factors, replay
#                  amplitude scales) into `factors`; bit for bit (-0.0 and 0.0 differ).
#   runs           concatenated: each run's body (its sources from its own table), then its
#                  reduction as REDUCE rows (the replay scale, when applied, as REDUCE's factor).
#   kernels        each binding a stub (its literals as scale slots), its relative projections the
#                  stub's inputs (parent 0/1 → IN_L/IN_R, destination → OUT, momentum 0/1 →
#                  MOMENTUM_U/V, factor → FACTOR); the saved applications decoded into the scratch
#                  space ("export-kernels", by SHA-256) and built as kernel_lib.jl builds pac2json's.
# Unsupported (an error): spin-2 sources, kernels with an amplitude destination or input, outputs
# that are not (re, im) pairs of one component.

const EXPORT_SCHEMA = 2
const EXPORT_SEMANTICS = "rusticol-batched-f64-v1"

# RustiCol's versioned intrinsic names → the operations' names (validate.jl's VERTEX_OPS, PROPAGATOR_OPS)
const EXPORT_VERTICES = Dict(
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1" => "weyl_vector_a",
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1" => "weyl_vector_b",
    "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1" => "tensor_vector",
    "rusticol.recurrence-intrinsic.vector-wedge-vector.v1" => "vector_wedge",
    "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1" => "three_vector",
)
const EXPORT_PROPAGATORS = Dict(
    "rusticol.recurrence-intrinsic.weyl-propagator-a.v1" => "weyl_a",
    "rusticol.recurrence-intrinsic.weyl-propagator-b.v1" => "weyl_b",
    "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1" => "vector",
)

struct ExportError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ExportError) = print(io, "recurrence export: ", e.msg)
export_error(msg) = throw(ExportError(msg))

export_kernels_dir() = @get_scratch!("export-kernels")

f64(x) = Float64(x)
cvalue(z) = complex(f64(z["re"]), f64(z["im"]))
index0(x) = x === nothing ? nothing : Int(x)    # a 0-based index or null, as read

# the file, checked: its schema and semantics
function read_export(path::AbstractString)
    d = JSON.parse(read(path, String))
    get(d, "schema_version", nothing) == EXPORT_SCHEMA ||
        export_error("$path: schema_version $(get(d, "schema_version", nothing)), this reader takes $EXPORT_SCHEMA")
    get(d, "execution_semantics", nothing) == EXPORT_SEMANTICS ||
        export_error("$path: execution_semantics $(get(d, "execution_semantics", nothing)), not $EXPORT_SEMANTICS")
    return d
end

# the kernels of the file, by SHA-256: each saved application decoded (and checked against its
# SHA-256) into the scratch space, and its library for the Julia runner (`kernels`: kernel_lib.jl's
# ways); a library's symbol is KERNEL_SYMBOL, its name in the C "k_" and the first 16 hex digits
function export_kernels(d; kernels::Symbol = :c)::Dict{String,KernelLibrary}
    kernels in KERNEL_WAYS || error("kernels = :$kernels: one of $(join(KERNEL_WAYS, ", "))")
    way = resolve_way(kernels)
    out = Dict{String,KernelLibrary}()
    for k in d["kernels"]
        sha = String(k["sha256"])
        bytes = base64decode(k["application_base64"])
        bytes2hex(sha256(bytes)) == sha || export_error("kernel $sha: its application has another SHA-256")
        source = joinpath(export_kernels_dir(), "$sha.plane.symjit")
        if !isfile(source)
            tmp = tempname(export_kernels_dir())
            write(tmp, bytes)
            mv(tmp, source; force = true)
        end
        app = read_sjb(source)
        (inputs, outputs) = (app.count_states, app.count_obs)
        name = "k_" * sha[1:16]
        out[sha] = if way == :none
            load_kernel("", name, inputs, outputs; source)
        else
            built = kernel_library(source, sha, inputs, outputs, way)
            load_kernel(something(built, ""), KERNEL_SYMBOL, inputs, outputs; source, name)
        end
    end
    return out
end

# ---- one program entry → Lowered ---------------------------------------------------------------

# a kernel binding of the file: the stub's inputs (literals as slots of `scaleset`), its output
# components, the kernel, and whether it stores its pairs directly
struct ExportBinding
    inputs::Vector{KernelInput}
    scaleset::Vector{ScaleIndex}
    outputs::Vector{ComponentIndex}
    kernel::KernelLibrary
    direct::Bool
end

# What import_program collects while it reads an entry: the Lowered's tables, and the rows with
# their d, l, r still edge requests (an index into `requests`), resolved to edges at the end
# (resolved_ops) when each current's own form is known.
struct ExportBuilder
    scales::Vector{Complex{Float64}}
    factors::Vector{Complex{Float64}}
    layouts::Vector{Tuple{Int,Int,PhysicalType}}                        # the currents: (offset, components, type)
    owns::Vector{FormIndex}                                             # each current's own form (0: none yet)
    requests::Vector{Tuple{CurrentIndex,Union{Nothing,FormIndex}}}      # (current, form or nothing: any edge of it)
    lists::Vector{Vector{FactorIndex}}
    embeddings::Vector{Vector{EmbeddingEntry}}
    stubs::Vector{LStub}
    scalesets::Vector{Vector{ScaleIndex}}
    instances::Vector{Tuple{StubIndex,ScaleSetIndex}}
    weights::Vector{Float64}
    bindings::Dict{Int,ExportBinding}                                   # by the file's id
    rows::Vector{LOp}
end

ExportBuilder() = ExportBuilder(Complex{Float64}[], Complex{Float64}[], Tuple{Int,Int,PhysicalType}[], FormIndex[],
                                Tuple{CurrentIndex,Union{Nothing,FormIndex}}[], Vector{FactorIndex}[],
                                Vector{EmbeddingEntry}[], LStub[], Vector{ScaleIndex}[], Tuple{StubIndex,ScaleSetIndex}[],
                                Float64[], Dict{Int,ExportBinding}(), LOp[])

scale!(b::ExportBuilder, z)::ScaleIndex = intern_value!(b.scales, z)
factor!(b::ExportBuilder, z)::FactorIndex = intern_value!(b.factors, z)
row!(b::ExportBuilder, op::LOp) = push!(b.rows, op)

# a request for an edge of the current at (offset, components, type), with `form` when the row
# reads its momentum; the first form asked of a current becomes its own
function want!(b::ExportBuilder, offset, components, ty::PhysicalType, form = nothing)::Int
    c = intern!(b.layouts, (Int(offset), Int(components), ty))
    c > length(b.owns) && push!(b.owns, 0)
    form === nothing || b.owns[c] != 0 || (b.owns[c] = form)
    push!(b.requests, (c, form))
    return length(b.requests)
end

form_of(x)::FormIndex = x === nothing ? export_error("an operand without a momentum form") : Int(x) + 1
set_mode(m)::Bool = m == "set" ? true : m == "add" ? false : export_error("mode $m")

# one input of a binding; a literal takes the next slot of `scaleset`
function binding_input!(b::ExportBuilder, scaleset::Vector{ScaleIndex}, i, id)::KernelInput
    kind = i["kind"]
    if kind == "parent"
        return KernelInput(Int(i["parent"]) == 0 ? IN_L : IN_R, Int(i["component"]) + 1, i["imaginary"])
    elseif kind == "destination"
        i["amplitude"] && export_error("binding $id: an amplitude input")
        return KernelInput(OUT, Int(i["component"]) + 1, i["imaginary"])
    elseif kind == "momentum"
        return KernelInput(Int(i["operand"]) == 0 ? MOMENTUM_U : MOMENTUM_V, Int(i["component"]) + 1, false)
    elseif kind == "literal"
        push!(scaleset, scale!(b, complex(f64(i["value"]), 0.0)))
        return KernelInput(SCALE, length(scaleset), false)
    elseif kind == "factor"
        return KernelInput(FACTOR, 0, i["imaginary"])
    end
    export_error("binding $id: input kind $kind")
end

# the output components of a binding: its outputs are (re, im) pairs of one destination component each
function binding_outputs(o, id)::Vector{ComponentIndex}
    iseven(length(o)) || export_error("binding $id: $(length(o)) outputs")
    outputs = ComponentIndex[]
    for j in 1:2:length(o)
        (re, im_) = (o[j], o[j+1])
        all(x -> x["kind"] == "destination" && !x["amplitude"], (re, im_)) && re["component"] == im_["component"] &&
            !re["imaginary"] && im_["imaginary"] || export_error("binding $id: outputs $j, $(j + 1) are not one component's (re, im)")
        push!(outputs, Int(re["component"]) + 1)
    end
    return outputs
end

function import_binding!(b::ExportBuilder, x, libraries::Dict{String,KernelLibrary})
    id, sha = x["id"], String(x["sha256"])
    haskey(libraries, sha) || export_error("binding $id: no kernel $sha in the file")
    scaleset = ScaleIndex[]
    inputs = KernelInput[binding_input!(b, scaleset, i, id) for i in x["inputs"]]
    b.bindings[Int(id)] = ExportBinding(inputs, scaleset, binding_outputs(x["outputs"], id), libraries[sha], Bool(x["direct"]))
end

# the binding instance (stub, scale set) of a binding, both interned
instance!(b::ExportBuilder, x::ExportBinding)::InstanceIndex =
    intern_instance!(k -> k.path == x.kernel.path && k.source == x.kernel.source, b.stubs, b.scalesets, b.instances,
                     x.kernel, x.inputs, x.outputs, x.scaleset)

# ---- the rows of the body, one function per instruction kind

function import_clear!(b::ExportBuilder, i)
    n = Int(i["component_count"])
    row!(b, LOp(CLEAR; d = want!(b, i["current"], n, UNTYPED), n = n))
end

import_clear_amplitude!(b::ExportBuilder, i) = row!(b, LOp(CLEAR_AMPLITUDE; d = Int(i["amplitude"]) + 1))

# `s`: the run's source the instruction names
function import_source!(b::ExportBuilder, s)
    family = s["family"]
    haskey(FAMILY_TYPES, family) || export_error("source family $family")
    ty = FAMILY_TYPES[family]
    # component_count is the wavefunction's; an embedding writes the components it lists
    # (union sources: a Weyl spinor into a 4-component slot, the rest +0)
    n = Int(s["component_count"])
    n == TYPE_COMPONENTS[ty] || export_error("a $family source of $n components")
    embedded = !isempty(s["embedding"])
    width = embedded ? maximum(Int(e["component"]) for e in s["embedding"]) + 1 : n
    d = want!(b, s["current"], width, embedded ? UNTYPED : ty, form_of(s["momentum_form"]))
    emb = 0
    if embedded
        push!(b.embeddings, [(Int(e["component"]) + 1, e["source_component"] === nothing ? 0 : Int(e["source_component"]) + 1,
                              factor!(b, cvalue(e["factor"]))) for e in s["embedding"]])
        emb = length(b.embeddings)
    end
    row!(b, source_op(d, ty, s["orientation"] == "antiparticle", Int(s["helicity"]), Int(s["chirality"]),
                      scale!(b, cvalue(s["mass"])), factor!(b, cvalue(s["factor"])); embedding = emb))
end

function import_vertex!(b::ExportBuilder, i)
    name = get(EXPORT_VERTICES, i["vertex"], nothing)
    name === nothing && export_error("vertex $(i["vertex"])")
    (opc, tl, tr, tout) = VERTEX_OPS[name]
    reads = opc == V_THREE_VECTOR
    i["right"] === nothing && export_error("vertex $(i["vertex"]) without a right operand")
    l = want!(b, i["left"], TYPE_COMPONENTS[tl], tl, reads ? form_of(i["left_momentum"]) : nothing)
    r = want!(b, i["right"], TYPE_COMPONENTS[tr], tr, reads ? form_of(i["right_momentum"]) : nothing)
    d = want!(b, i["current"], TYPE_COMPONENTS[tout], tout)
    row!(b, LOp(opc; set = set_mode(i["mode"]), first = true, last = true, d = d, t = scale!(b, cvalue(i["coupling"])),
                l = l, r = r, x = factor!(b, cvalue(i["factor"]))))
end

function import_propagator!(b::ExportBuilder, i)
    name = get(EXPORT_PROPAGATORS, i["propagator"], nothing)
    name === nothing && export_error("propagator $(i["propagator"])")
    (opc, ty) = PROPAGATOR_OPS[name]
    Int(i["component_count"]) == TYPE_COMPONENTS[ty] || export_error("propagator $name of $(i["component_count"]) components")
    d = want!(b, i["current"], TYPE_COMPONENTS[ty], ty, form_of(i["momentum_form"]))
    row!(b, LOp(opc; d = d, x = factor!(b, cvalue(i["factor"]))))
end

function import_scale!(b::ExportBuilder, i)
    n = Int(i["component_count"])
    row!(b, LOp(SCALE_ROW; d = want!(b, i["current"], n, UNTYPED), n = n, x = factor!(b, cvalue(i["factor"]))))
end

# a copy or a scatter (which has a mode)
function import_copy!(b::ExportBuilder, i, scatter::Bool)
    n = Int(i["component_count"])
    d = want!(b, i["current"], n, UNTYPED)
    l = want!(b, i["source"], n, UNTYPED)
    row!(b, LOp(scatter ? SCATTER : COPY; set = scatter && set_mode(i["mode"]), d = d, l = l, x = factor!(b, cvalue(i["factor"]))))
end

function import_closure!(b::ExportBuilder, i)
    cf = [factor!(b, cvalue(z)) for z in i["component_factors"]]
    n = length(cf)
    row!(b, LOp(CLOSURE; d = Int(i["amplitude"]) + 1, t = intern!(b.lists, cf), l = want!(b, i["left"], n, UNTYPED),
                r = want!(b, i["right"], n, UNTYPED), x = factor!(b, cvalue(i["factor"]))))
end

# the largest component of the given kinds a binding reads (0: none)
reach(x::ExportBinding, kinds...) = maximum((i.n for i in x.inputs if i.kind in kinds); init = 0)

# a kernel row's operand `at`, which the binding reads through the input kinds `part` (a component)
# and `momentum` (its form): its edge request, or 0 when the binding reads neither
function kernel_operand!(b::ExportBuilder, x::ExportBinding, at, part::InputKind, momentum::InputKind, form)::Int
    comps = reach(x, part)
    reads_form = any(i -> i.kind == momentum, x.inputs)
    comps == 0 && !reads_form && return 0
    at === nothing && export_error("a kernel input of a null operand")
    return want!(b, at, max(1, comps), UNTYPED, reads_form ? form_of(form) : nothing)
end

function import_kernel!(b::ExportBuilder, i)
    i["amplitude"] && export_error("a kernel with an amplitude destination")
    x = get(b.bindings, Int(i["binding"]), nothing)
    x === nothing && export_error("kernel row: no binding $(i["binding"])")
    # each operand's components: the largest one the binding reads (at least 1)
    nd = max(1, reach(x, OUT), maximum(x.outputs; init = 0))
    l = kernel_operand!(b, x, i["left"], IN_L, MOMENTUM_U, i["left_momentum"])
    r = kernel_operand!(b, x, i["right"], IN_R, MOMENTUM_V, i["right_momentum"])
    row!(b, LOp(x.direct ? KERNEL_DIRECT : KERNEL; set = set_mode(i["mode"]), d = want!(b, i["destination"], nd, UNTYPED),
                t = instance!(b, x), l = l, r = r, x = factor!(b, cvalue(i["factor"]))))
end

# one instruction of the body, with the run's sources
function import_row!(b::ExportBuilder, i, sources)
    kind = i["kind"]
    kind == "clear" ? import_clear!(b, i) :
    kind == "clear_amplitude" ? import_clear_amplitude!(b, i) :
    kind == "source" ? import_source!(b, sources[Int(i["source"])+1]) :
    kind == "vertex" ? import_vertex!(b, i) :
    kind == "propagator" ? import_propagator!(b, i) :
    kind == "scale" ? import_scale!(b, i) :
    kind == "copy" ? import_copy!(b, i, false) :
    kind == "scatter" ? import_copy!(b, i, true) :
    kind == "closure" ? import_closure!(b, i) :
    kind == "kernel" ? import_kernel!(b, i) : export_error("instruction $kind")
end

# one term of a run's reduction
function import_reduction!(b::ExportBuilder, t)
    x = t["apply_amplitude_scale"] ? factor!(b, cvalue(t["amplitude_scale"])) : 0
    row!(b, LOp(REDUCE; d = Int(t["helicity_output"]) + 1, t = Int(t["amplitude"]) + 1, x = x,
                n = intern_value!(b.weights, f64(t["weight"]))))
end

# The edges and the rows with their requests resolved: current c's own edge first (edge c), then
# the other (current, form) pairs in the order the rows ask for them.
function resolved_ops(b::ExportBuilder)::Tuple{Vector{LEdge},Vector{LOp}}
    edges = [LEdge(c, b.owns[c]) for c in eachindex(b.layouts)]
    resolve(k) = k == 0 ? 0 : ((c, f) = b.requests[k]; f === nothing || f == b.owns[c] ? c : intern!(edges, LEdge(c, f)))
    ops = LOp[]
    for o in b.rows
        d = no_destination(o) ? o.d : resolve(o.d)      # d first, then l and r: the edges' order
        (l, r) = isempty(operand_edges(o)) ? (o.l, o.r) : (resolve(o.l), resolve(o.r))
        push!(ops, LOp(o.op, o.set, o.first, o.last, d, o.t, l, r, o.n, o.x))
    end
    return (edges, ops)
end

function export_alpha_s(p)
    a = p["alpha_s"]
    a === nothing && return nothing
    power = a["squared_amplitude_power"]
    return (f64(a["value"]["re"]), power === nothing ? nothing : Int(power))
end

# The rows of one full program entry, as a Lowered (its own process, colour, crossing, group
# factor and helicities): the bindings, then per run the body (with the run's sources) and the
# run's reduction.
function import_program(p, libraries::Dict{String,KernelLibrary})::Lowered
    b = ExportBuilder()
    foreach(x -> import_binding!(b, x, libraries), p["kernel_bindings"])
    for run in p["runs"]
        foreach(i -> import_row!(b, i, run["sources"]), p["body"])
        foreach(t -> import_reduction!(b, t), run["reduction"])
    end
    (edges, ops) = resolved_ops(b)
    currents = [LCurrent(offset, n, ty) for (offset, n, ty) in b.layouts]
    legs = Int(p["external_count"])
    forms = [Form([(Int(t["leg"]) + 1, f64(t["coefficient"])) for t in f]) for f in p["momentum_forms"]]
    l = Lowered(UIR_VERSION, String(p["process_key"]), String(p["colour_id"]), legs, export_crossing(p, legs), forms, b.scales,
                b.factors, export_alpha_s(p), export_helicities(p), currents, Int(p["current_component_count"]), edges, b.lists,
                b.embeddings, b.stubs, b.scalesets, b.instances, ops, Int(p["amplitude_count"]), b.weights, export_group(p), true)
    validate(l)
    return l
end

# program leg l (1-based) → (caller leg, sign), or nothing: the identity
function export_crossing(p, legs::Int)
    x = p["crossing"]
    x === nothing && return nothing
    out = Vector{Tuple{LegIndex,Float64}}(undef, legs)
    seen = falses(legs)
    for e in x
        k = Int(e["program_leg"]) + 1
        out[k] = (Int(e["caller_leg"]) + 1, f64(e["sign"]))
        seen[k] = true
    end
    all(seen) || export_error("$(p["process_key"]): the crossing misses a leg")
    return out
end

export_helicities(p) = [Int.(h["helicities"]) for h in p["helicity_outputs"]]
export_group(p) = (Int(p["group_factor"]["numerator"]), Int(p["group_factor"]["denominator"]))

"""
    load_recurrence_export(path; kernels = :c) -> Vector{Lowered}

The programs of a RustiCol recurrence export (`pyamplicol export-recurrence`, schema 2), one
Lowered per entry of `programs` in the file's order; a member or alias entry is its
representative's rows with its own process, colour, crossing, group factor and helicities.
`kernels`: how the kernels' libraries are built for the Julia runner (kernel_lib.jl).
"""
function load_recurrence_export(path::AbstractString; kernels::Symbol = :c)::Vector{Lowered}
    d = read_export(path)
    libraries = export_kernels(d; kernels)
    # a (process, colour flow) names one program: a second entry would silently replace the first
    seen = Set{Tuple{String,String}}()
    for p in d["programs"]
        key = (String(p["process_key"]), String(p["colour_id"]))
        key in seen && export_error("$(key[1]) $(key[2]): more than one program")
        push!(seen, key)
    end
    full = Dict{Tuple{String,String},Lowered}()
    for p in d["programs"]
        haskey(p, "program_reference") && continue
        full[(String(p["process_key"]), String(p["colour_id"]))] = import_program(p, libraries)
    end
    out = Lowered[]
    for p in d["programs"]
        key = (String(p["process_key"]), String(p["colour_id"]))
        if haskey(p, "program_reference")
            r = p["program_reference"]
            rep = get(full, (String(r["process_key"]), String(r["colour_id"])), nothing)
            rep === nothing && export_error("$(key[1]) $(key[2]): no program $(r["process_key"]) $(r["colour_id"])")
            hel = export_helicities(p)
            length(hel) == output_count(rep) || export_error("$(key[1]) $(key[2]): $(length(hel)) helicities, its program $(output_count(rep))")
            push!(out, with(rep; process = key[1], colour = key[2], crossing = export_crossing(p, rep.legs),
                            group_factor = export_group(p), helicities = hel))
        else
            push!(out, full[key])
        end
    end
    return out
end

"""
    recurrence_references(path) -> Vector

The reference values of a recurrence export written with `--reference-points`, one entry per
program entry (as load_recurrence_export's order): nothing, or (momenta, values) with momenta[point]
the caller's legs as (E, px, py, pz) tuples and values[point] the outputs in helicity order.
"""
function recurrence_references(path::AbstractString)
    d = read_export(path)
    map(d["programs"]) do p
        r = get(p, "reference_values", nothing)
        r === nothing && return nothing
        Int.(length.(r["values"])) == fill(length(p["helicity_outputs"]), length(r["values"])) ||
            export_error("$(p["process_key"]): reference values not one per helicity output")
        momenta = [[Tuple(f64.(leg)) for leg in point] for point in r["momenta"]]
        values = [f64.(v) for v in r["values"]]
        (momenta, values)
    end
end
