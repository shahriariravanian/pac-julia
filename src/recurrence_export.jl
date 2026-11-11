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

# The rows of one full program entry, as a Lowered (its own process, colour, crossing, group
# factor and helicities)
function import_program(p, libraries::Dict{String,KernelLibrary})::Lowered
    scales = Complex{Float64}[]
    factors = Complex{Float64}[]
    scale(z) = intern_value!(scales, z)
    factor(z) = intern_value!(factors, z)

    # currents, keyed by (offset, components, type); the edge requests (current, form or nothing:
    # "any edge of it") as pairs, resolved to edges at the end
    keys = Tuple{Int,Int,PhysicalType}[]
    owns = FormIndex[]                       # each current's own form (0: none yet)
    pairs = Tuple{CurrentIndex,Union{Nothing,FormIndex}}[]
    function want(offset, components, ty::PhysicalType, form = nothing)::Int
        c = intern!(keys, (Int(offset), Int(components), ty))
        c > length(owns) && push!(owns, 0)
        form === nothing || owns[c] != 0 || (owns[c] = form)
        push!(pairs, (c, form))
        return length(pairs)
    end
    form_of(x) = x === nothing ? export_error("an operand without a momentum form") : Int(x) + 1

    lists = Vector{FactorIndex}[]
    embeddings = Vector{EmbeddingEntry}[]
    stubs = LStub[]
    scalesets = Vector{ScaleIndex}[]
    instances = Tuple{StubIndex,ScaleSetIndex}[]
    weights = Float64[]

    # the kernel bindings: by id, (stub inputs with scale slots, the scale set, outputs, kernel, direct)
    bindings = Dict{Int,Any}()
    for b in p["kernel_bindings"]
        sha = String(b["sha256"])
        haskey(libraries, sha) || export_error("binding $(b["id"]): no kernel $sha in the file")
        inputs = KernelInput[]
        scaleset = ScaleIndex[]
        for i in b["inputs"]
            kind = i["kind"]
            if kind == "parent"
                push!(inputs, KernelInput(Int(i["parent"]) == 0 ? IN_L : IN_R, Int(i["component"]) + 1, i["imaginary"]))
            elseif kind == "destination"
                i["amplitude"] && export_error("binding $(b["id"]): an amplitude input")
                push!(inputs, KernelInput(OUT, Int(i["component"]) + 1, i["imaginary"]))
            elseif kind == "momentum"
                push!(inputs, KernelInput(Int(i["operand"]) == 0 ? MOMENTUM_U : MOMENTUM_V, Int(i["component"]) + 1, false))
            elseif kind == "literal"
                push!(scaleset, scale(complex(f64(i["value"]), 0.0)))
                push!(inputs, KernelInput(SCALE, length(scaleset), false))
            elseif kind == "factor"
                push!(inputs, KernelInput(FACTOR, 0, i["imaginary"]))
            else
                export_error("binding $(b["id"]): input kind $kind")
            end
        end
        o = b["outputs"]
        iseven(length(o)) || export_error("binding $(b["id"]): $(length(o)) outputs")
        outputs = ComponentIndex[]
        for j in 1:2:length(o)
            (re, im_) = (o[j], o[j+1])
            all(x -> x["kind"] == "destination" && !x["amplitude"], (re, im_)) && re["component"] == im_["component"] &&
                !re["imaginary"] && im_["imaginary"] || export_error("binding $(b["id"]): outputs $j, $(j + 1) are not one component's (re, im)")
            push!(outputs, Int(re["component"]) + 1)
        end
        bindings[Int(b["id"])] = (inputs, scaleset, outputs, libraries[sha], Bool(b["direct"]))
    end
    function instance(binding)::InstanceIndex
        (inputs, scaleset, outputs, kernel, _) = binding
        k = findfirst(s -> s.kernel.path == kernel.path && s.kernel.source == kernel.source && s.inputs == inputs && s.outputs == outputs, stubs)
        k === nothing && (push!(stubs, LStub(kernel, inputs, outputs)); k = length(stubs))
        return intern!(instances, (k, intern!(scalesets, scaleset)))
    end
    set_mode(m) = m == "set" ? true : m == "add" ? false : export_error("mode $m")

    # one row of the body (pairs in d, l, r; resolved later), with the run's sources
    raw = LOp[]
    function row!(i, sources)
        kind = i["kind"]
        if kind == "clear"
            n = Int(i["component_count"])
            push!(raw, LOp(CLEAR; d = want(i["current"], n, UNTYPED), n = n))
        elseif kind == "clear_amplitude"
            push!(raw, LOp(CLEAR_AMPLITUDE; d = Int(i["amplitude"]) + 1))
        elseif kind == "source"
            s = sources[Int(i["source"])+1]
            family = s["family"]
            haskey(FAMILY_TYPES, family) || export_error("source family $family")
            ty = FAMILY_TYPES[family]
            # component_count is the wavefunction's; an embedding writes the components it lists
            # (union sources: a Weyl spinor into a 4-component slot, the rest +0)
            n = Int(s["component_count"])
            n == TYPE_COMPONENTS[ty] || export_error("a $family source of $n components")
            embedded = !isempty(s["embedding"])
            width = embedded ? maximum(Int(e["component"]) for e in s["embedding"]) + 1 : n
            d = want(s["current"], width, embedded ? UNTYPED : ty, form_of(s["momentum_form"]))
            emb = 0
            if embedded
                push!(embeddings, [(Int(e["component"]) + 1, e["source_component"] === nothing ? 0 : Int(e["source_component"]) + 1,
                                    factor(cvalue(e["factor"]))) for e in s["embedding"]])
                emb = length(embeddings)
            end
            push!(raw, source_op(d, ty, s["orientation"] == "antiparticle", Int(s["helicity"]), Int(s["chirality"]),
                                 scale(cvalue(s["mass"])), factor(cvalue(s["factor"])); embedding = emb))
        elseif kind == "vertex"
            name = get(EXPORT_VERTICES, i["vertex"], nothing)
            name === nothing && export_error("vertex $(i["vertex"])")
            (opc, tl, tr, tout) = VERTEX_OPS[name]
            reads = opc == V_THREE_VECTOR
            i["right"] === nothing && export_error("vertex $(i["vertex"]) without a right operand")
            l = want(i["left"], TYPE_COMPONENTS[tl], tl, reads ? form_of(i["left_momentum"]) : nothing)
            r = want(i["right"], TYPE_COMPONENTS[tr], tr, reads ? form_of(i["right_momentum"]) : nothing)
            d = want(i["current"], TYPE_COMPONENTS[tout], tout)
            push!(raw, LOp(opc; set = set_mode(i["mode"]), first = true, last = true, d = d, t = scale(cvalue(i["coupling"])),
                           l = l, r = r, x = factor(cvalue(i["factor"]))))
        elseif kind == "propagator"
            name = get(EXPORT_PROPAGATORS, i["propagator"], nothing)
            name === nothing && export_error("propagator $(i["propagator"])")
            (opc, ty) = PROPAGATOR_OPS[name]
            Int(i["component_count"]) == TYPE_COMPONENTS[ty] || export_error("propagator $name of $(i["component_count"]) components")
            d = want(i["current"], TYPE_COMPONENTS[ty], ty, form_of(i["momentum_form"]))
            push!(raw, LOp(opc; d = d, x = factor(cvalue(i["factor"]))))
        elseif kind == "scale"
            n = Int(i["component_count"])
            push!(raw, LOp(SCALE_ROW; d = want(i["current"], n, UNTYPED), n = n, x = factor(cvalue(i["factor"]))))
        elseif kind == "copy" || kind == "scatter"
            n = Int(i["component_count"])
            d = want(i["current"], n, UNTYPED)
            l = want(i["source"], n, UNTYPED)
            push!(raw, LOp(kind == "copy" ? COPY : SCATTER; set = kind == "scatter" && set_mode(i["mode"]), d = d, l = l,
                           x = factor(cvalue(i["factor"]))))
        elseif kind == "closure"
            cf = [factor(cvalue(z)) for z in i["component_factors"]]
            n = length(cf)
            push!(raw, LOp(CLOSURE; d = Int(i["amplitude"]) + 1, t = intern!(lists, cf), l = want(i["left"], n, UNTYPED),
                           r = want(i["right"], n, UNTYPED), x = factor(cvalue(i["factor"]))))
        elseif kind == "kernel"
            i["amplitude"] && export_error("a kernel with an amplitude destination")
            b = get(bindings, Int(i["binding"]), nothing)
            b === nothing && export_error("kernel row: no binding $(i["binding"])")
            (inputs, _, outputs, _, direct) = b
            # each operand's components: the largest one the binding reads (at least 1)
            reach(kinds...) = maximum((x.n for x in inputs if x.kind in kinds); init = 0)
            nd = max(1, reach(OUT), maximum(outputs; init = 0))
            # an operand the binding reads (a component or its momentum): its edge; else 0
            function operand(at, part::InputKind, momentum::InputKind, form)
                comps = reach(part)
                reads_form = any(x -> x.kind == momentum, inputs)
                comps == 0 && !reads_form && return 0
                at === nothing && export_error("a kernel input of a null operand")
                return want(at, max(1, comps), UNTYPED, reads_form ? form_of(form) : nothing)
            end
            l = operand(i["left"], IN_L, MOMENTUM_U, i["left_momentum"])
            r = operand(i["right"], IN_R, MOMENTUM_V, i["right_momentum"])
            push!(raw, LOp(direct ? KERNEL_DIRECT : KERNEL; set = set_mode(i["mode"]), d = want(i["destination"], nd, UNTYPED),
                           t = instance(b), l = l, r = r, x = factor(cvalue(i["factor"]))))
        else
            export_error("instruction $kind")
        end
    end

    for run in p["runs"]
        sources = run["sources"]
        for i in p["body"]
            row!(i, sources)
        end
        for t in run["reduction"]
            x = t["apply_amplitude_scale"] ? factor(cvalue(t["amplitude_scale"])) : 0
            push!(raw, LOp(REDUCE; d = Int(t["helicity_output"]) + 1, t = Int(t["amplitude"]) + 1, x = x,
                           n = intern_value!(weights, f64(t["weight"]))))
        end
    end

    # the edges: current c's own edge first (edge c), then the other (current, form) pairs
    edges = [LEdge(c, owns[c]) for c in eachindex(keys)]
    resolve(k) = k == 0 ? 0 : ((c, f) = pairs[k]; f === nothing || f == owns[c] ? c : intern!(edges, LEdge(c, f)))
    has_d(op) = !(op.op in (CLOSURE, CLEAR_AMPLITUDE, REDUCE))
    has_lr(op) = !(op.op in (SOURCE, CLEAR_AMPLITUDE, REDUCE))
    ops = [LOp(o.op, o.set, o.first, o.last, has_d(o) ? resolve(o.d) : o.d, o.t, has_lr(o) ? resolve(o.l) : o.l,
               has_lr(o) ? resolve(o.r) : o.r, o.n, o.x) for o in raw]
    currents = [LCurrent(offset, n, ty) for (offset, n, ty) in keys]

    legs = Int(p["external_count"])
    forms = [Form([(Int(t["leg"]) + 1, f64(t["coefficient"])) for t in f]) for f in p["momentum_forms"]]
    a = p["alpha_s"]
    alpha_s = a === nothing ? nothing : (f64(a["value"]["re"]), a["squared_amplitude_power"] === nothing ? nothing : Int(a["squared_amplitude_power"]))
    l = Lowered(UIR_VERSION, String(p["process_key"]), String(p["colour_id"]), legs, export_crossing(p, legs), forms, scales,
                factors, alpha_s, export_helicities(p), currents, Int(p["current_component_count"]), edges, lists, embeddings,
                stubs, scalesets, instances, ops, Int(p["amplitude_count"]), weights, export_group(p), true)
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
