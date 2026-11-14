# The tables shared by the readers that build a Lowered (v1/lower.jl from a main.json Program,
# recurrence_export.jl from RustiCol's export): the vertex, propagator and source types, the
# interning of table values, and `validate`, the checks of a Lowered.

# vertex type → (opcode, l, r, out types)
const VERTEX_OPS = Dict(
    "weyl_vector_a" => (V_WEYL_VECTOR_A, WEYL, VECTOR, WEYL),
    "weyl_vector_b" => (V_WEYL_VECTOR_B, WEYL, VECTOR, WEYL),
    "tensor_vector" => (V_TENSOR_VECTOR, TENSOR, VECTOR, VECTOR),
    "vector_wedge" => (V_VECTOR_WEDGE, VECTOR, VECTOR, TENSOR),
    "three_vector" => (V_THREE_VECTOR, VECTOR, VECTOR, VECTOR),
)
# propagator type → (opcode, the current's type)
const PROPAGATOR_OPS = Dict("weyl_a" => (P_WEYL_A, WEYL), "weyl_b" => (P_WEYL_B, WEYL), "vector" => (P_VECTOR, VECTOR))
# source family → type
const FAMILY_TYPES = Dict("scalar" => SCALAR, "weyl" => WEYL, "dirac" => DIRAC, "vector" => VECTOR)

const OPCODE_SHAPES = Dict(op => (l, r, out) for (op, l, r, out) in values(VERTEX_OPS))
const PROPAGATOR_TYPES = Dict(op => t for (op, t) in values(PROPAGATOR_OPS))

cname(c::CurrentIndex) = "c$(c - 1)"   # main.json's name, for messages

# the index of `x` in `v`, appended if new
function intern!(v::Vector{T}, x::T)::Int where {T}
    k = findfirst(==(x), v)
    k === nothing || return k
    push!(v, x)
    return length(v)
end

# the index of `x` in `v` by value, bit for bit (-0.0 and 0.0 differ), appended if new
function intern_value!(v::Vector{T}, x::T)::Int where {T}
    k = findfirst(y -> isequal(y, x), v)
    k === nothing || return k
    push!(v, x)
    return length(v)
end

# The binding instance (stub, scale set) of a kernel with these inputs, outputs and scale set, each
# interned in its table; `same(stub's kernel)` says whether a stub's kernel is this one (the readers
# name their kernels differently).
function intern_instance!(same, stubs::Vector{LStub}, scalesets, instances, kernel::KernelLibrary, inputs::Vector{KernelInput},
                          outputs::Vector{ComponentIndex}, scaleset::Vector{ScaleIndex})::InstanceIndex
    k = findfirst(s -> same(s.kernel) && s.inputs == inputs && s.outputs == outputs, stubs)
    k === nothing && (push!(stubs, LStub(kernel, inputs, outputs)); k = length(stubs))
    return intern!(instances, (k, intern!(scalesets, scaleset)))
end

is_one(z::Complex{Float64}) = reinterpret(UInt64, real(z)) == reinterpret(UInt64, 1.0) &&
                              reinterpret(UInt64, imag(z)) == reinterpret(UInt64, 0.0)

################################### validate ######################################

# The checks of a Lowered, each adding its problems (messages) to `out`. `at` names the row in a
# message ("row k (OPCODE)").

inrange(i, n) = 1 <= i <= n
edge_components(l::Lowered, e::EdgeIndex) = edge_layout(l, e).components

# the layout (even planes, inside the arena, a typed current's component count), the edges and the embeddings
function table_problems!(out::Vector{String}, l::Lowered)
    nc, nf = length(l.currents), length(l.forms)
    for (c, cur) in enumerate(l.currents)
        iseven(cur.start) || push!(out, "$(cname(c)) starts at odd plane $(cur.start)")
        cur.start >= 0 && cur.start + cur.components <= l.planes || push!(out, "$(cname(c)) lies outside the $(l.planes) planes")
        cur.type == UNTYPED || TYPE_COMPONENTS[cur.type] == cur.components ||
            push!(out, "$(cname(c)) is $(cur.type) with $(cur.components) components")
    end
    for (e, edge) in enumerate(l.edges)
        inrange(edge.current, nc) || push!(out, "edge $e: no current $(edge.current)")
        0 <= edge.form <= nf || push!(out, "edge $e: no form $(edge.form)")
        e <= nc && edge.current != e && push!(out, "edge $e is not current $(cname(e))'s own edge")
    end
    length(unique(l.edges)) == length(l.edges) || push!(out, "an edge is listed twice")
    for (k, list) in enumerate(l.embeddings)
        for (j, w, x) in list
            j >= 1 && w >= 0 && inrange(x, length(l.factors)) || push!(out, "embedding $k: entry ($j, $w, $x) out of range")
        end
        length(unique(first.(list))) == length(list) || push!(out, "embedding $k writes a component twice")
    end
end

# the edges each opcode names are in range: d (not for a closure, clear_amplitude or reduce); l and
# r (vertex, closure; a kernel's may be 0); l (copy, scatter); a source's l is its embedding
function row_edges_ok(l::Lowered, op::LOp)::Bool
    ne = length(l.edges)
    (no_destination(op) || inrange(op.d, ne)) || return false
    is_vertex(op.op) || op.op == CLOSURE ? inrange(op.l, ne) && inrange(op.r, ne) :
    is_kernel(op.op) ? 0 <= op.l <= ne && 0 <= op.r <= ne :
    op.op in (COPY, SCATTER) ? inrange(op.l, ne) && op.r == 0 :
    op.op == SOURCE ? 0 <= op.l <= length(l.embeddings) && op.r == 0 : op.l == 0 && op.r == 0
end

# the factor: none for clear and clear_amplitude, optional for reduce (its amplitude's scale) and
# an embedded source (its entries carry theirs)
row_factor_ok(l::Lowered, op::LOp)::Bool =
    op.op in (CLEAR, CLEAR_AMPLITUDE) ? op.x == 0 :
    op.op == REDUCE ? 0 <= op.x <= length(l.factors) :
    op.op == SOURCE && op.l != 0 ? op.x == 0 : inrange(op.x, length(l.factors))

# A vertex row: its operands' types, the momenta a three-vector vertex reads, its coupling, and its
# place in a fused run (SET only on the run's first row; the run's rows on one destination; no row
# after the first reads the destination's planes). `open`: the destination edge of the open run
# (0: none); returns it after this row.
function vertex_problems!(out::Vector{String}, l::Lowered, op::LOp, at::String, open::EdgeIndex)::EdgeIndex
    (tl, tr, tout) = OPCODE_SHAPES[op.op]
    for (e, t, what) in ((op.l, tl, "l"), (op.r, tr, "r"), (op.d, tout, "out"))
        have = edge_layout(l, e).type
        have == t || push!(out, "$at: $what is $(cname(edge_current(l, e))), a $have, not a $t")
    end
    op.op == V_THREE_VECTOR && (edge_form(l, op.l) == 0 || edge_form(l, op.r) == 0) && push!(out, "$at: an operand without a momentum")
    inrange(op.t, length(l.scales)) || push!(out, "$at: no coupling $(op.t)")
    if op.first
        open == 0 || push!(out, "$at: a run starts inside another run")
        open = op.d
    else
        open == op.d || push!(out, "$at: continues a run on another destination")
        op.set && push!(out, "$at: SET on a row that does not start its run")
        any(e -> overlaps(plane_range(l, e), plane_range(l, op.d)), (op.l, op.r)) &&
            push!(out, "$at: reads the planes of its run's destination")
    end
    return op.last ? 0 : open
end

function propagator_problems!(out::Vector{String}, l::Lowered, op::LOp, at::String)
    edge_layout(l, op.d).type == PROPAGATOR_TYPES[op.op] || push!(out, "$at: $(cname(edge_current(l, op.d))) is a $(edge_layout(l, op.d).type)")
    edge_form(l, op.d) == 0 && push!(out, "$at: the current has no momentum")
end

# a kernel row: every input of its stub against the row's operands, and its outputs
function kernel_problems!(out::Vector{String}, l::Lowered, op::LOp, at::String)
    inrange(op.t, length(l.instances)) || return push!(out, "$at: no binding instance $(op.t)")
    (s, set) = l.instances[op.t]
    stub, scales = l.stubs[s], l.scalesets[set]
    for i in stub.inputs
        if i.kind in (IN_L, IN_R, MOMENTUM_U, MOMENTUM_V)
            e = i.kind == IN_L || i.kind == MOMENTUM_U ? op.l : op.r
            e == 0 && (push!(out, "$at: input $(i.kind) of a null operand"); continue)
            if i.kind == IN_L || i.kind == IN_R
                i.n <= edge_components(l, e) || push!(out, "$at: input $(i.kind)[$(i.n)] beyond $(edge_components(l, e)) components")
            else
                edge_form(l, e) == 0 && push!(out, "$at: input $(i.kind) of an operand without a momentum")
                i.n <= 4 || push!(out, "$at: momentum component $(i.n)")
            end
        elseif i.kind == OUT
            i.n <= edge_components(l, op.d) || push!(out, "$at: input out[$(i.n)] beyond $(edge_components(l, op.d)) components")
        elseif i.kind == SCALE
            inrange(i.n, length(scales)) && inrange(scales[i.n], length(l.scales)) || push!(out, "$at: scale slot $(i.n)")
        end
    end
    all(j -> 1 <= j <= edge_components(l, op.d), stub.outputs) || push!(out, "$at: an output beyond $(edge_components(l, op.d)) components")
end

function closure_problems!(out::Vector{String}, l::Lowered, op::LOp, at::String)
    inrange(op.d, l.amplitudes) || push!(out, "$at: no amplitude $(op.d)")
    inrange(op.t, length(l.lists)) || return push!(out, "$at: no component-factor list $(op.t)")
    n = length(l.lists[op.t])
    n <= min(edge_components(l, op.l), edge_components(l, op.r)) || push!(out, "$at: $n components, beyond its operands'")
    all(x -> inrange(x, length(l.factors)), l.lists[op.t]) || push!(out, "$at: a component factor out of range")
end

function source_problems!(out::Vector{String}, l::Lowered, op::LOp, at::String)
    (family, _, h, ch) = source_fields(op)
    family in (SCALAR, WEYL, DIRAC, VECTOR) || push!(out, "$at: family $family")
    -2 <= h <= 2 && -1 <= ch <= 1 || push!(out, "$at: helicity $h, chirality $ch")
    inrange(op.t, length(l.scales)) || push!(out, "$at: no mass scale $(op.t)")
    edge_form(l, op.d) == 0 && push!(out, "$at: its edge has no momentum")
    if op.l == 0
        edge_layout(l, op.d).type == family || push!(out, "$at: $family into a $(edge_layout(l, op.d).type)")
    else
        n = TYPE_COMPONENTS[family]
        all(((j, w, _),) -> j <= edge_components(l, op.d) && w <= n, l.embeddings[op.l]) ||
            push!(out, "$at: embedding $(op.l) beyond the current's $(edge_components(l, op.d)) or the wavefunction's $n components")
    end
end

function reduce_problems!(out::Vector{String}, l::Lowered, op::LOp, at::String)
    inrange(op.d, output_count(l)) || push!(out, "$at: no output $(op.d)")
    inrange(op.t, l.amplitudes) || push!(out, "$at: no amplitude $(op.t)")
    inrange(op.n, length(l.weights)) || push!(out, "$at: no weight $(op.n)")
end

# a row that is not a vertex, by its opcode
function row_problems!(out::Vector{String}, l::Lowered, op::LOp, at::String)
    if is_propagator(op.op)
        propagator_problems!(out, l, op, at)
    elseif is_kernel(op.op)
        kernel_problems!(out, l, op, at)
    elseif op.op == CLOSURE
        closure_problems!(out, l, op, at)
    elseif op.op == SCALE_ROW || op.op == CLEAR
        0 <= op.n <= edge_components(l, op.d) || push!(out, "$at: $(op.n) components, beyond $(edge_components(l, op.d))")
    elseif op.op == SOURCE
        source_problems!(out, l, op, at)
    elseif op.op == CLEAR_AMPLITUDE
        inrange(op.d, l.amplitudes) || push!(out, "$at: no amplitude $(op.d)")
    elseif op.op == COPY || op.op == SCATTER
        edge_components(l, op.l) >= edge_components(l, op.d) ||
            push!(out, "$at: copies $(edge_components(l, op.d)) components from $(edge_components(l, op.l))")
    elseif op.op == REDUCE
        reduce_problems!(out, l, op, at)
    end
end

# The problems of `l`, empty when it is valid: the layout (even planes, inside the arena), the
# edges, the sources, each row's operands against its opcode (edges, physical types, component
# counts, the momenta it reads, table indices) and the fused runs (SET only on a run's first row; a
# run is vertex rows on one destination, and no row after the first reads the destination's planes).
function problems(l::Lowered)::Vector{String}
    out = String[]
    table_problems!(out, l)
    open = 0          # the destination edge of the open fused run (0: none)
    for (k, op) in enumerate(l.ops)
        at = "row $k ($(op.op))"
        op.op == END && (push!(out, "$at: an END row (the encoding's, not a Lowered's)"); continue)
        row_edges_ok(l, op) || (push!(out, "$at: an edge out of range"); continue)
        row_factor_ok(l, op) || push!(out, "$at: factor $(op.x)")
        (op.set && !(is_vertex(op.op) || is_kernel(op.op) || op.op == SCATTER)) && push!(out, "$at: SET on a row without a mode")
        ((op.first || op.last) && !is_vertex(op.op)) && push!(out, "$at: FIRST/LAST on a row that is not a vertex")
        if is_vertex(op.op)
            open = vertex_problems!(out, l, op, at, open)
        else
            open == 0 || push!(out, "$at: inside the fused run on edge $open")
            row_problems!(out, l, op, at)
        end
    end
    open == 0 || push!(out, "the fused run on edge $open does not end")
    return out
end

# throws when `l` is not valid, with its first problems
function validate(l::Lowered)
    p = problems(l)
    isempty(p) && return l
    error("invalid Lowered ($(l.process), $(l.colour)): $(length(p)) problems\n  " * join(first(p, 20), "\n  "))
end
