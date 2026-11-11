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

is_one(z::Complex{Float64}) = reinterpret(UInt64, real(z)) == reinterpret(UInt64, 1.0) &&
                              reinterpret(UInt64, imag(z)) == reinterpret(UInt64, 0.0)

################################### validate ######################################

# The problems of `l`, empty when it is valid: the layout (even planes, inside the arena), the
# edges, the sources, each row's operands against its opcode (edges, physical types, component
# counts, the momenta it reads, table indices) and the fused runs (SET only on a run's first row; a
# run is vertex rows on one destination, and no row after the first reads the destination's planes).
function problems(l::Lowered)::Vector{String}
    out = String[]
    bad(msg) = push!(out, msg)
    nc, nf, ne = length(l.currents), length(l.forms), length(l.edges)
    inrange(i, n) = 1 <= i <= n

    for (c, cur) in enumerate(l.currents)
        iseven(cur.start) || bad("$(cname(c)) starts at odd plane $(cur.start)")
        cur.start >= 0 && cur.start + cur.components <= l.planes || bad("$(cname(c)) lies outside the $(l.planes) planes")
        cur.type == UNTYPED || TYPE_COMPONENTS[cur.type] == cur.components ||
            bad("$(cname(c)) is $(cur.type) with $(cur.components) components")
    end
    for (e, edge) in enumerate(l.edges)
        inrange(edge.current, nc) || bad("edge $e: no current $(edge.current)")
        0 <= edge.form <= nf || bad("edge $e: no form $(edge.form)")
        e <= nc && edge.current != e && bad("edge $e is not current $(cname(e))'s own edge")
    end
    length(unique(l.edges)) == ne || bad("an edge is listed twice")
    for (k, list) in enumerate(l.embeddings)
        for (j, w, x) in list
            j >= 1 && w >= 0 && inrange(x, length(l.factors)) || bad("embedding $k: entry ($j, $w, $x) out of range")
        end
        length(unique(first.(list))) == length(list) || bad("embedding $k writes a component twice")
    end
    nw, nh = length(l.weights), output_count(l)

    components(e) = edge_layout(l, e).components
    planes(e) = (edge_plane(l, e), edge_plane(l, e) + components(e) - 1)
    overlaps(a, b) = a[1] <= b[2] && b[1] <= a[2]
    open = 0          # the destination edge of the open fused run (0: none)
    for (k, op) in enumerate(l.ops)
        at = "row $k ($(op.op))"
        op.op == END && (bad("$at: an END row (the encoding's, not a Lowered's)"); continue)
        # the edges each opcode names: d (not for a closure, clear_amplitude or reduce); l and r (vertex,
        # closure; a kernel's may be 0); l (copy, scatter); a source's l is its embedding
        d_edge = !(op.op in (CLOSURE, CLEAR_AMPLITUDE, REDUCE))
        edges_ok = (!d_edge || inrange(op.d, ne)) &&
                   (is_vertex(op.op) || op.op == CLOSURE ? inrange(op.l, ne) && inrange(op.r, ne) :
                    is_kernel(op.op) ? 0 <= op.l <= ne && 0 <= op.r <= ne :
                    op.op in (COPY, SCATTER) ? inrange(op.l, ne) && op.r == 0 :
                    op.op == SOURCE ? 0 <= op.l <= length(l.embeddings) && op.r == 0 : op.l == 0 && op.r == 0)
        edges_ok || (bad("$at: an edge out of range"); continue)
        # the factor: none for clear and clear_amplitude, optional for reduce (its amplitude's scale) and
        # an embedded source (its entries carry theirs)
        factor_ok = op.op in (CLEAR, CLEAR_AMPLITUDE) ? op.x == 0 :
                    op.op == REDUCE ? 0 <= op.x <= length(l.factors) :
                    op.op == SOURCE && op.l != 0 ? op.x == 0 : inrange(op.x, length(l.factors))
        factor_ok || bad("$at: factor $(op.x)")
        (op.set && !(is_vertex(op.op) || is_kernel(op.op) || op.op == SCATTER)) && bad("$at: SET on a row without a mode")
        ((op.first || op.last) && !is_vertex(op.op)) && bad("$at: FIRST/LAST on a row that is not a vertex")

        if is_vertex(op.op)
            (tl, tr, tout) = OPCODE_SHAPES[op.op]
            for (e, t, what) in ((op.l, tl, "l"), (op.r, tr, "r"), (op.d, tout, "out"))
                have = edge_layout(l, e).type
                have == t || bad("$at: $what is $(cname(edge_current(l, e))), a $have, not a $t")
            end
            op.op == V_THREE_VECTOR && (edge_form(l, op.l) == 0 || edge_form(l, op.r) == 0) && bad("$at: an operand without a momentum")
            inrange(op.t, length(l.scales)) || bad("$at: no coupling $(op.t)")
            # the fused run
            if op.first
                open == 0 || bad("$at: a run starts inside another run")
                open = op.d
            else
                open == op.d || bad("$at: continues a run on another destination")
                op.set && bad("$at: SET on a row that does not start its run")
                any(e -> overlaps(planes(e), planes(op.d)), (op.l, op.r)) && bad("$at: reads the planes of its run's destination")
            end
            op.last && (open = 0)
            continue
        end
        open == 0 || bad("$at: inside the fused run on edge $open")
        if is_propagator(op.op)
            edge_layout(l, op.d).type == PROPAGATOR_TYPES[op.op] || bad("$at: $(cname(edge_current(l, op.d))) is a $(edge_layout(l, op.d).type)")
            edge_form(l, op.d) == 0 && bad("$at: the current has no momentum")
        elseif is_kernel(op.op)
            inrange(op.t, length(l.instances)) || (bad("$at: no binding instance $(op.t)"); continue)
            (s, set) = l.instances[op.t]
            stub, scales = l.stubs[s], l.scalesets[set]
            for i in stub.inputs
                if i.kind == IN_L || i.kind == IN_R || i.kind == MOMENTUM_U || i.kind == MOMENTUM_V
                    e = i.kind == IN_L || i.kind == MOMENTUM_U ? op.l : op.r
                    e == 0 && (bad("$at: input $(i.kind) of a null operand"); continue)
                    if i.kind == IN_L || i.kind == IN_R
                        i.n <= components(e) || bad("$at: input $(i.kind)[$(i.n)] beyond $(components(e)) components")
                    else
                        edge_form(l, e) == 0 && bad("$at: input $(i.kind) of an operand without a momentum")
                        i.n <= 4 || bad("$at: momentum component $(i.n)")
                    end
                elseif i.kind == OUT
                    i.n <= components(op.d) || bad("$at: input out[$(i.n)] beyond $(components(op.d)) components")
                elseif i.kind == SCALE
                    inrange(i.n, length(scales)) && inrange(scales[i.n], length(l.scales)) || bad("$at: scale slot $(i.n)")
                end
            end
            all(j -> 1 <= j <= components(op.d), stub.outputs) || bad("$at: an output beyond $(components(op.d)) components")
        elseif op.op == CLOSURE
            inrange(op.d, l.amplitudes) || bad("$at: no amplitude $(op.d)")
            inrange(op.t, length(l.lists)) || (bad("$at: no component-factor list $(op.t)"); continue)
            n = length(l.lists[op.t])
            n <= min(components(op.l), components(op.r)) || bad("$at: $n components, beyond its operands'")
            all(x -> inrange(x, length(l.factors)), l.lists[op.t]) || bad("$at: a component factor out of range")
        elseif op.op == SCALE_ROW || op.op == CLEAR
            0 <= op.n <= components(op.d) || bad("$at: $(op.n) components, beyond $(components(op.d))")
        elseif op.op == SOURCE
            (family, _, h, ch) = source_fields(op)
            family in (SCALAR, WEYL, DIRAC, VECTOR) || bad("$at: family $family")
            -2 <= h <= 2 && -1 <= ch <= 1 || bad("$at: helicity $h, chirality $ch")
            inrange(op.t, length(l.scales)) || bad("$at: no mass scale $(op.t)")
            edge_form(l, op.d) == 0 && bad("$at: its edge has no momentum")
            if op.l == 0
                edge_layout(l, op.d).type == family || bad("$at: $family into a $(edge_layout(l, op.d).type)")
            else
                n = TYPE_COMPONENTS[family]
                all(((j, w, _),) -> j <= components(op.d) && w <= n, l.embeddings[op.l]) ||
                    bad("$at: embedding $(op.l) beyond the current's $(components(op.d)) or the wavefunction's $n components")
            end
        elseif op.op == CLEAR_AMPLITUDE
            inrange(op.d, l.amplitudes) || bad("$at: no amplitude $(op.d)")
        elseif op.op == COPY || op.op == SCATTER
            components(op.l) >= components(op.d) || bad("$at: copies $(components(op.d)) components from $(components(op.l))")
        elseif op.op == REDUCE
            inrange(op.d, nh) || bad("$at: no output $(op.d)")
            inrange(op.t, l.amplitudes) || bad("$at: no amplitude $(op.t)")
            inrange(op.n, nw) || bad("$at: no weight $(op.n)")
        end
    end
    open == 0 || bad("the fused run on edge $open does not end")
    return out
end

# throws when `l` is not valid, with its first problems
function validate(l::Lowered)
    p = problems(l)
    isempty(p) && return l
    error("invalid Lowered ($(l.process), $(l.colour)): $(length(p)) problems\n  " * join(first(p, 20), "\n  "))
end
