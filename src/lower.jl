# lower(p::Program) -> Lowered (UIR v0.6, uir.jl): the plain lowering. Every current gets its own
# planes (no reuse), at an even plane; the rows keep the body's order, one LOp per operation; each
# vertex row is its own fused run (first and last). The optional passes (plane reuse, the sort of
# contribution runs, longer fused runs) come later and must leave the values unchanged bit for bit.
# `validate` checks a Lowered (lower calls it on its result).

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

# the momenta an operation reads, as (current, form) pairs
function momentum_reads(op::Operation, p::Program)::Vector{Tuple{CurrentIndex,FormIndex}}
    if op isa Vertex
        return op.type == "three_vector" ? [(op.l, op.u), (op.r, op.v)] : Tuple{CurrentIndex,FormIndex}[]
    elseif op isa Propagator
        return [(op.current, op.momentum)]
    elseif op isa Kernel
        kinds = [i.kind for i in p.bindings[op.binding].inputs]
        reads = Tuple{CurrentIndex,FormIndex}[]
        MOMENTUM_U in kinds && push!(reads, (op.l, op.u))
        MOMENTUM_V in kinds && push!(reads, (op.r, op.v))
        return reads
    end
    return Tuple{CurrentIndex,FormIndex}[]
end

# the forms an operation lists for its operands, read or not
listed_forms(op::Union{Vertex,Kernel}) = [(op.l, op.u), (op.r, op.v)]
listed_forms(op::Operation) = Tuple{CurrentIndex,FormIndex}[]

# each current's physical type, from the sources, the vertices and the propagators
function physical_types(p::Program)::Vector{PhysicalType}
    types = fill(UNTYPED, length(p.currents))
    function assign!(c::CurrentIndex, t::PhysicalType, why)
        c == 0 && return
        types[c] == UNTYPED || types[c] == t || error("$(cname(c)) is $(types[c]) and $t ($why)")
        types[c] = t
    end
    for (c, cur) in enumerate(p.currents)
        cur.source === nothing || assign!(c, FAMILY_TYPES[cur.source.family], "its source")
    end
    for op in p.body
        if op isa Vertex
            haskey(VERTEX_OPS, op.type) || error("unknown vertex type $(op.type)")
            (_, l, r, out) = VERTEX_OPS[op.type]
            assign!(op.l, l, op.type)
            assign!(op.r, r, op.type)
            assign!(op.out, out, op.type)
        elseif op isa Propagator
            haskey(PROPAGATOR_OPS, op.type) || error("unknown propagator type $(op.type)")
            assign!(op.current, PROPAGATOR_OPS[op.type][2], "propagator $(op.type)")
        end
    end
    return types
end

# each current's own momentum: the first form a row reads it with, else its source's, else the
# first one a row lists for it, else none (0)
function own_forms(p::Program)::Vector{FormIndex}
    read = zeros(FormIndex, length(p.currents))
    listed = zeros(FormIndex, length(p.currents))
    for op in p.body
        for (c, f) in momentum_reads(op, p)
            c == 0 && error("an operation reads the momentum of a null operand: $op")
            f == 0 && error("an operation reads no form for $(cname(c)): $op")
            read[c] == 0 && (read[c] = f)
        end
        for (c, f) in listed_forms(op)
            c != 0 && f != 0 && listed[c] == 0 && (listed[c] = f)
        end
    end
    forms = zeros(FormIndex, length(p.currents))
    for (c, cur) in enumerate(p.currents)
        forms[c] = read[c] != 0 ? read[c] : cur.source !== nothing ? cur.source.form : listed[c]
    end
    return forms
end

# The plain layout: the currents in order, each at an even plane (a current with an odd number of
# components is followed by one padding plane)
function plain_layout(p::Program, types::Vector{PhysicalType})::Tuple{Vector{LCurrent},Int}
    currents = LCurrent[]
    start = 0
    for (c, cur) in enumerate(p.currents)
        push!(currents, LCurrent(start, cur.components, types[c]))
        start += cur.components + isodd(cur.components)
    end
    return (currents, start)
end

# the index of `x` in `v`, appended if new
function intern!(v::Vector{T}, x::T)::Int where {T}
    k = findfirst(==(x), v)
    k === nothing || return k
    push!(v, x)
    return length(v)
end

is_one(z::Complex{Float64}) = reinterpret(UInt64, real(z)) == reinterpret(UInt64, 1.0) &&
                              reinterpret(UInt64, imag(z)) == reinterpret(UInt64, 0.0)

function set_flag(mode::String)::Bool
    mode == "=" && return true
    mode == "+=" && return false
    error("unknown mode $mode")
end

function lower(p::Program)::Lowered
    types = physical_types(p)
    own = own_forms(p)
    (currents, planes) = plain_layout(p, types)

    # edge c is current c with its own momentum; a current read with another form gets one more edge
    edges = [LEdge(c, own[c]) for c in eachindex(p.currents)]
    edge(c::CurrentIndex, f::FormIndex) = c == 0 ? 0 : f == own[c] ? c : intern!(edges, LEdge(c, f))
    operand(c::CurrentIndex) = c   # an operand whose momentum the row does not read: its own edge

    sources = LSource[]
    for (c, cur) in enumerate(p.currents)
        s = cur.source
        s === nothing && continue
        s.orientation in ("particle", "antiparticle") || error("$(cname(c)): unknown orientation $(s.orientation)")
        # its edge: the current with the source's form (its own edge, unless a row reads it with another)
        push!(sources, LSource(edge(c, s.form), FAMILY_TYPES[s.family], s.orientation == "antiparticle", s.helicity,
                               s.chirality, s.mass, s.factor))
    end

    lists = Vector{FactorIndex}[]
    stubs = LStub[]
    scalesets = Vector{ScaleIndex}[]
    instances = Tuple{StubIndex,ScaleSetIndex}[]
    function instance(b::Binding)::InstanceIndex
        scaleset = ScaleIndex[]
        inputs = KernelInput[]
        for i in b.inputs
            if i.kind == SCALE
                push!(scaleset, i.n)
                push!(inputs, KernelInput(SCALE, length(scaleset), false))
            else
                push!(inputs, i)
            end
        end
        k = findfirst(s -> s.kernel.name == b.kernel.name && s.inputs == inputs && s.outputs == b.outputs, stubs)
        k === nothing && (push!(stubs, LStub(b.kernel, inputs, b.outputs)); k = length(stubs))
        return intern!(instances, (k, intern!(scalesets, scaleset)))
    end

    ops = LOp[]
    for op in p.body
        if op isa Vertex
            (opc, _, _, _) = VERTEX_OPS[op.type]
            reads = opc == V_THREE_VECTOR
            l = reads ? edge(op.l, op.u) : operand(op.l)
            r = reads ? edge(op.r, op.v) : operand(op.r)
            push!(ops, LOp(opc; set = set_flag(op.mode), first = true, last = true, d = operand(op.out), t = op.coupling,
                           l = l, r = r, x = op.factor))
        elseif op isa Propagator
            opc = PROPAGATOR_OPS[op.type][1]
            push!(ops, LOp(opc; d = edge(op.current, op.momentum), x = op.factor))
        elseif op isa Kernel
            b = p.bindings[op.binding]
            kinds = [i.kind for i in b.inputs]
            l = MOMENTUM_U in kinds ? edge(op.l, op.u) : operand(op.l)
            r = MOMENTUM_V in kinds ? edge(op.r, op.v) : operand(op.r)
            opc = b.direct && is_one(p.factors[op.factor]) ? KERNEL_DIRECT : KERNEL
            push!(ops, LOp(opc; set = set_flag(op.mode), d = operand(op.out), t = instance(b), l = l, r = r, x = op.factor))
        elseif op isa Closure
            push!(ops, LOp(CLOSURE; d = op.amplitude, t = intern!(lists, op.component_factors), l = operand(op.l),
                           r = operand(op.r), x = op.factor))
        elseif op isa Scale
            push!(ops, LOp(SCALE_ROW; d = operand(op.current), n = op.components, x = op.factor))
        elseif op isa Clear
            push!(ops, LOp(CLEAR; d = operand(op.current), n = op.components))
        else
            error("unknown operation $op")
        end
    end

    l = Lowered(UIR_VERSION, p.process, p.colour, p.legs, p.crossing, p.forms, p.scales, p.factors, p.alpha_s,
                p.helicities, currents, planes, edges, sources, lists, stubs, scalesets, instances, ops, p.amplitudes,
                p.reduction, p.group_factor)
    validate(l)
    return l
end

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
    for s in l.sources
        inrange(s.edge, ne) && inrange(s.mass, length(l.scales)) && inrange(s.factor, length(l.factors)) ||
            (bad("source on edge $(s.edge): an index out of range"); continue)
        at = "source of $(cname(edge_current(l, s.edge)))"
        edge_form(l, s.edge) == 0 && bad("$at: its edge has no momentum")
        edge_layout(l, s.edge).type == s.family || bad("$at: $(s.family) into a $(edge_layout(l, s.edge).type)")
        -2 <= s.helicity <= 2 && -1 <= s.chirality <= 1 || bad("$at: helicity $(s.helicity), chirality $(s.chirality)")
    end

    components(e) = edge_layout(l, e).components
    planes(e) = (edge_plane(l, e), edge_plane(l, e) + components(e) - 1)
    overlaps(a, b) = a[1] <= b[2] && b[1] <= a[2]
    open = 0          # the destination edge of the open fused run (0: none)
    for (k, op) in enumerate(l.ops)
        at = "row $k ($(op.op))"
        op.op == SOURCE && (bad("$at: a source among the rows (Lowered.sources holds them)"); continue)
        # the edges each opcode names: d (not for a closure), l and r (vertex, closure; a kernel's may be 0)
        edges_ok = (op.op == CLOSURE || inrange(op.d, ne)) &&
                   (is_vertex(op.op) || op.op == CLOSURE ? inrange(op.l, ne) && inrange(op.r, ne) :
                    is_kernel(op.op) ? 0 <= op.l <= ne && 0 <= op.r <= ne : op.l == 0 && op.r == 0)
        edges_ok || (bad("$at: an edge out of range"); continue)
        (op.op == CLEAR ? op.x == 0 : inrange(op.x, length(l.factors))) || bad("$at: factor $(op.x)")
        (op.set && !(is_vertex(op.op) || is_kernel(op.op))) && bad("$at: SET on a row without a mode")
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
            op.op == KERNEL_DIRECT && !is_one(l.factors[op.x]) && bad("$at: KERNEL_DIRECT with a factor other than 1")
        elseif op.op == CLOSURE
            inrange(op.d, l.amplitudes) || bad("$at: no amplitude $(op.d)")
            inrange(op.t, length(l.lists)) || (bad("$at: no component-factor list $(op.t)"); continue)
            n = length(l.lists[op.t])
            n <= min(components(op.l), components(op.r)) || bad("$at: $n components, beyond its operands'")
            all(x -> inrange(x, length(l.factors)), l.lists[op.t]) || bad("$at: a component factor out of range")
        elseif op.op == SCALE_ROW || op.op == CLEAR
            0 <= op.n <= components(op.d) || bad("$at: $(op.n) components, beyond $(components(op.d))")
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
