# lower(p::Program) -> Lowered (UIR v0.8, uir.jl): the plain lowering. Every current gets its own
# planes (no reuse), at an even plane; the sources are SOURCE rows at the head, then the body's rows
# in order, one LOp per operation, then the reduction as REDUCE rows (output by output, each output's
# terms in order: the sums of v0.7's reduction); each vertex row is its own fused run (first and last). The optional passes (plane reuse, the sort of
# contribution runs, longer fused runs) come later and must leave the values unchanged bit for bit.
# `validate` (../validate.jl) checks a Lowered; lower calls it on its result.

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

function set_flag(mode::String)::Bool
    mode == "=" && return true
    mode == "+=" && return false
    error("unknown mode $mode")
end

# What lower collects: the edges beyond each current's own, the rows, and the kernel tables.
struct Lowering
    own::Vector{FormIndex}                                 # each current's own form
    edges::Vector{LEdge}
    ops::Vector{LOp}
    lists::Vector{Vector{FactorIndex}}
    stubs::Vector{LStub}
    scalesets::Vector{Vector{ScaleIndex}}
    instances::Vector{Tuple{StubIndex,ScaleSetIndex}}
end

# edge c is current c with its own momentum; a current read with another form gets one more edge
Lowering(p::Program, own::Vector{FormIndex}) =
    Lowering(own, [LEdge(c, own[c]) for c in eachindex(p.currents)], LOp[], Vector{FactorIndex}[], LStub[], Vector{ScaleIndex}[],
             Tuple{StubIndex,ScaleSetIndex}[])

# the edge of current `c` read with form `f` (0: a null operand)
edge!(w::Lowering, c::CurrentIndex, f::FormIndex)::EdgeIndex = c == 0 ? 0 : f == w.own[c] ? c : intern!(w.edges, LEdge(c, f))

# the binding instance (stub, scale set) of a binding: its scale inputs become slots of the scale set
function instance!(w::Lowering, b::Binding)::InstanceIndex
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
    return intern_instance!(k -> k.name == b.kernel.name, w.stubs, w.scalesets, w.instances, b.kernel, inputs, b.outputs, scaleset)
end

# the SOURCE rows, at the head: each source current's edge is the current with the source's form
# (its own edge, unless a row reads it with another)
function lower_sources!(w::Lowering, p::Program)
    for (c, cur) in enumerate(p.currents)
        s = cur.source
        s === nothing && continue
        s.orientation in ("particle", "antiparticle") || error("$(cname(c)): unknown orientation $(s.orientation)")
        push!(w.ops, source_op(edge!(w, c, s.form), FAMILY_TYPES[s.family], s.orientation == "antiparticle", s.helicity,
                               s.chirality, s.mass, s.factor))
    end
end

# ---- one operation of the body as its row; an operand whose momentum the row does not read is its
# current's own edge (the current's index)

function lower_op!(w::Lowering, p::Program, op::Vertex)
    (opc, _, _, _) = VERTEX_OPS[op.type]
    reads = opc == V_THREE_VECTOR
    l = reads ? edge!(w, op.l, op.u) : op.l
    r = reads ? edge!(w, op.r, op.v) : op.r
    push!(w.ops, LOp(opc; set = set_flag(op.mode), first = true, last = true, d = op.out, t = op.coupling, l = l, r = r,
                     x = op.factor))
end

lower_op!(w::Lowering, p::Program, op::Propagator) =
    push!(w.ops, LOp(PROPAGATOR_OPS[op.type][1]; d = edge!(w, op.current, op.momentum), x = op.factor))

function lower_op!(w::Lowering, p::Program, op::Kernel)
    b = p.bindings[op.binding]
    kinds = [i.kind for i in b.inputs]
    l = MOMENTUM_U in kinds ? edge!(w, op.l, op.u) : op.l
    r = MOMENTUM_V in kinds ? edge!(w, op.r, op.v) : op.r
    # a direct binding with factor 1 stores its pairs as they are, whatever its mode (pac2json's
    # kernel_store!): KERNEL_DIRECT with SET
    direct = b.direct && is_one(p.factors[op.factor])
    push!(w.ops, LOp(direct ? KERNEL_DIRECT : KERNEL; set = direct || set_flag(op.mode), d = op.out, t = instance!(w, b),
                     l = l, r = r, x = op.factor))
end

lower_op!(w::Lowering, p::Program, op::Closure) =
    push!(w.ops, LOp(CLOSURE; d = op.amplitude, t = intern!(w.lists, op.component_factors), l = op.l, r = op.r, x = op.factor))
lower_op!(w::Lowering, p::Program, op::Scale) = push!(w.ops, LOp(SCALE_ROW; d = op.current, n = op.components, x = op.factor))
lower_op!(w::Lowering, p::Program, op::Clear) = push!(w.ops, LOp(CLEAR; d = op.current, n = op.components))
lower_op!(w::Lowering, p::Program, op::Operation) = error("unknown operation $op")

# the reduction as REDUCE rows, output by output, each output's terms in order; returns the weights
function lower_reduction!(w::Lowering, p::Program)::Vector{Float64}
    weights = Float64[]
    for (h, terms) in enumerate(p.reduction), (a, x) in terms
        push!(w.ops, LOp(REDUCE; d = h, t = a, n = intern_value!(weights, x)))
    end
    return weights
end

function lower(p::Program)::Lowered
    (currents, planes) = plain_layout(p, physical_types(p))
    w = Lowering(p, own_forms(p))
    lower_sources!(w, p)
    foreach(op -> lower_op!(w, p, op), p.body)
    weights = lower_reduction!(w, p)
    l = Lowered(UIR_VERSION, p.process, p.colour, p.legs, p.crossing, p.forms, p.scales, p.factors, p.alpha_s,
                p.helicities, currents, planes, w.edges, w.lists, Vector{EmbeddingEntry}[], w.stubs, w.scalesets, w.instances,
                w.ops, p.amplitudes, weights, p.group_factor, false)
    validate(l)
    return l
end
