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

function lower(p::Program)::Lowered
    types = physical_types(p)
    own = own_forms(p)
    (currents, planes) = plain_layout(p, types)

    # edge c is current c with its own momentum; a current read with another form gets one more edge
    edges = [LEdge(c, own[c]) for c in eachindex(p.currents)]
    edge(c::CurrentIndex, f::FormIndex) = c == 0 ? 0 : f == own[c] ? c : intern!(edges, LEdge(c, f))
    operand(c::CurrentIndex) = c   # an operand whose momentum the row does not read: its own edge

    ops = LOp[]
    for (c, cur) in enumerate(p.currents)
        s = cur.source
        s === nothing && continue
        s.orientation in ("particle", "antiparticle") || error("$(cname(c)): unknown orientation $(s.orientation)")
        # its edge: the current with the source's form (its own edge, unless a row reads it with another)
        push!(ops, source_op(edge(c, s.form), FAMILY_TYPES[s.family], s.orientation == "antiparticle", s.helicity,
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
            # a direct binding with factor 1 stores its pairs as they are, whatever its mode (pac2json's
            # kernel_store!): KERNEL_DIRECT with SET
            direct = b.direct && is_one(p.factors[op.factor])
            push!(ops, LOp(direct ? KERNEL_DIRECT : KERNEL; set = direct || set_flag(op.mode), d = operand(op.out),
                           t = instance(b), l = l, r = r, x = op.factor))
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

    weights = Float64[]
    for (h, terms) in enumerate(p.reduction), (a, w) in terms
        push!(ops, LOp(REDUCE; d = h, t = a, n = intern_value!(weights, w)))
    end

    l = Lowered(UIR_VERSION, p.process, p.colour, p.legs, p.crossing, p.forms, p.scales, p.factors, p.alpha_s,
                p.helicities, currents, planes, edges, lists, Vector{EmbeddingEntry}[], stubs, scalesets, instances, ops,
                p.amplitudes, weights, p.group_factor, false)
    validate(l)
    return l
end
