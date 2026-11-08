# evaluate(l::Lowered, a::LArena, momenta): the UIR rows run in Julia, the third oracle. Its values
# must equal evaluate(p::Program) bit for bit: the arithmetic is operations.jl's, called the same
# way, and only the storage differs (one plane array, reached through the edge table). A mismatch
# then points at `lower` or at a back end, never at both.
#
# Fused runs follow the UIR semantics (design doc, "The lowered IR"): a vertex row's value c goes to
# the run's sum, acc = c (FIRST with SET), acc = arena + c (FIRST), acc = acc + c (otherwise), and
# the sum is stored at LAST; the additions keep the unfused order, so the values are unchanged.

# The values of one point while a Lowered runs
struct LArena
    forms::Vector{NTuple{4,Float64}}      # the momentum forms, (E, px, py, pz)
    planes::Vector{Complex{Float64}}      # plane offset k (0-based) is planes[k + 1]
    amplitudes::Vector{Complex{Float64}}
    acc::Vector{Complex{Float64}}         # the open fused run's sum, one per component of its destination
end

LArena(l::Lowered) = LArena(fill((0.0, 0.0, 0.0, 0.0), length(l.forms)), zeros(Complex{Float64}, l.planes),
                            zeros(Complex{Float64}, l.amplitudes), Complex{Float64}[])

# an edge's planes as a vector (a view into the arena); a null operand: no components
function planes(a::LArena, l::Lowered, e::EdgeIndex)
    e == 0 && return NO_CURRENT
    c = edge_layout(l, e)
    return view(a.planes, c.start+1:c.start+c.components)
end

edge_momentum(a::LArena, l::Lowered, e::EdgeIndex) = a.forms[edge_form(l, e)]

# the names operations.jl takes, by opcode
const VERTEX_TYPES = Dict(op => t for (t, (op, _, _, _)) in VERTEX_OPS)
const PROPAGATOR_NAMES = Dict(op => t for (t, (op, _)) in PROPAGATOR_OPS)
const FAMILY_NAMES = Dict(t => f for (f, t) in FAMILY_TYPES)

# a vertex row: its values, × (coupling × factor) as vertex! does, into the run's sum
function run_vertex!(a::LArena, l::Lowered, op::LOp)
    ty = VERTEX_TYPES[op.op]
    reads = op.op == V_THREE_VECTOR
    u = reads ? edge_momentum(a, l, op.l) : nothing
    v = reads ? edge_momentum(a, l, op.r) : nothing
    scale = l.scales[op.t] * l.factors[op.x]
    o = vertex_values(ty, planes(a, l, op.l), planes(a, l, op.r), u, v)
    out = planes(a, l, op.d)
    if op.first
        resize!(a.acc, length(o))
        for j in eachindex(o)
            value = o[j] * scale
            a.acc[j] = op.set ? value : out[j] + value
        end
    else
        length(o) == length(a.acc) || error("a fused run changes its component count")
        for j in eachindex(o)
            a.acc[j] = a.acc[j] + o[j] * scale
        end
    end
    if op.last
        for j in eachindex(a.acc)
            out[j] = a.acc[j]
        end
    end
end

# a propagator row: the current's new values, as evaluate(::Propagator) computes them
function run_propagator!(a::LArena, l::Lowered, op::LOp)
    c = planes(a, l, op.d)
    v = propagator(PROPAGATOR_NAMES[op.op], c, edge_momentum(a, l, op.d), l.factors[op.x])
    c[1:length(v)] = v
end

# a closure row: the amplitude += Σ l[j] r[j] × component factor j × factor
function run_closure!(a::LArena, l::Lowered, op::LOp)
    cf = [l.factors[k] for k in l.lists[op.t]]
    a.amplitudes[op.d] = closure(a.amplitudes[op.d], planes(a, l, op.l), planes(a, l, op.r), cf, l.factors[op.x])
end

function kernel_input(i::KernelInput, op::LOp, a::LArena, l::Lowered, scaleset::Vector{ScaleIndex})::Float64
    part(z) = i.im ? imag(z) : real(z)
    if i.kind == IN_L
        return part(planes(a, l, op.l)[i.n])
    elseif i.kind == IN_R
        return part(planes(a, l, op.r)[i.n])
    elseif i.kind == OUT
        return part(planes(a, l, op.d)[i.n])
    elseif i.kind == MOMENTUM_U
        return edge_momentum(a, l, op.l)[i.n]
    elseif i.kind == MOMENTUM_V
        return edge_momentum(a, l, op.r)[i.n]
    elseif i.kind == SCALE
        return real(l.scales[scaleset[i.n]])
    else
        return part(l.factors[op.x])
    end
end

# a kernel row: the stub's inputs, the kernel, its pairs into out (KERNEL_DIRECT: as they are)
function run_kernel!(a::LArena, l::Lowered, op::LOp)
    (s, set) = l.instances[op.t]
    stub, scaleset = l.stubs[s], l.scalesets[set]
    values = kernel_call(stub.kernel, [kernel_input(i, op, a, l, scaleset) for i in stub.inputs])
    out = planes(a, l, op.d)
    factor = l.factors[op.x]
    for (n, j) in enumerate(stub.outputs)
        pair = complex(values[2n-1], values[2n])
        if op.op == KERNEL_DIRECT
            out[j] = pair
        else
            store!(out, j, pair * factor, op.set ? "=" : "+=")
        end
    end
end

function run_row!(a::LArena, l::Lowered, op::LOp)
    if is_vertex(op.op)
        run_vertex!(a, l, op)
    elseif is_propagator(op.op)
        run_propagator!(a, l, op)
    elseif is_kernel(op.op)
        run_kernel!(a, l, op)
    elseif op.op == CLOSURE
        run_closure!(a, l, op)
    elseif op.op == SCALE_ROW
        scale!(planes(a, l, op.d), op.n, l.factors[op.x])
    elseif op.op == CLEAR
        clear!(planes(a, l, op.d), op.n)
    else
        error("unknown opcode $(op.op)")
    end
end

# the sources into their currents' planes (every other plane is zero)
function fill_sources!(a::LArena, l::Lowered)
    fill!(a.planes, complex(0.0, 0.0))
    for s in l.sources
        w = wavefunction(FAMILY_NAMES[s.family], s.antiparticle ? "antiparticle" : "particle", a.forms[s.form],
                         s.helicity, s.chirality, real(l.scales[s.mass]))
        c = l.currents[s.current]
        factor = l.factors[s.factor]
        for j in 1:c.components
            a.planes[c.start+j] = factor * w[j]
        end
    end
end

# The outputs of `l` (one per helicity configuration, group factor applied) at one point, as
# evaluate(p::Program): momenta[leg] = (E, px, py, pz), the caller's momenta, 1-based legs.
function evaluate(l::Lowered, a::LArena, momenta)::Vector{Float64}
    fill_forms!(a, l, momenta)
    fill_sources!(a, l)
    fill!(a.amplitudes, complex(0.0, 0.0))
    for op in l.ops
        run_row!(a, l, op)
    end
    return [reduction([a.amplitudes[k] for (k, _) in terms], [w for (_, w) in terms], l.group_factor) for terms in l.reduction]
end
