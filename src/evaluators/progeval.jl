# evaluate(p::Program, a::Arena, momenta): a Program run in Julia on one phase-space point, the
# first oracle. The arithmetic is operations.jl's, the sources wavefunctions.jl's; RustiCol's in the
# same order, so the values are its bit for bit (up to its SIMD kernels' rounding).

Arena(p::Program) = Arena(length(p.forms), [c.components for c in p.currents], p.amplitudes)

function evaluate(op::Vertex, a::Arena, p::Program)
    vertex!(a.currents[op.out], op.type, op.mode, current(a, op.l), current(a, op.r), form(a, op.u), form(a, op.v),
            p.scales[op.coupling], p.factors[op.factor])
end

function evaluate(op::Propagator, a::Arena, p::Program)
    c = a.currents[op.current]
    v = propagator(op.type, c, form(a, op.momentum), p.factors[op.factor])
    c[1:length(v)] = v
end

evaluate(op::Scale, a::Arena, p::Program) = scale!(a.currents[op.current], op.components, p.factors[op.factor])

evaluate(op::Clear, a::Arena, p::Program) = clear!(a.currents[op.current], op.components)

function evaluate(op::Closure, a::Arena, p::Program)
    cf = [p.factors[k] for k in op.component_factors]
    a.amplitudes[op.amplitude] = closure(a.amplitudes[op.amplitude], current(a, op.l), current(a, op.r), cf, p.factors[op.factor])
end

function kernel_value(i::KernelInput, op::Kernel, a::Arena, p::Program, factor::Complex{Float64})::Float64
    part(z) = i.im ? imag(z) : real(z)
    if i.kind == IN_L
        return part(a.currents[op.l][i.n])
    elseif i.kind == IN_R
        return part(a.currents[op.r][i.n])
    elseif i.kind == OUT
        return part(a.currents[op.out][i.n])
    elseif i.kind == MOMENTUM_U
        return form(a, op.u)[i.n]
    elseif i.kind == MOMENTUM_V
        return form(a, op.v)[i.n]
    elseif i.kind == SCALE
        return real(p.scales[i.n])
    else
        return part(factor)
    end
end

function evaluate(op::Kernel, a::Arena, p::Program)
    b = p.bindings[op.binding]
    factor = p.factors[op.factor]
    values = kernel_call(b.kernel, [kernel_value(i, op, a, p, factor) for i in b.inputs])
    kernel_store!(a.currents[op.out], values, b.outputs, b.direct, op.mode, factor)
end

# the momentum forms from the caller's momenta (momenta[leg] = (E, px, py, pz), 1-based legs);
# `a`: an Arena or an LArena, `p`: a Program or a Lowered (both have `forms` and `crossing`)
function fill_forms!(a, p, momenta)
    ext = p.crossing === nothing ? [Tuple(Float64.(m)) for m in momenta] :
          [Tuple(sign .* Float64.(momenta[leg])) for (leg, sign) in p.crossing]
    for (i, f) in enumerate(p.forms)
        v = (0.0, 0.0, 0.0, 0.0)
        for (t, (leg, coef)) in enumerate(f.terms)
            term = coef .* ext[leg]
            v = t == 1 ? 0.0 .+ term : v .+ term
        end
        a.forms[i] = v
    end
end

# the currents' values before the body: zero, or factor × the source's wavefunction
function fill_currents!(a::Arena, p::Program)
    for (i, c) in enumerate(p.currents)
        cur = a.currents[i]
        s = c.source
        if s === nothing
            fill!(cur, complex(0.0, 0.0))
        else
            w = wavefunction(s.family, s.orientation, a.forms[s.form], s.helicity, s.chirality, real(p.scales[s.mass]))
            factor = p.factors[s.factor]
            for j in eachindex(cur)
                cur[j] = factor * w[j]
            end
        end
    end
end

# The outputs of program `p` (one per helicity configuration, group factor applied) at one point:
# momenta[leg] = (E, px, py, pz), the caller's momenta (before the crossing), 1-based legs.
function evaluate(p::Program, a::Arena, momenta)::Vector{Float64}
    fill_forms!(a, p, momenta)
    fill_currents!(a, p)
    fill!(a.amplitudes, complex(0.0, 0.0))
    for op in p.body
        evaluate(op, a, p)
    end
    return [reduction([a.amplitudes[k] for (k, _) in terms], [w for (_, w) in terms], p.group_factor) for terms in p.reduction]
end
