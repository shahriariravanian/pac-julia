# A runner for pac2json folders: `load_artifact` reads one program of main.json into a Program
# (the template: every name resolved to an index), `evaluate` runs it on one phase-space point
# with an Arena (the values). The instructions are operations.jl's, the sources wavefunctions.jl's.
#
#   p = load_artifact("folder/main.json", 1)        # or load_programs("folder/main.json")
#   a = Arena(p)
#   outputs = evaluate(p, a, momenta)               # momenta[leg] = (E, px, py, pz), the caller's
using JSON
using Libdl

include("types.jl")
include("wavefunctions.jl")
include("operations.jl")
include("arena.jl")

# "c12" → 13 (1-based), nothing → 0 (a null operand)
index(name::Nothing, prefix::String) = 0
function index(name::AbstractString, prefix::String)::Int
    startswith(name, prefix) || error("$name is not a $prefix<i>")
    return parse(Int, name[length(prefix)+1:end]) + 1
end

ci(name)::CurrentIndex = index(name, "c")
qi(name)::FormIndex = index(name, "q")
ai(name)::AmplitudeIndex = index(name, "a")
si(name)::ScaleIndex = index(name, "s")
xi(name)::FactorIndex = index(name, "x")
bi(name)::BindingIndex = index(name, "b")

cplx(v) = complex(Float64(v[1]), Float64(v[2]))

#################################### Form ########################################

# q = Σ coef × p[leg] over the terms, the first term added to 0.0, in term order
struct Form
    terms::Vector{Tuple{LegIndex,Float64}}  # (leg, coefficient)
end

load_forms(prog) = [Form([(LegIndex(t[1]) + 1, Float64(t[2])) for t in f.terms]) for f in prog.inputs.forms]

################################### Current #########################################

# a source current: factor × the wavefunction of (family, orientation) at form `form`
struct CurrentSource
    family::String
    orientation::String
    form::FormIndex
    helicity::Helicity
    chirality::Chirality
    mass::ScaleIndex  # its real part is the mass
    factor::FactorIndex
end

struct Current
    components::Int
    kind::CurrentKind                     # checked at load (check.jl)
    source::Union{Nothing,CurrentSource}  # nothing: the current starts at zero
end

# a current before its kind is known (load_currents → check_program → Current)
struct CurrentSpec
    components::Int
    source::Union{Nothing,CurrentSource}
    initially::Bool  # has a value before the body: a source or "zero" (not "written")
end

function load_currents(prog)::Vector{CurrentSpec}
    currents = CurrentSpec[]
    for (i, c) in enumerate(prog.currents)
        ci(c.name) == i || error("currents[$i] is $(c.name)")
        init = c.initial
        if init == "source"
            s = c.source
            src = CurrentSource(s.family, s.orientation, qi(s.momentum), s.helicity, s.chirality, si(s.mass), xi(s.factor))
            push!(currents, CurrentSpec(c.components, src, true))
        elseif init == "zero" || init == "written"  # written: the body's first write is "=" (checked)
            push!(currents, CurrentSpec(c.components, nothing, init == "zero"))
        else
            error("current $(c.name): initial \"$init\" is not supported")
        end
    end
    return currents
end

###################################### Kernel #########################################

# a binding input: a part of an operand's component (IN_L, IN_R, OUT), a momentum component
# (MOMENTUM_U, MOMENTUM_V), the real part of a scale, a part of the row's factor
@enum InputKind IN_L IN_R OUT MOMENTUM_U MOMENTUM_V SCALE FACTOR

struct KernelInput
    kind::InputKind
    n::Int      # a ComponentIndex, or the ScaleIndex of SCALE (0 for FACTOR)
    im::Bool    # the imaginary part
end

function load_input(code::AbstractString)::KernelInput
    if (m = match(r"^(re|im) in\[([01])\]\[(\d+)\]$", code)) !== nothing
        return KernelInput(m[2] == "0" ? IN_L : IN_R, parse(ComponentIndex, m[3]) + 1, m[1] == "im")
    elseif (m = match(r"^(re|im) out\[(\d+)\]$", code)) !== nothing
        return KernelInput(OUT, parse(ComponentIndex, m[2]) + 1, m[1] == "im")
    elseif (m = match(r"^momenta\[([01])\]\[(\d+)\]$", code)) !== nothing
        return KernelInput(m[1] == "0" ? MOMENTUM_U : MOMENTUM_V, parse(ComponentIndex, m[2]) + 1, false)
    elseif (m = match(r"^s(\d+)$", code)) !== nothing
        return KernelInput(SCALE, parse(ScaleIndex, m[1]) + 1, false)
    elseif (m = match(r"^(re|im) factor$", code)) !== nothing
        return KernelInput(FACTOR, 0, m[1] == "im")
    else
        error("unknown binding input $code")
    end
end

struct Binding
    kernel::KernelLibrary
    direct::Bool
    inputs::Vector{KernelInput}
    outputs::Vector{ComponentIndex}  # the components of out, one per output pair
end

# the kernels of main.json, by name; each needs its shared library (pac2json --sjb2obj)
function load_kernels(main, path)
    kernels = Dict{String,KernelLibrary}()
    for k in main.kernels
        lib = k.library
        lib === nothing && error("kernel $(k.name) has no library: run pac2json with --sjb2obj")
        kernels[k.name] = load_kernel(joinpath(dirname(path), lib.path), lib.symbol, lib.states, lib.outputs)
    end
    return kernels
end

function load_bindings(prog, kernels)
    bindings = Binding[]
    for (i, b) in enumerate(prog.bindings)
        bi(b.name) == i || error("bindings[$i] is $(b.name)")
        push!(bindings, Binding(kernels[b.kernel], b.direct, [load_input(c) for c in b.inputs],
                              [ComponentIndex(j) + 1 for j in b.outputs]))
    end
    return bindings
end

##################################### Operations ######################################

abstract type Operation end

struct Vertex <: Operation
    type::String
    out::CurrentIndex
    mode::String
    l::CurrentIndex
    r::CurrentIndex
    u::FormIndex  # momenta[0]
    v::FormIndex  # momenta[1]
    coupling::ScaleIndex
    factor::FactorIndex
end

struct Propagator <: Operation
    type::String
    current::CurrentIndex
    momentum::FormIndex
    factor::FactorIndex
end

struct Scale <: Operation
    current::CurrentIndex
    components::Int
    factor::FactorIndex
end

struct Clear <: Operation
    current::CurrentIndex
    components::Int
end

struct Closure <: Operation
    amplitude::AmplitudeIndex
    l::CurrentIndex
    r::CurrentIndex
    component_factors::Vector{FactorIndex}
    factor::FactorIndex
end

struct Kernel <: Operation
    binding::BindingIndex
    out::CurrentIndex
    mode::String
    l::CurrentIndex
    r::CurrentIndex
    u::FormIndex  # momenta[0]
    v::FormIndex  # momenta[1]
    factor::FactorIndex
end

function load_operation(o)::Operation
    op = o.op
    if op == "vertex"
        return Vertex(o.type, ci(o.out), o.mode, ci(o.in[1]), ci(o.in[2]), qi(o.momenta[1]), qi(o.momenta[2]),
                      si(o.coupling), xi(o.factor))
    elseif op == "propagator"
        return Propagator(o.type, ci(o.current), qi(o.momentum), xi(o.factor))
    elseif op == "scale"
        return Scale(ci(o.current), o.components, xi(o.factor))
    elseif op == "clear"
        return Clear(ci(o.current), o.components)
    elseif op == "closure"
        return Closure(ai(o.amplitude), ci(o.in[1]), ci(o.in[2]), [xi(f) for f in o.component_factors], xi(o.factor))
    elseif op == "kernel"
        return Kernel(bi(o.binding), ci(o.out), o.mode, ci(o.in[1]), ci(o.in[2]), qi(o.momenta[1]), qi(o.momenta[2]),
                      xi(o.factor))
    else
        error("unknown instruction $op")
    end
end

load_body(prog)::Vector{Operation} = [load_operation(o) for o in prog.body]

include("check.jl")

###################################### Program ########################################

struct Program
    process::String
    colour::String
    expression::String
    legs::Int
    crossing::Union{Nothing,Vector{Tuple{LegIndex,Float64}}}  # p<l> = sign × the caller's leg
    forms::Vector{Form}
    scales::Vector{Complex{Float64}}
    factors::Vector{Complex{Float64}}
    alpha_s::Union{Nothing,Tuple{Float64,Int}}  # (value, power): the scales are at this value
    helicities::Vector{Vector{Helicity}}        # output h: the helicity of each leg
    currents::Vector{Current}
    amplitudes::Int                             # the number of amplitudes
    bindings::Vector{Binding}
    body::Vector{Operation}
    reduction::Vector{Vector{Tuple{AmplitudeIndex,Float64}}}  # output h: (amplitude, weight) terms
    group_factor::Tuple{Int,Int}
end

count_legs(p::Program) = p.legs

Arena(p::Program) = Arena(length(p.forms), [c.components for c in p.currents], p.amplitudes)

function load_program(prog, kernels)::Program
    inputs = prog.inputs
    crossing = inputs.crossing === nothing ? nothing : [(LegIndex(c.leg) + 1, Float64(c.sign)) for c in inputs.crossing]
    for (k, s) in enumerate(inputs.scales)
        si(s.name) == k || error("scales[$k] is $(s.name)")
    end
    for (k, f) in enumerate(inputs.factors)
        xi(f.name) == k || error("factors[$k] is $(f.name)")
    end
    for (k, a) in enumerate(prog.amplitudes)
        ai(a.name) == k && a.initial == 0 || error("amplitudes[$k] is $(a.name), initial $(a.initial)")
    end
    reduction = Vector{Tuple{AmplitudeIndex,Float64}}[]
    for (h, r) in enumerate(prog.reduction)
        r.output == h - 1 || error("reduction[$h] is output $(r.output)")
        terms = [(ai(t[1]), Float64(t[2])) for t in r.terms]
        all(in_range(k, length(prog.amplitudes)) for (k, _) in terms) || throw(ProgramError("reduction[$h]: an amplitude out of range"))
        push!(reduction, terms)
    end
    legs = inputs.legs
    forms = load_forms(prog)
    all(in_range(l, legs) for f in forms for (l, _) in f.terms) || throw(ProgramError("a momentum form of a leg out of range"))
    crossing === nothing || all(in_range(l, legs) for (l, _) in crossing) || throw(ProgramError("a crossing of a leg out of range"))
    specs = load_currents(prog)
    bindings = load_bindings(prog, kernels)
    body = load_body(prog)
    kinds = check_program(specs, body, bindings; forms = length(forms), scales = length(inputs.scales),
                          factors = length(inputs.factors), amplitudes = length(prog.amplitudes))
    currents = [Current(s.components, k, s.source) for (s, k) in zip(specs, kinds)]
    out = prog.output
    alpha_s = inputs.alpha_s === nothing ? nothing : (Float64(inputs.alpha_s.value), Int(inputs.alpha_s.power))
    return Program(
        prog.process,
        prog.colour,
        prog.expression,
        legs,
        crossing,
        forms,
        [cplx(s.value) for s in inputs.scales],
        [cplx(f.value) for f in inputs.factors],
        alpha_s,
        [h.legs === nothing ? Helicity[] : Helicity.(h.legs) for h in out.helicities],
        currents,
        length(prog.amplitudes),
        bindings,
        body,
        reduction,
        (Int(out.group_factor[1]), Int(out.group_factor[2])),
    )
end

read_main(path::String) = JSON.parse(read(path, String))

# program `program` (1-based) of main.json at `path`
function load_artifact(path::String, program::Int = 1)::Program
    main = read_main(path)
    return load_program(main.programs[program], load_kernels(main, path))
end

# every program of main.json at `path`
function load_programs(path::String)::Vector{Program}
    main = read_main(path)
    kernels = load_kernels(main, path)
    return [load_program(prog, kernels) for prog in main.programs]
end

###################################### evaluate ########################################

# The arena holds each current as a tuple of its kind: `value(a, i)` reads it (() for a null
# operand), `set!(a, i, v)` replaces it (arena.jl). An arena element has the union type
# CurrentValue; the operations get concrete types (no boxing, no dynamic dispatch): a vertex or a
# propagator by its type, whose operand kinds the load-time checks guarantee (the type assertions
# never fail), the other instructions through `concrete`.

# f(v) with v's concrete type
@inline concrete(f, v::CurrentValue) = v isa Weyl ? f(v) : v isa Vec4 ? f(v) : v isa Tensor ? f(v) : f(v::Scalar)

function evaluate(op::Vertex, a::Arena, p::Program)
    (c, t) = (a.currents, op.type)
    (u, v) = (form(a, op.u), form(a, op.v))
    scale = p.scales[op.coupling] * p.factors[op.factor]
    if t == "weyl_vector_a"
        set!(a, op.out, assign(op.mode, c[op.out]::Weyl, vertex_weyl_vector_a(c[op.l]::Weyl, c[op.r]::Vec4), scale))
    elseif t == "weyl_vector_b"
        set!(a, op.out, assign(op.mode, c[op.out]::Weyl, vertex_weyl_vector_b(c[op.l]::Weyl, c[op.r]::Vec4), scale))
    elseif t == "tensor_vector"
        set!(a, op.out, assign(op.mode, c[op.out]::Vec4, vertex_tensor_vector(c[op.l]::Tensor, c[op.r]::Vec4), scale))
    elseif t == "vector_wedge"
        set!(a, op.out, assign(op.mode, c[op.out]::Tensor, vertex_vector_wedge(c[op.l]::Vec4, c[op.r]::Vec4), scale))
    elseif t == "three_vector"
        set!(a, op.out, assign(op.mode, c[op.out]::Vec4, vertex_three_vector(c[op.l]::Vec4, c[op.r]::Vec4, u, v), scale))
    else
        error("unknown vertex type $t")
    end
    return nothing
end

function evaluate(op::Propagator, a::Arena, p::Program)
    (c, i, t) = (a.currents, op.current, op.type)
    (k, f) = (form(a, op.momentum), p.factors[op.factor])
    if t == "weyl_a"
        set!(a, i, propagator_weyl_a(c[i]::Weyl, k, f))
    elseif t == "weyl_b"
        set!(a, i, propagator_weyl_b(c[i]::Weyl, k, f))
    elseif t == "vector"
        set!(a, i, propagator_vector(c[i]::Vec4, k, f))
    else
        error("unknown propagator type $t")
    end
    return nothing
end

function evaluate(op::Scale, a::Arena, p::Program)
    concrete(x -> set!(a, op.current, scale(x, op.components, p.factors[op.factor])), a.currents[op.current])
    return nothing
end

function evaluate(op::Clear, a::Arena, p::Program)
    concrete(x -> set!(a, op.current, clear(x, op.components)), a.currents[op.current])
    return nothing
end

function evaluate(op::Closure, a::Arena, p::Program)
    (cf, f) = (view(p.factors, op.component_factors), p.factors[op.factor])
    amplitude = a.amplitudes[op.amplitude]
    a.amplitudes[op.amplitude] = concrete(a.currents[op.l]) do l
        concrete(r -> closure(amplitude, l, r, cf, f), a.currents[op.r])
    end
    return nothing
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
    lib = b.kernel
    for (k, i) in enumerate(b.inputs)
        lib.input[k] = kernel_value(i, op, a, p, factor)
    end
    values = kernel_call!(lib)
    concrete(out -> set!(a, op.out, kernel_store(out, values, b.outputs, b.direct, op.mode, factor)), a.currents[op.out])
    return nothing
end

# the momentum forms from the caller's momenta (momenta[leg] = (E, px, py, pz), 1-based legs)
function fill_forms!(a::Arena, p::Program, momenta)
    # p<l>: the caller's leg, or sign × the leg it crosses from
    function external(l::LegIndex)::Momentum
        p.crossing === nothing && return Momentum(momenta[l])
        (leg, sign) = p.crossing[l]
        return sign .* Momentum(momenta[leg])
    end
    for (i, f) in enumerate(p.forms)
        v = (0.0, 0.0, 0.0, 0.0)
        for (t, (leg, coef)) in enumerate(f.terms)
            term = coef .* external(leg)
            v = t == 1 ? 0.0 .+ term : v .+ term
        end
        a.forms[i] = v
    end
end

# the currents' values before the body: zero, or factor × the source's wavefunction
function fill_currents!(a::Arena, p::Program)
    for (i, c) in enumerate(p.currents)
        s = c.source
        if s === nothing
            set!(a, i, zero_value(c.components))
        else
            w = wavefunction(s.family, s.orientation, a.forms[s.form], s.helicity, s.chirality, real(p.scales[s.mass]))
            factor = p.factors[s.factor]
            set!(a, i, map(x -> factor * x, w))
        end
    end
end

# The body, each instruction by its concrete type: static calls (a call dispatched at run time
# would box the Program and the Arena, immutable structs, 192 bytes per instruction)
function run_body!(a::Arena, p::Program)
    for op in p.body
        if op isa Vertex
            evaluate(op, a, p)
        elseif op isa Propagator
            evaluate(op, a, p)
        elseif op isa Kernel
            evaluate(op, a, p)
        elseif op isa Closure
            evaluate(op, a, p)
        elseif op isa Scale
            evaluate(op, a, p)
        elseif op isa Clear
            evaluate(op, a, p)
        else
            error("unknown instruction $(typeof(op))")
        end
    end
end

# The outputs of program `p` (one per helicity configuration, group factor applied) at one point:
# momenta[leg] = (E, px, py, pz), the caller's momenta (before the crossing), 1-based legs.
function evaluate(p::Program, a::Arena, momenta)::Vector{Float64}
    fill_forms!(a, p, momenta)
    fill_currents!(a, p)
    fill!(a.amplitudes, complex(0.0, 0.0))
    run_body!(a, p)
    outputs = Vector{Float64}(undef, length(p.reduction))
    for (h, terms) in enumerate(p.reduction)
        outputs[h] = reduction((a.amplitudes[k] for (k, _) in terms), (w for (_, w) in terms), p.group_factor)
    end
    return outputs
end
