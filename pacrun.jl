using JSON
using Libdl

include("arena.jl")
include("wavefunctions.jl")

#################################### Form ########################################

struct Term
    var::String
    coef::Float64
end

struct Form
    name::String
    value::String
    terms::Vector{Term}
end

function load_forms(prog)
    forms = Form[]

    for form in prog.inputs.forms
        terms = Term[]

        for term in form.terms
            push!(terms, Term("p$(term[1])", term[2]))
        end

        push!(forms, Form(form.name, form.value, terms))
    end

    return forms
end

#################################### Scale #########################################

struct Scale
    name::String
    value::Complex{Float64}
end

function load_scales(prog)
    scales = Scale[]

    for scale in prog.inputs.scales
        push!(scales, Scale(scale.name, complex(scale.value[1], scale.value[2])))
    end

    return scales
end

################################### Factor ########################################

struct Factor
    name::String
    value::Complex{Float64}
end

function load_factors(prog)
    factors = Factor[]

    for factor in prog.inputs.factors
        push!(factors, Factor(factor.name, complex(factor.value[1], factor.value[2])))
    end

    return factors
end

################################ Helicity ###########################################

struct Helicity
    legs::Vector{Int}
end

function load_helicities(prog)
    helicities = Helicity[]

    for h in prog.output.helicities
        push!(helicities, Helicity(h.legs))
    end

    return helicities
end

################################### Current #########################################

struct Current
    name::String
    components::Int
    momentum::Union{Nothing,String}
    source::Union{Nothing,Source}
end

function load_currents(prog)
    currents = Current[]

    for current in prog.currents
        if current.initial == "source"
            source = Source(
                current.source.wavefunction,
                current.source.family,
                current.source.orientation,
                current.source.leg,
                current.source.momentum,
                current.source.helicity,
                current.source.chirality,
                current.source.mass,
                current.source.factor
            )

            push!(currents, Current(
                current.name,
                current.components,
                current.momentum,
                source
            ))
        else
            push!(currents, Current(
                current.name,
                current.components,
                current.momentum,
                nothing
            ))
        end
    end

    return currents
end

function apply_wavefunctions(prog, arena)
    for c in prog.currents
        if c.source == nothing
            set_vector(arena, c.name, zeros(Complex{Float64}, c.components))
        else

        end
    end
end

##################################### Amplitude #######################################

struct Amplitude
   name::String
   initial::Complex{Float64}
end

function load_amplitudes(prog)
    amplitudes = Amplitude[]

    for a in prog.amplitudes
        push!(amplitudes, Amplitude(a.name, a.initial))
    end

    return amplitudes
end

###################################### Kernel #########################################

struct Func
    lib::Ptr{Nothing}
    sym::Ptr{Nothing}
    path::String
    count_inputs::Int
    count_outputs::Int
    mem::Vector{Float64}
end

function load_kernels(main, path)
    kernels = Dict{String, Func}()

    for k in main.kernels
        q = joinpath(splitpath(path)[1:end-1]..., k.library.path)
        lib = Libdl.dlopen(q)
        sym = Libdl.dlsym(lib, Symbol(k.library.symbol))
        mem = zeros(k.library.mem)
        kernel = Func(lib, sym, q, k.library.states, k.library.outputs, mem)
        kernels[k.name] = kernel
    end

    return kernels
end

struct Binding
    func::Func
    direct::Bool
    inputs::Vector{String}
    outputs::Vector{Int}
end

function load_bindings(prog, kernels)
    bindings = Dict{String, Binding}()

    for b in prog.bindings
        func = kernels[b.kernel]
        bindings[b.name] = Binding(func, b.direct, b.inputs, b.outputs)
    end

    return bindings
end


##################################### Operations ######################################

abstract type Operation end

struct Vertex <: Operation
    ty::String
    out::String
    mode::String
    in::Vector{String}
    momenta::Vector{String}
    coupling::String
    factor::String
end

function load_vertex(oper)
    return Vertex(
        oper.type,
        oper.out,
        oper.mode,
        oper.in,
        oper.momenta,
        oper.coupling,
        oper.factor,
    )
end


struct Propagator <: Operation
    ty::String
    current::String
    momentum::String
    factor::String
end

function load_propagator(oper)
    return Propagator(
        oper.type,
        oper.current,
        oper.momentum,
        oper.factor,
    )
end


struct Closure <: Operation
    amplitude::String
    in::Vector{String}
    components::Int
    component_factors::Vector{String}
    factor::String
end

function load_closure(oper)
    return Closure(
        oper.amplitude,
        oper.in,
        oper.components,
        oper.component_factors,
        oper.factor,
    )
end


struct Kernel <: Operation
    func::Func
    out::String
    mode::String
    inputs::Vector{String}
    outputs::Vector{String}
    factor::String
end

function load_kernel(oper, bindings)
    b = bindings[oper.binding]

    d = Dict{String, String}()
    for (i, v) in enumerate(oper.in)
        d["in[$(i-1)]"] = v
    end

    inputs = [replace(v, d...) for v in b.inputs]

    outputs = []
    for i in 0:b.func.count_outputs÷2-1
        push!(outputs, "re $(oper.out)[$i]")
        push!(outputs, "im $(oper.out)[$i]")
    end

    return Kernel(
        b.func,
        oper.out,
        oper.mode,
        inputs,
        outputs,
        oper.factor
    )
end


function load_body(prog, bindings)::Vector{Operation}
    body = Operation[]

    for oper in prog.body
        op = oper.op

        if op == "vertex"
            push!(body, load_vertex(oper))
        elseif op == "propagator"
            push!(body, load_propagator(oper))
        elseif op == "closure"
            push!(body, load_closure(oper))
        elseif op == "kernel"
            push!(body, load_kernel(oper, bindings))
        else
            error("op is not defined")
        end
    end

    return body
end


###################################### Program ########################################

struct Program
    path::String
    process::String
    expression::String
    incoming_legs::Int
    outgoint_legs::Int
    forms::Vector{Form}
    scales::Vector{Scale}
    factors::Vector{Factor}
    alpha_s_value::Float64
    alpha_s_power::Int
    helicities::Vector{Helicity}
    currents::Vector{Current}
    amplitudes::Vector{Amplitude}
    kernels::Dict{String, Func}
    bindings::Dict{String, Binding}
    body::Vector{Operation}
end

count_legs(p::Program) = p.incoming_legs + p.outgoint_legs

function load_artifact(path::String, program=1)
    fd = open(path, "r")
    json = String(read(fd))
    close(fd)
    main = JSON.parse(json)

    prog = main.programs[program]

    incoming_legs = sum(1 for q in prog.inputs.particles if q.side == "initial")
    outgoing_legs = sum(1 for q in prog.inputs.particles if q.side == "final")

    forms = load_forms(prog)
    scales = load_scales(prog)
    factors = load_factors(prog)
    helicities = load_helicities(prog)
    currents = load_currents(prog)
    amplitudes = load_amplitudes(prog)
    kernels = load_kernels(main, path)
    bindings = load_bindings(prog, kernels)
    body = load_body(prog, bindings)

    return Program(
        path,
        prog.process,
        prog.expression,
        incoming_legs,
        outgoing_legs,
        forms,
        scales,
        factors,
        prog.inputs.alpha_s.value,
        prog.inputs.alpha_s.power,
        helicities,
        currents,
        amplitudes,
        kernels,
        bindings,
        body
    )
end
