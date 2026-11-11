# The Program of a main.json entry (program.jl): its currents, operations (each naming its
# operands by index), bindings and reduction. The indices and the shared types (Form,
# KernelLibrary, KernelInput) are ../types.jl's.

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
    source::Union{Nothing,CurrentSource}  # nothing: the current starts at zero
end

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

struct Binding
    kernel::KernelLibrary
    direct::Bool
    inputs::Vector{KernelInput}
    outputs::Vector{ComponentIndex}  # the components of out, one per output pair
end

struct Program
    process::String
    colour::String
    expression::Union{Nothing,String}  # nothing: a process artifact.json does not list
    legs::Int
    crossing::Union{Nothing,Vector{Tuple{LegIndex,Float64}}}  # p<l> = sign × the caller's leg
    forms::Vector{Form}
    scales::Vector{Complex{Float64}}
    factors::Vector{Complex{Float64}}
    alpha_s::Union{Nothing,Tuple{Float64,Union{Int,Nothing}}}  # (value, power): the scales are at this value;
                                                # power nothing: not measured (alpha_s.jl)
    helicities::Vector{Vector{Helicity}}        # output h: the helicity of each leg
    currents::Vector{Current}
    amplitudes::Int                             # the number of amplitudes
    bindings::Vector{Binding}
    body::Vector{Operation}
    reduction::Vector{Vector{Tuple{AmplitudeIndex,Float64}}}  # output h: (amplitude, weight) terms
    group_factor::Tuple{Int,Int}
end
