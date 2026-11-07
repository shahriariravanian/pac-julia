# Names for the integers of a program. They are aliases of Int (documentation, not checked
# types): a FormIndex and a CurrentIndex are interchangeable to Julia. Every index is 1-based:
# main.json's q<i> is FormIndex i + 1, c<i> is CurrentIndex i + 1, and so on.

const LegIndex = Int        # an external leg (main.json's leg l → l + 1)
const FormIndex = Int       # a momentum form q<i>
const CurrentIndex = Int    # a current c<i>; 0: a null operand (one the instruction does not read)
const AmplitudeIndex = Int  # an amplitude a<i>
const ScaleIndex = Int      # a scale s<i>
const FactorIndex = Int     # a factor x<i>
const BindingIndex = Int    # a symjit binding b<i>
const ComponentIndex = Int  # a component of a current or a momentum (main.json's [j] → j + 1)

const Helicity = Int        # −1, +1 (and 0: the longitudinal state of a massive vector, a scalar)
const Chirality = Int       # −1 (left), +1 (right) for a Weyl source; 0 otherwise

# q = Σ coef × p[leg] over the terms, the first term added to 0.0, in term order
struct Form
    terms::Vector{Tuple{LegIndex,Float64}}  # (leg, coefficient)
end

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

struct SymjitSlice  # symjit_slice = {double *ptr; size_t len}
    ptr::Ptr{Float64}
    len::Csize_t
end

# a kernel's shared library (main.json kernels[i].library): its function and its counts
struct KernelLibrary
    name::String
    path::String
    handle::Ptr{Cvoid}
    fn::Ptr{Cvoid}
    inputs::Int    # library.states
    outputs::Int   # library.outputs: 2 × the output pairs
end

# a binding input: a part of an operand's component (IN_L, IN_R, OUT), a momentum component
# (MOMENTUM_U, MOMENTUM_V), the real part of a scale, a part of the row's factor
@enum InputKind IN_L IN_R OUT MOMENTUM_U MOMENTUM_V SCALE FACTOR

struct KernelInput
    kind::InputKind
    n::Int      # a ComponentIndex, or the ScaleIndex of SCALE (0 for FACTOR)
    im::Bool    # the imaginary part
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
    alpha_s::Union{Nothing,Tuple{Float64,Int}}  # (value, power): the scales are at this value
    helicities::Vector{Vector{Helicity}}        # output h: the helicity of each leg
    currents::Vector{Current}
    amplitudes::Int                             # the number of amplitudes
    bindings::Vector{Binding}
    body::Vector{Operation}
    reduction::Vector{Vector{Tuple{AmplitudeIndex,Float64}}}  # output h: (amplitude, weight) terms
    group_factor::Tuple{Int,Int}
end
