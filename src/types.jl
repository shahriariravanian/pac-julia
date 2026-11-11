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

struct SymjitSlice  # symjit_slice = {double *ptr; size_t len}
    ptr::Ptr{Float64}
    len::Csize_t
end

# a kernel's shared library (main.json kernels[i].library): its function and its counts
struct KernelLibrary
    name::String
    path::String       # the library sjb2obj built (the Julia runner calls it)
    source::String     # the kernel's .plane.symjit (the compiled runners' C is made from it: kernel_c.jl)
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
