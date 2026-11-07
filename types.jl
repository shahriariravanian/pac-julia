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

# the kind of a current: its wavefunction family, or what the instructions that write and read it
# make of it (Program.currents[i].kind, from the checks of check.jl)
@isdefined(CurrentKind) || @enum CurrentKind SCALAR WEYL DIRAC VECTOR TENSOR  # once, though included by several files

# the components of each kind
components(k::CurrentKind)::Int = k == SCALAR ? 1 : k == WEYL ? 2 : k == TENSOR ? 6 : 4

kind_name(k::CurrentKind)::String = ("scalar", "Weyl spinor", "Dirac spinor", "vector", "antisymmetric tensor")[Int(k)+1]

# The values of a current, by kind, and of a momentum form: immutable tuples (isbits: no heap
# allocation). They are aliases, so a Dirac spinor and a vector are the same type to Julia (their
# kinds are told apart by CurrentKind); the lengths tell the others apart.
const Scalar = NTuple{1,Complex{Float64}}
const Weyl = NTuple{2,Complex{Float64}}
const Dirac = NTuple{4,Complex{Float64}}
const Vec4 = NTuple{4,Complex{Float64}}
const Tensor = NTuple{6,Complex{Float64}}   # antisymmetric: components (01, 02, 03, 12, 13, 23)
const CurrentValue = Union{Scalar,Weyl,Vec4,Tensor}
const NoOperand = Tuple{}                   # a null operand: ()
const Momentum = NTuple{4,Float64}          # (E, px, py, pz)

const ZERO = complex(0.0, 0.0)
