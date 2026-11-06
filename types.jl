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
