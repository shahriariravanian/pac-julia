# The values of one phase-space point while a Program runs. Every name of main.json is an index
# here (1-based): c<i> → currents[i + 1], a<i> → amplitudes[i + 1], q<i> → forms[i + 1]. The scales
# and factors are constants of the Program (Program.scales, Program.factors).

include(joinpath(@__DIR__, "types.jl"))

struct Arena
    forms::Vector{Momentum}               # the momentum forms q<i>, (E, px, py, pz)
    currents::Vector{CurrentValue}        # the currents c<i>, each a tuple of its kind (stored inline)
    amplitudes::Vector{Complex{Float64}}  # the amplitudes a<i>
end

# a current of `components` zeros: a tuple of its kind
zero_value(components::Int)::CurrentValue = ntuple(_ -> ZERO, components)

# an arena sized for programs of `forms` forms, currents of `components` components, `amplitudes`
# amplitudes; Arena(p::Program) in pacrun.jl
Arena(forms::Int, components::Vector{Int}, amplitudes::Int) =
    Arena(fill((0.0, 0.0, 0.0, 0.0), forms), CurrentValue[zero_value(n) for n in components], zeros(Complex{Float64}, amplitudes))

# a current's value; () for a null operand (index 0, an operand the instruction does not read)
value(a::Arena, i::CurrentIndex) = i == 0 ? () : a.currents[i]

# a current's new value (of the same kind)
set!(a::Arena, i::CurrentIndex, v::CurrentValue) = (a.currents[i] = v)

form(a::Arena, i::FormIndex)::Momentum = a.forms[i]
