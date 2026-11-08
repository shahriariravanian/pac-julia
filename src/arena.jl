# The values of one phase-space point while a Program runs. Every name of main.json is an index
# here (1-based): c<i> → currents[i + 1], a<i> → amplitudes[i + 1], q<i> → forms[i + 1]. The scales
# and factors are constants of the Program (Program.scales, Program.factors).

struct Arena
    forms::Vector{NTuple{4,Float64}}            # the momentum forms q<i>, (E, px, py, pz)
    currents::Vector{Vector{Complex{Float64}}}  # the currents c<i>, `components` each
    amplitudes::Vector{Complex{Float64}}        # the amplitudes a<i>
end

# an arena sized for programs of `forms` forms, currents of `components` components, `amplitudes`
# amplitudes; Arena(p::Program) in pacrun.jl
Arena(forms::Int, components::Vector{Int}, amplitudes::Int) =
    Arena(fill((0.0, 0.0, 0.0, 0.0), forms), [zeros(Complex{Float64}, n) for n in components], zeros(Complex{Float64}, amplitudes))

const NO_CURRENT = Complex{Float64}[]  # a null operand (an operand the instruction does not read)

current(a::Arena, i::CurrentIndex) = i == 0 ? NO_CURRENT : a.currents[i]
form(a::Arena, i::FormIndex) = a.forms[i]
