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


###############################################################

struct ArenaIndex
    forms::Vector{Int}
    amplitudes::Vector{Int}
    currents::Vector{Int}
    outputs::Vector{Int}
    size::Int
end

function ArenaIndex(prog, a)
    forms = [4*i for i=0:length(a.forms)-1]
    size = 4 * length(forms)

    amplitudes = [size + 2*i for i=0:length(a.amplitudes)-1]
    size += 2 * length(amplitudes)

    currents = Int[]
    for c in a.currents
        push!(currents, size)
        size += 2 * length(c)
    end
    push!(currents, size)   # sentinel

    outputs = [size + 2*i for i=0:length(prog.helicities)-1]
    size += 2 * length(outputs)

    return ArenaIndex(forms, amplitudes, currents, outputs, size)
end

current_size(b::ArenaIndex, ci::CurrentIndex) = b.currents[ci+1] - b.currents[ci]
