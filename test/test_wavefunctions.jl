# wavefunctions.jl against RustiCol's wavefunctions.rs, bit for bit: every source family,
# orientation, energy sign, helicity and chirality, on random momenta and the edge cases (along
# ±z, along y, zero 3-momentum), each through apply_source with a random factor.
# The reference is wavefunctions_ref.rs (the f64 functions of rusticol-core/src/wavefunctions.rs,
# dispatched and multiplied by the factor as pacuda's runner does); this script builds it with rustc.
# usage: julia test_wavefunctions.jl [CASES_PER_KIND]
using Random

include(joinpath(@__DIR__, "..", "wavefunctions.jl"))

# a stand-in arena: one source's momentum, mass scale and factor
struct TestArena
    k::NTuple{4,Float64}
    mass::Complex{Float64}
    factor::Complex{Float64}
end
get_vector(a::TestArena, name) = a.k
get_scale(a::TestArena, name) = a.mass
get_factor(a::TestArena, name) = a.factor

const KINDS = [  # (wavefunction, family, orientation, masses, helicities)
    ("scalar", "scalar", "particle", [0.0], [0]),
    ("weyl particle", "weyl", "particle", [0.0], [-1, 1]),
    ("weyl antiparticle", "weyl", "antiparticle", [0.0], [-1, 1]),
    ("dirac particle", "dirac", "particle", [0.0, 4.7, 173.0], [-1, 1]),
    ("dirac antiparticle", "dirac", "antiparticle", [0.0, 4.7, 173.0], [-1, 1]),
    ("vector (massless)", "vector", "particle", [0.0], [-1, 1]),
    ("vector (massive)", "vector", "particle", [80.4, 91.188], [-1, 0, 1]),
]
const CODE = Dict("scalar" => 0, "weyl" => 1, "dirac" => 2, "vector" => 3)

function momentum(rng, m)
    r = rand(rng)
    p3 = if r < 0.15
        (0.0, 0.0, rand(rng, (-1, 1)) * (1 + 499 * rand(rng)))
    elseif r < 0.2 && m != 0
        (0.0, 0.0, 0.0)
    elseif r < 0.25
        (0.0, 1 + 499 * rand(rng), 0.0)
    else
        Tuple(1000 .* rand(rng, 3) .- 500)
    end
    E = √(p3[1]^2 + p3[2]^2 + p3[3]^2 + m^2)
    sg = rand(rng, (-1.0, 1.0))  # E < 0: an incoming leg (its form is minus its momentum)
    return (sg * E, sg * p3[1], sg * p3[2], sg * p3[3])
end

hex(x) = string(reinterpret(UInt64, Float64(x)), base = 16, pad = 16)
unhex(s) = reinterpret(Float64, parse(UInt64, s, base = 16))
same(a, b) = reinterpret(UInt64, a) == reinterpret(UInt64, b)

function main(per_kind)
    rng = MersenneTwister(7)
    cases = []
    for (name, family, orientation, masses, hs) in KINDS, _ in 1:per_kind
        m = rand(rng, masses)
        k = momentum(rng, m)
        h = rand(rng, hs)
        ch = family == "weyl" ? rand(rng, (-1, 1)) : 0
        f = rand(rng) < 0.2 ? complex(1.0, 0.0) : complex(randn(rng), randn(rng))
        s = Source(name, family, orientation, nothing, "q0", h, ch, "s0", "x0")
        push!(cases, (name, s, TestArena(k, complex(m, 0.0), f)))
    end

    ref = joinpath(mktempdir(), "wavefunctions_ref")
    run(`rustc -O --edition 2021 $(joinpath(@__DIR__, "wavefunctions_ref.rs")) -o $ref`)
    input = IOBuffer()
    for (_, s, a) in cases
        fields = [CODE[s.family], s.orientation == "antiparticle" ? 1 : 0, s.helicity, s.chirality,
                  hex(real(a.mass)), hex.(a.k)..., hex(real(a.factor)), hex(imag(a.factor))]
        println(input, join(fields, " "))
    end
    lines = readlines(pipeline(`$ref`, stdin = IOBuffer(take!(input))))
    length(lines) == length(cases) || error("the reference gave $(length(lines)) lines")

    counts, failures = Dict{String,Int}(), Dict{String,Int}()
    for ((name, s, a), line) in zip(cases, lines)
        t = split(line)
        n = parse(Int, t[1])
        want = [complex(unhex(t[2j]), unhex(t[2j+1])) for j in 1:n]
        got = apply_source(s, a)
        counts[name] = get(counts, name, 0) + 1
        ok = length(got) == n && all(same(real(g), real(w)) && same(imag(g), imag(w)) for (g, w) in zip(got, want))
        if !ok
            failures[name] = get(failures, name, 0) + 1
            if failures[name] <= 2
                println("  differs: $name k=$(a.k) h=$(s.helicity) ch=$(s.chirality) m=$(real(a.mass)) factor=$(a.factor)")
                println("    julia $got\n    rust  $want")
            end
        end
    end
    for (name,) in KINDS
        println(rpad(name, 20), lpad(counts[name], 6), " cases, ", get(failures, name, 0), " differ")
    end
    return isempty(failures) ? 0 : 1
end

exit(main(length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2000))
