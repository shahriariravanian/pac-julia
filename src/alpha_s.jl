# The alpha_s power of a program: p in |M|² ∝ alpha_s^p, measured as pac2json used to measure it
# with RustiCol (pac-miner's pac2json evaluates nothing and writes the power as null): the same
# programs recorded at alpha_s0 and at 2 alpha_s0, run by the Julia runner at fixed phase-space
# points; every point's ratio of totals must be (2 alpha_s0 / alpha_s0)^p for one integer p, to
# 1e-12, else the program is refused (its outputs are not one power of alpha_s).
#
#   alpha_s_power(base, doubled)      two folders (or main.json files) of the same programs
#   alpha_s_power(artifact)           a pyAmpliCol LC artifact: captured as it is and with its
#                                     alpha_s parameter doubled (capture's parameters)
#   load_programs(artifact; measure_alpha_s = true)  the programs with their power filled in

"""
    rambo_momenta(n, points; seed) -> Vector{Vector{NTuple{4,Float64}}}

`points` phase-space points of `n` massless legs at sqrt(s) = 1000: two incoming beams along z,
`n - 2` outgoing legs by RAMBO, from a fixed seed (pac2json's `momenta`, not bit for bit).
"""
function rambo_momenta(n::Int, points::Int; seed::UInt64 = 0x00000000_0000a1fa)
    state = seed | 1
    function uniform()
        state ⊻= state << 13
        state ⊻= state >> 7
        state ⊻= state << 17
        return (Float64(state >> 11) + 0.5) / 2.0^53
    end
    ecm = 1000.0
    out = Vector{Vector{NTuple{4,Float64}}}()
    for _ in 1:points
        q = map(1:n-2) do _
            c = 2.0 * uniform() - 1.0
            s = sqrt(1.0 - c * c)
            f = 2.0 * pi * uniform()
            e = -log(uniform() * uniform())
            (e, e * s * cos(f), e * s * sin(f), e * c)
        end
        total = ntuple(k -> sum(p[k] for p in q), 4)
        mass = sqrt(total[1]^2 - total[2]^2 - total[3]^2 - total[4]^2)
        b = (-total[2] / mass, -total[3] / mass, -total[4] / mass)
        x, g = ecm / mass, total[1] / mass
        a = 1.0 / (1.0 + g)
        legs = NTuple{4,Float64}[(ecm / 2, 0.0, 0.0, ecm / 2), (ecm / 2, 0.0, 0.0, -ecm / 2)]
        for p in q
            bq = b[1] * p[2] + b[2] * p[3] + b[3] * p[4]
            push!(legs, (x * (g * p[1] + bq), ntuple(k -> x * (p[k+1] + b[k] * (p[1] + a * bq)), 3)...))
        end
        push!(out, legs)
    end
    return out
end

# the power of one program from its copies at alpha_s0 (`p`) and alpha_s1 (`q`)
function measure_alpha_s_power(p::Program, q::Program; points::Int = 16)::Int
    what = "$(p.process) $(p.colour)"
    (p.process, p.colour) == (q.process, q.colour) || error("alpha_s_power: $what is paired with $(q.process) $(q.colour)")
    (p.alpha_s === nothing || q.alpha_s === nothing) && error("alpha_s_power: $what has no alpha_s")
    a0, a1 = p.alpha_s[1], q.alpha_s[1]
    a0 > 0 && a1 > 0 && a0 != a1 || error("alpha_s_power: $what is recorded at alpha_s $a0 and $a1")
    base, other = Arena(p), Arena(q)
    ratios = Float64[]
    for m in rambo_momenta(p.legs, points)
        t0 = sum(evaluate(p, base, m))
        t1 = sum(evaluate(q, other, m))
        t0 == 0.0 || push!(ratios, t1 / t0)
    end
    isempty(ratios) && error("alpha_s_power: $what: every total is zero")
    power = round(Int, log(ratios[1]) / log(a1 / a0))
    want = (a1 / a0)^power
    bad = findfirst(r -> abs(r / want - 1.0) > 1e-12, ratios)
    bad === nothing ||
        error("alpha_s_power: $what: |M|² is not a power of alpha_s (ratio $(ratios[bad]) at alpha_s × $(a1 / a0), $(a1 / a0)^$power = $want)")
    return power
end

"""
    alpha_s_power(base, doubled; kernels = :auto, points = 16) -> Vector{Union{Int,Nothing}}

Per program of `base` (a pac2json folder or main.json), the power p of |M|² ∝ alpha_s^p, from
the same programs recorded at another alpha_s (`doubled`); `nothing` for a program without
alpha_s. The programs run on the Julia runner (`kernels` as for load_programs).
"""
function alpha_s_power(base::AbstractString, doubled::AbstractString; kernels::Symbol = :auto, points::Int = 16)
    ps = load_programs(base; kernels)
    qs = load_programs(doubled; kernels)
    length(ps) == length(qs) || error("alpha_s_power: $(length(ps)) programs and $(length(qs))")
    return Union{Int,Nothing}[p.alpha_s === nothing ? nothing : measure_alpha_s_power(p, q; points) for (p, q) in zip(ps, qs)]
end

"""
    alpha_s_power(artifact; parameters = nothing, kernels = :auto, points = 16) -> Vector{Union{Int,Nothing}}

As `alpha_s_power(base, doubled)` for a pyAmpliCol LC artifact: captured with `parameters` and
again with its alpha_s parameter doubled (both captures kept, as `capture` keeps them).
"""
function alpha_s_power(artifact::AbstractString; parameters = nothing, kernels::Symbol = :auto, points::Int = 16)
    is_artifact(artifact) || error("alpha_s_power: $artifact is not a pyAmpliCol artifact (give the folders at two alpha_s)")
    base = capture(artifact; parameters)
    main = read_main(joinpath(base, "main.json"))
    named = [prog.inputs.alpha_s for prog in main.programs if prog.inputs.alpha_s !== nothing]
    isempty(named) && return Union{Int,Nothing}[nothing for _ in main.programs]
    names = unique(String(a.name) for a in named)
    length(names) == 1 || error("alpha_s_power: the programs name several alpha_s parameters: $(join(names, ", "))")
    values = unique(Float64(a.value) for a in named)
    length(values) == 1 || error("alpha_s_power: the programs are recorded at several alpha_s: $values")
    # the parameters given, with alpha_s doubled
    given = parameters === nothing ? Dict{String,Any}() : Dict{String,Any}(JSON.parse(read(parameters, String)))
    given[names[1]] = 2 * values[1]
    file, io = mktemp()
    try
        # sorted: the same parameters, the same file, the same capture key
        write(io, "{" * join(("$(JSON.json(k)): $(JSON.json(given[k]))" for k in sort!(collect(keys(given)))), ", ") * "}")
        close(io)
        doubled = capture(artifact; parameters = file)
        return alpha_s_power(base, doubled; kernels, points)
    finally
        rm(file; force = true)
    end
end

# `p` with alpha_s power `power`
with_alpha_s_power(p::Program, power) =
    p.alpha_s === nothing ? p :
    Program((f == :alpha_s ? (p.alpha_s[1], power) : getfield(p, f) for f in fieldnames(Program))...)
