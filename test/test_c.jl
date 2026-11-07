# The C functions of c/pacops.c against the Julia ones, bit for bit, on random inputs (with the
# special momenta: on the axes, at rest, E < 0) and random helicities, chiralities, masses, modes.
# usage: julia test_c.jl [N]   (c/libpacops.so built first: see c/pacops.h)
include(joinpath(@__DIR__, "..", "wavefunctions.jl"))
include(joinpath(@__DIR__, "..", "operations.jl"))
using Random

const LIB = joinpath(@__DIR__, "..", "c", "libpacops.so")

# a value as the C layout: re, im of each component
flat(v::Tuple) = Float64[x for z in v for x in (real(z), imag(z))]
flat(z::Complex{Float64}) = [real(z), imag(z)]
unflat(x::Vector{Float64}, n::Int) = ntuple(j -> complex(x[2j-1], x[2j]), n)
same(a, b) = length(a) == length(b) && all(reinterpret(UInt64, x) == reinterpret(UInt64, y) for (x, y) in zip(a, b))

rc() = complex(randn(), randn())
rv(n) = ntuple(_ -> rc(), n)
function rk()
    r = rand()
    E = 10 * randn()
    r < 0.1 && return (E, 0.0, 0.0, abs(E) * (rand(Bool) ? 1 : -1))  # along ±z
    r < 0.15 && return (E, 0.0, 0.0, 0.0)                              # at rest
    r < 0.2 && return (E, 0.0, 0.0, -E)                                # E + pz = 0
    return (E, randn(), randn(), randn())
end

const counts = Dict{String,Tuple{Int,Int}}()  # name → (cases, differences)
check(name, a, b) = (c = get(counts, name, (0, 0)); counts[name] = (c[1] + 1, c[2] + !same(a, b)))

const FAMILY = Dict("scalar" => 0, "weyl" => 1, "dirac" => 2, "vector" => 3)
const VTYPE = Dict("weyl_vector_a" => 0, "weyl_vector_b" => 1, "tensor_vector" => 2, "vector_wedge" => 3, "three_vector" => 4)
const VSHAPE = Dict("weyl_vector_a" => (2, 4, 2), "weyl_vector_b" => (2, 4, 2), "tensor_vector" => (6, 4, 4),
                    "vector_wedge" => (4, 4, 6), "three_vector" => (4, 4, 4))
const PTYPE = Dict("weyl_a" => (0, 2), "weyl_b" => (1, 2), "vector" => (2, 4))

function main(N)
    Random.seed!(1)
    lib = Libdl.dlopen(LIB)
    sym(s) = Libdl.dlsym(lib, s)
    for _ in 1:N
        k = rk()
        kv = collect(k)
        h = rand((-1, 0, 1))
        ch = rand((-1, 0, 1))
        mass = rand((0.0, 1e-9, 0.5 + rand()))

        # wavefunctions
        for (name, f, n, args) in [("weyl_particle", weyl_particle, 2, (h, ch)), ("weyl_antiparticle", weyl_antiparticle, 2, (h, ch))]
            out = zeros(2n)
            ccall(sym(name), Cvoid, (Ptr{Float64}, Ptr{Float64}, Cint, Cint), out, kv, args[1], args[2])
            check(name, out, flat(f(k, args...)))
        end
        for (name, f) in [("quark_dirac", quark_dirac), ("antiquark_dirac", antiquark_dirac), ("gluon", gluon)]
            hh = rand((-1, 1))
            out = zeros(8)
            ccall(sym(name), Cvoid, (Ptr{Float64}, Ptr{Float64}, Cint), out, kv, hh)
            check(name, out, flat(f(k, hh)))
        end
        for (name, f) in [("quark_dirac_massive", quark_dirac_massive), ("antiquark_dirac_massive", antiquark_dirac_massive),
                          ("massive_vector", massive_vector)]
            hh = name == "massive_vector" ? h : rand((-1, 1))
            out = zeros(8)
            ccall(sym(name), Cvoid, (Ptr{Float64}, Ptr{Float64}, Cint, Cdouble), out, kv, hh, mass)
            check(name, out, flat(f(k, hh, mass)))
        end
        for (family, fam) in FAMILY, orientation in ("particle", "antiparticle")
            factor = rc()
            out = zeros(8)
            n = ccall(sym("source"), Cint, (Ptr{Float64}, Cint, Cint, Ptr{Float64}, Cint, Cint, Cdouble, Ptr{Float64}),
                      out, fam, orientation == "antiparticle", kv, h, ch, mass, flat(factor))
            w = map(c -> factor * c, wavefunction(family, orientation, k, h, ch, mass))
            check("source", out[1:2n], flat(w))
        end

        # vertices (in place: out is the old value)
        for (type, t) in VTYPE, mode in ("=", "+=")
            (nl, nr, no) = VSHAPE[type]
            (l, r, old) = (rv(nl), rv(nr), rv(no))
            (u, v) = (rk(), rk())
            (coupling, factor) = (rc(), rc())
            out = flat(old)
            ccall(sym("vertex"), Cvoid, (Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                  out, t, mode == "+=", flat(l), flat(r), collect(u), collect(v), flat(coupling), flat(factor))
            check("vertex $type", out, flat(vertex(type, mode, old, l, r, u, v, coupling, factor)))
        end

        # propagators (in place)
        for (type, (t, n)) in PTYPE
            c = rv(n)
            factor = rc()
            out = flat(c)
            ccall(sym("propagator"), Cvoid, (Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), out, t, out, kv, flat(factor))
            check("propagator $type", out, flat(propagator(type, c, k, factor)))
        end

        # scale, clear, closure
        n = rand((1, 2, 4, 6))
        c = rv(n)
        m = rand(1:n)
        factor = rc()
        out = flat(c)
        ccall(sym("scale"), Cvoid, (Ptr{Float64}, Cint, Ptr{Float64}), out, m, flat(factor))
        check("scale", out, flat(scale(c, m, factor)))
        out = flat(c)
        ccall(sym("clear"), Cvoid, (Ptr{Float64}, Cint), out, m)
        check("clear", out, flat(clear(c, m)))
        (l, r, cf, a) = (rv(n), rv(n), [rc() for _ in 1:m], rc())
        out = flat(a)
        ccall(sym("closure"), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}),
              out, flat(l), flat(r), flat(Tuple(cf)), m, flat(factor))
        check("closure", out, flat(closure(a, l, r, cf, factor)))

        # kernel_store: random distinct outputs, direct with factor 1 + 0i at times
        pairs = rand(1:n)
        outputs = randperm(n)[1:pairs]
        values = randn(2pairs)
        direct = rand(Bool)
        factor = rand() < 0.3 ? complex(1.0, 0.0) : rc()
        mode = rand(("=", "+="))
        out = flat(c)
        ccall(sym("kernel_store"), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Cint, Cint, Cint, Ptr{Float64}),
              out, values, Cint.(outputs .- 1), pairs, direct, mode == "+=", flat(factor))
        check("kernel_store", out, flat(kernel_store(c, values, outputs, direct, mode, factor)))

        # reduction
        amps = [rc() for _ in 1:8]
        terms = rand(1:8, rand(1:6))
        weights = randn(length(terms))
        (num, den) = rand(((1, 1), (1, 3), (4, 9)))
        x = ccall(sym("reduction"), Cdouble, (Ptr{Float64}, Ptr{Cint}, Ptr{Float64}, Cint, Clong, Clong),
                  flat(Tuple(amps)), Cint.(terms .- 1), weights, length(terms), num, den)
        check("reduction", [x], [reduction(amps[terms], weights, (num, den))])
    end

    failed = 0
    for name in sort(collect(keys(counts)))
        (cases, diffs) = counts[name]
        failed += diffs
        println(rpad(name, 28), lpad(cases, 7), " cases, ", diffs, " differ")
    end
    println(failed == 0 ? "all bit for bit" : "$failed differ")
    return failed == 0 ? 0 : 1
end

exit(main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10000))
