# The Metal runner's sources (c/metal/: df64.h, pacops.metal, uirvm.metal and each program's
# generated runner.metal) run on the CPU through c/metal/msl_shim.h, against the f64 Julia runner:
#
#   julia --project test/test_metal_shim.jl [EXPORT.json ...]   (default: test/data/export/*.json)
#
# Needs a C++ compiler (CXX, default c++). Not bit for bit: Metal's mixed precision (f32 currents,
# df64 for the f64 parts) is compared with tolerances; the shim flushes f32 denormals as Apple GPUs do.
# Checks: the df64 operations against f64; every wavefunction (all families, orientations, helicities;
# generic and on-axis momenta) against wavefunctions.jl; the f32 vertices and propagators against
# operations.jl; then every program of the exports at their reference points against
# evaluate(::Lowered) (or 16 RAMBO points where an export has none): the matrix element (sum of the helicity outputs) and each output relative to it.

using PacRunner
using PacRunner: Lowered, LArena, evaluate, output_count, compile_metal, wavefunction, vertex_values, propagator
using Libdl, Printf, Random, Statistics, Test

const ROOT = dirname(@__DIR__)
const CXX = get(ENV, "CXX", "c++")
const CXXFLAGS = `-std=c++17 -O1 -ffp-contract=off -fPIC -shared -Wno-attributes -Wno-unused-function -Wno-unused-variable`

function build(source::String, out::String, includes::Vector{String})
    run(`$CXX $CXXFLAGS $(["-I" * i for i in includes]) -o $out $source`)
    return out
end

# ---- the operations ---------------------------------------------------------------------------

const OPS = Ref{Ptr{Cvoid}}(C_NULL)
ops(sym) = dlsym(OPS[], sym)
dd(op, a, b) = ccall(ops(:shim_dd), Cdouble, (Cint, Cdouble, Cdouble), op, a, b)

function check_df64(rng)
    worst = Dict{String,Float64}()
    note(k, e) = (worst[k] = max(get(worst, k, 0.0), e))
    for _ in 1:20000
        x = exp(randn(rng) * 10) * rand(rng, (-1, 1)); y = exp(randn(rng) * 10) * rand(rng, (-1, 1))
        # the f64 result of the df64-rounded inputs is the reference (the inputs' own rounding is 2^-48)
        note("+", abs(dd(0, x, y) - (x + y)) / max(abs(x), abs(y)))
        note("-", abs(dd(1, x, y) - (x - y)) / max(abs(x), abs(y)))
        note("*", abs(dd(2, x, y) - x * y) / abs(x * y))
        note("/", abs(dd(3, x, y) - x / y) / abs(x / y))
        note("sqrt", abs(dd(4, abs(x), 0.0) - sqrt(abs(x))) / sqrt(abs(x)))
        dd(5, x, y) == (x < y ? 1.0 : 0.0) || note("<", 1.0)
        rx, ry = dd(0, x, 0.0), dd(0, y, 0.0)        # the inputs as df64 holds them
        note("max", dd(8, x, y) == max(rx, ry) ? 0.0 : 1.0)
    end
    for x in (0.0, -0.0, 1.0, 2.5, -2.5, 3.5, 1e7 + 0.5, -7.25)
        note("floor", dd(6, x, 0.0) == floor(x) ? 0.0 : 1.0)
        note("rint", dd(7, x, 0.0) == round(x, RoundNearest) ? 0.0 : 1.0)
    end
    for (k, e) in sort(collect(worst)); @printf("  df64 %-5s worst relative error %.2e\n", k, e); end
    return worst
end

# a random massless or massive momentum (E > 0, or E < 0: an incoming leg crossed), and on-axis ones
function momenta(rng, mass)
    ks = Vector{NTuple{4,Float64}}()
    for _ in 1:200
        p = randn(rng, 3) .* 100
        E = sqrt(sum(abs2, p) + mass^2)
        push!(ks, (E, p...), (-E, (-p)...))
    end
    for pz in (50.0, -50.0), s in (1.0, -1.0)
        E = sqrt(pz^2 + mass^2)
        push!(ks, (s * E, 0.0, 0.0, s * pz))
    end
    return ks
end

function check_wavefunctions(rng)
    worst = 0.0
    out = zeros(8)
    for (family, fnum, masses) in (("scalar", 0, (0.0,)), ("weyl", 1, (0.0,)), ("dirac", 2, (0.0, 4.7, 173.0)),
                                    ("vector", 3, (0.0, 91.1876)))
        for mass in masses, k in momenta(rng, mass), (orientation, onum) in (("particle", 0), ("antiparticle", 1))
            for h in (family == "vector" && mass > 0 ? (-1, 0, 1) : (-1, 1)), ch in (family == "weyl" ? (-1, 1) : (0,))
                ref = wavefunction(family, orientation, k, h, ch, mass)
                n = ccall(ops(:shim_wavefunction), Cint, (Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Cint, Cint, Cdouble),
                          out, fnum, onum, collect(k), h, ch, mass)
                got = [complex(out[2j-1], out[2j]) for j in 1:n]
                scale = max(1.0, maximum(abs, ref))
                worst = max(worst, maximum(abs, got .- ref) / scale)
            end
        end
    end
    @printf("  wavefunctions: worst error %.2e (relative to the largest component)\n", worst)
    return worst
end

f32(x) = Float64(Float32(x))
randc(rng, n) = [complex(f32(randn(rng)), f32(randn(rng))) for _ in 1:n]
flat(z) = Float64[v for c in z for v in (real(c), imag(c))]

function check_currents(rng)
    worst = 0.0
    out = zeros(12)
    shapes = (("weyl_vector_a", 0, 2, 4), ("weyl_vector_b", 1, 2, 4), ("tensor_vector", 2, 6, 4),
              ("vector_wedge", 3, 4, 4), ("three_vector", 4, 4, 4))
    for _ in 1:2000, (type, tnum, nl, nr) in shapes
        l = randc(rng, nl); r = randc(rng, nr); u = f32.(randn(rng, 4)); v = f32.(randn(rng, 4))
        ref = vertex_values(type, l, r, u, v)
        lf = vcat(flat(l), zeros(12 - 2nl)); rf = vcat(flat(r), zeros(12 - 2nr))
        n = ccall(ops(:shim_vertex), Cint, (Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                  out, tnum, lf, rf, u, v)
        got = [complex(out[2j-1], out[2j]) for j in 1:n]
        worst = max(worst, maximum(abs, got .- ref) / max(1.0, maximum(abs, ref)))
    end
    for _ in 1:2000, (type, tnum, n) in (("weyl_a", 0, 2), ("weyl_b", 1, 2), ("vector", 2, 4))
        c = randc(rng, n); k = Tuple(randn(rng, 4) .* 50); factor = randc(rng, 1)[1]
        ref = propagator(type, c, k, factor)
        ccall(ops(:shim_propagator), Cint, (Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
              out, tnum, vcat(flat(c), zeros(8 - 2n)), collect(k), [real(factor), imag(factor)])
        got = [complex(out[2j-1], out[2j]) for j in 1:n]
        worst = max(worst, maximum(abs, got .- ref) / max(1e-30, maximum(abs, ref)))
    end
    @printf("  f32 vertices and propagators: worst error %.2e (relative to the largest component)\n", worst)
    return worst
end

# ---- whole programs ---------------------------------------------------------------------------

# every program of `path` at its reference points: (programs, the worst ME error, the worst output error)
function check_programs(path::String; quiet::Bool = false)
    ls = load_recurrence_export(path)
    refs = recurrence_references(path)
    me_errors = Float64[]
    out_worst = 0.0
    for (k, l) in enumerate(ls)
        # the export's reference points, else 16 RAMBO points (massless; both runners take the same momenta)
        points = refs[k] === nothing ? PacRunner.rambo_momenta(l.legs, 16; seed = UInt64(k)) : refs[k][1]
        dir = compile_metal(mktempdir(), l)
        lib = build(joinpath(ROOT, "test", "metal_shim_run.cpp"), joinpath(dir, "libshim_run.$(dlext)"),
                    [joinpath(ROOT, "c", "metal"), dir])
        h = dlopen(lib)
        try
            program = read(joinpath(dir, "program.bin"))
            n = output_count(l)
            momenta = Float64[c for point in points for leg in point for c in leg]
            out = zeros(n * length(points))
            status = ccall(dlsym(h, :shim_run), Cint, (Ptr{UInt8}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}),
                           program, momenta, length(points), out)
            status == 0 || error("$(l.process) $(l.colour): status $status")
            for (p, point) in enumerate(points)
                ref = evaluate(l, LArena(l), point)
                got = out[(p-1)*n+1:p*n]
                me = sum(ref)
                push!(me_errors, abs(sum(got) - me) / abs(me))
                out_worst = max(out_worst, maximum(abs, got .- ref) / abs(me))
            end
        finally
            dlclose(h)
        end
    end
    quiet || @printf("  %s: %d programs, %d points: ME error median %.1e, max %.1e; outputs (relative to the ME) max %.1e\n",
                     basename(path), length(ls), length(me_errors), median(me_errors), maximum(me_errors), out_worst)
    return (me_errors, out_worst)
end

# the operations' library built and loaded (once), denormals flushed
function setup()
    OPS[] == C_NULL || return
    lib = build(joinpath(ROOT, "test", "metal_shim_ops.cpp"), joinpath(mktempdir(), "libshim_ops.$(dlext)"),
                [joinpath(ROOT, "c", "metal")])
    OPS[] = dlopen(lib)
    ccall(ops(:shim_flush_denormals), Cvoid, ())
end

have_cxx() = Sys.which(CXX) !== nothing

function main(paths)
    setup()
    rng = Xoshiro(20261115)
    @testset "Metal sources on the CPU (msl_shim.h)" begin
        w = check_df64(rng)
        @test all(w[k] < 2.0^-44 for k in ("+", "-", "*", "/", "sqrt"))
        @test all(w[k] == 0 for k in ("max", "floor", "rint")) && !haskey(w, "<")
        @test check_wavefunctions(rng) < 1e-12
        @test check_currents(rng) < 1e-5
        for p in paths
            (me, out) = check_programs(p)
            @test maximum(me) < 1e-4 && median(me) < 1e-6
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? filter(f -> endswith(f, ".json"), readdir(joinpath(ROOT, "test", "data", "export"); join = true)) : ARGS)
end
