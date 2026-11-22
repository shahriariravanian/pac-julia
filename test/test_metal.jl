# The Metal runner on an Apple GPU (macOS only; elsewhere it prints why and exits 2):
#
#   julia --project test/test_metal.jl [--time N] [EXPORT.json ...]   (default: test/data/export/*.json)
#
# For every program of the exports (its reference points, else 16 RAMBO points):
#   - compile_metal's folder built (make: librunner_metal.dylib) and run on the GPU (run_metal);
#   - the GPU against the f64 Julia runner: the matrix element (sum of the helicity outputs) and each
#     output relative to it (mixed f32/df64: about 1e-7, worst ~1e-5);
#   - the GPU against the same source on the CPU (test/metal_shim_run.cpp through msl_shim.h): the
#     same correctly rounded f32 operations and flush-to-zero, so they should agree bit for bit
#     (reported: the points that differ, and the largest relative difference);
# then the UMAMI library (umami_libraries' device = :metal, test_umami.jl's check). With --time N:
# each program warm on N RAMBO points, µs per point (the second of two runs).

using PacRunner
using PacRunner: Lowered, LArena, evaluate, output_count, compile_metal
using Libdl, Printf, Statistics, Test

const ROOT = dirname(@__DIR__)
module MetalShim include(joinpath(@__DIR__, "test_metal_shim.jl")) end
module Umami include(joinpath(@__DIR__, "test_umami.jl")) end

have_metal() = Sys.isapple() && success(pipeline(`system_profiler SPDisplaysDataType`, `grep -q Metal`))

# run_metal of a built folder on `points` (each a vector of legs): the outputs, point-major
function run_gpu(h, l::Lowered, points)
    n = output_count(l)
    momenta = Float64[c for point in points for leg in point for c in leg]
    out = zeros(n * length(points))
    status = ccall(dlsym(h, :run_metal), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t), out, momenta, length(points))
    status == 0 || error("$(l.process) $(l.colour): run_metal status $status")
    return out
end

function run_shim(dir::String, l::Lowered, points)
    lib = MetalShim.build(joinpath(ROOT, "test", "metal_shim_run.cpp"), joinpath(dir, "libshim_run.$(dlext)"),
                          [joinpath(ROOT, "c", "metal"), dir])
    h = dlopen(lib)
    try
        n = output_count(l)
        momenta = Float64[c for point in points for leg in point for c in leg]
        out = zeros(n * length(points))
        status = ccall(dlsym(h, :shim_run), Cint, (Ptr{UInt8}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}),
                       read(joinpath(dir, "program.bin")), momenta, length(points), out)
        status == 0 || error("$(l.process) $(l.colour): shim status $status")
        return out
    finally
        dlclose(h)
    end
end

function check_gpu(path::String; time_points::Int = 0)
    ls = load_recurrence_export(path)
    refs = recurrence_references(path)
    me_errors = Float64[]; out_worst = 0.0; shim_differ = 0; shim_rel = 0.0; npoints = 0
    for (k, l) in enumerate(ls)
        points = refs[k] === nothing ? PacRunner.rambo_momenta(l.legs, 16; seed = UInt64(k)) : refs[k][1]
        dir = compile_metal(mktempdir(), l)
        run(`make -s -C $dir`)
        h = dlopen(joinpath(dir, "librunner_metal.dylib"))
        try
            n = output_count(l)
            gpu = run_gpu(h, l, points)
            shim = run_shim(dir, l, points)
            for (p, point) in enumerate(points)
                ref = evaluate(l, LArena(l), point)
                got = gpu[(p-1)*n+1:p*n]
                me = sum(ref)
                push!(me_errors, abs(sum(got) - me) / abs(me))
                out_worst = max(out_worst, maximum(abs, got .- ref) / abs(me))
                s = shim[(p-1)*n+1:p*n]
                any(reinterpret(UInt64, got) .!= reinterpret(UInt64, s)) && (shim_differ += 1)
                shim_rel = max(shim_rel, maximum(abs, got .- s) / abs(me))
                npoints += 1
            end
            if time_points > 0
                many = PacRunner.rambo_momenta(l.legs, time_points; seed = UInt64(1000 + k))
                run_gpu(h, l, many)                         # compile, warm up
                t = @elapsed run_gpu(h, l, many)
                @printf("    %s %s: %.3f µs/point (%d points)\n", l.process, l.colour, 1e6 * t / time_points, time_points)
            end
        finally
            dlclose(h)
        end
    end
    @printf("  %s: %d programs, %d points: ME error median %.1e, max %.1e; outputs max %.1e; GPU vs CPU shim: %d points differ, max %.1e\n",
            basename(path), length(ls), npoints, median(me_errors), maximum(me_errors), out_worst, shim_differ, shim_rel)
    return (me_errors, shim_differ)
end

function main(args)
    have_metal() || (println("no Metal (macOS with a Metal GPU needed)"); exit(2))
    time_points = 0
    if !isempty(args) && args[1] == "--time"
        time_points = parse(Int, args[2]); args = args[3:end]
    end
    paths = isempty(args) ? filter(f -> endswith(f, ".json"), readdir(joinpath(ROOT, "test", "data", "export"); join = true)) : args
    @testset "Metal runner on the GPU" begin
        for p in paths
            (me, differ) = check_gpu(p; time_points)
            @test maximum(me) < 1e-4 && median(me) < 1e-6
        end
        for p in filter(f -> occursin("test/data/export", f), paths)
            @test isempty(Umami.check(p; metal = true))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
