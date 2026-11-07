# Timing of the compiled C (compile.jl) against the Julia runner (pacrun.jl's evaluate), per
# phase-space point, on pac2json folders (default: every case in test/data). Each folder is
# prepared as in test_compile.jl (the kernels' libraries built by sjb2obj in a temporary copy);
# each program is compiled once and built with test/bench_driver.c at each of FLAGS (all with
# -ffp-contract=off, so the outputs are those of test_compile.jl bit for bit), then run on its
# examples.json points, best of 5 trials. The C runs on one point per call, single-threaded; for
# steady numbers pin it to one core of a quiet machine, e.g.  taskset -c 4 julia bench_compile.jl
# usage: julia --project test/bench_compile.jl SJB2OBJ [PAC2JSON_DIR...]     (CC: the C compiler, default cc)
include(joinpath(@__DIR__, "compile_common.jl"))

const DRIVER = joinpath(@__DIR__, "bench_driver.c")
const FLAGS = ["-O2" => `-O2 -ffp-contract=off`,
               "-O3 native" => `-O3 -march=native -ffp-contract=off`,
               "-O3 native lto" => `-O3 -march=native -ffp-contract=off -flto=auto`]
const SECONDS = 2.0  # per measurement, over the 5 trials

# the best time per point (ns) of the Julia runner on `ms`, over 5 trials
function julia_time(p::Program, a::Arena, ms)::Float64
    foreach(m -> evaluate(p, a, m), ms)  # the first calls compile
    best = Inf
    for _ in 1:5
        (n, t0) = (0, time_ns())
        while time_ns() - t0 < SECONDS / 5 * 1e9
            foreach(m -> evaluate(p, a, m), ms)
            n += length(ms)
        end
        best = min(best, (time_ns() - t0) / n)
    end
    return best
end

us(ns) = lpad(string(round(ns / 1000, digits = 2)), 9) * " µs/point"

function bench_folder(sjb2obj::String, source::String)
    dir = prepare(sjb2obj, source; quiet = true)
    programs = load_programs(joinpath(dir, "main.json"))
    ex = read_main(joinpath(dir, "examples.json"))
    for (n, (p, e)) in enumerate(zip(programs, ex.programs))
        out = joinpath(dir, "compiled-$n")
        compile(out, p)  # the first call compiles compile.jl
        tc = @elapsed compile(out, p)
        b = ArenaIndex(p, Arena(p))
        println("$(basename(dir)) program $n ($(p.process), $(p.colour)): $(length(p.body)) instructions, ",
                "$(length(p.reduction)) outputs, arena $(b.size) doubles; runner.c ",
                "$(round(Int, filesize(joinpath(out, "runner.c")) / 1024)) KB in $(round(tc, digits = 3)) s")
        input = points_text(e.points)
        for (name, flags) in FLAGS
            exe = joinpath(out, "bench-" * replace(name, " " => "-"))
            tb = @elapsed build(out, DRIVER, exe, flags)
            # one quick pass to size the repetitions to about SECONDS over the trials
            t1 = parse(Float64, split(read(pipeline(`$exe $(p.legs) $(length(p.reduction)) $(b.size) 1 1`; stdin = IOBuffer(input)), String))[1])
            reps = max(1, round(Int, SECONDS / 5 * 1e9 / (t1 * length(e.points))))
            t = parse(Float64, split(read(pipeline(`$exe $(p.legs) $(length(p.reduction)) $(b.size) $reps 5`; stdin = IOBuffer(input)), String))[1])
            println("  C $(rpad(name, 15)) $(us(t))   (cc $(round(tb, digits = 2)) s)")
        end
        ms = [[Tuple(Float64.(v)) for v in pt.momenta] for pt in e.points]
        println("  Julia runner      $(us(julia_time(p, Arena(p), ms)))")
    end
end

function main(args)
    sjb2obj = args[1]
    foreach(d -> bench_folder(sjb2obj, d), length(args) > 1 ? args[2:end] : cases())
    return 0
end

# as a script (runtests.jl includes this file for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || (println("usage: julia --project test/bench_compile.jl SJB2OBJ [PAC2JSON_DIR...]"); exit(2))
    exit(main(ARGS))
end
