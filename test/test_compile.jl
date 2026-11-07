# compile.jl end to end on pac2json folders (default: every case in test/data, u u~ > Z g g and
# u u~ > Z g g g g):
#   1. the folder is copied to a temporary directory, and each kernel's shared library (main.json
#      kernels[i].library) is built there from its .plane.symjit: symjit-utils' sjb2obj, then
#      cc -shared, as pac2json --sjb2obj builds them;
#   2. each program is compiled to C (compile.jl, with its examples), built with
#      test/compile_driver.c and the kernels' libraries (cc -ffp-contract=off -Werror), and run on
#      its examples.json points; the folder's own `make test` (testing.c) must pass too;
#   3. each output is compared bit for bit with the Julia runner (pacrun.jl's evaluate) and with
#      the CPU's outputs in examples.json (RustiCol) within TOLERANCE of the point's total, as
#      test_pacrun.jl does; the bit-identical outputs are counted (RustiCol's symjit 2.27.3 AVX
#      kernels may round differently from the scalar kernel libraries, see test_operations.jl).
# usage: julia --project test/test_compile.jl SJB2OBJ [PAC2JSON_DIR...]     (CC: the C compiler, default cc)
include(joinpath(@__DIR__, "compile_common.jl"))

const TOLERANCE = 1e-14
const DRIVER = joinpath(@__DIR__, "compile_driver.c")

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

# the outputs of the compiled program in `dir` (runner.c) on `points`, built with the driver
function run_compiled(dir::String, p::Program, points)::Vector{Vector{Float64}}
    exe = build(dir, DRIVER, joinpath(dir, "driver"), `-O2 -ffp-contract=off -Wall -Werror`)
    b = ArenaIndex(p, Arena(p))
    text = read(pipeline(`$exe $(p.legs) $(length(p.reduction)) $(b.size)`; stdin = IOBuffer(points_text(points))), String)
    rows = split(strip(text), "\n")
    length(rows) == length(points) || error("the driver printed $(length(rows)) lines for $(length(points)) points")
    return [[reinterpret(Float64, parse(UInt64, h; base = 16)) for h in split(r)] for r in rows]
end

function test_folder(sjb2obj::String, source::String)::Int
    println("$source:")
    dir = prepare(sjb2obj, source)
    programs = load_programs(joinpath(dir, "main.json"))
    ex = read_main(joinpath(dir, "examples.json"))
    failed = 0
    for (n, (p, e)) in enumerate(zip(programs, ex.programs))
        (p.process, p.colour) == (e.process, e.colour) || error("examples.json is not in main.json's order")
        out = joinpath(dir, "compiled-$n")
        compile(out, p; examples = e)
        got = run_compiled(out, p, e.points)
        # the folder's own test: make test (testing.c, against examples.json)
        made = success(pipeline(`make -s -C $out test`; stdout = devnull, stderr = devnull))
        a = Arena(p)
        (outputs, vs_julia, vs_cpu, identical, worst) = (0, 0, 0, 0, 0.0)
        for (pt, c) in zip(e.points, got)
            julia = evaluate(p, a, [Tuple(Float64.(v)) for v in pt.momenta])
            cpu = Float64.(pt.outputs)
            length(c) == length(julia) == length(cpu) || error("program $n: output counts differ")
            total = abs(sum(cpu))
            for (x, y, z) in zip(c, julia, cpu)
                outputs += 1
                same(x, y) || (vs_julia += 1)
                same(x, z) && (identical += 1)
                d = abs(x - z) / max(total, floatmin(Float64))
                worst = max(worst, d)
                d <= TOLERANCE || (vs_cpu += 1)
            end
        end
        ok = vs_julia == 0 && vs_cpu == 0 && made
        ok || (failed += 1)
        println("  program $n ($(p.process), $(p.colour)): $(length(e.points)) points, $outputs outputs; ",
                "C vs Julia: $vs_julia differ; C vs CPU: $vs_cpu beyond $TOLERANCE of the total ",
                "(worst $worst), $identical bit-identical; make test: $(made ? "passes" : "FAILS")", ok ? "" : "  FAILED")
    end
    return failed
end

function main(args)
    sjb2obj = args[1]
    dirs = length(args) > 1 ? args[2:end] : cases()
    failed = sum(d -> test_folder(sjb2obj, d), dirs)
    println(failed == 0 ? "all programs pass" : "$failed programs FAILED")
    return failed == 0 ? 0 : 1
end

# as a script (runtests.jl includes this file for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || (println("usage: julia --project test/test_compile.jl SJB2OBJ [PAC2JSON_DIR...]"); exit(2))
    exit(main(ARGS))
end
