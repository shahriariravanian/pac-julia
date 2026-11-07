# compile.jl end to end on a pac2json folder (default: test/data/uux_zgg, u u~ > Z g g):
#   1. the folder is copied to a temporary directory, and each kernel's shared library (main.json
#      kernels[i].library) is built there from its .plane.symjit: symjit-utils' sjb2obj, then
#      cc -shared, as pac2json --sjb2obj builds them;
#   2. each program is compiled to C (compile.jl), built with test/compile_driver.c and the
#      kernels' libraries (cc -ffp-contract=off -Werror), and run on its examples.json points;
#   3. each output is compared bit for bit with the Julia runner (pacrun.jl's evaluate) and with
#      the CPU's outputs in examples.json (RustiCol) within TOLERANCE of the point's total, as
#      test_pacrun.jl does; the bit-identical outputs are counted.
# usage: julia test_compile.jl SJB2OBJ [PAC2JSON_DIR...]     (CC: the C compiler, default cc)
include(joinpath(@__DIR__, "..", "PacRunner.jl"))

const TOLERANCE = 1e-14
const CC = get(ENV, "CC", "cc")
const DRIVER = joinpath(@__DIR__, "compile_driver.c")

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

# the value of `#define <NAME>_<what>` in a kernel's header
function define(header::String, name::String, what::String)::Int
    key = "#define $(uppercase(name))_$what "
    for l in eachline(header)
        startswith(l, key) && return parse(Int, strip(l[length(key)+1:end]))
    end
    error("$header: no $key")
end

# each kernel's library, built in `dir` (a copy of the pac2json folder) from its .plane.symjit
function build_kernels(sjb2obj::String, dir::String)
    main = read_main(joinpath(dir, "main.json"))
    for k in main.kernels
        lib = k.library
        lib === nothing && error("kernel $(k.name) has no library entry in main.json (pac2json --sjb2obj)")
        (plane, so, header) = (joinpath(dir, k.path), joinpath(dir, lib.path), joinpath(dir, lib.header))
        object = splitext(so)[1] * ".o"
        run(`$sjb2obj $plane -o $object --name $(lib.symbol) --header $header`)
        run(`$CC -shared -o $so $object`)
        rm(object)
        (define(header, lib.symbol, "COUNT_STATES"), define(header, lib.symbol, "COUNT_OBS")) == (lib.states, lib.outputs) ||
            error("kernel $(k.name): the header's counts differ from main.json's")
        println("  kernel $(k.name): $(lib.states) inputs, $(lib.outputs ÷ 2) output pairs, built by sjb2obj")
    end
end

# the outputs of the compiled program in `dir` (runner.c) on `points`, built with the driver
function run_compiled(dir::String, p::Program, points)::Vector{Vector{Float64}}
    libs = filter(f -> endswith(f, ".so"), readdir(dir; join = true))
    exe = joinpath(dir, "driver")
    run(`$CC -std=c99 -O2 -ffp-contract=off -Wall -Werror -o $exe $(joinpath(dir, "runner.c")) $(joinpath(dir, "pacops.c")) $DRIVER $libs -Wl,-rpath,$dir -lm`)
    input = join([join([repr(Float64(x)) for v in pt.momenta for x in v], " ") for pt in points], "\n") * "\n"
    b = ArenaIndex(p, Arena(p))
    text = read(pipeline(`$exe $(p.legs) $(length(p.reduction)) $(b.size)`; stdin = IOBuffer(input)), String)
    rows = split(strip(text), "\n")
    length(rows) == length(points) || error("the driver printed $(length(rows)) lines for $(length(points)) points")
    return [[reinterpret(Float64, parse(UInt64, h; base = 16)) for h in split(r)] for r in rows]
end

function test_folder(sjb2obj::String, source::String)::Int
    println("$source:")
    dir = joinpath(mktempdir(), basename(normpath(source)))
    cp(source, dir)
    build_kernels(sjb2obj, dir)
    programs = load_programs(joinpath(dir, "main.json"))
    ex = read_main(joinpath(dir, "examples.json"))
    failed = 0
    for (n, (p, e)) in enumerate(zip(programs, ex.programs))
        (p.process, p.colour) == (e.process, e.colour) || error("examples.json is not in main.json's order")
        out = joinpath(dir, "compiled-$n")
        compile(out, p)
        got = run_compiled(out, p, e.points)
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
        ok = vs_julia == 0 && vs_cpu == 0
        ok || (failed += 1)
        println("  program $n ($(p.process), $(p.colour)): $(length(e.points)) points, $outputs outputs; ",
                "C vs Julia: $vs_julia differ; C vs CPU: $vs_cpu beyond $TOLERANCE of the total ",
                "(worst $worst), $identical bit-identical", ok ? "" : "  FAILED")
    end
    return failed
end

function main(args)
    sjb2obj = args[1]
    dirs = length(args) > 1 ? args[2:end] : [joinpath(@__DIR__, "data", "uux_zgg")]
    failed = sum(d -> test_folder(sjb2obj, d), dirs)
    println(failed == 0 ? "all programs pass" : "$failed programs FAILED")
    return failed == 0 ? 0 : 1
end

length(ARGS) >= 1 || (println("usage: julia test_compile.jl SJB2OBJ [PAC2JSON_DIR...]"); exit(2))
exit(main(ARGS))
