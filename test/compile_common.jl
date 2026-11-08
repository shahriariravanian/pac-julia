# The parts of test_compile.jl and bench_compile.jl: the test cases, a pac2json folder copied to a
# temporary directory with its kernels' shared libraries built there, the points of examples.json
# as the drivers read them, and building a compiled program with a driver.
using PacRunner
using PacRunner: built

const CC = get(ENV, "CC", "cc")

# the test cases: every folder of test/data (u u~ > Z g g, u u~ > Z g g g g)
cases() = [d for d in readdir(joinpath(@__DIR__, "data"); join = true) if isfile(joinpath(d, "main.json"))]

# the value of `#define <NAME>_<what>` in a kernel's header
function define(header::String, name::String, what::String)::Int
    key = "#define $(uppercase(name))_$what "
    for l in eachline(header)
        startswith(l, key) && return parse(Int, strip(l[length(key)+1:end]))
    end
    error("$header: no $key")
end

# each kernel's library, built in `dir` (a copy of the pac2json folder) from its .plane.symjit:
# sjb2obj, then cc -shared, as pac2json --sjb2obj builds them
function build_kernels(sjb2obj::String, dir::String; quiet = false)
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
        quiet || println("  kernel $(k.name): $(lib.states) inputs, $(lib.outputs ÷ 2) output pairs, built by sjb2obj")
    end
end

# a copy of the pac2json folder `source` in a temporary directory, with its kernels' libraries
function prepare(sjb2obj::String, source::String; quiet = false)::String
    dir = joinpath(mktempdir(), basename(normpath(source)))
    cp(source, dir)
    build_kernels(sjb2obj, dir; quiet)
    return dir
end

# the arena size of the compiled program in `dir`, from its runner.h (RUNNER_ARENA_SIZE)
function arena_size(dir::String)::Int
    for l in eachline(joinpath(dir, "runner.h"))
        startswith(l, "#define RUNNER_ARENA_SIZE ") && return parse(Int, split(l)[3])
    end
    error("$dir/runner.h has no RUNNER_ARENA_SIZE")
end

# the points of examples.json as the drivers read them: one line of 4 × legs momenta per point
points_text(points) = join([join([repr(Float64(x)) for v in pt.momenta for x in v], " ") for pt in points], "\n") * "\n"

# the executable of the compiled program in `dir` (compile.jl's runner.c) with `driver`
function build(dir::String, driver::String, exe::String, flags::Cmd)::String
    libs = filter(f -> endswith(f, ".so"), readdir(dir; join = true))
    run(`$CC $flags -std=c99 -o $exe $(joinpath(dir, "runner.c")) $(joinpath(dir, "pacops.c")) $driver $libs -Wl,-rpath,$dir -lm`)
    return exe
end
