# Short tests of PacRunner on the cases of test/data (`] test`, or julia --project test/runtests.jl).
#
# Without external tools: the loader, the .sjb reader (a synthetic code buffer of every
# instruction, the kernels' headers), the Complexifier (the generic and the native lowering of each
# kernel give the same values), the Julia runner against RustiCol's outputs (programs without
# kernel rows), and compile's refusals.
# With a C compiler (CC, default cc): the compiled C against the Julia runner, the variants, and
# (with make) the compiled folders' own `make test`.
# With symjit-utils' sjb2obj (ENV["SJB2OBJ"], or sjb2obj on the PATH): the kernels' libraries are
# built and every program of every case runs, in Julia and in C.
# The longer comparisons with symjit and RustiCol themselves (sjbdump, wavefunctions_ref.rs, the
# reference interpreter) are the test_*.jl scripts.
using Test
using Random
using PacRunner
using PacRunner: built, Kernel, ComplexBi, Uni, Bi, Fused, LoadMath, LoadConstMath, Call, LoadArgs, SaveArgs, mem_size

# the scripts' functions, each in its own module (their names would clash)
module SjbScript include(joinpath(@__DIR__, "test_sjb.jl")) end
module ComplexifyScript include(joinpath(@__DIR__, "test_complexify.jl")) end
module CompileScript include(joinpath(@__DIR__, "test_compile.jl")) end
module VariantsScript include(joinpath(@__DIR__, "test_compile_variants.jl")) end
module LowerScript include(joinpath(@__DIR__, "test_lower.jl")) end

const CASES = CompileScript.cases()
const CC = CompileScript.CC
const HAVE_CC = Sys.which(CC) !== nothing
const HAVE_MAKE = Sys.which("make") !== nothing
const SJB2OBJ = let s = get(ENV, "SJB2OBJ", Sys.which("sjb2obj"))
    s === nothing || !isfile(s) ? nothing : s
end

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y) || (isnan(x) && isnan(y))
has_kernel_rows(p::Program) = any(op -> op isa Kernel, p.body)
kernel_files(case) = [joinpath(case, k.path) for k in read_main(joinpath(case, "main.json")).kernels]

@testset "PacRunner" begin
    @testset "load" begin
        counts = Dict(basename(c) => length(load_programs(joinpath(c, "main.json"))) for c in CASES)
        @test counts == Dict("uux_zgg" => 1, "uux_zgggg" => 1, "ppjjj_crossed" => 21)
        ps = load_programs(joinpath(@__DIR__, "data", "ppjjj_crossed", "main.json"))
        @test count(p -> p.crossing !== nothing, ps) == 12
        @test count(!has_kernel_rows, ps) == 3
        # test/data holds no libraries: they load as not built, and a call is an error
        p = load_artifact(joinpath(@__DIR__, "data", "uux_zgg", "main.json"))
        @test !built(p.bindings[1].kernel)
        @test_throws ErrorException evaluate(p, Arena(p), [(500.0, 0.0, 0.0, 500.0) for _ in 1:p.legs])
    end

    @testset "sjb reader" begin
        # every instruction and the edge cases of the encoding, encoded and decoded
        code = SjbScript.synthetic()
        bytes = UInt8[]
        foreach(i -> SjbScript.encode(bytes, i), code)
        @test SjbScript.lines(PacRunner.decode(bytes)) == SjbScript.lines(code)
        # each kernel: its counts are main.json's, and it lists
        for case in CASES, k in read_main(joinpath(case, "main.json")).kernels
            a = read_sjb(joinpath(case, k.path))
            @test (a.count_states, a.count_obs) == (k.inputs, 2 * k.output_pairs)
            @test occursin("instructions", sprint(show, MIME"text/plain"(), a)) || !isempty(listing(a.mir))
        end
    end

    @testset "complexify" begin
        rng = Xoshiro(1)
        for case in CASES, path in kernel_files(case)
            a = read_sjb(path)
            generic = read_sjb(path; lowered = true).mir
            native = read_sjb(path; lowered = true, native_complex = true).mir
            @test !any(i -> i isa ComplexBi, generic.code)
            # the same values from both lowerings on random memory (mir_run.jl, symjit's interpreter)
            any(i -> i isa Union{LoadArgs,SaveArgs} || (i isa Call && i.num_args > 0), generic.code) && continue
            for _ in 1:20
                mem = [ComplexifyScript.rand_value(rng) for _ in 1:max(64, mem_size(a) + 2)]
                stack = [ComplexifyScript.rand_value(rng) for _ in 1:max(64, a.stack_size + 2)]
                (m1, s1, m2, s2) = (copy(mem), copy(stack), copy(mem), copy(stack))
                ComplexifyScript.run_mir!(generic, m1, s1, Float64[])
                ComplexifyScript.run_mir!(native, m2, s2, Float64[])
                @test all(same.(m1, m2)) && all(same.(s1, s2))
            end
        end
    end

    @testset "Julia runner against RustiCol" begin
        # programs without kernel rows here; every program with the libraries (below)
        for case in CASES
            ps = load_programs(joinpath(case, "main.json"))
            ex = read_main(joinpath(case, "examples.json")).programs
            for (p, e) in zip(ps, ex)
                has_kernel_rows(p) && continue
                a = Arena(p)
                for pt in e.points
                    got = evaluate(p, a, [Tuple(Float64.(v)) for v in pt.momenta])
                    cpu = Float64.(pt.outputs)
                    @test maximum(abs.(got .- cpu)) <= 1e-14 * abs(sum(cpu))
                end
            end
        end
    end

    @testset "lower (UIR v0.6)" begin
        L = LowerScript
        for case in CASES, p in load_programs(joinpath(case, "main.json"))
            l = lower(p)
            @test isempty(L.roundtrip(p, l))
            @test all(c -> iseven(c.start), l.currents)
            @test length(l.edges) == length(l.currents)   # every current has one momentum in this data
        end
        # validate: a fused run is accepted; SET inside it, reading its destination, a run left open,
        # an odd plane and a wrong physical type are not
        l = lower(load_artifact(joinpath(@__DIR__, "data", "uux_zgggg", "main.json")))
        k = L.fusable(l)
        @test k !== nothing
        run = L.with_row(o -> L.setrow(o; first = false), L.with_row(o -> L.setrow(o; last = false), l, k), k + 1)
        @test isempty(PacRunner.problems(run))
        has(m, words) = any(occursin(words, s) for s in PacRunner.problems(m))
        @test has(L.with_row(o -> L.setrow(o; set = true), run, k + 1), "SET on a row that does not start its run")
        @test has(L.with_row(o -> L.setrow(o; l = run.ops[k].d), run, k + 1), "reads the planes of its run's destination")
        open = L.with_row(o -> L.setrow(o; last = false), l, k + 1)
        @test has(open, "inside the fused run") || has(open, "starts inside another run") || has(open, "does not end")
        odd = deepcopy(l)
        odd.currents[1] = PacRunner.LCurrent(1, odd.currents[1].components, odd.currents[1].type)
        @test has(odd, "starts at odd plane")
        prop = findfirst(o -> o.op == PacRunner.P_WEYL_A, l.ops)
        vec = findfirst(o -> o.op == PacRunner.P_VECTOR, l.ops)
        @test has(L.with_row(o -> L.setrow(o; d = l.ops[vec].d), l, prop), "a VECTOR")
        @test_throws ErrorException validate(odd)
    end

    @testset "evaluate(::Lowered) against evaluate(::Program)" begin
        # bit for bit on examples.json's points, plain and with adjacent rows fused (kernel rows: below)
        L = LowerScript
        joined = 0
        for case in CASES
            ps = load_programs(joinpath(case, "main.json"))
            ex = read_main(joinpath(case, "examples.json")).programs
            for (p, e) in zip(ps, ex)
                has_kernel_rows(p) && continue
                (plain, fused, j) = L.compare(p, e.points)
                @test (plain, fused) == (0, 0)
                joined += j
            end
        end
        @test joined > 0
        # the comparison sees a changed row: another factor on one vertex row
        case = joinpath(@__DIR__, "data", "ppjjj_crossed")
        n = findfirst(!has_kernel_rows, load_programs(joinpath(case, "main.json")))
        p = load_artifact(joinpath(case, "main.json"), n)
        l = lower(p)
        k = findfirst(o -> PacRunner.is_vertex(o.op), l.ops)
        m = L.with_row(o -> L.setrow(o; x = o.x == 1 ? 2 : 1), l, k)
        pt = read_main(joinpath(case, "examples.json")).programs[n].points[1]
        momenta = [Tuple(Float64.(v)) for v in pt.momenta]
        @test evaluate(l, LArena(l), momenta) == evaluate(p, Arena(p), momenta)
        @test evaluate(m, LArena(m), momenta) != evaluate(p, Arena(p), momenta)
        # the paths the data does not reach (compile's variants: crossing signs, scale and clear rows,
        # several terms per output, a group factor); a null vertex operand is refused
        points = read_main(joinpath(case, "examples.json")).programs[n].points
        for (name, v) in VariantsScript.variants(p)
            @test L.compare(v, points)[1:2] == (0, 0)
        end
        refusals = Dict(VariantsScript.refusals(p))
        @test_throws ErrorException lower(refusals["a null vertex operand"])
    end

    @testset "compile refuses" begin
        p = load_programs(joinpath(@__DIR__, "data", "ppjjj_crossed", "main.json"))[1]
        for (name, v) in VariantsScript.refusals(p)
            @test_throws ErrorException compile(mktempdir(), v)
        end
        # another program's examples
        ex = load_examples(joinpath(@__DIR__, "data", "ppjjj_crossed", "examples.json"), 2)
        @test_throws ErrorException compile(mktempdir(), p; examples = ex)
    end

    @testset "the compiled folder" begin
        case = joinpath(@__DIR__, "data", "ppjjj_crossed")
        n = findfirst(!has_kernel_rows, load_programs(joinpath(case, "main.json")))
        p = load_artifact(joinpath(case, "main.json"), n)
        files(dir) = sort(readdir(dir))
        @test files(compile(mktempdir(), p)) == ["Makefile", "pacops.c", "pacops.h", "runner.c", "runner.h"]
        dir = compile(mktempdir(), p; examples = load_examples(joinpath(case, "examples.json"), n))
        @test files(dir) == ["Makefile", "pacops.c", "pacops.h", "runner.c", "runner.h", "testing.c"]
        @test occursin("#define RUNNER_OUTPUTS $(length(p.reduction))", read(joinpath(dir, "runner.h"), String))
    end

    if HAVE_CC
        @testset "C against the Julia runner" begin
            for case in CASES
                ps = load_programs(joinpath(case, "main.json"))
                ex = read_main(joinpath(case, "examples.json")).programs
                for (p, e) in zip(ps, ex)
                    has_kernel_rows(p) && continue
                    (c, julia) = VariantsScript.both(p, e.points, mktempdir())
                    @test all(all(same.(cr, jr)) for (cr, jr) in zip(c, julia))
                end
            end
        end
        @testset "variants" begin
            p = load_programs(joinpath(@__DIR__, "data", "ppjjj_crossed", "main.json"))[1]
            points = read_main(joinpath(@__DIR__, "data", "ppjjj_crossed", "examples.json")).programs[1].points
            base = VariantsScript.both(p, points, mktempdir())[2]
            for (name, v) in VariantsScript.variants(p)
                (c, julia) = VariantsScript.both(v, points, mktempdir())
                @test all(all(same.(cr, jr)) for (cr, jr) in zip(c, julia))
                @test julia != base
            end
        end
        if HAVE_MAKE
            @testset "make test in the compiled folders" begin
                # testing.c of each program without kernel rows against examples.json (RustiCol)
                for case in CASES
                    ps = load_programs(joinpath(case, "main.json"))
                    for (n, p) in enumerate(ps)
                        has_kernel_rows(p) && continue
                        dir = compile(mktempdir(), p; examples = load_examples(joinpath(case, "examples.json"), n))
                        @test success(pipeline(`make -s -C $dir test`; stdout = devnull, stderr = devnull))
                    end
                end
            end
        else
            @info "no make: the compiled folders' make test is not run"
        end
    else
        @info "no C compiler ($CC): the compiled C is not tested"
    end

    if HAVE_CC && SJB2OBJ !== nothing
        @testset "end to end with the kernels (sjb2obj)" begin
            # every program of every case: C against Julia bit for bit, against RustiCol within 1e-14
            @test sum(c -> CompileScript.test_folder(SJB2OBJ, c), CASES) == 0
            @test VariantsScript.main(SJB2OBJ) == 0
            # evaluate(::Lowered) on every program, kernel rows included
            @test sum(c -> LowerScript.test_folder(c; sjb2obj = SJB2OBJ), CASES) == 0
            # and on compile's variants of programs with kernel rows (direct bindings, an input from out)
            for (case, n) in [("uux_zgg", 1), ("ppjjj_crossed", 4)]
                dir = CompileScript.prepare(SJB2OBJ, joinpath(@__DIR__, "data", case); quiet = true)
                p = load_programs(joinpath(dir, "main.json"))[n]
                points = read_main(joinpath(dir, "examples.json")).programs[n].points
                for (name, v) in VariantsScript.variants(p)
                    @test LowerScript.compare(v, points)[1:2] == (0, 0)
                end
                f = Dict(VariantsScript.refusals(p))["a FACTOR kernel input"]
                @test LowerScript.compare(f, points)[1:2] == (0, 0)   # lower reads it; compile does not yet
            end
        end
    else
        @info "no sjb2obj (ENV[\"SJB2OBJ\"] or the PATH): programs with kernel rows are not run"
    end
end
