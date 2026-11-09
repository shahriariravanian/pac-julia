# Short tests of PacRunner on the cases of test/data (`] test`, or julia --project test/runtests.jl).
#
# Without external tools: the loader, the .sjb reader (a synthetic code buffer of every
# instruction, the kernels' headers), the Complexifier (each kernel's lowering runs), the Julia runner against RustiCol's outputs (programs without
# kernel rows), and compile's refusals.
# With a C compiler (CC, default cc): the compiled C against the Julia runner, the variants, the
# kernels as C (kernel_c.jl), and (with make) the compiled folders' own `make test` of every program,
# kernel rows included, straight-line and on the UIR interpreter (c/uirvm.c, whose field macros are also
# checked, field by field, against decode_row on one-word rows).
# With nvcc and a GPU (NVCC, nvidia-smi): every program on the GPU build of the interpreter, bit for
# bit its CPU build.
# With symjit-utils' sjb2obj (ENV["SJB2OBJ"], or sjb2obj on the PATH): the kernels' libraries are
# built, the kernels' C is checked against them bit for bit, and every program of every case runs,
# in Julia and in C.
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
module KernelCScript include(joinpath(@__DIR__, "test_kernel_c.jl")) end
module GpuScript include(joinpath(@__DIR__, "test_gpu.jl")) end
module Uir08Script include(joinpath(@__DIR__, "test_uir08.jl")) end

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
        # test/data holds no libraries: with kernels = :none they load as not built, and a call is an error
        p = load_artifact(joinpath(@__DIR__, "data", "uux_zgg", "main.json"); kernels = :none)
        @test !built(p.bindings[1].kernel)
        @test_throws ErrorException evaluate(p, Arena(p), [(500.0, 0.0, 0.0, 500.0) for _ in 1:p.legs])
        # a folder is its main.json
        @test length(load_programs(joinpath(@__DIR__, "data", "ppjjj_crossed"); kernels = :none)) == 21
        @test load_examples(joinpath(@__DIR__, "data", "uux_zgg")).process == p.process
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
        # each kernel lowers to real registers (symjit's ComplexBi kept), and the lowering runs
        # (mir_run.jl, symjit's interpreter); its values are the kernels' C's below and, against
        # symjit's own lowering, test_complexify.jl's
        rng = Xoshiro(1)
        for case in CASES, path in kernel_files(case)
            a = read_sjb(path)
            l = read_sjb(path; lowered = true).mir
            regs = [getfield(i, f) for i in l.code for f in fieldnames(typeof(i)) if getfield(i, f) isa PacRunner.Reg]
            @test all(r -> r.kind != PacRunner.REG_STATIC && r.index < 64, regs)
            @test any(i -> i isa ComplexBi, l.code)
            mem = [ComplexifyScript.rand_value(rng) for _ in 1:max(64, mem_size(a) + 2)]
            stack = [ComplexifyScript.rand_value(rng) for _ in 1:max(64, a.stack_size + 2)]
            ComplexifyScript.run_mir!(l, mem, stack, Float64[])
            @test any(!iszero, mem[a.count_states+1:a.count_states+a.count_obs])
        end
    end

    if HAVE_CC
        @testset "Julia runner with kernel_c libraries" begin
            # every program, kernel rows included: the kernels built from kernel_c's C (kernel_lib.jl),
            # against RustiCol within 1e-14 of the total
            for case in CASES
                ps = load_programs(case; kernels = :c)
                @test all(k -> built(k), [b.kernel for p in ps for b in p.bindings])
                for (p, e) in zip(ps, read_main(joinpath(case, "examples.json")).programs), pt in e.points
                    got = evaluate(p, Arena(p), [Tuple(Float64.(v)) for v in pt.momenta])
                    cpu = Float64.(pt.outputs)
                    @test maximum(abs.(got .- cpu)) <= 1e-14 * abs(sum(cpu))
                end
                # and bit for bit the sjb2obj libraries' values
                if SJB2OBJ !== nothing
                    withenv("SJB2OBJ" => SJB2OBJ) do
                        qs = load_programs(case; kernels = :sjb2obj)
                        for (p, q, e) in zip(ps, qs, read_main(joinpath(case, "examples.json")).programs), pt in e.points
                            m = [Tuple(Float64.(v)) for v in pt.momenta]
                            @test all(same.(evaluate(p, Arena(p), m), evaluate(q, Arena(q), m)))
                        end
                    end
                end
            end
        end
    end

    @testset "artifacts" begin
        # check_artifact: what pac2json records (a pyAmpliCol LC process or process-set artifact of
        # schema 3, with its UMAMI provider), and the artifact_id that names its capture
        id = repeat("ab", 32)
        function fake(; kind = "pyamplicol-process", schema = 3, colour = "lc", provider = true, artifact_id = id)
            dir = mktempdir()
            write(joinpath(dir, "artifact.json"), PacRunner.JSON.json(Dict("artifact_id" => artifact_id, "kind" => kind,
                "schema_version" => schema, "runtime" => Dict("engine" => "rusticol"),
                "producer" => Dict("version" => "0.2.1", "versions" => Dict("process_artifact" => schema)),
                "processes" => [Dict("id" => "p", "color_accuracy" => colour)])))
            provider && (mkpath(joinpath(dir, "API", "umami")); touch(joinpath(dir, "API", "umami", "provider_p0.h")))
            return dir
        end
        @test PacRunner.is_artifact(fake())
        @test !PacRunner.is_artifact(joinpath(@__DIR__, "data", "uux_zgg"))
        @test PacRunner.check_artifact(fake()) == id
        @test PacRunner.check_artifact(fake(kind = "pyamplicol-process-set")) == id
        @test_throws ErrorException PacRunner.check_artifact(fake(kind = "something"))
        @test_throws ErrorException PacRunner.check_artifact(fake(schema = 4))
        @test_throws ErrorException PacRunner.check_artifact(fake(colour = "nlc"))
        @test_throws ErrorException PacRunner.check_artifact(fake(provider = false))
        @test_throws ErrorException PacRunner.check_artifact(fake(artifact_id = "not a sha"))
        @test_throws ErrorException PacRunner.check_artifact(joinpath(@__DIR__, "data", "uux_zgg"))
        # a pac2json folder or main.json is used as it is (no capture)
        @test PacRunner.main_json(joinpath(@__DIR__, "data", "uux_zgg")) == joinpath(@__DIR__, "data", "uux_zgg", "main.json")
        # with pac2json installed (PacRunner.install()) and ENV["PACRUNNER_ARTIFACT"] an LC artifact:
        # captured once, then found by its artifact_id
        art = get(ENV, "PACRUNNER_ARTIFACT", nothing)
        if art !== nothing && isfile(PacRunner.pac2json_path())
            dir = capture(art)
            @test basename(dir) == PacRunner.check_artifact(art)
            @test isfile(joinpath(dir, "main.json")) && isfile(joinpath(dir, "examples.json"))
            @test capture(art) == dir
            @test length(load_programs(art)) == length(read_main(joinpath(dir, "main.json")).programs)
            # the alpha_s powers, measured from a second capture at twice alpha_s
            powers = alpha_s_power(art)
            @test length(powers) == length(load_programs(art)) && all(p -> p === nothing || p >= 0, powers)
            @test [p.alpha_s === nothing ? nothing : p.alpha_s[2] for p in load_programs(art; measure_alpha_s = true)] == powers
        end
    end

    @testset "alpha_s power" begin
        # u u~ > Z g g at alpha_s and at 2 alpha_s (pac-miner's pac2json; RustiCol measured 2)
        base = joinpath(@__DIR__, "data", "uux_zgg")
        doubled = joinpath(@__DIR__, "data", "alpha_s", "uux_zgg_2as")
        @test alpha_s_power(base, doubled) == [2]
        @test alpha_s_power(doubled, base) == [2]
        @test_throws ErrorException alpha_s_power(base, base)       # the same alpha_s twice
        # a power not measured (null, as pac-miner writes it) reads as nothing
        @test load_programs(doubled)[1].alpha_s == (0.236, nothing)
        @test PacRunner.with_alpha_s_power(load_programs(doubled)[1], 2).alpha_s == (0.236, 2)
        # measured on load only for an artifact (a folder has no second capture)
        @test_throws ErrorException load_programs(doubled; measure_alpha_s = true)
        @test load_programs(base; measure_alpha_s = true)[1].alpha_s == (0.118, 2)  # measured by the capture: kept
        # the momenta: massless, momentum conserved
        for m in PacRunner.rambo_momenta(5, 4)
            @test all(abs(p[1]^2 - p[2]^2 - p[3]^2 - p[4]^2) <= 1e-9 * p[1]^2 for p in m)
            @test all(abs(m[1][k] + m[2][k] - sum(p[k] for p in m[3:end])) <= 1e-9 for k in 1:4)
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

    @testset "lower (UIR v0.8)" begin
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
                r = L.compare(p, e.points)
                @test r.differ == 0
                joined += r.fused
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
            @test L.compare(v, points).differ == 0
        end
        # reuse_planes on currents that start at zero (no fresh first write): the same values
        acc = L.first_writes_add(p)
        @test evaluate(acc, Arena(acc), momenta) == evaluate(p, Arena(p), momenta)
        @test L.compare(acc, points).differ == 0
        @test optimize(lower(acc)).planes > optimize(lower(p)).planes
        refusals = Dict(VariantsScript.refusals(p))
        @test_throws ErrorException lower(refusals["a null vertex operand"])
    end

    @testset "encode (UIR v0.8)" begin
        P = PacRunner
        # the word layout: op 0-4, EXT 5, SET 6, FIRST 7, LAST 8, D 9-21, T 22-31, L 32-44, R 45-57, X 58-63,
        # N 32-47; values 0-based
        o = P.LOp(P.V_THREE_VECTOR; set = true, first = true, d = 3, t = 2, l = 5, r = 7, x = 4)
        @test P.encode_row!(UInt64[], o) == [UInt64(4) | 1 << 6 | 1 << 7 | 2 << 9 | 1 << 22 | 4 << 32 | 6 << 45 | 3 << 58]
        c = P.LOp(P.CLEAR; d = 1, n = 6)
        @test P.encode_row!(UInt64[], c) == [UInt64(Int(P.CLEAR)) | UInt64(6) << 32]
        # a kernel's null operand is all ones; decoded, it is 0 again
        k = P.LOp(P.KERNEL; d = 1, t = 1, l = 2, r = 0, x = 1)
        @test (P.encode_row!(UInt64[], k)[1] >> 45) & 0x1fff == 0x1fff
        @test P.decode_row(P.encode_row!(UInt64[], k), 1) == (k, 2)
        # no field crosses bit 32, in word 0 or in word 1
        for f in (P.FD, P.FT, P.FL, P.FR, P.FN, P.FX)
            @test (f.shift + f.width <= 32 || f.shift >= 32) && (f.hishift + f.hiwidth <= 32 || f.hishift >= 32)
        end
        # the fields of an opcode never share a bit, in word 0 (with op and the flags, bits 0-8) or in word 1 (with EXT, bit 0)
        bits(shift, width) = width == 0 ? UInt64(0) : ((UInt64(1) << width) - 1) << shift
        for op in instances(P.Opcode)
            (w0, w1) = (bits(0, 9), bits(0, 1))
            for (_, f, _) in P.fields(op)
                @test w0 & bits(f.shift, f.width) == 0 && w1 & bits(f.hishift, f.hiwidth) == 0
                (w0, w1) = (w0 | bits(f.shift, f.width), w1 | bits(f.hishift, f.hiwidth))
            end
        end
        # random rows of every opcode: one word when the values fit, two when not, and back
        rng = Xoshiro(7)
        random_rows = P.LOp[]
        for op in instances(P.Opcode), wide in (false, true), _ in 1:50
            op == P.END && continue
            fs = P.fields(op)
            val(f, index) = index ? rand(rng, 1:(wide ? 2^(f.width + f.hiwidth) - 1 : 2^f.width - 1)) :
                                    rand(rng, 0:(wide ? 2^(f.width + f.hiwidth) - 1 : 2^f.width - 1))
            v = Dict(s => val(f, index) for (s, f, index) in fs)
            wide && (v[:d] = 2^13 + rand(rng, 0:1000))      # forces the wide form
            P.is_kernel(op) && rand(rng, Bool) && (v[:r] = 0)
            flags = P.is_vertex(op) ? (rand(rng, Bool), rand(rng, Bool), rand(rng, Bool)) :
                    P.is_kernel(op) ? (rand(rng, Bool), false, false) : (false, false, false)
            row = P.LOp(op, flags..., get(v, :d, 0), get(v, :t, 0), get(v, :l, 0), get(v, :r, 0), get(v, :n, 0), get(v, :x, 0))
            words = P.encode_row!(UInt64[], row)
            @test length(words) == (wide ? 2 : 1)
            @test P.decode_row(words, 1) == (row, length(words) + 1)
            push!(random_rows, row)
        end
        # uirvm.h's field macros on the one-word rows (uirvm reads no wide rows): each field as
        # decode_row reads it, 0-based
        narrow = filter(o -> length(P.encode_row!(UInt64[], o)) == 1, random_rows)
        if HAVE_CC
            dir = mktempdir()
            exe = joinpath(dir, "decode_driver")
            home = joinpath(@__DIR__, "..")
            run(`$CC -std=c99 -O2 -Wall -Werror -I$(joinpath(home, "c")) -o $exe $(joinpath(@__DIR__, "decode_driver.c")) $(joinpath(home, "c", "pacops.c")) -lm`)
            words = foldl(P.encode_row!, narrow; init = UInt64[])
            text = read(pipeline(`$exe`; stdin = IOBuffer(join(string.(words; base = 16), " "))), String)
            lines = split(strip(text), "\n")
            @test length(lines) == length(narrow)
            # an index 0-based, a Lowered's none (0) as -1; a count as it is
            # (an index field the driver prints as -1 for none; a REDUCE's N, an index, is printed raw: 0xffff)
            want(o) = Dict(s => getfield(o, s) - index for (s, _, index) in P.fields(o.op))
            decoded = 0
            for (o, line) in zip(narrow, lines)
                v = parse.(Int, split(line))
                n = o.op == P.REDUCE ? (v[9] == 0xffff ? -1 : v[9]) : v[9]
                got = Dict(:d => v[5], :t => v[6], :l => o.op == P.SOURCE ? v[11] : v[7], :r => v[8], :n => n, :x => v[10])
                ok = v[1:4] == [Int(o.op), o.set, o.first, o.last] && all(got[s] == w for (s, w) in want(o))
                decoded += ok
            end
            @test decoded == length(narrow)
        end
        # beyond the wide form: refused
        @test_throws ErrorException P.encode_row!(UInt64[], P.LOp(P.V_WEYL_VECTOR_A; first = true, last = true, d = 2^20 + 1, t = 1, l = 1, r = 1, x = 1))
        @test_throws ErrorException P.encode_row!(UInt64[], P.LOp(P.P_VECTOR; d = 1, x = 2^16 + 1))
        # a source: D edge, T mass scale, N template (family 0-2, antiparticle 3, helicity + 2 at 4-6,
        # chirality + 1 at 7-8), E embedding (bits 48-57; none: all ones), X factor; every template value reads back
        src = P.source_op(3, P.WEYL, true, -1, 1, 2, 5)
        @test P.encode_row!(UInt64[], src) ==
              [UInt64(Int(P.SOURCE)) | 2 << 9 | 1 << 22 | UInt64(2 | 1 << 3 | 1 << 4 | 2 << 7) << 32 | UInt64(0x3ff) << 48 |
               UInt64(4) << 58]
        for ty in (P.SCALAR, P.WEYL, P.DIRAC, P.VECTOR), anti in (false, true), h in -2:2, ch in -1:1
            s = P.source_op(7, ty, anti, h, ch, 1, 1)
            d = P.decode_row(P.encode_row!(UInt64[], s), 1)[1]
            @test d == s && P.source_fields(d) == (ty, anti, h, ch)
        end
        # an embedded source: its embedding in E, no factor (X none)
        emb = P.source_op(3, P.VECTOR, false, 1, 0, 2, 5; embedding = 4)
        @test (emb.l, emb.x) == (4, 0) && P.decode_row(P.encode_row!(UInt64[], emb), 1)[1] == emb
        # END: op 31, no fields, one word
        @test P.encode_row!(UInt64[], P.LOp(P.END)) == [UInt64(31)]
        # every program: the rows (SOURCE rows at the head), one word each, then END; the decoded program
        # evaluates as the Lowered (in the comparisons above, through LowerScript.passes)
        for case in CASES, p in load_programs(joinpath(case, "main.json"))
            l = optimize(lower(p))
            e = encode(l)
            @test P.decode_stream(e) == l.ops
            @test length(e.rows) == length(l.ops) + 1 && e.rows[end] == UInt64(31)
            @test [P.decode_edge(x)[1] for x in e.edges] == [P.edge_plane(l, k) for k in eachindex(l.edges)]
        end
    end

    @testset "compile refuses" begin
        p = load_programs(joinpath(@__DIR__, "data", "ppjjj_crossed", "main.json"))[1]
        for (name, v) in VariantsScript.refusals(p)
            @test_throws ErrorException compile(mktempdir(), v)
        end
        # another program's examples
        ex = load_examples(joinpath(@__DIR__, "data", "ppjjj_crossed", "examples.json"), 2)
        @test_throws ErrorException compile(mktempdir(), p; examples = ex)
        # the interpreter: a wide row (here a factor index beyond X's 6 bits)
        l = lower(p)
        factors = vcat(l.factors, fill(1.0 + 0.0im, 64))
        k = findfirst(o -> PacRunner.is_propagator(o.op), l.ops)
        o = l.ops[k]
        ops = copy(l.ops)
        ops[k] = PacRunner.LOp(o.op, o.set, o.first, o.last, o.d, o.t, o.l, o.r, o.n, length(factors))
        wide = PacRunner.with(l; factors, ops)
        @test length(encode(wide).rows) > length(wide.ops) + 1
        @test_throws ErrorException compile(mktempdir(), wide; interpreter = true)
        @test isdir(compile(mktempdir(), wide))   # the straight-line C has no word fields
        # the interpreter: closure and scale rows of more than 6 components (an UNTYPED current may
        # have up to 15); vm_unsupported finds them (the rows here are made up: not valid programs)
        @test isempty(PacRunner.vm_unsupported(l))
        k = findfirst(o -> o.op == PacRunner.CLOSURE, l.ops)
        lists = copy(l.lists)
        lists[l.ops[k].t] = fill(1, 8)
        ops = copy(l.ops)
        push!(ops, PacRunner.LOp(PacRunner.SCALE_ROW; d = 1, n = 7, x = 1))
        sharing = count(o -> o.op == PacRunner.CLOSURE && o.t == l.ops[k].t, l.ops)   # every closure of that list
        @test length(PacRunner.vm_unsupported(PacRunner.with(l; lists, ops))) == sharing + 1
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
        # optimize (the default) against the plain lowering: a smaller arena
        plain = compile(mktempdir(), p; optimize = false)
        @test CompileScript.arena_size(plain) == ArenaIndex(lower(p)).size > CompileScript.arena_size(dir) ==
              ArenaIndex(optimize(lower(p))).size
        # on the UIR interpreter: its source next to pacops, the same runner.h
        vm = compile(mktempdir(), p; interpreter = true)
        @test files(vm) == ["Makefile", "pacops.c", "pacops.h", "runner.c", "runner.h", "uirvm.c", "uirvm.h"]
        @test read(joinpath(vm, "runner.h"), String) == read(joinpath(compile(mktempdir(), p), "runner.h"), String)
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
                    # and the plain lowering (compile's optimize = false)
                    dir = compile(mktempdir(), p; optimize = false)
                    @test all(all(same.(cr, jr)) for (cr, jr) in zip(CompileScript.run_compiled(dir, p, e.points), julia))
                    # and on the UIR interpreter, optimized and plain
                    for optimize in (true, false)
                        dir = compile(mktempdir(), p; optimize, interpreter = true)
                        @test all(all(same.(cr, jr)) for (cr, jr) in zip(CompileScript.run_compiled(dir, p, e.points), julia))
                    end
                end
            end
        end
        @testset "UIR v0.8 rows (copy, scatter, embedded sources, scaled reduce) on every CPU back end" begin
            @test isempty(Uir08Script.check(joinpath(@__DIR__, "data", "ppjjj_crossed"); quiet = true))
        end
        @testset "C of an optimized lowering" begin
            # sorted, fused and with reused planes: the C writes a fused run row by row, and by the run
            # invariants the values are the same
            for case in CASES
                ps = load_programs(joinpath(case, "main.json"))
                ex = read_main(joinpath(case, "examples.json")).programs
                n = findfirst(!has_kernel_rows, ps)
                n === nothing && continue
                fused = optimize(lower(ps[n]))
                joined = PacRunner.fused_rows(fused)
                joined > 0 || continue
                got = CompileScript.run_compiled(compile(mktempdir(), fused), ps[n], ex[n].points)
                a = Arena(ps[n])
                want = [evaluate(ps[n], a, [Tuple(Float64.(v)) for v in pt.momenta]) for pt in ex[n].points]
                @test all(all(same.(g, w)) for (g, w) in zip(got, want))
            end
        end
        @testset "variants" begin
            p = load_programs(joinpath(@__DIR__, "data", "ppjjj_crossed", "main.json"))[1]
            points = read_main(joinpath(@__DIR__, "data", "ppjjj_crossed", "examples.json")).programs[1].points
            base = VariantsScript.both(p, points, mktempdir())[2]
            for (name, v) in VariantsScript.variants(p), interpreter in (false, true)
                (c, julia) = VariantsScript.both(v, points, mktempdir(); interpreter)
                @test all(all(same.(cr, jr)) for (cr, jr) in zip(c, julia))
                @test julia != base
            end
        end
        @testset "kernels as C" begin
            # each kernel's C compiles without warnings
            for case in CASES, k in read_main(joinpath(case, "main.json")).kernels
                src = joinpath(mktempdir(), "$(k.name).c")
                write(src, PacRunner.KERNEL_PRELUDE * kernel_c(k.name, joinpath(case, k.path)))
                @test success(pipeline(`$CC -std=c99 -O2 -Wall -Wextra -Werror -Wno-unused-function -c -o $src.o $src`; stderr = devnull))
            end
            # refused: parameters, a kernel not saved for a direct arena, a math function
            a = read_sjb(kernel_files(joinpath(@__DIR__, "data", "uux_zgg"))[1]; lowered = true)
            with(a; kw...) = PacRunner.SavedApplication((get(kw, f, getfield(a, f)) for f in fieldnames(PacRunner.SavedApplication))...)
            @test_throws PacRunner.KernelCError kernel_c("k", with(a; count_params = 1))
            c = a.config
            nodirect = PacRunner.Config(c.opt & ~PacRunner.CONFIG_OPTIONS.direct_arena, c.ty, c.stack_limit, c.num_args)
            @test_throws PacRunner.KernelCError kernel_c("k", with(a; config = nodirect))
            sin = PacRunner.MIR(vcat(a.mir.code, [Call("sin", 1)]), a.mir.consts)
            @test_throws PacRunner.KernelCError kernel_c("k", with(a; mir = sin))
            # bit for bit sjb2obj's libraries
            if SJB2OBJ !== nothing
                @test sum(c -> KernelCScript.test_folder(SJB2OBJ, c; quiet = true), CASES) == 0
            end
        end
        if HAVE_MAKE
            @testset "make test in the compiled folders" begin
                # testing.c of each program against examples.json (RustiCol); the kernels are C, so
                # no sjb2obj is needed
                for case in CASES
                    ps = load_programs(joinpath(case, "main.json"))
                    for (n, p) in enumerate(ps)
                        for interpreter in (false, true)
                            dir = compile(mktempdir(), p; examples = load_examples(joinpath(case, "examples.json"), n), interpreter)
                            @test success(pipeline(`make -s -C $dir test`; stdout = devnull, stderr = devnull))
                        end
                    end
                end
            end
        else
            @info "no make: the compiled folders' make test is not run"
        end
    else
        @info "no C compiler ($CC): the compiled C is not tested"
    end

    if HAVE_CC && HAVE_MAKE && GpuScript.have_gpu()
        @testset "the interpreter on the GPU" begin
            @test sum(c -> GpuScript.test_folder(c; quiet = true), CASES) == 0
        end
        @testset "UIR v0.8 rows on the GPU" begin
            @test isempty(Uir08Script.check(joinpath(@__DIR__, "data", "ppjjj_crossed"); gpu = GpuScript.NVCC, quiet = true))
        end
    else
        @info "no nvcc or no GPU: the GPU build is not run"
    end

    if HAVE_CC && SJB2OBJ !== nothing
        @testset "end to end with the kernels (sjb2obj)" begin
            # every program of every case: C against Julia bit for bit, against RustiCol within 1e-14
            @test sum(c -> CompileScript.test_folder(SJB2OBJ, c), CASES) == 0
            @test sum(c -> CompileScript.test_folder(SJB2OBJ, c; interpreter = true), CASES) == 0
            @test VariantsScript.main(SJB2OBJ) == 0
            # evaluate(::Lowered) on every program, kernel rows included
            @test sum(c -> LowerScript.test_folder(c; sjb2obj = SJB2OBJ), CASES) == 0
            # and on compile's variants of programs with kernel rows (direct bindings, an input from out)
            for (case, n) in [("uux_zgg", 1), ("ppjjj_crossed", 4)]
                dir = CompileScript.prepare(SJB2OBJ, joinpath(@__DIR__, "data", case); quiet = true)
                p = load_programs(joinpath(dir, "main.json"))[n]
                points = read_main(joinpath(dir, "examples.json")).programs[n].points
                for (name, v) in VariantsScript.variants(p)
                    @test LowerScript.compare(v, points).differ == 0
                end
            end
        end
    else
        @info "no sjb2obj (ENV[\"SJB2OBJ\"] or the PATH): the comparisons with its native kernel libraries are not run"
    end
end
