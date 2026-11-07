# complexify.jl against symjit's Complexifier, for each complex .sjb (or .plane.symjit) file given
# and for random complex programs (written as .sjb files: every instruction a saved complex model
# can hold, on random operands, with random real locations):
#   1. the MIR: read_sjb(path; lowered = true, native_complex = true) and `sjbdump --lowered`
#      (test/sjbdump), line for line, with the saved stable_math option and with it forced on, off;
#   2. the values: the default lowering (ComplexBi replaced by real operations) run by mir_run.jl
#      and symjit's lowering run by symjit's bytecode interpreter (`sjbdump --run`), the memory and
#      the stack bit for bit (any NaN equal to any NaN), on random inputs (programs without calls or branches). The native
#      lowering run by mir_run.jl is checked the same way (the interpreter), and the generic
#      division with symjit's scratch register, Gen(0), is counted (it is wrong when Gen(0) holds an
#      operand).
# A program both reject (symjit panics on some operands, e.g. a fused operation adding Ret) counts
# as agreement.
# usage: julia test_complexify.jl SJBDUMP [N] [FILE_OR_DIR...]   (N random programs, default 1000)
include(joinpath(@__DIR__, "..", "sjb.jl"))
include(joinpath(@__DIR__, "sjb_encode.jl"))
include(joinpath(@__DIR__, "mir_run.jl"))
using Random

lines(mir::MIR) = join([lpad(k - 1, 5, '0') * "\t" * (i isa Call ? "call $(i.label) / $(i.num_args)" : sprint(show, i)) * "\n"
                        for (k, i) in enumerate(mir.code)]) *
                  join(["mir const " * string(reinterpret(UInt64, x), base = 16, pad = 16) * "\n" for x in mir.consts])

# (outcome, text): :ok and the listing, or :rejected
function by_symjit(dump, path, stable)
    out = IOBuffer()
    cmd = stable === nothing ? `$dump --lowered $path` : `$dump --lowered --stable $stable $path`
    ok = success(pipeline(cmd, stdout = out, stderr = devnull))
    return ok ? (:ok, String(take!(out))) : (:rejected, "")
end

function by_julia(path, stable)
    try
        return (:ok, lines(read_sjb(path; lowered = true, stable_math = stable, native_complex = true).mir))
    catch e
        e isa ComplexifyError || e isa BoundsError || rethrow()
        return (:rejected, "")
    end
end

# compares the two lowerings of a file; (agree, rejected)
function compare(dump, path, stable; quiet = false)
    (a, b) = (by_julia(path, stable), by_symjit(dump, path, stable))
    a == b && return (true, a[1] == :rejected)
    if !quiet
        if a[1] != b[1]
            println("DIFFERS $path (stable_math $stable): sjb.jl $(a[1]), symjit $(b[1])")
        else
            (x, y) = (split(a[2], '\n'), split(b[2], '\n'))
            k = findfirst(i -> get(x, i, "") != get(y, i, ""), 1:max(length(x), length(y)))
            println("DIFFERS $path (stable_math $stable), line $k:\n  sjb.jl:  ", get(x, k, "(none)"), "\n  symjit:  ", get(y, k, "(none)"))
        end
    end
    return (false, false)
end

# ---- random complex programs -------------------------------------------------------------------------

const REGS = [[Reg(REG_GEN, k) for k in 0:9]; Reg(REG_RET, 0); Reg(REG_TEMP, 0); Reg(REG_LEFT, 0); Reg(REG_RIGHT, 0)]
reg_(rng) = rand(rng) < 0.85 ? Reg(REG_GEN, rand(rng, 0:9)) : rand(rng, REGS)
loc_(rng, kinds = (LOC_MEM, LOC_STACK, LOC_PARAM)) = Loc(rand(rng, kinds), 2rand(rng, 0:5) + (rand(rng) < 0.5 ? 0 : 20))

function random_instruction(rng, nconsts)
    r() = reg_(rng)
    k = rand(rng, 1:21)
    k == 1 ? Uni(rand(rng, UNI_OPS), r(), r()) :
    k == 2 ? Bi(rand(rng, BIN_OPS), r(), r(), r()) :
    k == 3 ? Mov(r(), r()) :
    k in (4, 5) ? Load(r(), loc_(rng)) :
    k == 6 ? Save(r(), loc_(rng, (LOC_MEM, LOC_STACK))) :
    k == 7 ? LoadConst(r(), rand(rng, 0:nconsts-1)) :
    k == 8 ? Fused(rand(rng, FUSED_OPS), r(), r(), r(), Reg(REG_GEN, rand(rng, 0:9))) :
    k == 9 ? IfElse(r(), r(), r(), loc_(rng, (LOC_STACK,))) :
    k == 10 ? LoadMath(rand(rng, ARITH_OPS), r(), r(), loc_(rng)) :
    k == 11 ? LoadConstMath(rand(rng, ARITH_OPS), r(), r(), rand(rng, 0:nconsts-1)) :
    k == 12 ? ComplexBi(rand(rng, (:Plus, :Minus)), r(), r(), r(), r(), r(), r()) :
    k == 13 ? rand(rng, [Call("sin", 1), Call("exp", 1), Call("power", 2), Call("ln", 1), Call("f", 0), Call("@self", 3)]) :
    k == 14 ? Label("L$(rand(rng, 1:3))") :
    k == 15 ? Branch(rand(rng, ["L1", "L2", ".ret"])) :
    k == 16 ? BranchIf(r(), "L$(rand(rng, 1:3))", rand(rng, Bool)) :
    k == 17 ? LoadArgs([loc_(rng) for _ in 1:rand(rng, 0:3)], rand(rng, Bool), rand(rng, Bool)) :
    k == 18 ? SaveArgs(rand(rng, 0:3), rand(rng, Bool), rand(rng, Bool)) :
    k == 19 ? Uni(rand(rng, (:Root, :RealRoot, :Abs, :Recip)), r(), r()) :  # the funclet, stable_math
    k == 20 ? Bi(:Divide, r(), r(), r()) :
    rand(rng, [Nop(), End()])
end

function random_programs(dump, n)
    rng = MersenneTwister(1)
    (agree, rejected) = (0, 0)
    path = tempname() * ".sjb"
    for p in 1:n
        consts = [rand(rng, (1.0, -1.0, 0.5, 2.0, 0.0, 3.25)) for _ in 1:rand(rng, 1:4)]
        code = [random_instruction(rng, length(consts)) for _ in 1:rand(rng, 5:40)]
        reals = unique([loc_(rng, (LOC_STACK, LOC_PARAM)) for _ in 1:rand(rng, 0:6)])
        opt = CONFIG_OPTIONS.complex | (rand(rng, Bool) ? CONFIG_OPTIONS.stable_math : 0x0)
        write(path, sjb_bytes(code; opt = UInt32(opt), consts, reals))
        (ok, rej) = compare(dump, path, nothing; quiet = true)
        if ok
            agree += 1
            rejected += rej
        else
            cp(path, "complexify_failure_$p.sjb"; force = true)
            compare(dump, path, nothing)
            println("  saved as complexify_failure_$p.sjb")
        end
    end
    rm(path; force = true)
    println("random programs: $n, $(n - agree) differ ($rejected rejected by both)")
    return agree == n
end

# ---- 2. the values ---------------------------------------------------------------------------------

# the text with every NaN as one (the sign and payload of a NaN are not specified by IEEE 754 and
# differ between compilers' instruction choices); all other values, ±0 included, bit for bit
canonical(text) = replace(text, r"\b[7f]ff[89a-f][0-9a-f]{12}\b" => "nan", r"\b[7f]ff[0-7](?!0{12})[0-9a-f]{12}\b" => "nan")

f64_section(xs) = vcat(reinterpret(UInt8, [htol(UInt64(length(xs)))]), reinterpret(UInt8, htol.(bits.(xs))))

# random inputs: mostly normal values, some zeros, ones and negative zeros
rand_value(rng) = (r = rand(rng); r < 0.1 ? 0.0 : r < 0.15 ? -0.0 : r < 0.2 ? 1.0 : randn(rng) * 10.0^rand(rng, -3:3))

# The values of the lowerings of `path` on one random input: (:ok, counts) with counts =
# (generic differs, native differs, Gen(0)-scratch differs) from symjit's run, or :skipped
function run_once(dump, path, rng)
    a = read_sjb(path)
    (nm, ns, np) = (max(64, mem_size(a) + 2), max(64, a.stack_size + 2), max(64, a.count_params + 2))
    (mem, stack, params) = ([rand_value(rng) for _ in 1:nm], [rand_value(rng) for _ in 1:ns], [rand_value(rng) for _ in 1:np])
    input = tempname()
    write(input, vcat(f64_section(mem), f64_section(stack), f64_section(params)))
    out = IOBuffer()
    ok = success(pipeline(`$dump --run $path $input`, stdout = out, stderr = devnull))
    rm(input)
    ok || return (:skipped, nothing)
    expected = String(take!(out))

    stable = a.config.opt & CONFIG_OPTIONS.stable_math != 0
    lowerings = try
        [read_sjb(path; lowered = true).mir, read_sjb(path; lowered = true, native_complex = true).mir,
         complexify(a.mir, a.reals; stable_math = stable, divide_scratch = Reg(REG_GEN, 0))]
    catch e
        e isa ComplexifyError || rethrow()
        return (:skipped, nothing)
    end
    # mir_run.jl runs no math functions (symjit's interpreter does)
    any(i -> i isa Union{LoadArgs,SaveArgs} || (i isa Call && i.num_args > 0), lowerings[1].code) && return (:skipped, nothing)
    differs = map(lowerings) do mir
        (m, st) = (copy(mem), copy(stack))
        run_mir!(mir, m, st, copy(params))
        got = join(["mem $(string(bits(x), base = 16, pad = 16))\n" for x in m]) *
              join(["stack $(string(bits(x), base = 16, pad = 16))\n" for x in st])
        canonical(got) != canonical(expected)
    end
    return (:ok, differs)
end

# the random programs of the run check: no calls, branches or argument lists (a square root still
# calls its funclet)
function runnable_instruction(rng, nconsts)
    while true
        i = random_instruction(rng, nconsts)
        i isa Union{Call,Branch,BranchIf,LoadArgs,SaveArgs} || return i
    end
end

function check_values(dump, files, n)
    rng = MersenneTwister(2)
    (runs, counts) = (0, [0, 0, 0])
    tally(r) = r[1] == :ok && (runs += 1; counts .+= r[2])
    for f in files, _ in 1:5
        tally(run_once(dump, f, rng))
    end
    file_runs = runs
    path = tempname() * ".sjb"
    for _ in 1:n
        consts = [rand(rng, (1.0, -1.0, 0.5, 2.0, 0.0, 3.25)) for _ in 1:rand(rng, 1:4)]
        code = [runnable_instruction(rng, length(consts)) for _ in 1:rand(rng, 5:40)]
        reals = unique([loc_(rng, (LOC_STACK, LOC_PARAM)) for _ in 1:rand(rng, 0:6)])
        opt = CONFIG_OPTIONS.complex | (rand(rng, Bool) ? CONFIG_OPTIONS.stable_math : 0x0)
        write(path, sjb_bytes(code; opt = UInt32(opt), consts, reals))
        r = run_once(dump, path, rng)
        tally(r)
        if r[1] == :ok && r[2][1]
            cp(path, "generic_failure_$(runs).sjb"; force = true)
            println("  the generic lowering differs: saved as generic_failure_$(runs).sjb")
        end
    end
    rm(path; force = true)
    println("values: $runs runs ($file_runs of the files, $(runs - file_runs) of random programs): generic lowering ",
            "$(counts[1]) differ, native lowering $(counts[2]) differ; with Gen(0) as the division scratch $(counts[3]) differ")
    return counts[1] == 0 && counts[2] == 0
end

function main(dump, n, args)
    files = String[]
    for a in args
        isdir(a) ? append!(files, [joinpath(r, f) for (r, _, fs) in walkdir(a) for f in fs if endswith(f, ".sjb") || endswith(f, ".symjit")]) :
                   push!(files, a)
    end
    (checked, failed, rejected, lowered) = (0, 0, 0, 0)
    for f in files
        a = try
            read_sjb(f)
        catch e
            e isa SjbError || rethrow()
            continue  # not readable (test_sjb.jl checks those)
        end
        is_complex(a.config) || continue
        for stable in (nothing, false, true)
            (ok, rej) = compare(dump, f, stable)
            checked += 1
            failed += !ok
            rejected += ok && rej
            lowered += ok && !rej && stable === nothing ? length(read_sjb(f; lowered = true).mir.code) : 0
        end
    end
    println("complex files: $(checked ÷ 3) × 3 stable_math settings, $failed differ ($rejected rejected by both); ",
            "$lowered lowered instructions (saved setting)")
    complex_files = filter(f -> try is_complex(read_sjb(f).config) catch; false end, files)
    ok = failed == 0
    ok &= random_programs(dump, n)
    ok &= check_values(dump, complex_files, n)
    return ok ? 0 : 1
end

length(ARGS) >= 1 || (println("usage: julia test_complexify.jl SJBDUMP [N] [FILE_OR_DIR...]"); exit(2))
n = length(ARGS) >= 2 && all(isdigit, ARGS[2]) ? parse(Int, ARGS[2]) : 1000
exit(main(ARGS[1], n, ARGS[(length(ARGS) >= 2 && all(isdigit, ARGS[2]) ? 3 : 2):end]))
