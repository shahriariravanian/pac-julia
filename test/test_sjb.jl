# sjb.jl against symjit's own reader: for each .sjb (or .plane.symjit) file, the header and the MIR
# as sjb.jl reads them and as `sjbdump` (test/sjbdump: Application::read_sjb of the symjit crate)
# prints them, line for line.
# usage: julia --project test/test_sjb.jl SJBDUMP FILE_OR_DIR...   (a directory: every .sjb, .symjit file in it)
# SJBDUMP: cargo build --release in test/sjbdump (target/release/sjbdump). Also checks a synthetic
# code buffer of every instruction and the edge cases of the encoding, decoded by sjbdump --code.
using PacRunner
using PacRunner: ARITH_OPS, BIN_OPS, FUSED_OPS, UNI_OPS, Instruction, Nop, End, Uni, Bi, Mov, Load, Save, LoadConst, LoadComplex,
    SaveComplex, LoadArgs, SaveArgs, Fused, IfElse, Label, Branch, BranchIf, Call, LoadMath, LoadConstMath, ComplexBi, Reg, Loc,
    REG_GEN, REG_STATIC, REG_RET, REG_TEMP, REG_LEFT, REG_RIGHT, LOC_MEM, LOC_PARAM, LOC_STACK, SjbError, decode, is_complex,
    mem_size, CONFIG_OPTIONS, ComplexifyError
include(joinpath(@__DIR__, "sjb_encode.jl"))

hex16(x::Float64) = string(reinterpret(UInt64, x), base = 16, pad = 16)
loc_code(l::Loc) = (l.kind == LOC_MEM ? 1 : l.kind == LOC_STACK ? 2 : 3) << 32 | l.index

# the text of sjbdump
function dump_text(a::SavedApplication)::String
    io = IOBuffer()
    c = a.config
    println(io, "config 0x", string(c.opt, base = 16), " ", c.stack_limit, " ", c.num_args)
    println(io, "counts $(a.count_states) $(a.count_params) $(a.count_obs) $(a.count_diffs) $(a.count_loops)")
    println(io, "stack Some($(a.stack_size)) loops $(a.count_loops)")
    foreach(x -> println(io, "const ", hex16(x)), a.consts)
    foreach(f -> println(io, "function ", f), a.functions)
    for (k, ins) in enumerate(a.mir.code)
        text = ins isa Call ? "call $(ins.label) / $(ins.num_args)" : sprint(show, ins)
        println(io, lpad(k - 1, 5, '0'), "\t", text)
    end
    foreach(x -> println(io, "mir const ", hex16(x)), a.mir.consts)
    foreach(l -> println(io, "real ", string(loc_code(l), base = 16)), sort(a.reals, by = loc_code))
    return String(take!(io))
end

# ---- a synthetic code buffer: every instruction and the edge cases of the encoding --------------

function synthetic()::Vector{Instruction}
    g(k) = Reg(REG_GEN, k)
    regs = [Reg(REG_RET, 0), Reg(REG_TEMP, 0), Reg(REG_LEFT, 0), Reg(REG_RIGHT, 0), g(0), g(63), Reg(REG_STATIC, 0),
            Reg(REG_STATIC, 300), Reg(REG_STATIC, 70000), Reg(REG_STATIC, 0xffffffff)]
    locs = [Loc(k, n) for k in (LOC_MEM, LOC_PARAM, LOC_STACK) for n in (0, 255, 256, 65536, 0xffffffff)]
    labels = ["L1", "", "a"^255, "b"^300, "λ", "\$x \\ \"q\" \n\t\0\x01\x7f"]
    code = Instruction[Nop(), End()]
    append!(code, [Uni(op, regs[k], regs[end-k+1]) for (k, op) in enumerate(UNI_OPS) if k <= length(regs)])
    append!(code, [Uni(op, g(1), g(2)) for op in UNI_OPS])
    append!(code, [Bi(op, g(1), regs[mod1(k, end)], g(3)) for (k, op) in enumerate(BIN_OPS)])
    append!(code, [Mov(r, g(5)) for r in regs])
    append!(code, [Load(g(1), l) for l in locs])
    append!(code, [Save(regs[mod1(k, end)], l) for (k, l) in enumerate(locs)])
    append!(code, [LoadConst(g(2), n) for n in (0, 255, 256, 0xffffffff)])
    append!(code, [LoadComplex(g(1), g(2), l) for l in locs])
    append!(code, [SaveComplex(g(1), Reg(REG_STATIC, 9), l) for l in locs])
    append!(code, [Branch(s) for s in labels])
    append!(code, [BranchIf(regs[k], s, k % 2 == 0) for (k, s) in enumerate(labels)])
    append!(code, [Call(s, n) for (s, n) in zip(labels, (0, 1, 127, 128, 32767))])
    append!(code, [Label(s) for s in labels])
    append!(code, [IfElse(g(1), g(2), regs[k], locs[k]) for k in 1:length(regs)])
    append!(code, [LoadArgs(locs[1:n], c, u) for n in (0, 1, 15) for c in (false, true) for u in (false, true)])
    append!(code, [SaveArgs(n, c, u) for n in (0, 1, 63) for c in (false, true) for u in (false, true)])
    append!(code, [Fused(op, g(1), g(2), regs[k], g(4)) for (k, op) in enumerate(FUSED_OPS)])
    append!(code, [LoadMath(op, g(1), g(2), locs[k]) for (k, op) in enumerate(ARITH_OPS)])
    append!(code, [LoadConstMath(op, g(1), regs[k+5], 1000k) for (k, op) in enumerate(ARITH_OPS)])
    append!(code, [ComplexBi(op, g(1), g(2), g(3), g(4), regs[k], Reg(REG_STATIC, 1 << 16)) for (k, op) in enumerate(ARITH_OPS)])
    return code
end

lines(code) = join([lpad(k - 1, 5, '0') * "\t" * (ins isa Call ? "call $(ins.label) / $(ins.num_args)" : sprint(show, ins)) * "\n"
                    for (k, ins) in enumerate(code)])

function check_synthetic(dump)::Int
    code = synthetic()
    bytes = UInt8[]
    foreach(i -> encode(bytes, i), code)
    path = tempname() * ".mir"
    write(path, bytes)
    expected = read(`$dump --code $path`, String)
    rm(path)
    decoded = decode(bytes)
    (want, got, from_symjit) = (lines(code), lines(decoded), expected)
    ok = got == want && got == from_symjit
    println("synthetic code: $(length(code)) instructions, $(length(bytes)) bytes: ",
            ok ? "decoded as symjit decodes them" : "DIFFERS")
    if !ok
        for (k, (a, b, c)) in enumerate(zip(split(want, '\n'), split(got, '\n'), split(from_symjit, '\n')))
            a == b == c || (println("  $k encoded: $a\n  $k sjb.jl:  $b\n  $k symjit:  $c"); break)
        end
    end
    return ok ? 0 : 1
end

function main(dump, args)
    files = String[]
    for a in args
        isdir(a) ? append!(files, [joinpath(r, f) for (r, _, fs) in walkdir(a) for f in fs if endswith(f, ".sjb") || endswith(f, ".symjit")]) :
                   push!(files, a)
    end
    (failed, rejected, kinds) = (0, 0, Dict{String,Int}())
    for f in files
        out = IOBuffer()
        ok = success(pipeline(`$dump $f`, stdout = out, stderr = devnull))
        expected = String(take!(out))
        a = try
            read_sjb(f)
        catch e
            e isa SjbError || rethrow()
            e
        end
        if !ok || a isa SjbError  # a file both readers reject (e.g. a version symjit 2.28 cannot read)
            !ok == (a isa SjbError) ? (rejected += 1) : (failed += 1; println("DIFFERS $f: one reader rejects it: ", a))
            continue
        end
        got = dump_text(a)
        for ins in a.mir.code
            kinds[string(nameof(typeof(ins)))] = get(kinds, string(nameof(typeof(ins))), 0) + 1
        end
        if got != expected
            failed += 1
            (g, e) = (split(got, '\n'), split(expected, '\n'))
            k = findfirst(i -> i > length(g) || i > length(e) || g[i] != e[i], 1:max(length(g), length(e)))
            println("DIFFERS $f, line $k:\n  sjb.jl:  ", get(g, k, "(none)"), "\n  symjit:  ", get(e, k, "(none)"))
        end
    end
    println("$(length(files)) files ($rejected rejected by both), $(failed) differ; instructions: ",
            join(["$k $(kinds[k])" for k in sort(collect(keys(kinds)))], ", "))
    return (failed == 0 ? 0 : 1) + check_synthetic(dump)
end

# as a script (runtests.jl includes this file for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || (println("usage: julia --project test/test_sjb.jl SJBDUMP FILE_OR_DIR..."); exit(2))
    exit(main(ARGS[1], ARGS[2:end]))
end
