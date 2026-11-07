# sjb.jl against symjit's own reader: for each .sjb (or .plane.symjit) file, the header and the MIR
# as sjb.jl reads them and as `sjbdump` (test/sjbdump: Application::read_sjb of the symjit crate)
# prints them, line for line.
# usage: julia test_sjb.jl SJBDUMP FILE_OR_DIR...   (a directory: every .sjb, .symjit file in it)
# SJBDUMP: cargo build --release in test/sjbdump (target/release/sjbdump). Also checks a synthetic
# code buffer of every instruction and the edge cases of the encoding, decoded by sjbdump --code.
include(joinpath(@__DIR__, "..", "sjb.jl"))

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

# the encoder of symjit's MirWriter (serializer.rs), for the test only
function enc_num(out, prefix::UInt8, n::Int)
    m = 1 + ((32 - leading_zeros(UInt32(n))) >> 3)
    push!(out, prefix | UInt8(m))
    for k in 0:m-1
        push!(out, UInt8((n >> (8k)) & 0xff))
    end
end
function enc(out, r::Reg)
    r.kind == REG_GEN ? push!(out, 0x40 | UInt8(r.index)) :
    r.kind == REG_STATIC ? enc_num(out, 0x80, r.index) :
    push!(out, (0xc0, 0xc1, 0xc2, 0xc3)[Int(r.kind)+1])
end
enc(out, l::Loc) = enc_num(out, (0x40, 0x80, 0xc0)[Int(l.kind)+1], l.index)
function enc(out, s::String)
    n = ncodeunits(s)
    n <= 255 ? push!(out, UInt8(n)) : append!(out, [0xff, 0xff], reinterpret(UInt8, [htol(UInt64(n))]))
    append!(out, codeunits(s))
end
enc(out, xs...) = foreach(x -> enc(out, x), xs)
flags(n, complex, ultra) = UInt8(n) | (complex ? 0x80 : 0x00) | (ultra ? 0x40 : 0x00)
opcode(ops, op) = UInt8(findfirst(==(op), ops) - 1)

function encode(out, i::Instruction)
    i isa Nop ? push!(out, 0) :
    i isa End ? push!(out, 1) :
    i isa Uni ? (push!(out, 0x40 | opcode(UNI_OPS, i.op)); enc(out, i.dst, i.s1)) :
    i isa Bi ? (push!(out, 0x80 | opcode(BIN_OPS, i.op)); enc(out, i.dst, i.s1, i.s2)) :
    i isa Mov ? (push!(out, 2); enc(out, i.dst, i.s1)) :
    i isa Load ? (push!(out, 3); enc(out, i.dst, i.loc)) :
    i isa Save ? (push!(out, 4); enc(out, i.src, i.loc)) :
    i isa LoadConst ? (push!(out, 5); enc(out, i.dst); enc_num(out, 0x00, i.idx)) :
    i isa LoadComplex ? (push!(out, 6); enc(out, i.xd, i.yd, i.loc)) :
    i isa SaveComplex ? (push!(out, 7); enc(out, i.xs, i.ys, i.loc)) :
    i isa Branch ? (push!(out, 8); enc(out, i.label)) :
    i isa BranchIf ? (push!(out, i.is_else ? 10 : 9); enc(out, i.cond, i.label)) :
    i isa Call ? (push!(out, 11);
                  i.num_args < 128 ? push!(out, UInt8(i.num_args)) : push!(out, UInt8(i.num_args & 0x7f) | 0x80, UInt8(i.num_args >> 7));
                  enc(out, i.label)) :
    i isa Label ? (push!(out, 12); enc(out, i.label)) :
    i isa IfElse ? (push!(out, 13); enc(out, i.dst, i.true_val, i.false_val, i.cond)) :
    i isa LoadArgs ? (push!(out, 14, flags(length(i.locs), i.complex, i.ultra)); foreach(l -> enc(out, l), i.locs)) :
    i isa SaveArgs ? push!(out, 15, flags(i.num_args, i.complex, i.ultra)) :
    i isa Fused ? (push!(out, 32 + opcode(FUSED_OPS, i.op)); enc(out, i.dst, i.a, i.b, i.c)) :
    i isa LoadMath ? (push!(out, 36 + opcode(ARITH_OPS, i.op)); enc(out, i.dst, i.s1, i.loc)) :
    i isa LoadConstMath ? (push!(out, 40 + opcode(ARITH_OPS, i.op)); enc(out, i.dst, i.s1); enc_num(out, 0x00, i.idx)) :
    i isa ComplexBi ? (push!(out, 44 + opcode(ARITH_OPS, i.op)); enc(out, i.xd, i.yd, i.x1, i.y1, i.x2, i.y2)) :
    error("no encoding of $i")
end

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

length(ARGS) >= 2 || (println("usage: julia test_sjb.jl SJBDUMP FILE_OR_DIR..."); exit(2))
exit(main(ARGS[1], ARGS[2:end]))
