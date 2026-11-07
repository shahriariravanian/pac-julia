# The encoder of symjit's MIR (serializer.rs `MirWriter`) and of .sjb files, for the tests only:
# test_sjb.jl (a synthetic code buffer) and test_complexify.jl (random complex programs).

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

# A whole .sjb file (version 4) of a model with `code` as its MIR (symjit's `Application::save`)
function sjb_bytes(code::Vector{<:Instruction}; opt::UInt32, consts::Vector{Float64} = Float64[], reals::Vector{Loc} = Loc[],
                   states = 4, params = 2, obs = 2, stack = 64)::Vector{UInt8}
    out = UInt8[]
    u64!(x) = append!(out, reinterpret(UInt8, [htol(UInt64(x))]))
    f64!(xs) = (u64!(length(xs)); foreach(x -> u64!(reinterpret(UInt64, x)), xs))
    u64!(APPLICATION_MAGIC); u64!(4)
    u64!(PROGRAM_MAGIC)
    u64!(CONFIG_MAGIC); u64!(UInt64(opt) | UInt64(6) << 32); u64!(1 << 20); u64!(64)  # bytecode
    u64!(states); u64!(params); u64!(obs); u64!(0); u64!(0)
    u64!(BUILDER_MAGIC); u64!(0); u64!(stack); f64!(consts); u64!(0)
    u64!(0)  # mask
    u64!(MIR_MAGIC)
    bytes = UInt8[]
    foreach(i -> encode(bytes, i), code)
    u64!(length(bytes)); append!(out, bytes); f64!(consts)
    codes = sort([(l.kind == LOC_MEM ? 1 : l.kind == LOC_STACK ? 2 : 3) << 32 | l.index for l in reals])
    u64!(length(codes)); foreach(u64!, codes)
    return out
end
