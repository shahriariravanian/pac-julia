# symjit's bytecode interpreter (rust/mir/interpreter.rs `Mir::exec_instruction`) for lowered (real)
# MIR, for the tests: the same operations on an array of registers (Ret and Left share one, Temp
# and Right another), the memory, the stack and the parameters. ComplexBi is num_complex's
# arithmetic (no fma). Calls with arguments and argument lists are not run (an error).
using PacRunner
using PacRunner: ARITH_OPS, BIN_OPS, FUSED_OPS, UNI_OPS, Instruction, Nop, End, Uni, Bi, Mov, Load, Save, LoadConst, LoadComplex,
    SaveComplex, LoadArgs, SaveArgs, Fused, IfElse, Label, Branch, BranchIf, Call, LoadMath, LoadConstMath, ComplexBi, Reg, Loc,
    REG_GEN, REG_STATIC, REG_RET, REG_TEMP, REG_LEFT, REG_RIGHT, LOC_MEM, LOC_PARAM, LOC_STACK, SjbError, decode, is_complex,
    mem_size, CONFIG_OPTIONS, ComplexifyError

mask(b::Bool) = b ? reinterpret(Float64, ~UInt64(0)) : 0.0
bits(x) = reinterpret(UInt64, x)
float(b::UInt64) = reinterpret(Float64, b)

function slot(r::Reg)::Int
    r.kind in (REG_RET, REG_LEFT) && return 1
    r.kind in (REG_TEMP, REG_RIGHT) && return 2
    r.kind == REG_GEN && return r.index + 3
    error("a static register")
end

function fetch(loc::Loc, mem, stack, params, k = 0)
    a = loc.kind == LOC_MEM ? mem : loc.kind == LOC_STACK ? stack : params
    return a[loc.index+k+1]
end

function store!(loc::Loc, x, mem, stack, k = 0)
    loc.kind == LOC_PARAM && error("a save to a parameter")
    (loc.kind == LOC_MEM ? mem : stack)[loc.index+k+1] = x
end

function uniop(op::Symbol, x::Float64)::Float64
    op == :Neg ? -x : op == :Not ? float(~bits(x)) : op == :Abs ? abs(x) : op == :Abs2 ? abs(x)^2 :
    op in (:Root, :RealRoot) ? Base.sqrt_llvm(x) : op == :Recip ? 1.0 / x :
    op == :Round ? round(x, RoundNearestTiesAway) : op == :Floor ? floor(x) : op == :Ceiling ? ceil(x) :
    op == :Trunc ? trunc(x) : op in (:Real, :Conjugate) ? x : op == :Imaginary ? 0.0 : op == :Half ? x / 2.0 :
    op == :IsZero ? mask(x == 0.0) : op == :IsNotZero ? mask(x != 0.0) : op == :Sign ? (x < 0.0 ? -0.0 : 0.0) :
    op in (:TimesI, :TimesNegI) ? 0.0 : error("unary $op")
end

function binop(op::Symbol, x::Float64, y::Float64)::Float64
    op == :Plus ? x + y : op == :Minus ? x - y : op == :Times ? x * y : op == :Divide ? x / y :
    op == :GreaterThan ? mask(x > y) : op == :GreaterThanEqual ? mask(x >= y) :
    op == :LittleThan ? mask(x < y) : op == :LittleThanEqual ? mask(x <= y) :
    op == :Equal ? mask(x == y) : op == :NotEqual ? mask(x != y) :
    op == :And ? float(bits(x) & bits(y)) : op == :AndNot ? float(~bits(x) & bits(y)) :
    op == :Or ? float(bits(x) | bits(y)) : op == :Xor ? float(bits(x) ⊻ bits(y)) : op == :Complex ? x :
    error("binary $op")
end

arith(op, x, y) = op == :Plus ? x + y : op == :Minus ? x - y : op == :Times ? x * y : x / y

# num_complex's z1 op z2
function complex_arith(op, a, b, c, d)
    op == :Plus && return (a + c, b + d)
    op == :Minus && return (a - c, b - d)
    op == :Times && return (a * c - b * d, a * d + b * c)
    n = c * c + d * d
    return ((a * c + b * d) / n, (b * c - a * d) / n)
end

# runs `mir` on mem and stack (changed in place) and params; at most `limit` steps
function run_mir!(mir::MIR, mem::Vector{Float64}, stack::Vector{Float64}, params::Vector{Float64}; limit = 10^6)
    regs = zeros(66)
    get(r) = regs[slot(r)]
    set!(r, x) = (regs[slot(r)] = x)
    code = mir.code
    labels = Dict(i.label => k for (k, i) in enumerate(code) if i isa Label)
    returns = Int[]
    ip = 1
    steps = 0
    while ip <= length(code)
        (steps += 1) > limit && error("too many steps")
        i = code[ip]
        if i isa Uni
            set!(i.dst, uniop(i.op, get(i.s1)))
        elseif i isa Bi
            set!(i.dst, binop(i.op, get(i.s1), get(i.s2)))
        elseif i isa Mov
            set!(i.dst, get(i.s1))
        elseif i isa Load
            set!(i.dst, fetch(i.loc, mem, stack, params))
        elseif i isa Save
            store!(i.loc, get(i.src), mem, stack)
        elseif i isa LoadComplex
            (x, y) = (fetch(i.loc, mem, stack, params), fetch(i.loc, mem, stack, params, 1))
            set!(i.xd, x)
            set!(i.yd, y)
        elseif i isa SaveComplex
            (x, y) = (get(i.xs), get(i.ys))
            store!(i.loc, x, mem, stack)
            store!(i.loc, y, mem, stack, 1)
        elseif i isa LoadConst
            set!(i.dst, mir.consts[i.idx+1])
        elseif i isa Call && i.num_args == 0
            push!(returns, ip)
            ip = labels[i.label]
        elseif i isa Fused
            (a, b, c) = (get(i.a), get(i.b), get(i.c))
            set!(i.dst, i.op == :MulAdd ? a * b + c : i.op == :MulSub ? a * b - c : i.op == :NegMulAdd ? -a * b + c : -a * b - c)
        elseif i isa IfElse
            set!(i.dst, fetch(i.cond, mem, stack, params) == 0 ? get(i.false_val) : get(i.true_val))
        elseif i isa Branch
            ip = i.label == ".ret" ? pop!(returns) : labels[i.label]
        elseif i isa BranchIf
            (get(i.cond) == 0.0) ⊻ i.is_else && (ip = labels[i.label])
        elseif i isa LoadMath
            set!(i.dst, arith(i.op, get(i.s1), fetch(i.loc, mem, stack, params)))
        elseif i isa LoadConstMath
            set!(i.dst, arith(i.op, get(i.s1), mir.consts[i.idx+1]))
        elseif i isa ComplexBi
            (x, y) = complex_arith(i.op, get(i.x1), get(i.y1), get(i.x2), get(i.y2))
            set!(i.xd, x)
            set!(i.yd, y)
        elseif !(i isa Union{Nop,End,Label})
            error("cannot run $(nameof(typeof(i)))")
        end
        ip += 1
    end
end
