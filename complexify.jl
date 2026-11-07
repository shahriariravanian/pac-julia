# The lowering of a complex model's MIR to real MIR: symjit's Complexifier (rust/complexify.rs),
# driven as symjit drives it (`Mir::rerun` in rust/mir.rs replays each saved instruction into the
# Complexifier, which emits real instructions). The same instructions in the same order, so the
# result is the MIR symjit's bytecode and kernels run (`SavedModel::read` of symjit-utils).
#
# Registers: a complex register is a pair of real ones: Gen(r) → (Gen(4 + 2r), Gen(5 + 2r)),
# Ret → (Ret, Temp), Left → (Left, Temp), Temp and Right → (Gen(0), Gen(1)); Gen(2), Gen(3) are
# scratch (T0, T1). A complex location holds re at index i, im at i + 1 (LoadComplex/SaveComplex).
# The lowering tracks which registers hold a real number (an imaginary part known to be zero) and
# skips the imaginary halves of their operations; the saved file's `reals` seed the real locations.
#
# Used by sjb.jl (`read_sjb(path; lowered = true)`); include sjb.jl, not this file.

mutable struct Complexifier
    code::Vector{Instruction}
    consts::Vector{Float64}
    real_locs::Set{Loc}
    real_regs::Vector{Bool}  # by ϕ(reg)
    root_defined::Bool
    stable_math::Bool
end

struct ComplexifyError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ComplexifyError) = print(io, "complexify: ", e.msg)

const RET = Reg(REG_RET, 0)
const TEMP = Reg(REG_TEMP, 0)
const LEFT = Reg(REG_LEFT, 0)
const RIGHT = Reg(REG_RIGHT, 0)
gen(r) = Reg(REG_GEN, r)
const T0 = gen(2)
const T1 = gen(3)

no_static(r::Reg) = r.kind == REG_STATIC && throw(ComplexifyError("a static register $r in the MIR"))

function re_(r::Reg)::Reg
    no_static(r)
    r.kind == REG_RET ? RET : r.kind == REG_TEMP ? gen(0) : r.kind == REG_LEFT ? LEFT :
    r.kind == REG_RIGHT ? gen(0) : gen(4 + 2r.index)
end

function im_(r::Reg)::Reg
    no_static(r)
    r.kind == REG_RET ? TEMP : r.kind == REG_TEMP ? gen(1) : r.kind == REG_LEFT ? TEMP :
    r.kind == REG_RIGHT ? gen(1) : gen(4 + 2r.index + 1)
end

# the slot of a register in real_regs (1-based; symjit's ϕ + 1): 32 slots, Gen(0)..Gen(29)
function ϕ(r::Reg)::Int
    no_static(r)
    k = r.kind == REG_RET ? 0 : r.kind == REG_TEMP ? 1 : r.kind == REG_LEFT ? 0 : r.kind == REG_RIGHT ? 1 : r.index + 2
    k < 32 || throw(ComplexifyError("register $r out of range"))
    return k + 1
end

assert(cond, what) = cond || throw(ComplexifyError("assertion failed: $what"))

# ---- the emitted (real) MIR: symjit's Mir methods ---------------------------------------------------

emit!(c::Complexifier, i::Instruction) = push!(c.code, i)
bi!(c, op, d, a, b) = emit!(c, Bi(op, d, a, b))
uni!(c, op, d, a) = emit!(c, Uni(op, d, a))
mov!(c, d, a) = emit!(c, Mov(d, a))
fma!(c, d, a, b, e) = emit!(c, Fused(:MulAdd, d, a, b, e))
cnst!(c, d, idx) = emit!(c, LoadConst(d, idx))

# ---- the state of the registers ----------------------------------------------------------------------

is_real(c::Complexifier, r::Reg) = c.real_regs[ϕ(r)]
is_real(c::Complexifier, l::Loc) = l in c.real_locs
set_real!(c, r::Reg) = (c.real_regs[ϕ(r)] = true)
set_complex!(c, r::Reg) = (c.real_regs[ϕ(r)] = false)
copy_real!(c, d, s) = (c.real_regs[ϕ(d)] = is_real(c, s))
promote_real!(c, d, s1, s2) = (c.real_regs[ϕ(d)] = is_real(c, s1) && is_real(c, s2))

function ensure_complex!(c::Complexifier, r::Reg)
    is_real(c, r) && bi!(c, :Xor, im_(r), im_(r), im_(r))
    set_complex!(c, r)
end

# :RR, :RC, :CR, :CC: whether s1, s2 are real (R) or complex (C)
types(c, s1, s2) = Symbol(is_real(c, s1) ? "R" : "C", is_real(c, s2) ? "R" : "C")

# ---- stable_math ------------------------------------------------------------------------------------

# the index of constant `val` (by its bits), appended if it is not there yet
function stable_const!(c::Complexifier, val::Float64)::Int
    k = findfirst(x -> reinterpret(UInt64, x) == reinterpret(UInt64, val), c.consts)
    k !== nothing && return k - 1
    push!(c.consts, val)
    return length(c.consts) - 1
end

const EXP_MASK_FINITE = reinterpret(Float64, 0x7fe0_0000_0000_0000)  # 2^1023
const FLOOR_FINITE = reinterpret(Float64, 0x0020_0000_0000_0000)     # 2^-1021

# T0 := (|x|/2 + |y|/2 + floor) & mask; T1 is the mask afterwards
function stable_exponent!(c, x, y, mask, floor)
    uni!(c, :Abs, T0, x)
    uni!(c, :Half, T0, T0)
    uni!(c, :Abs, T1, y)
    uni!(c, :Half, T1, T1)
    bi!(c, :Plus, T0, T0, T1)
    cnst!(c, T1, stable_const!(c, floor))
    bi!(c, :Plus, T0, T0, T1)
    cnst!(c, T1, stable_const!(c, mask))
    bi!(c, :And, T0, T0, T1)
end

function stable_divisor_scale!(c, x, y)
    stable_exponent!(c, x, y, Inf, floatmin(Float64))
    bi!(c, :Xor, T0, T0, T1)
    cnst!(c, T1, stable_const!(c, 0.25))
    bi!(c, :Times, T0, T0, T1)
end

function stable_divide!(c, dst, s1, s2)
    (a, b, cc, d) = (re_(s1), im_(s1), re_(s2), im_(s2))
    (a_, b_, c_, d_) = dst.kind != REG_GEN ? (re_(dst), im_(dst), gen(0), gen(1)) : (gen(0), gen(1), re_(dst), im_(dst))
    assert(s1 != RET && s1 != TEMP, "s1 != Ret && s1 != Temp")
    assert(s2 != RET && s2 != TEMP, "s2 != Ret && s2 != Temp")
    stable_divisor_scale!(c, cc, d)
    if is_real(c, s1)
        bi!(c, :Times, a_, a, T0)
        bi!(c, :Times, c_, cc, T0)
        bi!(c, :Times, d_, d, T0)
        bi!(c, :Times, T1, c_, c_)
        bi!(c, :Times, b_, a_, d_)
        bi!(c, :Times, a_, a_, c_)
        fma!(c, T1, d_, d_, T1)
        uni!(c, :Neg, b_, b_)
        bi!(c, :Divide, im_(dst), b_, T1)
        bi!(c, :Divide, re_(dst), a_, T1)
    else
        bi!(c, :Times, a_, a, T0)
        bi!(c, :Times, b_, b, T0)
        bi!(c, :Times, c_, cc, T0)
        bi!(c, :Times, d_, d, T0)
        emit!(c, ComplexBi(:Divide, re_(dst), im_(dst), a_, b_, c_, d_))
    end
    set_complex!(c, dst)
end

function stable_recip!(c, dst, s1)
    (cc, d) = (re_(s1), im_(s1))
    (c_, d_) = (gen(0), gen(1))
    stable_divisor_scale!(c, cc, d)
    bi!(c, :Times, c_, cc, T0)
    bi!(c, :Times, d_, d, T0)
    bi!(c, :Times, T1, c_, c_)
    fma!(c, T1, d_, d_, T1)
    bi!(c, :Divide, c_, c_, T1)
    bi!(c, :Divide, d_, d_, T1)
    uni!(c, :Neg, d_, d_)
    bi!(c, :Times, re_(dst), c_, T0)
    bi!(c, :Times, im_(dst), d_, T0)
    set_complex!(c, dst)
end

function stable_abs!(c, dst, s1)
    (a, b) = (re_(s1), im_(s1))
    (b_, a_, k) = (T1, RET, T1)
    assert(s1 != RET, "s1 != Ret")
    stable_exponent!(c, a, b, EXP_MASK_FINITE, FLOOR_FINITE)
    cnst!(c, k, stable_const!(c, Inf))
    bi!(c, :Xor, k, T0, k)
    uni!(c, :Half, k, k)
    bi!(c, :Times, a_, a, k)
    bi!(c, :Times, b_, b, k)
    bi!(c, :Times, a_, a_, a_)
    bi!(c, :Times, b_, b_, b_)
    bi!(c, :Plus, a_, a_, b_)
    uni!(c, :Root, a_, a_)
    bi!(c, :Times, re_(dst), a_, T0)
    bi!(c, :Xor, im_(dst), im_(dst), im_(dst))
    set_real!(c, dst)
end

function stable_root_magnitude!(c)
    (x, y) = (re_(TEMP), im_(TEMP))
    u = RET
    stable_exponent!(c, x, y, EXP_MASK_FINITE, FLOOR_FINITE)
    uni!(c, :Root, T0, T0)
    bi!(c, :And, T0, T0, T1)
    bi!(c, :Times, T1, T0, T0)
    cnst!(c, u, stable_const!(c, Inf))
    bi!(c, :Xor, T1, T1, u)
    uni!(c, :Half, T1, T1)
    bi!(c, :Times, u, x, T1)
    bi!(c, :Times, T1, y, T1)
    bi!(c, :Times, x, u, u)
    fma!(c, x, T1, T1, x)
    uni!(c, :Root, x, x)
    uni!(c, :Abs, T1, u)
    bi!(c, :Plus, x, x, T1)
    uni!(c, :Half, x, x)
    uni!(c, :Root, x, x)
    bi!(c, :Times, T0, x, T0)
end

# ---- the generator methods (the Complexifier's `impl Generator`) ------------------------------------

function c_fmov!(c, d, s)
    mov!(c, re_(d), re_(s))
    is_real(c, s) || mov!(c, im_(d), im_(s))
    copy_real!(c, d, s)
end

function c_load_const!(c, d, idx)
    cnst!(c, re_(d), idx)
    set_real!(c, d)
end

function c_load!(c, d, l::Loc)
    if l.kind != LOC_MEM && is_real(c, l)  # Mem is always complex
        emit!(c, Load(re_(d), l))
        set_real!(c, d)
    else
        emit!(c, LoadComplex(re_(d), im_(d), l))
        set_complex!(c, d)
    end
end

function c_save!(c, s, l::Loc)
    l.kind == LOC_PARAM && throw(ComplexifyError("a save to a parameter"))
    ensure_complex!(c, s)
    emit!(c, SaveComplex(re_(s), im_(s), l))
    l.kind == LOC_STACK && delete!(c.real_locs, l)
end

# an operation on both halves, the imaginary one skipped for a real operand (neg, sign, round, ...)
function c_elementwise!(c, op, d, s)
    uni!(c, op, re_(d), re_(s))
    is_real(c, s) || uni!(c, op, im_(d), im_(s))
    copy_real!(c, d, s)
end

function c_abs!(c, d, s)
    if is_real(c, s)
        uni!(c, :Abs, re_(d), re_(s))
        set_real!(c, d)
    elseif c.stable_math
        stable_abs!(c, d, s)
        set_complex!(c, d)
    else
        bi!(c, :Times, T0, re_(s), re_(s))
        bi!(c, :Times, T1, im_(s), im_(s))
        bi!(c, :Plus, re_(d), T0, T1)
        uni!(c, :Root, re_(d), re_(d))
        bi!(c, :Xor, im_(d), im_(d), im_(d))
        set_complex!(c, d)
    end
end

function c_abs2!(c, d, s)
    if is_real(c, s)
        bi!(c, :Times, re_(d), re_(s), re_(s))
        set_real!(c, d)
    else
        bi!(c, :Times, T0, re_(s), re_(s))
        bi!(c, :Times, T1, im_(s), im_(s))
        bi!(c, :Plus, re_(d), T0, T1)
        bi!(c, :Xor, im_(d), im_(d), im_(d))
        set_complex!(c, d)
    end
end

function c_times_i!(c, d, s)
    if is_real(c, s)
        mov!(c, im_(d), re_(s))
        bi!(c, :Xor, re_(d), re_(d), re_(d))
    else
        uni!(c, :Neg, T1, im_(s))
        mov!(c, im_(d), re_(s))
        mov!(c, re_(d), T1)
    end
    set_complex!(c, d)
end

function c_times_neg_i!(c, d, s)
    if is_real(c, s)
        uni!(c, :Neg, im_(d), re_(s))
        bi!(c, :Xor, re_(d), re_(d), re_(d))
    else
        mov!(c, T1, im_(s))
        uni!(c, :Neg, im_(d), re_(s))
        mov!(c, re_(d), T1)
    end
    set_complex!(c, d)
end

c_set_label!(c, label) = emit!(c, Label(label))
c_branch!(c, label) = emit!(c, Branch(label))

function c_call_funclet!(c, label)
    emit!(c, Call(label, 0))
    set_complex!(c, RET)
end

function c_ret!(c)
    ensure_complex!(c, RET)
    c_branch!(c, ".ret")
end

# the square root: a call of the `@complex_root` funclet, defined (inline, jumped over) at its first use
function c_root!(c, d, s)
    ensure_complex!(c, s)
    c_fmov!(c, TEMP, s)
    c_call_funclet!(c, "@complex_root")
    set_complex!(c, RET)
    c_fmov!(c, d, RET)

    if !c.root_defined
        c.root_defined = true
        c_branch!(c, "@jump_over_complex_root")
        c_set_label!(c, "@complex_root")
        s1 = TEMP
        (x, y) = (T0, T1)
        bi!(c, :Xor, x, x, x)
        bi!(c, :LittleThan, x, x, re_(s1))
        emit!(c, Save(x, Loc(LOC_STACK, 2)))
        if c.stable_math
            stable_root_magnitude!(c)
        else
            bi!(c, :Times, x, re_(s1), re_(s1))
            fma!(c, x, im_(s1), im_(s1), x)
            uni!(c, :Root, x, x)
            uni!(c, :Abs, y, re_(s1))
            bi!(c, :Plus, x, x, y)
            uni!(c, :Half, x, x)
            uni!(c, :Root, x, x)
        end
        bi!(c, :Divide, y, im_(s1), x)
        uni!(c, :Half, y, y)
        bi!(c, :Equal, re_(s1), y, y)
        bi!(c, :And, y, y, re_(s1))
        emit!(c, IfElse(re_(s1), x, y, Loc(LOC_STACK, 2)))
        emit!(c, IfElse(im_(s1), y, x, Loc(LOC_STACK, 2)))
        uni!(c, :Sign, x, re_(s1))
        bi!(c, :Xor, re_(RET), re_(s1), x)
        bi!(c, :Xor, im_(RET), im_(s1), x)
        c_ret!(c)
        c_set_label!(c, "@jump_over_complex_root")
    end
end

function c_real_root!(c, d, s)
    if is_real(c, s)
        uni!(c, :Root, re_(d), re_(s))
        set_real!(c, d)
    else
        c_root!(c, d, s)
    end
end

function c_recip!(c, d, s)
    if is_real(c, s)
        uni!(c, :Recip, re_(d), re_(s))
        set_real!(c, d)
    elseif c.stable_math
        stable_recip!(c, d, s)
        set_complex!(c, d)
    else
        bi!(c, :Times, T0, re_(s), re_(s))
        fma!(c, T0, im_(s), im_(s), T0)
        bi!(c, :Divide, re_(d), re_(s), T0)
        bi!(c, :Divide, im_(d), im_(s), T0)
        uni!(c, :Neg, im_(d), im_(d))
        set_complex!(c, d)
    end
end

# both halves, whatever the operand (the realness of d is left as it was, as in symjit)
function c_half!(c, d, s)
    uni!(c, :Half, im_(d), im_(s))
    uni!(c, :Half, re_(d), re_(s))
end

function c_plus!(c, d, s1, s2)
    t = types(c, s1, s2)
    if t == :RC
        bi!(c, :Plus, re_(d), re_(s1), re_(s2))
        mov!(c, im_(d), im_(s2))
    elseif t == :CR
        bi!(c, :Plus, re_(d), re_(s1), re_(s2))
        mov!(c, im_(d), im_(s1))
    elseif t == :CC
        emit!(c, ComplexBi(:Plus, re_(d), im_(d), re_(s1), im_(s1), re_(s2), im_(s2)))
    else
        bi!(c, :Plus, re_(d), re_(s1), re_(s2))
    end
    promote_real!(c, d, s1, s2)
end

function c_minus!(c, d, s1, s2)
    t = types(c, s1, s2)
    if t == :RC
        bi!(c, :Minus, re_(d), re_(s1), re_(s2))
        uni!(c, :Neg, im_(d), im_(s2))
    elseif t == :CR
        bi!(c, :Minus, re_(d), re_(s1), re_(s2))
        mov!(c, im_(d), im_(s1))
    elseif t == :CC
        emit!(c, ComplexBi(:Minus, re_(d), im_(d), re_(s1), im_(s1), re_(s2), im_(s2)))
    else
        bi!(c, :Minus, re_(d), re_(s1), re_(s2))
    end
    promote_real!(c, d, s1, s2)
end

function c_times!(c, d, s1, s2)
    t = types(c, s1, s2)
    if t == :RR
        bi!(c, :Times, re_(d), re_(s1), re_(s2))
    elseif t == :RC
        bi!(c, :Times, im_(d), re_(s1), im_(s2))
        bi!(c, :Times, re_(d), re_(s1), re_(s2))
    elseif t == :CR
        bi!(c, :Times, im_(d), im_(s1), re_(s2))
        bi!(c, :Times, re_(d), re_(s1), re_(s2))
    else
        emit!(c, ComplexBi(:Times, re_(d), im_(d), re_(s1), im_(s1), re_(s2), im_(s2)))
    end
    promote_real!(c, d, s1, s2)
end

function c_divide!(c, d, s1, s2)
    tt = re_(TEMP)
    t = types(c, s1, s2)
    if t == :RR
        bi!(c, :Divide, re_(d), re_(s1), re_(s2))
    elseif t == :CR
        bi!(c, :Divide, im_(d), im_(s1), re_(s2))
        bi!(c, :Divide, re_(d), re_(s1), re_(s2))
    elseif c.stable_math
        stable_divide!(c, d, s1, s2)
    elseif t == :RC
        bi!(c, :Times, T0, re_(s2), re_(s2))
        fma!(c, tt, im_(s2), im_(s2), T0)
        bi!(c, :Times, T0, re_(s1), re_(s2))
        bi!(c, :Times, T1, re_(s1), im_(s2))
        uni!(c, :Neg, im_(d), T1)
        bi!(c, :Divide, im_(d), im_(d), tt)
        bi!(c, :Divide, re_(d), T0, tt)
    else
        emit!(c, ComplexBi(:Divide, re_(d), im_(d), re_(s1), im_(s1), re_(s2), im_(s2)))
    end
    promote_real!(c, d, s1, s2)
end

function c_real!(c, d, s)
    mov!(c, re_(d), re_(s))
    bi!(c, :Xor, im_(d), im_(d), im_(d))
    set_real!(c, d)
end

function c_imaginary!(c, d, s)
    is_real(c, s) ? bi!(c, :Xor, re_(d), re_(d), re_(d)) : mov!(c, re_(d), im_(s))
    bi!(c, :Xor, im_(d), im_(d), im_(d))
    set_real!(c, d)
end

function c_conjugate!(c, d, s)
    mov!(c, re_(d), re_(s))
    if !is_real(c, s)
        mov!(c, im_(d), im_(s))
        uni!(c, :Neg, im_(d), im_(d))
    end
    copy_real!(c, d, s)
end

function c_complex!(c, d, s1, s2)
    mov!(c, im_(d), re_(s2))  # the imaginary part first, for d == s2
    mov!(c, re_(d), re_(s1))
    set_complex!(c, d)
end

# >, >=, <, <=: on the real parts
function c_compare!(c, op, d, s1, s2)
    bi!(c, op, re_(d), re_(s1), re_(s2))
    set_real!(c, d)
end

# ==, !=: both parts, combined (and for ==, or for !=)
function c_equality!(c, op, d, s1, s2)
    ensure_complex!(c, s1)
    ensure_complex!(c, s2)
    bi!(c, op, im_(d), im_(s1), im_(s2))
    bi!(c, op, re_(d), re_(s1), re_(s2))
    bi!(c, op == :Equal ? :And : :Or, re_(d), re_(d), im_(d))
    set_real!(c, d)
end

# and, andnot, or, xor: bitwise on both parts (the imaginary one skipped for two real operands)
function c_bitwise!(c, op, d, s1, s2)
    t = types(c, s1, s2)
    t == :RC ? bi!(c, op, im_(d), re_(s1), im_(s2)) :
    t == :CR ? bi!(c, op, im_(d), im_(s1), re_(s2)) :
    t == :CC ? bi!(c, op, im_(d), im_(s1), im_(s2)) : nothing
    bi!(c, op, re_(d), re_(s1), re_(s2))
    promote_real!(c, d, s1, s2)
end

function c_not!(c, d, s)
    uni!(c, :Not, re_(d), re_(s))
    set_real!(c, d)
end

function c_fused!(c, op, d, a, b, e)
    assert(e != RET, "s3 != Ret")
    c_times!(c, RET, a, b)
    if op == :MulAdd
        c_plus!(c, d, RET, e)
    elseif op == :MulSub
        c_minus!(c, d, RET, e)
    elseif op == :NegMulAdd
        c_minus!(c, d, e, RET)
    else
        c_plus!(c, d, RET, e)
        c_elementwise!(c, :Neg, d, d)
    end
end

function c_ifelse!(c, d, t, f, cond::Loc)
    cond.kind == LOC_STACK || throw(ComplexifyError("an IfElse condition not on the stack"))
    ensure_complex!(c, t)
    ensure_complex!(c, f)
    emit!(c, IfElse(re_(d), re_(t), re_(f), cond))
    emit!(c, IfElse(im_(d), im_(t), im_(f), cond))
    set_complex!(c, d)
end

# The math functions a complex model calls (symjit's VirtualTable, `cplx_<name>`): complex ones,
# with 1 or 2 arguments
const COMPLEX_FUNCTIONS = Set(["sin", "sinc", "cos", "tan", "csc", "sec", "cot", "sinh", "cosh", "tanh", "csch",
                               "sech", "coth", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "cbrt",
                               "exp", "ln", "log", "power", "random", "arg", "_yield_"])

function c_call!(c, label, num_args)
    if num_args == 0
        c_call_funclet!(c, label)
    elseif label == "@self"  # a recursive function: real arguments in memory, complex result
        ensure_complex!(c, RET)
        emit!(c, Call("@self", num_args))
    elseif label in COMPLEX_FUNCTIONS
        ensure_complex!(c, RET)
        num_args == 2 ? ensure_complex!(c, TEMP) :
        num_args == 1 || throw(ComplexifyError("complex functions expect 1 or 2 arguments"))
        emit!(c, Call(label, num_args))
    else
        throw(ComplexifyError("the model calls `$label`, which a saved application does not hold"))
    end
end

# ---- the replay (`Mir::rerun`) -----------------------------------------------------------------------

const ELEMENTWISE = (:Neg, :Sign, :Round, :Floor, :Ceiling, :Trunc)

function replay!(c::Complexifier, i::Uni)
    (op, d, s) = (i.op, i.dst, i.s1)
    op in ELEMENTWISE ? c_elementwise!(c, op, d, s) :
    op == :Not ? c_not!(c, d, s) :
    op == :Abs ? c_abs!(c, d, s) :
    op == :Abs2 ? c_abs2!(c, d, s) :
    op == :TimesI ? c_times_i!(c, d, s) :
    op == :TimesNegI ? c_times_neg_i!(c, d, s) :
    op == :Root ? c_root!(c, d, s) :
    op == :RealRoot ? c_real_root!(c, d, s) :
    op == :Recip ? c_recip!(c, d, s) :
    op == :Real ? c_real!(c, d, s) :
    op == :Imaginary ? c_imaginary!(c, d, s) :
    op == :Conjugate ? c_conjugate!(c, d, s) :
    op == :Half ? c_half!(c, d, s) :
    op in (:IsZero, :IsNotZero) ? (c_bitwise!(c, :Xor, TEMP, TEMP, TEMP);
                                   c_equality!(c, op == :IsZero ? :Equal : :NotEqual, d, s, TEMP)) :
    throw(ComplexifyError("unknown unary operation $op"))
end

function replay!(c::Complexifier, i::Bi)
    (op, d, s1, s2) = (i.op, i.dst, i.s1, i.s2)
    op == :Plus ? c_plus!(c, d, s1, s2) :
    op == :Minus ? c_minus!(c, d, s1, s2) :
    op == :Times ? c_times!(c, d, s1, s2) :
    op == :Divide ? c_divide!(c, d, s1, s2) :
    op in (:GreaterThan, :GreaterThanEqual, :LittleThan, :LittleThanEqual) ? c_compare!(c, op, d, s1, s2) :
    op in (:Equal, :NotEqual) ? c_equality!(c, op, d, s1, s2) :
    op in (:And, :AndNot, :Or, :Xor) ? c_bitwise!(c, op, d, s1, s2) :
    op == :Complex ? c_complex!(c, d, s1, s2) :
    throw(ComplexifyError("unknown binary operation $op"))
end

replay!(c::Complexifier, ::Union{Nop,End}) = nothing
replay!(c::Complexifier, i::Mov) = i.dst != i.s1 && c_fmov!(c, i.dst, i.s1)
replay!(c::Complexifier, i::Load) = c_load!(c, i.dst, i.loc)
replay!(c::Complexifier, i::Save) = c_save!(c, i.src, i.loc)
replay!(c::Complexifier, i::LoadConst) = c_load_const!(c, i.dst, i.idx)
replay!(c::Complexifier, i::Union{LoadComplex,SaveComplex}) =
    throw(ComplexifyError("$(nameof(typeof(i))) in the saved MIR of a complex model"))
# real argument lists become complex ones; complex ones (already lowered) are dropped, as in symjit
replay!(c::Complexifier, i::LoadArgs) = i.complex || emit!(c, LoadArgs(i.locs, true, i.ultra))
replay!(c::Complexifier, i::SaveArgs) = i.complex || emit!(c, SaveArgs(i.num_args, true, i.ultra))
replay!(c::Complexifier, i::Call) = c_call!(c, i.label, i.num_args)
replay!(c::Complexifier, i::Fused) = c_fused!(c, i.op, i.dst, i.a, i.b, i.c)
replay!(c::Complexifier, i::IfElse) = c_ifelse!(c, i.dst, i.true_val, i.false_val, i.cond)
replay!(c::Complexifier, i::Label) = c_set_label!(c, i.label)
replay!(c::Complexifier, i::Branch) = i.label == ".ret" ? c_ret!(c) : c_branch!(c, i.label)
replay!(c::Complexifier, i::BranchIf) = emit!(c, BranchIf(re_(i.cond), i.label, i.is_else))

# x op y with y loaded into Temp first (a complex model's LoadMath, LoadConstMath)
function arith!(c, op, d, s1, s2)
    op == :Plus ? c_plus!(c, d, s1, s2) : op == :Minus ? c_minus!(c, d, s1, s2) :
    op == :Times ? c_times!(c, d, s1, s2) : c_divide!(c, d, s1, s2)
end
replay!(c::Complexifier, i::LoadMath) = (c_load!(c, TEMP, i.loc); arith!(c, i.op, i.dst, i.s1, TEMP))
replay!(c::Complexifier, i::LoadConstMath) = (c_load_const!(c, TEMP, i.idx); arith!(c, i.op, i.dst, i.s1, TEMP))

# a saved complex operation on pairs of registers: + and - as two real operations through the
# lowering (symjit's generic_complex_plus/minus); × and ÷ are not lowered by symjit
function replay!(c::Complexifier, i::ComplexBi)
    i.op in (:Times, :Divide) && throw(ComplexifyError("a saved complex $(i.op) (ComplexBi) cannot be lowered"))
    f = i.op == :Plus ? c_plus! : c_minus!
    f(c, i.xd, i.x1, i.x2)
    f(c, i.yd, i.y1, i.y2)
end

# The real MIR of a complex model's `mir` (`reals`: the locations that hold real numbers);
# `stable_math` as the model's configuration has it
function complexify(mir::MIR, reals::Vector{Loc}; stable_math::Bool = false)::MIR
    c = Complexifier(Instruction[], copy(mir.consts), Set(reals), fill(false, 32), false, stable_math)
    for i in mir.code
        replay!(c, i)
    end
    return MIR(c.code, c.consts)
end
