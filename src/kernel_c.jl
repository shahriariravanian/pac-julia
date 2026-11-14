# A symjit kernel as C: kernel_c(name, app) writes the MIR of a saved application, lowered as the
# native generators run it (read_sjb's `lowered`, ComplexBi kept), as one C function, `static int name(double *mem)`, so the compiled runners need no
# library built by symjit-utils' sjb2obj (the Julia runner still calls those). The kernel is the
# direct-arena form pac2json saves: its inputs at mem[0 .. states), its outputs (re, im of each
# output pair) at mem[states .. states + observables), no parameters.
#
# The operations are those of the native code (sjb2obj's x86-64 generator), which is the reference
# the C must match bit for bit: Fused is one fma; Sign is the sign bit (x & -0.0); Half is x × 0.5;
# comparisons and IsZero give all ones or zero; And, Or, Xor, AndNot, Not act on the bits. A complex
# × or ÷ (ComplexBi) is the generator's sequence: fused, through Gen(2), Gen(3) (and Temp for ÷);
# + and − are two real operations. LoadMath and LoadConstMath of a complex model go through Temp.
# Registers follow symjit's: Ret and Left are one register (r0), Temp and Right another (r1),
# Gen(k) is gk, and stack slot k the local stackk. A funclet (a Call with no arguments, its label in the code, ".ret" to return) is
# a goto with the return point kept in `ret`, so the C has no calls and no computed goto (nvcc).
# Every function is PAC_FN: under nvcc a device function too.

struct KernelCError <: Exception
    msg::String
end
Base.showerror(io::IO, e::KernelCError) = print(io, "kernel_c: ", e.msg)
refuse(msg) = throw(KernelCError(msg))

# the helpers the kernels use, once per C file before them
const KERNEL_PRELUDE = """
#include <math.h>
#include <stdint.h>
#include <string.h>

/* a function of the host and, under nvcc, of the device (as pacops.h's) */
#ifndef PAC_FN
#ifdef __CUDACC__
#define PAC_FN __host__ __device__
#else
#define PAC_FN
#endif
#endif

#ifdef __CUDACC__
#pragma nv_diag_suppress 177  /* a helper a program's kernels do not use */
#endif

/* the symjit kernels' bit operations on doubles (kernel_c.jl) */
PAC_FN static inline uint64_t sj_bits(double x) { uint64_t u; memcpy(&u, &x, sizeof u); return u; }
PAC_FN static inline double sj_double(uint64_t u) { double x; memcpy(&x, &u, sizeof x); return x; }
PAC_FN static inline double sj_mask(int b) { return sj_double(b ? ~(uint64_t)0 : 0); }
PAC_FN static inline double sj_and(double a, double b) { return sj_double(sj_bits(a) & sj_bits(b)); }
PAC_FN static inline double sj_andnot(double a, double b) { return sj_double(~sj_bits(a) & sj_bits(b)); }
PAC_FN static inline double sj_or(double a, double b) { return sj_double(sj_bits(a) | sj_bits(b)); }
PAC_FN static inline double sj_xor(double a, double b) { return sj_double(sj_bits(a) ^ sj_bits(b)); }
PAC_FN static inline double sj_not(double a) { return sj_double(~sj_bits(a)); }
PAC_FN static inline double sj_sign(double a) { return sj_double(sj_bits(a) & ((uint64_t)1 << 63)); }
"""

function reg_c(r::Reg)::String
    r.kind in (REG_RET, REG_LEFT) && return "r0"
    r.kind in (REG_TEMP, REG_RIGHT) && return "r1"
    r.kind == REG_GEN && return "g$(r.index)"
    refuse("a static register $r")
end

# a location as C: mem[k], or the local stack<k> (symjit's stack slots are fixed indices, the low
# ones mostly unused, so each slot a kernel names is a variable of its own)
function loc_c(l::Loc, offset::Int = 0)::String
    l.kind == LOC_MEM && return "mem[$(l.index + offset)]"
    l.kind == LOC_STACK && return "stack$(l.index + offset)"
    refuse("a parameter location $l (the kernel must take no parameters)")
end

const UNI_C = Dict(
    :Neg => x -> "-$x", :Not => x -> "sj_not($x)", :Abs => x -> "fabs($x)", :Abs2 => x -> "$x * $x",
    :Root => x -> "sqrt($x)", :RealRoot => x -> "sqrt($x)", :Recip => x -> "1.0 / $x",
    :Round => x -> "nearbyint($x)",  # vroundsd mode 0: to nearest, ties to even
    :Floor => x -> "floor($x)", :Ceiling => x -> "ceil($x)", :Trunc => x -> "trunc($x)",
    :Real => x -> x, :Conjugate => x -> x, :Imaginary => _ -> "0.0", :Half => x -> "$x * 0.5",
    :IsZero => x -> "sj_mask($x == 0.0)", :IsNotZero => x -> "sj_mask($x != 0.0)", :Sign => x -> "sj_sign($x)",
)

const BI_C = Dict(
    :Plus => (a, b) -> "$a + $b", :Minus => (a, b) -> "$a - $b", :Times => (a, b) -> "$a * $b",
    :Divide => (a, b) -> "$a / $b",
    :GreaterThan => (a, b) -> "sj_mask($a > $b)", :GreaterThanEqual => (a, b) -> "sj_mask($a >= $b)",
    :LittleThan => (a, b) -> "sj_mask($a < $b)", :LittleThanEqual => (a, b) -> "sj_mask($a <= $b)",
    :Equal => (a, b) -> "sj_mask($a == $b)", :NotEqual => (a, b) -> "sj_mask($a != $b)",
    :And => (a, b) -> "sj_and($a, $b)", :AndNot => (a, b) -> "sj_andnot($a, $b)",
    :Or => (a, b) -> "sj_or($a, $b)", :Xor => (a, b) -> "sj_xor($a, $b)",
)

const FUSED_C = Dict(
    :MulAdd => (a, b, c) -> "fma($a, $b, $c)", :MulSub => (a, b, c) -> "fma($a, $b, -$c)",
    :NegMulAdd => (a, b, c) -> "fma(-$a, $b, $c)", :NegMulSub => (a, b, c) -> "fma(-$a, $b, -$c)",
)

# a label as a C identifier (labels are symjit's strings: "@complex_root", ...)
label_c(s::String) = "L_" * replace(s, r"[^A-Za-z0-9_]" => c -> "_$(string(UInt32(c[1]); base = 16))_")

# a double as a C literal that reads back exactly
const_c(x::Float64) = isnan(x) ? "sj_double(0x$(string(reinterpret(UInt64, x); base = 16))ull)" :
                      isinf(x) ? (x > 0 ? "HUGE_VAL" : "-HUGE_VAL") : repr(x)

# What kernel_c collects while it writes a kernel: the C lines of the body, the registers and stack
# slots they use (declared at the top), and the funclet calls (each a return point).
mutable struct KernelWriter
    name::String
    mir::MIR
    complex::Bool                # a complex model: the operand of a LoadMath goes to Temp, else to Ret
    labels::Set{String}
    regs::Set{String}
    slots::Set{Int}
    body::Vector{String}
    returns::Int                 # funclet calls so far: the return points R_0, R_1, ...
    has_ret::Bool                # a funclet returns (a branch to ".ret")
end

KernelWriter(name::String, app::SavedApplication) =
    KernelWriter(name, app.mir, is_complex(app.config), Set(i.label for i in app.mir.code if i isa Label), Set{String}(),
                 Set{Int}(), String[], 0, false)

# a kernel compile can write: no parameters, saved for a direct arena
function check_kernel(name::String, app::SavedApplication)
    app.count_params == 0 || refuse("$name takes $(app.count_params) parameters")
    app.config.opt & CONFIG_OPTIONS.direct_arena != 0 || refuse("$name was not saved for a direct arena")
end

line!(g::KernelWriter, s::String) = push!(g.body, s)
used!(g::KernelWriter, r::Reg)::String = (s = reg_c(r); push!(g.regs, s); s)
at!(g::KernelWriter, l::Loc, offset::Int = 0)::String = (l.kind == LOC_STACK && push!(g.slots, l.index + offset); loc_c(l, offset))
known_label(labels::Set{String}, name::String, label::String) = label in labels || refuse("$name: no label $label")
funclet(labels::Set{String}, name::String, i::Call) =
    i.num_args == 0 && i.label in labels || refuse("$name: a call of $(i.label) (math functions are not supported)")

# ---- one MIR instruction as C, by its type

emit_c!(g::KernelWriter, i::Union{Nop,End}) = nothing
emit_c!(g::KernelWriter, i::Instruction) = refuse("$(g.name): the instruction $i")

function emit_c!(g::KernelWriter, i::Uni)
    f = get(UNI_C, i.op, nothing)
    f === nothing && refuse("$(g.name): the operation $(i.op)")
    line!(g, "$(used!(g, i.dst)) = $(f(used!(g, i.s1)));")
end

function emit_c!(g::KernelWriter, i::Bi)
    f = get(BI_C, i.op, nothing)
    f === nothing && refuse("$(g.name): the operation $(i.op)")
    line!(g, "$(used!(g, i.dst)) = $(f(used!(g, i.s1), used!(g, i.s2)));")
end

emit_c!(g::KernelWriter, i::Fused) = line!(g, "$(used!(g, i.dst)) = $(FUSED_C[i.op](used!(g, i.a), used!(g, i.b), used!(g, i.c)));")
emit_c!(g::KernelWriter, i::Mov) = line!(g, "$(used!(g, i.dst)) = $(used!(g, i.s1));")
emit_c!(g::KernelWriter, i::Load) = line!(g, "$(used!(g, i.dst)) = $(at!(g, i.loc));")
emit_c!(g::KernelWriter, i::Save) = line!(g, "$(at!(g, i.loc)) = $(used!(g, i.src));")
emit_c!(g::KernelWriter, i::LoadConst) = line!(g, "$(used!(g, i.dst)) = $(const_c(g.mir.consts[i.idx+1]));")
emit_c!(g::KernelWriter, i::LoadComplex) = line!(g, "$(used!(g, i.xd)) = $(at!(g, i.loc)); $(used!(g, i.yd)) = $(at!(g, i.loc, 1));")
emit_c!(g::KernelWriter, i::SaveComplex) = line!(g, "$(at!(g, i.loc)) = $(used!(g, i.xs)); $(at!(g, i.loc, 1)) = $(used!(g, i.ys));")

# the operand goes to Temp (complex) or Ret (real) first, as symjit replays it
function emit_c!(g::KernelWriter, i::Union{LoadMath,LoadConstMath})
    t = used!(g, Reg(g.complex ? REG_TEMP : REG_RET, 0))
    operand = i isa LoadMath ? at!(g, i.loc) : const_c(g.mir.consts[i.idx+1])
    line!(g, "$t = $operand; $(used!(g, i.dst)) = $(BI_C[i.op](used!(g, i.s1), t));")
end

# +, -, ×, ÷ of register pairs, as symjit's x86-64 generator (amd/scalar.rs times_complex, divide_complex)
function emit_c!(g::KernelWriter, i::ComplexBi)
    (xd, yd, x1, y1, x2, y2) = (used!(g, r) for r in (i.xd, i.yd, i.x1, i.y1, i.x2, i.y2))
    (xt, yt, t) = (used!(g, r) for r in (Reg(REG_GEN, 2), Reg(REG_GEN, 3), Reg(REG_TEMP, 0)))
    if i.op == :Plus || i.op == :Minus
        o = i.op == :Plus ? "+" : "-"
        line!(g, "$xd = $x1 $o $x2; $yd = $y1 $o $y2;")
    elseif i.op == :Times
        line!(g, "$xt = $y1 * $y2; $xt = fma($x1, $x2, -$xt); $yt = $x1 * $y2; $yd = fma($x2, $y1, $yt); $xd = $xt;")
    elseif i.op == :Divide
        line!(g, "$xt = $y1 * $y2; $xt = fma($x1, $x2, $xt); $yt = $x1 * $y2; $yt = fma($x2, $y1, -$yt); " *
                 "$t = $x2 * $x2; $t = fma($y2, $y2, $t); $xd = $xt / $t; $yd = $yt / $t;")
    else
        refuse("$(g.name): the complex operation $(i.op)")
    end
end

emit_c!(g::KernelWriter, i::IfElse) =
    line!(g, "$(used!(g, i.dst)) = $(at!(g, i.cond)) != 0.0 ? $(used!(g, i.true_val)) : $(used!(g, i.false_val));")
emit_c!(g::KernelWriter, i::Label) = line!(g, "$(label_c(i.label)):;")

function emit_c!(g::KernelWriter, i::Branch)
    if i.label == ".ret"
        line!(g, "goto L_return;")
        g.has_ret = true
    else
        known_label(g.labels, g.name, i.label)
        line!(g, "goto $(label_c(i.label));")
    end
end

function emit_c!(g::KernelWriter, i::BranchIf)
    known_label(g.labels, g.name, i.label)
    line!(g, "if ($(i.is_else ? "" : "!")($(used!(g, i.cond)) == 0.0)) goto $(label_c(i.label));")
end

# a funclet call: its return point is the next R_k
function emit_c!(g::KernelWriter, i::Call)
    funclet(g.labels, g.name, i)
    line!(g, "ret = $(g.returns); goto $(label_c(i.label)); R_$(g.returns):;")
    g.returns += 1
end

# the names of a declaration, shortest first
declared(names) = sort(collect(names); by = r -> (length(r), r))

# The C function of a kernel: `comment`, the declarations, the body, and the funclets' return
# switch (`returns` return points; `has_ret`: a funclet returns).
function kernel_function(name::String, comment::String, declarations::Vector{String}, body::Vector{String}, returns::Int,
                         has_ret::Bool)::String
    out = IOBuffer()
    println(out, comment)
    println(out, "PAC_FN static int $name(double *mem) {")
    foreach(d -> println(out, d), declarations)
    has_ret && println(out, "int ret = -1;")
    foreach(l -> println(out, l), body)
    println(out, "return 0;")
    if has_ret
        println(out, "L_return:")
        println(out, "switch (ret) {")
        foreach(k -> println(out, "case $k: goto R_$k;"), 0:returns-1)
        println(out, "}")
        println(out, "return -1;  /* a return outside a funclet */")
    end
    println(out, "}")
    return String(take!(out))
end

# the kernel `name` of `app` as C (KERNEL_PRELUDE before it)
function kernel_c(name::String, app::SavedApplication)::String
    check_kernel(name, app)
    g = KernelWriter(name, app)
    foreach(i -> emit_c!(g, i), g.mir.code)
    declarations = String[]
    isempty(g.slots) || push!(declarations, "double " * join(("stack$k = 0.0" for k in sort(collect(g.slots))), ", ") * ";")
    isempty(g.regs) || push!(declarations, "double " * join(("$r = 0.0" for r in declared(g.regs)), ", ") * ";")
    comment = "/* the symjit kernel $name ($(length(g.mir.code)) MIR instructions), from its .plane.symjit by kernel_c.jl */"
    return kernel_function(name, comment, declarations, g.body, g.returns, g.has_ret)
end

kernel_c(name::String, path::AbstractString) = kernel_c(name, read_sjb(path; lowered = true))

# ---- the mixed-precision kernel (PAC_MIXED, pacops.h) ----------------------------------------
#
# kernel_c_mixed(name, app): the same kernel with each value an f32 or an f64, as sjb2ptx's
# `--mixed need` (symjit-utils src/ptx/mixed.rs) that pacuda's mixed kernels used: the divisions and
# reciprocals are f64, and so is every value a divisor or a reciprocal's operand depends on (the
# backward slices of the denominators, e.g. p² − m² + i m Γ); the bit operations (comparisons,
# masks, sign) stay f64 on their exact operands; every other value is f32. mem stays double: an
# input is rounded where an f32 value loads it, an output widened where it is saved.
#
# The kernel is first a list of steps (one value each: a C expression of its operands, rendered in
# the step's type); a register or stack slot `x` has a double `x` and a float `x_f`, and the
# current type of each name is followed through the straight-line code. The analysis does not
# follow control flow: before every label, jump, funclet call and return, the f32 values are
# widened (exactly) into their doubles, and after them every value is f64.

struct MixedStep
    dst::String                       # the name it defines ("" for none: a store to mem, control)
    args::Vector{String}              # the names it reads
    kind::Symbol                      # :arith (f32 unless needed), :f64 (always), :move (its source's type unless needed),
                                      # :select (f64 when a branch is; its condition a mask, f64), :load (from mem),
                                      # :save (to mem), :ctrl (control: raw C)
    render::Function                  # (operand expressions, type) -> the C expression (or, :ctrl, the C)
    seeds::Vector{String}             # the operands it needs in f64 (a divisor)
end

mfn(t, f) = t == :f32 ? f * "f" : f            # sqrt → sqrtf
mlit(t, x::Float64) = t == :f32 ? "((float)$(const_c(x)))" : const_c(x)

# the unary operations that follow their value's type, and the fused multiply-adds: (operands, type) -> C
const MIXED_UNI = Dict{Symbol,Function}(
    :Neg => (a, t) -> "-$(a[1])", :Abs => (a, t) -> "$(mfn(t, "fabs"))($(a[1]))", :Abs2 => (a, t) -> "$(a[1]) * $(a[1])",
    :Root => (a, t) -> "$(mfn(t, "sqrt"))($(a[1]))", :RealRoot => (a, t) -> "$(mfn(t, "sqrt"))($(a[1]))",
    :Round => (a, t) -> "$(mfn(t, "nearbyint"))($(a[1]))", :Floor => (a, t) -> "$(mfn(t, "floor"))($(a[1]))",
    :Ceiling => (a, t) -> "$(mfn(t, "ceil"))($(a[1]))", :Trunc => (a, t) -> "$(mfn(t, "trunc"))($(a[1]))",
    :Half => (a, t) -> "$(a[1]) * $(mlit(t, 0.5))")
const MIXED_FUSED = Dict{Symbol,Function}(
    :MulAdd => (a, t) -> "$(mfn(t, "fma"))($(a[1]), $(a[2]), $(a[3]))",
    :MulSub => (a, t) -> "$(mfn(t, "fma"))($(a[1]), $(a[2]), -$(a[3]))",
    :NegMulAdd => (a, t) -> "$(mfn(t, "fma"))(-$(a[1]), $(a[2]), $(a[3]))",
    :NegMulSub => (a, t) -> "$(mfn(t, "fma"))(-$(a[1]), $(a[2]), -$(a[3]))")
const MIXED_COPY = (a, _) -> a[1]

# what mixed_steps collects: the steps, and the funclets as KernelWriter does
mutable struct MixedWriter
    name::String
    mir::MIR
    complex::Bool
    labels::Set{String}
    steps::Vector{MixedStep}
    returns::Int
    has_ret::Bool
end

step!(m::MixedWriter, dst, args, kind::Symbol, render; seeds = String[]) =
    push!(m.steps, MixedStep(dst, collect(String, args), kind, render, collect(String, seeds)))
arith!(m::MixedWriter, dst, args, render) = step!(m, dst, args, :arith, render)
move!(m::MixedWriter, dst, src) = step!(m, dst, [src], :move, MIXED_COPY)
ctrl!(m::MixedWriter, c::String) = step!(m, "", String[], :ctrl, (_, _) -> c)
const_step!(m::MixedWriter, dst, x::Float64) = arith!(m, dst, String[], (_, t) -> mlit(t, x))

# a load from / a save to a location: mem is double (rounded or widened there); a stack slot is a name
function load!(m::MixedWriter, dst, l::Loc, offset = 0)
    n = loc_c(l, offset)
    l.kind == LOC_MEM ? step!(m, dst, String[], :load, (_, t) -> t == :f32 ? "(float)$n" : n) : move!(m, dst, n)
end

function save!(m::MixedWriter, src, l::Loc, offset = 0)
    n = loc_c(l, offset)
    l.kind == LOC_MEM ? step!(m, "", [src], :save, (a, _) -> "$n = $(a[1])") : move!(m, n, src)
end

# a real binary operation: the division f64 with its divisor a seed, +, -, × following their value's
# type, the others (comparisons, masks: BI_C's) f64 on their exact operands
function bi!(m::MixedWriter, op::Symbol, dst, s1, s2)
    if op == :Divide
        step!(m, dst, [s1, s2], :f64, (a, _) -> "$(a[1]) / $(a[2])"; seeds = [s2])
    elseif op in (:Plus, :Minus, :Times)
        o = op == :Plus ? "+" : op == :Minus ? "-" : "*"
        arith!(m, dst, [s1, s2], (a, _) -> "$(a[1]) $o $(a[2])")
    else
        f = get(BI_C, op, nothing)
        f === nothing && refuse("$(m.name): the operation $op")
        step!(m, dst, [s1, s2], :f64, (a, _) -> f(a[1], a[2]))
    end
end

# ---- one MIR instruction as steps, by its type

step_c!(m::MixedWriter, i::Union{Nop,End}) = nothing
step_c!(m::MixedWriter, i::Instruction) = refuse("$(m.name): the instruction $i")

function step_c!(m::MixedWriter, i::Uni)
    d, s = reg_c(i.dst), reg_c(i.s1)
    if i.op == :Recip
        step!(m, d, [s], :f64, (a, _) -> "1.0 / $(a[1])"; seeds = [s])
    elseif i.op in (:Real, :Conjugate)
        move!(m, d, s)
    elseif i.op == :Imaginary
        const_step!(m, d, 0.0)
    elseif haskey(MIXED_UNI, i.op)
        arith!(m, d, [s], MIXED_UNI[i.op])
    else
        f = get(UNI_C, i.op, nothing)
        f === nothing && refuse("$(m.name): the operation $(i.op)")
        step!(m, d, [s], :f64, (a, _) -> f(a[1]))
    end
end

step_c!(m::MixedWriter, i::Bi) = bi!(m, i.op, reg_c(i.dst), reg_c(i.s1), reg_c(i.s2))
step_c!(m::MixedWriter, i::Fused) = arith!(m, reg_c(i.dst), reg_c.([i.a, i.b, i.c]), MIXED_FUSED[i.op])
step_c!(m::MixedWriter, i::Mov) = move!(m, reg_c(i.dst), reg_c(i.s1))
step_c!(m::MixedWriter, i::Load) = load!(m, reg_c(i.dst), i.loc)
step_c!(m::MixedWriter, i::Save) = save!(m, reg_c(i.src), i.loc)
step_c!(m::MixedWriter, i::LoadConst) = const_step!(m, reg_c(i.dst), m.mir.consts[i.idx+1])
step_c!(m::MixedWriter, i::LoadComplex) = (load!(m, reg_c(i.xd), i.loc); load!(m, reg_c(i.yd), i.loc, 1))
step_c!(m::MixedWriter, i::SaveComplex) = (save!(m, reg_c(i.xs), i.loc); save!(m, reg_c(i.ys), i.loc, 1))

function step_c!(m::MixedWriter, i::Union{LoadMath,LoadConstMath})
    t = reg_c(Reg(m.complex ? REG_TEMP : REG_RET, 0))
    i isa LoadMath ? load!(m, t, i.loc) : const_step!(m, t, m.mir.consts[i.idx+1])
    bi!(m, i.op, reg_c(i.dst), reg_c(i.s1), t)
end

# as emit_c!'s ComplexBi, one step per operation
function step_c!(m::MixedWriter, i::ComplexBi)
    (xd, yd, x1, y1, x2, y2) = reg_c.((i.xd, i.yd, i.x1, i.y1, i.x2, i.y2))
    (xt, yt, t) = reg_c.((Reg(REG_GEN, 2), Reg(REG_GEN, 3), Reg(REG_TEMP, 0)))
    if i.op == :Plus || i.op == :Minus
        bi!(m, i.op, xd, x1, x2); bi!(m, i.op, yd, y1, y2)
    elseif i.op == :Times
        bi!(m, :Times, xt, y1, y2); arith!(m, xt, [x1, x2, xt], MIXED_FUSED[:MulSub]); bi!(m, :Times, yt, x1, y2)
        arith!(m, yd, [x2, y1, yt], MIXED_FUSED[:MulAdd]); move!(m, xd, xt)
    elseif i.op == :Divide
        bi!(m, :Times, xt, y1, y2); arith!(m, xt, [x1, x2, xt], MIXED_FUSED[:MulAdd]); bi!(m, :Times, yt, x1, y2)
        arith!(m, yt, [x2, y1, yt], MIXED_FUSED[:MulSub]); bi!(m, :Times, t, x2, x2); arith!(m, t, [y2, y2, t], MIXED_FUSED[:MulAdd])
        bi!(m, :Divide, xd, xt, t); bi!(m, :Divide, yd, yt, t)
    else
        refuse("$(m.name): the complex operation $(i.op)")
    end
end

step_c!(m::MixedWriter, i::IfElse) =
    step!(m, reg_c(i.dst), [reg_c(i.true_val), reg_c(i.false_val), loc_c(i.cond, 0)], :select,
          (a, _) -> "$(a[3]) != 0.0 ? $(a[1]) : $(a[2])")
step_c!(m::MixedWriter, i::Label) = ctrl!(m, "$(label_c(i.label)):;")

function step_c!(m::MixedWriter, i::Branch)
    if i.label == ".ret"
        ctrl!(m, "goto L_return;")
        m.has_ret = true
    else
        known_label(m.labels, m.name, i.label)
        ctrl!(m, "goto $(label_c(i.label));")
    end
end

function step_c!(m::MixedWriter, i::BranchIf)
    known_label(m.labels, m.name, i.label)
    step!(m, "", [reg_c(i.cond)], :ctrl, (a, _) -> "if ($(i.is_else ? "" : "!")($(a[1]) == 0.0)) goto $(label_c(i.label));")
end

function step_c!(m::MixedWriter, i::Call)
    funclet(m.labels, m.name, i)
    ctrl!(m, "ret = $(m.returns); goto $(label_c(i.label)); R_$(m.returns):;")
    m.returns += 1
end

# the steps of the kernel, with its funclet return points and whether a funclet returns
function mixed_steps(name::String, app::SavedApplication)
    m = MixedWriter(name, app.mir, is_complex(app.config), Set(i.label for i in app.mir.code if i isa Label), MixedStep[], 0, false)
    foreach(i -> step_c!(m, i), m.mir.code)
    return (m.steps, m.returns, m.has_ret)
end

# backward: the steps whose value a denominator needs in f64 (the seeds' backward slices)
function needed_steps(steps::Vector{MixedStep})::BitVector
    needed = Set{String}()
    need = falses(length(steps))
    for k in reverse(eachindex(steps))
        s = steps[k]
        if !isempty(s.dst) && s.dst in needed
            need[k] = true
            delete!(needed, s.dst)
            union!(needed, s.args)
        end
        union!(needed, s.seeds)
    end
    return need
end

# What the forward pass keeps: each name's current type (absent: f64, as declared), the names
# declared as double and as float, the C lines, and the counts for the kernel's comment.
struct MixedBody
    cur::Dict{String,Symbol}
    doubles::Set{String}
    floats::Set{String}
    body::Vector{String}
    counts::Dict{Symbol,Int}
end

MixedBody() = MixedBody(Dict{String,Symbol}(), Set{String}(), Set{String}(), String[], Dict(:f64 => 0, :f32 => 0, :conversions => 0))

current_type(b::MixedBody, n::String)::Symbol = get(b.cur, n, :f64)

# the name `n` read as type `t`: its variable of its current type, converted when that differs
function ref!(b::MixedBody, n::String, t::Symbol)::String
    startswith(n, "mem[") && return t == :f32 ? "(float)$n" : n
    have = current_type(b, n)
    have == :f32 ? push!(b.floats, n) : push!(b.doubles, n)
    have == t && return have == :f32 ? "$(n)_f" : n
    return t == :f64 ? "(double)$(n)_f" : "(float)$n"
end

# before control flow: every f32 value widened (exactly) into its double; after it all are f64
function sync!(b::MixedBody)
    for (n, t) in sort(collect(b.cur))
        t == :f32 || continue
        push!(b.body, "$n = (double)$(n)_f;"); push!(b.doubles, n)
        b.counts[:conversions] += 1
    end
    empty!(b.cur)
end

# the type of a step's value: f64 when a denominator needs it
step_type(b::MixedBody, s::MixedStep, needed::Bool)::Symbol =
    needed || s.kind == :f64 ? :f64 :
    s.kind == :move ? current_type(b, s.args[1]) :
    s.kind == :select ? (any(a -> current_type(b, a) == :f64, s.args[1:2]) ? :f64 : :f32) :
    s.kind == :save ? :f64 : :f32

# one step as C: its operands converted where their current type differs from the step's
function emit_step!(b::MixedBody, s::MixedStep, needed::Bool)
    if s.kind == :ctrl
        sync!(b)
        push!(b.body, s.render([ref!(b, a, :f64) for a in s.args], :f64))
        return
    end
    t = step_type(b, s, needed)
    operands = String[]
    for (j, a) in enumerate(s.args)
        want = s.kind == :select && j == 3 ? :f64 : t      # the condition: a mask (f64)
        r = ref!(b, a, want)
        (startswith(r, "(double)") || startswith(r, "(float)")) && (b.counts[:conversions] += 1)
        push!(operands, r)
    end
    s.kind == :load && t == :f32 && (b.counts[:conversions] += 1)
    expr = s.render(operands, t)
    if s.kind == :save
        push!(b.body, expr * ";")
        return
    end
    s.kind in (:move, :select) || (b.counts[t] += 1)
    b.cur[s.dst] = t
    if t == :f32
        push!(b.floats, s.dst); push!(b.body, "$(s.dst)_f = $expr;")
    else
        push!(b.doubles, s.dst); push!(b.body, "$(s.dst) = $expr;")
    end
end

# the kernel `name` of `app` in mixed precision (KERNEL_PRELUDE before it); same prototype as kernel_c's
function kernel_c_mixed(name::String, app::SavedApplication)::String
    check_kernel(name, app)
    (steps, returns, has_ret) = mixed_steps(name, app)
    need = needed_steps(steps)
    b = MixedBody()
    foreach(k -> emit_step!(b, steps[k], need[k]), eachindex(steps))
    declarations = String[]
    isempty(b.doubles) || push!(declarations, "double " * join(("$n = 0.0" for n in declared(b.doubles)), ", ") * ";")
    isempty(b.floats) || push!(declarations, "float " * join(("$(n)_f = 0.0f" for n in declared(b.floats)), ", ") * ";")
    comment = "/* the symjit kernel $name in mixed precision (kernel_c.jl's kernel_c_mixed): " *
              "$(b.counts[:f32]) f32 and $(b.counts[:f64]) f64 operations, $(b.counts[:conversions]) conversions */"
    return kernel_function(name, comment, declarations, b.body, returns, has_ret)
end

kernel_c_mixed(name::String, path::AbstractString) = kernel_c_mixed(name, read_sjb(path; lowered = true))
