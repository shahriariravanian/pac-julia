# A symjit kernel as C: kernel_c(name, app) writes the MIR of a saved application, lowered as the
# native generators run it (by symjit's Complexifier: `complexified`, ComplexBi kept), as one C function, `static int name(double *mem)`, so the compiled runners need no
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

# the kernel `name` of `app` as C (KERNEL_PRELUDE before it)
function kernel_c(name::String, app::SavedApplication)::String
    app.count_params == 0 || refuse("$name takes $(app.count_params) parameters")
    app.config.opt & CONFIG_OPTIONS.direct_arena != 0 || refuse("$name was not saved for a direct arena")
    mir = app.mir
    complex = is_complex(app.config)
    complex && !is_complexified(mir) && refuse("$name's MIR is complex: read complexified(path)")
    labels = Set(i.label for i in mir.code if i isa Label)
    regs = Set{String}()
    used(r::Reg) = (s = reg_c(r); push!(regs, s); s)
    slots = Set{Int}()
    at(l::Loc, offset::Int = 0) = (l.kind == LOC_STACK && push!(slots, l.index + offset); loc_c(l, offset))
    returns = 0
    has_ret = false
    body = String[]
    line(s) = push!(body, s)
    for i in mir.code
        if i isa Nop || i isa End
        elseif i isa Uni
            f = get(UNI_C, i.op, nothing)
            f === nothing && refuse("$name: the operation $(i.op)")
            line("$(used(i.dst)) = $(f(used(i.s1)));")
        elseif i isa Bi
            f = get(BI_C, i.op, nothing)
            f === nothing && refuse("$name: the operation $(i.op)")
            line("$(used(i.dst)) = $(f(used(i.s1), used(i.s2)));")
        elseif i isa Fused
            line("$(used(i.dst)) = $(FUSED_C[i.op](used(i.a), used(i.b), used(i.c)));")
        elseif i isa Mov
            line("$(used(i.dst)) = $(used(i.s1));")
        elseif i isa Load
            line("$(used(i.dst)) = $(at(i.loc));")
        elseif i isa Save
            line("$(at(i.loc)) = $(used(i.src));")
        elseif i isa LoadConst
            line("$(used(i.dst)) = $(const_c(mir.consts[i.idx+1]));")
        elseif i isa LoadComplex
            line("$(used(i.xd)) = $(at(i.loc)); $(used(i.yd)) = $(at(i.loc, 1));")
        elseif i isa SaveComplex
            line("$(at(i.loc)) = $(used(i.xs)); $(at(i.loc, 1)) = $(used(i.ys));")
        elseif i isa LoadMath || i isa LoadConstMath
            # the operand goes to Temp (complex) or Ret (real) first, as symjit replays it
            t = used(Reg(complex ? REG_TEMP : REG_RET, 0))
            line("$t = $(i isa LoadMath ? at(i.loc) : const_c(mir.consts[i.idx+1])); $(used(i.dst)) = $(BI_C[i.op](used(i.s1), t));")
        elseif i isa ComplexBi
            (xd, yd, x1, y1, x2, y2) = used.((i.xd, i.yd, i.x1, i.y1, i.x2, i.y2))
            (xt, yt, t) = used.((Reg(REG_GEN, 2), Reg(REG_GEN, 3), Reg(REG_TEMP, 0)))
            if i.op == :Plus || i.op == :Minus
                o = i.op == :Plus ? "+" : "-"
                line("$xd = $x1 $o $x2; $yd = $y1 $o $y2;")
            elseif i.op == :Times    # amd/scalar.rs times_complex
                line("$xt = $y1 * $y2; $xt = fma($x1, $x2, -$xt); $yt = $x1 * $y2; $yd = fma($x2, $y1, $yt); $xd = $xt;")
            elseif i.op == :Divide   # amd/scalar.rs divide_complex
                line("$xt = $y1 * $y2; $xt = fma($x1, $x2, $xt); $yt = $x1 * $y2; $yt = fma($x2, $y1, -$yt); " *
                     "$t = $x2 * $x2; $t = fma($y2, $y2, $t); $xd = $xt / $t; $yd = $yt / $t;")
            else
                refuse("$name: the complex operation $(i.op)")
            end
        elseif i isa IfElse
            line("$(used(i.dst)) = $(at(i.cond)) != 0.0 ? $(used(i.true_val)) : $(used(i.false_val));")
        elseif i isa Label
            line("$(label_c(i.label)):;")
        elseif i isa Branch
            if i.label == ".ret"
                line("goto L_return;")
                has_ret = true
            else
                i.label in labels || refuse("$name: no label $(i.label)")
                line("goto $(label_c(i.label));")
            end
        elseif i isa BranchIf
            i.label in labels || refuse("$name: no label $(i.label)")
            line("if ($(i.is_else ? "" : "!")($(used(i.cond)) == 0.0)) goto $(label_c(i.label));")
        elseif i isa Call
            i.num_args == 0 && i.label in labels || refuse("$name: a call of $(i.label) (math functions are not supported)")
            line("ret = $returns; goto $(label_c(i.label)); R_$returns:;")
            returns += 1
        else
            refuse("$name: the instruction $i")
        end
    end
    out = IOBuffer()
    println(out, "/* the symjit kernel $name ($(length(mir.code)) MIR instructions), from its .plane.symjit by kernel_c.jl */")
    println(out, "PAC_FN static int $name(double *mem) {")
    isempty(slots) || println(out, "double ", join(("stack$k = 0.0" for k in sort(collect(slots))), ", "), ";")
    isempty(regs) || println(out, "double ", join(("$r = 0.0" for r in sort(collect(regs); by = r -> (length(r), r))), ", "), ";")
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

kernel_c(name::String, path::AbstractString) = kernel_c(name, read_sjb(complexified(path)))
