/* df64.h: a double emulated by a pair of floats (hi, lo), value hi + lo with |lo| <= ulp(hi)/2, for
 * Apple GPUs (Metal has no double). The algorithms are pacuda's (runner/recurrence_runner.cu):
 * Joldes, Muller, Popescu, "Tight and rigorous error bounds for basic building blocks of
 * double-word arithmetic", ACM TOMS 44(2), 2017: + AccurateDWPlusDW (Alg. 6), x DWTimesDW3
 * (Alg. 12), / DWDivDW2 (Alg. 17) with DWTimesFP3 (Alg. 9). Relative error about 2^-46 or better.
 *
 * They need correctly rounded f32 operations without contraction or reassociation (Metal:
 * MTLMathModeSafe and `#pragma METAL fp contract(off)`, which Safe does not imply; measured on the
 * M4: f32 +, -, x, /, fma and sqrt are the CPU's bits) and a real fma. Apple GPUs flush f32
 * denormals to zero: a low part below 2^-126 is lost (a relative error of 2^-126 / |x|, only for
 * values near the f32 range's bottom), and the range is f32's (~1e-38..3e38).
 *
 * Plain C++ over float: the same source is MSL (with <metal_stdlib>) and C++ on the CPU (with
 * msl_shim.h), so the Linux tests run these very functions. The `df64` struct is not a float2 on
 * purpose: a float2's + would be componentwise. Buffers hold df64 values as float2 (hi, lo). */

#ifndef DF64_H
#define DF64_H

struct df64 {
    float hi, lo;
};

static inline df64 dd(float hi, float lo) { df64 z; z.hi = hi; z.lo = lo; return z; }
static inline df64 dd_f(float x) { return dd(x, 0.0f); }                 /* exact */
static inline float dd_to_f(df64 x) { return x.hi + x.lo; }              /* rounded once */
static inline df64 dd_load(float2 v) { return dd(v.x, v.y); }
static inline float2 dd_store(df64 x) { return float2(x.hi, x.lo); }
/* an integer (|n| < 2^48) exactly */
static inline df64 dd_int(long n) { float h = (float)n; return dd(h, (float)(n - (long)h)); }

/* ---- error-free transformations ---- */
static inline df64 two_sum(float a, float b)
{
    float s = a + b, bb = s - a;
    return dd(s, (a - (s - bb)) + (b - bb));
}
static inline df64 fast_two_sum(float a, float b)    /* |a| >= |b| (or a = 0) */
{
    float s = a + b;
    return dd(s, b - (s - a));
}
static inline df64 two_prod(float a, float b)
{
    float p = a * b;
    return dd(p, fma(a, b, -p));
}

/* ---- arithmetic ---- */
static inline df64 operator-(df64 x) { return dd(-x.hi, -x.lo); }

static inline df64 operator+(df64 x, df64 y)          /* AccurateDWPlusDW */
{
    df64 s = two_sum(x.hi, y.hi), t = two_sum(x.lo, y.lo);
    df64 v = fast_two_sum(s.hi, s.lo + t.hi);
    return fast_two_sum(v.hi, t.lo + v.lo);
}
static inline df64 operator-(df64 x, df64 y) { return x + (-y); }

static inline df64 operator*(df64 x, df64 y)          /* DWTimesDW3 */
{
    float ch = x.hi * y.hi;
    float cl1 = fma(x.hi, y.hi, -ch);
    float tl0 = x.lo * y.lo;
    float tl1 = fma(x.hi, y.lo, tl0);
    float cl2 = fma(x.lo, y.hi, tl1);
    return fast_two_sum(ch, cl1 + cl2);
}

static inline df64 operator/(df64 x, df64 y)          /* DWDivDW2 */
{
    float th = x.hi / y.hi;
    float rh = y.hi * th;                              /* r = y x th (DWTimesFP3) */
    float rl1 = fma(y.hi, th, -rh);
    float rl3 = fma(y.lo, th, rl1);
    df64 r = fast_two_sum(rh, rl3);
    float pi_h = x.hi - r.hi;
    float delta = pi_h + (x.lo - r.lo);
    return fast_two_sum(th, delta / y.hi);
}

static inline thread df64 &operator+=(thread df64 &x, df64 y) { x = x + y; return x; }

/* a × b + c (the kernels' fused multiply-add: two df64 roundings, not one) */
static inline df64 dd_fma(df64 a, df64 b, df64 c) { return a * b + c; }

/* sqrt: Karp-Markstein, one correction of the f32 root; sqrt(0) = 0, sqrt(<0) = NaN */
static inline df64 dd_sqrt(df64 x)
{
    if (!(x.hi > 0.0f)) return dd(sqrt(x.hi), 0.0f);
    float s = sqrt(x.hi);
    df64 r = x - two_prod(s, s);
    return fast_two_sum(s, r.hi / (2.0f * s));
}

/* ---- comparisons (normalized pairs: hi decides, lo breaks ties) ---- */
static inline bool operator==(df64 a, df64 b) { return a.hi == b.hi && a.lo == b.lo; }
static inline bool operator!=(df64 a, df64 b) { return !(a == b); }
static inline bool operator<(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
static inline bool operator<=(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo); }
static inline bool operator>(df64 a, df64 b) { return b < a; }
static inline bool operator>=(df64 a, df64 b) { return b <= a; }
static inline bool dd_isnan(df64 x) { return isnan(x.hi) || isnan(x.lo); }
static inline bool dd_signbit(df64 x) { return (as_type<uint>(x.hi) >> 31) != 0; }
static inline bool dd_is_zero(df64 x) { return x.hi == 0.0f && x.lo == 0.0f; }

static inline df64 dd_abs(df64 x) { return dd_signbit(x) ? -x : x; }
/* sign of s on the magnitude of x (as copysign: on hi's sign bit) */
static inline df64 dd_copysign(df64 x, df64 s) { return dd_signbit(x) != dd_signbit(s) ? -x : x; }

/* Julia's max and min (pacops' jmax, jmin): NaN propagates, max(-0, 0) = 0, min(-0, 0) = -0 */
static inline df64 dd_max(df64 x, df64 y)
{
    if (dd_isnan(x) || dd_isnan(y)) return x + y;
    if (x != y) return x > y ? x : y;
    return dd_signbit(x) ? y : x;
}
static inline df64 dd_min(df64 x, df64 y)
{
    if (dd_isnan(x) || dd_isnan(y)) return x + y;
    if (x != y) return x < y ? x : y;
    return dd_signbit(x) ? x : y;
}

/* ---- roundings to an integer (the kernels' Floor, Ceiling, Trunc, Round) ---- */
static inline df64 dd_floor(df64 x)
{
    float h = floor(x.hi);
    return h != x.hi ? dd(h, 0.0f) : fast_two_sum(h, floor(x.lo));
}
static inline df64 dd_ceil(df64 x)
{
    float h = ceil(x.hi);
    return h != x.hi ? dd(h, 0.0f) : fast_two_sum(h, ceil(x.lo));
}
static inline df64 dd_trunc(df64 x) { return dd_signbit(x) ? dd_ceil(x) : dd_floor(x); }
/* to nearest, ties to even (vroundsd mode 0) */
static inline df64 dd_rint(df64 x)
{
    df64 f = dd_floor(x), d = x - f;
    if (d < dd_f(0.5f)) return f;
    if (d > dd_f(0.5f)) return f + dd_f(1.0f);
    df64 halved = dd_floor(f * dd_f(0.5f));               /* a tie: the even neighbour */
    return halved * dd_f(2.0f) == f ? f : f + dd_f(1.0f);
}

/* ---- the kernels' bit operations (kernel_msl.jl): componentwise on (hi, lo) ----
 * A mask (comparisons, IsZero) has all bits set in both halves, and a sign (Sign) the sign bit in
 * both, so that x And mask selects x, x Xor x is 0 and x Xor Sign(y) is ±x, as in f64. */
static inline df64 dd_bits(uint h, uint l) { return dd(as_type<float>(h), as_type<float>(l)); }
static inline df64 dd_mask(bool b) { uint m = b ? 0xffffffffu : 0u; return dd_bits(m, m); }
static inline df64 dd_sign(df64 x) { uint s = as_type<uint>(x.hi) & 0x80000000u; return dd_bits(s, s); }
static inline df64 dd_and(df64 a, df64 b)
{
    return dd_bits(as_type<uint>(a.hi) & as_type<uint>(b.hi), as_type<uint>(a.lo) & as_type<uint>(b.lo));
}
static inline df64 dd_andnot(df64 a, df64 b)
{
    return dd_bits(~as_type<uint>(a.hi) & as_type<uint>(b.hi), ~as_type<uint>(a.lo) & as_type<uint>(b.lo));
}
static inline df64 dd_or(df64 a, df64 b)
{
    return dd_bits(as_type<uint>(a.hi) | as_type<uint>(b.hi), as_type<uint>(a.lo) | as_type<uint>(b.lo));
}
static inline df64 dd_xor(df64 a, df64 b)
{
    return dd_bits(as_type<uint>(a.hi) ^ as_type<uint>(b.hi), as_type<uint>(a.lo) ^ as_type<uint>(b.lo));
}
static inline df64 dd_not(df64 a) { return dd_bits(~as_type<uint>(a.hi), ~as_type<uint>(a.lo)); }
/* a condition (IfElse, BranchIf): x != 0, as the C kernels' `x != 0.0` (NaN masks are true) */
static inline bool dd_true(df64 x) { return !(x.hi == 0.0f && x.lo == 0.0f); }

/* ---- complex df64 ---- */
struct cdf64 {
    df64 re, im;
};
static inline cdf64 cdd(df64 re, df64 im) { cdf64 z; z.re = re; z.im = im; return z; }
static inline cdf64 cdd_add(cdf64 a, cdf64 b) { return cdd(a.re + b.re, a.im + b.im); }
static inline cdf64 cdd_sub(cdf64 a, cdf64 b) { return cdd(a.re - b.re, a.im - b.im); }
static inline cdf64 cdd_neg(cdf64 a) { return cdd(-a.re, -a.im); }
static inline cdf64 cdd_mul(cdf64 a, cdf64 b) { return cdd(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
static inline cdf64 cdd_mulr(cdf64 a, df64 x) { return cdd(a.re * x, a.im * x); }
static inline cdf64 cdd_divr(cdf64 a, df64 x) { return cdd(a.re / x, a.im / x); }

#endif
