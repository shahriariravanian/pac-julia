/* The wavefunctions (wavefunctions.jl) and the instructions (operations.jl) of a pac2json program,
 * in plain C, for one phase-space point.
 *
 * Layout: a complex number is two doubles (re, im); a current is its components in order, each
 * (re, im): a scalar 2 doubles, a Weyl spinor 4, a Dirac spinor or a vector 8, an antisymmetric
 * tensor 12 (components 01, 02, 03, 12, 13, 23). A momentum form is 4 real doubles (E, px, py, pz).
 * Scales and factors are complex (2 doubles); a mass is a real double. Components count from 0.
 *
 * The output comes first. Every function computes its result before it writes `out`, so `out` may
 * be one of the inputs (an instruction that updates a current in place).
 *
 * Bit-exactness with the Julia runner needs the plain IEEE operations: compile without contraction
 * into fma (gcc/clang: -ffp-contract=off; nvcc: --fmad=false). Under nvcc every function is also a
 * device function (PAC_FN), but kernel_call (a host function pointer). */

#ifndef PACOPS_H
#define PACOPS_H

#include <stddef.h>
#include <math.h>

/* a function of the host and, under nvcc, of the device: the same arithmetic on both */
#ifndef PAC_FN
#ifdef __CUDACC__
#define PAC_FN __host__ __device__
#else
#define PAC_FN
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* main.json's strings as numbers */
enum { MODE_SET = 0, MODE_ADD = 1 };                        /* "=", "+=" */
enum { FAMILY_SCALAR, FAMILY_WEYL, FAMILY_DIRAC, FAMILY_VECTOR };
enum { PARTICLE = 0, ANTIPARTICLE = 1 };                   /* orientation */
enum { VERTEX_WEYL_VECTOR_A, VERTEX_WEYL_VECTOR_B, VERTEX_TENSOR_VECTOR, VERTEX_VECTOR_WEDGE,
       VERTEX_THREE_VECTOR };
enum { PROPAGATOR_WEYL_A, PROPAGATOR_WEYL_B, PROPAGATOR_VECTOR };

/* ---- complex arithmetic, as Julia's Complex{Float64} (no fma) ---- */

typedef struct { double re, im; } cplx;

PAC_FN static inline cplx cx(double re, double im) { cplx z = {re, im}; return z; }
PAC_FN static inline cplx get(const double *p, int j) { return cx(p[2 * j], p[2 * j + 1]); }
PAC_FN static inline void put(double *p, int j, cplx z) { p[2 * j] = z.re; p[2 * j + 1] = z.im; }

PAC_FN static inline cplx add(cplx a, cplx b) { return cx(a.re + b.re, a.im + b.im); }
PAC_FN static inline cplx sub(cplx a, cplx b) { return cx(a.re - b.re, a.im - b.im); }
PAC_FN static inline cplx neg(cplx a) { return cx(-a.re, -a.im); }
PAC_FN static inline cplx mul(cplx a, cplx b) { return cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
PAC_FN static inline cplx mulr(cplx a, double x) { return cx(a.re * x, a.im * x); }  /* complex × real */
PAC_FN static inline cplx divr(cplx a, double x) { return cx(a.re / x, a.im / x); }  /* complex / real */
PAC_FN static inline cplx mul_i(cplx a) { return cx(-a.im, a.re); }                  /* i × a */

#define ZERO (cx(0.0, 0.0))

/* ---- the currents' numbers ----
 * pac_real: a current's components and the arithmetic of the current rows (vertex, propagator
 * numerator, scale, closure products, copy, scatter); pac_amp: an amplitude. Both are double, unless
 * the translation unit defines PAC_MIXED (the GPU's mixed precision): pac_real is then float and
 * pac_amp double (float with PAC_AMP32). The momentum forms, the propagator denominators
 * (reciprocal_norm), the wavefunctions and the reduction stay double (the kernels: mixed, kernel_c.jl); the factors the
 * current rows take are pac_real (the program's rounded copy). rcplx is cplx over pac_real: in the
 * double build the same operations on the same type, so the same bits. */
#ifdef PAC_MIXED
typedef float pac_real;
#ifdef PAC_AMP32
typedef float pac_amp;
#else
typedef double pac_amp;
#endif
#else
typedef double pac_real;
typedef double pac_amp;
#endif

typedef struct { pac_real re, im; } rcplx;

PAC_FN static inline rcplx rx(pac_real re, pac_real im) { rcplx z = {re, im}; return z; }
PAC_FN static inline rcplx rget(const pac_real *p, int j) { return rx(p[2 * j], p[2 * j + 1]); }
PAC_FN static inline void rput(pac_real *p, int j, rcplx z) { p[2 * j] = z.re; p[2 * j + 1] = z.im; }
PAC_FN static inline rcplx radd(rcplx a, rcplx b) { return rx(a.re + b.re, a.im + b.im); }
PAC_FN static inline rcplx rsub(rcplx a, rcplx b) { return rx(a.re - b.re, a.im - b.im); }
PAC_FN static inline rcplx rneg(rcplx a) { return rx(-a.re, -a.im); }
PAC_FN static inline rcplx rmul(rcplx a, rcplx b) { return rx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
PAC_FN static inline rcplx rmulr(rcplx a, pac_real x) { return rx(a.re * x, a.im * x); }
PAC_FN static inline rcplx rmul_i(rcplx a) { return rx(-a.im, a.re); }
PAC_FN static inline rcplx rcombine(rcplx old, rcplx value, int mode) { return mode == MODE_ADD ? radd(old, value) : value; }
PAC_FN static inline void rput2(pac_real *out, rcplx a, rcplx b) { rput(out, 0, a); rput(out, 1, b); }
PAC_FN static inline void rput4(pac_real *out, rcplx a, rcplx b, rcplx c, rcplx d)
{
    rput(out, 0, a); rput(out, 1, b); rput(out, 2, c); rput(out, 3, d);
}
/* to and from double (no-ops in the double build) */
PAC_FN static inline rcplx rof(cplx z) { return rx((pac_real)z.re, (pac_real)z.im); }
PAC_FN static inline cplx dof(rcplx z) { return cx((double)z.re, (double)z.im); }

#define RZERO (rx(0, 0))

/* Julia's max and min: NaN propagates, and max(-0.0, 0.0) = 0.0, min(-0.0, 0.0) = -0.0 */
PAC_FN static inline double jmax(double x, double y)
{
    if (isnan(x) || isnan(y)) return x + y;
    if (x != y) return x > y ? x : y;
    return signbit(x) ? y : x;
}

PAC_FN static inline double jmin(double x, double y)
{
    if (isnan(x) || isnan(y)) return x + y;
    if (x != y) return x < y ? x : y;
    return signbit(x) ? x : y;
}

/* the new value of a component: value ("=") or old + value ("+=") */
PAC_FN static inline cplx combine(cplx old, cplx value, int mode) { return mode == MODE_ADD ? add(old, value) : value; }

PAC_FN static inline void put2(double *out, cplx a, cplx b) { put(out, 0, a); put(out, 1, b); }
PAC_FN static inline void put4(double *out, cplx a, cplx b, cplx c, cplx d)
{
    put(out, 0, a); put(out, 1, b); put(out, 2, c); put(out, 3, d);
}

/* ---- wavefunctions (k: a momentum, h: helicity, ch: chirality) ---- */
PAC_FN void scalar(double *out);                                                     /* 1 component */
PAC_FN void weyl_particle(double *out, const double *k, int h, int ch);              /* 2 */
PAC_FN void weyl_antiparticle(double *out, const double *k, int h, int ch);          /* 2 */
PAC_FN void quark_dirac(double *out, const double *k, int h);                        /* 4 */
PAC_FN void antiquark_dirac(double *out, const double *k, int h);                    /* 4 */
PAC_FN void quark_dirac_massive(double *out, const double *k, int h, double mass);   /* 4 */
PAC_FN void antiquark_dirac_massive(double *out, const double *k, int h, double mass); /* 4 */
PAC_FN void gluon(double *out, const double *k, int h);                              /* 4 */
PAC_FN void massive_vector(double *out, const double *k, int h, double mass);        /* 4 */

/* the wavefunction of a source by family and orientation; returns its components */
PAC_FN int wavefunction(double *out, int family, int orientation, const double *k, int h, int ch, double mass);
/* a source current: factor × its wavefunction; returns its components */
PAC_FN int source(double *out, int family, int orientation, const double *k, int h, int ch, double mass,
           const double *factor);
/* an embedded source (UIR v0.8): for each of its n entries, out[components[e]] = factors[e] × the
 * wavefunction's component waves[e], or 0 when waves[e] < 0 (factors: (re, im) pairs) */
PAC_FN void source_embedded(double *out, int family, int orientation, const double *k, int h, int ch, double mass,
                     const int *components, const int *waves, const double *factors, int n);

/* ---- vertex: o, before the coupling and the factor ---- */
PAC_FN void vertex_weyl_vector_a(pac_real *out, const pac_real *l, const pac_real *r);   /* Weyl, vector → Weyl */
PAC_FN void vertex_weyl_vector_b(pac_real *out, const pac_real *l, const pac_real *r);   /* Weyl, vector → Weyl */
PAC_FN void vertex_tensor_vector(pac_real *out, const pac_real *l, const pac_real *r);   /* tensor, vector → vector */
PAC_FN void vertex_vector_wedge(pac_real *out, const pac_real *l, const pac_real *r);    /* vector, vector → tensor */
PAC_FN void vertex_three_vector(pac_real *out, const pac_real *l, const pac_real *r, const pac_real *u,
                         const pac_real *v);                                  /* vector, vector → vector */
/* the vertex instruction: out[j] (mode)= o[j] × (coupling × factor); u, v unused but by three_vector
 * (the straight-line runner's: the double build only) */
#ifndef PAC_MIXED
PAC_FN void vertex(double *out, int type, int mode, const double *l, const double *r, const double *u,
            const double *v, const double *coupling, const double *factor);
#endif

/* ---- propagator: out = o × factor / (k·k) (1 / (k·k) in double, then rounded to pac_real) ---- */
PAC_FN void propagator_weyl_a(pac_real *out, const pac_real *c, const double *k, const pac_real *factor);
PAC_FN void propagator_weyl_b(pac_real *out, const pac_real *c, const double *k, const pac_real *factor);
PAC_FN void propagator_vector(pac_real *out, const pac_real *c, const double *k, const pac_real *factor);
PAC_FN void propagator(pac_real *out, int type, const pac_real *c, const double *k, const pac_real *factor);

/* ---- scale, clear, closure (in place) ---- */
PAC_FN void scale(pac_real *c, int components, const pac_real *factor);   /* c[j] = factor × c[j], j < components */
PAC_FN void clear(pac_real *c, int components);                         /* c[j] = 0, j < components */
/* amplitude += Σ_j (l[j] × r[j]) × (component_factors[j] × factor), j < components, in turn (the
 * products in pac_real, the sum in pac_amp) */
PAC_FN void closure(pac_amp *amplitude, const pac_real *l, const pac_real *r, const pac_real *component_factors,
             int components, const pac_real *factor);

/* ---- kernel ---- */
typedef struct { double *ptr; size_t len; } symjit_slice;
typedef int (*symjit_kernel)(double *, symjit_slice *, size_t, double *);
/* one call of a kernel (main.json kernels[i].symbol) on one point; `states` is scratch for
 * inputs + outputs slices; returns the kernel's status (0: ok) */
int kernel_call(symjit_kernel fn, double *output, int outputs, double *input, int inputs,
                symjit_slice *states);
/* the kernel's output pairs into `out`: pair n × factor (mode)= out[outputs[n]] (0-based, distinct),
 * or with `direct` the pair as it is (no factor), with the mode */
PAC_FN void kernel_store(double *out, const double *values, const int *outputs, int pairs, int direct,
                  int mode, const double *factor);

/* ---- copy, scatter (UIR v0.8: RustiCol's certified reuse and fanout) ---- */
/* out[j] = src[j] × factor (value first), j < n; a zero factor writes 0 */
PAC_FN void copy_row(pac_real *out, const pac_real *src, int n, const pac_real *factor);
/* out[j] (mode)= src[j] × factor with RustiCol's fanout shortcuts (a real factor: ±1 copies or
 * negates, else componentwise; a purely imaginary one: (-im k, re k); else the complex product) */
PAC_FN void scatter(pac_real *out, const pac_real *src, int n, int mode, const pac_real *factor);
/* ---- reduction ---- */
/* acc + weight × |a|², a = the amplitude × scale (value first) when scale is not NULL */
PAC_FN double reduce_term(double acc, const pac_amp *amplitude, const double *scale, double weight);
/* x × the group factor: x when numerator = denominator, else x × numerator ÷ denominator */
PAC_FN double group(double x, long numerator, long denominator);
/* group_factor × Σ_n weights[n] × |amplitudes[terms[n]]|² (terms 0-based) */
PAC_FN double reduction(const double *amplitudes, const int *terms, const double *weights, int n,
                 long numerator, long denominator);

PAC_FN double reduction_single(const double *amplitudes, int term, double weight, long numerator, long denominator);

#ifdef __cplusplus
}
#endif

#endif
