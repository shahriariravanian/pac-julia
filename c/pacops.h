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
 * into fma (gcc/clang: -ffp-contract=off; nvcc: --fmad=false). */

#ifndef PACOPS_H
#define PACOPS_H

#include <stddef.h>
#include <math.h>

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

static inline cplx cx(double re, double im) { cplx z = {re, im}; return z; }
static inline cplx get(const double *p, int j) { return cx(p[2 * j], p[2 * j + 1]); }
static inline void put(double *p, int j, cplx z) { p[2 * j] = z.re; p[2 * j + 1] = z.im; }

static inline cplx add(cplx a, cplx b) { return cx(a.re + b.re, a.im + b.im); }
static inline cplx sub(cplx a, cplx b) { return cx(a.re - b.re, a.im - b.im); }
static inline cplx neg(cplx a) { return cx(-a.re, -a.im); }
static inline cplx mul(cplx a, cplx b) { return cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
static inline cplx mulr(cplx a, double x) { return cx(a.re * x, a.im * x); }  /* complex × real */
static inline cplx divr(cplx a, double x) { return cx(a.re / x, a.im / x); }  /* complex / real */
static inline cplx mul_i(cplx a) { return cx(-a.im, a.re); }                  /* i × a */

static const cplx ZERO = {0.0, 0.0};

/* Julia's max and min: NaN propagates, and max(-0.0, 0.0) = 0.0, min(-0.0, 0.0) = -0.0 */
static inline double jmax(double x, double y)
{
    if (isnan(x) || isnan(y)) return x + y;
    if (x != y) return x > y ? x : y;
    return signbit(x) ? y : x;
}

static inline double jmin(double x, double y)
{
    if (isnan(x) || isnan(y)) return x + y;
    if (x != y) return x < y ? x : y;
    return signbit(x) ? x : y;
}

/* the new value of a component: value ("=") or old + value ("+=") */
static inline cplx combine(cplx old, cplx value, int mode) { return mode == MODE_ADD ? add(old, value) : value; }

static inline void put2(double *out, cplx a, cplx b) { put(out, 0, a); put(out, 1, b); }
static inline void put4(double *out, cplx a, cplx b, cplx c, cplx d)
{
    put(out, 0, a); put(out, 1, b); put(out, 2, c); put(out, 3, d);
}

/* ---- wavefunctions (k: a momentum, h: helicity, ch: chirality) ---- */
void scalar(double *out);                                                     /* 1 component */
void weyl_particle(double *out, const double *k, int h, int ch);              /* 2 */
void weyl_antiparticle(double *out, const double *k, int h, int ch);          /* 2 */
void quark_dirac(double *out, const double *k, int h);                        /* 4 */
void antiquark_dirac(double *out, const double *k, int h);                    /* 4 */
void quark_dirac_massive(double *out, const double *k, int h, double mass);   /* 4 */
void antiquark_dirac_massive(double *out, const double *k, int h, double mass); /* 4 */
void gluon(double *out, const double *k, int h);                              /* 4 */
void massive_vector(double *out, const double *k, int h, double mass);        /* 4 */

/* the wavefunction of a source by family and orientation; returns its components */
int wavefunction(double *out, int family, int orientation, const double *k, int h, int ch, double mass);
/* a source current: factor × its wavefunction; returns its components */
int source(double *out, int family, int orientation, const double *k, int h, int ch, double mass,
           const double *factor);

/* ---- vertex: o, before the coupling and the factor ---- */
void vertex_weyl_vector_a(double *out, const double *l, const double *r);   /* Weyl, vector → Weyl */
void vertex_weyl_vector_b(double *out, const double *l, const double *r);   /* Weyl, vector → Weyl */
void vertex_tensor_vector(double *out, const double *l, const double *r);   /* tensor, vector → vector */
void vertex_vector_wedge(double *out, const double *l, const double *r);    /* vector, vector → tensor */
void vertex_three_vector(double *out, const double *l, const double *r, const double *u,
                         const double *v);                                  /* vector, vector → vector */
/* the vertex instruction: out[j] (mode)= o[j] × (coupling × factor); u, v unused but by three_vector */
void vertex(double *out, int type, int mode, const double *l, const double *r, const double *u,
            const double *v, const double *coupling, const double *factor);

/* ---- propagator: out = o × factor / (k·k) ---- */
void propagator_weyl_a(double *out, const double *c, const double *k, const double *factor);
void propagator_weyl_b(double *out, const double *c, const double *k, const double *factor);
void propagator_vector(double *out, const double *c, const double *k, const double *factor);
void propagator(double *out, int type, const double *c, const double *k, const double *factor);

/* ---- scale, clear, closure (in place) ---- */
void scale(double *c, int components, const double *factor);   /* c[j] = factor × c[j], j < components */
void clear(double *c, int components);                         /* c[j] = 0, j < components */
/* amplitude += Σ_j (l[j] × r[j]) × (component_factors[j] × factor), j < components, in turn */
void closure(double *amplitude, const double *l, const double *r, const double *component_factors,
             int components, const double *factor);

/* ---- kernel ---- */
typedef struct { double *ptr; size_t len; } symjit_slice;
typedef int (*symjit_kernel)(double *, symjit_slice *, size_t, double *);
/* one call of a kernel (main.json kernels[i].symbol) on one point; `states` is scratch for
 * inputs + outputs slices; returns the kernel's status (0: ok) */
int kernel_call(symjit_kernel fn, double *output, int outputs, double *input, int inputs,
                symjit_slice *states);
/* the kernel's output pairs into `out`: pair n × factor (mode)= out[outputs[n]] (0-based, distinct),
 * or the pair as it is with `direct` and factor exactly 1 + 0i */
void kernel_store(double *out, const double *values, const int *outputs, int pairs, int direct,
                  int mode, const double *factor);

/* ---- reduction ---- */
/* group_factor × Σ_n weights[n] × |amplitudes[terms[n]]|² (terms 0-based) */
double reduction(const double *amplitudes, const int *terms, const double *weights, int n,
                 long numerator, long denominator);

double reduction_single(const double *amplitudes, int term, double weight, long numerator, long denominator);

#ifdef __cplusplus
}
#endif

#endif
