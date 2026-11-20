/* pacops.metal: c/pacops.{h,c} for Metal, in the GPU's mixed precision with df64 (df64.h) for f64:
 * the same functions, the same operations in the same order, as PAC_MIXED does on CUDA:
 *
 *   f32 (pac_real): a current's components and the current rows' arithmetic (vertices, propagator
 *                   numerators, scale, closure products, copy, scatter), the rows' factors
 *   df64:           momentum forms, propagator denominators (rounded once to f32), wavefunctions
 *                   (then rounded into the f32 planes), amplitudes, the reduction, the group factor
 *
 * Differences from c/pacops.c, all forced by MSL: no double (df64); pointer arguments name their
 * address space (`thread`: the interpreter's local operand arrays), so factors come by value
 * (rcplx, cdf64); no recursion (massive_vector flips its momentum in place). The straight-line
 * runner's `vertex`, `kernel_call` and `kernel_store` are not here: Metal runs the interpreter,
 * and uirvm.metal stores kernel outputs itself.
 *
 * Keep in step with c/pacops.c: a change there belongs here too (test/test_metal_shim.jl compares
 * the two on the CPU). Needs df64.h, and <metal_stdlib> or msl_shim.h, before it. */

#ifndef PACOPS_METAL
#define PACOPS_METAL

/* main.json's strings as numbers (pacops.h's) */
enum { MODE_SET = 0, MODE_ADD = 1 };
enum { FAMILY_SCALAR, FAMILY_WEYL, FAMILY_DIRAC, FAMILY_VECTOR };
enum { PARTICLE = 0, ANTIPARTICLE = 1 };
enum { VERTEX_WEYL_VECTOR_A, VERTEX_WEYL_VECTOR_B, VERTEX_TENSOR_VECTOR, VERTEX_VECTOR_WEDGE,
       VERTEX_THREE_VECTOR };
enum { PROPAGATOR_WEYL_A, PROPAGATOR_WEYL_B, PROPAGATOR_VECTOR };

/* ---- f32 complex numbers: the currents (pacops.h's rcplx under PAC_MIXED) ---- */

typedef float pac_real;
struct rcplx {
    float re, im;
};

static inline rcplx rx(float re, float im) { rcplx z; z.re = re; z.im = im; return z; }
static inline rcplx rget(thread const float *p, int j) { return rx(p[2 * j], p[2 * j + 1]); }
static inline void rput(thread float *p, int j, rcplx z) { p[2 * j] = z.re; p[2 * j + 1] = z.im; }
static inline rcplx radd(rcplx a, rcplx b) { return rx(a.re + b.re, a.im + b.im); }
static inline rcplx rsub(rcplx a, rcplx b) { return rx(a.re - b.re, a.im - b.im); }
static inline rcplx rneg(rcplx a) { return rx(-a.re, -a.im); }
static inline rcplx rmul(rcplx a, rcplx b) { return rx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
static inline rcplx rmulr(rcplx a, float x) { return rx(a.re * x, a.im * x); }
static inline rcplx rmul_i(rcplx a) { return rx(-a.im, a.re); }
static inline rcplx rcombine(rcplx old, rcplx value, int mode) { return mode == MODE_ADD ? radd(old, value) : value; }
static inline void rput2(thread float *out, rcplx a, rcplx b) { rput(out, 0, a); rput(out, 1, b); }
static inline void rput4(thread float *out, rcplx a, rcplx b, rcplx c, rcplx d)
{
    rput(out, 0, a); rput(out, 1, b); rput(out, 2, c); rput(out, 3, d);
}
/* to and from df64: rounded once, widened exactly */
static inline rcplx rof(cdf64 z) { return rx(dd_to_f(z.re), dd_to_f(z.im)); }
static inline cdf64 dof(rcplx z) { return cdd(dd_f(z.re), dd_f(z.im)); }

#define RZERO (rx(0.0f, 0.0f))

/* ---- df64 complex helpers of the wavefunctions ---- */

static inline df64 D(float x) { return dd_f(x); }                 /* an exact small constant */
static inline cdf64 creal(df64 x) { return cdd(x, D(0.0f)); }
#define CZERO (cdd(D(0.0f), D(0.0f)))
static inline void put2(thread cdf64 *out, cdf64 a, cdf64 b) { out[0] = a; out[1] = b; }
static inline void put4(thread cdf64 *out, cdf64 a, cdf64 b, cdf64 c, cdf64 d)
{
    out[0] = a; out[1] = b; out[2] = c; out[3] = d;
}
static inline cdf64 combine(cdf64 old, cdf64 value, int mode) { return mode == MODE_ADD ? cdd_add(old, value) : value; }

/* ==== wavefunctions (k: a momentum, h: helicity, ch: chirality) ================================ */

static inline void scalar(thread cdf64 *out) { out[0] = creal(D(1.0f)); }

/* ---- Weyl spinors (ext_quark_weyl_array, ext_antiquark_weyl_array) ---- */

static inline void weyl_particle(thread cdf64 *out, thread const df64 *k, int h, int ch)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(hd * px / s, -py / s);
        if (h == +1 && ch == +1) put2(out, a, b);
        else if (h == -1 && ch == -1) put2(out, b, a);
        else put2(out, CZERO, CZERO);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : -dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(hd * px / s, py / s);
        if (h == -1 && ch == +1) put2(out, a, b);
        else if (h == +1 && ch == -1) put2(out, b, a);
        else put2(out, CZERO, CZERO);
    }
}

static inline void weyl_antiparticle(thread cdf64 *out, thread const df64 *k, int h, int ch)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : -dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(-hd * px / s, py / s);
        if (h == +1 && ch == +1) put2(out, b, a);
        else if (h == -1 && ch == -1) put2(out, a, b);
        else put2(out, CZERO, CZERO);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(-hd * px / s, -py / s);
        if (h == -1 && ch == +1) put2(out, b, a);
        else if (h == +1 && ch == -1) put2(out, a, b);
        else put2(out, CZERO, CZERO);
    }
}

/* ---- massless Dirac spinors (ext_quark_dirac, ext_antiquark_dirac) ---- */

static inline void quark_dirac(thread cdf64 *out, thread const df64 *k, int h)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(hd * px / s, -py / s);
        if (h == +1) put4(out, a, b, CZERO, CZERO);
        else put4(out, CZERO, CZERO, b, a);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : -dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(hd * px / s, py / s);
        if (h == -1) put4(out, a, b, CZERO, CZERO);
        else put4(out, CZERO, CZERO, b, a);
    }
}

static inline void antiquark_dirac(thread cdf64 *out, thread const df64 *k, int h)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : -dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(-hd * px / s, py / s);
        if (h == -1) put4(out, CZERO, CZERO, a, b);
        else put4(out, b, a, CZERO, CZERO);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(-hd * px / s, -py / s);
        if (h == +1) put4(out, CZERO, CZERO, a, b);
        else put4(out, b, a, CZERO, CZERO);
    }
}

/* ---- Dirac spinors (ext_quark_dirac_massive, ext_antiquark_dirac_massive) ----
 * |mass| < 1e-8 falls back to the massless spinors (the threshold compared in f32: a mass is never
 * near it). The two differ in nsf, the sign of py in chi2 and the order of the products. */

static inline void dirac_massive(thread cdf64 *out, thread const df64 *k, int h, df64 mass, int anti)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f);
    int nsf = anti ? (E > zero ? -1 : 1) : (E > zero ? 1 : -1);
    int nh = nsf * h;

    df64 pp = dd_sqrt(px * px + py * py + pz * pz);
    df64 omega1 = dd_sqrt(dd_abs(E) + pp);
    df64 omega2 = mass / omega1;

    df64 sf1 = D((float)(1 + nsf + (1 - nsf) * nh)) * D(0.5f);
    df64 sf2 = D((float)(1 + nsf - (1 - nsf) * nh)) * D(0.5f);

    df64 spx = E > zero ? px : -px, spy = E > zero ? py : -py, spz = E > zero ? pz : -pz;
    df64 pp3 = dd_max(pp + spz, zero);
    cdf64 chi1 = dd_is_zero(pp) ? creal(D(1.0f)) : creal(dd_sqrt(pp3 * D(0.5f) / pp));
    cdf64 chi2 = dd_is_zero(pp3) || dd_is_zero(pp) ? creal(D((float)-nh))
                                                  : cdd_divr(cdd(D((float)nh) * spx, anti ? spy : -spy), dd_sqrt(D(2.0f) * pp * pp3));

    /* the quark's (upper, lower) weights; the antiquark's are swapped */
    df64 up1 = anti ? sf1 * omega2 : sf2 * omega1, lo1 = anti ? sf2 * omega1 : sf1 * omega2;
    df64 up2 = anti ? sf1 * omega1 : sf2 * omega2, lo2 = anti ? sf2 * omega2 : sf1 * omega1;
    if (nh > 0) put4(out, cdd_mulr(chi1, up1), cdd_mulr(chi2, up1), cdd_mulr(chi1, lo1), cdd_mulr(chi2, lo1));
    else put4(out, cdd_mulr(chi2, up2), cdd_mulr(chi1, up2), cdd_mulr(chi2, lo2), cdd_mulr(chi1, lo2));
}

static inline void quark_dirac_massive(thread cdf64 *out, thread const df64 *k, int h, df64 mass)
{
    if (dd_abs(mass).hi < 1.0e-8f) quark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 0);
}

static inline void antiquark_dirac_massive(thread cdf64 *out, thread const df64 *k, int h, df64 mass)
{
    if (dd_abs(mass).hi < 1.0e-8f) antiquark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 1);
}

/* ---- massless vectors: gluon, photon (ext_gluon) ---- */

static inline void gluon(thread cdf64 *out, thread const df64 *k, int h)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f);
    df64 sqh = dd_sqrt(D(0.5f));
    cdf64 wf1, wf2, wf3;
    if (E > zero) {
        df64 hel = D((float)h);
        df64 pp = E;
        df64 pt = dd_sqrt(px * px + py * py);
        wf3 = creal(hel * pt / pp * sqh);
        if (!dd_is_zero(pt)) {
            df64 pzpt = pz / (pp * pt) * sqh * hel;
            wf1 = cdd(-px * pzpt, -py / pt * sqh);
            wf2 = cdd(-py * pzpt, px / pt * sqh);
        } else {
            wf1 = creal(-hel * sqh);
            wf2 = cdd(zero, dd_copysign(sqh, pz));
        }
    } else {
        df64 hel = D((float)-h);
        df64 pp = -E;
        df64 pt = dd_sqrt(px * px + py * py);
        wf3 = creal(hel * pt / pp * sqh);
        if (!dd_is_zero(pt)) {
            df64 pzpt = -pz / (pp * pt) * sqh * hel;
            wf1 = cdd(px * pzpt, py / pt * sqh);
            wf2 = cdd(py * pzpt, -px / pt * sqh);
        } else {
            wf1 = creal(-hel * sqh);
            wf2 = cdd(zero, -dd_copysign(sqh, pz));
        }
    }
    put4(out, CZERO, wf1, wf2, wf3);
}

/* ---- massive vectors: Z, W (ext_massive_vector) ---- */

static inline void massive_vector(thread cdf64 *out, thread const df64 *k, int h, df64 mass)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f);
    if (E < zero) {   /* c/pacops.c recurses with (-k, -h) */
        E = -E; px = -px; py = -py; pz = -pz; h = -h;
    }

    df64 sqh = dd_sqrt(D(0.5f));
    df64 hel = D((float)h);
    df64 nsvahl = D((float)abs(h));
    df64 pt2 = px * px + py * py;
    df64 pp = dd_min(E, dd_sqrt(pt2 + pz * pz));
    df64 pt = dd_min(pp, dd_sqrt(pt2));
    df64 hel0 = D(1.0f) - dd_abs(hel);

    if (dd_is_zero(pp)) {
        put4(out, CZERO, creal(-hel * sqh), cdd(zero, nsvahl * sqh), creal(hel0));
        return;
    }

    df64 emp = E / (mass * pp);
    cdf64 wf0 = creal(hel0 * pp / mass);
    cdf64 wf3 = creal(hel0 * pz * emp + hel * pt / pp * sqh);
    cdf64 wf1, wf2;
    if (!dd_is_zero(pt)) {
        df64 pzpt = pz / (pp * pt) * sqh * hel;
        wf1 = cdd(hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh);
        wf2 = cdd(hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh);
    } else {
        wf1 = creal(-hel * sqh);
        wf2 = cdd(zero, nsvahl * dd_copysign(sqh, pz));
    }
    put4(out, wf0, wf1, wf2, wf3);
}

/* ---- a source current ---- */

/* the runner's pick: a Dirac source always through the massive spinors, a vector by whether its
 * mass is exactly 0; returns the components */
static inline int wavefunction(thread cdf64 *out, int family, int orientation, thread const df64 *k, int h, int ch,
                               df64 mass)
{
    int anti = orientation == ANTIPARTICLE;
    switch (family) {
    case FAMILY_SCALAR: scalar(out); return 1;
    case FAMILY_WEYL:
        if (anti) weyl_antiparticle(out, k, h, ch); else weyl_particle(out, k, h, ch);
        return 2;
    case FAMILY_DIRAC:
        if (anti) antiquark_dirac_massive(out, k, h, mass); else quark_dirac_massive(out, k, h, mass);
        return 4;
    case FAMILY_VECTOR:
        if (dd_is_zero(mass)) gluon(out, k, h); else massive_vector(out, k, h, mass);
        return 4;
    default: return 0;
    }
}

/* a source current: factor × its wavefunction; returns its components */
static inline int source(thread cdf64 *out, int family, int orientation, thread const df64 *k, int h, int ch, df64 mass,
                         cdf64 factor)
{
    int n = wavefunction(out, family, orientation, k, h, ch, mass);
    for (int j = 0; j < n; j++) out[j] = cdd_mul(factor, out[j]);
    return n;
}

/* ==== vertex: o, before the coupling and the factor (f32) ======================================== */

static inline void vertex_weyl_vector_a(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o0 = radd(radd(rsub(rneg(rmul_i(rmul(l0, r3))), rmul_i(rmul(l1, r1))), rmul(l1, r2)), rmul_i(rmul(l0, r0)));
    rcplx o1 = radd(radd(rsub(rneg(rmul(l0, r2)), rmul_i(rmul(l0, r1))), rmul_i(rmul(l1, r0))), rmul_i(rmul(l1, r3)));
    rput2(out, o0, o1);
}

static inline void vertex_weyl_vector_b(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o0 = radd(radd(radd(rneg(rmul(l1, r2)), rmul_i(rmul(l0, r0))), rmul_i(rmul(l0, r3))), rmul_i(rmul(l1, r1)));
    rcplx o1 = radd(radd(radd(rneg(rmul_i(rmul(l1, r3))), rmul(l0, r2)), rmul_i(rmul(l0, r1))), rmul_i(rmul(l1, r0)));
    rput2(out, o0, o1);
}

static inline void vertex_tensor_vector(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1), l2 = rget(l, 2), l3 = rget(l, 3), l4 = rget(l, 4), l5 = rget(l, 5);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rput4(out,
          radd(radd(rmul(l0, r1), rmul(l1, r2)), rmul(l2, r3)),
          radd(radd(rmul(l0, r0), rmul(l3, r2)), rmul(l4, r3)),
          radd(rsub(rmul(l1, r0), rmul(l3, r1)), rmul(l5, r3)),
          rsub(rsub(rmul(l2, r0), rmul(l4, r1)), rmul(l5, r2)));
}

static inline void vertex_vector_wedge(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1), l2 = rget(l, 2), l3 = rget(l, 3);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o[6] = {rsub(rmul(l0, r1), rmul(l1, r0)), rsub(rmul(l0, r2), rmul(l2, r0)), rsub(rmul(l0, r3), rmul(l3, r0)),
                  rsub(rmul(l1, r2), rmul(l2, r1)), rsub(rmul(l1, r3), rmul(l3, r1)), rsub(rmul(l2, r3), rmul(l3, r2))};
    for (int j = 0; j < 6; j++) rput(out, j, o[j]);
}

static inline rcplx dot(thread const float *a, thread const float *b)
{
    return rsub(rsub(rsub(rmul(rget(a, 0), rget(b, 0)), rmul(rget(a, 1), rget(b, 1))), rmul(rget(a, 2), rget(b, 2))),
                rmul(rget(a, 3), rget(b, 3)));
}

static inline rcplx dot_real(thread const float *a, thread const float *b)
{
    return rsub(rsub(rsub(rmulr(rget(a, 0), b[0]), rmulr(rget(a, 1), b[1])), rmulr(rget(a, 2), b[2])), rmulr(rget(a, 3), b[3]));
}

/* the colour-ordered three-gluon vertex; u = momenta[0], v = momenta[1] (rounded to f32) */
static inline void vertex_three_vector(thread float *out, thread const float *l, thread const float *r, thread const float *u,
                                       thread const float *v)
{
    rcplx lr = dot(l, r), lv = dot_real(l, v), ru = dot_real(r, u);
    rcplx o[4];
    for (int j = 0; j < 4; j++)
        o[j] = radd(rmulr(lr, u[j] - v[j]), rmulr(rsub(rmul(lv, rget(r, j)), rmul(ru, rget(l, j))), 2.0f));
    for (int j = 0; j < 4; j++) rput(out, j, o[j]);
}

/* ==== propagator: out[j] = o[j] × factor / (k·k) ================================================= */

/* 1 / (k·k) in df64; the numerators take k rounded to f32 (kr) */
static inline df64 reciprocal_norm(thread const df64 *k)
{
    return D(1.0f) / (((k[0] * k[0] - k[1] * k[1]) - k[2] * k[2]) - k[3] * k[3]);
}

static inline void propagator_weyl_a(thread float *out, thread const float *c, thread const df64 *k, rcplx factor)
{
    float kr[4] = {dd_to_f(k[0]), dd_to_f(k[1]), dd_to_f(k[2]), dd_to_f(k[3])};
    rcplx c0 = rget(c, 0), c1 = rget(c, 1);
    rcplx s = rmulr(factor, dd_to_f(reciprocal_norm(k)));
    rcplx o0 = radd(radd(rsub(rneg(rmul_i(rmulr(c0, kr[3]))), rmul_i(rmulr(c1, kr[1]))), rmulr(c1, kr[2])), rmul_i(rmulr(c0, kr[0])));
    rcplx o1 = radd(radd(rsub(rneg(rmulr(c0, kr[2])), rmul_i(rmulr(c0, kr[1]))), rmul_i(rmulr(c1, kr[0]))), rmul_i(rmulr(c1, kr[3])));
    rput2(out, rmul(o0, s), rmul(o1, s));
}

static inline void propagator_weyl_b(thread float *out, thread const float *c, thread const df64 *k, rcplx factor)
{
    float kr[4] = {dd_to_f(k[0]), dd_to_f(k[1]), dd_to_f(k[2]), dd_to_f(k[3])};
    rcplx c0 = rget(c, 0), c1 = rget(c, 1);
    rcplx s = rmulr(factor, dd_to_f(reciprocal_norm(k)));
    rcplx o0 = radd(radd(radd(rneg(rmulr(c1, kr[2])), rmul_i(rmulr(c0, kr[0]))), rmul_i(rmulr(c0, kr[3]))), rmul_i(rmulr(c1, kr[1])));
    rcplx o1 = radd(radd(radd(rneg(rmul_i(rmulr(c1, kr[3]))), rmulr(c0, kr[2])), rmul_i(rmulr(c0, kr[1]))), rmul_i(rmulr(c1, kr[0])));
    rput2(out, rmul(o0, s), rmul(o1, s));
}

/* c vector, Feynman gauge: o[j] = -i c[j] */
static inline void propagator_vector(thread float *out, thread const float *c, thread const df64 *k, rcplx factor)
{
    rcplx s = rmulr(rmul(factor, rx(0.0f, -1.0f)), dd_to_f(reciprocal_norm(k)));
    for (int j = 0; j < 4; j++) rput(out, j, rmul(rget(c, j), s));
}

static inline void propagator(thread float *out, int type, thread const float *c, thread const df64 *k, rcplx factor)
{
    switch (type) {
    case PROPAGATOR_WEYL_A: propagator_weyl_a(out, c, k, factor); break;
    case PROPAGATOR_WEYL_B: propagator_weyl_b(out, c, k, factor); break;
    case PROPAGATOR_VECTOR: propagator_vector(out, c, k, factor); break;
    }
}

/* ==== scale, clear, closure ====================================================================== */

static inline void scale(thread float *c, int components, rcplx f)
{
    for (int j = 0; j < components; j++) rput(c, j, rmul(f, rget(c, j)));
}

static inline void clear(thread float *c, int components)
{
    for (int j = 0; j < components; j++) rput(c, j, RZERO);
}

/* amplitude += Σ_j (l[j] × r[j]) × (component_factors[j] × factor): products in f32, the sum in df64 */
static inline void closure(thread cdf64 &amplitude, thread const float *l, thread const float *r,
                           thread const float *component_factors, int components, rcplx f)
{
    df64 re = amplitude.re, im = amplitude.im;
    for (int j = 0; j < components; j++) {
        rcplx t = rmul(rmul(rget(l, j), rget(r, j)), rmul(rget(component_factors, j), f));
        re = re + dd_f(t.re);
        im = im + dd_f(t.im);
    }
    amplitude = cdd(re, im);
}

/* ==== copy, scatter (f32) ======================================================================== */

static inline void copy_row(thread float *out, thread const float *src, int n, rcplx f)
{
    bool zero = f.re == 0.0f && f.im == 0.0f;
    for (int j = 0; j < n; j++) rput(out, j, zero ? RZERO : rmul(rget(src, j), f));
}

static inline rcplx scatter_value(rcplx v, rcplx f)
{
    if (f.im == 0.0f) {
        if (f.re == 1.0f) return v;
        if (f.re == -1.0f) return rneg(v);
        return rx(v.re * f.re, v.im * f.re);
    }
    if (f.re == 0.0f) return rx(-v.im * f.im, v.re * f.im);
    return rmul(v, f);
}

static inline void scatter(thread float *out, thread const float *src, int n, int mode, rcplx f)
{
    for (int j = 0; j < n; j++) rput(out, j, rcombine(rget(out, j), scatter_value(rget(src, j), f), mode));
}

/* ==== reduction (df64) =========================================================================== */

/* acc + weight × |a|², a = the amplitude × scale when there is one (pacops' fma(re, re, im im): here
 * re re + im im, each product and the sum in df64) */
static inline df64 reduce_term(df64 acc, cdf64 a, bool scaled, cdf64 scale_, df64 weight)
{
    if (scaled) a = cdd_mul(a, scale_);
    return acc + weight * (a.re * a.re + a.im * a.im);
}

/* x × the group factor: x when numerator = denominator, else x × numerator ÷ denominator */
static inline df64 group(df64 x, long numerator, long denominator)
{
    return numerator == denominator ? x : x * dd_int(numerator) / dd_int(denominator);
}

#endif
