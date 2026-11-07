/* The wavefunctions (wavefunctions.jl) and the instructions (operations.jl) in plain C: the same
 * operations in the same order, so the values are bit for bit the Julia runner's (and RustiCol's).
 * The layout and the conventions are in pacops.h. Compile with -ffp-contract=off. */

#include <math.h>
#include <stdlib.h>
#include "pacops.h"

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

/* ==== wavefunctions ============================================================================ */

void scalar(double *out) { put(out, 0, cx(1.0, 0.0)); }

/* ---- Weyl spinors (ext_quark_weyl_array, ext_antiquark_weyl_array) ---- */

void weyl_particle(double *out, const double *k, int h, int ch)
{
    double E = k[0], px = k[1], py = k[2], pz = k[3];
    if (E > 0) {
        double s = px == 0 && py == 0 && pz < 0 ? 0.0 : sqrt(jmax(E + pz, 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * E), 0.0) : cx(h * px / s, -py / s);
        if (h == +1 && ch == +1) put2(out, a, b);
        else if (h == -1 && ch == -1) put2(out, b, a);
        else put2(out, ZERO, ZERO);
    } else {
        double s = px == 0 && py == 0 && pz > 0 ? 0.0 : -sqrt(jmax(-(E + pz), 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * fabs(E)), 0.0) : cx(h * px / s, py / s);
        if (h == -1 && ch == +1) put2(out, a, b);
        else if (h == +1 && ch == -1) put2(out, b, a);
        else put2(out, ZERO, ZERO);
    }
}

void weyl_antiparticle(double *out, const double *k, int h, int ch)
{
    double E = k[0], px = k[1], py = k[2], pz = k[3];
    if (E > 0) {
        double s = px == 0 && py == 0 && pz < 0 ? 0.0 : -sqrt(jmax(E + pz, 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * E), 0.0) : cx(-h * px / s, py / s);
        if (h == +1 && ch == +1) put2(out, b, a);
        else if (h == -1 && ch == -1) put2(out, a, b);
        else put2(out, ZERO, ZERO);
    } else {
        double s = px == 0 && py == 0 && pz > 0 ? 0.0 : sqrt(jmax(-(E + pz), 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * fabs(E)), 0.0) : cx(-h * px / s, -py / s);
        if (h == -1 && ch == +1) put2(out, b, a);
        else if (h == +1 && ch == -1) put2(out, a, b);
        else put2(out, ZERO, ZERO);
    }
}

/* ---- massless Dirac spinors (ext_quark_dirac, ext_antiquark_dirac) ---- */

void quark_dirac(double *out, const double *k, int h)
{
    double E = k[0], px = k[1], py = k[2], pz = k[3];
    if (E > 0) {
        double s = px == 0 && py == 0 && pz < 0 ? 0.0 : sqrt(jmax(E + pz, 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * E), 0.0) : cx(h * px / s, -py / s);
        if (h == +1) put4(out, a, b, ZERO, ZERO);
        else put4(out, ZERO, ZERO, b, a);
    } else {
        double s = px == 0 && py == 0 && pz > 0 ? 0.0 : -sqrt(jmax(-(E + pz), 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * fabs(E)), 0.0) : cx(h * px / s, py / s);
        if (h == -1) put4(out, a, b, ZERO, ZERO);
        else put4(out, ZERO, ZERO, b, a);
    }
}

void antiquark_dirac(double *out, const double *k, int h)
{
    double E = k[0], px = k[1], py = k[2], pz = k[3];
    if (E > 0) {
        double s = px == 0 && py == 0 && pz < 0 ? 0.0 : -sqrt(jmax(E + pz, 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * E), 0.0) : cx(-h * px / s, py / s);
        if (h == -1) put4(out, ZERO, ZERO, a, b);
        else put4(out, b, a, ZERO, ZERO);
    } else {
        double s = px == 0 && py == 0 && pz > 0 ? 0.0 : sqrt(jmax(-(E + pz), 0.0));
        cplx a = cx(s, 0.0);
        cplx b = s == 0 ? cx(-h * sqrt(2 * fabs(E)), 0.0) : cx(-h * px / s, -py / s);
        if (h == +1) put4(out, ZERO, ZERO, a, b);
        else put4(out, b, a, ZERO, ZERO);
    }
}

/* ---- Dirac spinors (ext_quark_dirac_massive, ext_antiquark_dirac_massive) ----
 * |mass| < 1e-8 falls back to the massless spinors. The two differ in nsf, the sign of py in chi2
 * and the order of the products; `anti` picks. */

static void dirac_massive(double *out, const double *k, int h, double mass, int anti)
{
    double E = k[0], px = k[1], py = k[2], pz = k[3];
    int nsf = anti ? (E > 0 ? -1 : 1) : (E > 0 ? 1 : -1);
    int nh = nsf * h;

    double pp = sqrt(px * px + py * py + pz * pz);
    double omega1 = sqrt(fabs(E) + pp);
    double omega2 = mass / omega1;

    double sf1 = (1 + nsf + (1 - nsf) * nh) * 0.5;
    double sf2 = (1 + nsf - (1 - nsf) * nh) * 0.5;

    double spx = E > 0 ? px : -px, spy = E > 0 ? py : -py, spz = E > 0 ? pz : -pz;
    double pp3 = jmax(pp + spz, 0.0);
    cplx chi1 = pp == 0 ? cx(1.0, 0.0) : cx(sqrt(pp3 * 0.5 / pp), 0.0);
    cplx chi2 = pp3 == 0 || pp == 0 ? cx(-(double)nh, 0.0)
                                    : divr(cx(nh * spx, anti ? spy : -spy), sqrt(2.0 * pp * pp3));

    /* the quark's (upper, lower) weights; the antiquark's are swapped */
    double up1 = anti ? sf1 * omega2 : sf2 * omega1, lo1 = anti ? sf2 * omega1 : sf1 * omega2;
    double up2 = anti ? sf1 * omega1 : sf2 * omega2, lo2 = anti ? sf2 * omega2 : sf1 * omega1;
    if (nh > 0) put4(out, mulr(chi1, up1), mulr(chi2, up1), mulr(chi1, lo1), mulr(chi2, lo1));
    else put4(out, mulr(chi2, up2), mulr(chi1, up2), mulr(chi2, lo2), mulr(chi1, lo2));
}

void quark_dirac_massive(double *out, const double *k, int h, double mass)
{
    if (fabs(mass) < 1.0e-8) quark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 0);
}

void antiquark_dirac_massive(double *out, const double *k, int h, double mass)
{
    if (fabs(mass) < 1.0e-8) antiquark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 1);
}

/* ---- massless vectors: gluon, photon (ext_gluon) ---- */

void gluon(double *out, const double *k, int h)
{
    double E = k[0], px = k[1], py = k[2], pz = k[3];
    double sqh = sqrt(0.5);
    cplx wf1, wf2, wf3;
    if (E > 0) {
        double hel = (double)h;
        double pp = E;
        double pt = sqrt(px * px + py * py);
        wf3 = cx(hel * pt / pp * sqh, 0.0);
        if (pt != 0) {
            double pzpt = pz / (pp * pt) * sqh * hel;
            wf1 = cx(-px * pzpt, -py / pt * sqh);
            wf2 = cx(-py * pzpt, px / pt * sqh);
        } else {
            wf1 = cx(-hel * sqh, 0.0);
            wf2 = cx(0.0, copysign(sqh, pz));
        }
    } else {
        double hel = (double)(-h);
        double pp = -E;
        double pt = sqrt(px * px + py * py);
        wf3 = cx(hel * pt / pp * sqh, 0.0);
        if (pt != 0) {
            double pzpt = -pz / (pp * pt) * sqh * hel;
            wf1 = cx(px * pzpt, py / pt * sqh);
            wf2 = cx(py * pzpt, -px / pt * sqh);
        } else {
            wf1 = cx(-hel * sqh, 0.0);
            wf2 = cx(0.0, -copysign(sqh, pz));
        }
    }
    put4(out, cx(0.0, 0.0), wf1, wf2, wf3);
}

/* ---- massive vectors: Z, W (ext_massive_vector) ---- */

void massive_vector(double *out, const double *k, int h, double mass)
{
    double E = k[0], px = k[1], py = k[2], pz = k[3];
    if (E < 0) {
        double kk[4] = {-E, -px, -py, -pz};
        massive_vector(out, kk, -h, mass);
        return;
    }

    double sqh = sqrt(0.5);
    double hel = (double)h;
    double nsvahl = (double)abs(h);
    double pt2 = px * px + py * py;
    double pp = jmin(E, sqrt(pt2 + pz * pz));
    double pt = jmin(pp, sqrt(pt2));
    double hel0 = 1.0 - fabs(hel);

    if (pp == 0) {
        put4(out, cx(0.0, 0.0), cx(-hel * sqh, 0.0), cx(0.0, nsvahl * sqh), cx(hel0, 0.0));
        return;
    }

    double emp = E / (mass * pp);
    cplx wf0 = cx(hel0 * pp / mass, 0.0);
    cplx wf3 = cx(hel0 * pz * emp + hel * pt / pp * sqh, 0.0);
    cplx wf1, wf2;
    if (pt != 0) {
        double pzpt = pz / (pp * pt) * sqh * hel;
        wf1 = cx(hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh);
        wf2 = cx(hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh);
    } else {
        wf1 = cx(-hel * sqh, 0.0);
        wf2 = cx(0.0, nsvahl * copysign(sqh, pz));
    }
    put4(out, wf0, wf1, wf2, wf3);
}

/* ---- a source current ---- */

/* the runner's pick (`source_values`): a Dirac source always through the massive spinors, a vector
 * by whether its mass is exactly 0 */
int wavefunction(double *out, int family, int orientation, const double *k, int h, int ch, double mass)
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
        if (mass == 0) gluon(out, k, h); else massive_vector(out, k, h, mass);
        return 4;
    default: return 0;
    }
}

int source(double *out, int family, int orientation, const double *k, int h, int ch, double mass,
           const double *factor)
{
    int n = wavefunction(out, family, orientation, k, h, ch, mass);
    cplx f = get(factor, 0);
    for (int j = 0; j < n; j++) put(out, j, mul(f, get(out, j)));
    return n;
}

/* ==== vertex ===================================================================================== */

/* l Weyl, r vector → Weyl:
 *   o[0] = -i l[0] r[3] - i l[1] r[1] + l[1] r[2] + i l[0] r[0]
 *   o[1] = -l[0] r[2] - i l[0] r[1] + i l[1] r[0] + i l[1] r[3] */
void vertex_weyl_vector_a(double *out, const double *l, const double *r)
{
    cplx l0 = get(l, 0), l1 = get(l, 1);
    cplx r0 = get(r, 0), r1 = get(r, 1), r2 = get(r, 2), r3 = get(r, 3);
    cplx o0 = add(add(sub(neg(mul_i(mul(l0, r3))), mul_i(mul(l1, r1))), mul(l1, r2)), mul_i(mul(l0, r0)));
    cplx o1 = add(add(sub(neg(mul(l0, r2)), mul_i(mul(l0, r1))), mul_i(mul(l1, r0))), mul_i(mul(l1, r3)));
    put2(out, o0, o1);
}

/* l Weyl, r vector → Weyl:
 *   o[0] = -l[1] r[2] + i l[0] r[0] + i l[0] r[3] + i l[1] r[1]
 *   o[1] = -i l[1] r[3] + l[0] r[2] + i l[0] r[1] + i l[1] r[0] */
void vertex_weyl_vector_b(double *out, const double *l, const double *r)
{
    cplx l0 = get(l, 0), l1 = get(l, 1);
    cplx r0 = get(r, 0), r1 = get(r, 1), r2 = get(r, 2), r3 = get(r, 3);
    cplx o0 = add(add(add(neg(mul(l1, r2)), mul_i(mul(l0, r0))), mul_i(mul(l0, r3))), mul_i(mul(l1, r1)));
    cplx o1 = add(add(add(neg(mul_i(mul(l1, r3))), mul(l0, r2)), mul_i(mul(l0, r1))), mul_i(mul(l1, r0)));
    put2(out, o0, o1);
}

/* l antisymmetric tensor, r vector → vector:
 *   o[0] = l[0] r[1] + l[1] r[2] + l[2] r[3]
 *   o[1] = l[0] r[0] + l[3] r[2] + l[4] r[3]
 *   o[2] = l[1] r[0] - l[3] r[1] + l[5] r[3]
 *   o[3] = l[2] r[0] - l[4] r[1] - l[5] r[2] */
void vertex_tensor_vector(double *out, const double *l, const double *r)
{
    cplx l0 = get(l, 0), l1 = get(l, 1), l2 = get(l, 2), l3 = get(l, 3), l4 = get(l, 4), l5 = get(l, 5);
    cplx r0 = get(r, 0), r1 = get(r, 1), r2 = get(r, 2), r3 = get(r, 3);
    put4(out,
         add(add(mul(l0, r1), mul(l1, r2)), mul(l2, r3)),
         add(add(mul(l0, r0), mul(l3, r2)), mul(l4, r3)),
         add(sub(mul(l1, r0), mul(l3, r1)), mul(l5, r3)),
         sub(sub(mul(l2, r0), mul(l4, r1)), mul(l5, r2)));
}

/* l vector, r vector → antisymmetric tensor:
 *   o = (l[0] r[1] - l[1] r[0], l[0] r[2] - l[2] r[0], l[0] r[3] - l[3] r[0],
 *        l[1] r[2] - l[2] r[1], l[1] r[3] - l[3] r[1], l[2] r[3] - l[3] r[2]) */
void vertex_vector_wedge(double *out, const double *l, const double *r)
{
    cplx l0 = get(l, 0), l1 = get(l, 1), l2 = get(l, 2), l3 = get(l, 3);
    cplx r0 = get(r, 0), r1 = get(r, 1), r2 = get(r, 2), r3 = get(r, 3);
    cplx o[6] = {sub(mul(l0, r1), mul(l1, r0)), sub(mul(l0, r2), mul(l2, r0)), sub(mul(l0, r3), mul(l3, r0)),
                 sub(mul(l1, r2), mul(l2, r1)), sub(mul(l1, r3), mul(l3, r1)), sub(mul(l2, r3), mul(l3, r2))};
    for (int j = 0; j < 6; j++) put(out, j, o[j]);
}

/* a·b of two complex vectors; of a complex vector and a momentum */
static cplx dot(const double *a, const double *b)
{
    return sub(sub(sub(mul(get(a, 0), get(b, 0)), mul(get(a, 1), get(b, 1))), mul(get(a, 2), get(b, 2))),
               mul(get(a, 3), get(b, 3)));
}

static cplx dot_real(const double *a, const double *b)
{
    return sub(sub(sub(mulr(get(a, 0), b[0]), mulr(get(a, 1), b[1])), mulr(get(a, 2), b[2])), mulr(get(a, 3), b[3]));
}

/* l vector, r vector → vector, the colour-ordered three-gluon vertex; u = momenta[0],
 * v = momenta[1]: o[j] = (l·r)(u[j] - v[j]) + 2((l·v) r[j] - (r·u) l[j]) */
void vertex_three_vector(double *out, const double *l, const double *r, const double *u, const double *v)
{
    cplx lr = dot(l, r), lv = dot_real(l, v), ru = dot_real(r, u);
    cplx o[4];
    for (int j = 0; j < 4; j++)
        o[j] = add(mulr(lr, u[j] - v[j]), mulr(sub(mul(lv, get(r, j)), mul(ru, get(l, j))), 2.0));
    for (int j = 0; j < 4; j++) put(out, j, o[j]);
}

/* the vertex instruction: out[j] (mode)= o[j] × (coupling × factor) */
void vertex(double *out, int type, int mode, const double *l, const double *r, const double *u,
            const double *v, const double *coupling, const double *factor)
{
    double o[12];
    int n;
    switch (type) {
    case VERTEX_WEYL_VECTOR_A: vertex_weyl_vector_a(o, l, r); n = 2; break;
    case VERTEX_WEYL_VECTOR_B: vertex_weyl_vector_b(o, l, r); n = 2; break;
    case VERTEX_TENSOR_VECTOR: vertex_tensor_vector(o, l, r); n = 4; break;
    case VERTEX_VECTOR_WEDGE: vertex_vector_wedge(o, l, r); n = 6; break;
    case VERTEX_THREE_VECTOR: vertex_three_vector(o, l, r, u, v); n = 4; break;
    default: return;
    }
    cplx s = mul(get(coupling, 0), get(factor, 0));
    for (int j = 0; j < n; j++) put(out, j, combine(get(out, j), mul(get(o, j), s), mode));
}

/* ==== propagator: out[j] = o[j] × factor / (k·k) ================================================= */

static double reciprocal_norm(const double *k) { return 1.0 / (((k[0] * k[0] - k[1] * k[1]) - k[2] * k[2]) - k[3] * k[3]); }

/* c Weyl:
 *   o[0] = -i c[0] k[3] - i c[1] k[1] + c[1] k[2] + i c[0] k[0]
 *   o[1] = -c[0] k[2] - i c[0] k[1] + i c[1] k[0] + i c[1] k[3] */
void propagator_weyl_a(double *out, const double *c, const double *k, const double *factor)
{
    cplx c0 = get(c, 0), c1 = get(c, 1);
    cplx s = mulr(get(factor, 0), reciprocal_norm(k));
    cplx o0 = add(add(sub(neg(mul_i(mulr(c0, k[3]))), mul_i(mulr(c1, k[1]))), mulr(c1, k[2])), mul_i(mulr(c0, k[0])));
    cplx o1 = add(add(sub(neg(mulr(c0, k[2])), mul_i(mulr(c0, k[1]))), mul_i(mulr(c1, k[0]))), mul_i(mulr(c1, k[3])));
    put2(out, mul(o0, s), mul(o1, s));
}

/* c Weyl:
 *   o[0] = -c[1] k[2] + i c[0] k[0] + i c[0] k[3] + i c[1] k[1]
 *   o[1] = -i c[1] k[3] + c[0] k[2] + i c[0] k[1] + i c[1] k[0] */
void propagator_weyl_b(double *out, const double *c, const double *k, const double *factor)
{
    cplx c0 = get(c, 0), c1 = get(c, 1);
    cplx s = mulr(get(factor, 0), reciprocal_norm(k));
    cplx o0 = add(add(add(neg(mulr(c1, k[2])), mul_i(mulr(c0, k[0]))), mul_i(mulr(c0, k[3]))), mul_i(mulr(c1, k[1])));
    cplx o1 = add(add(add(neg(mul_i(mulr(c1, k[3]))), mulr(c0, k[2])), mul_i(mulr(c0, k[1]))), mul_i(mulr(c1, k[0])));
    put2(out, mul(o0, s), mul(o1, s));
}

/* c vector, Feynman gauge: o[j] = -i c[j] */
void propagator_vector(double *out, const double *c, const double *k, const double *factor)
{
    cplx s = mulr(mul(get(factor, 0), cx(0.0, -1.0)), reciprocal_norm(k));
    for (int j = 0; j < 4; j++) put(out, j, mul(get(c, j), s));
}

void propagator(double *out, int type, const double *c, const double *k, const double *factor)
{
    switch (type) {
    case PROPAGATOR_WEYL_A: propagator_weyl_a(out, c, k, factor); break;
    case PROPAGATOR_WEYL_B: propagator_weyl_b(out, c, k, factor); break;
    case PROPAGATOR_VECTOR: propagator_vector(out, c, k, factor); break;
    }
}

/* ==== scale, clear, closure ====================================================================== */

void scale(double *c, int components, const double *factor)
{
    cplx f = get(factor, 0);
    for (int j = 0; j < components; j++) put(c, j, mul(f, get(c, j)));
}

void clear(double *c, int components)
{
    for (int j = 0; j < components; j++) put(c, j, ZERO);
}

void closure(double *amplitude, const double *l, const double *r, const double *component_factors,
             int components, const double *factor)
{
    cplx a = get(amplitude, 0), f = get(factor, 0);
    for (int j = 0; j < components; j++)
        a = add(a, mul(mul(get(l, j), get(r, j)), mul(get(component_factors, j), f)));
    put(amplitude, 0, a);
}

/* ==== kernel ===================================================================================== */

int kernel_call(symjit_kernel fn, double *output, int outputs, double *input, int inputs, symjit_slice *states)
{
    for (int j = 0; j < inputs; j++) { states[j].ptr = input + j; states[j].len = 1; }
    for (int j = 0; j < outputs; j++) { states[inputs + j].ptr = output + j; states[inputs + j].len = 1; }
    return fn(NULL, states, 0, NULL);
}

void kernel_store(double *out, const double *values, const int *outputs, int pairs, int direct, int mode,
                  const double *factor)
{
    cplx f = get(factor, 0);
    int as_is = direct && f.re == 1.0 && f.im == 0.0 && !signbit(f.im);  /* exactly 1 + 0i */
    for (int n = 0; n < pairs; n++) {
        int j = outputs[n];
        cplx pair = get(values, n);
        put(out, j, as_is ? pair : combine(get(out, j), mul(pair, f), mode));
    }
}

/* ==== reduction ================================================================================== */

double reduction(const double *amplitudes, const int *terms, const double *weights, int n, long numerator,
                 long denominator)
{
    double acc = 0.0;
    for (int t = 0; t < n; t++) {
        cplx a = get(amplitudes, terms[t]);
        acc = acc + weights[t] * fma(a.re, a.re, a.im * a.im);
    }
    return numerator == denominator ? acc : acc * (double)numerator / (double)denominator;
}
