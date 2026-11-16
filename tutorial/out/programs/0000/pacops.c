/* The wavefunctions (wavefunctions.jl) and the instructions (operations.jl) in plain C: the same
 * operations in the same order, so the values are bit for bit the Julia runner's (and RustiCol's).
 * The layout and the conventions are in pacops.h. Compile with -ffp-contract=off. */

#include <math.h>
#include <stdlib.h>
#include "pacops.h"

/* ==== wavefunctions ============================================================================ */

PAC_FN void scalar(double *out) { put(out, 0, cx(1.0, 0.0)); }

/* ---- Weyl spinors (ext_quark_weyl_array, ext_antiquark_weyl_array) ---- */

PAC_FN void weyl_particle(double *out, const double *k, int h, int ch)
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

PAC_FN void weyl_antiparticle(double *out, const double *k, int h, int ch)
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

PAC_FN void quark_dirac(double *out, const double *k, int h)
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

PAC_FN void antiquark_dirac(double *out, const double *k, int h)
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

PAC_FN static void dirac_massive(double *out, const double *k, int h, double mass, int anti)
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

PAC_FN void quark_dirac_massive(double *out, const double *k, int h, double mass)
{
    if (fabs(mass) < 1.0e-8) quark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 0);
}

PAC_FN void antiquark_dirac_massive(double *out, const double *k, int h, double mass)
{
    if (fabs(mass) < 1.0e-8) antiquark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 1);
}

/* ---- massless vectors: gluon, photon (ext_gluon) ---- */

PAC_FN void gluon(double *out, const double *k, int h)
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

PAC_FN void massive_vector(double *out, const double *k, int h, double mass)
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
PAC_FN int wavefunction(double *out, int family, int orientation, const double *k, int h, int ch, double mass)
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

PAC_FN int source(double *out, int family, int orientation, const double *k, int h, int ch, double mass,
           const double *factor)
{
    int n = wavefunction(out, family, orientation, k, h, ch, mass);
    cplx f = get(factor, 0);
    for (int j = 0; j < n; j++) put(out, j, mul(f, get(out, j)));
    return n;
}

PAC_FN void source_embedded(double *out, int family, int orientation, const double *k, int h, int ch, double mass,
                     const int *components, const int *waves, const double *factors, int n)
{
    double w[8];
    wavefunction(w, family, orientation, k, h, ch, mass);
    for (int e = 0; e < n; e++)
        put(out, components[e], waves[e] < 0 ? ZERO : mul(get(factors, e), get(w, waves[e])));
}

/* ==== vertex ===================================================================================== */

/* l Weyl, r vector → Weyl:
 *   o[0] = -i l[0] r[3] - i l[1] r[1] + l[1] r[2] + i l[0] r[0]
 *   o[1] = -l[0] r[2] - i l[0] r[1] + i l[1] r[0] + i l[1] r[3] */
PAC_FN void vertex_weyl_vector_a(pac_real *out, const pac_real *l, const pac_real *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o0 = radd(radd(rsub(rneg(rmul_i(rmul(l0, r3))), rmul_i(rmul(l1, r1))), rmul(l1, r2)), rmul_i(rmul(l0, r0)));
    rcplx o1 = radd(radd(rsub(rneg(rmul(l0, r2)), rmul_i(rmul(l0, r1))), rmul_i(rmul(l1, r0))), rmul_i(rmul(l1, r3)));
    rput2(out, o0, o1);
}

/* l Weyl, r vector → Weyl:
 *   o[0] = -l[1] r[2] + i l[0] r[0] + i l[0] r[3] + i l[1] r[1]
 *   o[1] = -i l[1] r[3] + l[0] r[2] + i l[0] r[1] + i l[1] r[0] */
PAC_FN void vertex_weyl_vector_b(pac_real *out, const pac_real *l, const pac_real *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o0 = radd(radd(radd(rneg(rmul(l1, r2)), rmul_i(rmul(l0, r0))), rmul_i(rmul(l0, r3))), rmul_i(rmul(l1, r1)));
    rcplx o1 = radd(radd(radd(rneg(rmul_i(rmul(l1, r3))), rmul(l0, r2)), rmul_i(rmul(l0, r1))), rmul_i(rmul(l1, r0)));
    rput2(out, o0, o1);
}

/* l antisymmetric tensor, r vector → vector:
 *   o[0] = l[0] r[1] + l[1] r[2] + l[2] r[3]
 *   o[1] = l[0] r[0] + l[3] r[2] + l[4] r[3]
 *   o[2] = l[1] r[0] - l[3] r[1] + l[5] r[3]
 *   o[3] = l[2] r[0] - l[4] r[1] - l[5] r[2] */
PAC_FN void vertex_tensor_vector(pac_real *out, const pac_real *l, const pac_real *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1), l2 = rget(l, 2), l3 = rget(l, 3), l4 = rget(l, 4), l5 = rget(l, 5);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rput4(out,
          radd(radd(rmul(l0, r1), rmul(l1, r2)), rmul(l2, r3)),
          radd(radd(rmul(l0, r0), rmul(l3, r2)), rmul(l4, r3)),
          radd(rsub(rmul(l1, r0), rmul(l3, r1)), rmul(l5, r3)),
          rsub(rsub(rmul(l2, r0), rmul(l4, r1)), rmul(l5, r2)));
}

/* l vector, r vector → antisymmetric tensor:
 *   o = (l[0] r[1] - l[1] r[0], l[0] r[2] - l[2] r[0], l[0] r[3] - l[3] r[0],
 *        l[1] r[2] - l[2] r[1], l[1] r[3] - l[3] r[1], l[2] r[3] - l[3] r[2]) */
PAC_FN void vertex_vector_wedge(pac_real *out, const pac_real *l, const pac_real *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1), l2 = rget(l, 2), l3 = rget(l, 3);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o[6] = {rsub(rmul(l0, r1), rmul(l1, r0)), rsub(rmul(l0, r2), rmul(l2, r0)), rsub(rmul(l0, r3), rmul(l3, r0)),
                  rsub(rmul(l1, r2), rmul(l2, r1)), rsub(rmul(l1, r3), rmul(l3, r1)), rsub(rmul(l2, r3), rmul(l3, r2))};
    for (int j = 0; j < 6; j++) rput(out, j, o[j]);
}

/* a·b of two complex vectors; of a complex vector and a momentum */
PAC_FN static rcplx dot(const pac_real *a, const pac_real *b)
{
    return rsub(rsub(rsub(rmul(rget(a, 0), rget(b, 0)), rmul(rget(a, 1), rget(b, 1))), rmul(rget(a, 2), rget(b, 2))),
                rmul(rget(a, 3), rget(b, 3)));
}

PAC_FN static rcplx dot_real(const pac_real *a, const pac_real *b)
{
    return rsub(rsub(rsub(rmulr(rget(a, 0), b[0]), rmulr(rget(a, 1), b[1])), rmulr(rget(a, 2), b[2])), rmulr(rget(a, 3), b[3]));
}

/* l vector, r vector → vector, the colour-ordered three-gluon vertex; u = momenta[0],
 * v = momenta[1]: o[j] = (l·r)(u[j] - v[j]) + 2((l·v) r[j] - (r·u) l[j]) */
PAC_FN void vertex_three_vector(pac_real *out, const pac_real *l, const pac_real *r, const pac_real *u, const pac_real *v)
{
    rcplx lr = dot(l, r), lv = dot_real(l, v), ru = dot_real(r, u);
    rcplx o[4];
    for (int j = 0; j < 4; j++)
        o[j] = radd(rmulr(lr, u[j] - v[j]), rmulr(rsub(rmul(lv, rget(r, j)), rmul(ru, rget(l, j))), 2));
    for (int j = 0; j < 4; j++) rput(out, j, o[j]);
}

#ifndef PAC_MIXED
/* the vertex instruction: out[j] (mode)= o[j] × (coupling × factor) */
PAC_FN void vertex(double *out, int type, int mode, const double *l, const double *r, const double *u,
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
#endif

/* ==== propagator: out[j] = o[j] × factor / (k·k) ================================================= */

/* 1 / (k·k) in double; the propagators' numerators take k rounded to pac_real (kr) */
PAC_FN static double reciprocal_norm(const double *k) { return 1.0 / (((k[0] * k[0] - k[1] * k[1]) - k[2] * k[2]) - k[3] * k[3]); }

/* c Weyl:
 *   o[0] = -i c[0] k[3] - i c[1] k[1] + c[1] k[2] + i c[0] k[0]
 *   o[1] = -c[0] k[2] - i c[0] k[1] + i c[1] k[0] + i c[1] k[3] */
PAC_FN void propagator_weyl_a(pac_real *out, const pac_real *c, const double *k, const pac_real *factor)
{
    pac_real kr[4] = {(pac_real)k[0], (pac_real)k[1], (pac_real)k[2], (pac_real)k[3]};
    rcplx c0 = rget(c, 0), c1 = rget(c, 1);
    rcplx s = rmulr(rget(factor, 0), (pac_real)reciprocal_norm(k));
    rcplx o0 = radd(radd(rsub(rneg(rmul_i(rmulr(c0, kr[3]))), rmul_i(rmulr(c1, kr[1]))), rmulr(c1, kr[2])), rmul_i(rmulr(c0, kr[0])));
    rcplx o1 = radd(radd(rsub(rneg(rmulr(c0, kr[2])), rmul_i(rmulr(c0, kr[1]))), rmul_i(rmulr(c1, kr[0]))), rmul_i(rmulr(c1, kr[3])));
    rput2(out, rmul(o0, s), rmul(o1, s));
}

/* c Weyl:
 *   o[0] = -c[1] k[2] + i c[0] k[0] + i c[0] k[3] + i c[1] k[1]
 *   o[1] = -i c[1] k[3] + c[0] k[2] + i c[0] k[1] + i c[1] k[0] */
PAC_FN void propagator_weyl_b(pac_real *out, const pac_real *c, const double *k, const pac_real *factor)
{
    pac_real kr[4] = {(pac_real)k[0], (pac_real)k[1], (pac_real)k[2], (pac_real)k[3]};
    rcplx c0 = rget(c, 0), c1 = rget(c, 1);
    rcplx s = rmulr(rget(factor, 0), (pac_real)reciprocal_norm(k));
    rcplx o0 = radd(radd(radd(rneg(rmulr(c1, kr[2])), rmul_i(rmulr(c0, kr[0]))), rmul_i(rmulr(c0, kr[3]))), rmul_i(rmulr(c1, kr[1])));
    rcplx o1 = radd(radd(radd(rneg(rmul_i(rmulr(c1, kr[3]))), rmulr(c0, kr[2])), rmul_i(rmulr(c0, kr[1]))), rmul_i(rmulr(c1, kr[0])));
    rput2(out, rmul(o0, s), rmul(o1, s));
}

/* c vector, Feynman gauge: o[j] = -i c[j] */
PAC_FN void propagator_vector(pac_real *out, const pac_real *c, const double *k, const pac_real *factor)
{
    rcplx s = rmulr(rmul(rget(factor, 0), rx(0, -1)), (pac_real)reciprocal_norm(k));
    for (int j = 0; j < 4; j++) rput(out, j, rmul(rget(c, j), s));
}

PAC_FN void propagator(pac_real *out, int type, const pac_real *c, const double *k, const pac_real *factor)
{
    switch (type) {
    case PROPAGATOR_WEYL_A: propagator_weyl_a(out, c, k, factor); break;
    case PROPAGATOR_WEYL_B: propagator_weyl_b(out, c, k, factor); break;
    case PROPAGATOR_VECTOR: propagator_vector(out, c, k, factor); break;
    }
}

/* ==== scale, clear, closure ====================================================================== */

PAC_FN void scale(pac_real *c, int components, const pac_real *factor)
{
    rcplx f = rget(factor, 0);
    for (int j = 0; j < components; j++) rput(c, j, rmul(f, rget(c, j)));
}

PAC_FN void clear(pac_real *c, int components)
{
    for (int j = 0; j < components; j++) rput(c, j, RZERO);
}

PAC_FN void closure(pac_amp *amplitude, const pac_real *l, const pac_real *r, const pac_real *component_factors,
             int components, const pac_real *factor)
{
    pac_amp re = amplitude[0], im = amplitude[1];
    rcplx f = rget(factor, 0);
    for (int j = 0; j < components; j++) {
        rcplx t = rmul(rmul(rget(l, j), rget(r, j)), rmul(rget(component_factors, j), f));
        re = re + (pac_amp)t.re;
        im = im + (pac_amp)t.im;
    }
    amplitude[0] = re;
    amplitude[1] = im;
}

/* ==== kernel ===================================================================================== */

int kernel_call(symjit_kernel fn, double *output, int outputs, double *input, int inputs, symjit_slice *states)
{
    for (int j = 0; j < inputs; j++) { states[j].ptr = input + j; states[j].len = 1; }
    for (int j = 0; j < outputs; j++) { states[inputs + j].ptr = output + j; states[inputs + j].len = 1; }
    return fn(NULL, states, 0, NULL);
}

PAC_FN void kernel_store(double *out, const double *values, const int *outputs, int pairs, int direct, int mode,
                  const double *factor)
{
    cplx f = get(factor, 0);
    for (int n = 0; n < pairs; n++) {
        int j = outputs[n];
        cplx pair = get(values, n);
        put(out, j, combine(get(out, j), direct ? pair : mul(pair, f), mode));
    }
}

/* ==== copy, scatter ============================================================================== */

PAC_FN void copy_row(pac_real *out, const pac_real *src, int n, const pac_real *factor)
{
    rcplx f = rget(factor, 0);
    int zero = f.re == 0 && f.im == 0;
    for (int j = 0; j < n; j++) rput(out, j, zero ? RZERO : rmul(rget(src, j), f));
}

PAC_FN static inline rcplx scatter_value(rcplx v, rcplx f)
{
    if (f.im == 0) {
        if (f.re == 1) return v;
        if (f.re == -1) return rneg(v);
        return rx(v.re * f.re, v.im * f.re);
    }
    if (f.re == 0) return rx(-v.im * f.im, v.re * f.im);
    return rmul(v, f);
}

PAC_FN void scatter(pac_real *out, const pac_real *src, int n, int mode, const pac_real *factor)
{
    rcplx f = rget(factor, 0);
    for (int j = 0; j < n; j++) rput(out, j, rcombine(rget(out, j), scatter_value(rget(src, j), f), mode));
}

/* ==== reduction ================================================================================== */

PAC_FN double reduce_term(double acc, const pac_amp *amplitude, const double *scale, double weight)
{
    cplx a = cx(amplitude[0], amplitude[1]);
    if (scale) a = mul(a, get(scale, 0));
    return acc + weight * fma(a.re, a.re, a.im * a.im);
}

PAC_FN double group(double x, long numerator, long denominator)
{
    return numerator == denominator ? x : x * (double)numerator / (double)denominator;
}

PAC_FN double reduction(const double *amplitudes, const int *terms, const double *weights, int n, long numerator,
                 long denominator)
{
    double acc = 0.0;
    for (int t = 0; t < n; t++) {
        cplx a = get(amplitudes, terms[t]);
        acc = acc + weights[t] * fma(a.re, a.re, a.im * a.im);
    }
    return numerator == denominator ? acc : acc * (double)numerator / (double)denominator;
}

PAC_FN double reduction_single(const double *amplitudes, int term, double weight, long numerator, long denominator)
{
    cplx a = get(amplitudes, term);
    double acc = 0.0 + weight * fma(a.re, a.re, a.im * a.im);  /* as reduction(): 0.0 + the first term */
    return numerator == denominator ? acc : acc * (double)numerator / (double)denominator;
}
