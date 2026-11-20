// metal_shim_ops.cpp: c/metal/df64.h and pacops.metal on the CPU, through msl_shim.h, for
// test/test_metal_shim.jl (built: c++ -std=c++17 -O1 -ffp-contract=off -fPIC -shared -Wno-attributes
// -I c/metal). Doubles in, doubles out: a double becomes a df64 (hi = the nearest float, lo = the
// rest, as the Metal host stages them), a df64 a double (hi + lo, exact).

#include "msl_shim.h"
#include "df64.h"
#include "pacops.metal"

static df64 todd(double x)
{
    float h = (float)x;
    return dd(h, std::isfinite(h) ? (float)(x - (double)h) : 0.0f);
}
static double fromdd(df64 x) { return (double)x.hi + (double)x.lo; }

extern "C" {

void shim_flush_denormals(void) { msl_shim_flush_denormals(); }

// op: 0 +, 1 -, 2 ×, 3 ÷, 4 sqrt(a), 5 a < b, 6 floor(a), 7 rint(a), 8 max(a, b), 9 min(a, b)
double shim_dd(int op, double a, double b)
{
    df64 x = todd(a), y = todd(b);
    switch (op) {
    case 0: return fromdd(x + y);
    case 1: return fromdd(x - y);
    case 2: return fromdd(x * y);
    case 3: return fromdd(x / y);
    case 4: return fromdd(dd_sqrt(x));
    case 5: return x < y ? 1.0 : 0.0;
    case 6: return fromdd(dd_floor(x));
    case 7: return fromdd(dd_rint(x));
    case 8: return fromdd(dd_max(x, y));
    default: return fromdd(dd_min(x, y));
    }
}

// the wavefunction of a source: out[2j], out[2j + 1] = component j (re, im); returns the components
int shim_wavefunction(double *out, int family, int orientation, const double *k, int h, int ch, double mass)
{
    df64 kk[4] = {todd(k[0]), todd(k[1]), todd(k[2]), todd(k[3])};
    cdf64 w[4];
    int n = wavefunction(w, family, orientation, kk, h, ch, todd(mass));
    for (int j = 0; j < n; j++) {
        out[2 * j] = fromdd(w[j].re);
        out[2 * j + 1] = fromdd(w[j].im);
    }
    return n;
}

// a vertex's o (f32: the operands rounded to float); returns its components
int shim_vertex(double *out, int type, const double *l, const double *r, const double *u, const double *v)
{
    float lf[12], rf[12], uf[4], vf[4], o[12];
    for (int j = 0; j < 12; j++) { lf[j] = (float)l[j]; rf[j] = (float)r[j]; }
    for (int j = 0; j < 4; j++) { uf[j] = (float)u[j]; vf[j] = (float)v[j]; }
    int n;
    switch (type) {
    case VERTEX_WEYL_VECTOR_A: vertex_weyl_vector_a(o, lf, rf); n = 2; break;
    case VERTEX_WEYL_VECTOR_B: vertex_weyl_vector_b(o, lf, rf); n = 2; break;
    case VERTEX_TENSOR_VECTOR: vertex_tensor_vector(o, lf, rf); n = 4; break;
    case VERTEX_VECTOR_WEDGE: vertex_vector_wedge(o, lf, rf); n = 6; break;
    default: vertex_three_vector(o, lf, rf, uf, vf); n = 4; break;
    }
    for (int j = 0; j < 2 * n; j++) out[j] = o[j];
    return n;
}

// a propagator: the current c (f32), the momentum k (df64), the factor (f32); returns its components
int shim_propagator(double *out, int type, const double *c, const double *k, const double *factor)
{
    float cf[8], o[8];
    for (int j = 0; j < 8; j++) cf[j] = (float)c[j];
    df64 kk[4] = {todd(k[0]), todd(k[1]), todd(k[2]), todd(k[3])};
    propagator(o, type, cf, kk, rx((float)factor[0], (float)factor[1]));
    int n = type == PROPAGATOR_VECTOR ? 4 : 2;
    for (int j = 0; j < 2 * n; j++) out[j] = o[j];
    return n;
}

}
