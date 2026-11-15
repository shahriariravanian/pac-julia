// metal_probe_body.metal: f32 and df64 operations on (a.hi, a.lo, b.hi, b.lo) per element, as MSL
// (after <metal_stdlib> and c/metal/df64.h) and as C++ (after msl_shim.h and df64.h), for
// test/metal_probe.mm. out: NOPS (hi, lo) pairs per element.
#define NOPS 16

static inline void probe_one(uint i, device const float *in, device float *out)
{
    float a = in[4 * i], al = in[4 * i + 1], b = in[4 * i + 2], bl = in[4 * i + 3];
    df64 x = dd(a, al), y = dd(b, bl);
    df64 r[NOPS];
    r[0] = dd_f(a + b);
    r[1] = dd_f(a - b);
    r[2] = dd_f(a * b);
    r[3] = dd_f(a / b);
    r[4] = dd_f(fma(a, b, al));
    r[5] = dd_f(sqrt(fabs(a)));
    r[6] = x + y;
    r[7] = x - y;
    r[8] = x * y;
    r[9] = x / y;
    r[10] = dd_sqrt(dd_abs(x));
    r[11] = dd_floor(x);
    r[12] = dd_rint(x);
    r[13] = two_prod(a, b);
    r[14] = two_sum(a, b);
    r[15] = dd_fma(x, y, dd(al, 0.0f));
    for (uint k = 0; k < NOPS; k++) {
        out[2 * (NOPS * i + k)] = r[k].hi;
        out[2 * (NOPS * i + k) + 1] = r[k].lo;
    }
}

kernel void probe(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
                  constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n) probe_one(i, in, out);
}
