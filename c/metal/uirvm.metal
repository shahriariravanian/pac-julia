/* uirvm.metal: c/uirvm.{h,c}, the interpreter of UIR v0.8, for Metal, in the GPU's mixed precision
 * with df64 for f64 (pacops.metal): f32 planes and current rows, df64 forms, amplitudes, outputs,
 * sources and kernels' wide values. One thread per point (uir_points, at the end).
 *
 * Differences from c/uirvm.c:
 *   - The program is one device buffer: a uir_header (written by compile_metal.jl's metal_program, its
 *     fields in METAL_TABLES order) of counts and byte offsets, then its tables. uir_view makes a thread's
 *     uir_program, its tables as device pointers. df64 tables (scales, factors, coefficients,
 *     crossing signs, weights) are float2 (hi, lo); rscales and rfactors f32.
 *   - The arena (uir_arena): planes f32, forms and amplitudes df64 (float2), plane-major with
 *     ld = the launch's points, as on CUDA.
 *   - Momenta and outputs have their own strides (uir_launch), so the host stages MadSpace's
 *     layout as it is and a call runs in chunks of points.
 *   - The arena is cleared by each thread for its point (no host memset).
 *   - Factors are passed by value (pacops.metal). No PAC_ACC64: the vertex runs' sums are f32.
 *
 * Keep in step with c/uirvm.c. Needs df64.h and pacops.metal before it; the program's generated
 * source (compile_metal.jl) defines uir_stub after it. */

#ifndef UIRVM_METAL
#define UIRVM_METAL

#define UIR_MAX_COMPONENTS 6
#define UIR_VERSION ((0u << 16) | 8u)

enum {
    UIR_V_WEYL_VECTOR_A, UIR_V_WEYL_VECTOR_B, UIR_V_TENSOR_VECTOR, UIR_V_VECTOR_WEDGE, UIR_V_THREE_VECTOR,
    UIR_P_WEYL_A, UIR_P_WEYL_B, UIR_P_VECTOR,
    UIR_CLOSURE, UIR_SCALE, UIR_CLEAR, UIR_KERNEL, UIR_KERNEL_DIRECT, UIR_SOURCE,
    UIR_CLEAR_AMPLITUDE, UIR_COPY, UIR_SCATTER, UIR_REDUCE,
    UIR_END = 31
};
enum { UIR_IN_L, UIR_IN_R, UIR_OUT, UIR_MOMENTUM_U, UIR_MOMENTUM_V, UIR_IN_SCALE, UIR_IN_FACTOR };

/* a row's fields (uirvm.h's; w: the row's two 32-bit halves) */
#define UIR_OP(w) ((w)[0] & 31u)
#define UIR_SET(w) (((w)[0] >> 6) & 1u)
#define UIR_FIRST(w) (((w)[0] >> 7) & 1u)
#define UIR_LAST(w) (((w)[0] >> 8) & 1u)
#define UIR_D(w) (((w)[0] >> 9) & 0x1fffu)
#define UIR_T(w) ((w)[0] >> 22)
#define UIR_L(w) ((w)[1] & 0x1fffu)
#define UIR_R(w) (((w)[1] >> 13) & 0x1fffu)
#define UIR_N(w) ((w)[1] & 0xffffu)
#define UIR_X(w) ((w)[1] >> 26)
#define UIR_E(w) (((w)[1] >> 16) & 0x3ffu)

/* the program buffer's header: counts, then the byte offset of each table in the buffer (every
 * offset a multiple of 8), as compile_metal.jl writes it (METAL_TABLES; 128 bytes with padding) */
struct uir_header {
    uint version, legs, forms, planes, amplitudes, outputs, words, has_crossing;
    int group_numerator, group_denominator;
    uint rows, edges, scales, factors, rscales, rfactors, form_first, term_leg, term_coefficient, crossing_leg,
        crossing_sign, list_first, list_factor, embedding_first, embedding_component, embedding_wave, embedding_factor,
        scaleset_first, scaleset_scale, instance, weights;
};

/* a thread's view of the program: its tables as device pointers */
struct uir_program {
    uint legs, forms, planes, amplitudes, outputs;
    bool has_crossing;
    long group_numerator, group_denominator;
    device const uint *rows, *edges;
    device const float2 *scales, *factors;              /* df64 (re, im) each */
    device const float *rscales, *rfactors;             /* f32 (re, im) each */
    device const uint *form_first, *term_leg;
    device const float2 *term_coefficient;
    device const uint *crossing_leg;
    device const float2 *crossing_sign;
    device const uint *list_first, *list_factor, *embedding_first;
    device const int *embedding_component, *embedding_wave;
    device const uint *embedding_factor, *scaleset_first, *scaleset_scale, *instance;
    device const float2 *weights;
};

#define UIR_TABLE(T, off) ((device const T *)(blob + (off)))

static inline uir_program uir_view(device const uchar *blob)
{
    device const uir_header *h = (device const uir_header *)blob;
    uir_program g;
    g.legs = h->legs; g.forms = h->forms; g.planes = h->planes; g.amplitudes = h->amplitudes; g.outputs = h->outputs;
    g.has_crossing = h->has_crossing != 0;
    g.group_numerator = h->group_numerator; g.group_denominator = h->group_denominator;
    g.rows = UIR_TABLE(uint, h->rows); g.edges = UIR_TABLE(uint, h->edges);
    g.scales = UIR_TABLE(float2, h->scales); g.factors = UIR_TABLE(float2, h->factors);
    g.rscales = UIR_TABLE(float, h->rscales); g.rfactors = UIR_TABLE(float, h->rfactors);
    g.form_first = UIR_TABLE(uint, h->form_first); g.term_leg = UIR_TABLE(uint, h->term_leg);
    g.term_coefficient = UIR_TABLE(float2, h->term_coefficient);
    g.crossing_leg = UIR_TABLE(uint, h->crossing_leg); g.crossing_sign = UIR_TABLE(float2, h->crossing_sign);
    g.list_first = UIR_TABLE(uint, h->list_first); g.list_factor = UIR_TABLE(uint, h->list_factor);
    g.embedding_first = UIR_TABLE(uint, h->embedding_first);
    g.embedding_component = UIR_TABLE(int, h->embedding_component); g.embedding_wave = UIR_TABLE(int, h->embedding_wave);
    g.embedding_factor = UIR_TABLE(uint, h->embedding_factor);
    g.scaleset_first = UIR_TABLE(uint, h->scaleset_first); g.scaleset_scale = UIR_TABLE(uint, h->scaleset_scale);
    g.instance = UIR_TABLE(uint, h->instance); g.weights = UIR_TABLE(float2, h->weights);
    return g;
}

/* the arena of one point: plane k (re, im) at planes[(2k) ld + i], [(2k + 1) ld + i]; form f's
 * component j at forms[(4f + j) ld + i]; amplitude k at amplitudes[(2k) ld + i], [(2k + 1) ld + i] */
struct uir_arena {
    device float2 *forms;
    device float2 *amplitudes;
    device float *planes;
    uint ld, i;
};

/* a table's complex value: f32 (rscales, rfactors) or df64 (scales, factors) */
static inline rcplx rgetd(device const float *p, uint j) { return rx(p[2 * j], p[2 * j + 1]); }
static inline cdf64 cgetd(device const float2 *p, uint j) { return cdd(dd_load(p[2 * j]), dd_load(p[2 * j + 1])); }

/* an edge-table entry's fields */
static inline uint edge_start(thread const uir_program &g, uint e) { return (g.edges[e] & 0x7fffu) * 2; }
static inline uint edge_form(thread const uir_program &g, uint e) { return (g.edges[e] >> 15) & 0xffu; }
static inline int edge_components(thread const uir_program &g, uint e) { return (int)((g.edges[e] >> 23) & 0xfu); }

/* ---- the arena ------------------------------------------------------------------------------- */

static inline device float *plane_re(thread const uir_arena &a, uint k) { return a.planes + (2 * k) * a.ld + a.i; }
static inline device float *plane_im(thread const uir_arena &a, uint k) { return a.planes + (2 * k + 1) * a.ld + a.i; }

static inline void load(thread const uir_arena &a, uint start, int n, thread float *v)
{
    for (int j = 0; j < n; j++) {
        v[2 * j] = *plane_re(a, start + j);
        v[2 * j + 1] = *plane_im(a, start + j);
    }
}

static inline void store(thread const uir_arena &a, uint start, int n, thread const float *v)
{
    for (int j = 0; j < n; j++) {
        *plane_re(a, start + j) = v[2 * j];
        *plane_im(a, start + j) = v[2 * j + 1];
    }
}

static inline rcplx load1(thread const uir_arena &a, uint k) { return rx(*plane_re(a, k), *plane_im(a, k)); }
static inline void store1(thread const uir_arena &a, uint k, rcplx z) { *plane_re(a, k) = z.re; *plane_im(a, k) = z.im; }
/* a component as df64 (exact), and a df64 value into a component (rounded once) */
static inline cdf64 load1d(thread const uir_arena &a, uint k) { return dof(load1(a, k)); }
static inline void store1d(thread const uir_arena &a, uint k, cdf64 z) { store1(a, k, rof(z)); }

/* momentum form f, in df64 and rounded to f32 (the three-vector vertex's) */
static inline void load_form(thread const uir_arena &a, uint f, thread df64 *q)
{
    for (int j = 0; j < 4; j++) q[j] = dd_load(a.forms[(4 * f + j) * a.ld + a.i]);
}
static inline void load_form_r(thread const uir_arena &a, uint f, thread float *q)
{
    for (int j = 0; j < 4; j++) q[j] = dd_to_f(dd_load(a.forms[(4 * f + j) * a.ld + a.i]));
}

static inline device float2 *amplitude_re(thread const uir_arena &a, uint k) { return a.amplitudes + (2 * k) * a.ld + a.i; }
static inline device float2 *amplitude_im(thread const uir_arena &a, uint k) { return a.amplitudes + (2 * k + 1) * a.ld + a.i; }

/* ---- the rows -------------------------------------------------------------------------------- */

static inline void vertex_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w,
                              thread rcplx *acc, uint op, int nl, int nr, int n)
{
    float l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], v[2 * UIR_MAX_COMPONENTS];
    load(a, edge_start(g, UIR_L(w)), nl, l);
    load(a, edge_start(g, UIR_R(w)), nr, r);
    switch (op) {
    case UIR_V_WEYL_VECTOR_A: vertex_weyl_vector_a(v, l, r); break;
    case UIR_V_WEYL_VECTOR_B: vertex_weyl_vector_b(v, l, r); break;
    case UIR_V_TENSOR_VECTOR: vertex_tensor_vector(v, l, r); break;
    case UIR_V_VECTOR_WEDGE: vertex_vector_wedge(v, l, r); break;
    default: {
        float u[4], q[4];
        load_form_r(a, edge_form(g, UIR_L(w)), u);
        load_form_r(a, edge_form(g, UIR_R(w)), q);
        vertex_three_vector(v, l, r, u, q);
    }
    }
    uint d = edge_start(g, UIR_D(w));
    rcplx s = rmul(rgetd(g.rscales, UIR_T(w)), rgetd(g.rfactors, UIR_X(w)));
    uint first = UIR_FIRST(w), set = UIR_SET(w);
    for (int j = 0; j < n; j++) {
        rcplx c = rmul(rget(v, j), s);
        if (!first) acc[j] = radd(acc[j], c);
        else if (set) acc[j] = c;
        else acc[j] = radd(load1(a, d + j), c);
    }
    if (UIR_LAST(w))
        for (int j = 0; j < n; j++) store1(a, d + j, acc[j]);
}

static inline void run_vertex(thread const uir_program &g, thread const uir_arena &a, device const uint *w, thread rcplx *acc)
{
    switch (UIR_OP(w)) {
    case UIR_V_WEYL_VECTOR_A: vertex_row(g, a, w, acc, UIR_V_WEYL_VECTOR_A, 2, 4, 2); break;
    case UIR_V_WEYL_VECTOR_B: vertex_row(g, a, w, acc, UIR_V_WEYL_VECTOR_B, 2, 4, 2); break;
    case UIR_V_TENSOR_VECTOR: vertex_row(g, a, w, acc, UIR_V_TENSOR_VECTOR, 6, 4, 4); break;
    case UIR_V_VECTOR_WEDGE: vertex_row(g, a, w, acc, UIR_V_VECTOR_WEDGE, 4, 4, 6); break;
    default: vertex_row(g, a, w, acc, UIR_V_THREE_VECTOR, 4, 4, 4); break;
    }
}

static inline void propagator_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int type, int n)
{
    float c[8], v[8];
    df64 q[4];
    uint d = edge_start(g, UIR_D(w));
    load(a, d, n, c);
    load_form(a, edge_form(g, UIR_D(w)), q);
    propagator(v, type, c, q, rgetd(g.rfactors, UIR_X(w)));
    store(a, d, n, v);
}

static inline void run_propagator(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    switch (UIR_OP(w)) {
    case UIR_P_WEYL_A: propagator_row(g, a, w, PROPAGATOR_WEYL_A, 2); break;
    case UIR_P_WEYL_B: propagator_row(g, a, w, PROPAGATOR_WEYL_B, 2); break;
    default: propagator_row(g, a, w, PROPAGATOR_VECTOR, 4); break;
    }
}

static inline void closure_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, uint first, int n)
{
    float l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], cf[2 * UIR_MAX_COMPONENTS];
    for (int j = 0; j < n; j++) {
        uint f = g.list_factor[first + j];
        cf[2 * j] = g.rfactors[2 * f];
        cf[2 * j + 1] = g.rfactors[2 * f + 1];
    }
    load(a, edge_start(g, UIR_L(w)), n, l);
    load(a, edge_start(g, UIR_R(w)), n, r);
    uint d = UIR_D(w);
    cdf64 amp = cdd(dd_load(*amplitude_re(a, d)), dd_load(*amplitude_im(a, d)));
    closure(amp, l, r, cf, n, rgetd(g.rfactors, UIR_X(w)));
    *amplitude_re(a, d) = dd_store(amp.re);
    *amplitude_im(a, d) = dd_store(amp.im);
}

static inline void run_closure(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    uint t = UIR_T(w), first = g.list_first[t];
    switch (g.list_first[t + 1] - first) {   /* 0 to 6 (compile refuses more), each a constant count */
    case 0: break;
    case 1: closure_row(g, a, w, first, 1); break;
    case 2: closure_row(g, a, w, first, 2); break;
    case 3: closure_row(g, a, w, first, 3); break;
    case 4: closure_row(g, a, w, first, 4); break;
    case 5: closure_row(g, a, w, first, 5); break;
    case 6: closure_row(g, a, w, first, 6); break;
    default: break;
    }
}

static inline void scale_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int n)
{
    float c[2 * UIR_MAX_COMPONENTS];
    uint d = edge_start(g, UIR_D(w));
    load(a, d, n, c);
    scale(c, n, rgetd(g.rfactors, UIR_X(w)));
    store(a, d, n, c);
}

static inline void run_scale(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    switch (UIR_N(w)) {
    case 0: break;
    case 1: scale_row(g, a, w, 1); break;
    case 2: scale_row(g, a, w, 2); break;
    case 3: scale_row(g, a, w, 3); break;
    case 4: scale_row(g, a, w, 4); break;
    case 5: scale_row(g, a, w, 5); break;
    case 6: scale_row(g, a, w, 6); break;
    default: break;
    }
}

static inline void run_clear(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    uint d = edge_start(g, UIR_D(w)), n = UIR_N(w);
    for (uint j = 0; j < n; j++) store1(a, d + j, RZERO);
}

/* kernel row w's input of code `code` (kind bits 0-2, imaginary part 3, n from 4), as df64 */
static inline df64 uir_input(thread const uir_program &g, thread const uir_arena &a, device const uint *w, uint code,
                             device const uint *scaleset)
{
    uint kind = code & 7, im = (code >> 3) & 1, n = code >> 4;
    df64 q[4];
    switch (kind) {
    case UIR_IN_L: return dd_f(im ? *plane_im(a, edge_start(g, UIR_L(w)) + n) : *plane_re(a, edge_start(g, UIR_L(w)) + n));
    case UIR_IN_R: return dd_f(im ? *plane_im(a, edge_start(g, UIR_R(w)) + n) : *plane_re(a, edge_start(g, UIR_R(w)) + n));
    case UIR_OUT: return dd_f(im ? *plane_im(a, edge_start(g, UIR_D(w)) + n) : *plane_re(a, edge_start(g, UIR_D(w)) + n));
    case UIR_MOMENTUM_U: load_form(a, edge_form(g, UIR_L(w)), q); return q[n];
    case UIR_MOMENTUM_V: load_form(a, edge_form(g, UIR_R(w)), q); return q[n];
    case UIR_IN_SCALE: return dd_load(g.scales[2 * scaleset[n]]);
    default: return dd_load(g.factors[2 * UIR_X(w) + im]);
    }
}

/* kernel row w's output pair into component `component` of its current, with the row's mode
 * (KERNEL_DIRECT: the pair as it is; else × factor), in df64, rounded once into the plane */
static inline void uir_output(thread const uir_program &g, thread const uir_arena &a, device const uint *w, uint component,
                              cdf64 pair)
{
    uint j = edge_start(g, UIR_D(w)) + component;
    cdf64 value = UIR_OP(w) == UIR_KERNEL_DIRECT ? pair : cdd_mul(pair, cgetd(g.factors, UIR_X(w)));
    store1d(a, j, combine(load1d(a, j), value, UIR_SET(w) ? MODE_SET : MODE_ADD));
}

/* the program's kernel stubs (its generated source defines it, after this file) */
static int uir_stub(uint stub, thread const uir_program &g, thread const uir_arena &a, device const uint *w,
                    device const uint *scaleset);

static inline int run_kernel(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    uint t = UIR_T(w), s = g.instance[2 * t], set = g.instance[2 * t + 1];
    return uir_stub(s, g, a, w, g.scaleset_scale + g.scaleset_first[set]);
}

static inline void source_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int family, int n)
{
    df64 q[4];
    cdf64 v[4];
    uint t = UIR_N(w), e = UIR_E(w), d = edge_start(g, UIR_D(w));
    int orientation = (t >> 3) & 1 ? ANTIPARTICLE : PARTICLE, h = (int)((t >> 4) & 7) - 2, ch = (int)((t >> 7) & 3) - 1;
    df64 mass = dd_load(g.scales[2 * UIR_T(w)]);
    load_form(a, edge_form(g, UIR_D(w)), q);
    if (e == 0x3ffu) {
        source(v, family, orientation, q, h, ch, mass, cgetd(g.factors, UIR_X(w)));
        for (int j = 0; j < n; j++) store1d(a, d + j, v[j]);
        return;
    }
    wavefunction(v, family, orientation, q, h, ch, mass);
    for (uint k = g.embedding_first[e]; k < g.embedding_first[e + 1]; k++) {
        int wave = g.embedding_wave[k];
        store1d(a, d + (uint)g.embedding_component[k], wave < 0 ? CZERO : cdd_mul(cgetd(g.factors, g.embedding_factor[k]), v[wave]));
    }
}

static inline void run_source(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    switch (UIR_N(w) & 7) {
    case 1: source_row(g, a, w, FAMILY_SCALAR, 1); break;
    case 2: source_row(g, a, w, FAMILY_WEYL, 2); break;
    case 3: source_row(g, a, w, FAMILY_DIRAC, 4); break;
    case 4: source_row(g, a, w, FAMILY_VECTOR, 4); break;
    default: break;
    }
}

static inline void copy_row_n(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int scatter_, int n)
{
    float s[2 * UIR_MAX_COMPONENTS], d[2 * UIR_MAX_COMPONENTS];
    uint dst = edge_start(g, UIR_D(w));
    load(a, edge_start(g, UIR_L(w)), n, s);
    if (scatter_) {
        load(a, dst, n, d);
        scatter(d, s, n, UIR_SET(w) ? MODE_SET : MODE_ADD, rgetd(g.rfactors, UIR_X(w)));
    } else {
        copy_row(d, s, n, rgetd(g.rfactors, UIR_X(w)));
    }
    store(a, dst, n, d);
}

static inline void run_copy(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int scatter_)
{
    switch (edge_components(g, UIR_D(w))) {
    case 1: copy_row_n(g, a, w, scatter_, 1); break;
    case 2: copy_row_n(g, a, w, scatter_, 2); break;
    case 3: copy_row_n(g, a, w, scatter_, 3); break;
    case 4: copy_row_n(g, a, w, scatter_, 4); break;
    case 5: copy_row_n(g, a, w, scatter_, 5); break;
    case 6: copy_row_n(g, a, w, scatter_, 6); break;
    default: break;
    }
}

/* REDUCE: output D += weight N × |amplitude T (× its scale X, unless none)|², in df64 */
static inline void run_reduce(thread const uir_program &g, thread const uir_arena &a, device const uint *w,
                              device float2 *out, uint out_step)
{
    uint t = UIR_T(w), x = UIR_X(w);
    uint d = UIR_D(w) * out_step;
    cdf64 amp = cdd(dd_load(*amplitude_re(a, t)), dd_load(*amplitude_im(a, t)));
    bool scaled = x != 0x3fu;
    cdf64 s = scaled ? cgetd(g.factors, x) : CZERO;
    out[d] = dd_store(reduce_term(dd_load(out[d]), amp, scaled, s, dd_load(g.weights[UIR_N(w)])));
}

static inline int uir_rows(thread const uir_program &g, thread const uir_arena &a, device float2 *out, uint out_step)
{
    rcplx acc[UIR_MAX_COMPONENTS];   /* the fused run's sum */
    for (uint k = 0;;) {
        device const uint *w = g.rows + 2 * k++;
        uint op = UIR_OP(w);
        if (op == UIR_END) break;
        if (op <= UIR_V_THREE_VECTOR) {
            run_vertex(g, a, w, acc);
        } else if (op <= UIR_P_VECTOR) {
            run_propagator(g, a, w);
        } else if (op == UIR_CLOSURE) {
            run_closure(g, a, w);
        } else if (op == UIR_SCALE) {
            run_scale(g, a, w);
        } else if (op == UIR_CLEAR) {
            run_clear(g, a, w);
        } else if (op == UIR_SOURCE) {
            run_source(g, a, w);
        } else if (op == UIR_CLEAR_AMPLITUDE) {
            *amplitude_re(a, UIR_D(w)) = float2(0.0f, 0.0f);
            *amplitude_im(a, UIR_D(w)) = float2(0.0f, 0.0f);
        } else if (op == UIR_COPY || op == UIR_SCATTER) {
            run_copy(g, a, w, op == UIR_SCATTER);
        } else if (op == UIR_REDUCE) {
            run_reduce(g, a, w, out, out_step);
        } else {
            int status = run_kernel(g, a, w);
            if (status != 0) return status;
        }
    }
    return 0;
}

/* ---- forms, outputs, group factor ------------------------------------------------------------ */

/* the forms of one point: component j of leg l at momenta[leg_step l + component_step j] (df64);
 * q_f = 0 + Σ coefficient × leg, in term order; with a crossing, leg l is sign_l × the caller's
 * leg crossing_leg_l */
static inline void uir_forms_layout(thread const uir_program &g, thread const uir_arena &a, device const float2 *momenta,
                                    uint leg_step, uint component_step)
{
    for (uint f = 0; f < g.forms; f++)
        for (uint j = 0; j < 4; j++) {
            df64 v = dd_f(0.0f);
            for (uint t = g.form_first[f]; t < g.form_first[f + 1]; t++) {
                uint leg = g.term_leg[t];
                df64 x = g.has_crossing
                    ? dd_load(g.crossing_sign[leg]) * dd_load(momenta[leg_step * g.crossing_leg[leg] + component_step * j])
                    : dd_load(momenta[leg_step * leg + component_step * j]);
                v = v + dd_load(g.term_coefficient[t]) * x;
            }
            a.forms[(4 * f + j) * a.ld + a.i] = dd_store(v);
        }
}

static inline void uir_group(thread const uir_program &g, device float2 *out, uint out_step)
{
    if (g.group_numerator == g.group_denominator) return;
    for (uint h = 0; h < g.outputs; h++)
        out[h * out_step] = dd_store(group(dd_load(out[h * out_step]), g.group_numerator, g.group_denominator));
}

/* one point: its arena cleared, the forms, its outputs at zero, the rows, the group factor.
 * Returns 0, or the first failing kernel's status. */
static inline int uir_run_layout(thread const uir_program &g, thread const uir_arena &a, device const float2 *momenta,
                                 uint leg_step, uint component_step, device float2 *out, uint out_step)
{
    for (uint k = 0; k < 2 * g.planes; k++) a.planes[k * a.ld + a.i] = 0.0f;
    for (uint k = 0; k < 2 * g.amplitudes; k++) a.amplitudes[k * a.ld + a.i] = float2(0.0f, 0.0f);
    uir_forms_layout(g, a, momenta, leg_step, component_step);
    for (uint h = 0; h < g.outputs; h++) out[h * out_step] = float2(0.0f, 0.0f);
    int status = uir_rows(g, a, out, out_step);
    if (status != 0) return status;
    uir_group(g, out, out_step);
    return 0;
}

/* ---- the compute function: one thread per point ----------------------------------------------
 * A launch of `points` points, the call's points first .. first + points: the momenta of point p
 * (of the call) at momenta[p + leg_step l + component_step j], its outputs at out[p + out_step h];
 * the arena holds the launch's points (ld = points). A failing point stores its status. */
struct uir_launch {
    uint points, first, leg_step, component_step, out_step;
};

kernel void uir_points(device const uchar *program [[buffer(0)]], constant uir_launch &launch [[buffer(1)]],
                       device const float2 *momenta [[buffer(2)]], device float2 *out [[buffer(3)]],
                       device float2 *forms [[buffer(4)]], device float2 *amplitudes [[buffer(5)]],
                       device float *planes [[buffer(6)]], device atomic_int *status [[buffer(7)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= launch.points) return;
    uir_program g = uir_view(program);
    uir_arena a;
    a.forms = forms; a.amplitudes = amplitudes; a.planes = planes; a.ld = launch.points; a.i = i;
    uint p = launch.first + i;
    int s = uir_run_layout(g, a, momenta + p, launch.leg_step, launch.component_step, out + p, launch.out_step);
    if (s != 0) atomic_store_explicit(status, s, memory_order_relaxed);
}

#endif
