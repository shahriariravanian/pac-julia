/* uirvm: the interpreter of UIR v0.7 (uirvm.h). The arithmetic is pacops' (pacops.h), on the
 * operands loaded into local arrays, so the values are the straight-line C's bit for bit; only the
 * storage differs. A row is its word (`w`, uirvm.h), its fields read by the UIR_ macros where used. A program's runner.c includes this file (one translation unit, as nvcc wants it)
 * and defines uir_stub, which uses uir_input and uir_output. */

#include "pacops.h"
#include "uirvm.h"

#define UIR_MAX_COMPONENTS 6

/* inlined at every call: with the component counts constant there, the operand loops unroll and the
 * operand arrays become registers (no local memory on a GPU) */
#if defined(__CUDACC__)
#define UIR_INLINE __forceinline__
#elif defined(__GNUC__)
#define UIR_INLINE inline __attribute__((always_inline))
#else
#define UIR_INLINE inline
#endif

/* an edge-table entry's fields */
PAC_FN static inline uint32_t edge_start(const uir_program *g, uint32_t e) { return (g->edges[e] & 0x7fff) * 2; }
PAC_FN static inline uint32_t edge_form(const uir_program *g, uint32_t e) { return (g->edges[e] >> 15) & 0xff; }
PAC_FN static inline int edge_components(const uir_program *g, uint32_t e) { return (g->edges[e] >> 23) & 0xf; }

/* ---- the arena ------------------------------------------------------------------------------- */

PAC_FN static inline double *plane_re(const uir_arena *a, uint32_t k) { return a->planes + (2 * (size_t)k) * a->ld + a->i; }
PAC_FN static inline double *plane_im(const uir_arena *a, uint32_t k) { return a->planes + (2 * (size_t)k + 1) * a->ld + a->i; }

/* n components from plane `start` into v (re, im each), and back */
PAC_FN static UIR_INLINE void load(const uir_arena *a, uint32_t start, int n, double *v)
{
    for (int j = 0; j < n; j++) {
        v[2 * j] = *plane_re(a, start + j);
        v[2 * j + 1] = *plane_im(a, start + j);
    }
}

PAC_FN static UIR_INLINE void store(const uir_arena *a, uint32_t start, int n, const double *v)
{
    for (int j = 0; j < n; j++) {
        *plane_re(a, start + j) = v[2 * j];
        *plane_im(a, start + j) = v[2 * j + 1];
    }
}

PAC_FN static UIR_INLINE cplx load1(const uir_arena *a, uint32_t k) { return cx(*plane_re(a, k), *plane_im(a, k)); }
PAC_FN static UIR_INLINE void store1(const uir_arena *a, uint32_t k, cplx z) { *plane_re(a, k) = z.re; *plane_im(a, k) = z.im; }

/* momentum form f into q[4] */
PAC_FN static UIR_INLINE void load_form(const uir_arena *a, uint32_t f, double *q)
{
    for (int j = 0; j < 4; j++) q[j] = a->forms[(4 * (size_t)f + j) * a->ld + a->i];
}

PAC_FN static inline double *amplitude_re(const uir_arena *a, uint32_t k) { return a->amplitudes + (2 * (size_t)k) * a->ld + a->i; }
PAC_FN static inline double *amplitude_im(const uir_arena *a, uint32_t k) { return a->amplitudes + (2 * (size_t)k + 1) * a->ld + a->i; }

/* ---- the rows -------------------------------------------------------------------------------- */

/* A vertex row of opcode `op`: l has nl components, r nr, the vertex writes n (the opcode's shape,
 * which validate checks); its values × (coupling × factor) into the run's sum `acc`, stored at LAST.
 * Called with constant op, nl, nr, n (run_vertex), so every loop has a constant bound. */
PAC_FN static UIR_INLINE void vertex_row(const uir_program *g, const uir_arena *a, const uint32_t *w, cplx *acc,
                                         uint32_t op, int nl, int nr, int n)
{
    double l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], v[2 * UIR_MAX_COMPONENTS];
    load(a, edge_start(g, UIR_L(w)), nl, l);
    load(a, edge_start(g, UIR_R(w)), nr, r);
    switch (op) {
    case UIR_V_WEYL_VECTOR_A: vertex_weyl_vector_a(v, l, r); break;
    case UIR_V_WEYL_VECTOR_B: vertex_weyl_vector_b(v, l, r); break;
    case UIR_V_TENSOR_VECTOR: vertex_tensor_vector(v, l, r); break;
    case UIR_V_VECTOR_WEDGE: vertex_vector_wedge(v, l, r); break;
    default: {
        double u[4], q[4];
        load_form(a, edge_form(g, UIR_L(w)), u);
        load_form(a, edge_form(g, UIR_R(w)), q);
        vertex_three_vector(v, l, r, u, q);
    }
    }
    uint32_t d = edge_start(g, UIR_D(w));
    cplx s = mul(get(g->scales, UIR_T(w)), get(g->factors, UIR_X(w)));
    uint32_t first = UIR_FIRST(w), set = UIR_SET(w);
    for (int j = 0; j < n; j++) {
        cplx c = mul(get(v, j), s);
        if (!first) acc[j] = add(acc[j], c);
        else if (set) acc[j] = c;
        else acc[j] = add(load1(a, d + j), c);
    }
    if (UIR_LAST(w))
        for (int j = 0; j < n; j++) store1(a, d + j, acc[j]);
}

PAC_FN static UIR_INLINE void run_vertex(const uir_program *g, const uir_arena *a, const uint32_t *w, cplx *acc)
{
    switch (UIR_OP(w)) {
    case UIR_V_WEYL_VECTOR_A: vertex_row(g, a, w, acc, UIR_V_WEYL_VECTOR_A, 2, 4, 2); break;
    case UIR_V_WEYL_VECTOR_B: vertex_row(g, a, w, acc, UIR_V_WEYL_VECTOR_B, 2, 4, 2); break;
    case UIR_V_TENSOR_VECTOR: vertex_row(g, a, w, acc, UIR_V_TENSOR_VECTOR, 6, 4, 4); break;
    case UIR_V_VECTOR_WEDGE: vertex_row(g, a, w, acc, UIR_V_VECTOR_WEDGE, 4, 4, 6); break;
    default: vertex_row(g, a, w, acc, UIR_V_THREE_VECTOR, 4, 4, 4); break;
    }
}

/* a propagator row of type `type` on a current of n components: the current's new values */
PAC_FN static UIR_INLINE void propagator_row(const uir_program *g, const uir_arena *a, const uint32_t *w, int type, int n)
{
    double c[8], q[4], v[8];
    uint32_t d = edge_start(g, UIR_D(w));
    load(a, d, n, c);
    load_form(a, edge_form(g, UIR_D(w)), q);
    propagator(v, type, c, q, g->factors + 2 * (size_t)UIR_X(w));
    store(a, d, n, v);
}

PAC_FN static UIR_INLINE void run_propagator(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    switch (UIR_OP(w)) {
    case UIR_P_WEYL_A: propagator_row(g, a, w, PROPAGATOR_WEYL_A, 2); break;
    case UIR_P_WEYL_B: propagator_row(g, a, w, PROPAGATOR_WEYL_B, 2); break;
    default: propagator_row(g, a, w, PROPAGATOR_VECTOR, 4); break;
    }
}

/* a closure row of n components: amplitude += Σ l[j] r[j] × component factor j × factor */
PAC_FN static UIR_INLINE void closure_row(const uir_program *g, const uir_arena *a, const uint32_t *w, uint32_t first, int n)
{
    double l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], cf[2 * UIR_MAX_COMPONENTS], amp[2];
    for (int j = 0; j < n; j++) {
        uint32_t f = g->list_factor[first + j];
        cf[2 * j] = g->factors[2 * (size_t)f];
        cf[2 * j + 1] = g->factors[2 * (size_t)f + 1];
    }
    load(a, edge_start(g, UIR_L(w)), n, l);
    load(a, edge_start(g, UIR_R(w)), n, r);
    uint32_t d = UIR_D(w);
    amp[0] = *amplitude_re(a, d);
    amp[1] = *amplitude_im(a, d);
    closure(amp, l, r, cf, n, g->factors + 2 * (size_t)UIR_X(w));
    *amplitude_re(a, d) = amp[0];
    *amplitude_im(a, d) = amp[1];
}

PAC_FN static UIR_INLINE void run_closure(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    uint32_t t = UIR_T(w), first = g->list_first[t];
    switch (g->list_first[t + 1] - first) {   /* 0 to 6 (compile refuses more), each a constant count */
    case 0: break;
    case 1: closure_row(g, a, w, first, 1); break;
    case 2: closure_row(g, a, w, first, 2); break;
    case 3: closure_row(g, a, w, first, 3); break;
    case 4: closure_row(g, a, w, first, 4); break;
    case 5: closure_row(g, a, w, first, 5); break;
    case 6: closure_row(g, a, w, first, 6); break;
    default: break;   /* refused by compile (vm_unsupported) */
    }
}

/* a scale row of n components: c[j] = factor × c[j] */
PAC_FN static UIR_INLINE void scale_row(const uir_program *g, const uir_arena *a, const uint32_t *w, int n)
{
    double c[2 * UIR_MAX_COMPONENTS];
    uint32_t d = edge_start(g, UIR_D(w));
    load(a, d, n, c);
    scale(c, n, g->factors + 2 * (size_t)UIR_X(w));
    store(a, d, n, c);
}

PAC_FN static UIR_INLINE void run_scale(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    switch (UIR_N(w)) {   /* 0 to 6 (compile refuses more), each a constant count */
    case 0: break;
    case 1: scale_row(g, a, w, 1); break;
    case 2: scale_row(g, a, w, 2); break;
    case 3: scale_row(g, a, w, 3); break;
    case 4: scale_row(g, a, w, 4); break;
    case 5: scale_row(g, a, w, 5); break;
    case 6: scale_row(g, a, w, 6); break;
    default: break;   /* refused by compile (vm_unsupported) */
    }
}

PAC_FN static UIR_INLINE void run_clear(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    uint32_t d = edge_start(g, UIR_D(w)), n = UIR_N(w);
    for (uint32_t j = 0; j < n; j++) store1(a, d + j, ZERO);
}

/* kernel row w's input of code `code` (kind bits 0-2, imaginary part 3, n from 4; uirvm.h) */
PAC_FN static UIR_INLINE double uir_input(const uir_program *g, const uir_arena *a, const uint32_t *w, uint32_t code,
                                          const uint32_t *scaleset)
{
    uint32_t kind = code & 7, im = (code >> 3) & 1, n = code >> 4;
    double q[4];
    switch (kind) {
    case UIR_IN_L: return im ? *plane_im(a, edge_start(g, UIR_L(w)) + n) : *plane_re(a, edge_start(g, UIR_L(w)) + n);
    case UIR_IN_R: return im ? *plane_im(a, edge_start(g, UIR_R(w)) + n) : *plane_re(a, edge_start(g, UIR_R(w)) + n);
    case UIR_OUT: return im ? *plane_im(a, edge_start(g, UIR_D(w)) + n) : *plane_re(a, edge_start(g, UIR_D(w)) + n);
    case UIR_MOMENTUM_U: load_form(a, edge_form(g, UIR_L(w)), q); return q[n];
    case UIR_MOMENTUM_V: load_form(a, edge_form(g, UIR_R(w)), q); return q[n];
    case UIR_IN_SCALE: return g->scales[2 * (size_t)scaleset[n]];
    default: return g->factors[2 * (size_t)UIR_X(w) + im];
    }
}

/* kernel row w's output pair into component `component` of its current (KERNEL_DIRECT: as it is;
 * else × factor, with the mode); the pairs in order */
PAC_FN static UIR_INLINE void uir_output(const uir_program *g, const uir_arena *a, const uint32_t *w, uint32_t component,
                                         cplx pair)
{
    uint32_t j = edge_start(g, UIR_D(w)) + component;
    store1(a, j, UIR_OP(w) == UIR_KERNEL_DIRECT ? pair
                 : combine(load1(a, j), mul(pair, get(g->factors, UIR_X(w))), UIR_SET(w) ? MODE_SET : MODE_ADD));
}

/* a kernel row: its binding instance's stub and scale set, run by the program's uir_stub */
PAC_FN static UIR_INLINE int run_kernel(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    uint32_t t = UIR_T(w), s = g->instance[2 * (size_t)t], set = g->instance[2 * (size_t)t + 1];
    return uir_stub(s, g, a, w, g->scaleset_scale + g->scaleset_first[set]);
}

/* a source row of family `family` (pacops' number: the physical type's − 1), n components: factor ×
 * the wavefunction at its edge's form, into its edge's current */
PAC_FN static UIR_INLINE void source_row(const uir_program *g, const uir_arena *a, const uint32_t *w, int family, int n)
{
    double q[4], v[8];
    uint32_t t = UIR_N(w);
    load_form(a, edge_form(g, UIR_D(w)), q);
    source(v, family, (t >> 3) & 1 ? ANTIPARTICLE : PARTICLE, q, (int)((t >> 4) & 7) - 2, (int)((t >> 7) & 3) - 1,
           g->scales[2 * (size_t)UIR_T(w)], g->factors + 2 * (size_t)UIR_X(w));
    store(a, edge_start(g, UIR_D(w)), n, v);
}

/* the template's family (bits 0-2, the physical type's number: SCALAR 1, WEYL 2, DIRAC 3, VECTOR 4) */
PAC_FN static UIR_INLINE void run_source(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    switch (UIR_N(w) & 7) {
    case 1: source_row(g, a, w, FAMILY_SCALAR, 1); break;
    case 2: source_row(g, a, w, FAMILY_WEYL, 2); break;
    case 3: source_row(g, a, w, FAMILY_DIRAC, 4); break;
    case 4: source_row(g, a, w, FAMILY_VECTOR, 4); break;
    default: break;   /* not a source family (validate) */
    }
}

PAC_FN int uir_rows(const uir_program *g, const uir_arena *a)
{
    cplx acc[UIR_MAX_COMPONENTS];   /* the fused run's sum (registers: every index is a constant where used) */
    for (uint32_t k = g->source_words;;) {
        const uint32_t *w = g->rows + 2 * (size_t)k++;
        uint32_t op = UIR_OP(w);
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
        } else {
            int status = run_kernel(g, a, w);
            if (status != 0) return status;
        }
    }
    return 0;
}

/* ---- forms, sources, reduction --------------------------------------------------------------- */

PAC_FN void uir_forms(const uir_program *g, const uir_arena *a, const double *momenta)
{
    for (uint32_t f = 0; f < g->forms; f++)
        for (int j = 0; j < 4; j++) {
            double v = 0.0;
            for (uint32_t t = g->form_first[f]; t < g->form_first[f + 1]; t++) {
                uint32_t leg = g->term_leg[t];
                double x = g->crossing_leg ? g->crossing_sign[leg] * momenta[4 * g->crossing_leg[leg] + j]
                                           : momenta[4 * leg + j];
                v = v + g->term_coefficient[t] * x;
            }
            a->forms[(4 * (size_t)f + j) * a->ld + a->i] = v;
        }
}

/* the sources: the SOURCE rows at the head of the stream (a pass of their own, so a GPU runner may
 * run them in f64 before converting the arena) */
PAC_FN void uir_sources(const uir_program *g, const uir_arena *a)
{
    for (uint32_t k = 0;;) {
        const uint32_t *w = g->rows + 2 * (size_t)k++;
        if (UIR_OP(w) != UIR_SOURCE) break;
        run_source(g, a, w);
    }
}

/* out[h] = group factor × Σ weight × |amplitude|² over output h's terms, as pacops' reduction */
PAC_FN void uir_reduce(const uir_program *g, const uir_arena *a, double *out)
{
    for (uint32_t h = 0; h < g->outputs; h++) {
        uint32_t first = g->output_first[h], last = g->output_first[h + 1];
        if (first == last) {
            out[h] = 0.0;
            continue;
        }
        double acc = 0.0;
        for (uint32_t t = first; t < last; t++) {
            uint32_t k = g->term_amplitude[t];
            double re = *amplitude_re(a, k), im = *amplitude_im(a, k);
            acc = acc + g->term_weight[t] * fma(re, re, im * im);
        }
        out[h] = g->group_numerator == g->group_denominator
                     ? acc
                     : acc * (double)g->group_numerator / (double)g->group_denominator;
    }
}

PAC_FN int uir_run(const uir_program *g, const uir_arena *a, const double *momenta, double *out)
{
    uir_forms(g, a, momenta);
    uir_sources(g, a);
    int status = uir_rows(g, a);
    if (status != 0) return status;
    uir_reduce(g, a, out);
    return 0;
}
