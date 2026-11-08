/* uirvm: the interpreter of UIR v0.7 (uirvm.h). The arithmetic is pacops' (pacops.h), on the
 * operands loaded into local arrays, so the values are the straight-line C's bit for bit; only the
 * storage differs. A row is its word (`w`, uirvm.h), its fields read by the UIR_ macros where used. A program's runner.c includes this file (one translation unit, as nvcc wants it)
 * after defining UIR_KERNEL_IN and UIR_KERNEL_OUT, the largest kernel's input and output doubles. */

#include "pacops.h"
#include "uirvm.h"

#ifndef UIR_KERNEL_IN
#define UIR_KERNEL_IN 64
#endif
#ifndef UIR_KERNEL_OUT
#define UIR_KERNEL_OUT 64
#endif

#define UIR_MAX_COMPONENTS 6

/* an edge-table entry's fields */
PAC_FN static inline uint32_t edge_start(const uir_program *g, uint32_t e) { return (g->edges[e] & 0x7fff) * 2; }
PAC_FN static inline uint32_t edge_form(const uir_program *g, uint32_t e) { return (g->edges[e] >> 15) & 0xff; }
PAC_FN static inline int edge_components(const uir_program *g, uint32_t e) { return (g->edges[e] >> 23) & 0xf; }

/* ---- the arena ------------------------------------------------------------------------------- */

PAC_FN static inline double *plane_re(const uir_arena *a, uint32_t k) { return a->planes + (2 * (size_t)k) * a->ld + a->i; }
PAC_FN static inline double *plane_im(const uir_arena *a, uint32_t k) { return a->planes + (2 * (size_t)k + 1) * a->ld + a->i; }

/* n components from plane `start` into v (re, im each), and back */
PAC_FN static inline void load(const uir_arena *a, uint32_t start, int n, double *v)
{
    for (int j = 0; j < n; j++) {
        v[2 * j] = *plane_re(a, start + j);
        v[2 * j + 1] = *plane_im(a, start + j);
    }
}

PAC_FN static inline void store(const uir_arena *a, uint32_t start, int n, const double *v)
{
    for (int j = 0; j < n; j++) {
        *plane_re(a, start + j) = v[2 * j];
        *plane_im(a, start + j) = v[2 * j + 1];
    }
}

PAC_FN static inline cplx load1(const uir_arena *a, uint32_t k) { return cx(*plane_re(a, k), *plane_im(a, k)); }
PAC_FN static inline void store1(const uir_arena *a, uint32_t k, cplx z) { *plane_re(a, k) = z.re; *plane_im(a, k) = z.im; }

/* momentum form f into q[4] */
PAC_FN static inline void load_form(const uir_arena *a, uint32_t f, double *q)
{
    for (int j = 0; j < 4; j++) q[j] = a->forms[(4 * (size_t)f + j) * a->ld + a->i];
}

PAC_FN static inline double *amplitude_re(const uir_arena *a, uint32_t k) { return a->amplitudes + (2 * (size_t)k) * a->ld + a->i; }
PAC_FN static inline double *amplitude_im(const uir_arena *a, uint32_t k) { return a->amplitudes + (2 * (size_t)k + 1) * a->ld + a->i; }

/* ---- the rows -------------------------------------------------------------------------------- */

/* the components a vertex writes (a function, not a table: a host table is not the device's) */
PAC_FN static inline int vertex_out(uint32_t op) { return op == UIR_V_TENSOR_VECTOR || op == UIR_V_THREE_VECTOR ? 4 : op == UIR_V_VECTOR_WEDGE ? 6 : 2; }

/* a vertex row: its values × (coupling × factor) into the run's sum `acc`; stored at LAST */
PAC_FN static void run_vertex(const uir_program *g, const uir_arena *a, const uint32_t *w, cplx *acc)
{
    double l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], v[2 * UIR_MAX_COMPONENTS];
    load(a, edge_start(g, UIR_L(w)), edge_components(g, UIR_L(w)), l);
    load(a, edge_start(g, UIR_R(w)), edge_components(g, UIR_R(w)), r);
    switch (UIR_OP(w)) {
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
    int n = vertex_out(UIR_OP(w));
    uint32_t d = edge_start(g, UIR_D(w));
    cplx s = mul(get(g->scales, UIR_T(w)), get(g->factors, UIR_X(w)));
    for (int j = 0; j < n; j++) {
        cplx c = mul(get(v, j), s);
        if (!UIR_FIRST(w)) acc[j] = add(acc[j], c);
        else if (UIR_SET(w)) acc[j] = c;
        else acc[j] = add(load1(a, d + j), c);
    }
    if (UIR_LAST(w))
        for (int j = 0; j < n; j++) store1(a, d + j, acc[j]);
}

/* a propagator row: the current's new values */
PAC_FN static void run_propagator(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    double c[8], q[4], v[8];
    uint32_t d = edge_start(g, UIR_D(w));
    int n = UIR_OP(w) == UIR_P_VECTOR ? 4 : 2;
    load(a, d, n, c);
    load_form(a, edge_form(g, UIR_D(w)), q);
    propagator(v, UIR_OP(w) - UIR_P_WEYL_A, c, q, g->factors + 2 * (size_t)UIR_X(w));
    store(a, d, n, v);
}

/* a closure row: amplitude += Σ l[j] r[j] × component factor j × factor */
PAC_FN static void run_closure(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    double l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], cf[2 * UIR_MAX_COMPONENTS], amp[2];
    uint32_t first = g->list_first[UIR_T(w)];
    int n = (int)(g->list_first[UIR_T(w) + 1] - first);
    for (int j = 0; j < n; j++) {
        uint32_t f = g->list_factor[first + j];
        cf[2 * j] = g->factors[2 * (size_t)f];
        cf[2 * j + 1] = g->factors[2 * (size_t)f + 1];
    }
    load(a, edge_start(g, UIR_L(w)), n, l);
    load(a, edge_start(g, UIR_R(w)), n, r);
    amp[0] = *amplitude_re(a, UIR_D(w));
    amp[1] = *amplitude_im(a, UIR_D(w));
    closure(amp, l, r, cf, n, g->factors + 2 * (size_t)UIR_X(w));
    *amplitude_re(a, UIR_D(w)) = amp[0];
    *amplitude_im(a, UIR_D(w)) = amp[1];
}

/* a kernel input */
PAC_FN static double kernel_input(const uir_program *g, const uir_arena *a, const uint32_t *w, uint32_t code,
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

/* a kernel row: the stub's inputs, the kernel, its pairs into out (KERNEL_DIRECT: as they are) */
PAC_FN static int run_kernel(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    double in[UIR_KERNEL_IN], out[UIR_KERNEL_OUT];
    uint32_t s = g->instance[2 * (size_t)UIR_T(w)], set = g->instance[2 * (size_t)UIR_T(w) + 1];
    const uint32_t *scaleset = g->scaleset_scale + g->scaleset_first[set];
    uint32_t i0 = g->stub_inputs[s], inputs = g->stub_inputs[s + 1] - i0;
    uint32_t o0 = g->stub_outputs[s], pairs = g->stub_outputs[s + 1] - o0;
    for (uint32_t k = 0; k < inputs; k++) in[k] = kernel_input(g, a, w, g->input_code[i0 + k], scaleset);
    int status = uir_kernel(g->stub_kernel[s], in, out);
    if (status != 0) return status;
    uint32_t d = edge_start(g, UIR_D(w));
    cplx f = get(g->factors, UIR_X(w));
    int mode = UIR_SET(w) ? MODE_SET : MODE_ADD;
    for (uint32_t n = 0; n < pairs; n++) {
        uint32_t j = d + g->output_component[o0 + n];
        cplx pair = get(out, n);
        store1(a, j, UIR_OP(w) == UIR_KERNEL_DIRECT ? pair : combine(load1(a, j), mul(pair, f), mode));
    }
    return 0;
}

/* a source row: factor × the wavefunction at its edge's form, into its edge's current */
PAC_FN static void run_source(const uir_program *g, const uir_arena *a, const uint32_t *w)
{
    double q[4], v[2 * UIR_MAX_COMPONENTS];
    uint32_t t = UIR_N(w);
    load_form(a, edge_form(g, UIR_D(w)), q);
    /* pacops' families count from 0 (the physical type's number − 1) */
    int n = source(v, (int)(t & 7) - 1, (t >> 3) & 1 ? ANTIPARTICLE : PARTICLE, q, (int)((t >> 4) & 7) - 2,
                   (int)((t >> 7) & 3) - 1, g->scales[2 * (size_t)UIR_T(w)], g->factors + 2 * (size_t)UIR_X(w));
    store(a, edge_start(g, UIR_D(w)), n, v);
}

PAC_FN int uir_rows(const uir_program *g, const uir_arena *a)
{
    cplx acc[UIR_MAX_COMPONENTS];
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
            double c[2 * UIR_MAX_COMPONENTS];
            uint32_t d = edge_start(g, UIR_D(w));
            int n = (int)UIR_N(w);
            load(a, d, n, c);
            scale(c, n, g->factors + 2 * (size_t)UIR_X(w));
            store(a, d, n, c);
        } else if (op == UIR_CLEAR) {
            uint32_t d = edge_start(g, UIR_D(w)), n = UIR_N(w);
            for (uint32_t j = 0; j < n; j++) store1(a, d + j, ZERO);
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
