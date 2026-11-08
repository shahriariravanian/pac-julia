/* uirvm: the interpreter of UIR v0.6 (uirvm.h). The arithmetic is pacops' (pacops.h), on the
 * operands loaded into local arrays, so the values are the straight-line C's bit for bit; only the
 * storage differs. A program's runner.c includes this file (one translation unit, as nvcc wants it)
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

/* ---- the encoding ---------------------------------------------------------------------------- */

/* word 0, low half: op 0-4, EXT 5, SET 6, FIRST 7, LAST 8, D 9-21, T 22-31;
 * high half: L 0-12, R 13-25, N 0-15, X 26-31 (bits 32-63 of the word) */
uint32_t uir_decode(const uint64_t *w, uir_row *o)
{
    uint32_t lo = (uint32_t)w[0], hi = (uint32_t)(w[0] >> 32);
    uint32_t ext = (lo >> 5) & 1;
    o->op = lo & 31;
    o->set = (lo >> 6) & 1;
    o->first = (lo >> 7) & 1;
    o->last = (lo >> 8) & 1;
    o->d = (lo >> 9) & 0x1fff;
    o->t = lo >> 22;
    o->l = hi & 0x1fff;
    o->r = (hi >> 13) & 0x1fff;
    o->n = hi & 0xffff;
    o->x = hi >> 26;
    uint32_t nd = 0x1fff, nt = 0x3ff, nlr = 0x1fff, nx = 0x3f;
    if (ext) {
        /* word 1, low half: EXT 0, D 1-7, L 8-14, R 15-21, T 22-27, N 8-23; high half: X 0-9 */
        uint32_t lo1 = (uint32_t)w[1], hi1 = (uint32_t)(w[1] >> 32);
        o->d |= ((lo1 >> 1) & 0x7f) << 13;
        o->l |= ((lo1 >> 8) & 0x7f) << 13;
        o->r |= ((lo1 >> 15) & 0x7f) << 13;
        o->t |= ((lo1 >> 22) & 0x3f) << 10;
        o->n |= ((lo1 >> 8) & 0xffff) << 16;
        o->x |= (hi1 & 0x3ff) << 6;
        nd = 0xfffff, nt = 0xffff, nlr = 0xfffff, nx = 0xffff;
    }
    if (o->d == nd) o->d = UIR_NONE;
    if (o->t == nt) o->t = UIR_NONE;
    if (o->l == nlr) o->l = UIR_NONE;
    if (o->r == nlr) o->r = UIR_NONE;
    if (o->x == nx) o->x = UIR_NONE;
    return ext ? 2 : 1;
}

/* an edge-table entry's fields */
static inline uint32_t edge_start(const uir_program *g, uint32_t e) { return (g->edges[e] & 0x7fff) * 2; }
static inline uint32_t edge_form(const uir_program *g, uint32_t e) { return (g->edges[e] >> 15) & 0xff; }
static inline int edge_components(const uir_program *g, uint32_t e) { return (g->edges[e] >> 23) & 0xf; }

/* ---- the arena ------------------------------------------------------------------------------- */

static inline double *plane_re(const uir_arena *a, uint32_t k) { return a->planes + (2 * (size_t)k) * a->ld + a->i; }
static inline double *plane_im(const uir_arena *a, uint32_t k) { return a->planes + (2 * (size_t)k + 1) * a->ld + a->i; }

/* n components from plane `start` into v (re, im each), and back */
static inline void load(const uir_arena *a, uint32_t start, int n, double *v)
{
    for (int j = 0; j < n; j++) {
        v[2 * j] = *plane_re(a, start + j);
        v[2 * j + 1] = *plane_im(a, start + j);
    }
}

static inline void store(const uir_arena *a, uint32_t start, int n, const double *v)
{
    for (int j = 0; j < n; j++) {
        *plane_re(a, start + j) = v[2 * j];
        *plane_im(a, start + j) = v[2 * j + 1];
    }
}

static inline cplx load1(const uir_arena *a, uint32_t k) { return cx(*plane_re(a, k), *plane_im(a, k)); }
static inline void store1(const uir_arena *a, uint32_t k, cplx z) { *plane_re(a, k) = z.re; *plane_im(a, k) = z.im; }

/* momentum form f into q[4] */
static inline void load_form(const uir_arena *a, uint32_t f, double *q)
{
    for (int j = 0; j < 4; j++) q[j] = a->forms[(4 * (size_t)f + j) * a->ld + a->i];
}

static inline double *amplitude_re(const uir_arena *a, uint32_t k) { return a->amplitudes + (2 * (size_t)k) * a->ld + a->i; }
static inline double *amplitude_im(const uir_arena *a, uint32_t k) { return a->amplitudes + (2 * (size_t)k + 1) * a->ld + a->i; }

/* ---- the rows -------------------------------------------------------------------------------- */

/* the components a vertex writes */
static const int VERTEX_OUT[5] = {2, 2, 4, 6, 4};

/* a vertex row: its values × (coupling × factor) into the run's sum `acc`; stored at LAST */
static void run_vertex(const uir_program *g, const uir_arena *a, const uir_row *o, cplx *acc)
{
    double l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], v[2 * UIR_MAX_COMPONENTS];
    load(a, edge_start(g, o->l), edge_components(g, o->l), l);
    load(a, edge_start(g, o->r), edge_components(g, o->r), r);
    switch (o->op) {
    case UIR_V_WEYL_VECTOR_A: vertex_weyl_vector_a(v, l, r); break;
    case UIR_V_WEYL_VECTOR_B: vertex_weyl_vector_b(v, l, r); break;
    case UIR_V_TENSOR_VECTOR: vertex_tensor_vector(v, l, r); break;
    case UIR_V_VECTOR_WEDGE: vertex_vector_wedge(v, l, r); break;
    default: {
        double u[4], w[4];
        load_form(a, edge_form(g, o->l), u);
        load_form(a, edge_form(g, o->r), w);
        vertex_three_vector(v, l, r, u, w);
    }
    }
    int n = VERTEX_OUT[o->op];
    uint32_t d = edge_start(g, o->d);
    cplx s = mul(get(g->scales, o->t), get(g->factors, o->x));
    for (int j = 0; j < n; j++) {
        cplx c = mul(get(v, j), s);
        if (!o->first) acc[j] = add(acc[j], c);
        else if (o->set) acc[j] = c;
        else acc[j] = add(load1(a, d + j), c);
    }
    if (o->last)
        for (int j = 0; j < n; j++) store1(a, d + j, acc[j]);
}

/* a propagator row: the current's new values */
static void run_propagator(const uir_program *g, const uir_arena *a, const uir_row *o)
{
    double c[8], q[4], v[8];
    uint32_t d = edge_start(g, o->d);
    int n = o->op == UIR_P_VECTOR ? 4 : 2;
    load(a, d, n, c);
    load_form(a, edge_form(g, o->d), q);
    propagator(v, o->op - UIR_P_WEYL_A, c, q, g->factors + 2 * (size_t)o->x);
    store(a, d, n, v);
}

/* a closure row: amplitude += Σ l[j] r[j] × component factor j × factor */
static void run_closure(const uir_program *g, const uir_arena *a, const uir_row *o)
{
    double l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], cf[2 * UIR_MAX_COMPONENTS], amp[2];
    uint32_t first = g->list_first[o->t];
    int n = (int)(g->list_first[o->t + 1] - first);
    for (int j = 0; j < n; j++) {
        uint32_t f = g->list_factor[first + j];
        cf[2 * j] = g->factors[2 * (size_t)f];
        cf[2 * j + 1] = g->factors[2 * (size_t)f + 1];
    }
    load(a, edge_start(g, o->l), n, l);
    load(a, edge_start(g, o->r), n, r);
    amp[0] = *amplitude_re(a, o->d);
    amp[1] = *amplitude_im(a, o->d);
    closure(amp, l, r, cf, n, g->factors + 2 * (size_t)o->x);
    *amplitude_re(a, o->d) = amp[0];
    *amplitude_im(a, o->d) = amp[1];
}

/* a kernel input */
static double kernel_input(const uir_program *g, const uir_arena *a, const uir_row *o, uint32_t code,
                           const uint32_t *scaleset)
{
    uint32_t kind = code & 7, im = (code >> 3) & 1, n = code >> 4;
    double q[4];
    switch (kind) {
    case UIR_IN_L: return im ? *plane_im(a, edge_start(g, o->l) + n) : *plane_re(a, edge_start(g, o->l) + n);
    case UIR_IN_R: return im ? *plane_im(a, edge_start(g, o->r) + n) : *plane_re(a, edge_start(g, o->r) + n);
    case UIR_OUT: return im ? *plane_im(a, edge_start(g, o->d) + n) : *plane_re(a, edge_start(g, o->d) + n);
    case UIR_MOMENTUM_U: load_form(a, edge_form(g, o->l), q); return q[n];
    case UIR_MOMENTUM_V: load_form(a, edge_form(g, o->r), q); return q[n];
    case UIR_IN_SCALE: return g->scales[2 * (size_t)scaleset[n]];
    default: return g->factors[2 * (size_t)o->x + im];
    }
}

/* a kernel row: the stub's inputs, the kernel, its pairs into out (KERNEL_DIRECT: as they are) */
static int run_kernel(const uir_program *g, const uir_arena *a, const uir_row *o)
{
    double in[UIR_KERNEL_IN], out[UIR_KERNEL_OUT];
    uint32_t s = g->instance[2 * (size_t)o->t], set = g->instance[2 * (size_t)o->t + 1];
    const uint32_t *scaleset = g->scaleset_scale + g->scaleset_first[set];
    uint32_t i0 = g->stub_inputs[s], inputs = g->stub_inputs[s + 1] - i0;
    uint32_t o0 = g->stub_outputs[s], pairs = g->stub_outputs[s + 1] - o0;
    for (uint32_t k = 0; k < inputs; k++) in[k] = kernel_input(g, a, o, g->input_code[i0 + k], scaleset);
    int status = uir_kernel(g->stub_kernel[s], in, out);
    if (status != 0) return status;
    uint32_t d = edge_start(g, o->d);
    cplx f = get(g->factors, o->x);
    int mode = o->set ? MODE_SET : MODE_ADD;
    for (uint32_t n = 0; n < pairs; n++) {
        uint32_t j = d + g->output_component[o0 + n];
        cplx pair = get(out, n);
        store1(a, j, o->op == UIR_KERNEL_DIRECT ? pair : combine(load1(a, j), mul(pair, f), mode));
    }
    return 0;
}

int uir_rows(const uir_program *g, const uir_arena *a)
{
    cplx acc[UIR_MAX_COMPONENTS];
    uir_row o;
    for (uint32_t k = 0; k < g->words;) {
        k += uir_decode(g->rows + k, &o);
        if (o.op <= UIR_V_THREE_VECTOR) {
            run_vertex(g, a, &o, acc);
        } else if (o.op <= UIR_P_VECTOR) {
            run_propagator(g, a, &o);
        } else if (o.op == UIR_CLOSURE) {
            run_closure(g, a, &o);
        } else if (o.op == UIR_SCALE) {
            double c[2 * UIR_MAX_COMPONENTS];
            uint32_t d = edge_start(g, o.d);
            load(a, d, (int)o.n, c);
            scale(c, (int)o.n, g->factors + 2 * (size_t)o.x);
            store(a, d, (int)o.n, c);
        } else if (o.op == UIR_CLEAR) {
            uint32_t d = edge_start(g, o.d);
            for (uint32_t j = 0; j < o.n; j++) store1(a, d + j, ZERO);
        } else {
            int status = run_kernel(g, a, &o);
            if (status != 0) return status;
        }
    }
    return 0;
}

/* ---- forms, sources, reduction --------------------------------------------------------------- */

void uir_forms(const uir_program *g, const uir_arena *a, const double *momenta)
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

void uir_sources(const uir_program *g, const uir_arena *a)
{
    for (uint32_t k = 0; k < g->sources; k++) {
        const int32_t *s = g->source + 8 * (size_t)k;
        double q[4], v[2 * UIR_MAX_COMPONENTS];
        load_form(a, (uint32_t)s[1], q);
        /* the physical type's number: SCALAR 1, WEYL 2, DIRAC 3, VECTOR 4; pacops' families count from 0 */
        int n = source(v, s[4] - 1, s[5] ? ANTIPARTICLE : PARTICLE, q, s[6], s[7], g->scales[2 * (size_t)s[3]],
                       g->factors + 2 * (size_t)s[2]);
        store(a, edge_start(g, (uint32_t)s[0]), n, v);
    }
}

/* out[h] = group factor × Σ weight × |amplitude|² over output h's terms, as pacops' reduction */
void uir_reduce(const uir_program *g, const uir_arena *a, double *out)
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

int uir_run(const uir_program *g, const uir_arena *a, const double *momenta, double *out)
{
    uir_forms(g, a, momenta);
    uir_sources(g, a);
    int status = uir_rows(g, a);
    if (status != 0) return status;
    uir_reduce(g, a, out);
    return 0;
}
