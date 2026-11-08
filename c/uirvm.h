/* uirvm: the interpreter of UIR v0.7 (the pac-julia design doc, "The lowered IR: UIR v0.7" and "The
 * 8-byte encoding"), in portable C: one source for the CPU (cc, a test of the encoding against the
 * straight-line C and the Julia runners) and the GPU (nvcc, hipcc: one thread per point).
 *
 * A program is data (uir_program): one stream of encoded words, the sources (SOURCE rows) first and
 * then the rows, then one END row; the edge table, and the tables the rows index. Every index is
 * 0-based; in a row's index field all ones means none (a null operand, which no row reads: validate
 * checks it).
 *
 * The arena of a point (uir_arena): plane k (one complex component) is (re, im) at
 * planes[(2k) ld + i], planes[(2k + 1) ld + i]; component j of momentum form f at forms[(4f + j) ld + i];
 * amplitude a at amplitudes[(2a) ld + i], [(2a + 1) ld + i]. With ld = 1, i = 0 this is the
 * straight-line runner's layout of one point (doubles in order); with ld = points, i = the point,
 * every plane is a contiguous row of points (coalesced on a GPU).
 *
 * The program supplies its kernels: uir_kernel(kernel, in, out) runs kernel `kernel` on `in` (its
 * inputs) into `out` (its output pairs, re and im) and returns its status (0: ok).
 *
 * Bit-exactness with the Julia runners needs the plain IEEE operations: -ffp-contract=off (cc),
 * --fmad=false (nvcc).
 *
 * The same runner.c builds for both: cc gives run(), one point on the host; nvcc -x cu gives
 * run_gpu(), every point on the device, one thread each (the tables are then __device__ data). */

#ifndef UIRVM_H
#define UIRVM_H

#include <stddef.h>
#include <stdint.h>

/* a function of the host and, under nvcc, of the device (as pacops.h's) */
#ifndef PAC_FN
#ifdef __CUDACC__
#define PAC_FN __host__ __device__
#else
#define PAC_FN
#endif
#endif

/* a program's tables (runner.c's): the device's under nvcc, the host's otherwise */
#ifdef __CUDACC__
#define UIR_DATA __device__ const
#else
#define UIR_DATA static const
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define UIR_VERSION ((0u << 16) | 7u)   /* major << 16 | minor */

/* the opcodes (bits 0-4 of word 0) */
enum {
    UIR_V_WEYL_VECTOR_A, UIR_V_WEYL_VECTOR_B, UIR_V_TENSOR_VECTOR, UIR_V_VECTOR_WEDGE, UIR_V_THREE_VECTOR,
    UIR_P_WEYL_A, UIR_P_WEYL_B, UIR_P_VECTOR,
    UIR_CLOSURE, UIR_SCALE, UIR_CLEAR, UIR_KERNEL, UIR_KERNEL_DIRECT, UIR_SOURCE,
    UIR_END = 31                      /* the stream's last word: no fields */
};

/* a SOURCE row's template (its N): family 0-2 (the physical type: SCALAR 1, WEYL 2, DIRAC 3, VECTOR 4),
 * antiparticle 3, helicity + 2 at 4-6, chirality + 1 at 7-8 */

/* a kernel input's code: kind (bits 0-2), imaginary part (bit 3), n (bits 4-31: the component, or
 * the slot in the scale set) */
enum { UIR_IN_L, UIR_IN_R, UIR_OUT, UIR_MOMENTUM_U, UIR_MOMENTUM_V, UIR_IN_SCALE, UIR_IN_FACTOR };

/* A table "first" holds n + 1 offsets: entry k's items are items[first[k] .. first[k + 1]). */
typedef struct {
    uint32_t version;                 /* UIR_VERSION */
    uint32_t legs, forms, planes, amplitudes, outputs;

    const uint64_t *rows;             /* the encoded words: the sources', the rows', END */
    uint32_t words, source_words;     /* all of them (END's included); the sources' at the head */
    const uint32_t *edges;            /* the edge table: start / 2 (15), form (8, 255 none), components (4), type (4) */
    const double *scales, *factors;   /* (re, im) each */

    /* the momentum forms: q_f = 0.0 + Σ coefficient × leg, in term order; with a crossing, program
       leg l is sign_l × the caller's leg crossing_leg_l */
    const uint32_t *form_first, *term_leg;
    const double *term_coefficient;
    const uint32_t *crossing_leg;     /* NULL: no crossing */
    const double *crossing_sign;

    const uint32_t *list_first, *list_factor;     /* the closures' component factors */
    /* the kernel stubs: stub s's kernel, its input codes and the components of out its output pairs go to */
    const uint32_t *stub_kernel, *stub_inputs, *input_code, *stub_outputs, *output_component;
    const uint32_t *scaleset_first, *scaleset_scale;
    const uint32_t *instance;         /* 2 each: stub, scale set */

    const uint32_t *output_first, *term_amplitude; /* the reduction: output h's terms */
    const double *term_weight;
    long group_numerator, group_denominator;
} uir_program;

typedef struct {
    double *forms, *amplitudes, *planes;
    size_t ld, i;
} uir_arena;

/* a row decoded: its fields 0-based as they are in the word (a null operand all ones of its width:
 * 13 or 20 bits for D, L, R; 10 or 16 for T; 6 or 16 for X) */
typedef struct {
    uint32_t op, set, first, last, d, t, l, r, n, x;
} uir_row;

/* the program's kernels (the program defines it) */
PAC_FN int uir_kernel(uint32_t kernel, const double *in, double *out);

/* the row at w, into o; returns its words (1 or 2) */
PAC_FN uint32_t uir_decode(const uint64_t *w, uir_row *o);

/* one point: the forms from momenta[4 legs] (the caller's (E, px, py, pz) per leg), the sources (the
 * SOURCE rows at the head), the rows (from word source_words up to END), the reduction into
 * out[outputs]. The planes and amplitudes must be zero before. Returns 0, or the first failing kernel's status. */
PAC_FN void uir_forms(const uir_program *g, const uir_arena *a, const double *momenta);
PAC_FN void uir_sources(const uir_program *g, const uir_arena *a);
PAC_FN int uir_rows(const uir_program *g, const uir_arena *a);
PAC_FN void uir_reduce(const uir_program *g, const uir_arena *a, double *out);
PAC_FN int uir_run(const uir_program *g, const uir_arena *a, const double *momenta, double *out);

#ifdef __CUDACC__
/* the program on `points` points (the program defines it, runner.c under nvcc): momenta[point][4 legs],
 * out[point][outputs], both on the host; returns 0, a kernel's failing status, or -1 (a CUDA error,
 * printed to stderr) */
int run_gpu(double *out, const double *momenta, size_t points);
#endif

#ifdef __cplusplus
}
#endif

#endif
