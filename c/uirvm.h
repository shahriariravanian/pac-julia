/* uirvm: the interpreter of UIR v0.8 (the pac-julia design doc, "The lowered IR" and "The 8-byte
 * encoding"), in portable C: one source for the CPU (cc, a test of the encoding against the
 * straight-line C and the Julia runners) and the GPU (nvcc, hipcc: one thread per point).
 *
 * A program is data (uir_program): one stream of encoded words, the rows (SOURCE rows among them),
 * then one END row; the edge table, and the tables the rows index. Every index is
 * 0-based; in a row's index field all ones means none (a null operand, which no row reads: validate
 * checks it).
 *
 * The arena of a point (uir_arena): plane k (one complex component) is (re, im) at
 * planes[(2k) ld + i], planes[(2k + 1) ld + i]; component j of momentum form f at forms[(4f + j) ld + i];
 * amplitude a at amplitudes[(2a) ld + i], [(2a + 1) ld + i]. With ld = 1, i = 0 this is the
 * straight-line runner's layout of one point (doubles in order); with ld = points, i = the point,
 * every plane is a contiguous row of points (coalesced on a GPU).
 *
 * The program supplies its kernel stubs: uir_stub(stub, g, a, w, scaleset) runs kernel row w through
 * stub `stub` (a kernel and its input layout): its inputs gathered (uir_input), the kernel, its
 * output pairs stored (uir_output); it returns the kernel's status (0: ok). Generated per program,
 * so every input code and output component is a constant there.
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

#define UIR_VERSION ((0u << 16) | 8u)   /* major << 16 | minor */

/* the opcodes (bits 0-4 of word 0) */
enum {
    UIR_V_WEYL_VECTOR_A, UIR_V_WEYL_VECTOR_B, UIR_V_TENSOR_VECTOR, UIR_V_VECTOR_WEDGE, UIR_V_THREE_VECTOR,
    UIR_P_WEYL_A, UIR_P_WEYL_B, UIR_P_VECTOR,
    UIR_CLOSURE, UIR_SCALE, UIR_CLEAR, UIR_KERNEL, UIR_KERNEL_DIRECT, UIR_SOURCE,
    UIR_CLEAR_AMPLITUDE, UIR_COPY, UIR_SCATTER, UIR_REDUCE,
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

    const uint32_t *rows;             /* the encoded words, each as two halves (low first): the rows', END */
    uint32_t words;                   /* the rows (one 64-bit word each), END included */
    const uint32_t *edges;            /* the edge table: start / 2 (15), form (8, 255 none), components (4), type (4) */
    const double *scales, *factors;   /* (re, im) each */

    /* the momentum forms: q_f = 0.0 + Σ coefficient × leg, in term order; with a crossing, program
       leg l is sign_l × the caller's leg crossing_leg_l */
    const uint32_t *form_first, *term_leg;
    const double *term_coefficient;
    const uint32_t *crossing_leg;     /* NULL: no crossing */
    const double *crossing_sign;

    const uint32_t *list_first, *list_factor;     /* the closures' component factors */
    /* the sources' embeddings: entry e writes component embedding_component[e] of the current, the
       wavefunction's component embedding_wave[e] (-1: zero) × factor embedding_factor[e] */
    const uint32_t *embedding_first;
    const int32_t *embedding_component, *embedding_wave;
    const uint32_t *embedding_factor;
    /* the kernel rows' scale sets (a stub's inputs and outputs are in the program's uir_stub) */
    const uint32_t *scaleset_first, *scaleset_scale;
    const uint32_t *instance;         /* 2 each: stub, scale set */

    const double *weights;            /* the REDUCE rows' weights */
    long group_numerator, group_denominator;
} uir_program;

typedef struct {
    double *forms, *amplitudes, *planes;
    size_t ld, i;
} uir_arena;

/* A row is its word, as two 32-bit halves: w[0] the low, w[1] the high. A field is read where it is
 * used: low half op 0-4, EXT 5, SET 6, FIRST 7, LAST 8, D 9-21, T 22-31; high half L 0-12, R 13-25,
 * N 0-15, X 26-31; a SOURCE's E (its embedding) 16-25. Index fields are 0-based; a null operand
 * (or an absent embedding, amplitude scale) is all ones.
 * The interpreter reads one-word rows only: a wide row (EXT, a second word for indices beyond these
 * widths) is not supported, and compile(...; interpreter = true) refuses a program that has one. */
#define UIR_OP(w) ((w)[0] & 31u)
#define UIR_SET(w) (((w)[0] >> 6) & 1u)
#define UIR_FIRST(w) (((w)[0] >> 7) & 1u)
#define UIR_LAST(w) (((w)[0] >> 8) & 1u)
#define UIR_D(w) (((w)[0] >> 9) & 0x1fffu)
#define UIR_T(w) ((w)[0] >> 22)
#define UIR_L(w) ((w)[1] & 0x1fffu)
#define UIR_R(w) (((w)[1] >> 13) & 0x1fffu)
#define UIR_N(w) ((w)[1] & 0xffffu)                     /* a count, or a SOURCE's template */
#define UIR_X(w) ((w)[1] >> 26)
#define UIR_E(w) (((w)[1] >> 16) & 0x3ffu)              /* a SOURCE's embedding (bits 48-57) */

/* the program's kernel stubs (the program defines it, after including uirvm.c) */
PAC_FN int uir_stub(uint32_t stub, const uir_program *g, const uir_arena *a, const uint32_t *w, const uint32_t *scaleset);

/* one point: the forms from its momenta (the caller's (E, px, py, pz) per leg), its outputs at
 * zero, the rows up to END (REDUCE rows add to the outputs), the group factor. The planes and
 * amplitudes must be zero before. Momenta and outputs are strided as the arena: component j of
 * leg l at momenta[(4 l + j) ld], output h at out[h ld] (the caller passes the point's first
 * element; ld = 1: the point's values in order). Returns 0, or the first failing kernel's status. */
PAC_FN void uir_forms(const uir_program *g, const uir_arena *a, const double *momenta);
PAC_FN int uir_rows(const uir_program *g, const uir_arena *a, double *out);
PAC_FN void uir_group(const uir_program *g, const uir_arena *a, double *out);
PAC_FN int uir_run(const uir_program *g, const uir_arena *a, const double *momenta, double *out);

#ifdef __CUDACC__
/* the program on the GPU (uirgpu.c, which runner.c includes under nvcc) */
#include "uirgpu.h"
#endif

#ifdef __cplusplus
}
#endif

#endif
