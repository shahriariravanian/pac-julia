/* uirgpu: one program (runner.c under nvcc) on a CUDA GPU, one thread per point, through a
 * persistent context: device buffers and pinned host staging that grow to the largest call and are
 * kept, one non-blocking stream, asynchronous copies. The arena, momenta and outputs are
 * plane-major (ld = points: coalesced).
 *
 * The bits are the CPU interpreter's (uirvm.c with ld = 1) when built with --fmad=false.
 *
 * A context is not thread safe: one caller at a time (the UMAMI wrapper holds a lock). */

#ifndef UIRGPU_H
#define UIRGPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pac_gpu pac_gpu;

/* a context on the current device (no buffers until the first run); 0, or -1 (a CUDA error,
 * printed to stderr) */
int pac_gpu_create(pac_gpu **ctx);
void pac_gpu_destroy(pac_gpu *ctx);

/* the program on `points` points: momenta[(4 leg + component) points + point] (E, px, py, pz per
 * leg, the program's leg order), out[output points + point]; both host memory (any). Returns 0, a
 * kernel's failing status, or -1 (a CUDA error, printed to stderr). */
int pac_gpu_run(pac_gpu *ctx, const double *momenta, double *out, size_t points);

/* the test entry: momenta[point][4 legs], out[point][outputs] (point-major) on a context of its own */
int run_gpu(double *out, const double *momenta, size_t points);

#ifdef __cplusplus
}
#endif

#endif
