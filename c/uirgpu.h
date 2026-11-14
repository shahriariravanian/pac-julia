/* uirgpu: one program (runner.c under nvcc) on a CUDA GPU, one thread per point, on a slot of the
 * process-wide pool (pool.h): a call borrows a free slot, carves the program's arena out of the
 * slot's device block, runs on the slot's stream, and gives the slot back. The arena, momenta and
 * outputs are plane-major (ld = the launch's points: coalesced); a call larger than the slot's
 * budget runs as several launches (every point is computed alone: the same values).
 *
 * The bits are the CPU interpreter's (uirvm.c with ld = 1) when built with --fmad=false.
 * Thread safe: any number of threads may call at once (up to the pool's slots run together). */

#ifndef UIRGPU_H
#define UIRGPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* the program on `points` points: momenta[(4 leg + component) points + point] (E, px, py, pz per
 * leg, the program's leg order), out[output points + point]; both host memory (any). Returns 0, a
 * kernel's failing status, or -1 (a CUDA error, printed to stderr). */
int pac_gpu_run(const double *momenta, double *out, size_t points);

/* UIRGPU_UMAMI (a UMAMI library, umami_finish.h): `count` events of MadSpace's momentum layout,
 * component k of particle p at momenta[(n k + p) stride + event] (n = the legs; `momenta` at the
 * call's first event), run with the program's crossing (which carries the library's leg
 * permutation) and finished on the device (pac_finish): me[event], label[event] (either NULL: not
 * wanted). Per event, optional (NULL): scale (the alpha_s rescale, multiplied in), fixed (an
 * exported helicity, else the sum and, with `draw`, a drawn one), random (the draw's number; else
 * 0.5). All host memory. Returns 0, a kernel's failing status, -2 (a non-finite momentum or
 * matrix element) or -1 (a CUDA error, printed to stderr). */
int pac_gpu_umami(const double *momenta, size_t stride, size_t count, const double *scale, const int *fixed,
                  const double *random, int draw, double *me, int *label);

/* the test entry: momenta[point][4 legs], out[point][outputs] (point-major) */
int run_gpu(double *out, const double *momenta, size_t points);

#ifdef __cplusplus
}
#endif

#endif
