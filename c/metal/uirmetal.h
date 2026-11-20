/* uirmetal.h: one program (runner_metal.c: its Metal source and program) on an Apple GPU, through
 * Metal (uirmetal.m), one GPU thread per point, in mixed f32/df64 precision (c/metal/PLAN.md).
 *
 * The first call compiles the program's source at run time (MTLMathModeSafe: no fast math, no
 * contraction; Metal caches the compiled library, so later processes start fast) and loads the
 * program into a shared buffer. A call borrows a slot of a pool (its buffers in unified memory,
 * grown on demand), stages the momenta as df64, runs the points in launches of at most
 * PAC_METAL_CHUNK points (default 4096) on the GPU, waits, and reads the outputs back as doubles.
 * Thread safe: up to the pool's slots run at once (PAC_METAL_SLOTS, default the online CPUs), the
 * others wait. Errors (no device, a compile error, a failed command buffer) are printed to stderr.
 *
 * Not bit for bit the CPU build: the currents are f32 and the f64 parts df64 (relative errors
 * about 1e-7 on a matrix element; c/metal/PLAN.md section 2). */

#ifndef UIRMETAL_H
#define UIRMETAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* the program on `points` points: momenta[(4 leg + component) points + point] (E, px, py, pz per
 * leg, the caller's legs: the program's crossing applies), out[output points + point], host memory.
 * Returns 0, a kernel's failing status, or -1 (a Metal error). */
int pac_metal_run(const double *momenta, double *out, size_t points);

/* MadSpace's momentum layout: component k of particle p of event e at momenta[(n k + p) stride + e]
 * (n = RUNNER_LEGS; `momenta` at the call's first event), `count` events; values[h count + e] the
 * helicity outputs (the group factor applied), for the host's pac_finish (umami_finish.h).
 * Returns as pac_metal_run. */
int pac_metal_values(const double *momenta, size_t stride, size_t count, double *values);

/* the test entry: momenta[point][4 legs], out[point][outputs] (point-major) */
int run_metal(double *out, const double *momenta, size_t points);

#ifdef __cplusplus
}
#endif

#endif
