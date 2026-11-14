/* uirgpu.c: uirgpu.h, included by runner.c under nvcc after the program's tables and uir_stub,
 * with its sizes defined:
 *   UIRGPU_LEGS, UIRGPU_OUTPUTS, UIRGPU_FORMS, UIRGPU_AMPLITUDES, UIRGPU_PLANES
 * (legs; helicity outputs; momentum forms; amplitudes; planes = complex components).
 *
 * This file holds what every build shares: the arena's regions per point, their carving out of
 * one device block, the CUDA error check. Then it includes the entries of one build:
 *
 *   (default)         uirgpu_pool.c      pac_gpu_run, run_gpu on the slot pool (pool.h): the
 *                                        standalone runner (tests)
 *   UIRGPU_UMAMI      + uirgpu_umami.c   pac_gpu_umami: a UMAMI library's call on the pool,
 *                                        finished on the device (umami_finish.h)
 *   UIRGPU_PROVIDER   uirgpu_provider.c  pac_gpu_provider: a CUDA provider on MadSpace's device
 *                                        arrays and stream (no pool)
 *
 * The pool's size (the first program library of the process decides): UIRGPU_SLOTS (0: the online
 * CPUs), UIRGPU_SLOT_BYTES (device bytes per slot; the CUDA provider's workspace budget of a
 * launch). Without PACPOOL_EXTERNAL the pool itself (pool.cu) is compiled in: a standalone runner
 * (tests); the UMAMI libraries link the process-wide libpacpool.so instead. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "uirgpu.h"

#ifndef UIRGPU_SLOTS
#define UIRGPU_SLOTS 0
#endif
#ifndef UIRGPU_SLOT_BYTES
#define UIRGPU_SLOT_BYTES ((size_t)256 << 20)
#endif

/* numbers per point of each region (at least 1: no empty regions); momenta, outputs and forms are
 * double, amplitudes pac_amp, planes pac_real (pacops.h: float under PAC_MIXED) */
#define UIRGPU_MAX1(x) ((x) > 0 ? (x) : 1)
#define UIRGPU_N_MOMENTA UIRGPU_MAX1(4 * UIRGPU_LEGS)
#define UIRGPU_N_OUT UIRGPU_MAX1(UIRGPU_OUTPUTS)
#define UIRGPU_N_FORMS UIRGPU_MAX1(4 * UIRGPU_FORMS)
#define UIRGPU_N_AMPLITUDES UIRGPU_MAX1(2 * UIRGPU_AMPLITUDES)
#define UIRGPU_N_PLANES UIRGPU_MAX1(2 * UIRGPU_PLANES)
#define UIRGPU_REGIONS 5
#define UIRGPU_ALIGN ((size_t)256)

/* bytes per point of each region: momenta, outputs, forms, amplitudes, planes */
static const size_t uirgpu_per_point[UIRGPU_REGIONS] = {
    UIRGPU_N_MOMENTA * sizeof(double), UIRGPU_N_OUT * sizeof(double), UIRGPU_N_FORMS * sizeof(double),
    UIRGPU_N_AMPLITUDES * sizeof(pac_amp), UIRGPU_N_PLANES * sizeof(pac_real)};

static size_t uirgpu_aligned(size_t bytes) { return (bytes + UIRGPU_ALIGN - 1) / UIRGPU_ALIGN * UIRGPU_ALIGN; }

/* A launch's regions: `n` arrays of per_point[r] bytes a point, each 256-byte aligned.
 * uirgpu_chunk: the most points of one launch within `budget` bytes, at most `points` (0, with a
 * message: the budget holds no point); uirgpu_bytes: the bytes of `chunk` points; uirgpu_carve:
 * the regions' addresses in a block of that size. */
static size_t uirgpu_chunk(const size_t *per_point, int n, size_t budget, size_t points)
{
    size_t each = 0, slack = (size_t)n * UIRGPU_ALIGN;
    for (int r = 0; r < n; r++) each += per_point[r];
    size_t most = budget > slack ? (budget - slack) / each : 0;
    if (most == 0)
        fprintf(stderr, "uirgpu (%s %s): %zu bytes hold no point (%zu bytes each)\n", RUNNER_PROCESS, RUNNER_COLOUR,
                budget, each);
    return points < most ? points : most;
}

static size_t uirgpu_bytes(const size_t *per_point, int n, size_t chunk)
{
    size_t total = 0;
    for (int r = 0; r < n; r++) total += uirgpu_aligned(per_point[r] * chunk);
    return total;
}

static void uirgpu_carve(void *block, const size_t *per_point, int n, size_t chunk, void **region)
{
    char *p = (char *)block;
    for (int r = 0; r < n; r++) {
        region[r] = p;
        p += uirgpu_aligned(per_point[r] * chunk);
    }
}

/* UIRGPU_REPORT: a failed CUDA call `x` (its error `e_`) printed to stderr; UIRGPU_CHECK: a CUDA
 * call that, failing, is reported and sets status = -1, then goto done (the entry's cleanup) */
#define UIRGPU_REPORT(x) fprintf(stderr, "uirgpu (%s %s): %s: %s\n", RUNNER_PROCESS, RUNNER_COLOUR, #x, cudaGetErrorString(e_))
#define UIRGPU_CHECK(x)                                                                            \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            UIRGPU_REPORT(x);                                                                      \
            status = -1;                                                                           \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

/* the entries of this build */
#ifdef UIRGPU_PROVIDER
#include "uirgpu_provider.c"
#else
#include "uirgpu_pool.c"
#ifdef UIRGPU_UMAMI
#include "uirgpu_umami.c"
#endif
#endif
