/* umami_finish.h: the end of one event in PacRunner's UMAMI libraries, shared by the CPU build
 * (umami_pac.c, plain C) and the GPU build (uirgpu.c under nvcc with UIRGPU_UMAMI, a __device__
 * function run by the point's own thread right after the program): from the program's runtime
 * helicity values to the matrix element and the helicity index. The operations are those of the
 * CPU wrapper in the same order, so both builds give the same bits.
 *
 * Includes umami_tables.h (written per library by umami_libraries: the helicity tables, the factor,
 * the colour index and the alpha_s constants), its tables __device__ under nvcc. Needs
 * RUNNER_OUTPUTS (runner.h). */

#ifndef UMAMI_FINISH_H
#define UMAMI_FINISH_H

#include <math.h>
#include <stddef.h>

#ifdef __CUDACC__
#define PAC_TABLE __device__ static const
#define PAC_FINISH_FN __device__ static inline
#else
#define PAC_TABLE static const
#define PAC_FINISH_FN static inline
#endif

#include "umami_tables.h"

/* One event: its runtime helicity values at values[h ld] (h < RUNNER_OUTPUTS), each × `scale` first
 * when `scaled` (the alpha_s rescale). fixed ≥ 0: the exported helicity `fixed` alone, × the
 * factor (one the program does not have contributes nothing); else the sum in order, × the factor,
 * and with `draw` the helicity drawn in proportion to its value at `random` (an exported index).
 * Writes *me and *label (`fixed`, the drawn helicity, or 0); returns 0, or -1 (a non-finite total). */
PAC_FINISH_FN int pac_finish(const double *values, size_t ld, int scaled, double scale, int fixed, double random,
                             int draw, double *me, int *label)
{
    int l = 0;
    double total = 0.0;
    if (fixed >= 0) {
        l = fixed;
        int runtime = pac_helicity_runtime[fixed];
        if (runtime >= 0) {
            double v = values[(size_t)runtime * ld];
            if (scaled) v *= scale;
            total += PAC_FACTOR * v;
        }
    } else {
        for (size_t h = 0; h < RUNNER_OUTPUTS; ++h) {
            double v = values[h * ld];
            if (scaled) v *= scale;
            total += v;
        }
        if (draw) {
            double target = random * total, cumulative = 0.0;
            size_t chosen = 0;
            for (size_t h = 0; h < RUNNER_OUTPUTS; ++h) {
                double v = values[h * ld];
                if (scaled) v *= scale;
                if (v <= 0.0) continue;
                chosen = h;
                cumulative += v;
                if (cumulative > target) break;
            }
            l = pac_helicity_export[chosen];
        }
        total *= PAC_FACTOR;
    }
    if (!isfinite(total)) return -1;
    *me = total;
    *label = l;
    return 0;
}

#endif
