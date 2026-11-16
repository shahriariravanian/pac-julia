/* umami_finish.h: the end of one event in PacRunner's UMAMI libraries, shared by the CPU build
 * (umami_pac.c, plain C) and the GPU build (uirgpu.c under nvcc with UIRGPU_UMAMI, a __device__
 * function run by the point's own thread right after the program): from the program's runtime
 * helicity values to the matrix element and the helicity index. The operations are those of the
 * CPU wrapper in the same order, so both builds give the same bits.
 *
 * pac_check is the events' input checks and alpha_s scale, shared the same way.
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

/* why an event fails (0: it does not) */
enum { PAC_E_FLAVOR = 1, PAC_E_RANDOM, PAC_E_HELICITY, PAC_E_ALPHA_S, PAC_E_MOMENTA, PAC_E_KERNEL, PAC_E_RESULT };

PAC_FINISH_FN int pac_close_to(double x, double y)
{
    return x == y || fabs(x - y) <= 1e-12 * fmax(fabs(x), fabs(y));
}

/* One event's inputs (has_*: the input is present): the flavour must be 0, the random number in
 * [0, 1), a fixed helicity an exported one, alpha_s finite and not negative. *scale, *scaled: the
 * alpha_s rescale, (alpha_s / alpha_s0)^power when alpha_s is not the exported one and the power is
 * proven; without a proven power only the exported alpha_s (relative 1e-12) passes. Returns 0 or a
 * PAC_E_ code. */
PAC_FINISH_FN int pac_check(int has_flavor, int flavor, double random, int has_fixed, int fixed, int has_alpha_s,
                            double alpha_s, double *scale, int *scaled)
{
    *scale = 1.0;
    *scaled = 0;
    if (has_flavor && flavor != 0) return PAC_E_FLAVOR;
    if (!isfinite(random) || random < 0.0 || random >= 1.0) return PAC_E_RANDOM;
    if (has_fixed && (fixed < 0 || fixed >= PAC_EXPORTED_HELICITY_COUNT)) return PAC_E_HELICITY;
    if (has_alpha_s) {
        if (!isfinite(alpha_s) || alpha_s < 0.0) return PAC_E_ALPHA_S;
        if (alpha_s != PAC_ALPHA_S0) {
            if (PAC_ALPHA_S_SCALES) {
                *scale = pow(alpha_s / PAC_ALPHA_S0, (double)PAC_ALPHA_S_POWER);
                *scaled = 1;
            } else if (!pac_close_to(alpha_s, PAC_ALPHA_S0)) {
                return PAC_E_ALPHA_S;
            }
        }
    }
    return 0;
}

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
