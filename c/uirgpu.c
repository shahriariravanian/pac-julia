/* uirgpu.c: uirgpu.h on the slot pool (pool.h). Included by runner.c under nvcc after the
 * program's tables and uir_stub, with its sizes defined:
 *   UIRGPU_LEGS, UIRGPU_OUTPUTS, UIRGPU_FORMS, UIRGPU_AMPLITUDES, UIRGPU_PLANES
 * (legs; helicity outputs; momentum forms; amplitudes; planes = complex components), and the pool
 * the program asks for (the first program library of the process decides):
 *   UIRGPU_SLOTS (0: the online CPUs), UIRGPU_SLOT_BYTES (device bytes per slot).
 * Without PACPOOL_EXTERNAL the pool itself (pool.cu) is compiled in: a standalone
 * runner (tests); the UMAMI libraries link the process-wide libpacpool.so instead. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "uirgpu.h"
#include "pool.h"
#ifndef PACPOOL_EXTERNAL
#include "pool.cu"
#endif
#ifdef UIRGPU_UMAMI
#include "umami_finish.h"
#endif

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

/* bytes per point of each region */
static const size_t uirgpu_per_point[UIRGPU_REGIONS] = {
    UIRGPU_N_MOMENTA * sizeof(double), UIRGPU_N_OUT * sizeof(double), UIRGPU_N_FORMS * sizeof(double),
    UIRGPU_N_AMPLITUDES * sizeof(pac_amp), UIRGPU_N_PLANES * sizeof(pac_real)};

static size_t uirgpu_aligned(size_t bytes) { return (bytes + UIRGPU_ALIGN - 1) / UIRGPU_ALIGN * UIRGPU_ALIGN; }

/* the device bytes of an arena of `points` points (each region 256-byte aligned) */
static size_t uirgpu_arena_bytes(size_t points)
{
    size_t total = 0;
    for (int r = 0; r < UIRGPU_REGIONS; r++) total += uirgpu_aligned(uirgpu_per_point[r] * points);
    return total;
}

#define UIRGPU_CHECK(x)                                                                            \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "uirgpu (%s %s): %s: %s\n", RUNNER_PROCESS, RUNNER_COLOUR, #x,          \
                    cudaGetErrorString(e_));                                                       \
            status = -1;                                                                           \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

__global__ static void uirgpu_points(const double *momenta, double *out, double *forms, pac_amp *amplitudes,
                                     pac_real *planes, size_t points, int *status)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    uir_arena a = {forms, amplitudes, planes, points, i};
    int s = uir_run(&program, &a, momenta + i, out + i);
    if (s != 0) *status = s;   /* any failing point's status */
}

int pac_gpu_run(const double *momenta, double *out, size_t points)
{
    if (points == 0) return 0;
    pacpool_configure(UIRGPU_SLOTS, UIRGPU_SLOT_BYTES, PACPOOL_ABI);   /* the first library decides */
    const size_t d = sizeof(double), rows_m = 4 * (size_t)UIRGPU_LEGS, rows_o = UIRGPU_OUTPUTS;
    /* the most points of one launch within the slot's budget */
    size_t budget = pacpool_slot_bytes(), per_point = 0;
    for (int r = 0; r < UIRGPU_REGIONS; r++) per_point += uirgpu_per_point[r];
    size_t slack = UIRGPU_REGIONS * UIRGPU_ALIGN;
    size_t most = budget > slack ? (budget - slack) / per_point : 0;
    if (most == 0) {
        fprintf(stderr, "uirgpu (%s %s): a %zu-byte slot holds no point (%zu bytes each)\n", RUNNER_PROCESS,
                RUNNER_COLOUR, budget, per_point);
        return -1;
    }
    size_t chunk = points < most ? points : most;
    int status = 0;
    pacpool_slot *s = pacpool_acquire();
    if (s == NULL) return -1;
    cudaStream_t stream = (cudaStream_t)s->stream;
    if (pacpool_reserve(s, uirgpu_arena_bytes(chunk), (rows_m + rows_o) * points * d) != 0) {
        status = -1;
        goto done;
    }
    {
        /* the arena's regions, for launches of up to `chunk` points */
        void *region[UIRGPU_REGIONS];
        char *p = (char *)s->device;
        for (int r = 0; r < UIRGPU_REGIONS; r++) {
            region[r] = p;
            p += uirgpu_aligned(uirgpu_per_point[r] * chunk);
        }
        double *host_momenta = (double *)s->host, *host_out = host_momenta + rows_m * points;
        memcpy(host_momenta, momenta, rows_m * points * d);
        UIRGPU_CHECK(cudaMemsetAsync(s->status, 0, sizeof(int), stream));
        for (size_t first = 0; first < points; first += chunk) {
            size_t q = points - first < chunk ? points - first : chunk;   /* this launch's points: ld = q */
            if (rows_m > 0)
                UIRGPU_CHECK(cudaMemcpy2DAsync(region[0], q * d, host_momenta + first, points * d, q * d, rows_m,
                                               cudaMemcpyHostToDevice, stream));
            /* the arena at zero (uir_run writes the forms and the outputs) */
            UIRGPU_CHECK(cudaMemsetAsync(region[3], 0, uirgpu_per_point[3] * q, stream));
            UIRGPU_CHECK(cudaMemsetAsync(region[4], 0, uirgpu_per_point[4] * q, stream));
            uirgpu_points<<<(unsigned)((q + 127) / 128), 128, 0, stream>>>(
                (const double *)region[0], (double *)region[1], (double *)region[2], (pac_amp *)region[3],
                (pac_real *)region[4], q, s->status);
            UIRGPU_CHECK(cudaGetLastError());
            if (rows_o > 0)
                UIRGPU_CHECK(cudaMemcpy2DAsync(host_out + first, points * d, region[1], q * d, q * d, rows_o,
                                               cudaMemcpyDeviceToHost, stream));
        }
        UIRGPU_CHECK(cudaMemcpyAsync(s->host_status, s->status, sizeof(int), cudaMemcpyDeviceToHost, stream));
        UIRGPU_CHECK(cudaStreamSynchronize(stream));
        status = *s->host_status;
        if (status == 0) memcpy(out, host_out, rows_o * points * d);
    }
done:
    pacpool_release(s);
    return status;
}

#ifdef UIRGPU_UMAMI
/* per point of a UMAMI call, besides the arena: scale, random, me (double); fixed, label (int) */
#define UIRGPU_EXTRAS 5
static const size_t uirgpu_extra_per_point[UIRGPU_EXTRAS] = {sizeof(double), sizeof(double), sizeof(double),
                                                              sizeof(int), sizeof(int)};

__global__ static void uirgpu_umami_points(const double *momenta, double *values, double *forms, pac_amp *amplitudes,
                                           pac_real *planes, size_t points, int *status, const double *scale,
                                           const int *fixed, const double *random, int draw, double *me, int *label)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    for (size_t r = 0; r < 4 * (size_t)UIRGPU_LEGS; r++)
        if (!isfinite(momenta[r * points + i])) {
            *status = -2;
            return;
        }
    uir_arena a = {forms, amplitudes, planes, points, i};
    /* MadSpace's layout: leg step 1, component step n (rows n k + p, ld = the launch's points) */
    int s = uir_run_layout(&program, &a, momenta + i, 1, UIRGPU_LEGS, values + i);
    if (s != 0) {
        *status = s;
        return;
    }
    double m;
    int l;
    if (pac_finish(values + i, points, scale != NULL, scale ? scale[i] : 1.0, fixed ? fixed[i] : -1,
                   random ? random[i] : 0.5, draw, &m, &l) != 0) {
        *status = -2;
        return;
    }
    me[i] = m;
    label[i] = l;
}

int pac_gpu_umami(const double *momenta, size_t stride, size_t count, const double *scale, const int *fixed,
                  const double *random, int draw, double *me, int *label)
{
    if (count == 0) return 0;
    pacpool_configure(UIRGPU_SLOTS, UIRGPU_SLOT_BYTES, PACPOOL_ABI);   /* the first library decides */
    const size_t d = sizeof(double), rows_m = 4 * (size_t)UIRGPU_LEGS;
    size_t budget = pacpool_slot_bytes(), per_point = 0;
    for (int r = 0; r < UIRGPU_REGIONS; r++) per_point += uirgpu_per_point[r];
    for (int r = 0; r < UIRGPU_EXTRAS; r++) per_point += uirgpu_extra_per_point[r];
    size_t slack = (UIRGPU_REGIONS + UIRGPU_EXTRAS) * UIRGPU_ALIGN;
    size_t most = budget > slack ? (budget - slack) / per_point : 0;
    if (most == 0) {
        fprintf(stderr, "uirgpu (%s %s): a %zu-byte slot holds no point (%zu bytes each)\n", RUNNER_PROCESS,
                RUNNER_COLOUR, budget, per_point);
        return -1;
    }
    size_t chunk = count < most ? count : most;
    size_t device_bytes = uirgpu_arena_bytes(chunk);
    for (int r = 0; r < UIRGPU_EXTRAS; r++) device_bytes += uirgpu_aligned(uirgpu_extra_per_point[r] * chunk);
    /* host staging: momenta rows, scale, random, me (double); fixed, label (int) */
    size_t host_bytes = (rows_m + 3) * count * d + 2 * count * sizeof(int);
    int status = 0;
    pacpool_slot *s = pacpool_acquire();
    if (s == NULL) return -1;
    cudaStream_t stream = (cudaStream_t)s->stream;
    if (pacpool_reserve(s, device_bytes, host_bytes) != 0) {
        status = -1;
        goto done;
    }
    {
        void *region[UIRGPU_REGIONS + UIRGPU_EXTRAS];
        char *p = (char *)s->device;
        for (int r = 0; r < UIRGPU_REGIONS; r++) {
            region[r] = p;
            p += uirgpu_aligned(uirgpu_per_point[r] * chunk);
        }
        for (int r = 0; r < UIRGPU_EXTRAS; r++) {
            region[UIRGPU_REGIONS + r] = p;
            p += uirgpu_aligned(uirgpu_extra_per_point[r] * chunk);
        }
        double *d_scale = (double *)region[UIRGPU_REGIONS], *d_random = (double *)region[UIRGPU_REGIONS + 1];
        double *d_me = (double *)region[UIRGPU_REGIONS + 2];
        int *d_fixed = (int *)region[UIRGPU_REGIONS + 3], *d_label = (int *)region[UIRGPU_REGIONS + 4];
        double *h_m = (double *)s->host, *h_scale = h_m + rows_m * count, *h_random = h_scale + count;
        double *h_me = h_random + count;
        int *h_fixed = (int *)(h_me + count), *h_label = h_fixed + count;
        /* the call's rows, contiguous: no transpose (row n k + p of `count` events) */
        for (size_t r = 0; r < rows_m; r++) memcpy(h_m + r * count, momenta + r * stride, count * d);
        if (scale) memcpy(h_scale, scale, count * d);
        if (random) memcpy(h_random, random, count * d);
        if (fixed) memcpy(h_fixed, fixed, count * sizeof(int));
        UIRGPU_CHECK(cudaMemsetAsync(s->status, 0, sizeof(int), stream));
        for (size_t first = 0; first < count; first += chunk) {
            size_t q = count - first < chunk ? count - first : chunk;   /* this launch's points: ld = q */
            UIRGPU_CHECK(cudaMemcpy2DAsync(region[0], q * d, h_m + first, count * d, q * d, rows_m,
                                           cudaMemcpyHostToDevice, stream));
            if (scale) UIRGPU_CHECK(cudaMemcpyAsync(d_scale, h_scale + first, q * d, cudaMemcpyHostToDevice, stream));
            if (random) UIRGPU_CHECK(cudaMemcpyAsync(d_random, h_random + first, q * d, cudaMemcpyHostToDevice, stream));
            if (fixed)
                UIRGPU_CHECK(cudaMemcpyAsync(d_fixed, h_fixed + first, q * sizeof(int), cudaMemcpyHostToDevice, stream));
            UIRGPU_CHECK(cudaMemsetAsync(region[3], 0, uirgpu_per_point[3] * q, stream));
            UIRGPU_CHECK(cudaMemsetAsync(region[4], 0, uirgpu_per_point[4] * q, stream));
            uirgpu_umami_points<<<(unsigned)((q + 127) / 128), 128, 0, stream>>>(
                (const double *)region[0], (double *)region[1], (double *)region[2], (pac_amp *)region[3],
                (pac_real *)region[4], q, s->status, scale ? d_scale : NULL, fixed ? d_fixed : NULL,
                random ? d_random : NULL, draw, d_me, d_label);
            UIRGPU_CHECK(cudaGetLastError());
            if (me) UIRGPU_CHECK(cudaMemcpyAsync(h_me + first, d_me, q * d, cudaMemcpyDeviceToHost, stream));
            if (label)
                UIRGPU_CHECK(cudaMemcpyAsync(h_label + first, d_label, q * sizeof(int), cudaMemcpyDeviceToHost, stream));
        }
        UIRGPU_CHECK(cudaMemcpyAsync(s->host_status, s->status, sizeof(int), cudaMemcpyDeviceToHost, stream));
        UIRGPU_CHECK(cudaStreamSynchronize(stream));
        status = *s->host_status;
        if (status == 0) {
            if (me) memcpy(me, h_me, count * d);
            if (label) memcpy(label, h_label, count * sizeof(int));
        }
    }
done:
    pacpool_release(s);
    return status;
}
#endif

int run_gpu(double *out, const double *momenta, size_t points)
{
    if (points == 0) return 0;
    size_t nm = 4 * (size_t)UIRGPU_LEGS, no = UIRGPU_OUTPUTS;
    double *m = (double *)malloc(UIRGPU_MAX1(nm) * points * sizeof(double));
    double *o = (double *)malloc(UIRGPU_MAX1(no) * points * sizeof(double));
    int status = -1;
    if (m != NULL && o != NULL) {
        for (size_t p = 0; p < points; p++)
            for (size_t k = 0; k < nm; k++) m[k * points + p] = momenta[p * nm + k];
        status = pac_gpu_run(m, o, points);
        if (status == 0)
            for (size_t p = 0; p < points; p++)
                for (size_t h = 0; h < no; h++) out[p * no + h] = o[h * points + p];
    }
    free(m);
    free(o);
    return status;
}
