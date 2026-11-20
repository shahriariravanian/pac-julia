/* uirgpu_umami.c: pac_gpu_umami (uirgpu.h), a UMAMI library's call on the slot pool: MadSpace's
 * momentum layout, the program, and the event's end on the device (umami_finish.h's pac_finish).
 * Included by uirgpu.c after uirgpu_pool.c under UIRGPU_UMAMI. */

#include "umami_finish.h"

/* per point of a UMAMI call: the arena's regions, then scale, random, me (double); fixed, label (int) */
#define UIRGPU_EXTRAS 5
#define UIRGPU_UMAMI_REGIONS (UIRGPU_REGIONS + UIRGPU_EXTRAS)
static const size_t uirgpu_umami_per_point[UIRGPU_UMAMI_REGIONS] = {
    UIRGPU_N_MOMENTA * sizeof(double), UIRGPU_N_OUT * sizeof(double), UIRGPU_N_FORMS * sizeof(double),
    UIRGPU_N_AMPLITUDES * sizeof(pac_amp), UIRGPU_N_PLANES * sizeof(pac_real),
    sizeof(double), sizeof(double), sizeof(double), sizeof(int), sizeof(int)};

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
    int s = uir_run_layout(&program, &a, momenta + i, points, (size_t)UIRGPU_LEGS * points, values + i);
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
    size_t chunk = uirgpu_chunk(uirgpu_umami_per_point, UIRGPU_UMAMI_REGIONS, pacpool_slot_bytes(), count);
    if (chunk == 0) return -1;
    size_t device_bytes = uirgpu_bytes(uirgpu_umami_per_point, UIRGPU_UMAMI_REGIONS, chunk);
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
        void *region[UIRGPU_UMAMI_REGIONS];
        uirgpu_carve(s->device, uirgpu_umami_per_point, UIRGPU_UMAMI_REGIONS, chunk, region);
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
