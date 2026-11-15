/* uirgpu_pool.c: the entries of uirgpu.h on the slot pool (pool.h): pac_gpu_run and the test entry
 * run_gpu. A call borrows a free slot, carves the program's arena out of the slot's device block,
 * runs on the slot's stream, and gives the slot back. Included by uirgpu.c (not UIRGPU_PROVIDER),
 * after the regions and their carving. Without PACPOOL_EXTERNAL the pool itself (pool.cu) is
 * compiled in (a standalone runner); the UMAMI libraries link libpacpool.so instead. */

#include "pool.h"
#ifndef PACPOOL_EXTERNAL
#include "pool.cu"
#endif

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
    size_t chunk = uirgpu_chunk(uirgpu_per_point, UIRGPU_REGIONS, pacpool_slot_bytes(), points);
    if (chunk == 0) return -1;
    int status = 0;
    pacpool_slot *s = pacpool_acquire();
    if (s == NULL) return -1;
    cudaStream_t stream = (cudaStream_t)s->stream;
    if (pacpool_reserve(s, uirgpu_bytes(uirgpu_per_point, UIRGPU_REGIONS, chunk), (rows_m + rows_o) * points * d) != 0) {
        status = -1;
        goto done;
    }
    {
        /* the arena's regions, for launches of up to `chunk` points */
        void *region[UIRGPU_REGIONS];
        uirgpu_carve(s->device, uirgpu_per_point, UIRGPU_REGIONS, chunk, region);
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
