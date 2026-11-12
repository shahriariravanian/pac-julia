/* uirgpu.c: the context of uirgpu.h. Included by runner.c under nvcc after the program's tables and
 * uir_stub, with its sizes defined:
 *   UIRGPU_LEGS, UIRGPU_OUTPUTS, UIRGPU_FORMS, UIRGPU_AMPLITUDES, UIRGPU_PLANES
 * (legs; helicity outputs; momentum forms; amplitudes; planes = complex components), and
 *   UIRGPU_MAX_POINTS
 * the most points of one launch: the device buffers never hold more (their memory is about
 * UIRGPU_MAX_POINTS × the arena's doubles × 8 bytes); a larger call runs as several launches. The
 * values do not depend on the split: every point is computed alone. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "uirgpu.h"

/* doubles per point of each buffer (at least 1: no empty allocations) */
#define UIRGPU_MAX1(x) ((x) > 0 ? (x) : 1)
#define UIRGPU_N_MOMENTA UIRGPU_MAX1(4 * UIRGPU_LEGS)
#define UIRGPU_N_OUT UIRGPU_MAX1(UIRGPU_OUTPUTS)
#define UIRGPU_N_FORMS UIRGPU_MAX1(4 * UIRGPU_FORMS)
#define UIRGPU_N_AMPLITUDES UIRGPU_MAX1(2 * UIRGPU_AMPLITUDES)
#define UIRGPU_N_PLANES UIRGPU_MAX1(2 * UIRGPU_PLANES)

struct pac_gpu {
    cudaStream_t stream;
    size_t device_capacity;                                  /* points of one launch the device buffers hold */
    size_t host_capacity;                                    /* points of one call the staging holds */
    double *momenta, *out, *forms, *amplitudes, *planes;     /* device */
    int *status;                                             /* device */
    double *host_momenta, *host_out;                         /* pinned staging, [rows][points] */
    int *host_status;                                        /* pinned */
};

#define UIRGPU_CHECK(x)                                                                            \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "uirgpu (%s %s): %s: %s\n", RUNNER_PROCESS, RUNNER_COLOUR, #x,          \
                    cudaGetErrorString(e_));                                                       \
            return -1;                                                                             \
        }                                                                                          \
    } while (0)

__global__ static void uirgpu_points(const double *momenta, double *out, double *forms, double *amplitudes,
                                     double *planes, size_t points, int *status)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    uir_arena a = {forms, amplitudes, planes, points, i};
    int s = uir_run(&program, &a, momenta + i, out + i);
    if (s != 0) *status = s;   /* any failing point's status */
}

static void uirgpu_release_device(pac_gpu *c)
{
    cudaFree(c->momenta); cudaFree(c->out); cudaFree(c->forms); cudaFree(c->amplitudes); cudaFree(c->planes);
    c->momenta = c->out = c->forms = c->amplitudes = c->planes = NULL;
    c->device_capacity = 0;
}

static void uirgpu_release_host(pac_gpu *c)
{
    cudaFreeHost(c->host_momenta); cudaFreeHost(c->host_out);
    c->host_momenta = c->host_out = NULL;
    c->host_capacity = 0;
}

/* capacity for n points: grown geometrically from `have`, at least 256, at most `most` */
static size_t uirgpu_grow(size_t have, size_t n, size_t most)
{
    size_t m = have * 2 > n ? have * 2 : n;
    if (m < 256) m = 256;
    return m < most ? m : most;
}

/* device buffers for launches of `points` (≤ UIRGPU_MAX_POINTS); cudaMalloc synchronises the device */
static int uirgpu_reserve_device(pac_gpu *c, size_t points)
{
    if (points <= c->device_capacity) return 0;
    size_t n = uirgpu_grow(c->device_capacity, points, UIRGPU_MAX_POINTS);
    UIRGPU_CHECK(cudaStreamSynchronize(c->stream));
    uirgpu_release_device(c);
    UIRGPU_CHECK(cudaMalloc((void **)&c->momenta, UIRGPU_N_MOMENTA * n * sizeof(double)));
    UIRGPU_CHECK(cudaMalloc((void **)&c->out, UIRGPU_N_OUT * n * sizeof(double)));
    UIRGPU_CHECK(cudaMalloc((void **)&c->forms, UIRGPU_N_FORMS * n * sizeof(double)));
    UIRGPU_CHECK(cudaMalloc((void **)&c->amplitudes, UIRGPU_N_AMPLITUDES * n * sizeof(double)));
    UIRGPU_CHECK(cudaMalloc((void **)&c->planes, UIRGPU_N_PLANES * n * sizeof(double)));
    c->device_capacity = n;
    return 0;
}

/* pinned staging for a call of `points` (momenta and outputs only: small) */
static int uirgpu_reserve_host(pac_gpu *c, size_t points)
{
    if (points <= c->host_capacity) return 0;
    size_t n = uirgpu_grow(c->host_capacity, points, (size_t)-1);
    uirgpu_release_host(c);
    UIRGPU_CHECK(cudaMallocHost((void **)&c->host_momenta, UIRGPU_N_MOMENTA * n * sizeof(double)));
    UIRGPU_CHECK(cudaMallocHost((void **)&c->host_out, UIRGPU_N_OUT * n * sizeof(double)));
    c->host_capacity = n;
    return 0;
}

int pac_gpu_create(pac_gpu **ctx)
{
    *ctx = NULL;
    pac_gpu *c = (pac_gpu *)calloc(1, sizeof(pac_gpu));
    if (c == NULL) return -1;
#ifdef UIRGPU_STREAM_PER_THREAD
    /* the calling thread's default stream (nvcc --default-stream per-thread): one GPU stream per CPU
     * thread, so a context must be used by one thread (the UMAMI wrapper: one per instance) */
    c->stream = cudaStreamPerThread;
    cudaError_t e = cudaSuccess;
#else
    cudaError_t e = cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking);
#endif
    if (e == cudaSuccess) e = cudaMalloc((void **)&c->status, sizeof(int));
    if (e == cudaSuccess) e = cudaMallocHost((void **)&c->host_status, sizeof(int));
    if (e != cudaSuccess) {
        fprintf(stderr, "uirgpu (%s %s): creating the context: %s\n", RUNNER_PROCESS, RUNNER_COLOUR,
                cudaGetErrorString(e));
        pac_gpu_destroy(c);
        return -1;
    }
    *ctx = c;
    return 0;
}

void pac_gpu_destroy(pac_gpu *c)
{
    if (c == NULL) return;
#ifdef UIRGPU_STREAM_PER_THREAD
    cudaStreamSynchronize(c->stream);
#else
    if (c->stream) cudaStreamSynchronize(c->stream);
#endif
    uirgpu_release_device(c);
    uirgpu_release_host(c);
    cudaFree(c->status);
    cudaFreeHost(c->host_status);
#ifndef UIRGPU_STREAM_PER_THREAD
    if (c->stream) cudaStreamDestroy(c->stream);
#endif
    free(c);
}

int pac_gpu_run(pac_gpu *c, const double *momenta, double *out, size_t points)
{
    if (points == 0) return 0;
    size_t chunk = points < (size_t)UIRGPU_MAX_POINTS ? points : (size_t)UIRGPU_MAX_POINTS;
    if (uirgpu_reserve_host(c, points) != 0 || uirgpu_reserve_device(c, chunk) != 0) return -1;
    const size_t d = sizeof(double), rows_m = 4 * (size_t)UIRGPU_LEGS, rows_o = UIRGPU_OUTPUTS;
    memcpy(c->host_momenta, momenta, rows_m * points * d);
    UIRGPU_CHECK(cudaMemsetAsync(c->status, 0, sizeof(int), c->stream));
    for (size_t first = 0; first < points; first += chunk) {
        size_t q = points - first < chunk ? points - first : chunk;   /* this launch's points: ld = q */
        if (rows_m > 0)
            UIRGPU_CHECK(cudaMemcpy2DAsync(c->momenta, q * d, c->host_momenta + first, points * d, q * d, rows_m,
                                           cudaMemcpyHostToDevice, c->stream));
        /* the arena at zero (uir_run writes the forms and the outputs) */
        UIRGPU_CHECK(cudaMemsetAsync(c->amplitudes, 0, UIRGPU_N_AMPLITUDES * q * d, c->stream));
        UIRGPU_CHECK(cudaMemsetAsync(c->planes, 0, UIRGPU_N_PLANES * q * d, c->stream));
        uirgpu_points<<<(unsigned)((q + 127) / 128), 128, 0, c->stream>>>(
            c->momenta, c->out, c->forms, c->amplitudes, c->planes, q, c->status);
        UIRGPU_CHECK(cudaGetLastError());
        if (rows_o > 0)
            UIRGPU_CHECK(cudaMemcpy2DAsync(c->host_out + first, points * d, c->out, q * d, q * d, rows_o,
                                           cudaMemcpyDeviceToHost, c->stream));
    }
    UIRGPU_CHECK(cudaMemcpyAsync(c->host_status, c->status, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    UIRGPU_CHECK(cudaStreamSynchronize(c->stream));
    if (*c->host_status != 0) return *c->host_status;
    memcpy(out, c->host_out, rows_o * points * d);
    return 0;
}

int run_gpu(double *out, const double *momenta, size_t points)
{
    if (points == 0) return 0;
    size_t nm = 4 * (size_t)UIRGPU_LEGS, no = UIRGPU_OUTPUTS;
    double *m = (double *)malloc(UIRGPU_MAX1(nm) * points * sizeof(double));
    double *o = (double *)malloc(UIRGPU_MAX1(no) * points * sizeof(double));
    pac_gpu *c = NULL;
    int status = -1;
    if (m == NULL || o == NULL || pac_gpu_create(&c) != 0) goto done;
    for (size_t p = 0; p < points; p++)
        for (size_t k = 0; k < nm; k++) m[k * points + p] = momenta[p * nm + k];
    status = pac_gpu_run(c, m, o, points);
    if (status == 0)
        for (size_t p = 0; p < points; p++)
            for (size_t h = 0; h < no; h++) out[p * no + h] = o[h * points + p];
done:
    pac_gpu_destroy(c);
    free(m);
    free(o);
    return status;
}
