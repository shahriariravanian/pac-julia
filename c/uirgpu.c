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
#ifndef UIRGPU_PROVIDER
#include "pool.h"
#ifndef PACPOOL_EXTERNAL
#include "pool.cu"
#endif
#endif
#if defined(UIRGPU_UMAMI) || defined(UIRGPU_PROVIDER)
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

#ifndef UIRGPU_PROVIDER   /* the slot pool's entries (the CPU provider, the tests) */
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

#ifdef UIRGPU_UMAMI
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
#endif   /* !UIRGPU_PROVIDER */

#ifdef UIRGPU_PROVIDER
/* ---- the CUDA provider: MadSpace's device pointers and stream ------------------------------- */

/* one event of a call: the checks (pac_check), the program, pac_finish, the outputs. A failing
 * event's matrix element is NaN, its helicity 0, and its code (PAC_E_) goes to *error. */
__global__ static void uirgpu_provider_points(uirgpu_call c, size_t first, double *values, double *forms,
                                              pac_amp *amplitudes, pac_real *planes, size_t points, int *error)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    size_t e = first + i;   /* the event, from the call's first */
    double random = c.random ? c.random[e] : 0.5, scale, me = NAN;
    int fixed = c.fixed ? c.fixed[e] : -1, scaled, label = 0;
    int code = pac_check(c.flavor != NULL, c.flavor ? c.flavor[e] : 0, random, c.fixed != NULL, fixed,
                         c.alpha_s != NULL, c.alpha_s ? c.alpha_s[e] : 0.0, &scale, &scaled);
    const double *m = c.momenta + e;
    if (code == 0)
        for (size_t r = 0; r < 4 * (size_t)UIRGPU_LEGS; r++)
            if (!isfinite(m[r * c.stride])) code = PAC_E_MOMENTA;
    if (code == 0) {
        uir_arena a = {forms, amplitudes, planes, points, i};
        /* the caller's layout in place: component k of particle p at [stride (n k + p) + event] */
        if (uir_run_layout(&program, &a, m, c.stride, (size_t)UIRGPU_LEGS * c.stride, values + i) != 0)
            code = PAC_E_KERNEL;
        else if (pac_finish(values + i, points, scaled, scale, fixed, random, c.draw, &me, &label) != 0)
            code = PAC_E_RESULT;
    }
    if (code != 0) {
        *error = code;
        me = NAN;
        label = 0;
    }
    if (c.me) c.me[e] = me;
    if (c.label) c.label[e] = label;
    if (c.colour) c.colour[e] = PAC_COLOR_INDEX;
}

#define UIRGPU_REPORT(x) fprintf(stderr, "uirgpu (%s %s): %s: %s\n", RUNNER_PROCESS, RUNNER_COLOUR, #x, cudaGetErrorString(e_))
#define UIRGPU_TRY(x)                                                                              \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            UIRGPU_REPORT(x);                                                                      \
            status = -1;                                                                           \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int *pac_gpu_error_flag(void)
{
    int *flag = NULL;
    if (cudaHostAlloc((void **)&flag, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return NULL;
    *flag = 0;
    return flag;
}

void pac_gpu_free_error_flag(int *flag)
{
    if (flag) cudaFreeHost(flag);
}

/* the arena without the momenta (read in place): values, forms, amplitudes, planes */
#define UIRGPU_WORK (UIRGPU_REGIONS - 1)

int pac_gpu_provider(const uirgpu_call *call, size_t count, void *stream_, int *error)
{
    if (count == 0) return 0;
    cudaStream_t stream = (cudaStream_t)stream_;
    const size_t *per_point = uirgpu_per_point + 1;
    int status = 0, before = -1, device = -1;
    char *workspace = NULL;
    /* the stream's GPU for this call (the default stream: the current one); restored below */
    UIRGPU_TRY(cudaGetDevice(&before));
    device = before;
    if (stream != NULL) UIRGPU_TRY(cudaStreamGetDevice(stream, &device));
    if (device != before) UIRGPU_TRY(cudaSetDevice(device));
    {
        int *error_device;
        UIRGPU_TRY(cudaHostGetDevicePointer((void **)&error_device, error, 0));
        size_t chunk = uirgpu_chunk(per_point, UIRGPU_WORK, UIRGPU_SLOT_BYTES, count);
        if (chunk == 0) {
            status = -1;
            goto done;
        }
        UIRGPU_TRY(cudaMallocAsync((void **)&workspace, uirgpu_bytes(per_point, UIRGPU_WORK, chunk), stream));
        void *region[UIRGPU_WORK];   /* values, forms, amplitudes, planes */
        uirgpu_carve(workspace, per_point, UIRGPU_WORK, chunk, region);
        for (size_t first = 0; first < count; first += chunk) {
            size_t q = count - first < chunk ? count - first : chunk;   /* this launch's points: the arena's ld */
            UIRGPU_TRY(cudaMemsetAsync(region[2], 0, per_point[2] * q, stream));
            UIRGPU_TRY(cudaMemsetAsync(region[3], 0, per_point[3] * q, stream));
            uirgpu_provider_points<<<(unsigned)((q + 127) / 128), 128, 0, stream>>>(
                *call, first, (double *)region[0], (double *)region[1], (pac_amp *)region[2], (pac_real *)region[3], q,
                error_device);
            UIRGPU_TRY(cudaGetLastError());
        }
    }
done:
    /* freed in stream order, after the launches queued so far; also on a failure */
    if (workspace != NULL) {
        cudaError_t e_ = cudaFreeAsync(workspace, stream);
        if (e_ != cudaSuccess) {
            UIRGPU_REPORT(cudaFreeAsync(workspace, stream));
            status = -1;
        }
    }
    if (before >= 0 && device != before) cudaSetDevice(before);
    return status;
}
#endif
