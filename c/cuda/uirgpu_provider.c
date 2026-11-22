/* uirgpu_provider.c: pac_gpu_provider (uirgpu.h), a UMAMI library that identifies as a CUDA
 * provider: MadSpace's device pointers and stream, no slot pool, nothing synchronized; the
 * workspace (at most UIRGPU_SLOT_BYTES a launch) is allocated and freed in stream order. Failing
 * events are reported through a mapped pinned error flag (pac_gpu_error_flag). Included by
 * uirgpu.c under UIRGPU_PROVIDER. */

#include "umami_finish.h"

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
    UIRGPU_CHECK(cudaGetDevice(&before));
    device = before;
    if (stream != NULL) UIRGPU_CHECK(cudaStreamGetDevice(stream, &device));
    if (device != before) UIRGPU_CHECK(cudaSetDevice(device));
    {
        int *error_device;
        UIRGPU_CHECK(cudaHostGetDevicePointer((void **)&error_device, error, 0));
        size_t chunk = uirgpu_chunk(per_point, UIRGPU_WORK, UIRGPU_SLOT_BYTES, count);
        if (chunk == 0) {
            status = -1;
            goto done;
        }
        UIRGPU_CHECK(cudaMallocAsync((void **)&workspace, uirgpu_bytes(per_point, UIRGPU_WORK, chunk), stream));
        void *region[UIRGPU_WORK];   /* values, forms, amplitudes, planes */
        uirgpu_carve(workspace, per_point, UIRGPU_WORK, chunk, region);
        for (size_t first = 0; first < count; first += chunk) {
            size_t q = count - first < chunk ? count - first : chunk;   /* this launch's points: the arena's ld */
            UIRGPU_CHECK(cudaMemsetAsync(region[2], 0, per_point[2] * q, stream));
            UIRGPU_CHECK(cudaMemsetAsync(region[3], 0, per_point[3] * q, stream));
            uirgpu_provider_points<<<(unsigned)((q + 127) / 128), 128, 0, stream>>>(
                *call, first, (double *)region[0], (double *)region[1], (pac_amp *)region[2], (pac_real *)region[3], q,
                error_device);
            UIRGPU_CHECK(cudaGetLastError());
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
