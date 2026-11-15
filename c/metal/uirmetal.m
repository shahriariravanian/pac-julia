/* uirmetal.m: uirmetal.h on Metal (macOS, Apple GPUs), on the process-wide slot pool (metalpool.h).
 * Built with the program's runner_metal.c (compile_metal), and either with metalpool.m (a standalone
 * folder) or against libpacmetal.dylib (the UMAMI libraries: one pool per process):
 *   cc -O2 -ffp-contract=off -fobjc-arc -framework Metal -framework Foundation.
 *
 * Once per library (pthread_once): the pool configured, the program's source compiled
 * (newLibraryWithSource, MTLMathModeSafe), its uir_points pipeline, the program in a shared buffer.
 * Per call: a slot of the pool, carved into the call's momenta (df64 rows) and outputs, the status
 * word and the arena of one chunk (each region 256-byte aligned); one command buffer with one
 * serial compute encoder, a dispatch per chunk of at most `chunk` points (the arena holds one
 * chunk; serial dispatches do not overlap); commit, wait; the status and the outputs read back.
 * No Objective-C classes: several program libraries live in one process.
 *
 * Environment: PAC_METAL_CHUNK (points per launch, default 4096: the M4's best), PAC_METAL_SLOT_BYTES
 * (the arena's budget, default 256 MiB: a smaller chunk when a point's arena is large),
 * PAC_METAL_SLOTS (metalpool.h). */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runner_metal.h"
#include "metalpool.h"

#define UIRMETAL_MAX1(x) ((x) > 0 ? (size_t)(x) : (size_t)1)
#define UIRMETAL_ALIGN ((size_t)256)

/* uirvm.metal's uir_launch */
typedef struct {
    uint32_t points, first, leg_step, component_step, out_step;
} uirmetal_launch;

static pthread_once_t uirmetal_once = PTHREAD_ONCE_INIT;
static int uirmetal_ok = 0;
static id<MTLComputePipelineState> uirmetal_pipeline = nil;
static id<MTLBuffer> uirmetal_program = nil;
static size_t uirmetal_chunk = 4096;

/* bytes per point: the call's momenta (4 legs × 4 df64) and outputs (df64); the arena's forms and
 * amplitudes (df64) and planes (f32) */
static const size_t uirmetal_momenta_bytes = UIRMETAL_MAX1(4 * RUNNER_LEGS) * 8;
static const size_t uirmetal_out_bytes = UIRMETAL_MAX1(RUNNER_OUTPUTS) * 8;
static const size_t uirmetal_forms_bytes = UIRMETAL_MAX1(4 * RUNNER_FORMS) * 8;
static const size_t uirmetal_amplitudes_bytes = UIRMETAL_MAX1(2 * RUNNER_AMPLITUDES) * 8;
static const size_t uirmetal_planes_bytes = UIRMETAL_MAX1(2 * RUNNER_PLANES) * 4;

static size_t uirmetal_aligned(size_t bytes) { return (bytes + UIRMETAL_ALIGN - 1) / UIRMETAL_ALIGN * UIRMETAL_ALIGN; }

static void uirmetal_error(const char *what, NSError *error)
{
    fprintf(stderr, "uirmetal (%s %s): %s%s%s\n", RUNNER_PROCESS, RUNNER_COLOUR, what, error ? ": " : "",
            error ? error.localizedDescription.UTF8String : "");
}

static size_t uirmetal_env(const char *name, size_t fallback)
{
    const char *s = getenv(name);
    if (s == NULL || *s == 0) return fallback;
    char *end;
    unsigned long long v = strtoull(s, &end, 10);
    return *end == 0 && v > 0 ? (size_t)v : fallback;
}

static void uirmetal_init(void)
{
    @autoreleasepool {
        if (pacmetal_configure(0, PACMETAL_ABI) != 0) return;
        id<MTLDevice> device = (__bridge id<MTLDevice>)pacmetal_device();
        NSString *source = [[NSString alloc] initWithBytes:uir_metal_source length:uir_metal_source_size
                                                  encoding:NSUTF8StringEncoding];
        MTLCompileOptions *options = [MTLCompileOptions new];
        /* IEEE f32: no fast math, no reassociation (df64.h needs them); no contraction comes from
         * the source's `#pragma METAL fp contract(off)` (compile_metal): Safe alone contracts */
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil) { uirmetal_error("the program's source does not compile", error); return; }
        id<MTLFunction> function = [library newFunctionWithName:@"uir_points"];
        if (function == nil) { uirmetal_error("no function uir_points", nil); return; }
        uirmetal_pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (uirmetal_pipeline == nil) { uirmetal_error("no pipeline", error); return; }
        uirmetal_program = [device newBufferWithBytes:uir_metal_program length:uir_metal_program_size
                                              options:MTLResourceStorageModeShared];
        if (uirmetal_program == nil) { uirmetal_error("no program buffer", nil); return; }

        size_t per_point = uirmetal_forms_bytes + uirmetal_amplitudes_bytes + uirmetal_planes_bytes;
        size_t budget = uirmetal_env("PAC_METAL_SLOT_BYTES", (size_t)256 << 20);
        uirmetal_chunk = uirmetal_env("PAC_METAL_CHUNK", 4096);
        if (uirmetal_chunk > budget / per_point) uirmetal_chunk = budget / per_point;
        if (uirmetal_chunk == 0) {
            fprintf(stderr, "uirmetal (%s %s): %zu bytes hold no point (%zu bytes each)\n", RUNNER_PROCESS,
                    RUNNER_COLOUR, budget, per_point);
            return;
        }
        uirmetal_ok = 1;
    }
}

/* `count` events whose momentum row r (4 legs × 4 components, in the caller's order) starts at
 * momenta + r * row_stride, contiguous over the events; component j of leg l is row
 * leg_row * l + component_row * j. values[h count + e]: the outputs. */
static int uirmetal_call(const double *momenta, size_t row_stride, size_t leg_row, size_t component_row, size_t count,
                         double *values)
{
    if (count == 0) return 0;
    pthread_once(&uirmetal_once, uirmetal_init);
    if (!uirmetal_ok) return -1;
    if (count > UINT32_MAX / UIRMETAL_MAX1(4 * RUNNER_LEGS) || count > UINT32_MAX / UIRMETAL_MAX1(RUNNER_OUTPUTS)) {
        fprintf(stderr, "uirmetal (%s %s): %zu events in one call (too many)\n", RUNNER_PROCESS, RUNNER_COLOUR, count);
        return -1;
    }
    size_t chunk = count < uirmetal_chunk ? count : uirmetal_chunk;
    /* the slot's regions: momenta, outputs, status, forms, amplitudes, planes */
    size_t region[6] = {uirmetal_momenta_bytes * count, uirmetal_out_bytes * count, sizeof(int),
                        uirmetal_forms_bytes * chunk, uirmetal_amplitudes_bytes * chunk, uirmetal_planes_bytes * chunk};
    size_t offset[6], total = 0;
    for (int r = 0; r < 6; r++) {
        offset[r] = total;
        total += uirmetal_aligned(region[r]);
    }
    int status = 0;
    pacmetal_slot *slot = pacmetal_acquire();
    @autoreleasepool {
        if (pacmetal_reserve(slot, total) != 0) {
            status = -1;
        } else {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)slot->buffer;
            char *base = (char *)buffer.contents;
            /* the momenta as df64 rows: row r at (float2) r count .. , hi = the nearest float, lo the rest */
            float *staged = (float *)(base + offset[0]);
            for (size_t r = 0; r < 4 * (size_t)RUNNER_LEGS; r++) {
                const double *row = momenta + r * row_stride;
                float *to = staged + 2 * r * count;
                for (size_t e = 0; e < count; e++) {
                    double x = row[e];
                    float hi = (float)x;
                    to[2 * e] = hi;
                    to[2 * e + 1] = isfinite(hi) ? (float)(x - (double)hi) : 0.0f;
                }
            }
            *(int *)(base + offset[2]) = 0;
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)pacmetal_queue();
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];   /* serial dispatches */
            [encoder setComputePipelineState:uirmetal_pipeline];
            [encoder setBuffer:uirmetal_program offset:0 atIndex:0];
            for (int r = 0; r < 6; r++) {
                static const int index[6] = {2, 3, 7, 4, 5, 6};   /* uir_points' buffers */
                [encoder setBuffer:buffer offset:offset[r] atIndex:index[r]];
            }
            NSUInteger width = uirmetal_pipeline.maxTotalThreadsPerThreadgroup;
            NSUInteger group = width < 128 ? width : 128;
            for (size_t first = 0; first < count; first += chunk) {
                size_t q = count - first < chunk ? count - first : chunk;
                uirmetal_launch launch = {(uint32_t)q, (uint32_t)first, (uint32_t)(leg_row * count),
                                          (uint32_t)(component_row * count), (uint32_t)count};
                [encoder setBytes:&launch length:sizeof(launch) atIndex:1];
                [encoder dispatchThreads:MTLSizeMake(q, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
            }
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.status != MTLCommandBufferStatusCompleted) {
                uirmetal_error("the command buffer failed", commands.error);
                status = -1;
            } else {
                status = *(int *)(base + offset[2]);
                if (status == 0) {
                    const float *o = (const float *)(base + offset[1]);
                    for (size_t k = 0; k < (size_t)RUNNER_OUTPUTS * count; k++)
                        values[k] = (double)o[2 * k] + (double)o[2 * k + 1];
                }
            }
        }
    }
    pacmetal_release(slot);
    return status;
}

int pac_metal_run(const double *momenta, double *out, size_t points)
{
    /* row 4 l + j at momenta + (4 l + j) points */
    return uirmetal_call(momenta, points, 4, 1, points, out);
}

int pac_metal_values(const double *momenta, size_t stride, size_t count, double *values)
{
    /* MadSpace: row n k + p at momenta + (n k + p) stride: leg p is row p, component k row n k */
    return uirmetal_call(momenta, stride, 1, RUNNER_LEGS, count, values);
}

int run_metal(double *out, const double *momenta, size_t points)
{
    if (points == 0) return 0;
    size_t nm = 4 * (size_t)RUNNER_LEGS, no = RUNNER_OUTPUTS;
    double *m = (double *)malloc(UIRMETAL_MAX1(nm) * points * sizeof(double));
    double *o = (double *)malloc(UIRMETAL_MAX1(no) * points * sizeof(double));
    int status = -1;
    if (m != NULL && o != NULL) {
        for (size_t p = 0; p < points; p++)
            for (size_t k = 0; k < nm; k++) m[k * points + p] = momenta[p * nm + k];
        status = pac_metal_run(m, o, points);
        if (status == 0)
            for (size_t p = 0; p < points; p++)
                for (size_t h = 0; h < no; h++) out[p * no + h] = o[h * points + p];
    }
    free(m);
    free(o);
    return status;
}
