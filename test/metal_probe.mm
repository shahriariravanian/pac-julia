// metal_probe.mm (macOS, phase 0): metal_probe_body.metal's f32 and df64 operations on the GPU
// (MTLMathModeSafe and precise functions, as c/metal/uirmetal.m) and on the CPU through msl_shim.h
// (denormals flushed, as the shim tests), on the same N inputs (default 2^20); the bits compared per
// operation. Exit status 0 when every operation agrees bit for bit.
//
//   c++ -std=c++17 -O1 -ffp-contract=off -fobjc-arc -Wno-attributes -Ic/metal -Itest \
//       -o /tmp/metal_probe test/metal_probe.mm -framework Metal -framework Foundation
//   /tmp/metal_probe c/metal test/metal_probe_body.metal [N]
//
// Negative controls (each should report differences): PROBE_NO_FTZ=1 (the CPU keeps denormals),
// PROBE_FAST=1 (the GPU compiled with MTLMathModeFast).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "msl_shim.h"
#include "df64.h"
#include "metal_probe_body.metal"

static const char *NAMES[NOPS] = {"f32 +", "f32 -", "f32 *", "f32 /", "f32 fma", "f32 sqrt",
    "df64 +", "df64 -", "df64 *", "df64 /", "df64 sqrt", "df64 floor", "df64 rint", "two_prod",
    "two_sum", "df64 fma"};

static NSString *slurp(const char *path)
{
    NSError *e = nil;
    NSString *s = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&e];
    if (!s) { fprintf(stderr, "%s: %s\n", path, e.localizedDescription.UTF8String); exit(2); }
    return s;
}

static void split(double x, float *hi, float *lo)
{
    *hi = (float)x;
    *lo = std::isfinite(*hi) ? (float)(x - (double)*hi) : 0.0f;
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: metal_probe C_METAL_DIR PROBE_BODY [N]\n"); return 2; }
        uint n = argc > 3 ? (uint)atoi(argv[3]) : (1u << 20);
        // inputs: random doubles over 2^-60..2^60 (both signs), every 8th b = -a(1 + tiny) (a
        // cancellation), every 16th pair near 2^-120 (lo parts below f32's normal range)
        std::mt19937_64 rng(20261115);
        std::normal_distribution<double> g(0.0, 1.0);
        std::uniform_real_distribution<double> u(-60.0, 60.0);
        std::vector<float> in(4 * (size_t)n);
        for (uint i = 0; i < n; i++) {
            double a = std::ldexp(1.0 + std::fabs(g(rng)), (int)u(rng)) * (g(rng) < 0 ? -1 : 1);
            double b = std::ldexp(1.0 + std::fabs(g(rng)), (int)u(rng)) * (g(rng) < 0 ? -1 : 1);
            if (i % 8 == 3) b = -a * (1.0 + std::ldexp(g(rng), -30));
            if (i % 16 == 5) { a = std::ldexp(a, -100 - (int)std::fabs(u(rng)) / 6); b = std::ldexp(b, -100); }
            split(a, &in[4 * i], &in[4 * i + 1]);
            split(b, &in[4 * i + 2], &in[4 * i + 3]);
        }
        // the CPU
        std::vector<float> cpu(2 * NOPS * (size_t)n);
        if (!getenv("PROBE_NO_FTZ")) msl_shim_flush_denormals();
        for (uint i = 0; i < n; i++) probe_one(i, in.data(), cpu.data());
        // the GPU
        NSString *dir = @(argv[1]);
        NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n#pragma METAL fp contract(off)\n%@\n%@\n",
                         slurp([[dir stringByAppendingPathComponent:@"df64.h"] UTF8String]), slurp(argv[2])];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions *o = [MTLCompileOptions new];
        o.mathMode = getenv("PROBE_FAST") ? MTLMathModeFast : MTLMathModeSafe;
        o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&e];
        if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&e];
        if (!pso) { printf("pipeline: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLBuffer> bin = [dev newBufferWithBytes:in.data() length:in.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [dev newBufferWithLength:cpu.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bin offset:0 atIndex:0];
        [enc setBuffer:bout offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof n atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { printf("run: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        const float *gpu = (const float *)bout.contents;
        // compare bits; df64 ops also against f64 (relative error of hi + lo)
        printf("%s, %u inputs, MTLMathModeSafe + precise functions\n", dev.name.UTF8String, n);
        int bad = 0;
        for (int k = 0; k < NOPS; k++) {
            size_t diff = 0, first = (size_t)-1;
            for (size_t i = 0; i < n; i++) {
                size_t j = 2 * (NOPS * i + k);
                if (memcmp(&gpu[j], &cpu[j], 8) != 0) { if (!diff) first = i; diff++; }
            }
            printf("  %-11s %s", NAMES[k], diff ? "DIFFERS" : "bit for bit");
            if (diff) {
                size_t j = 2 * (NOPS * first + k);
                printf(" at %zu of %u inputs; first: in (%a %a %a %a) gpu (%a, %a) cpu (%a, %a)", diff, n,
                       in[4 * first], in[4 * first + 1], in[4 * first + 2], in[4 * first + 3],
                       gpu[j], gpu[j + 1], cpu[j], cpu[j + 1]);
                bad++;
            }
            printf("\n");
        }
        return bad ? 1 : 0;
    }
}
