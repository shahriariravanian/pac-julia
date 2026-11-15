/* msl_shim.h: just enough of the Metal Shading Language for the Metal sources (df64.h,
 * pacops.metal, uirvm.metal and a program's generated runner.metal) to compile as C++ on the CPU,
 * so that their arithmetic is tested on Linux. Compile with -std=c++17 -ffp-contract=off (no fma
 * contraction, as MTLMathModeSafe) and call msl_shim_flush_denormals() first: Apple GPUs flush f32
 * denormals to zero, and on x86-64 FTZ + DAZ do the same.
 *
 * What it maps: the address spaces (device, constant, thread, threadgroup) to nothing, `kernel` to
 * nothing, MSL's attributes ([[buffer(n)]], [[thread_position_in_grid]]) are ignored by the C++
 * compiler (-Wno-attributes), float2 / uint2 / uchar, as_type<T>, MSL's float math (fma, sqrt, ...),
 * and the atomics a kernel uses for its status. */

#ifndef MSL_SHIM_H
#define MSL_SHIM_H

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <atomic>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#define device
#define constant
#define thread
#define threadgroup
#define kernel

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

struct float2 {
    float x, y;
    float2() : x(0), y(0) {}
    float2(float a, float b) : x(a), y(b) {}
};
struct uint2 {
    uint x, y;
    uint2() : x(0), y(0) {}
    uint2(uint a, uint b) : x(a), y(b) {}
};

template <class T, class F> static inline T as_type(F f)
{
    static_assert(sizeof(T) == sizeof(F), "as_type: sizes differ");
    T t;
    std::memcpy(&t, &f, sizeof t);
    return t;
}

/* MSL's float math (the float overloads; the sources use nothing else) */
using std::fma;
using std::sqrt;
using std::fabs;
using std::floor;
using std::ceil;
using std::trunc;
using std::rint;
using std::isnan;
using std::isinf;
using std::copysign;
using std::abs;   /* abs(int), abs(float) */

/* the status word a kernel sets when a point fails (MSL: device atomic_int) */
typedef std::atomic<int> atomic_int;
enum memory_order { memory_order_relaxed };
static inline void atomic_store_explicit(device atomic_int *p, int v, memory_order) { p->store(v, std::memory_order_relaxed); }

/* f32 denormals flushed to zero, inputs and results (as an Apple GPU) */
static inline void msl_shim_flush_denormals()
{
#if defined(__x86_64__) || defined(_M_X64)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#elif defined(__aarch64__)
    uint64_t fpcr;                                         /* FPCR.FZ (bit 24): denormal inputs and */
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));     /* results flushed to zero, as FTZ + DAZ */
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr | (UINT64_C(1) << 24)));
#endif
}

#endif
