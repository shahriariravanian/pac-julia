/* metalpool.m: metalpool.h. Built once per output folder into libpacmetal.dylib (umami_libraries),
 * which every program library links; a standalone program folder (compile_metal) compiles it into
 * its own library instead. cc -O2 -fobjc-arc -dynamiclib -framework Metal -framework Foundation. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "metalpool.h"

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_free = PTHREAD_COND_INITIALIZER;
static int pool_configured = 0;     /* 1: ready, -1: failed */
static int pool_slots = 0;
static pacmetal_slot *pool = NULL;
/* the device and queue, kept for the process (retained through these strong statics) */
static id<MTLDevice> pool_device = nil;
static id<MTLCommandQueue> pool_queue = nil;

int pacmetal_configure(int slots, int abi)
{
    if (abi != PACMETAL_ABI) {
        fprintf(stderr, "metalpool: a library of ABI %d, the pool's is %d: rebuild the libraries together\n", abi,
                PACMETAL_ABI);
        return -1;
    }
    pthread_mutex_lock(&pool_lock);
    if (pool_configured == 0) {
        pool_configured = -1;
        pool_device = MTLCreateSystemDefaultDevice();
        pool_queue = pool_device ? [pool_device newCommandQueue] : nil;
        const char *env = getenv("PAC_METAL_SLOTS");
        int n = env && atoi(env) > 0 ? atoi(env) : slots;
        if (n <= 0) {
            long cpus = sysconf(_SC_NPROCESSORS_ONLN);
            n = cpus > 0 ? (int)cpus : 1;
        }
        pool = (pacmetal_slot *)calloc((size_t)n, sizeof(pacmetal_slot));
        if (pool_device == nil || pool_queue == nil) {
            fprintf(stderr, "metalpool: no Metal device\n");
        } else if (pool != NULL) {
            pool_slots = n;
            pool_configured = 1;
        }
    }
    int ok = pool_configured == 1 ? 0 : -1;
    pthread_mutex_unlock(&pool_lock);
    return ok;
}

void *pacmetal_device(void) { return (__bridge void *)pool_device; }
void *pacmetal_queue(void) { return (__bridge void *)pool_queue; }

pacmetal_slot *pacmetal_acquire(void)
{
    pthread_mutex_lock(&pool_lock);
    for (;;) {
        for (int k = 0; k < pool_slots; k++)
            if (!pool[k].busy) {
                pool[k].busy = 1;
                pthread_mutex_unlock(&pool_lock);
                return &pool[k];
            }
        pthread_cond_wait(&pool_free, &pool_lock);
    }
}

int pacmetal_reserve(pacmetal_slot *slot, size_t bytes)
{
    if (bytes <= slot->bytes) return 0;
    size_t n = bytes > 2 * slot->bytes ? bytes : 2 * slot->bytes;
    @autoreleasepool {
        id<MTLBuffer> b = [pool_device newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (b == nil) {
            fprintf(stderr, "metalpool: no buffer of %zu bytes\n", n);
            return -1;
        }
        if (slot->buffer != NULL) CFRelease(slot->buffer);
        slot->buffer = (__bridge_retained void *)b;
        slot->bytes = n;
    }
    return 0;
}

void pacmetal_release(pacmetal_slot *slot)
{
    pthread_mutex_lock(&pool_lock);
    slot->busy = 0;
    pthread_cond_signal(&pool_free);
    pthread_mutex_unlock(&pool_lock);
}
