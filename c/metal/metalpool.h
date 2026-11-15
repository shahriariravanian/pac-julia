/* metalpool.h: the process-wide slot pool of PacRunner's Metal libraries (libpacmetal.dylib, loaded
 * once per process whatever the number of program libraries; the Metal analogue of c/cuda/pool.h).
 *
 * It holds the Metal device and one command queue, and K slots. A slot is one buffer in shared
 * storage (unified memory: the CPU writes and reads it directly), grown on demand and kept: a call
 * borrows any free slot (pacmetal_acquire), carves its momenta, outputs and arena out of it by
 * offset, runs, and gives it back (pacmetal_release); when every slot is busy the caller waits.
 * The pool's memory is at most (slots that ran at once) × their largest call.
 *
 * The first configure call wins: `slots` = 0 means the online CPUs (MadSpace's default thread pool),
 * PAC_METAL_SLOTS overrides it. Objective-C objects are passed as void * (bridged, not retained:
 * they live as long as the process). */

#ifndef METALPOOL_H
#define METALPOOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PACMETAL_ABI 1

typedef struct {
    void *buffer;        /* id<MTLBuffer> (shared storage), NULL until first reserved */
    size_t bytes;        /* its length */
    int busy;
} pacmetal_slot;

/* the pool's size (only the first call counts; `abi` must be PACMETAL_ABI). Returns 0, or -1 (no
 * Metal device, or another ABI; printed to stderr) */
int pacmetal_configure(int slots, int abi);
/* the device (id<MTLDevice>) and the command queue (id<MTLCommandQueue>); NULL before configure */
void *pacmetal_device(void);
void *pacmetal_queue(void);
/* a free slot (waits for one) */
pacmetal_slot *pacmetal_acquire(void);
/* the slot's buffer at least `bytes` long (its contents are not kept when it grows); 0 or -1 */
int pacmetal_reserve(pacmetal_slot *slot, size_t bytes);
void pacmetal_release(pacmetal_slot *slot);

#ifdef __cplusplus
}
#endif

#endif
