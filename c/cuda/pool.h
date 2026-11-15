/* pool.h: the process-wide GPU slot pool of PacRunner's CUDA UMAMI libraries (libpacpool.so,
 * loaded once per process, whatever the number of program libraries).
 *
 * A slot is a device memory block of at most `slot_bytes`, pinned host staging, a status word and
 * its own non-blocking stream. A call borrows any free slot (pacpool_acquire), carves its arena out
 * of it, runs on its stream, and gives it back (pacpool_release); when every slot is busy, the
 * caller waits. Slots get their memory the first time they are used and keep it, so the device
 * memory is at most (slots that ran at once) × slot_bytes.
 *
 * The first configure call wins (the first program library to run): `slots` = 0 means the number
 * of online CPUs (MadSpace's default thread pool), so that every thread can have one. */

#ifndef POOL_H
#define POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PACPOOL_ABI 1

typedef struct {
    void *stream;          /* cudaStream_t */
    void *device;          /* device block, device_bytes */
    size_t device_bytes;
    void *host;            /* pinned host block, host_bytes */
    size_t host_bytes;
    int *status;           /* device */
    int *host_status;      /* pinned */
} pacpool_slot;

/* the pool's size: slots (0: the online CPUs) and the device bytes per slot; only the first call
 * counts. Returns 0, or -1 (a different configuration than the first, reported on stderr; the
 * first stays) */
int pacpool_configure(int slots, size_t slot_bytes, int abi);

/* the slot budget (bytes) and the number of slots, after configure */
size_t pacpool_slot_bytes(void);
int pacpool_slots(void);

/* a free slot (waits while all are busy); NULL on a CUDA error (reported on stderr) */
pacpool_slot *pacpool_acquire(void);

/* the slot's blocks grown to at least these sizes (device_bytes ≤ the budget); 0, or -1 */
int pacpool_reserve(pacpool_slot *slot, size_t device_bytes, size_t host_bytes);

void pacpool_release(pacpool_slot *slot);

#ifdef __cplusplus
}
#endif

#endif
