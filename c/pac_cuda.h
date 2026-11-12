/* pac_cuda: the process-wide GPU slot pool of PacRunner's CUDA UMAMI libraries (libpac_cuda.so,
 * loaded once per process, whatever the number of program libraries).
 *
 * A slot is a device memory block of at most `slot_bytes`, pinned host staging, a status word and
 * its own non-blocking stream. A call borrows any free slot (pac_cuda_acquire), carves its arena out
 * of it, runs on its stream, and gives it back (pac_cuda_release); when every slot is busy, the
 * caller waits. Slots get their memory the first time they are used and keep it, so the device
 * memory is at most (slots that ran at once) × slot_bytes.
 *
 * The first configure call wins (the first program library to run): `slots` = 0 means the number
 * of online CPUs (MadSpace's default thread pool), so that every thread can have one. */

#ifndef PAC_CUDA_H
#define PAC_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAC_CUDA_ABI 1

typedef struct {
    void *stream;          /* cudaStream_t */
    void *device;          /* device block, device_bytes */
    size_t device_bytes;
    void *host;            /* pinned host block, host_bytes */
    size_t host_bytes;
    int *status;           /* device */
    int *host_status;      /* pinned */
} pac_cuda_slot;

/* the pool's size: slots (0: the online CPUs) and the device bytes per slot; only the first call
 * counts. Returns 0, or -1 (a different configuration than the first, reported on stderr; the
 * first stays) */
int pac_cuda_configure(int slots, size_t slot_bytes, int abi);

/* the slot budget (bytes) and the number of slots, after configure */
size_t pac_cuda_slot_bytes(void);
int pac_cuda_slots(void);

/* a free slot (waits while all are busy); NULL on a CUDA error (reported on stderr) */
pac_cuda_slot *pac_cuda_acquire(void);

/* the slot's blocks grown to at least these sizes (device_bytes ≤ the budget); 0, or -1 */
int pac_cuda_reserve(pac_cuda_slot *slot, size_t device_bytes, size_t host_bytes);

void pac_cuda_release(pac_cuda_slot *slot);

#ifdef __cplusplus
}
#endif

#endif
