/* pool.cu: the slot pool of pool.h. Built once per output folder by nvcc into
 * libpacpool.so (shared CUDA runtime), which every program library links. */

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>
#include "pool.h"

namespace {

struct Pool {
    std::mutex lock;
    std::condition_variable freed;
    bool configured = false;
    int slots = 0;
    size_t slot_bytes = 0;
    std::vector<pacpool_slot *> all;    // created lazily, up to `slots`
    std::vector<pacpool_slot *> idle;   // free ones, the most recently used last
};

Pool &pool()
{
    static Pool *p = new Pool();   // never destroyed: the libraries may unload in any order at exit
    return *p;
}

bool check(cudaError_t e, const char *what)
{
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "pacpool: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

pacpool_slot *new_slot()
{
    auto *s = static_cast<pacpool_slot *>(std::calloc(1, sizeof(pacpool_slot)));
    if (s == nullptr) return nullptr;
    cudaStream_t stream;
    if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate") ||
        !check(cudaMalloc((void **)&s->status, sizeof(int)), "cudaMalloc") ||
        !check(cudaMallocHost((void **)&s->host_status, sizeof(int)), "cudaMallocHost")) {
        std::free(s);
        return nullptr;
    }
    s->stream = stream;
    return s;
}

size_t grow(size_t have, size_t need, size_t most)
{
    size_t n = have * 2 > need ? have * 2 : need;
    return n < most ? n : most;
}

}  // namespace

extern "C" int pacpool_configure(int slots, size_t slot_bytes, int abi)
{
    Pool &p = pool();
    std::lock_guard<std::mutex> g(p.lock);
    if (abi != PACPOOL_ABI) {
        std::fprintf(stderr, "pacpool: a library of ABI %d, the runtime's is %d\n", abi, PACPOOL_ABI);
        return -1;
    }
    if (slots <= 0) {
        long cpus = sysconf(_SC_NPROCESSORS_ONLN);
        slots = cpus > 0 ? (int)cpus : 1;
    }
    if (!p.configured) {
        p.configured = true;
        p.slots = slots;
        p.slot_bytes = slot_bytes;
        return 0;
    }
    if (slots != p.slots || slot_bytes != p.slot_bytes) {
        static bool warned = false;
        if (!warned)
            std::fprintf(stderr, "pacpool: a library asks for %d slots of %zu bytes; the pool has %d of %zu\n",
                         slots, slot_bytes, p.slots, p.slot_bytes);
        warned = true;
        return -1;
    }
    return 0;
}

extern "C" size_t pacpool_slot_bytes(void) { return pool().slot_bytes; }
extern "C" int pacpool_slots(void) { return pool().slots; }

extern "C" pacpool_slot *pacpool_acquire(void)
{
    Pool &p = pool();
    std::unique_lock<std::mutex> g(p.lock);
    for (;;) {
        if (!p.idle.empty()) {
            pacpool_slot *s = p.idle.back();
            p.idle.pop_back();
            return s;
        }
        if ((int)p.all.size() < p.slots) {
            pacpool_slot *s = new_slot();
            if (s == nullptr) return nullptr;
            p.all.push_back(s);
            return s;
        }
        p.freed.wait(g);
    }
}

extern "C" void pacpool_release(pacpool_slot *s)
{
    if (s == nullptr) return;
    Pool &p = pool();
    {
        std::lock_guard<std::mutex> g(p.lock);
        p.idle.push_back(s);
    }
    p.freed.notify_one();
}

extern "C" int pacpool_reserve(pacpool_slot *s, size_t device_bytes, size_t host_bytes)
{
    size_t budget = pool().slot_bytes;
    if (device_bytes > budget) {
        std::fprintf(stderr, "pacpool: %zu device bytes asked of a %zu-byte slot\n", device_bytes, budget);
        return -1;
    }
    cudaStream_t stream = (cudaStream_t)s->stream;
    if (device_bytes > s->device_bytes) {
        size_t n = grow(s->device_bytes, device_bytes, budget);
        if (!check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) return -1;
        cudaFree(s->device);
        s->device = nullptr;
        s->device_bytes = 0;
        if (!check(cudaMalloc(&s->device, n), "cudaMalloc")) return -1;
        s->device_bytes = n;
    }
    if (host_bytes > s->host_bytes) {
        size_t n = grow(s->host_bytes, host_bytes, (size_t)-1);
        if (!check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) return -1;
        cudaFreeHost(s->host);
        s->host = nullptr;
        s->host_bytes = 0;
        if (!check(cudaMallocHost(&s->host, n), "cudaMallocHost")) return -1;
        s->host_bytes = n;
    }
    return 0;
}
