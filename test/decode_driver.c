/* The driver of runtests.jl's test of uirvm's field macros (c/uirvm.h, UIR_OP ... UIR_X): reads
 * encoded words from stdin (hex, whitespace-separated), reads them row by row as uirvm does (as
 * 32-bit halves; one-word rows only) and prints one line per row:
 * op set first last d t l r n x, the index fields 0-based and -1 for none (all ones).
 * build: cc -std=c99 -Ic -o decode_driver test/decode_driver.c c/pacops.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include "uirvm.c"

int uir_stub(uint32_t stub, const uir_program *g, const uir_arena *a, const uint32_t *w, const uint32_t *scaleset)
{
    (void)stub; (void)g; (void)a; (void)w; (void)scaleset;
    return -1;
}

static long field(uint32_t v, uint32_t none) { return v == none ? -1 : (long)v; }

int main(void)
{
    size_t n = 0, cap = 1024;
    uint64_t *w = malloc(cap * sizeof *w);
    unsigned long long x;
    while (scanf("%llx", &x) == 1) {
        if (n == cap) w = realloc(w, (cap *= 2) * sizeof *w);
        w[n++] = x;
    }
    /* as uirvm reads them: each word as two 32-bit halves, low first */
    uint32_t *h = malloc((2 * n + 4) * sizeof *h);
    for (size_t k = 0; k < n; k++) {
        h[2 * k] = (uint32_t)w[k];
        h[2 * k + 1] = (uint32_t)(w[k] >> 32);
    }
    for (size_t k = 0; k < n; k++) {
        const uint32_t *r = h + 2 * k;
        printf("%u %u %u %u %ld %ld %ld %ld %u %ld\n", UIR_OP(r), UIR_SET(r), UIR_FIRST(r), UIR_LAST(r),
               field(UIR_D(r), 0x1fff), field(UIR_T(r), 0x3ff), field(UIR_L(r), 0x1fff), field(UIR_R(r), 0x1fff),
               UIR_N(r), field(UIR_X(r), 0x3f));
    }
    return 0;
}
