/* The driver of runtests.jl's test of uirvm's decoder (c/uirvm.c, uir_decode): reads encoded words
 * from stdin (hex, whitespace-separated), decodes them row by row and prints one line per row:
 * op set first last d t l r n x, the index fields 0-based and -1 for none.
 * build: cc -std=c99 -Ic -o decode_driver test/decode_driver.c c/pacops.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include "uirvm.c"

int uir_kernel(uint32_t kernel, const double *in, double *out) { (void)kernel; (void)in; (void)out; return -1; }

static long field(uint32_t v) { return v == UIR_NONE ? -1 : (long)v; }

int main(void)
{
    size_t n = 0, cap = 1024;
    uint64_t *w = malloc(cap * sizeof *w);
    unsigned long long x;
    while (scanf("%llx", &x) == 1) {
        if (n == cap) w = realloc(w, (cap *= 2) * sizeof *w);
        w[n++] = x;
    }
    for (size_t k = 0; k < n;) {
        uir_row o;
        k += uir_decode(w + k, &o);
        printf("%u %u %u %u %ld %ld %ld %ld %u %ld\n", o.op, o.set, o.first, o.last, field(o.d), field(o.t),
               field(o.l), field(o.r), o.n, field(o.x));
    }
    return 0;
}
