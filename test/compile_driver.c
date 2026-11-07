/* The driver of test_compile.jl: runs a compiled program (compile.jl's runner.c) on the points
 * read from stdin, one line of 4 × LEGS momenta (E, px, py, pz of each leg, the caller's legs) per
 * point, and prints one line of the OUTPUTS outputs per point, each as its 64 bits in hex.
 * usage: driver LEGS OUTPUTS ARENA_SIZE < points */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int run(double *out, double *p, const double *momenta);

int main(int argc, char **argv)
{
    if (argc != 4) return 2;
    int legs = atoi(argv[1]), outputs = atoi(argv[2]);
    size_t size = (size_t)atol(argv[3]);
    double *momenta = malloc(4 * legs * sizeof(double)), *p = malloc(size * sizeof(double)),
           *out = malloc((outputs + 1) * sizeof(double));
    for (;;) {
        for (int i = 0; i < 4 * legs; i++)
            if (scanf("%lf", &momenta[i]) != 1) return 0;
        run(out, p, momenta);
        for (int h = 0; h < outputs; h++) {
            uint64_t bits;
            memcpy(&bits, &out[h], sizeof bits);
            printf("%s%016llx", h ? " " : "", (unsigned long long)bits);
        }
        printf("\n");
    }
}
