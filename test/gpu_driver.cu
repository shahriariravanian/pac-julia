/* The driver of test_gpu.jl: runs a compiled program on the GPU (compile(...; interpreter = true)'s
 * runner.c built by nvcc: run_gpu) on the points read from stdin, one line of 4 × LEGS momenta per
 * point, and prints one line of the OUTPUTS outputs per point, each as its 64 bits in hex (as
 * compile_driver.c does for the CPU).
 * usage: gpu_driver LEGS OUTPUTS < points */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern "C" int run_gpu(double *out, const double *momenta, size_t points);  /* uirvm.h */

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    int legs = atoi(argv[1]), outputs = atoi(argv[2]);
    size_t n = 0, cap = 64;
    double *momenta = (double *)malloc(cap * 4 * legs * sizeof(double));
    for (;;) {
        if (n == cap) momenta = (double *)realloc(momenta, (cap *= 2) * 4 * legs * sizeof(double));
        int k = 0;
        for (; k < 4 * legs; k++)
            if (scanf("%lf", &momenta[n * 4 * legs + k]) != 1) break;
        if (k < 4 * legs) break;
        n++;
    }
    double *out = (double *)malloc((n * outputs + 1) * sizeof(double));
    int status = run_gpu(out, momenta, n);
    if (status != 0) {
        fprintf(stderr, "run_gpu: status %d\n", status);
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        for (int h = 0; h < outputs; h++) {
            uint64_t bits;
            memcpy(&bits, &out[i * outputs + h], sizeof bits);
            printf("%s%016llx", h ? " " : "", (unsigned long long)bits);
        }
        printf("\n");
    }
    return 0;
}
