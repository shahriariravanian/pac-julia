/* The driver of bench_compile.jl: times a compiled program (compile.jl's runner.c) on the points
 * read from stdin (one line of 4 × LEGS momenta per point, as compile_driver.c reads them): REPS
 * passes over the points, TRIALS times; prints the best time per point in ns and a checksum (the
 * sum of output 0, so that the calls are not optimized away).
 * usage: bench LEGS OUTPUTS ARENA_SIZE REPS TRIALS < points */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int run(double *out, double *p, const double *momenta);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc != 6) return 2;
    int legs = atoi(argv[1]), outputs = atoi(argv[2]), reps = atoi(argv[4]), trials = atoi(argv[5]);
    double *p = malloc(atol(argv[3]) * sizeof(double)), *out = malloc((outputs + 1) * sizeof(double));
    int n = 0, cap = 64;
    double *m = malloc(cap * 4 * legs * sizeof(double));
    for (;;) {
        if (n == cap) m = realloc(m, (cap *= 2) * 4 * legs * sizeof(double));
        int i;
        for (i = 0; i < 4 * legs; i++)
            if (scanf("%lf", &m[n * 4 * legs + i]) != 1) break;
        if (i < 4 * legs) break;
        n++;
    }
    if (n == 0) return 1;
    double best = 1e30, sum = 0.0;
    for (int t = 0; t < trials; t++) {
        double t0 = now();
        for (int r = 0; r < reps; r++)
            for (int k = 0; k < n; k++) {
                run(out, p, m + k * 4 * legs);
                sum += out[0];
            }
        double dt = (now() - t0) / ((double)reps * n);
        if (dt < best) best = dt;
    }
    printf("%.1f %g\n", best * 1e9, sum);
    return 0;
}
