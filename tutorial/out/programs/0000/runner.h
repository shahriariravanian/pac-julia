/* u_ubar_to_z_g_g flow:2,4,5,1: the program compiled by PacRunner's compile (runner.c).
 * run(out, p, momenta) evaluates one phase-space point: momenta[4 * RUNNER_LEGS] are the caller's
 * legs (E, px, py, pz each, before the crossing), out[RUNNER_OUTPUTS] receives the helicity
 * outputs (group factor applied; the matrix element is their sum), p[RUNNER_ARENA_SIZE] is
 * scratch. Returns 0. */
#ifndef RUNNER_H
#define RUNNER_H

#define RUNNER_PROCESS "u_ubar_to_z_g_g"
#define RUNNER_COLOUR "flow:2,4,5,1"
#define RUNNER_LEGS 5
#define RUNNER_OUTPUTS 48
#define RUNNER_ARENA_SIZE 372

#ifdef __cplusplus
extern "C" {
#endif

int run(double *out, double *p, const double *momenta);

#ifdef __cplusplus
}
#endif

#endif
