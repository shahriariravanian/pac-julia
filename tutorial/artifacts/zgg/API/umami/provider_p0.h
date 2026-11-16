/* SPDX-License-Identifier: 0BSD */
/* Generated from the same records as metadata.json. */
#include "umami_provider.h"
#define UMAMI_HAS_ALPHA_S 0
#define UMAMI_HAS_COLOR_FLOW 1
#define UMAMI_ARTIFACT_ID "d2800389720ee621819d5075d5a2c7faca577a9b047423ab87ad8dfcf70de48b"
static const double umami_masses[] = {0, 0, 91.188000000000002, 0, 0};
static const int umami_p0_pdgs[] = {2, -2, 23, 21, 21};
static const char *const umami_p0_helicities[] = {"h:-1,-1,-1,-1,-1", "h:-1,-1,-1,-1,+1", "h:-1,-1,-1,+1,-1", "h:-1,-1,-1,+1,+1", "h:-1,-1,+0,-1,-1", "h:-1,-1,+0,-1,+1", "h:-1,-1,+0,+1,-1", "h:-1,-1,+0,+1,+1", "h:-1,-1,+1,-1,-1", "h:-1,-1,+1,-1,+1", "h:-1,-1,+1,+1,-1", "h:-1,-1,+1,+1,+1", "h:-1,+1,-1,-1,-1", "h:-1,+1,-1,-1,+1", "h:-1,+1,-1,+1,-1", "h:-1,+1,-1,+1,+1", "h:-1,+1,+0,-1,-1", "h:-1,+1,+0,-1,+1", "h:-1,+1,+0,+1,-1", "h:-1,+1,+0,+1,+1", "h:-1,+1,+1,-1,-1", "h:-1,+1,+1,-1,+1", "h:-1,+1,+1,+1,-1", "h:-1,+1,+1,+1,+1", "h:+1,-1,-1,-1,-1", "h:+1,-1,-1,-1,+1", "h:+1,-1,-1,+1,-1", "h:+1,-1,-1,+1,+1", "h:+1,-1,+0,-1,-1", "h:+1,-1,+0,-1,+1", "h:+1,-1,+0,+1,-1", "h:+1,-1,+0,+1,+1", "h:+1,-1,+1,-1,-1", "h:+1,-1,+1,-1,+1", "h:+1,-1,+1,+1,-1", "h:+1,-1,+1,+1,+1", "h:+1,+1,-1,-1,-1", "h:+1,+1,-1,-1,+1", "h:+1,+1,-1,+1,-1", "h:+1,+1,-1,+1,+1", "h:+1,+1,+0,-1,-1", "h:+1,+1,+0,-1,+1", "h:+1,+1,+0,+1,-1", "h:+1,+1,+0,+1,+1", "h:+1,+1,+1,-1,-1", "h:+1,+1,+1,-1,+1", "h:+1,+1,+1,+1,-1", "h:+1,+1,+1,+1,+1"};
static const UmamiParameter umami_p0_parameters[] = {
    {"normalization.alpha_ew", 0.0075467711139788835, 0, true, false},
    {"normalization.alpha_s_me_check", 0.11799999999999999, 0, true, false},
    {"particle.23.mass", 91.188000000000002, 0, true, false},
    {"particle.23.width", 2.4414039999999999, 0, true, false},
    {"particle.24.mass", 80.419002445756163, 0, true, false},
    {"particle.24.width", 2.0476000000000001, 0, true, false},
    {"particle.25.mass", 125, 0, true, false},
    {"particle.25.width", 0.0063823389999999999, 0, true, false},
    {"particle.6.mass", 173, 0, true, false},
    {"particle.6.width", 1.4915, 0, true, false},
};
static const UmamiProcess umami_processes[] = {
    {"u_ubar_to_z_g_g", umami_p0_pdgs, NULL, "lc", 48, umami_p0_helicities, 10, umami_p0_parameters},
};
static const size_t umami_c0_f0_permutation[] = {0, 1, 2, 3, 4};
static const int umami_c0_f0_helicity_runtime[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
static const size_t umami_c0_f0_helicity_export[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
static const UmamiContribution umami_c0_f0_contributions[] = {
    {0, "flow:2,4,5,1", 0, umami_c0_f0_permutation, 1, umami_c0_f0_helicity_runtime, umami_c0_f0_helicity_export},
};
static const UmamiFlavour umami_c0_flavours[] = {
    {1, umami_c0_f0_contributions},
};
static const UmamiChannel umami_channels[] = {
    {1, umami_c0_flavours},
};
static const UmamiProvider umami_provider = {
    "p0", 5, 2, 48, 2, 48,
    umami_masses, 1, umami_processes,
    1, umami_channels, false
};
