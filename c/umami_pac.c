/* A UMAMI matrix element (madspace/include/madspace/umami.h) for one program: one process, one LC
 * colour flow, one leg permutation, compiled by PacRunner (runner.c: run(out, p, momenta), the
 * outputs of every runtime helicity at one point). PacRunner's umami_libraries writes one library per
 * (process, colour flow, permutation) of a pyAmpliCol artifact's UMAMI provider, with this file, the
 * program's runner.c and pacops.c, and umami_program.h (its constants, below). The flavour
 * splitting is MadSpace's: a flavour of the provider is a cross section of its own
 * (flavor_diff_xs_indices), and each library sees flavour 0.
 *
 * Per event, as the stock pyAmpliCol wrapper (API/umami/umami.c) evaluates one contribution:
 *   - the momenta, input[event + stride*(particle + particle_count*component)], gathered in the
 *     program's leg order: runtime leg j = input leg permutation[j];
 *   - run(): the runtime helicities' values;
 *   - with a helicity index input (exported numbering, the reweight pass): that helicity's value ×
 *     factor (0 when the program does not have it);
 *   - else the values summed in order, × factor; with a helicity index output (the LC pass) one
 *     helicity drawn by inverse CDF from the random helicity (zeros skipped).
 * Bit for bit the stock library in both passes MadGraph's [amplicol] makes. Without either index
 * the stock library asks RustiCol for the helicity sum, which RustiCol takes over helicity orbits
 * (one representative × the orbit's weight): the export does not describe that shortcut, and the
 * sum here agrees with it to rounding (about 2e-16).
 *   - with an alpha_s input different from the export's: each value × (alpha_s / alpha_s0)^power,
 *     the power the export proved (the programs' couplings do not depend on alpha_s; RustiCol
 *     recomputes (4 pi alpha_s)^power instead, so values at another alpha_s agree to rounding);
 *   - the colour index is the program's.
 * The model parameters are those of the export: umami_set_parameter accepts a known parameter at
 * its exported value (relative 1e-12) and refuses any other value.
 * The flavour input must be 0; the channel input is not read (the library is one program). */
#include "umami.h"
#include "runner.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    double real, imaginary;
} PacParameter;

#include "umami_program.h"

#define PAC_EXPORT __attribute__((visibility("default")))

typedef struct {
    double *arena;    /* run()'s scratch, RUNNER_ARENA_SIZE */
    double momenta[4 * PAC_PARTICLES];
    double values[RUNNER_OUTPUTS];
} Instance;

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = PAC_HAS_ALPHA_S,
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_RANDOM_HELICITY] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
    [UMAMI_IN_CHANNEL_INDEX] = true,
};
static bool const required_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
};
static bool const supported_outputs[UMAMI_OUTPUT_KEY_COUNT] = {
    [UMAMI_OUT_MATRIX_ELEMENT] = true,
    [UMAMI_OUT_COLOR_INDEX] = true,
    [UMAMI_OUT_HELICITY_INDEX] = true,
};

PAC_EXPORT UmamiStatus umami_get_meta(UmamiMetaKey key, void *result) {
    if (result == NULL) return UMAMI_ERROR;
    switch (key) {
    case UMAMI_META_DEVICE: *(UmamiDevice *)result = UMAMI_DEVICE_CPU; break;
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = PAC_PARTICLES; break;
    case UMAMI_META_HELICITY_COUNT: *(int *)result = PAC_HELICITY_COUNT; break;
    case UMAMI_META_COLOR_COUNT: *(int *)result = PAC_COLOR_COUNT; break;
    case UMAMI_META_MASSES: memcpy(result, pac_masses, sizeof(pac_masses)); break;
    default: return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_supported_inputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_required_inputs(bool const **required, int *count) {
    if (required == NULL || count == NULL) return UMAMI_ERROR;
    *required = required_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_supported_outputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_outputs; *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

/* param_card_path is not read: the program carries its parameters */
PAC_EXPORT UmamiStatus umami_initialize(UmamiHandle *handle, char const *param_card_path) {
    (void)param_card_path;
    if (handle == NULL) return UMAMI_ERROR;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) return UMAMI_ERROR;
    instance->arena = calloc(RUNNER_ARENA_SIZE, sizeof(double));
    if (instance->arena == NULL) { free(instance); return UMAMI_ERROR; }
    *handle = instance;
    return UMAMI_SUCCESS;
}

static PacParameter const *find_parameter(char const *name) {
    for (size_t i = 0; i < sizeof(pac_parameters) / sizeof(pac_parameters[0]); ++i)
        if (pac_parameters[i].name != NULL && strcmp(pac_parameters[i].name, name) == 0)
            return &pac_parameters[i];
    return NULL;
}

static bool close_to(double x, double y) {
    return x == y || fabs(x - y) <= 1e-12 * fmax(fabs(x), fabs(y));
}

PAC_EXPORT UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name,
                                           double real, double imaginary) {
    if (handle == NULL || name == NULL) return UMAMI_ERROR;
    PacParameter const *p = find_parameter(name);
    if (p == NULL) return UMAMI_ERROR_UNKNOWN_PARAMETER;
    if (close_to(real, p->real) && close_to(imaginary, p->imaginary)) return UMAMI_SUCCESS;
    fprintf(stderr, "umami (%s %s): parameter %s = %.17g + %.17gi, but the program was exported "
            "with %.17g + %.17gi: export it again with these parameters\n", RUNNER_PROCESS,
            RUNNER_COLOUR, name, real, imaginary, p->real, p->imaginary);
    return UMAMI_ERROR;
}

PAC_EXPORT UmamiStatus umami_get_parameter(UmamiHandle handle, char const *name,
                                           double *real, double *imaginary) {
    if (handle == NULL || name == NULL || real == NULL) return UMAMI_ERROR;
    PacParameter const *p = find_parameter(name);
    if (p == NULL) return UMAMI_ERROR_UNKNOWN_PARAMETER;
    *real = p->real;
    if (imaginary != NULL) *imaginary = p->imaginary;
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const *input_keys, void const *const *inputs,
    size_t output_count, UmamiOutputKey const *output_keys, void *const *outputs) {
    if (handle == NULL || offset > stride || count > stride - offset ||
        (input_count && (input_keys == NULL || inputs == NULL)) ||
        (output_count && (output_keys == NULL || outputs == NULL))) return UMAMI_ERROR;
    void const *in[UMAMI_INPUT_KEY_COUNT] = {0};
    void *out[UMAMI_OUTPUT_KEY_COUNT] = {0};
    for (size_t i = 0; i < input_count; ++i) {
        int key = input_keys[i];
        if (key < 0 || key >= UMAMI_INPUT_KEY_COUNT || !supported_inputs[key])
            return UMAMI_ERROR_UNSUPPORTED_INPUT;
        if (inputs[i] == NULL || in[key] != NULL) return UMAMI_ERROR;
        in[key] = inputs[i];
    }
    for (size_t i = 0; i < output_count; ++i) {
        int key = output_keys[i];
        if (key < 0 || key >= UMAMI_OUTPUT_KEY_COUNT || !supported_outputs[key])
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
        if (outputs[i] == NULL || out[key] != NULL) return UMAMI_ERROR;
        out[key] = outputs[i];
    }
    if (count == 0) return UMAMI_SUCCESS;
    if (in[UMAMI_IN_MOMENTA] == NULL) return UMAMI_ERROR_MISSING_INPUT;
    Instance *instance = handle;
    double const *momenta = in[UMAMI_IN_MOMENTA];
    size_t const n = PAC_PARTICLES;
    for (size_t event = offset; event < offset + count; ++event) {
        if (in[UMAMI_IN_FLAVOR_INDEX] && ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] != 0)
            return UMAMI_ERROR;
        double random = in[UMAMI_IN_RANDOM_HELICITY] ?
            ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        if (!isfinite(random) || random < 0.0 || random >= 1.0) return UMAMI_ERROR;
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < 4; ++k) {
                double value = momenta[event + stride * (n * k + pac_permutation[j])];
                if (!isfinite(value)) return UMAMI_ERROR;
                instance->momenta[4 * j + k] = value;
            }
        double scale = 1.0;
        bool scaled = false;
        if (in[UMAMI_IN_ALPHA_S]) {
            double alpha_s = ((double const *)in[UMAMI_IN_ALPHA_S])[event];
            if (!isfinite(alpha_s) || alpha_s < 0.0) return UMAMI_ERROR;
            if (alpha_s != PAC_ALPHA_S0) {
                scale = pow(alpha_s / PAC_ALPHA_S0, PAC_ALPHA_S_POWER);
                scaled = true;
            }
        }
        int label = 0;
        double total = 0.0;
        double *values = instance->values;
        if (run(values, instance->arena, instance->momenta) != 0) return UMAMI_ERROR;
        if (scaled)
            for (size_t h = 0; h < RUNNER_OUTPUTS; ++h) values[h] *= scale;
        if (in[UMAMI_IN_HELICITY_INDEX]) {
            /* a fixed exported helicity; one the program does not have contributes nothing */
            label = ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event];
            if (label < 0 || label >= PAC_EXPORTED_HELICITY_COUNT) return UMAMI_ERROR;
            int runtime = pac_helicity_runtime[label];
            if (runtime >= 0) total += PAC_FACTOR * values[runtime];
        } else {
            for (size_t h = 0; h < RUNNER_OUTPUTS; ++h) total += values[h];
            if (out[UMAMI_OUT_HELICITY_INDEX]) {
                /* the helicity drawn in proportion to its value */
                double target = random * total, cumulative = 0.0;
                size_t chosen = 0;
                for (size_t h = 0; h < RUNNER_OUTPUTS; ++h) {
                    if (values[h] <= 0.0) continue;
                    chosen = h;
                    cumulative += values[h];
                    if (cumulative > target) break;
                }
                label = pac_helicity_export[chosen];
            }
            total *= PAC_FACTOR;
        }
        if (!isfinite(total)) return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT]) ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[event] = total;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = label;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] = PAC_COLOR_INDEX;
    }
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    free(instance->arena);
    free(instance);
    return UMAMI_SUCCESS;
}
