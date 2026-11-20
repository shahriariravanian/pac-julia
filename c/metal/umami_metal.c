/* umami_metal.c: c/umami_pac.c's CPU library with the program on an Apple GPU (uirmetal.h), for
 * umami_libraries(...; device = :metal). The library identifies as a CPU provider (UMAMI and
 * MadSpace have no Metal device): MadSpace passes host memory from its CPU threads; unified memory
 * makes the host's share cheap.
 *
 * Per call, three passes:
 *   1. every event's inputs checked on the host (pac_check: flavour 0, the random number, a fixed
 *      helicity, alpha_s and its scale) and its momenta (finite), as the CPU library;
 *   2. one GPU run of the call's events (pac_metal_values: MadSpace's momentum rows as they are,
 *      staged as df64; the runtime helicities' values back as doubles);
 *   3. every event finished on the host in f64 (pac_finish, umami_finish.h: the scale, the fixed
 *      or summed helicity, the draw, the factor), the colour index.
 * Everything else (the metadata, the parameters, the inputs and outputs accepted) is
 * c/umami_pac.c's. Not bit for bit: the program runs in mixed f32/df64 (c/metal/PLAN.md). */
#include "umami.h"
#include "runner_metal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    double real, imaginary;
} PacParameter;

#include "umami_finish.h"
#include "umami_program.h"

#define PAC_EXPORT __attribute__((visibility("default")))

typedef struct {
    size_t capacity;  /* events the buffers hold */
    double *values;   /* the call's runtime helicity values: values[h capacity' + e], h < RUNNER_OUTPUTS */
    double *scales;   /* each event's alpha_s scale */
    int *scaled;      /* whether it applies */
} Instance;

/* the instance's buffers for `count` events */
static bool reserve_batch(Instance *instance, size_t count) {
    if (count <= instance->capacity) return true;
    size_t n = count > 2 * instance->capacity ? count : 2 * instance->capacity;
    double *v = realloc(instance->values, ((size_t)RUNNER_OUTPUTS * n + 1) * sizeof(double));
    if (v == NULL) return false;
    instance->values = v;
    double *s = realloc(instance->scales, (n + 1) * sizeof(double));
    if (s == NULL) return false;
    instance->scales = s;
    int *f = realloc(instance->scaled, (n + 1) * sizeof(int));
    if (f == NULL) return false;
    instance->scaled = f;
    instance->capacity = n;
    return true;
}

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = true,   /* as the stock library; see c/umami_pac.c */
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
    *handle = instance;
    return UMAMI_SUCCESS;
}

static PacParameter const *find_parameter(char const *name) {
    for (size_t i = 0; i < sizeof(pac_parameters) / sizeof(pac_parameters[0]); ++i)
        if (pac_parameters[i].name != NULL && strcmp(pac_parameters[i].name, name) == 0)
            return &pac_parameters[i];
    return NULL;
}

static bool alpha_s_warned = false;   /* the unknown-power refusal is reported once */

PAC_EXPORT UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name,
                                           double real, double imaginary) {
    if (handle == NULL || name == NULL) return UMAMI_ERROR;
    PacParameter const *p = find_parameter(name);
    if (p == NULL) return UMAMI_ERROR_UNKNOWN_PARAMETER;
    if (pac_close_to(real, p->real) && pac_close_to(imaginary, p->imaginary)) return UMAMI_SUCCESS;
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
    if (!reserve_batch(instance, count)) return UMAMI_ERROR;
    double const *momenta = in[UMAMI_IN_MOMENTA];
    size_t const n = PAC_PARTICLES;
    bool const draw = out[UMAMI_OUT_HELICITY_INDEX] != NULL && in[UMAMI_IN_HELICITY_INDEX] == NULL;

    /* 1. the inputs */
    for (size_t event = offset; event < offset + count; ++event) {
        double random = in[UMAMI_IN_RANDOM_HELICITY] ? ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        int fixed = in[UMAMI_IN_HELICITY_INDEX] ? ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event] : -1;
        double alpha_s = in[UMAMI_IN_ALPHA_S] ? ((double const *)in[UMAMI_IN_ALPHA_S])[event] : 0.0;
        double scale;
        int scaled;
        int code = pac_check(in[UMAMI_IN_FLAVOR_INDEX] != NULL,
                             in[UMAMI_IN_FLAVOR_INDEX] ? ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0, random,
                             in[UMAMI_IN_HELICITY_INDEX] != NULL, fixed, in[UMAMI_IN_ALPHA_S] != NULL, alpha_s,
                             &scale, &scaled);
        if (code != 0) {
            if (code == PAC_E_ALPHA_S && isfinite(alpha_s) && alpha_s >= 0.0 && !alpha_s_warned) {
                alpha_s_warned = true;
                fprintf(stderr, "umami (%s %s): alpha_s = %.17g, but the program was exported with "
                        "%.17g and its alpha_s power is unknown: export it again at this alpha_s\n",
                        RUNNER_PROCESS, RUNNER_COLOUR, alpha_s, PAC_ALPHA_S0);
            }
            return UMAMI_ERROR;
        }
        for (size_t r = 0; r < 4 * n; ++r)
            if (!isfinite(momenta[event + stride * r])) return UMAMI_ERROR;
        instance->scales[event - offset] = scale;
        instance->scaled[event - offset] = scaled;
    }

    /* 2. the program on the GPU: values[h count + e] */
    if (pac_metal_values(momenta + offset, stride, count, instance->values) != 0) return UMAMI_ERROR;

    /* 3. each event's end, in f64 */
    for (size_t event = offset; event < offset + count; ++event) {
        size_t e = event - offset;
        double random = in[UMAMI_IN_RANDOM_HELICITY] ? ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        int fixed = in[UMAMI_IN_HELICITY_INDEX] ? ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event] : -1;
        double me;
        int label;
        if (pac_finish(instance->values + e, count, instance->scaled[e], instance->scales[e], fixed, random, draw,
                       &me, &label) != 0)
            return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT]) ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[event] = me;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = label;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] = PAC_COLOR_INDEX;
    }
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    free(instance->values);
    free(instance->scales);
    free(instance->scaled);
    free(instance);
    return UMAMI_SUCCESS;
}
