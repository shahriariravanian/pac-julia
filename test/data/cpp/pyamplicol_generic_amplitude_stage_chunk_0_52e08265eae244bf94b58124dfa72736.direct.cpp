// pyAmpliCol genuine native DirectApplication producer v1
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if !defined(__clang__) && !defined(__GNUC__)
#error "pyAmpliCol native DirectApplication requires Clang or GCC vector extensions"
#endif

namespace pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736 {

constexpr std::uint32_t kInputPlaneCount = 72u;
constexpr std::uint32_t kScalarInputCount = 0u;
constexpr std::uint32_t kOutputPlaneCount = 20u;
constexpr std::uint32_t kSimdLaneWidth = 2u;
constexpr std::uint32_t kInstructionCount = 70u;
constexpr std::uint32_t kTemporaryCount = 14u;
constexpr std::uint32_t kLogicalStackBytes = 768u;
static_assert(kLogicalStackBytes <= 65536u);

using DirectVector = double __attribute__((vector_size(16)));

template <typename Lane>
struct DirectComplex {
    Lane real;
    Lane imaginary;
};

template <typename Lane>
inline Lane direct_broadcast(double value) noexcept {
    if constexpr (std::is_same_v<Lane, double>) {
        return value;
    } else {
        Lane result{};
        for (std::size_t lane = 0; lane < sizeof(Lane) / sizeof(double); ++lane) {
            result[lane] = value;
        }
        return result;
    }
}

template <typename Lane>
inline Lane direct_load_lane(const double* values, std::uint32_t point) noexcept {
    Lane result;
    std::memcpy(&result, values + point, sizeof(Lane));
    return result;
}

template <typename Lane>
inline void direct_store_lane(
    double* values,
    std::uint32_t point,
    Lane value
) noexcept {
    std::memcpy(values + point, &value, sizeof(Lane));
}

template <typename Lane>
inline DirectComplex<Lane> direct_constant(double real, double imaginary) noexcept {
    return {direct_broadcast<Lane>(real), direct_broadcast<Lane>(imaginary)};
}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_plane(
    const double* real,
    const double* imaginary,
    std::uint32_t point
) noexcept {
    return {direct_load_lane<Lane>(real, point),
             direct_load_lane<Lane>(imaginary, point)};
}

template <typename Lane>
inline DirectComplex<Lane> direct_real_plane(
    const double* real,
    std::uint32_t point
) noexcept {
    return {direct_load_lane<Lane>(real, point), direct_broadcast<Lane>(0.0)};
}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_scalar(
    double real,
    double imaginary
) noexcept {
    return direct_constant<Lane>(real, imaginary);
}

template <typename Lane>
inline DirectComplex<Lane> direct_real_scalar(double real) noexcept {
    return direct_constant<Lane>(real, 0.0);
}

template <typename Lane>
inline DirectComplex<Lane> direct_add(
    DirectComplex<Lane> left,
    DirectComplex<Lane> right
) noexcept {
    return {left.real + right.real, left.imaginary + right.imaginary};
}

template <typename Lane>
inline DirectComplex<Lane> direct_multiply(
    DirectComplex<Lane> left,
    DirectComplex<Lane> right
) noexcept {
    return {
        left.real * right.real - left.imaginary * right.imaginary,
        left.real * right.imaginary + left.imaginary * right.real,
    };
}

template <typename Lane>
inline DirectComplex<Lane> direct_real_reciprocal(
    DirectComplex<Lane> value
) noexcept {
    return {direct_broadcast<Lane>(1.0) / value.real,
             direct_broadcast<Lane>(0.0)};
}

template <typename Lane>
inline DirectComplex<Lane> direct_complex_reciprocal(
    DirectComplex<Lane> value
) noexcept {
    const Lane denominator =
        value.real * value.real + value.imaginary * value.imaginary;
    return {value.real / denominator, -value.imaginary / denominator};
}

struct NativeCompiledDirectInputPlaneV1 {
    const double* values;
};

struct NativeCompiledDirectOutputPlaneV1 {
    double* values;
};

struct NativeCompiledDirectScalarV1 {
    const double* value;
};

struct NativeCompiledDirectMetadataV1 {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::uint32_t flags;
    std::uint32_t input_plane_count;
    std::uint32_t scalar_input_count;
    std::uint32_t output_plane_count;
    std::uint32_t simd_lane_width;
    std::uint32_t reserved;
};

template <typename Lane>
inline void evaluate_direct_bundle(
    const NativeCompiledDirectInputPlaneV1* inputs,
    const NativeCompiledDirectScalarV1* scalars,
    const NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t point
) noexcept {
    std::array<DirectComplex<Lane>, 14> temporary;
    std::array<DirectComplex<Lane>, 10> output;
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point));
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point));
    output[0] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point));
    temporary[1] = direct_add(direct_add(temporary[1], temporary[2]), output[0]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[1]);
    output[0] = direct_add(temporary[0], temporary[2]);
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point));
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point));
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point));
    temporary[0] = direct_add(direct_add(temporary[0], temporary[2]), output[1]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[0]);
    output[1] = direct_add(temporary[1], temporary[2]);
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point));
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point));
    output[2] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point));
    temporary[1] = direct_add(direct_add(temporary[1], temporary[2]), output[2]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[1]);
    output[2] = direct_add(temporary[0], temporary[2]);
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point));
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point));
    output[3] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[70].values, inputs[71].values, point));
    temporary[0] = direct_add(direct_add(temporary[0], temporary[2]), output[3]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[0]);
    output[3] = direct_add(temporary[1], temporary[2]);
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    temporary[1] = direct_add(direct_add(temporary[1], temporary[2]), output[4]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[1]);
    output[4] = direct_add(temporary[0], temporary[2]);
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point));
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    temporary[0] = direct_add(direct_add(temporary[0], temporary[2]), output[5]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[0]);
    output[5] = direct_add(temporary[1], temporary[2]);
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point));
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    temporary[1] = direct_add(direct_add(temporary[1], temporary[2]), output[6]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[1]);
    output[6] = direct_add(temporary[0], temporary[2]);
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point));
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point));
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    temporary[0] = direct_add(direct_add(temporary[0], temporary[2]), output[7]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[0]);
    output[7] = direct_add(temporary[1], temporary[2]);
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point));
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point));
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    temporary[1] = direct_add(direct_add(temporary[1], temporary[2]), output[8]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[1]);
    output[8] = direct_add(temporary[0], temporary[2]);
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point));
    temporary[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point));
    temporary[0] = direct_add(direct_add(temporary[0], temporary[2]), output[9]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[0]);
    output[9] = direct_add(temporary[1], temporary[2]);
    direct_store_lane(outputs[0].values, point, output[0].real);
    direct_store_lane(outputs[1].values, point, output[0].imaginary);
    direct_store_lane(outputs[2].values, point, output[1].real);
    direct_store_lane(outputs[3].values, point, output[1].imaginary);
    direct_store_lane(outputs[4].values, point, output[2].real);
    direct_store_lane(outputs[5].values, point, output[2].imaginary);
    direct_store_lane(outputs[6].values, point, output[3].real);
    direct_store_lane(outputs[7].values, point, output[3].imaginary);
    direct_store_lane(outputs[8].values, point, output[4].real);
    direct_store_lane(outputs[9].values, point, output[4].imaginary);
    direct_store_lane(outputs[10].values, point, output[5].real);
    direct_store_lane(outputs[11].values, point, output[5].imaginary);
    direct_store_lane(outputs[12].values, point, output[6].real);
    direct_store_lane(outputs[13].values, point, output[6].imaginary);
    direct_store_lane(outputs[14].values, point, output[7].real);
    direct_store_lane(outputs[15].values, point, output[7].imaginary);
    direct_store_lane(outputs[16].values, point, output[8].real);
    direct_store_lane(outputs[17].values, point, output[8].imaginary);
    direct_store_lane(outputs[18].values, point, output[9].real);
    direct_store_lane(outputs[19].values, point, output[9].imaginary);
}

}  // namespace pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736

extern "C" pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736::NativeCompiledDirectMetadataV1
pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736_direct_application_v1_metadata() noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736;
    return {
        1u,
        static_cast<std::uint32_t>(sizeof(NativeCompiledDirectMetadataV1)),
        63u,
        kInputPlaneCount,
        kScalarInputCount,
        kOutputPlaneCount,
        kSimdLaneWidth,
        0u,
    };
}

extern "C" const char*
pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736_direct_application_v1_target_triple() noexcept {
    return "x86_64-unknown-linux-gnu";
}

extern "C" const char*
pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736_direct_application_v1_cpu_features() noexcept {
    return "";
}

extern "C" const char*
pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736_direct_application_v1_source_sha256() noexcept {
    return "21312a574e9b73bc6ebda1c577c2ba0ae5e1abf151c4a4c92938c902603e232e";
}

extern "C" std::uint32_t
pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736_direct_application_v1_logical_stack_bytes() noexcept {
    return pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736::kLogicalStackBytes;
}

extern "C" int pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736_direct_application_v1(
    const pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736::
        NativeCompiledDirectInputPlaneV1* inputs,
    std::uint32_t input_count,
    const pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736::
        NativeCompiledDirectScalarV1* scalars,
    std::uint32_t scalar_count,
    const pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736::
        NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t output_count,
    std::uint32_t point_start,
    std::uint32_t point_count
) noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_amplitude_stage_chunk_0_52e08265eae244bf94b58124dfa72736;
    try {
        if (input_count != kInputPlaneCount ||
            scalar_count != kScalarInputCount ||
            output_count != kOutputPlaneCount) {
            return 2;
        }
        if ((input_count != 0u && inputs == nullptr) ||
            (scalar_count != 0u && scalars == nullptr) ||
            outputs == nullptr) {
            return 1;
        }
        for (std::uint32_t input = 0; input < input_count; ++input) {
            if (inputs[input].values == nullptr) {
                return 3;
            }
        }
        for (std::uint32_t scalar = 0; scalar < scalar_count; ++scalar) {
            if (scalars[scalar].value == nullptr) {
                return 3;
            }
        }
        for (std::uint32_t output = 0; output < output_count; ++output) {
            if (outputs[output].values == nullptr) {
                return 3;
            }
            for (std::uint32_t input = 0; input < input_count; ++input) {
                if (outputs[output].values == inputs[input].values) {
                    return 4;
                }
            }
            for (std::uint32_t earlier = 0; earlier < output; ++earlier) {
                if (outputs[output].values == outputs[earlier].values) {
                    return 4;
                }
            }
        }
        if (point_count == 0u) {
            return 5;
        }
        const std::uint64_t point_stop_wide =
            static_cast<std::uint64_t>(point_start) +
            static_cast<std::uint64_t>(point_count);
        if (point_stop_wide >
            static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            return 5;
        }
        const std::uint32_t point_stop =
            static_cast<std::uint32_t>(point_stop_wide);
        const std::uint32_t vector_points =
            (point_count / kSimdLaneWidth) * kSimdLaneWidth;
        const std::uint32_t vector_stop = point_start + vector_points;
        std::uint32_t point = point_start;
        for (; point < vector_stop; point += kSimdLaneWidth) {
            evaluate_direct_bundle<DirectVector>(inputs, scalars, outputs, point);
        }
        for (; point < point_stop; ++point) {
            evaluate_direct_bundle<double>(inputs, scalars, outputs, point);
        }
        return 0;
    } catch (...) {
        return 6;
    }
}
