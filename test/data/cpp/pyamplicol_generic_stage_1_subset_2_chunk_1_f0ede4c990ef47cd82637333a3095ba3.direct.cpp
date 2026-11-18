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

namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3 {

constexpr std::uint32_t kInputPlaneCount = 36u;
constexpr std::uint32_t kScalarInputCount = 0u;
constexpr std::uint32_t kOutputPlaneCount = 40u;
constexpr std::uint32_t kSimdLaneWidth = 2u;
constexpr std::uint32_t kInstructionCount = 150u;
constexpr std::uint32_t kTemporaryCount = 24u;
constexpr std::uint32_t kLogicalStackBytes = 1408u;
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
    std::array<DirectComplex<Lane>, 24> temporary;
    std::array<DirectComplex<Lane>, 20> output;
    output[12] = direct_constant<Lane>(0x1.6a09e667f3bcdp-1, 0x0.0p+0);
    output[14] = direct_multiply(direct_real_plane<Lane>(inputs[28].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[14] = direct_add(direct_real_plane<Lane>(inputs[24].values, point), output[14]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[0] = direct_add(direct_add(output[0], output[11]), output[5]);
    output[11] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[0]);
    output[5] = direct_add(output[1], output[11]);
    output[0] = direct_multiply(output[14], output[5]);
    output[1] = direct_multiply(direct_real_plane<Lane>(inputs[24].values, point), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[11] = direct_multiply(direct_real_plane<Lane>(inputs[25].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[13] = direct_multiply(direct_real_plane<Lane>(inputs[26].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[3] = direct_multiply(direct_real_plane<Lane>(inputs[27].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[11] = direct_add(direct_add(output[11], output[13]), output[3]);
    output[13] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[11]);
    output[3] = direct_add(output[1], output[13]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[3]);
    output[1] = direct_multiply(direct_real_plane<Lane>(inputs[28].values, point), direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point));
    output[13] = direct_multiply(direct_real_plane<Lane>(inputs[29].values, point), direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point));
    output[2] = direct_multiply(direct_real_plane<Lane>(inputs[30].values, point), direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point));
    output[18] = direct_multiply(direct_real_plane<Lane>(inputs[31].values, point), direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point));
    output[13] = direct_add(direct_add(output[13], output[2]), output[18]);
    output[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[13]);
    output[18] = direct_add(output[1], output[2]);
    output[13] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[18]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), output[13]);
    output[2] = direct_add(output[11], output[1]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[2]);
    output[1] = direct_add(output[0], output[18]);
    output[11] = direct_multiply(direct_real_plane<Lane>(inputs[32].values, point), direct_real_plane<Lane>(inputs[32].values, point));
    output[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[11]);
    output[0] = direct_multiply(direct_real_plane<Lane>(inputs[33].values, point), direct_real_plane<Lane>(inputs[33].values, point));
    output[18] = direct_multiply(direct_real_plane<Lane>(inputs[34].values, point), direct_real_plane<Lane>(inputs[34].values, point));
    output[11] = direct_multiply(direct_real_plane<Lane>(inputs[35].values, point), direct_real_plane<Lane>(inputs[35].values, point));
    output[0] = direct_add(direct_add(direct_add(output[2], output[0]), output[18]), output[11]);
    output[11] = direct_real_reciprocal(output[0]);
    output[2] = direct_multiply(output[11], output[12]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[2]);
    output[0] = direct_multiply(output[1], output[18]);
    output[12] = direct_multiply(direct_real_plane<Lane>(inputs[29].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[11] = direct_add(direct_real_plane<Lane>(inputs[25].values, point), output[12]);
    output[2] = direct_multiply(output[11], output[5]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[3]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), output[13]);
    output[12] = direct_add(output[1], output[12]);
    output[1] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[12]);
    output[12] = direct_add(output[2], output[1]);
    output[1] = direct_multiply(output[18], output[12]);
    output[2] = direct_multiply(direct_real_plane<Lane>(inputs[30].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[12] = direct_add(direct_real_plane<Lane>(inputs[26].values, point), output[2]);
    output[2] = direct_multiply(output[12], output[5]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[3]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), output[13]);
    output[17] = direct_add(output[17], output[4]);
    output[4] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[17]);
    output[17] = direct_add(output[2], output[4]);
    output[2] = direct_multiply(output[18], output[17]);
    output[4] = direct_multiply(direct_real_plane<Lane>(inputs[31].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[17] = direct_add(direct_real_plane<Lane>(inputs[27].values, point), output[4]);
    output[4] = direct_multiply(output[17], output[5]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), output[3]);
    output[3] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), output[13]);
    output[5] = direct_add(output[5], output[3]);
    output[3] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[5]);
    output[5] = direct_add(output[4], output[3]);
    output[3] = direct_multiply(output[18], output[5]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[5] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[5]);
    output[4] = direct_add(output[4], output[6]);
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[5]);
    output[5] = direct_add(output[6], output[5]);
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    temporary[13] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point));
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), temporary[13]);
    output[6] = direct_add(output[6], output[7]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    temporary[15] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point));
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[15]);
    output[7] = direct_add(output[7], output[8]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[13]);
    output[8] = direct_add(output[8], output[9]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), temporary[13]);
    output[9] = direct_add(output[9], output[19]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    temporary[21] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[16] = direct_add(direct_add(output[16], output[10]), temporary[21]);
    output[10] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[16]);
    temporary[21] = direct_add(output[19], output[10]);
    output[16] = direct_multiply(output[14], temporary[21]);
    output[19] = direct_multiply(direct_real_plane<Lane>(inputs[24].values, point), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[10] = direct_multiply(direct_real_plane<Lane>(inputs[25].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[14] = direct_multiply(direct_real_plane<Lane>(inputs[26].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[15] = direct_multiply(direct_real_plane<Lane>(inputs[27].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[14] = direct_add(direct_add(output[10], output[14]), output[15]);
    output[10] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[14]);
    output[15] = direct_add(output[19], output[10]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[15]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), output[13]);
    output[10] = direct_add(output[14], output[19]);
    output[14] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[10]);
    output[19] = direct_add(output[16], output[14]);
    output[10] = direct_multiply(output[18], output[19]);
    output[14] = direct_multiply(output[11], temporary[21]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[15]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), output[13]);
    output[11] = direct_add(output[16], output[19]);
    output[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[11]);
    output[16] = direct_add(output[14], output[19]);
    output[11] = direct_multiply(output[18], output[16]);
    output[14] = direct_multiply(output[12], temporary[21]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[15]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), output[13]);
    output[12] = direct_add(output[19], output[16]);
    output[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[12]);
    output[16] = direct_add(output[14], output[19]);
    output[12] = direct_multiply(output[18], output[16]);
    output[14] = direct_multiply(output[17], temporary[21]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), output[15]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), output[13]);
    output[17] = direct_add(output[19], output[16]);
    temporary[21] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[17]);
    output[15] = direct_add(output[14], temporary[21]);
    output[13] = direct_multiply(output[18], output[15]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[16]);
    output[14] = direct_add(output[19], output[17]);
    temporary[21] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[16]);
    output[15] = direct_add(temporary[21], output[18]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), temporary[13]);
    output[16] = direct_add(output[17], output[19]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    temporary[21] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[15]);
    output[17] = direct_add(output[18], temporary[21]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[13]);
    output[18] = direct_add(output[19], temporary[15]);
    temporary[21] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), temporary[13]);
    output[19] = direct_add(temporary[21], temporary[15]);
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
    direct_store_lane(outputs[20].values, point, output[10].real);
    direct_store_lane(outputs[21].values, point, output[10].imaginary);
    direct_store_lane(outputs[22].values, point, output[11].real);
    direct_store_lane(outputs[23].values, point, output[11].imaginary);
    direct_store_lane(outputs[24].values, point, output[12].real);
    direct_store_lane(outputs[25].values, point, output[12].imaginary);
    direct_store_lane(outputs[26].values, point, output[13].real);
    direct_store_lane(outputs[27].values, point, output[13].imaginary);
    direct_store_lane(outputs[28].values, point, output[14].real);
    direct_store_lane(outputs[29].values, point, output[14].imaginary);
    direct_store_lane(outputs[30].values, point, output[15].real);
    direct_store_lane(outputs[31].values, point, output[15].imaginary);
    direct_store_lane(outputs[32].values, point, output[16].real);
    direct_store_lane(outputs[33].values, point, output[16].imaginary);
    direct_store_lane(outputs[34].values, point, output[17].real);
    direct_store_lane(outputs[35].values, point, output[17].imaginary);
    direct_store_lane(outputs[36].values, point, output[18].real);
    direct_store_lane(outputs[37].values, point, output[18].imaginary);
    direct_store_lane(outputs[38].values, point, output[19].real);
    direct_store_lane(outputs[39].values, point, output[19].imaginary);
}

}  // namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3

extern "C" pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3::NativeCompiledDirectMetadataV1
pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3_direct_application_v1_metadata() noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3;
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
pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3_direct_application_v1_target_triple() noexcept {
    return "x86_64-unknown-linux-gnu";
}

extern "C" const char*
pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3_direct_application_v1_cpu_features() noexcept {
    return "";
}

extern "C" const char*
pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3_direct_application_v1_source_sha256() noexcept {
    return "96b8779c59b55be0e0fec3ec0813a186134c25c293cbac3cc5f4f718179b74ef";
}

extern "C" std::uint32_t
pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3_direct_application_v1_logical_stack_bytes() noexcept {
    return pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3::kLogicalStackBytes;
}

extern "C" int pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3_direct_application_v1(
    const pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3::
        NativeCompiledDirectInputPlaneV1* inputs,
    std::uint32_t input_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3::
        NativeCompiledDirectScalarV1* scalars,
    std::uint32_t scalar_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3::
        NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t output_count,
    std::uint32_t point_start,
    std::uint32_t point_count
) noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_1_f0ede4c990ef47cd82637333a3095ba3;
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
