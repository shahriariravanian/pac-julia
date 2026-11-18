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

namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1 {

constexpr std::uint32_t kInputPlaneCount = 60u;
constexpr std::uint32_t kScalarInputCount = 0u;
constexpr std::uint32_t kOutputPlaneCount = 100u;
constexpr std::uint32_t kSimdLaneWidth = 2u;
constexpr std::uint32_t kInstructionCount = 359u;
constexpr std::uint32_t kTemporaryCount = 55u;
constexpr std::uint32_t kLogicalStackBytes = 3360u;
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
    std::array<DirectComplex<Lane>, 55> temporary;
    std::array<DirectComplex<Lane>, 50> output;
    output[12] = direct_constant<Lane>(0x1.6a09e667f3bcdp-1, 0x0.0p+0);
    output[14] = direct_multiply(direct_real_plane<Lane>(inputs[44].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[14] = direct_add(direct_real_plane<Lane>(inputs[40].values, point), output[14]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[0] = direct_add(direct_add(output[0], output[11]), output[5]);
    output[11] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[0]);
    output[5] = direct_add(output[1], output[11]);
    output[0] = direct_multiply(output[14], output[5]);
    output[1] = direct_multiply(direct_real_plane<Lane>(inputs[40].values, point), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[11] = direct_multiply(direct_real_plane<Lane>(inputs[41].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[13] = direct_multiply(direct_real_plane<Lane>(inputs[42].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[3] = direct_multiply(direct_real_plane<Lane>(inputs[43].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[11] = direct_add(direct_add(output[11], output[13]), output[3]);
    output[13] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[11]);
    output[3] = direct_add(output[1], output[13]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[3]);
    output[1] = direct_multiply(direct_real_plane<Lane>(inputs[44].values, point), direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point));
    output[13] = direct_multiply(direct_real_plane<Lane>(inputs[45].values, point), direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point));
    temporary[8] = direct_multiply(direct_real_plane<Lane>(inputs[46].values, point), direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point));
    output[47] = direct_multiply(direct_real_plane<Lane>(inputs[47].values, point), direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point));
    output[13] = direct_add(direct_add(output[13], temporary[8]), output[47]);
    temporary[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[13]);
    output[47] = direct_add(output[1], temporary[8]);
    output[13] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[47]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), output[13]);
    temporary[8] = direct_add(output[11], output[1]);
    output[47] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[8]);
    output[1] = direct_add(output[0], output[47]);
    output[11] = direct_multiply(direct_real_plane<Lane>(inputs[52].values, point), direct_real_plane<Lane>(inputs[52].values, point));
    temporary[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[11]);
    output[0] = direct_multiply(direct_real_plane<Lane>(inputs[53].values, point), direct_real_plane<Lane>(inputs[53].values, point));
    output[47] = direct_multiply(direct_real_plane<Lane>(inputs[54].values, point), direct_real_plane<Lane>(inputs[54].values, point));
    output[11] = direct_multiply(direct_real_plane<Lane>(inputs[55].values, point), direct_real_plane<Lane>(inputs[55].values, point));
    output[0] = direct_add(direct_add(direct_add(temporary[8], output[0]), output[47]), output[11]);
    output[11] = direct_real_reciprocal(output[0]);
    temporary[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[12]);
    output[47] = direct_multiply(output[11], temporary[8]);
    output[0] = direct_multiply(output[1], output[47]);
    output[12] = direct_multiply(direct_real_plane<Lane>(inputs[45].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[11] = direct_add(direct_real_plane<Lane>(inputs[41].values, point), output[12]);
    output[1] = direct_multiply(output[11], output[5]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[3]);
    output[2] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), output[13]);
    output[12] = direct_add(output[12], output[2]);
    output[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[12]);
    output[12] = direct_add(output[1], output[2]);
    output[1] = direct_multiply(output[47], output[12]);
    output[2] = direct_multiply(direct_real_plane<Lane>(inputs[46].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[12] = direct_add(direct_real_plane<Lane>(inputs[42].values, point), output[2]);
    output[2] = direct_multiply(output[12], output[5]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[3]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), output[13]);
    output[17] = direct_add(output[17], output[4]);
    output[4] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[17]);
    output[17] = direct_add(output[2], output[4]);
    output[2] = direct_multiply(output[47], output[17]);
    output[4] = direct_multiply(direct_real_plane<Lane>(inputs[47].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[17] = direct_add(direct_real_plane<Lane>(inputs[43].values, point), output[4]);
    output[4] = direct_multiply(output[17], output[5]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), output[3]);
    output[3] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), output[13]);
    output[5] = direct_add(output[5], output[3]);
    output[3] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[5]);
    output[5] = direct_add(output[4], output[3]);
    output[3] = direct_multiply(output[47], output[5]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[5] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[5]);
    output[4] = direct_add(output[4], output[6]);
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[5]);
    output[5] = direct_add(output[6], output[5]);
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point));
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), output[19]);
    output[6] = direct_add(output[6], output[7]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[38] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[38]);
    output[7] = direct_add(output[7], output[8]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), output[19]);
    output[8] = direct_add(output[8], output[9]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[37] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), output[37]);
    output[9] = direct_add(output[9], output[46]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[16] = direct_add(direct_add(output[16], output[10]), output[18]);
    output[10] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[16]);
    output[18] = direct_add(output[46], output[10]);
    output[16] = direct_multiply(output[14], output[18]);
    output[46] = direct_multiply(direct_real_plane<Lane>(inputs[40].values, point), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[10] = direct_multiply(direct_real_plane<Lane>(inputs[41].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[14] = direct_multiply(direct_real_plane<Lane>(inputs[42].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[15] = direct_multiply(direct_real_plane<Lane>(inputs[43].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[14] = direct_add(direct_add(output[10], output[14]), output[15]);
    output[10] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[14]);
    output[15] = direct_add(output[46], output[10]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[15]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), output[13]);
    output[10] = direct_add(output[14], output[46]);
    output[14] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[10]);
    output[46] = direct_add(output[16], output[14]);
    output[10] = direct_multiply(output[47], output[46]);
    output[14] = direct_multiply(output[11], output[18]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[15]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), output[13]);
    output[11] = direct_add(output[16], output[46]);
    output[46] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[11]);
    output[16] = direct_add(output[14], output[46]);
    output[11] = direct_multiply(output[47], output[16]);
    output[14] = direct_multiply(output[12], output[18]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[15]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), output[13]);
    output[12] = direct_add(output[46], output[16]);
    output[46] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[12]);
    output[16] = direct_add(output[14], output[46]);
    output[12] = direct_multiply(output[47], output[16]);
    output[14] = direct_multiply(output[17], output[18]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), output[15]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), output[13]);
    output[17] = direct_add(output[46], output[16]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[17]);
    output[15] = direct_add(output[14], output[18]);
    output[13] = direct_multiply(output[47], output[15]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[16]);
    output[14] = direct_add(output[46], output[17]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[16]);
    output[15] = direct_add(output[18], output[47]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), output[19]);
    output[16] = direct_add(output[17], output[46]);
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[18] = direct_multiply(direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point)), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[17] = direct_add(output[47], output[18]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), output[19]);
    output[18] = direct_add(output[46], output[47]);
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), output[19]);
    output[19] = direct_add(output[47], output[46]);
    output[47] = direct_multiply(direct_real_plane<Lane>(inputs[48].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[46] = direct_add(direct_real_plane<Lane>(inputs[44].values, point), output[47]);
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point));
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    temporary[27] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[21] = direct_add(direct_add(output[21], output[22]), temporary[27]);
    output[22] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[21]);
    temporary[27] = direct_add(output[47], output[22]);
    output[21] = direct_multiply(output[46], temporary[27]);
    output[47] = direct_multiply(direct_real_plane<Lane>(inputs[44].values, point), direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point));
    output[22] = direct_multiply(direct_real_plane<Lane>(inputs[45].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    output[35] = direct_multiply(direct_real_plane<Lane>(inputs[46].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[43] = direct_multiply(direct_real_plane<Lane>(inputs[47].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[22] = direct_add(direct_add(output[22], output[35]), output[43]);
    output[35] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[22]);
    output[43] = direct_add(output[47], output[35]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), output[43]);
    output[47] = direct_multiply(direct_real_plane<Lane>(inputs[48].values, point), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[35] = direct_multiply(direct_real_plane<Lane>(inputs[49].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    temporary[30] = direct_multiply(direct_real_plane<Lane>(inputs[50].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[20] = direct_multiply(direct_real_plane<Lane>(inputs[51].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[35] = direct_add(direct_add(output[35], temporary[30]), output[20]);
    temporary[30] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[35]);
    output[20] = direct_add(output[47], temporary[30]);
    output[35] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[20]);
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), output[35]);
    temporary[30] = direct_add(output[22], output[47]);
    output[20] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[30]);
    output[47] = direct_add(output[21], output[20]);
    output[22] = direct_multiply(direct_real_plane<Lane>(inputs[56].values, point), direct_real_plane<Lane>(inputs[56].values, point));
    temporary[30] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[22]);
    output[21] = direct_multiply(direct_real_plane<Lane>(inputs[57].values, point), direct_real_plane<Lane>(inputs[57].values, point));
    output[20] = direct_multiply(direct_real_plane<Lane>(inputs[58].values, point), direct_real_plane<Lane>(inputs[58].values, point));
    output[22] = direct_multiply(direct_real_plane<Lane>(inputs[59].values, point), direct_real_plane<Lane>(inputs[59].values, point));
    output[21] = direct_add(direct_add(direct_add(temporary[30], output[21]), output[20]), output[22]);
    output[22] = direct_real_reciprocal(output[21]);
    temporary[30] = direct_multiply(output[22], temporary[8]);
    output[20] = direct_multiply(output[47], temporary[30]);
    output[21] = direct_multiply(direct_real_plane<Lane>(inputs[49].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[8] = direct_add(direct_real_plane<Lane>(inputs[45].values, point), output[21]);
    output[22] = direct_multiply(temporary[8], temporary[27]);
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), output[43]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), output[35]);
    output[47] = direct_add(output[47], output[21]);
    output[21] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[47]);
    output[47] = direct_add(output[22], output[21]);
    output[21] = direct_multiply(temporary[30], output[47]);
    output[22] = direct_multiply(direct_real_plane<Lane>(inputs[50].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[47] = direct_add(direct_real_plane<Lane>(inputs[46].values, point), output[22]);
    output[22] = direct_multiply(output[47], temporary[27]);
    output[45] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), output[43]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), output[35]);
    output[45] = direct_add(output[45], output[23]);
    output[23] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[45]);
    output[45] = direct_add(output[22], output[23]);
    output[22] = direct_multiply(temporary[30], output[45]);
    output[23] = direct_multiply(direct_real_plane<Lane>(inputs[51].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[45] = direct_add(direct_real_plane<Lane>(inputs[47].values, point), output[23]);
    output[23] = direct_multiply(output[45], temporary[27]);
    temporary[27] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), output[43]);
    output[24] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), output[35]);
    temporary[27] = direct_add(temporary[27], output[24]);
    output[24] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[27]);
    temporary[27] = direct_add(output[23], output[24]);
    output[23] = direct_multiply(temporary[30], temporary[27]);
    output[24] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    temporary[27] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[27]);
    output[24] = direct_add(output[24], output[25]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), temporary[27]);
    output[25] = direct_add(output[25], output[26]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), temporary[27]);
    output[26] = direct_add(output[26], output[27]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[49] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), output[49]);
    output[27] = direct_add(output[27], output[28]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[39] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[29] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), output[39]);
    output[28] = direct_add(output[28], output[29]);
    output[29] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), output[39]);
    output[29] = direct_add(output[29], output[31]);
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point));
    output[36] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point));
    output[32] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point));
    output[33] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    output[36] = direct_add(direct_add(output[36], output[32]), output[33]);
    output[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[36]);
    output[33] = direct_add(output[31], output[32]);
    output[36] = direct_multiply(output[46], output[33]);
    output[31] = direct_multiply(direct_real_plane<Lane>(inputs[44].values, point), direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point));
    output[32] = direct_multiply(direct_real_plane<Lane>(inputs[45].values, point), direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point));
    output[30] = direct_multiply(direct_real_plane<Lane>(inputs[46].values, point), direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point));
    output[40] = direct_multiply(direct_real_plane<Lane>(inputs[47].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    output[32] = direct_add(direct_add(output[32], output[30]), output[40]);
    output[30] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[32]);
    output[40] = direct_add(output[31], output[30]);
    output[32] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), output[40]);
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), output[35]);
    output[30] = direct_add(output[32], output[31]);
    output[31] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[30]);
    output[32] = direct_add(output[36], output[31]);
    output[30] = direct_multiply(temporary[30], output[32]);
    output[31] = direct_multiply(temporary[8], output[33]);
    output[36] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), output[40]);
    output[32] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), output[35]);
    output[36] = direct_add(output[36], output[32]);
    output[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[36]);
    output[36] = direct_add(output[31], output[32]);
    output[31] = direct_multiply(temporary[30], output[36]);
    output[32] = direct_multiply(output[47], output[33]);
    output[36] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), output[40]);
    output[34] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), output[35]);
    output[36] = direct_add(output[36], output[34]);
    output[34] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[36]);
    output[36] = direct_add(output[32], output[34]);
    output[32] = direct_multiply(temporary[30], output[36]);
    output[34] = direct_multiply(output[45], output[33]);
    output[36] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), output[40]);
    output[33] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), output[35]);
    output[40] = direct_add(output[36], output[33]);
    output[35] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[40]);
    output[36] = direct_add(output[34], output[35]);
    output[33] = direct_multiply(temporary[30], output[36]);
    output[40] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point));
    output[35] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), output[38]);
    output[34] = direct_add(output[40], output[35]);
    output[36] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point));
    output[38] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), output[37]);
    output[35] = direct_add(output[36], output[38]);
    output[40] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    output[38] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), output[39]);
    output[36] = direct_add(output[40], output[38]);
    output[38] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point));
    output[40] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), output[37]);
    output[37] = direct_add(output[38], output[40]);
    output[38] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    output[40] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), output[39]);
    output[38] = direct_add(output[38], output[40]);
    output[40] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    output[39] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), output[39]);
    output[39] = direct_add(output[40], output[39]);
    output[40] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point));
    output[42] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    temporary[50] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[48] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[42] = direct_add(direct_add(output[42], temporary[50]), output[48]);
    temporary[50] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[42]);
    output[48] = direct_add(output[40], temporary[50]);
    output[42] = direct_multiply(output[46], output[48]);
    output[40] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), output[43]);
    temporary[50] = direct_multiply(direct_real_plane<Lane>(inputs[48].values, point), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[46] = direct_multiply(direct_real_plane<Lane>(inputs[49].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[41] = direct_multiply(direct_real_plane<Lane>(inputs[50].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[44] = direct_multiply(direct_real_plane<Lane>(inputs[51].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[46] = direct_add(direct_add(output[46], output[41]), output[44]);
    output[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[46]);
    output[44] = direct_add(temporary[50], output[41]);
    output[46] = direct_multiply(temporary[27], output[44]);
    temporary[50] = direct_add(output[40], output[46]);
    output[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[50]);
    output[46] = direct_add(output[42], output[41]);
    output[40] = direct_multiply(temporary[30], output[46]);
    temporary[50] = direct_multiply(temporary[8], output[48]);
    output[42] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), output[43]);
    output[41] = direct_multiply(output[49], output[44]);
    output[46] = direct_add(output[42], output[41]);
    temporary[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[46]);
    output[42] = direct_add(temporary[50], temporary[8]);
    output[41] = direct_multiply(temporary[30], output[42]);
    output[46] = direct_multiply(output[47], output[48]);
    temporary[8] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), output[43]);
    temporary[50] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[44]);
    output[42] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), temporary[50]);
    output[47] = direct_add(temporary[8], output[42]);
    output[44] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[47]);
    temporary[8] = direct_add(output[46], output[44]);
    output[42] = direct_multiply(temporary[30], temporary[8]);
    output[47] = direct_multiply(output[45], output[48]);
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), output[43]);
    output[44] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), temporary[50]);
    temporary[8] = direct_add(output[46], output[44]);
    output[45] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[8]);
    output[48] = direct_add(output[47], output[45]);
    output[43] = direct_multiply(temporary[30], output[48]);
    temporary[50] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    output[46] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[27]);
    output[44] = direct_add(temporary[50], output[46]);
    temporary[8] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[47] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), temporary[27]);
    output[45] = direct_add(temporary[8], output[47]);
    temporary[30] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[48] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), temporary[27]);
    output[46] = direct_add(temporary[30], output[48]);
    temporary[50] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    temporary[8] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), output[49]);
    output[47] = direct_add(temporary[50], temporary[8]);
    temporary[27] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    temporary[30] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), output[49]);
    output[48] = direct_add(temporary[27], temporary[30]);
    temporary[8] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    temporary[50] = direct_multiply(direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point)), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    output[49] = direct_add(temporary[8], temporary[50]);
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
    direct_store_lane(outputs[40].values, point, output[20].real);
    direct_store_lane(outputs[41].values, point, output[20].imaginary);
    direct_store_lane(outputs[42].values, point, output[21].real);
    direct_store_lane(outputs[43].values, point, output[21].imaginary);
    direct_store_lane(outputs[44].values, point, output[22].real);
    direct_store_lane(outputs[45].values, point, output[22].imaginary);
    direct_store_lane(outputs[46].values, point, output[23].real);
    direct_store_lane(outputs[47].values, point, output[23].imaginary);
    direct_store_lane(outputs[48].values, point, output[24].real);
    direct_store_lane(outputs[49].values, point, output[24].imaginary);
    direct_store_lane(outputs[50].values, point, output[25].real);
    direct_store_lane(outputs[51].values, point, output[25].imaginary);
    direct_store_lane(outputs[52].values, point, output[26].real);
    direct_store_lane(outputs[53].values, point, output[26].imaginary);
    direct_store_lane(outputs[54].values, point, output[27].real);
    direct_store_lane(outputs[55].values, point, output[27].imaginary);
    direct_store_lane(outputs[56].values, point, output[28].real);
    direct_store_lane(outputs[57].values, point, output[28].imaginary);
    direct_store_lane(outputs[58].values, point, output[29].real);
    direct_store_lane(outputs[59].values, point, output[29].imaginary);
    direct_store_lane(outputs[60].values, point, output[30].real);
    direct_store_lane(outputs[61].values, point, output[30].imaginary);
    direct_store_lane(outputs[62].values, point, output[31].real);
    direct_store_lane(outputs[63].values, point, output[31].imaginary);
    direct_store_lane(outputs[64].values, point, output[32].real);
    direct_store_lane(outputs[65].values, point, output[32].imaginary);
    direct_store_lane(outputs[66].values, point, output[33].real);
    direct_store_lane(outputs[67].values, point, output[33].imaginary);
    direct_store_lane(outputs[68].values, point, output[34].real);
    direct_store_lane(outputs[69].values, point, output[34].imaginary);
    direct_store_lane(outputs[70].values, point, output[35].real);
    direct_store_lane(outputs[71].values, point, output[35].imaginary);
    direct_store_lane(outputs[72].values, point, output[36].real);
    direct_store_lane(outputs[73].values, point, output[36].imaginary);
    direct_store_lane(outputs[74].values, point, output[37].real);
    direct_store_lane(outputs[75].values, point, output[37].imaginary);
    direct_store_lane(outputs[76].values, point, output[38].real);
    direct_store_lane(outputs[77].values, point, output[38].imaginary);
    direct_store_lane(outputs[78].values, point, output[39].real);
    direct_store_lane(outputs[79].values, point, output[39].imaginary);
    direct_store_lane(outputs[80].values, point, output[40].real);
    direct_store_lane(outputs[81].values, point, output[40].imaginary);
    direct_store_lane(outputs[82].values, point, output[41].real);
    direct_store_lane(outputs[83].values, point, output[41].imaginary);
    direct_store_lane(outputs[84].values, point, output[42].real);
    direct_store_lane(outputs[85].values, point, output[42].imaginary);
    direct_store_lane(outputs[86].values, point, output[43].real);
    direct_store_lane(outputs[87].values, point, output[43].imaginary);
    direct_store_lane(outputs[88].values, point, output[44].real);
    direct_store_lane(outputs[89].values, point, output[44].imaginary);
    direct_store_lane(outputs[90].values, point, output[45].real);
    direct_store_lane(outputs[91].values, point, output[45].imaginary);
    direct_store_lane(outputs[92].values, point, output[46].real);
    direct_store_lane(outputs[93].values, point, output[46].imaginary);
    direct_store_lane(outputs[94].values, point, output[47].real);
    direct_store_lane(outputs[95].values, point, output[47].imaginary);
    direct_store_lane(outputs[96].values, point, output[48].real);
    direct_store_lane(outputs[97].values, point, output[48].imaginary);
    direct_store_lane(outputs[98].values, point, output[49].real);
    direct_store_lane(outputs[99].values, point, output[49].imaginary);
}

}  // namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1

extern "C" pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1::NativeCompiledDirectMetadataV1
pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1_direct_application_v1_metadata() noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1;
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
pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1_direct_application_v1_target_triple() noexcept {
    return "x86_64-unknown-linux-gnu";
}

extern "C" const char*
pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1_direct_application_v1_cpu_features() noexcept {
    return "";
}

extern "C" const char*
pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1_direct_application_v1_source_sha256() noexcept {
    return "800c522d4e767af7065763f6d658a3b279b52063683b7dde23181651839d0dc9";
}

extern "C" std::uint32_t
pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1_direct_application_v1_logical_stack_bytes() noexcept {
    return pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1::kLogicalStackBytes;
}

extern "C" int pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1_direct_application_v1(
    const pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1::
        NativeCompiledDirectInputPlaneV1* inputs,
    std::uint32_t input_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1::
        NativeCompiledDirectScalarV1* scalars,
    std::uint32_t scalar_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1::
        NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t output_count,
    std::uint32_t point_start,
    std::uint32_t point_count
) noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_1_subset_2_chunk_0_57273a31086b451bb863c923cebb18b1;
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
