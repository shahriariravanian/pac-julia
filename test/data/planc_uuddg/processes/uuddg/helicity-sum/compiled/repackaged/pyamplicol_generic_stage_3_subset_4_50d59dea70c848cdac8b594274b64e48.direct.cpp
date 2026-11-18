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

namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48 {

constexpr std::uint32_t kInputPlaneCount = 148u;
constexpr std::uint32_t kScalarInputCount = 0u;
constexpr std::uint32_t kOutputPlaneCount = 64u;
constexpr std::uint32_t kSimdLaneWidth = 2u;
constexpr std::uint32_t kInstructionCount = 254u;
constexpr std::uint32_t kTemporaryCount = 48u;
constexpr std::uint32_t kLogicalStackBytes = 2560u;
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
    std::array<DirectComplex<Lane>, 48> temporary;
    std::array<DirectComplex<Lane>, 32> output;
    output[9] = direct_constant<Lane>(0x1.6a09e667f3bcdp-1, 0x0.0p+0);
    temporary[1] = direct_add(direct_complex_plane<Lane>(inputs[88].values, inputs[89].values, point), direct_complex_plane<Lane>(inputs[94].values, inputs[95].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[1]);
    temporary[3] = direct_multiply(direct_complex_plane<Lane>(inputs[92].values, inputs[93].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    temporary[3] = direct_add(direct_complex_plane<Lane>(inputs[90].values, inputs[91].values, point), temporary[3]);
    temporary[4] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), temporary[3]);
    output[21] = direct_add(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    output[0] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), output[21]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    temporary[7] = direct_add(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), temporary[7]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), temporary[7]);
    temporary[2] = direct_add(direct_add(direct_add(temporary[2], temporary[4]), output[0]), output[7]);
    temporary[4] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0), output[9]);
    output[0] = direct_multiply(temporary[2], temporary[4]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[92].values, inputs[93].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    temporary[2] = direct_add(direct_complex_plane<Lane>(inputs[90].values, inputs[91].values, point), output[7]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[2]);
    output[5] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[94].values, inputs[95].values, point));
    output[5] = direct_add(direct_complex_plane<Lane>(inputs[88].values, inputs[89].values, point), output[5]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[5]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    output[12] = direct_add(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), output[12]);
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), output[12]);
    output[6] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point));
    output[6] = direct_add(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), output[6]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), output[6]);
    output[7] = direct_add(direct_add(direct_add(output[7], output[1]), output[10]), output[4]);
    output[1] = direct_multiply(temporary[4], output[7]);
    output[10] = direct_add(direct_complex_plane<Lane>(inputs[96].values, inputs[97].values, point), direct_complex_plane<Lane>(inputs[102].values, inputs[103].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[10]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[100].values, inputs[101].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    output[7] = direct_add(direct_complex_plane<Lane>(inputs[98].values, inputs[99].values, point), output[7]);
    output[2] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[7]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), output[21]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), temporary[7]);
    output[4] = direct_add(direct_add(direct_add(output[4], output[2]), output[8]), output[14]);
    output[2] = direct_multiply(temporary[4], output[4]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[100].values, inputs[101].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    output[14] = direct_add(direct_complex_plane<Lane>(inputs[98].values, inputs[99].values, point), output[8]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[14]);
    output[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[102].values, inputs[103].values, point));
    output[8] = direct_add(direct_complex_plane<Lane>(inputs[96].values, inputs[97].values, point), output[8]);
    output[3] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[8]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), output[12]);
    temporary[20] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), output[6]);
    output[4] = direct_add(direct_add(direct_add(output[4], output[3]), output[11]), temporary[20]);
    output[3] = direct_multiply(temporary[4], output[4]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[5]);
    temporary[20] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point));
    output[4] = direct_multiply(temporary[3], temporary[20]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), output[6]);
    temporary[3] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point));
    output[15] = direct_multiply(temporary[7], temporary[3]);
    output[5] = direct_add(direct_add(direct_add(output[11], output[4]), output[5]), output[15]);
    output[4] = direct_multiply(temporary[4], output[5]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), temporary[2]);
    output[15] = direct_multiply(temporary[1], temporary[20]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), output[12]);
    temporary[2] = direct_multiply(output[21], temporary[3]);
    temporary[1] = direct_add(direct_add(direct_add(output[11], output[15]), output[5]), temporary[2]);
    temporary[2] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0), output[9]);
    output[5] = direct_multiply(temporary[1], temporary[2]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[8]);
    output[15] = direct_multiply(output[7], temporary[20]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), output[6]);
    temporary[1] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point));
    output[8] = direct_multiply(temporary[7], temporary[1]);
    output[7] = direct_add(direct_add(direct_add(output[11], output[15]), output[9]), output[8]);
    output[6] = direct_multiply(temporary[4], output[7]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[14]);
    output[9] = direct_multiply(output[10], temporary[20]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), output[12]);
    output[11] = direct_multiply(output[21], temporary[1]);
    output[15] = direct_add(direct_add(direct_add(temporary[7], output[9]), output[8]), output[11]);
    output[7] = direct_multiply(temporary[2], output[15]);
    output[14] = direct_add(direct_complex_plane<Lane>(inputs[72].values, inputs[73].values, point), direct_complex_plane<Lane>(inputs[78].values, inputs[79].values, point));
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[14]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[76].values, inputs[77].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    output[21] = direct_add(direct_complex_plane<Lane>(inputs[74].values, inputs[75].values, point), output[12]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[21]);
    temporary[7] = direct_add(output[10], output[9]);
    output[8] = direct_multiply(temporary[4], temporary[7]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[76].values, inputs[77].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    output[15] = direct_add(direct_complex_plane<Lane>(inputs[74].values, inputs[75].values, point), output[11]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[15]);
    output[9] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[78].values, inputs[79].values, point));
    output[10] = direct_add(direct_complex_plane<Lane>(inputs[72].values, inputs[73].values, point), output[9]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[10]);
    output[11] = direct_add(output[12], temporary[7]);
    output[9] = direct_multiply(temporary[4], output[11]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[10]);
    output[12] = direct_multiply(temporary[20], output[21]);
    output[11] = direct_add(temporary[7], output[12]);
    output[10] = direct_multiply(temporary[4], output[11]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[15]);
    temporary[7] = direct_multiply(temporary[20], output[14]);
    output[12] = direct_add(output[21], temporary[7]);
    output[11] = direct_multiply(temporary[2], output[12]);
    output[15] = direct_add(direct_complex_plane<Lane>(inputs[80].values, inputs[81].values, point), direct_complex_plane<Lane>(inputs[86].values, inputs[87].values, point));
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[15]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[84].values, inputs[85].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    temporary[7] = direct_add(direct_complex_plane<Lane>(inputs[82].values, inputs[83].values, point), output[21]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), temporary[7]);
    output[21] = direct_add(output[14], output[12]);
    output[12] = direct_multiply(temporary[4], output[21]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[84].values, inputs[85].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    output[21] = direct_add(direct_complex_plane<Lane>(inputs[82].values, inputs[83].values, point), output[14]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[21]);
    output[22] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[86].values, inputs[87].values, point));
    output[22] = direct_add(direct_complex_plane<Lane>(inputs[80].values, inputs[81].values, point), output[22]);
    output[13] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[22]);
    output[14] = direct_add(output[14], output[13]);
    output[13] = direct_multiply(temporary[4], output[14]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[22]);
    output[22] = direct_multiply(temporary[20], temporary[7]);
    temporary[7] = direct_add(output[14], output[22]);
    output[14] = direct_multiply(temporary[4], temporary[7]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[21]);
    temporary[7] = direct_multiply(temporary[20], output[15]);
    output[21] = direct_add(output[22], temporary[7]);
    output[15] = direct_multiply(temporary[2], output[21]);
    temporary[7] = direct_add(direct_complex_plane<Lane>(inputs[104].values, inputs[105].values, point), direct_complex_plane<Lane>(inputs[110].values, inputs[111].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[7]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[108].values, inputs[109].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    output[21] = direct_add(direct_complex_plane<Lane>(inputs[106].values, inputs[107].values, point), output[21]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[21]);
    output[29] = direct_add(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    output[24] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), output[29]);
    temporary[27] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    temporary[27] = direct_add(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[27]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), temporary[27]);
    output[22] = direct_add(direct_add(direct_add(output[22], output[16]), output[24]), output[28]);
    output[16] = direct_multiply(temporary[4], output[22]);
    output[24] = direct_multiply(direct_complex_plane<Lane>(inputs[108].values, inputs[109].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    output[28] = direct_add(direct_complex_plane<Lane>(inputs[106].values, inputs[107].values, point), output[24]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[28]);
    output[24] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[110].values, inputs[111].values, point));
    output[24] = direct_add(direct_complex_plane<Lane>(inputs[104].values, inputs[105].values, point), output[24]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[24]);
    temporary[30] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    temporary[30] = direct_add(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[30]);
    temporary[31] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), temporary[30]);
    temporary[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    temporary[32] = direct_add(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), temporary[32]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), temporary[32]);
    output[22] = direct_add(direct_add(direct_add(output[22], output[17]), temporary[31]), output[20]);
    output[17] = direct_multiply(temporary[4], output[22]);
    temporary[31] = direct_add(direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[31]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    output[22] = direct_add(direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point), output[22]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[22]);
    temporary[35] = direct_add(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    output[30] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), temporary[35]);
    temporary[37] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    temporary[37] = direct_add(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[37]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), temporary[37]);
    output[20] = direct_add(direct_add(direct_add(output[20], output[18]), output[30]), output[23]);
    output[18] = direct_multiply(temporary[4], output[20]);
    output[30] = direct_multiply(direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    output[23] = direct_add(direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point), output[30]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[23]);
    output[30] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    output[30] = direct_add(direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point), output[30]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[30]);
    temporary[40] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    temporary[40] = direct_add(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[40]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), temporary[40]);
    temporary[42] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    temporary[42] = direct_add(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), temporary[42]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), temporary[42]);
    output[20] = direct_add(direct_add(direct_add(output[20], output[19]), output[25]), output[26]);
    output[19] = direct_multiply(temporary[4], output[20]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[24]);
    output[26] = direct_multiply(temporary[20], output[21]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), temporary[32]);
    output[24] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point));
    output[21] = direct_multiply(temporary[27], output[24]);
    output[21] = direct_add(direct_add(direct_add(output[25], output[26]), output[20]), output[21]);
    output[20] = direct_multiply(temporary[4], output[21]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[28]);
    output[26] = direct_multiply(temporary[20], temporary[7]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), temporary[30]);
    output[28] = direct_multiply(output[29], output[24]);
    temporary[7] = direct_add(direct_add(direct_add(output[25], output[26]), output[21]), output[28]);
    output[21] = direct_multiply(temporary[2], temporary[7]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[30]);
    output[25] = direct_multiply(temporary[20], output[22]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), temporary[42]);
    temporary[7] = direct_multiply(temporary[37], output[24]);
    output[30] = direct_add(direct_add(direct_add(output[28], output[25]), output[26]), temporary[7]);
    output[22] = direct_multiply(temporary[4], output[30]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[23]);
    output[28] = direct_multiply(temporary[20], temporary[31]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), temporary[40]);
    output[26] = direct_multiply(temporary[35], output[24]);
    output[30] = direct_add(direct_add(direct_add(temporary[7], output[28]), output[25]), output[26]);
    output[23] = direct_multiply(temporary[2], output[30]);
    temporary[20] = direct_add(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    temporary[31] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), temporary[20]);
    output[24] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0));
    temporary[7] = direct_add(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), output[24]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), temporary[7]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point), output[29]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point), temporary[27]);
    output[30] = direct_add(direct_add(direct_add(temporary[31], output[28]), output[25]), output[26]);
    output[24] = direct_multiply(temporary[4], output[30]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p+0));
    temporary[31] = direct_add(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), output[28]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), temporary[31]);
    output[26] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    output[30] = direct_add(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), output[26]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), output[30]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point), temporary[30]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point), temporary[32]);
    output[28] = direct_add(direct_add(direct_add(output[25], output[28]), output[26]), output[27]);
    output[25] = direct_multiply(temporary[4], output[28]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), temporary[20]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), temporary[7]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point), temporary[35]);
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point), temporary[37]);
    output[28] = direct_add(direct_add(direct_add(output[26], output[27]), output[28]), output[31]);
    output[26] = direct_multiply(temporary[4], output[28]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), temporary[31]);
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), output[30]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point), temporary[40]);
    temporary[46] = direct_multiply(direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point), temporary[42]);
    output[28] = direct_add(direct_add(direct_add(output[27], output[31]), output[28]), temporary[46]);
    output[27] = direct_multiply(temporary[4], output[28]);
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), output[30]);
    temporary[46] = direct_multiply(temporary[3], temporary[7]);
    output[28] = direct_multiply(direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point), temporary[32]);
    temporary[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[70].values, inputs[71].values, point));
    temporary[27] = direct_multiply(temporary[27], temporary[32]);
    temporary[27] = direct_add(direct_add(direct_add(output[31], temporary[46]), output[28]), temporary[27]);
    output[28] = direct_multiply(temporary[4], temporary[27]);
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), temporary[31]);
    temporary[46] = direct_multiply(temporary[3], temporary[20]);
    temporary[27] = direct_multiply(direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point), temporary[30]);
    temporary[3] = direct_multiply(output[29], temporary[32]);
    temporary[30] = direct_add(direct_add(direct_add(output[31], temporary[46]), temporary[27]), temporary[3]);
    output[29] = direct_multiply(temporary[2], temporary[30]);
    temporary[3] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), output[30]);
    temporary[27] = direct_multiply(temporary[1], temporary[7]);
    output[31] = direct_multiply(direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point), temporary[42]);
    temporary[46] = direct_multiply(temporary[37], temporary[32]);
    temporary[30] = direct_add(direct_add(direct_add(temporary[3], temporary[27]), output[31]), temporary[46]);
    output[30] = direct_multiply(temporary[4], temporary[30]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), temporary[31]);
    temporary[42] = direct_multiply(temporary[1], temporary[20]);
    temporary[37] = direct_multiply(direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point), temporary[40]);
    temporary[3] = direct_multiply(temporary[35], temporary[32]);
    temporary[27] = direct_add(direct_add(direct_add(temporary[7], temporary[42]), temporary[37]), temporary[3]);
    output[31] = direct_multiply(temporary[2], temporary[27]);
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
}

}  // namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48

extern "C" pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48::NativeCompiledDirectMetadataV1
pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48_direct_application_v1_metadata() noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48;
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
pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48_direct_application_v1_target_triple() noexcept {
    return "x86_64-unknown-linux-gnu";
}

extern "C" const char*
pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48_direct_application_v1_cpu_features() noexcept {
    return "";
}

extern "C" const char*
pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48_direct_application_v1_source_sha256() noexcept {
    return "99b9fd09177f20a6ca021d190257d373d07d62b8e5c05e1e2218f1ae66df58d7";
}

extern "C" std::uint32_t
pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48_direct_application_v1_logical_stack_bytes() noexcept {
    return pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48::kLogicalStackBytes;
}

extern "C" int pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48_direct_application_v1(
    const pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48::
        NativeCompiledDirectInputPlaneV1* inputs,
    std::uint32_t input_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48::
        NativeCompiledDirectScalarV1* scalars,
    std::uint32_t scalar_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48::
        NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t output_count,
    std::uint32_t point_start,
    std::uint32_t point_count
) noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_50d59dea70c848cdac8b594274b64e48;
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
