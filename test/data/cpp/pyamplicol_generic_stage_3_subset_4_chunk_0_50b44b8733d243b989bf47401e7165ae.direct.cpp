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

namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae {

constexpr std::uint32_t kInputPlaneCount = 388u;
constexpr std::uint32_t kScalarInputCount = 0u;
constexpr std::uint32_t kOutputPlaneCount = 56u;
constexpr std::uint32_t kSimdLaneWidth = 2u;
constexpr std::uint32_t kInstructionCount = 1291u;
constexpr std::uint32_t kTemporaryCount = 69u;
constexpr std::uint32_t kLogicalStackBytes = 3104u;
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
    std::array<DirectComplex<Lane>, 69> temporary;
    std::array<DirectComplex<Lane>, 28> output;
    output[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[232].values, inputs[233].values, point));
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[234].values, inputs[235].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[236].values, inputs[237].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point));
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point));
    temporary[5] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point));
    output[0] = direct_add(direct_add(direct_add(direct_add(direct_add(output[0], temporary[1]), temporary[2]), output[4]), output[1]), temporary[5]);
    temporary[1] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), output[0]);
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[152].values, inputs[153].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[154].values, inputs[155].values, point));
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[156].values, inputs[157].values, point));
    temporary[5] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    output[0] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    temporary[6] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    output[0] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[2], output[4]), output[1]), temporary[5]), output[0]), temporary[6]);
    temporary[2] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), output[0]);
    output[4] = direct_constant<Lane>(0x1.6a09e667f3bcdp-1, 0x0.0p+0);
    output[1] = direct_multiply(direct_real_plane<Lane>(inputs[384].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[5] = direct_add(direct_real_plane<Lane>(inputs[364].values, point), output[1]);
    temporary[6] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[224].values, inputs[225].values, point));
    output[0] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[226].values, inputs[227].values, point));
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[228].values, inputs[229].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[230].values, inputs[231].values, point));
    output[0] = direct_add(direct_add(output[0], output[1]), temporary[7]);
    output[1] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[0]);
    temporary[7] = direct_add(temporary[6], output[1]);
    output[0] = direct_multiply(temporary[5], temporary[7]);
    output[1] = direct_multiply(direct_real_plane<Lane>(inputs[380].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[6] = direct_add(direct_real_plane<Lane>(inputs[368].values, point), output[1]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[144].values, inputs[145].values, point));
    temporary[8] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[146].values, inputs[147].values, point));
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[148].values, inputs[149].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[150].values, inputs[151].values, point));
    temporary[8] = direct_add(direct_add(temporary[8], output[27]), temporary[10]);
    output[27] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[8]);
    temporary[10] = direct_add(output[1], output[27]);
    temporary[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[6]);
    output[1] = direct_multiply(temporary[8], temporary[10]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[376].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[6] = direct_add(direct_real_plane<Lane>(inputs[372].values, point), output[27]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    temporary[12] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    temporary[11] = direct_add(direct_add(temporary[11], temporary[12]), output[12]);
    temporary[12] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[11]);
    output[12] = direct_add(output[27], temporary[12]);
    temporary[11] = direct_multiply(temporary[6], output[12]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[364].values, point), direct_complex_plane<Lane>(inputs[224].values, inputs[225].values, point));
    temporary[12] = direct_multiply(direct_real_plane<Lane>(inputs[365].values, point), direct_complex_plane<Lane>(inputs[226].values, inputs[227].values, point));
    temporary[14] = direct_multiply(direct_real_plane<Lane>(inputs[366].values, point), direct_complex_plane<Lane>(inputs[228].values, inputs[229].values, point));
    temporary[15] = direct_multiply(direct_real_plane<Lane>(inputs[367].values, point), direct_complex_plane<Lane>(inputs[230].values, inputs[231].values, point));
    temporary[12] = direct_add(direct_add(temporary[12], temporary[14]), temporary[15]);
    temporary[14] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[12]);
    temporary[15] = direct_add(output[27], temporary[14]);
    temporary[12] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[15]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[384].values, point), direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point));
    temporary[14] = direct_multiply(direct_real_plane<Lane>(inputs[385].values, point), direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point));
    temporary[16] = direct_multiply(direct_real_plane<Lane>(inputs[386].values, point), direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point));
    temporary[17] = direct_multiply(direct_real_plane<Lane>(inputs[387].values, point), direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point));
    temporary[14] = direct_add(direct_add(temporary[14], temporary[16]), temporary[17]);
    temporary[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[14]);
    temporary[17] = direct_add(output[27], temporary[16]);
    temporary[14] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[17]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[224].values, inputs[225].values, point), temporary[14]);
    temporary[16] = direct_add(temporary[12], output[27]);
    temporary[17] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[16]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[368].values, point), direct_complex_plane<Lane>(inputs[144].values, inputs[145].values, point));
    temporary[12] = direct_multiply(direct_real_plane<Lane>(inputs[369].values, point), direct_complex_plane<Lane>(inputs[146].values, inputs[147].values, point));
    temporary[16] = direct_multiply(direct_real_plane<Lane>(inputs[370].values, point), direct_complex_plane<Lane>(inputs[148].values, inputs[149].values, point));
    output[8] = direct_multiply(direct_real_plane<Lane>(inputs[371].values, point), direct_complex_plane<Lane>(inputs[150].values, inputs[151].values, point));
    temporary[12] = direct_add(direct_add(temporary[12], temporary[16]), output[8]);
    temporary[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[12]);
    output[8] = direct_add(output[27], temporary[16]);
    temporary[12] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), output[8]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[380].values, point), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    temporary[16] = direct_multiply(direct_real_plane<Lane>(inputs[381].values, point), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    temporary[19] = direct_multiply(direct_real_plane<Lane>(inputs[382].values, point), direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point));
    output[2] = direct_multiply(direct_real_plane<Lane>(inputs[383].values, point), direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point));
    temporary[16] = direct_add(direct_add(temporary[16], temporary[19]), output[2]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[16]);
    output[2] = direct_add(output[27], temporary[19]);
    temporary[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[2]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[144].values, inputs[145].values, point), temporary[16]);
    temporary[19] = direct_add(temporary[12], output[27]);
    output[2] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[19]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[372].values, point), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    temporary[12] = direct_multiply(direct_real_plane<Lane>(inputs[373].values, point), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    temporary[19] = direct_multiply(direct_real_plane<Lane>(inputs[374].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    output[25] = direct_multiply(direct_real_plane<Lane>(inputs[375].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    temporary[12] = direct_add(direct_add(temporary[12], temporary[19]), output[25]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[12]);
    output[25] = direct_add(output[27], temporary[19]);
    temporary[12] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), output[25]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[376].values, point), direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point));
    temporary[19] = direct_multiply(direct_real_plane<Lane>(inputs[377].values, point), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    temporary[22] = direct_multiply(direct_real_plane<Lane>(inputs[378].values, point), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    temporary[23] = direct_multiply(direct_real_plane<Lane>(inputs[379].values, point), direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point));
    temporary[19] = direct_add(direct_add(temporary[19], temporary[22]), temporary[23]);
    temporary[22] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[19]);
    temporary[23] = direct_add(output[27], temporary[22]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[23]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point), temporary[19]);
    temporary[22] = direct_add(temporary[12], output[27]);
    temporary[23] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[22]);
    output[27] = direct_add(direct_add(direct_add(direct_add(direct_add(output[0], output[1]), temporary[11]), temporary[17]), output[2]), temporary[23]);
    temporary[12] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p+0), output[4]);
    temporary[22] = direct_multiply(output[27], temporary[12]);
    output[0] = direct_add(direct_add(temporary[1], temporary[2]), temporary[22]);
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[232].values, inputs[233].values, point));
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[238].values, inputs[239].values, point));
    temporary[17] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[240].values, inputs[241].values, point));
    output[2] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point));
    temporary[23] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[70].values, inputs[71].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[72].values, inputs[73].values, point));
    output[27] = direct_add(direct_add(direct_add(direct_add(direct_add(output[1], temporary[11]), temporary[17]), output[2]), temporary[23]), output[4]);
    temporary[1] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), output[27]);
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[152].values, inputs[153].values, point));
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[158].values, inputs[159].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[160].values, inputs[161].values, point));
    output[1] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    temporary[17] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    output[2] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[2], temporary[22]), output[4]), output[1]), temporary[11]), temporary[17]);
    temporary[23] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), output[2]);
    output[27] = direct_multiply(direct_real_plane<Lane>(inputs[385].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[2] = direct_add(direct_real_plane<Lane>(inputs[365].values, point), output[27]);
    output[4] = direct_multiply(temporary[2], temporary[7]);
    output[1] = direct_multiply(direct_real_plane<Lane>(inputs[381].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[11] = direct_add(direct_real_plane<Lane>(inputs[369].values, point), output[1]);
    temporary[17] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[11]);
    temporary[22] = direct_multiply(temporary[17], temporary[10]);
    output[2] = direct_multiply(direct_real_plane<Lane>(inputs[377].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[27] = direct_add(direct_real_plane<Lane>(inputs[373].values, point), output[2]);
    output[1] = direct_multiply(output[27], output[12]);
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), temporary[15]);
    output[2] = direct_multiply(direct_complex_plane<Lane>(inputs[226].values, inputs[227].values, point), temporary[14]);
    temporary[11] = direct_add(temporary[11], output[2]);
    output[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[11]);
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), output[8]);
    temporary[24] = direct_multiply(direct_complex_plane<Lane>(inputs[146].values, inputs[147].values, point), temporary[16]);
    temporary[11] = direct_add(temporary[11], temporary[24]);
    temporary[24] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[11]);
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), output[25]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point), temporary[19]);
    temporary[11] = direct_add(temporary[11], output[5]);
    output[5] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[11]);
    temporary[11] = direct_add(direct_add(direct_add(direct_add(direct_add(output[4], temporary[22]), output[1]), output[2]), temporary[24]), output[5]);
    output[4] = direct_multiply(temporary[12], temporary[11]);
    output[1] = direct_add(direct_add(temporary[1], temporary[23]), output[4]);
    output[2] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[238].values, inputs[239].values, point));
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[154].values, inputs[155].values, point));
    temporary[24] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[158].values, inputs[159].values, point), temporary[24]);
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[162].values, inputs[163].values, point));
    temporary[1] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[70].values, inputs[71].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    temporary[23] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    temporary[26] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), temporary[23]);
    temporary[27] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    temporary[28] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[234].values, inputs[235].values, point));
    temporary[29] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[242].values, inputs[243].values, point));
    temporary[28] = direct_add(temporary[28], temporary[29]);
    temporary[29] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[28]);
    temporary[28] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point));
    temporary[30] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[74].values, inputs[75].values, point));
    temporary[28] = direct_add(temporary[28], temporary[30]);
    temporary[30] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[28]);
    temporary[28] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[2], temporary[22]), output[5]), temporary[11]), temporary[1]), output[4]), temporary[26]), temporary[27]), temporary[29]), temporary[30]);
    temporary[1] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[28]);
    output[4] = direct_multiply(direct_real_plane<Lane>(inputs[386].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[11] = direct_add(direct_real_plane<Lane>(inputs[366].values, point), output[4]);
    output[2] = direct_multiply(temporary[11], temporary[7]);
    temporary[22] = direct_multiply(direct_real_plane<Lane>(inputs[382].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[5] = direct_add(direct_real_plane<Lane>(inputs[370].values, point), temporary[22]);
    temporary[26] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[5]);
    temporary[27] = direct_multiply(temporary[26], temporary[10]);
    temporary[29] = direct_multiply(direct_real_plane<Lane>(inputs[378].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[30] = direct_add(direct_real_plane<Lane>(inputs[374].values, point), temporary[29]);
    temporary[28] = direct_multiply(temporary[30], output[12]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), temporary[15]);
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[228].values, inputs[229].values, point), temporary[14]);
    output[5] = direct_add(output[4], temporary[22]);
    temporary[29] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[5]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), output[8]);
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[148].values, inputs[149].values, point), temporary[16]);
    output[5] = direct_add(output[4], temporary[22]);
    output[4] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[5]);
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), output[25]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point), temporary[19]);
    temporary[22] = direct_add(temporary[22], output[5]);
    output[5] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[22]);
    temporary[22] = direct_add(direct_add(direct_add(direct_add(direct_add(output[2], temporary[27]), temporary[28]), temporary[29]), output[4]), output[5]);
    output[4] = direct_multiply(temporary[12], temporary[22]);
    output[2] = direct_add(temporary[1], output[4]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[236].values, inputs[237].values, point));
    temporary[27] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point));
    temporary[28] = direct_multiply(direct_complex_plane<Lane>(inputs[240].values, inputs[241].values, point), temporary[27]);
    temporary[29] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point));
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[242].values, inputs[243].values, point), temporary[29]);
    temporary[1] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[156].values, inputs[157].values, point), temporary[1]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[160].values, inputs[161].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[162].values, inputs[163].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point));
    temporary[34] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point));
    temporary[35] = direct_multiply(direct_complex_plane<Lane>(inputs[72].values, inputs[73].values, point), temporary[34]);
    output[23] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point));
    temporary[37] = direct_multiply(direct_complex_plane<Lane>(inputs[74].values, inputs[75].values, point), output[23]);
    temporary[38] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    output[3] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), temporary[38]);
    temporary[40] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    output[4] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[5], temporary[28]), temporary[22]), output[4]), output[9]), temporary[32]), output[22]), temporary[35]), temporary[37]), output[3]), temporary[40]), temporary[41]);
    temporary[22] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), output[4]);
    output[5] = direct_multiply(direct_real_plane<Lane>(inputs[387].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[28] = direct_add(direct_real_plane<Lane>(inputs[367].values, point), output[5]);
    output[9] = direct_multiply(temporary[28], temporary[7]);
    temporary[32] = direct_multiply(direct_real_plane<Lane>(inputs[383].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    output[22] = direct_add(direct_real_plane<Lane>(inputs[371].values, point), temporary[32]);
    temporary[35] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[22]);
    temporary[37] = direct_multiply(temporary[35], temporary[10]);
    output[3] = direct_multiply(direct_real_plane<Lane>(inputs[379].values, point), direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0));
    temporary[40] = direct_add(direct_real_plane<Lane>(inputs[375].values, point), output[3]);
    temporary[41] = direct_multiply(temporary[40], output[12]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), temporary[15]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[230].values, inputs[231].values, point), temporary[14]);
    temporary[7] = direct_add(output[4], output[5]);
    temporary[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[7]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), output[8]);
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[150].values, inputs[151].values, point), temporary[16]);
    output[3] = direct_add(output[22], temporary[10]);
    output[12] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[3]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), output[25]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point), temporary[19]);
    output[5] = direct_add(temporary[15], output[4]);
    temporary[7] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[5]);
    temporary[10] = direct_add(direct_add(direct_add(direct_add(direct_add(output[9], temporary[37]), temporary[41]), temporary[32]), output[12]), temporary[7]);
    output[22] = direct_multiply(temporary[12], temporary[10]);
    output[3] = direct_add(temporary[22], output[22]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[252].values, inputs[253].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[254].values, inputs[255].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[256].values, inputs[257].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[76].values, inputs[77].values, point));
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[78].values, inputs[79].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[80].values, inputs[81].values, point));
    temporary[32] = direct_add(direct_add(direct_add(direct_add(direct_add(output[4], temporary[15]), output[5]), temporary[7]), output[12]), output[9]);
    temporary[37] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[32]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[152].values, inputs[153].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[154].values, inputs[155].values, point));
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[156].values, inputs[157].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    output[12] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[41], temporary[10]), temporary[22]), output[22]), output[4]), temporary[7]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), output[12]);
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[244].values, inputs[245].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[246].values, inputs[247].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[248].values, inputs[249].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[250].values, inputs[251].values, point));
    temporary[7] = direct_add(direct_add(output[9], temporary[32]), output[4]);
    temporary[10] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[7]);
    temporary[22] = direct_add(output[5], temporary[10]);
    output[22] = direct_multiply(temporary[5], temporary[22]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[144].values, inputs[145].values, point));
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[146].values, inputs[147].values, point));
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[148].values, inputs[149].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[150].values, inputs[151].values, point));
    temporary[32] = direct_add(direct_add(output[12], output[4]), output[9]);
    temporary[7] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[32]);
    temporary[10] = direct_add(temporary[41], temporary[7]);
    output[5] = direct_multiply(temporary[8], temporary[10]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    temporary[7] = direct_add(direct_add(output[12], output[9]), temporary[32]);
    temporary[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[7]);
    output[12] = direct_add(output[4], temporary[41]);
    output[9] = direct_multiply(temporary[6], output[12]);
    temporary[32] = direct_multiply(direct_real_plane<Lane>(inputs[364].values, point), direct_complex_plane<Lane>(inputs[244].values, inputs[245].values, point));
    temporary[7] = direct_multiply(direct_real_plane<Lane>(inputs[365].values, point), direct_complex_plane<Lane>(inputs[246].values, inputs[247].values, point));
    output[4] = direct_multiply(direct_real_plane<Lane>(inputs[366].values, point), direct_complex_plane<Lane>(inputs[248].values, inputs[249].values, point));
    temporary[41] = direct_multiply(direct_real_plane<Lane>(inputs[367].values, point), direct_complex_plane<Lane>(inputs[250].values, inputs[251].values, point));
    output[4] = direct_add(direct_add(temporary[7], output[4]), temporary[41]);
    temporary[7] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[4]);
    temporary[41] = direct_add(temporary[32], temporary[7]);
    output[4] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[41]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[244].values, inputs[245].values, point), temporary[14]);
    temporary[32] = direct_add(output[4], temporary[7]);
    output[4] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[32]);
    temporary[7] = direct_multiply(direct_real_plane<Lane>(inputs[380].values, point), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    temporary[32] = direct_multiply(direct_real_plane<Lane>(inputs[381].values, point), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    temporary[42] = direct_multiply(direct_real_plane<Lane>(inputs[382].values, point), direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point));
    output[13] = direct_multiply(direct_real_plane<Lane>(inputs[383].values, point), direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point));
    temporary[32] = direct_add(direct_add(temporary[32], temporary[42]), output[13]);
    temporary[42] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[32]);
    output[13] = direct_add(temporary[7], temporary[42]);
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[144].values, inputs[145].values, point), output[13]);
    temporary[7] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[8]);
    temporary[42] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), temporary[7]);
    output[8] = direct_add(temporary[32], temporary[42]);
    temporary[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[8]);
    temporary[42] = direct_multiply(direct_real_plane<Lane>(inputs[372].values, point), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    output[8] = direct_multiply(direct_real_plane<Lane>(inputs[373].values, point), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    output[18] = direct_multiply(direct_real_plane<Lane>(inputs[374].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    output[24] = direct_multiply(direct_real_plane<Lane>(inputs[375].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    output[8] = direct_add(direct_add(output[8], output[18]), output[24]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[8]);
    output[24] = direct_add(temporary[42], output[18]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), output[24]);
    temporary[42] = direct_multiply(direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point), temporary[19]);
    output[18] = direct_add(output[8], temporary[42]);
    output[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[18]);
    temporary[42] = direct_add(direct_add(direct_add(direct_add(direct_add(output[22], output[5]), output[9]), output[4]), temporary[32]), output[8]);
    output[18] = direct_multiply(temporary[12], temporary[42]);
    output[4] = direct_add(direct_add(temporary[37], temporary[15]), output[18]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[252].values, inputs[253].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[258].values, inputs[259].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[260].values, inputs[261].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[76].values, inputs[77].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[82].values, inputs[83].values, point));
    temporary[42] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[84].values, inputs[85].values, point));
    temporary[15] = direct_add(direct_add(direct_add(direct_add(direct_add(output[8], output[5]), output[9]), temporary[32]), output[22]), temporary[42]);
    temporary[37] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[15]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[152].values, inputs[153].values, point));
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[158].values, inputs[159].values, point));
    output[5] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[160].values, inputs[161].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    temporary[42] = direct_add(direct_add(direct_add(direct_add(direct_add(output[18], output[8]), output[5]), output[9]), temporary[32]), output[22]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[42]);
    output[8] = direct_multiply(temporary[2], temporary[22]);
    output[5] = direct_multiply(temporary[17], temporary[10]);
    output[9] = direct_multiply(output[27], output[12]);
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), temporary[41]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[246].values, inputs[247].values, point), temporary[14]);
    output[18] = direct_add(temporary[32], output[22]);
    temporary[42] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[18]);
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[146].values, inputs[147].values, point), output[13]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[7]);
    output[18] = direct_add(temporary[32], output[22]);
    temporary[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[18]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), output[24]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point), temporary[19]);
    output[22] = direct_add(output[22], output[18]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[22]);
    output[22] = direct_add(direct_add(direct_add(direct_add(direct_add(output[8], output[5]), output[9]), temporary[42]), temporary[32]), output[18]);
    output[8] = direct_multiply(temporary[12], output[22]);
    output[5] = direct_add(direct_add(temporary[37], temporary[15]), output[8]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[258].values, inputs[259].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[154].values, inputs[155].values, point));
    temporary[42] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point));
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[158].values, inputs[159].values, point), temporary[42]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[162].values, inputs[163].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[82].values, inputs[83].values, point));
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    temporary[37] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), temporary[37]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[254].values, inputs[255].values, point));
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[262].values, inputs[263].values, point));
    temporary[48] = direct_add(temporary[48], output[7]);
    output[7] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[48]);
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[78].values, inputs[79].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[86].values, inputs[87].values, point));
    temporary[48] = direct_add(temporary[48], output[15]);
    output[15] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[48]);
    temporary[48] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[9], temporary[32]), output[18]), output[22]), temporary[15]), output[8]), output[6]), output[16]), output[7]), output[15]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[48]);
    output[8] = direct_multiply(temporary[11], temporary[22]);
    output[9] = direct_multiply(temporary[26], temporary[10]);
    temporary[32] = direct_multiply(temporary[30], output[12]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), temporary[41]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[248].values, inputs[249].values, point), temporary[14]);
    output[6] = direct_add(output[22], output[18]);
    output[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[6]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[148].values, inputs[149].values, point), output[13]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), temporary[7]);
    temporary[48] = direct_add(output[7], output[15]);
    output[22] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[48]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), output[24]);
    output[6] = direct_multiply(direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point), temporary[19]);
    output[7] = direct_add(output[18], output[6]);
    output[15] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[7]);
    temporary[48] = direct_add(direct_add(direct_add(direct_add(direct_add(output[8], output[9]), temporary[32]), output[16]), output[22]), output[15]);
    output[18] = direct_multiply(temporary[12], temporary[48]);
    output[6] = direct_add(temporary[15], output[18]);
    output[7] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[256].values, inputs[257].values, point));
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[260].values, inputs[261].values, point), temporary[27]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[262].values, inputs[263].values, point), temporary[29]);
    temporary[32] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[156].values, inputs[157].values, point), temporary[32]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[160].values, inputs[161].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[162].values, inputs[163].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[80].values, inputs[81].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[84].values, inputs[85].values, point), temporary[34]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[86].values, inputs[87].values, point), output[23]);
    temporary[51] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    temporary[52] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), temporary[51]);
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    temporary[15] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[7], output[8]), output[9]), output[22]), output[16]), output[15]), temporary[48]), temporary[15]), output[18]), temporary[52]), temporary[53]), output[10]);
    output[8] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[15]);
    output[9] = direct_multiply(temporary[28], temporary[22]);
    output[22] = direct_multiply(temporary[35], temporary[10]);
    output[18] = direct_multiply(temporary[40], output[12]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), temporary[41]);
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[250].values, inputs[251].values, point), temporary[14]);
    output[7] = direct_add(output[16], temporary[48]);
    output[15] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[7]);
    temporary[52] = direct_multiply(direct_complex_plane<Lane>(inputs[150].values, inputs[151].values, point), output[13]);
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), temporary[7]);
    output[10] = direct_add(temporary[52], temporary[53]);
    temporary[15] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[10]);
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), output[24]);
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point), temporary[19]);
    output[12] = direct_add(temporary[22], temporary[10]);
    temporary[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[12]);
    output[16] = direct_add(direct_add(direct_add(direct_add(direct_add(output[9], output[22]), output[18]), output[15]), temporary[15]), temporary[41]);
    temporary[48] = direct_multiply(temporary[12], output[16]);
    output[7] = direct_add(output[8], temporary[48]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[272].values, inputs[273].values, point));
    temporary[52] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[274].values, inputs[275].values, point));
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[276].values, inputs[277].values, point));
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[88].values, inputs[89].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[90].values, inputs[91].values, point));
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[92].values, inputs[93].values, point));
    output[12] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[7], temporary[52]), temporary[53]), output[10]), temporary[10]), temporary[22]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), output[12]);
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[172].values, inputs[173].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[174].values, inputs[175].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[176].values, inputs[177].values, point));
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    output[8] = direct_add(direct_add(direct_add(direct_add(direct_add(output[9], output[22]), temporary[41]), output[18]), output[15]), output[16]);
    temporary[48] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), output[8]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[264].values, inputs[265].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[266].values, inputs[267].values, point));
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[268].values, inputs[269].values, point));
    temporary[52] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[270].values, inputs[271].values, point));
    temporary[53] = direct_add(direct_add(temporary[10], temporary[22]), temporary[52]);
    output[10] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[53]);
    output[12] = direct_add(temporary[7], output[10]);
    output[9] = direct_multiply(temporary[5], output[12]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[164].values, inputs[165].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[166].values, inputs[167].values, point));
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[168].values, inputs[169].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[170].values, inputs[171].values, point));
    output[15] = direct_add(direct_add(temporary[41], output[18]), output[16]);
    output[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[15]);
    temporary[10] = direct_add(output[22], output[8]);
    temporary[22] = direct_multiply(temporary[8], temporary[10]);
    temporary[52] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    temporary[41] = direct_add(direct_add(temporary[53], temporary[7]), output[10]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[41]);
    output[16] = direct_add(temporary[52], output[18]);
    output[15] = direct_multiply(temporary[6], output[16]);
    output[8] = direct_multiply(direct_real_plane<Lane>(inputs[364].values, point), direct_complex_plane<Lane>(inputs[264].values, inputs[265].values, point));
    output[22] = direct_multiply(direct_real_plane<Lane>(inputs[365].values, point), direct_complex_plane<Lane>(inputs[266].values, inputs[267].values, point));
    temporary[7] = direct_multiply(direct_real_plane<Lane>(inputs[366].values, point), direct_complex_plane<Lane>(inputs[268].values, inputs[269].values, point));
    temporary[53] = direct_multiply(direct_real_plane<Lane>(inputs[367].values, point), direct_complex_plane<Lane>(inputs[270].values, inputs[271].values, point));
    output[10] = direct_add(direct_add(output[22], temporary[7]), temporary[53]);
    temporary[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[10]);
    output[18] = direct_add(output[8], temporary[41]);
    temporary[52] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[18]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[264].values, inputs[265].values, point), temporary[14]);
    output[22] = direct_add(temporary[52], temporary[7]);
    temporary[53] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[22]);
    output[10] = direct_multiply(direct_real_plane<Lane>(inputs[368].values, point), direct_complex_plane<Lane>(inputs[164].values, inputs[165].values, point));
    output[8] = direct_multiply(direct_real_plane<Lane>(inputs[369].values, point), direct_complex_plane<Lane>(inputs[166].values, inputs[167].values, point));
    temporary[41] = direct_multiply(direct_real_plane<Lane>(inputs[370].values, point), direct_complex_plane<Lane>(inputs[168].values, inputs[169].values, point));
    temporary[7] = direct_multiply(direct_real_plane<Lane>(inputs[371].values, point), direct_complex_plane<Lane>(inputs[170].values, inputs[171].values, point));
    temporary[52] = direct_add(direct_add(output[8], temporary[41]), temporary[7]);
    output[22] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[52]);
    temporary[7] = direct_add(output[10], output[22]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), temporary[7]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[164].values, inputs[165].values, point), temporary[16]);
    temporary[52] = direct_add(output[8], temporary[41]);
    output[22] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[52]);
    output[10] = direct_multiply(direct_real_plane<Lane>(inputs[372].values, point), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    output[8] = direct_multiply(direct_real_plane<Lane>(inputs[373].values, point), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    temporary[41] = direct_multiply(direct_real_plane<Lane>(inputs[374].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    temporary[52] = direct_multiply(direct_real_plane<Lane>(inputs[375].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    output[8] = direct_add(direct_add(output[8], temporary[41]), temporary[52]);
    temporary[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[8]);
    temporary[52] = direct_add(output[10], temporary[41]);
    output[8] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), temporary[52]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point), temporary[19]);
    output[10] = direct_add(output[8], temporary[41]);
    output[8] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[10]);
    temporary[41] = direct_add(direct_add(direct_add(direct_add(direct_add(output[9], temporary[22]), output[15]), temporary[53]), output[22]), output[8]);
    output[10] = direct_multiply(temporary[12], temporary[41]);
    output[8] = direct_add(direct_add(temporary[15], temporary[48]), output[10]);
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[272].values, inputs[273].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[278].values, inputs[279].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[280].values, inputs[281].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[88].values, inputs[89].values, point));
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[94].values, inputs[95].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[96].values, inputs[97].values, point));
    temporary[15] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[22], output[9]), output[22]), output[15]), temporary[53]), temporary[41]);
    temporary[48] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[15]);
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[172].values, inputs[173].values, point));
    temporary[22] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[178].values, inputs[179].values, point));
    output[9] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[180].values, inputs[181].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    temporary[53] = direct_add(direct_add(direct_add(direct_add(direct_add(output[10], temporary[22]), output[9]), output[22]), temporary[41]), output[15]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[53]);
    temporary[22] = direct_multiply(temporary[2], output[12]);
    output[9] = direct_multiply(temporary[17], temporary[10]);
    output[22] = direct_multiply(output[27], output[16]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[18]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[266].values, inputs[267].values, point), temporary[14]);
    output[10] = direct_add(temporary[41], output[15]);
    temporary[53] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[10]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[7]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[166].values, inputs[167].values, point), temporary[16]);
    output[10] = direct_add(temporary[41], output[15]);
    temporary[41] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[10]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), temporary[52]);
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point), temporary[19]);
    output[15] = direct_add(output[15], output[10]);
    output[10] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[15]);
    output[15] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[22], output[9]), output[22]), temporary[53]), temporary[41]), output[10]);
    temporary[22] = direct_multiply(temporary[12], output[15]);
    output[9] = direct_add(direct_add(temporary[48], temporary[15]), temporary[22]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[278].values, inputs[279].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[174].values, inputs[175].values, point));
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[178].values, inputs[179].values, point), temporary[24]);
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[182].values, inputs[183].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[94].values, inputs[95].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    temporary[22] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), temporary[22]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[274].values, inputs[275].values, point));
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[282].values, inputs[283].values, point));
    temporary[56] = direct_add(temporary[56], output[11]);
    output[11] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[56]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[90].values, inputs[91].values, point));
    temporary[58] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[98].values, inputs[99].values, point));
    temporary[56] = direct_add(temporary[56], temporary[58]);
    temporary[58] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[56]);
    temporary[56] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[22], temporary[41]), temporary[53]), output[10]), output[15]), temporary[15]), temporary[48]), output[14]), output[11]), temporary[58]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[56]);
    output[22] = direct_multiply(temporary[11], output[12]);
    temporary[41] = direct_multiply(temporary[26], temporary[10]);
    temporary[48] = direct_multiply(temporary[30], output[16]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[18]);
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[268].values, inputs[269].values, point), temporary[14]);
    output[10] = direct_add(output[15], temporary[53]);
    output[14] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[10]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), temporary[7]);
    temporary[58] = direct_multiply(direct_complex_plane<Lane>(inputs[168].values, inputs[169].values, point), temporary[16]);
    temporary[56] = direct_add(output[11], temporary[58]);
    output[15] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[56]);
    temporary[53] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), temporary[52]);
    output[10] = direct_multiply(direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point), temporary[19]);
    output[11] = direct_add(temporary[53], output[10]);
    temporary[58] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[11]);
    temporary[56] = direct_add(direct_add(direct_add(direct_add(direct_add(output[22], temporary[41]), temporary[48]), output[14]), output[15]), temporary[58]);
    temporary[53] = direct_multiply(temporary[12], temporary[56]);
    output[10] = direct_add(temporary[15], temporary[53]);
    output[11] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[276].values, inputs[277].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[280].values, inputs[281].values, point), temporary[27]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[282].values, inputs[283].values, point), temporary[29]);
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[176].values, inputs[177].values, point), temporary[1]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[180].values, inputs[181].values, point));
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[182].values, inputs[183].values, point));
    temporary[58] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[92].values, inputs[93].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[96].values, inputs[97].values, point), temporary[34]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[98].values, inputs[99].values, point), output[23]);
    temporary[53] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), temporary[53]);
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    temporary[15] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[11], output[22]), temporary[41]), temporary[48]), output[15]), output[14]), temporary[58]), temporary[56]), temporary[15]), output[17]), temporary[60]), output[19]);
    output[22] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[15]);
    temporary[41] = direct_multiply(temporary[28], output[12]);
    temporary[48] = direct_multiply(temporary[35], temporary[10]);
    output[15] = direct_multiply(temporary[40], output[16]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), output[18]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[270].values, inputs[271].values, point), temporary[14]);
    output[11] = direct_add(output[14], temporary[56]);
    temporary[58] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[11]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), temporary[7]);
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[170].values, inputs[171].values, point), temporary[16]);
    output[19] = direct_add(output[17], temporary[60]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[19]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), temporary[52]);
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point), temporary[19]);
    output[16] = direct_add(output[12], temporary[10]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[16]);
    output[14] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[41], temporary[48]), output[15]), temporary[58]), temporary[15]), output[18]);
    temporary[56] = direct_multiply(temporary[12], output[14]);
    output[11] = direct_add(output[22], temporary[56]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[292].values, inputs[293].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[294].values, inputs[295].values, point));
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[296].values, inputs[297].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[100].values, inputs[101].values, point));
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[102].values, inputs[103].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[104].values, inputs[105].values, point));
    temporary[15] = direct_add(direct_add(direct_add(direct_add(direct_add(output[17], temporary[60]), output[19]), temporary[10]), output[12]), output[16]);
    temporary[41] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[15]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[172].values, inputs[173].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[174].values, inputs[175].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[176].values, inputs[177].values, point));
    temporary[58] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[138].values, inputs[139].values, point));
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[140].values, inputs[141].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point), direct_complex_plane<Lane>(inputs[142].values, inputs[143].values, point));
    temporary[56] = direct_add(direct_add(direct_add(direct_add(direct_add(output[18], temporary[48]), output[15]), temporary[58]), output[14]), output[22]);
    temporary[10] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[56]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[284].values, inputs[285].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[286].values, inputs[287].values, point));
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[288].values, inputs[289].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[290].values, inputs[291].values, point));
    output[19] = direct_add(direct_add(output[16], output[17]), temporary[60]);
    temporary[15] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[19]);
    output[22] = direct_add(output[12], temporary[15]);
    output[18] = direct_multiply(temporary[5], output[22]);
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[164].values, inputs[165].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[166].values, inputs[167].values, point));
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[168].values, inputs[169].values, point));
    temporary[58] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[170].values, inputs[171].values, point));
    temporary[56] = direct_add(direct_add(output[15], output[14]), temporary[58]);
    output[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[56]);
    output[17] = direct_add(temporary[48], output[16]);
    temporary[60] = direct_multiply(temporary[8], output[17]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[136].values, inputs[137].values, point));
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[138].values, inputs[139].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[140].values, inputs[141].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[142].values, inputs[143].values, point));
    output[14] = direct_add(direct_add(output[12], temporary[15]), output[15]);
    temporary[58] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[14]);
    temporary[56] = direct_add(output[19], temporary[58]);
    output[16] = direct_multiply(temporary[6], temporary[56]);
    temporary[48] = direct_multiply(direct_real_plane<Lane>(inputs[364].values, point), direct_complex_plane<Lane>(inputs[284].values, inputs[285].values, point));
    output[12] = direct_multiply(direct_real_plane<Lane>(inputs[365].values, point), direct_complex_plane<Lane>(inputs[286].values, inputs[287].values, point));
    temporary[15] = direct_multiply(direct_real_plane<Lane>(inputs[366].values, point), direct_complex_plane<Lane>(inputs[288].values, inputs[289].values, point));
    output[15] = direct_multiply(direct_real_plane<Lane>(inputs[367].values, point), direct_complex_plane<Lane>(inputs[290].values, inputs[291].values, point));
    output[14] = direct_add(direct_add(output[12], temporary[15]), output[15]);
    temporary[58] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[14]);
    output[19] = direct_add(temporary[48], temporary[58]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), output[19]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[284].values, inputs[285].values, point), temporary[14]);
    output[15] = direct_add(output[12], temporary[15]);
    output[14] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[15]);
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), temporary[7]);
    temporary[58] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[13]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[164].values, inputs[165].values, point), temporary[58]);
    temporary[15] = direct_add(temporary[48], output[12]);
    output[15] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[15]);
    output[13] = direct_multiply(direct_real_plane<Lane>(inputs[372].values, point), direct_complex_plane<Lane>(inputs[136].values, inputs[137].values, point));
    output[12] = direct_multiply(direct_real_plane<Lane>(inputs[373].values, point), direct_complex_plane<Lane>(inputs[138].values, inputs[139].values, point));
    temporary[48] = direct_multiply(direct_real_plane<Lane>(inputs[374].values, point), direct_complex_plane<Lane>(inputs[140].values, inputs[141].values, point));
    temporary[15] = direct_multiply(direct_real_plane<Lane>(inputs[375].values, point), direct_complex_plane<Lane>(inputs[142].values, inputs[143].values, point));
    output[12] = direct_add(direct_add(output[12], temporary[48]), temporary[15]);
    temporary[15] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[12]);
    temporary[48] = direct_add(output[13], temporary[15]);
    output[12] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), temporary[48]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[136].values, inputs[137].values, point), temporary[19]);
    output[13] = direct_add(output[12], temporary[15]);
    output[12] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[13]);
    temporary[15] = direct_add(direct_add(direct_add(direct_add(direct_add(output[18], temporary[60]), output[16]), output[14]), output[15]), output[12]);
    output[13] = direct_multiply(temporary[12], temporary[15]);
    output[12] = direct_add(direct_add(temporary[41], temporary[10]), output[13]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[292].values, inputs[293].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[298].values, inputs[299].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[300].values, inputs[301].values, point));
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[100].values, inputs[101].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), direct_complex_plane<Lane>(inputs[106].values, inputs[107].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[108].values, inputs[109].values, point));
    temporary[10] = direct_add(direct_add(direct_add(direct_add(direct_add(output[18], output[16]), output[15]), output[14]), temporary[60]), temporary[15]);
    temporary[41] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[10]);
    output[13] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[172].values, inputs[173].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[178].values, inputs[179].values, point));
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[180].values, inputs[181].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[32].values, inputs[33].values, point), direct_complex_plane<Lane>(inputs[136].values, inputs[137].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point), direct_complex_plane<Lane>(inputs[140].values, inputs[141].values, point));
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[142].values, inputs[143].values, point));
    temporary[60] = direct_add(direct_add(direct_add(direct_add(direct_add(output[13], temporary[15]), output[18]), output[16]), output[15]), output[14]);
    temporary[10] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[60]);
    temporary[15] = direct_multiply(temporary[2], output[22]);
    output[13] = direct_multiply(temporary[17], output[17]);
    output[18] = direct_multiply(output[27], temporary[56]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), output[19]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[286].values, inputs[287].values, point), temporary[14]);
    output[14] = direct_add(output[16], output[15]);
    temporary[60] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[14]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[7]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[166].values, inputs[167].values, point), temporary[58]);
    output[14] = direct_add(output[16], output[15]);
    output[16] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[14]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), temporary[48]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[138].values, inputs[139].values, point), temporary[19]);
    output[15] = direct_add(output[15], output[14]);
    output[14] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[15]);
    output[15] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[15], output[13]), output[18]), temporary[60]), output[16]), output[14]);
    temporary[15] = direct_multiply(temporary[12], output[15]);
    output[13] = direct_add(direct_add(temporary[41], temporary[10]), temporary[15]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[298].values, inputs[299].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[174].values, inputs[175].values, point));
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[178].values, inputs[179].values, point), temporary[42]);
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[182].values, inputs[183].values, point));
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[26].values, inputs[27].values, point), direct_complex_plane<Lane>(inputs[106].values, inputs[107].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[34].values, inputs[35].values, point), direct_complex_plane<Lane>(inputs[136].values, inputs[137].values, point));
    temporary[15] = direct_multiply(direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[38].values, inputs[39].values, point)), direct_complex_plane<Lane>(inputs[138].values, inputs[139].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[142].values, inputs[143].values, point));
    temporary[62] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[294].values, inputs[295].values, point));
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[302].values, inputs[303].values, point));
    temporary[62] = direct_add(temporary[62], output[21]);
    output[21] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[62]);
    temporary[62] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[102].values, inputs[103].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), direct_complex_plane<Lane>(inputs[110].values, inputs[111].values, point));
    temporary[62] = direct_add(temporary[62], temporary[64]);
    temporary[64] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[62]);
    temporary[62] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[18], output[16]), output[14]), temporary[60]), output[15]), temporary[10]), temporary[15]), temporary[41]), output[21]), temporary[64]);
    temporary[10] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[62]);
    temporary[15] = direct_multiply(temporary[11], output[22]);
    temporary[41] = direct_multiply(temporary[26], output[17]);
    output[18] = direct_multiply(temporary[30], temporary[56]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), output[19]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[288].values, inputs[289].values, point), temporary[14]);
    output[14] = direct_add(output[16], output[15]);
    temporary[60] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[14]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), temporary[7]);
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[168].values, inputs[169].values, point), temporary[58]);
    temporary[62] = direct_add(output[21], temporary[64]);
    output[16] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[62]);
    output[15] = direct_multiply(direct_complex_plane<Lane>(inputs[28].values, inputs[29].values, point), temporary[48]);
    output[14] = direct_multiply(direct_complex_plane<Lane>(inputs[140].values, inputs[141].values, point), temporary[19]);
    output[21] = direct_add(output[15], output[14]);
    temporary[64] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[21]);
    temporary[62] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[15], temporary[41]), output[18]), temporary[60]), output[16]), temporary[64]);
    output[15] = direct_multiply(temporary[12], temporary[62]);
    output[14] = direct_add(temporary[10], output[15]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[296].values, inputs[297].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[300].values, inputs[301].values, point), temporary[27]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[302].values, inputs[303].values, point), temporary[29]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[176].values, inputs[177].values, point), temporary[32]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[180].values, inputs[181].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[182].values, inputs[183].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[24].values, inputs[25].values, point), direct_complex_plane<Lane>(inputs[104].values, inputs[105].values, point));
    temporary[62] = direct_multiply(direct_complex_plane<Lane>(inputs[108].values, inputs[109].values, point), temporary[34]);
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[110].values, inputs[111].values, point), output[23]);
    output[15] = direct_multiply(direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[36].values, inputs[37].values, point)), direct_complex_plane<Lane>(inputs[136].values, inputs[137].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[40].values, inputs[41].values, point), direct_complex_plane<Lane>(inputs[138].values, inputs[139].values, point));
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[42].values, inputs[43].values, point), direct_complex_plane<Lane>(inputs[140].values, inputs[141].values, point));
    temporary[10] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[21], temporary[15]), temporary[41]), output[18]), output[16]), temporary[60]), temporary[64]), temporary[62]), temporary[10]), output[15]), temporary[34]), output[23]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[10]);
    temporary[34] = direct_multiply(temporary[28], output[22]);
    output[23] = direct_multiply(temporary[35], output[17]);
    temporary[41] = direct_multiply(temporary[40], temporary[56]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), output[19]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[290].values, inputs[291].values, point), temporary[14]);
    output[15] = direct_add(output[18], output[16]);
    temporary[60] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[15]);
    temporary[62] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), temporary[7]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[170].values, inputs[171].values, point), temporary[58]);
    temporary[64] = direct_add(temporary[62], output[21]);
    temporary[10] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[64]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[30].values, inputs[31].values, point), temporary[48]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[142].values, inputs[143].values, point), temporary[19]);
    temporary[56] = direct_add(output[22], output[17]);
    output[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[56]);
    output[18] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[34], output[23]), temporary[41]), temporary[60]), temporary[10]), output[19]);
    output[16] = direct_multiply(temporary[12], output[18]);
    output[15] = direct_add(temporary[15], output[16]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[312].values, inputs[313].values, point));
    temporary[62] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[314].values, inputs[315].values, point));
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[316].values, inputs[317].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point));
    output[22] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[7], temporary[62]), output[21]), temporary[64]), temporary[48]), temporary[19]);
    output[17] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), output[22]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[192].values, inputs[193].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[194].values, inputs[195].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[196].values, inputs[197].values, point));
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    output[19] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[56], temporary[10]), temporary[34]), output[23]), temporary[41]), temporary[60]);
    output[18] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), output[19]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[304].values, inputs[305].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[306].values, inputs[307].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[308].values, inputs[309].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[310].values, inputs[311].values, point));
    temporary[48] = direct_add(direct_add(output[16], temporary[7]), temporary[19]);
    temporary[62] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[48]);
    output[21] = direct_add(temporary[15], temporary[62]);
    temporary[64] = direct_multiply(temporary[5], output[21]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[184].values, inputs[185].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[186].values, inputs[187].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[188].values, inputs[189].values, point));
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[190].values, inputs[191].values, point));
    temporary[41] = direct_add(direct_add(temporary[10], temporary[34]), output[23]);
    temporary[56] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[41]);
    temporary[60] = direct_add(output[22], temporary[56]);
    output[19] = direct_multiply(temporary[8], temporary[60]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    temporary[15] = direct_add(direct_add(temporary[19], output[16]), temporary[48]);
    temporary[62] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[15]);
    temporary[10] = direct_add(temporary[7], temporary[62]);
    temporary[34] = direct_multiply(temporary[6], temporary[10]);
    output[23] = direct_multiply(direct_real_plane<Lane>(inputs[364].values, point), direct_complex_plane<Lane>(inputs[304].values, inputs[305].values, point));
    temporary[41] = direct_multiply(direct_real_plane<Lane>(inputs[365].values, point), direct_complex_plane<Lane>(inputs[306].values, inputs[307].values, point));
    output[22] = direct_multiply(direct_real_plane<Lane>(inputs[366].values, point), direct_complex_plane<Lane>(inputs[308].values, inputs[309].values, point));
    temporary[56] = direct_multiply(direct_real_plane<Lane>(inputs[367].values, point), direct_complex_plane<Lane>(inputs[310].values, inputs[311].values, point));
    temporary[19] = direct_add(direct_add(temporary[41], output[22]), temporary[56]);
    output[16] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[19]);
    temporary[48] = direct_add(output[23], output[16]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[48]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[304].values, inputs[305].values, point), temporary[14]);
    temporary[62] = direct_add(temporary[15], temporary[7]);
    output[22] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[62]);
    temporary[41] = direct_multiply(direct_real_plane<Lane>(inputs[368].values, point), direct_complex_plane<Lane>(inputs[184].values, inputs[185].values, point));
    temporary[56] = direct_multiply(direct_real_plane<Lane>(inputs[369].values, point), direct_complex_plane<Lane>(inputs[186].values, inputs[187].values, point));
    temporary[19] = direct_multiply(direct_real_plane<Lane>(inputs[370].values, point), direct_complex_plane<Lane>(inputs[188].values, inputs[189].values, point));
    output[23] = direct_multiply(direct_real_plane<Lane>(inputs[371].values, point), direct_complex_plane<Lane>(inputs[190].values, inputs[191].values, point));
    output[16] = direct_add(direct_add(temporary[56], temporary[19]), output[23]);
    temporary[7] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[16]);
    temporary[15] = direct_add(temporary[41], temporary[7]);
    temporary[62] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), temporary[15]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[184].values, inputs[185].values, point), temporary[16]);
    output[23] = direct_add(temporary[62], temporary[19]);
    temporary[56] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[23]);
    output[16] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), output[25]);
    temporary[7] = direct_multiply(direct_real_plane<Lane>(inputs[376].values, point), direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point));
    temporary[41] = direct_multiply(direct_real_plane<Lane>(inputs[377].values, point), direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point));
    temporary[19] = direct_multiply(direct_real_plane<Lane>(inputs[378].values, point), direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point));
    temporary[62] = direct_multiply(direct_real_plane<Lane>(inputs[379].values, point), direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point));
    output[23] = direct_add(direct_add(temporary[41], temporary[19]), temporary[62]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[23]);
    temporary[41] = direct_add(temporary[7], temporary[19]);
    temporary[62] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[41]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point), temporary[62]);
    temporary[7] = direct_add(output[16], output[23]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[7]);
    temporary[41] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[64], output[19]), temporary[34]), output[22]), temporary[56]), temporary[19]);
    output[23] = direct_multiply(temporary[12], temporary[41]);
    output[16] = direct_add(direct_add(output[17], output[18]), output[23]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[312].values, inputs[313].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[318].values, inputs[319].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[320].values, inputs[321].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[64].values, inputs[65].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[70].values, inputs[71].values, point));
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[72].values, inputs[73].values, point));
    temporary[64] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[7], temporary[19]), output[22]), temporary[34]), temporary[56]), output[19]);
    temporary[41] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[64]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[192].values, inputs[193].values, point));
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[198].values, inputs[199].values, point));
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[200].values, inputs[201].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    temporary[34] = direct_add(direct_add(direct_add(direct_add(direct_add(output[23], output[18]), output[17]), temporary[7]), temporary[19]), output[22]);
    temporary[56] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[34]);
    output[19] = direct_multiply(temporary[2], output[21]);
    temporary[64] = direct_multiply(temporary[17], temporary[60]);
    temporary[7] = direct_multiply(output[27], temporary[10]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), temporary[48]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[306].values, inputs[307].values, point), temporary[14]);
    output[23] = direct_add(temporary[19], output[22]);
    output[18] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[23]);
    output[17] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[15]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[186].values, inputs[187].values, point), temporary[16]);
    temporary[19] = direct_add(output[17], temporary[34]);
    output[22] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[19]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), output[25]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point), temporary[62]);
    output[17] = direct_add(output[23], temporary[34]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[17]);
    temporary[34] = direct_add(direct_add(direct_add(direct_add(direct_add(output[19], temporary[64]), temporary[7]), output[18]), output[22]), temporary[19]);
    output[23] = direct_multiply(temporary[12], temporary[34]);
    output[17] = direct_add(direct_add(temporary[41], temporary[56]), output[23]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[318].values, inputs[319].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[194].values, inputs[195].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[198].values, inputs[199].values, point), temporary[24]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[202].values, inputs[203].values, point));
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[70].values, inputs[71].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point), direct_complex_plane<Lane>(inputs[112].values, inputs[113].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), temporary[23]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point), direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point));
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[314].values, inputs[315].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[322].values, inputs[323].values, point));
    temporary[23] = direct_add(temporary[41], temporary[56]);
    temporary[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[23]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[66].values, inputs[67].values, point));
    temporary[23] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[74].values, inputs[75].values, point));
    temporary[23] = direct_add(temporary[56], temporary[23]);
    temporary[56] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[23]);
    temporary[23] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(temporary[7], temporary[19]), output[22]), output[18]), output[19]), temporary[64]), temporary[34]), output[23]), temporary[41]), temporary[56]);
    temporary[7] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[23]);
    temporary[19] = direct_multiply(temporary[11], output[21]);
    output[22] = direct_multiply(temporary[26], temporary[60]);
    temporary[34] = direct_multiply(temporary[30], temporary[10]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), temporary[48]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[308].values, inputs[309].values, point), temporary[14]);
    output[18] = direct_add(output[23], temporary[41]);
    temporary[56] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[18]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), temporary[15]);
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[188].values, inputs[189].values, point), temporary[16]);
    temporary[23] = direct_add(output[19], temporary[64]);
    output[23] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[23]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), output[25]);
    output[18] = direct_multiply(direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point), temporary[62]);
    output[19] = direct_add(temporary[41], output[18]);
    temporary[64] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[19]);
    temporary[23] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[19], output[22]), temporary[34]), temporary[56]), output[23]), temporary[64]);
    temporary[41] = direct_multiply(temporary[12], temporary[23]);
    output[18] = direct_add(temporary[7], temporary[41]);
    output[19] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[316].values, inputs[317].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[320].values, inputs[321].values, point), temporary[27]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[322].values, inputs[323].values, point), temporary[29]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[196].values, inputs[197].values, point), temporary[1]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[200].values, inputs[201].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[202].values, inputs[203].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[68].values, inputs[69].values, point));
    temporary[23] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[72].values, inputs[73].values, point), temporary[23]);
    temporary[41] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point));
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[74].values, inputs[75].values, point), temporary[41]);
    temporary[38] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), temporary[38]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), direct_complex_plane<Lane>(inputs[114].values, inputs[115].values, point));
    temporary[67] = direct_multiply(direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point), direct_complex_plane<Lane>(inputs[116].values, inputs[117].values, point));
    temporary[7] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(output[19], temporary[19]), output[22]), temporary[34]), output[23]), temporary[56]), temporary[64]), temporary[7]), output[20]), temporary[38]), output[26]), temporary[67]);
    temporary[19] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[7]);
    output[22] = direct_multiply(temporary[28], output[21]);
    temporary[34] = direct_multiply(temporary[35], temporary[60]);
    output[23] = direct_multiply(temporary[40], temporary[10]);
    temporary[38] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), temporary[48]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[310].values, inputs[311].values, point), temporary[14]);
    output[19] = direct_add(temporary[38], temporary[56]);
    temporary[64] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[19]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), temporary[15]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[190].values, inputs[191].values, point), temporary[16]);
    temporary[67] = direct_add(output[20], output[26]);
    temporary[7] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[67]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), output[25]);
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[118].values, inputs[119].values, point), temporary[62]);
    temporary[10] = direct_add(output[21], temporary[60]);
    temporary[48] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[10]);
    temporary[38] = direct_add(direct_add(direct_add(direct_add(direct_add(output[22], temporary[34]), output[23]), temporary[64]), temporary[7]), temporary[48]);
    temporary[56] = direct_multiply(temporary[12], temporary[38]);
    output[19] = direct_add(temporary[19], temporary[56]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[332].values, inputs[333].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[334].values, inputs[335].values, point));
    temporary[67] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[336].values, inputs[337].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[76].values, inputs[77].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[78].values, inputs[79].values, point));
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[80].values, inputs[81].values, point));
    temporary[10] = direct_add(direct_add(direct_add(direct_add(direct_add(output[20], output[26]), temporary[67]), output[25]), temporary[60]), output[21]);
    temporary[7] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[10]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[192].values, inputs[193].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[194].values, inputs[195].values, point));
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[196].values, inputs[197].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    temporary[38] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    temporary[19] = direct_add(direct_add(direct_add(direct_add(direct_add(output[22], temporary[34]), output[23]), temporary[48]), temporary[64]), temporary[38]);
    temporary[56] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[19]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[324].values, inputs[325].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[326].values, inputs[327].values, point));
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[328].values, inputs[329].values, point));
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[330].values, inputs[331].values, point));
    output[26] = direct_add(direct_add(temporary[60], output[21]), output[20]);
    temporary[67] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[26]);
    temporary[10] = direct_add(output[25], temporary[67]);
    output[22] = direct_multiply(temporary[5], temporary[10]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[184].values, inputs[185].values, point));
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[186].values, inputs[187].values, point));
    temporary[38] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[188].values, inputs[189].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[190].values, inputs[191].values, point));
    temporary[64] = direct_add(direct_add(output[23], temporary[38]), temporary[48]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[64]);
    temporary[60] = direct_add(temporary[34], temporary[19]);
    output[21] = direct_multiply(temporary[8], temporary[60]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    temporary[67] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    output[23] = direct_add(direct_add(output[26], output[25]), temporary[67]);
    temporary[38] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), output[23]);
    temporary[48] = direct_add(output[20], temporary[38]);
    temporary[64] = direct_multiply(temporary[6], temporary[48]);
    temporary[19] = direct_multiply(direct_real_plane<Lane>(inputs[364].values, point), direct_complex_plane<Lane>(inputs[324].values, inputs[325].values, point));
    temporary[34] = direct_multiply(direct_real_plane<Lane>(inputs[365].values, point), direct_complex_plane<Lane>(inputs[326].values, inputs[327].values, point));
    output[25] = direct_multiply(direct_real_plane<Lane>(inputs[366].values, point), direct_complex_plane<Lane>(inputs[328].values, inputs[329].values, point));
    output[26] = direct_multiply(direct_real_plane<Lane>(inputs[367].values, point), direct_complex_plane<Lane>(inputs[330].values, inputs[331].values, point));
    temporary[67] = direct_add(direct_add(temporary[34], output[25]), output[26]);
    output[23] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[67]);
    temporary[38] = direct_add(temporary[19], output[23]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[38]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[324].values, inputs[325].values, point), temporary[14]);
    temporary[34] = direct_add(output[20], output[25]);
    output[26] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[34]);
    temporary[67] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), temporary[15]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[184].values, inputs[185].values, point), temporary[58]);
    output[23] = direct_add(temporary[67], temporary[19]);
    output[25] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[23]);
    output[20] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), output[24]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point), temporary[62]);
    temporary[19] = direct_add(output[20], temporary[34]);
    temporary[67] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[19]);
    output[23] = direct_add(direct_add(direct_add(direct_add(direct_add(output[22], output[21]), temporary[64]), output[26]), output[25]), temporary[67]);
    temporary[34] = direct_multiply(temporary[12], output[23]);
    output[20] = direct_add(direct_add(temporary[7], temporary[56]), temporary[34]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[332].values, inputs[333].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[338].values, inputs[339].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[340].values, inputs[341].values, point));
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[76].values, inputs[77].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[82].values, inputs[83].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[84].values, inputs[85].values, point));
    temporary[67] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[19], output[25]), output[22]), output[21]), temporary[64]), output[26]);
    output[23] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[67]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[192].values, inputs[193].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[198].values, inputs[199].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[200].values, inputs[201].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    output[21] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[7], temporary[34]), temporary[56]), temporary[19]), output[25]), output[22]);
    temporary[64] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), output[21]);
    output[26] = direct_multiply(temporary[2], temporary[10]);
    temporary[67] = direct_multiply(temporary[17], temporary[60]);
    temporary[7] = direct_multiply(output[27], temporary[48]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), temporary[38]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[326].values, inputs[327].values, point), temporary[14]);
    output[22] = direct_add(temporary[19], output[25]);
    temporary[34] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[22]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), temporary[15]);
    output[21] = direct_multiply(direct_complex_plane<Lane>(inputs[186].values, inputs[187].values, point), temporary[58]);
    temporary[19] = direct_add(temporary[56], output[21]);
    output[25] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[19]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), output[24]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point), temporary[62]);
    output[21] = direct_add(output[22], temporary[56]);
    temporary[19] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[21]);
    output[22] = direct_add(direct_add(direct_add(direct_add(direct_add(output[26], temporary[67]), temporary[7]), temporary[34]), output[25]), temporary[19]);
    temporary[56] = direct_multiply(temporary[12], output[22]);
    output[21] = direct_add(direct_add(output[23], temporary[64]), temporary[56]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[338].values, inputs[339].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[16].values, inputs[17].values, point), direct_complex_plane<Lane>(inputs[194].values, inputs[195].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[198].values, inputs[199].values, point), temporary[42]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), direct_complex_plane<Lane>(inputs[202].values, inputs[203].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[82].values, inputs[83].values, point));
    temporary[67] = direct_multiply(direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point), direct_complex_plane<Lane>(inputs[128].values, inputs[129].values, point));
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), temporary[37]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point), direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[334].values, inputs[335].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[342].values, inputs[343].values, point));
    temporary[42] = direct_add(temporary[56], temporary[64]);
    temporary[37] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[42]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[78].values, inputs[79].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[86].values, inputs[87].values, point));
    temporary[42] = direct_add(temporary[56], temporary[64]);
    temporary[56] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[42]);
    temporary[64] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(temporary[7], temporary[19]), output[25]), temporary[34]), output[26]), temporary[67]), output[22]), output[23]), temporary[37]), temporary[56]);
    temporary[42] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[64]);
    temporary[7] = direct_multiply(temporary[11], temporary[10]);
    temporary[19] = direct_multiply(temporary[26], temporary[60]);
    output[25] = direct_multiply(temporary[30], temporary[48]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), temporary[38]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[328].values, inputs[329].values, point), temporary[14]);
    output[23] = direct_add(output[22], temporary[34]);
    temporary[37] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[23]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), temporary[15]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[188].values, inputs[189].values, point), temporary[58]);
    temporary[67] = direct_add(temporary[56], output[26]);
    temporary[64] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[67]);
    output[22] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), output[24]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point), temporary[62]);
    output[23] = direct_add(output[22], temporary[34]);
    temporary[56] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[23]);
    output[26] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[7], temporary[19]), output[25]), temporary[37]), temporary[64]), temporary[56]);
    temporary[67] = direct_multiply(temporary[12], output[26]);
    output[22] = direct_add(temporary[42], temporary[67]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[336].values, inputs[337].values, point));
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[340].values, inputs[341].values, point), temporary[27]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[342].values, inputs[343].values, point), temporary[29]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[196].values, inputs[197].values, point), temporary[32]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[18].values, inputs[19].values, point), direct_complex_plane<Lane>(inputs[200].values, inputs[201].values, point));
    temporary[37] = direct_multiply(direct_complex_plane<Lane>(inputs[20].values, inputs[21].values, point), direct_complex_plane<Lane>(inputs[202].values, inputs[203].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[80].values, inputs[81].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[84].values, inputs[85].values, point), temporary[23]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[86].values, inputs[87].values, point), temporary[41]);
    temporary[42] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), temporary[51]);
    temporary[67] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), direct_complex_plane<Lane>(inputs[130].values, inputs[131].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point), direct_complex_plane<Lane>(inputs[132].values, inputs[133].values, point));
    temporary[51] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(temporary[34], output[23]), temporary[7]), temporary[19]), output[25]), temporary[37]), temporary[56]), temporary[64]), output[26]), temporary[42]), temporary[67]), temporary[32]);
    temporary[7] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[51]);
    temporary[19] = direct_multiply(temporary[28], temporary[10]);
    output[25] = direct_multiply(temporary[35], temporary[60]);
    temporary[32] = direct_multiply(temporary[40], temporary[48]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), temporary[38]);
    output[23] = direct_multiply(direct_complex_plane<Lane>(inputs[330].values, inputs[331].values, point), temporary[14]);
    temporary[37] = direct_add(temporary[34], output[23]);
    temporary[42] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[37]);
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[22].values, inputs[23].values, point), temporary[15]);
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[190].values, inputs[191].values, point), temporary[58]);
    output[26] = direct_add(temporary[56], temporary[64]);
    temporary[67] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[26]);
    temporary[51] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), output[24]);
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[134].values, inputs[135].values, point), temporary[62]);
    temporary[60] = direct_add(temporary[51], temporary[10]);
    temporary[48] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[60]);
    temporary[38] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[19], output[25]), temporary[32]), temporary[42]), temporary[67]), temporary[48]);
    temporary[34] = direct_multiply(temporary[12], temporary[38]);
    output[23] = direct_add(temporary[7], temporary[34]);
    temporary[37] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[352].values, inputs[353].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[354].values, inputs[355].values, point));
    temporary[58] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[356].values, inputs[357].values, point));
    temporary[56] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[88].values, inputs[89].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[90].values, inputs[91].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[92].values, inputs[93].values, point));
    output[24] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[37], temporary[15]), temporary[58]), temporary[56]), temporary[64]), output[26]);
    temporary[10] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), output[24]);
    temporary[51] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[212].values, inputs[213].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[214].values, inputs[215].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[216].values, inputs[217].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    temporary[42] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    temporary[48] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[51], temporary[60]), temporary[19]), output[25]), temporary[32]), temporary[42]);
    temporary[67] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[48]);
    temporary[38] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[344].values, inputs[345].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[346].values, inputs[347].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[348].values, inputs[349].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[350].values, inputs[351].values, point));
    temporary[37] = direct_add(direct_add(temporary[7], temporary[34]), temporary[15]);
    temporary[56] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[37]);
    temporary[58] = direct_add(temporary[38], temporary[56]);
    temporary[64] = direct_multiply(temporary[5], temporary[58]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[204].values, inputs[205].values, point));
    output[24] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[206].values, inputs[207].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[208].values, inputs[209].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[210].values, inputs[211].values, point));
    temporary[32] = direct_add(direct_add(output[24], temporary[19]), output[25]);
    temporary[42] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[32]);
    temporary[51] = direct_add(output[26], temporary[42]);
    temporary[60] = direct_multiply(temporary[8], temporary[51]);
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    temporary[37] = direct_add(direct_add(temporary[7], temporary[15]), temporary[34]);
    temporary[38] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[37]);
    temporary[56] = direct_add(temporary[48], temporary[38]);
    temporary[5] = direct_multiply(temporary[6], temporary[56]);
    temporary[19] = direct_multiply(direct_real_plane<Lane>(inputs[364].values, point), direct_complex_plane<Lane>(inputs[344].values, inputs[345].values, point));
    output[25] = direct_multiply(direct_real_plane<Lane>(inputs[365].values, point), direct_complex_plane<Lane>(inputs[346].values, inputs[347].values, point));
    output[24] = direct_multiply(direct_real_plane<Lane>(inputs[366].values, point), direct_complex_plane<Lane>(inputs[348].values, inputs[349].values, point));
    temporary[32] = direct_multiply(direct_real_plane<Lane>(inputs[367].values, point), direct_complex_plane<Lane>(inputs[350].values, inputs[351].values, point));
    temporary[42] = direct_add(direct_add(output[25], output[24]), temporary[32]);
    output[26] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[42]);
    temporary[8] = direct_add(temporary[19], output[26]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), temporary[8]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[344].values, inputs[345].values, point), temporary[14]);
    temporary[34] = direct_add(temporary[7], temporary[15]);
    temporary[37] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[34]);
    temporary[38] = direct_multiply(direct_real_plane<Lane>(inputs[368].values, point), direct_complex_plane<Lane>(inputs[204].values, inputs[205].values, point));
    temporary[48] = direct_multiply(direct_real_plane<Lane>(inputs[369].values, point), direct_complex_plane<Lane>(inputs[206].values, inputs[207].values, point));
    temporary[6] = direct_multiply(direct_real_plane<Lane>(inputs[370].values, point), direct_complex_plane<Lane>(inputs[208].values, inputs[209].values, point));
    output[25] = direct_multiply(direct_real_plane<Lane>(inputs[371].values, point), direct_complex_plane<Lane>(inputs[210].values, inputs[211].values, point));
    temporary[32] = direct_add(direct_add(temporary[48], temporary[6]), output[25]);
    output[24] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[32]);
    temporary[42] = direct_add(temporary[38], output[24]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), temporary[42]);
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[204].values, inputs[205].values, point), temporary[16]);
    temporary[7] = direct_add(temporary[19], output[26]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[7]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), temporary[52]);
    temporary[6] = direct_multiply(direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point), temporary[62]);
    output[25] = direct_add(temporary[34], temporary[6]);
    temporary[48] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[25]);
    temporary[32] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[64], temporary[60]), temporary[5]), temporary[37]), temporary[15]), temporary[48]);
    temporary[38] = direct_multiply(temporary[12], temporary[32]);
    output[24] = direct_add(direct_add(temporary[10], temporary[67]), temporary[38]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[352].values, inputs[353].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), direct_complex_plane<Lane>(inputs[358].values, inputs[359].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[360].values, inputs[361].values, point));
    temporary[6] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[88].values, inputs[89].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), direct_complex_plane<Lane>(inputs[94].values, inputs[95].values, point));
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[96].values, inputs[97].values, point));
    temporary[5] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[19], output[26]), temporary[7]), temporary[6]), temporary[34]), output[25]);
    temporary[15] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[5]);
    temporary[37] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[212].values, inputs[213].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[218].values, inputs[219].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[220].values, inputs[221].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[52].values, inputs[53].values, point), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    temporary[38] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[37], temporary[48]), temporary[60]), temporary[64]), temporary[32]), temporary[10]);
    temporary[67] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[38]);
    temporary[6] = direct_multiply(temporary[2], temporary[58]);
    temporary[7] = direct_multiply(temporary[17], temporary[51]);
    temporary[19] = direct_multiply(output[27], temporary[56]);
    output[25] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), temporary[8]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[346].values, inputs[347].values, point), temporary[14]);
    output[26] = direct_add(output[25], temporary[34]);
    temporary[5] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), output[26]);
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), temporary[42]);
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[206].values, inputs[207].values, point), temporary[16]);
    temporary[37] = direct_add(temporary[10], temporary[32]);
    temporary[48] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[37]);
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), temporary[52]);
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point), temporary[62]);
    temporary[38] = direct_add(temporary[60], temporary[64]);
    temporary[2] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[38]);
    temporary[17] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[6], temporary[7]), temporary[19]), temporary[5]), temporary[48]), temporary[2]);
    output[27] = direct_multiply(temporary[12], temporary[17]);
    output[25] = direct_add(direct_add(temporary[15], temporary[67]), output[27]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[2].values, inputs[3].values, point), direct_complex_plane<Lane>(inputs[358].values, inputs[359].values, point));
    output[26] = direct_multiply(direct_complex_plane<Lane>(inputs[8].values, inputs[9].values, point), direct_complex_plane<Lane>(inputs[214].values, inputs[215].values, point));
    temporary[10] = direct_multiply(direct_complex_plane<Lane>(inputs[218].values, inputs[219].values, point), temporary[24]);
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), direct_complex_plane<Lane>(inputs[222].values, inputs[223].values, point));
    temporary[37] = direct_multiply(direct_complex_plane<Lane>(inputs[46].values, inputs[47].values, point), direct_complex_plane<Lane>(inputs[94].values, inputs[95].values, point));
    temporary[60] = direct_multiply(direct_complex_plane<Lane>(inputs[54].values, inputs[55].values, point), direct_complex_plane<Lane>(inputs[120].values, inputs[121].values, point));
    temporary[64] = direct_multiply(direct_complex_plane<Lane>(inputs[58].values, inputs[59].values, point), temporary[22]);
    temporary[38] = direct_multiply(direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point), direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point));
    temporary[2] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[354].values, inputs[355].values, point));
    temporary[5] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), direct_complex_plane<Lane>(inputs[362].values, inputs[363].values, point));
    temporary[6] = direct_add(temporary[2], temporary[5]);
    temporary[7] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[6]);
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[90].values, inputs[91].values, point));
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), direct_complex_plane<Lane>(inputs[98].values, inputs[99].values, point));
    temporary[17] = direct_add(temporary[19], temporary[48]);
    output[27] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+0, 0x0.0p+0), temporary[17]);
    temporary[15] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(temporary[34], output[26]), temporary[10]), temporary[32]), temporary[37]), temporary[60]), temporary[64]), temporary[38]), temporary[7]), output[27]);
    temporary[67] = direct_multiply(direct_constant<Lane>(0x0.0p+0, 0x1.0000000000000p-1), temporary[15]);
    temporary[24] = direct_multiply(temporary[11], temporary[58]);
    temporary[22] = direct_multiply(temporary[26], temporary[51]);
    temporary[2] = direct_multiply(temporary[30], temporary[56]);
    temporary[5] = direct_multiply(direct_complex_plane<Lane>(inputs[4].values, inputs[5].values, point), temporary[8]);
    temporary[6] = direct_multiply(direct_complex_plane<Lane>(inputs[348].values, inputs[349].values, point), temporary[14]);
    temporary[19] = direct_add(temporary[5], temporary[6]);
    temporary[48] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[19]);
    temporary[17] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), temporary[42]);
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[208].values, inputs[209].values, point), temporary[16]);
    output[27] = direct_add(temporary[17], temporary[7]);
    temporary[10] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), output[27]);
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[48].values, inputs[49].values, point), temporary[52]);
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point), temporary[62]);
    temporary[37] = direct_add(temporary[32], temporary[34]);
    temporary[38] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[37]);
    temporary[60] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[24], temporary[22]), temporary[2]), temporary[48]), temporary[10]), temporary[38]);
    temporary[64] = direct_multiply(temporary[12], temporary[60]);
    output[26] = direct_add(temporary[67], temporary[64]);
    temporary[15] = direct_multiply(direct_complex_plane<Lane>(inputs[0].values, inputs[1].values, point), direct_complex_plane<Lane>(inputs[356].values, inputs[357].values, point));
    temporary[11] = direct_multiply(direct_complex_plane<Lane>(inputs[360].values, inputs[361].values, point), temporary[27]);
    temporary[26] = direct_multiply(direct_complex_plane<Lane>(inputs[362].values, inputs[363].values, point), temporary[29]);
    temporary[30] = direct_multiply(direct_complex_plane<Lane>(inputs[216].values, inputs[217].values, point), temporary[1]);
    temporary[5] = direct_multiply(direct_complex_plane<Lane>(inputs[10].values, inputs[11].values, point), direct_complex_plane<Lane>(inputs[220].values, inputs[221].values, point));
    temporary[6] = direct_multiply(direct_complex_plane<Lane>(inputs[12].values, inputs[13].values, point), direct_complex_plane<Lane>(inputs[222].values, inputs[223].values, point));
    temporary[19] = direct_multiply(direct_complex_plane<Lane>(inputs[44].values, inputs[45].values, point), direct_complex_plane<Lane>(inputs[92].values, inputs[93].values, point));
    temporary[7] = direct_multiply(direct_complex_plane<Lane>(inputs[96].values, inputs[97].values, point), temporary[23]);
    temporary[17] = direct_multiply(direct_complex_plane<Lane>(inputs[98].values, inputs[99].values, point), temporary[41]);
    output[27] = direct_multiply(direct_complex_plane<Lane>(inputs[56].values, inputs[57].values, point), temporary[53]);
    temporary[32] = direct_multiply(direct_complex_plane<Lane>(inputs[60].values, inputs[61].values, point), direct_complex_plane<Lane>(inputs[122].values, inputs[123].values, point));
    temporary[34] = direct_multiply(direct_complex_plane<Lane>(inputs[62].values, inputs[63].values, point), direct_complex_plane<Lane>(inputs[124].values, inputs[125].values, point));
    temporary[37] = direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(direct_add(temporary[15], temporary[11]), temporary[26]), temporary[30]), temporary[5]), temporary[6]), temporary[19]), temporary[7]), temporary[17]), output[27]), temporary[32]), temporary[34]);
    temporary[2] = direct_multiply(direct_constant<Lane>(0x0.0p+0, -0x1.0000000000000p-1), temporary[37]);
    temporary[10] = direct_multiply(temporary[28], temporary[58]);
    temporary[22] = direct_multiply(temporary[35], temporary[51]);
    temporary[24] = direct_multiply(temporary[40], temporary[56]);
    temporary[38] = direct_multiply(direct_complex_plane<Lane>(inputs[6].values, inputs[7].values, point), temporary[8]);
    temporary[48] = direct_multiply(direct_complex_plane<Lane>(inputs[350].values, inputs[351].values, point), temporary[14]);
    temporary[60] = direct_add(temporary[38], temporary[48]);
    temporary[64] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[60]);
    temporary[67] = direct_multiply(direct_complex_plane<Lane>(inputs[14].values, inputs[15].values, point), temporary[42]);
    temporary[27] = direct_multiply(direct_complex_plane<Lane>(inputs[210].values, inputs[211].values, point), temporary[16]);
    temporary[29] = direct_add(temporary[67], temporary[27]);
    temporary[1] = direct_multiply(direct_constant<Lane>(0x1.0000000000000p+1, 0x0.0p+0), temporary[29]);
    temporary[23] = direct_multiply(direct_complex_plane<Lane>(inputs[50].values, inputs[51].values, point), temporary[52]);
    temporary[41] = direct_multiply(direct_complex_plane<Lane>(inputs[126].values, inputs[127].values, point), temporary[62]);
    temporary[53] = direct_add(temporary[23], temporary[41]);
    temporary[5] = direct_multiply(direct_constant<Lane>(-0x1.0000000000000p+1, 0x0.0p+0), temporary[53]);
    temporary[6] = direct_add(direct_add(direct_add(direct_add(direct_add(temporary[10], temporary[22]), temporary[24]), temporary[64]), temporary[1]), temporary[5]);
    temporary[7] = direct_multiply(temporary[12], temporary[6]);
    output[27] = direct_add(temporary[2], temporary[7]);
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
}

}  // namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae

extern "C" pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae::NativeCompiledDirectMetadataV1
pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae_direct_application_v1_metadata() noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae;
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
pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae_direct_application_v1_target_triple() noexcept {
    return "x86_64-unknown-linux-gnu";
}

extern "C" const char*
pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae_direct_application_v1_cpu_features() noexcept {
    return "";
}

extern "C" const char*
pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae_direct_application_v1_source_sha256() noexcept {
    return "8f2759eafaa375db0298c37290f80c91ebff127dd4ed2e22039f789b6d9a51bf";
}

extern "C" std::uint32_t
pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae_direct_application_v1_logical_stack_bytes() noexcept {
    return pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae::kLogicalStackBytes;
}

extern "C" int pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae_direct_application_v1(
    const pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae::
        NativeCompiledDirectInputPlaneV1* inputs,
    std::uint32_t input_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae::
        NativeCompiledDirectScalarV1* scalars,
    std::uint32_t scalar_count,
    const pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae::
        NativeCompiledDirectOutputPlaneV1* outputs,
    std::uint32_t output_count,
    std::uint32_t point_start,
    std::uint32_t point_count
) noexcept {
    using namespace pyamplicol_native_direct_pyamplicol_generic_stage_3_subset_4_chunk_0_50b44b8733d243b989bf47401e7165ae;
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
