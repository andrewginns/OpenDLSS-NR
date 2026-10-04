// Exact captured-browser F13 GEMM prototype: integer E4 mantissas and shifts.
// Isolated from runtime policies. Canonical Ada half-group rounding is absent.
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include "browser_numerics.cuh"

#define NR_F13_INT_HD __host__ __device__ __forceinline__
namespace nr {
namespace f13_integer_detail {
using H = __half;

// Finite E4 operands decoded to half. Low byte: signed mantissa -15..15;
// high byte: floor exponent + 6, in 0..14. x = mantissa * 2^(high-9).
// Zero loses its sign because an integer FDPA sum publishes +0 either way.
NR_F13_INT_HD uint32_t pack_operand(H value) {
    const unsigned bits = halfBits(value), magnitude = bits & 0x7fffu;
    if (!magnitude) return 0;
    const int actual_exponent = int(magnitude >> 10) - 15;
    const int exponent = actual_exponent < -6 ? -6 : actual_exponent;
    const int mantissa = int((1024u + (magnitude & 1023u)) >> (7 + exponent - actual_exponent));
    const int signed_mantissa = bits & 0x8000u ? -mantissa : mantissa;
    return uint32_t(signed_mantissa & 255) | (uint32_t(exponent + 6) << 8);
}
NR_F13_INT_HD int mantissa(uint32_t packed) { return int(static_cast<int8_t>(packed & 255u)); }

template<class Packet>
NR_F13_INT_HD float step16(const Packet* a, int as, const Packet* b, int bs, float incoming) {
    const unsigned bits = browser_detail::floatBits(incoming);
    if ((bits & 0x7f800000u) == 0x7f800000u) return incoming;
    const int ie = int((bits >> 23) & 255) - 127;
    int maximum = (bits & 0x7fffffffu) ? (ie < -14 ? -14 : ie) : -21;
    // One 14-bit register record per term: signed nine-bit product and a
    // five-bit exponent biased by twelve. No per-product float conversion.
    unsigned terms[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const unsigned left = a[i * as], right = b[i * bs];
        const int product = mantissa(left) * mantissa(right);
        const int exponent = int(left >> 8) + int(right >> 8) - 12;
        terms[i] = unsigned(product & 511) | (unsigned(exponent + 12) << 9);
        if (product && exponent > maximum) maximum = exponent;
    }
    int units;
#if defined(__CUDA_ARCH__)
    units = __float2int_rz(__fmul_rn(incoming, __int_as_float((140 - maximum) << 23)));
#else
    units = int(std::trunc(std::ldexp(incoming, 13 - maximum)));
#endif
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int product = int(terms[i] & 511u);
        if (product & 256) product -= 512;
        const unsigned magnitude = unsigned(product < 0 ? -product : product);
        const int shift = int(terms[i] >> 9) - maximum - 5;
        const int right = -shift < 0 ? 0 : (-shift > 31 ? 31 : -shift);
        const int left = shift > 0 ? shift : 0;
        const unsigned aligned = shift >= 0 ? magnitude << left : magnitude >> right;
        units += product < 0 ? -int(aligned) : int(aligned);
    }
#if defined(__CUDA_ARCH__)
    return __fmul_rn(float(units), __int_as_float((maximum + 114) << 23));
#else
    return std::ldexp(float(units), maximum - 13);
#endif
}

static __global__ void gemm(const H* a, const H* weights, H* output, H* raw,
                           const H* residual, const H* scale, float* diagnostic_wide,
                           int M, int K, int N, int experts, bool broadcast,
                           bool silu, int partition, bool quantize) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = int(blockIdx.y) * 16 + ty, column = int(blockIdx.x) * 16 + tx, expert = blockIdx.z;
    const size_t as = size_t(K) * (broadcast ? 1 : experts), os = size_t(N) * experts;
    const size_t at = size_t(row) * os + size_t(expert) * N + column;
    const bool valid = row < M && column < N;
    float accumulator = valid && residual ? __fmul_rn(halfToFloat(residual[at]), halfToFloat(scale[expert * N + column])) : 0.f;
    float total = 0.f;
    __shared__ unsigned left[16][16], right[16][16];
    for (int start = 0; start < K; start += 16) {
        left[ty][tx] = row < M ? pack_operand(a[size_t(row) * as + (broadcast ? 0 : size_t(expert) * K) + start + tx]) : 0u;
        right[ty][tx] = column < N ? pack_operand(weights[(size_t(expert) * K + start + ty) * N + column]) : 0u;
        __syncthreads();
        if (valid) {
            accumulator = step16(&left[ty][0], 1, &right[0][tx], 16, accumulator);
            if (partition && ((start + 16) % partition == 0 || start + 16 == K)) {
                total = start < partition ? accumulator : __fadd_rn(total, accumulator);
                accumulator = 0.f;
            }
        }
        __syncthreads();
    }
    if (valid) {
        const float value = partition ? total : accumulator;
        if (diagnostic_wide) diagnostic_wide[at] = value;
        const H index = browserHalfRtz(value), published = silu ? browserSiluHalf(index) : index;
        if (raw) raw[at] = published;
        output[at] = quantize ? (silu ? browserSiluE4(index) : browserPublishE4(value)) : published;
    }
}
} // namespace f13_integer_detail

inline bool f13_integer_ranges_overlap(const void* first, size_t first_bytes,
                                       const void* second, size_t second_bytes) {
    if (!first || !second) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(first), b = reinterpret_cast<uintptr_t>(second);
    return a <= b ? b - a < first_bytes : a - b < second_bytes;
}

// Same expert layout as the existing exact-F13 diagnostic. Optional wide output
// exposes pre-publication FP32 only for independent fixtures and diagnostics.
inline void nr_gemm_f13_integer(const __half* a, const __half* weights, __half* output,
    __half* raw, const __half* residual, const __half* scale,
    int M, int K, int N, int experts, int broadcast, int silu,
    int partition, int quantize, cudaStream_t stream, float* diagnostic_wide = nullptr) {
    if (!a || !weights || !output || M <= 0 || K <= 0 || N <= 0 || K % 16
        || experts < 1 || experts > 65535 || M > 65535 * 16 || N > INT_MAX - 15
        || partition < 0 || partition % 16 || (residual && !scale)
        || (broadcast != 0 && broadcast != 1) || (silu != 0 && silu != 1)
        || (quantize != 0 && quantize != 1))
        throw std::invalid_argument("Invalid integer F13 shape or pointers");
    const size_t os = size_t(N) * experts;
    if (os > INT_MAX || size_t(M) > size_t(INT_MAX) / os || size_t(K) > size_t(INT_MAX) / os
        || size_t(K) * (broadcast ? 1 : experts) > INT_MAX)
        throw std::invalid_argument("Integer F13 tensor exceeds bounded element count");
    const size_t input_bytes = size_t(M) * K * (broadcast ? 1 : experts) * sizeof(__half);
    const size_t weights_bytes = size_t(K) * os * sizeof(__half);
    const size_t output_bytes = size_t(M) * os * sizeof(__half), scale_bytes = os * sizeof(__half);
    for (const void* target : {static_cast<const void*>(output), static_cast<const void*>(raw)}) {
        if (f13_integer_ranges_overlap(target, output_bytes, a, input_bytes)
            || f13_integer_ranges_overlap(target, output_bytes, weights, weights_bytes)
            || f13_integer_ranges_overlap(target, output_bytes, scale, scale_bytes)
            || (target != residual && f13_integer_ranges_overlap(target, output_bytes, residual, output_bytes)))
            throw std::invalid_argument("Integer F13 output overlaps an input");
    }
    if (f13_integer_ranges_overlap(output, output_bytes, raw, output_bytes))
        throw std::invalid_argument("Integer F13 output overlaps raw output");
    if (diagnostic_wide) {
        const size_t wide_bytes = size_t(M) * os * sizeof(float);
        if (f13_integer_ranges_overlap(diagnostic_wide, wide_bytes, a, input_bytes)
            || f13_integer_ranges_overlap(diagnostic_wide, wide_bytes, weights, weights_bytes)
            || f13_integer_ranges_overlap(diagnostic_wide, wide_bytes, residual, output_bytes)
            || f13_integer_ranges_overlap(diagnostic_wide, wide_bytes, scale, scale_bytes)
            || f13_integer_ranges_overlap(diagnostic_wide, wide_bytes, output, output_bytes)
            || f13_integer_ranges_overlap(diagnostic_wide, wide_bytes, raw, output_bytes))
            throw std::invalid_argument("Integer F13 wide diagnostic overlaps a tensor");
    }
    f13_integer_detail::gemm<<<dim3((N + 15) / 16, (M + 15) / 16, experts), dim3(16,16), 0, stream>>>(
        a, weights, output, raw, residual, scale, diagnostic_wide, M, K, N, experts,
        broadcast != 0, silu != 0, partition, quantize != 0);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace nr
#undef NR_F13_INT_HD
