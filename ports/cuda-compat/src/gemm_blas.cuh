// FP16 operands with FP32 Tensor Core accumulation for the fp32all experiment.
// Half/E4 publication semantics derived from OpenDLSS-NR (MIT, 2026 maan).
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "kernels/numerics/numerics.cuh"

namespace nr {
namespace blas_detail {
using Half = __half;

inline void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("fp32all cuBLAS error " + std::to_string(status));
}

static __global__ void seed_residual(const Half* residual, const Half* scale,
                                    Half* output, size_t count, int row_stride) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = mulHalf(residual[i], scale[i % row_stride]);
}

static __global__ void add_partition(const Half* partial, Half* output,
                                    size_t count, int columns, int output_stride) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const size_t target = (i / columns) * output_stride + i % columns;
        output[target] = addHalf(output[target], partial[i]);
    }
}

static __global__ void publish_result(Half* output, Half* raw, size_t count,
                                     bool activation, bool quantize) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        Half value = output[i];
        if (activation) value = siluHalf(value);
        if (raw) raw[i] = value;
        output[i] = quantize ? publishE4(value) : value;
    }
}
} // namespace blas_detail

// A is [M, K*batches], or [M,K] when broadcast is true. W is [batches,K,N].
// Output/residual are [M,N*batches], scale is [N*batches]. Expert views are
// interleaved by row, not separate contiguous matrices. All work is enqueued on
// the supplied stream. Scratch is an existing M*N-half allocation reused for
// every later partition and expert; it must not alias any operand/output.
//
// A partition is accumulated in float then published once to half. The first
// partition includes the scaled-half residual via beta=1; subsequent partials
// are rounded to half before half addition to the preceding partition sum.
// No device allocations, synchronizations or host output copies occur here.
inline void nr_gemm_blas(const __half* a, const __half* w, __half* out, __half* raw,
                         const __half* residual, const __half* scale,
                         int M, int K, int N, int batches, int broadcast,
                         int silu, int partition, int quantize,
                         cudaStream_t stream, cublasHandle_t blas, __half* scratch,
                         size_t scratch_elements = size_t(32) * 1024 * 1024) {
    using namespace blas_detail;
    if (!a || !w || !out || !blas || M <= 0 || K <= 0 || N <= 0 || batches <= 0 ||
        M % 16 || K % 16 || N % 16 || partition < 0 || partition % 16 ||
        (residual && !scale) || (broadcast != 0 && broadcast != 1))
        throw std::runtime_error("Invalid fp32all GEMM arguments");
    const size_t input_stride_long = size_t(K) * (broadcast ? 1 : batches);
    const size_t output_stride_long = size_t(N) * batches;
    const size_t output_count = size_t(M) * output_stride_long;
    const size_t partial_count = size_t(M) * N;
    if (input_stride_long > INT_MAX || output_stride_long > INT_MAX || output_count > INT_MAX)
        throw std::runtime_error("fp32all GEMM exceeds bounded strides/count");
    const int input_stride = int(input_stride_long), output_stride = int(output_stride_long);
    const int partition_width = partition ? std::min(partition, K) : K;
    if (partition_width < K && (!scratch || scratch_elements < partial_count ||
                               scratch == out || scratch == a || scratch == w || scratch == residual))
        throw std::runtime_error("fp32all partition scratch is missing, too small or aliased");
    cudaStream_t handle_stream;
    cublasPointerMode_t pointer_mode;
    check_blas(cublasGetStream(blas, &handle_stream));
    check_blas(cublasGetPointerMode(blas, &pointer_mode));
    if (handle_stream != stream || pointer_mode != CUBLAS_POINTER_MODE_HOST)
        throw std::runtime_error("fp32all requires the caller stream and host cuBLAS scalars");
    // FP32 compute alone permits reduced-precision split-K reductions. Forbid
    // those while retaining Tensor Core eligibility for the half operands.
    const auto math_mode = static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    check_blas(cublasSetMathMode(blas, math_mode));
    if (residual) {
        seed_residual<<<unsigned((output_count + 255) / 256), 256, 0, stream>>>(
            residual, scale, out, output_count, output_stride);
        check_cuda(cudaGetLastError());
    }
    const float one = 1.f, zero = 0.f;
    for (int expert = 0; expert < batches; ++expert) {
        const __half* input = a + (broadcast ? 0 : size_t(expert) * K);
        const __half* weights = w + size_t(expert) * K * N;
        __half* output = out + size_t(expert) * N;
        for (int start = 0; start < K; start += partition_width) {
            const int width = std::min(partition_width, K - start);
            const bool first = start == 0;
            __half* target = first ? output : scratch;
            const int target_stride = first ? output_stride : N;
            const float* beta = (first && residual) ? &one : &zero;
            // C^T = W^T A^T maps row-major views to column-major cuBLAS
            // without transposes or packing. Offsets retain original strides.
            check_blas(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, width,
                &one, weights + size_t(start) * N, CUDA_R_16F, N,
                input + start, CUDA_R_16F, input_stride,
                beta, target, CUDA_R_16F, target_stride,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            if (!first) {
                add_partition<<<unsigned((partial_count + 255) / 256), 256, 0, stream>>>(
                    scratch, output, partial_count, N, output_stride);
                check_cuda(cudaGetLastError());
            }
        }
    }
    publish_result<<<unsigned((output_count + 255) / 256), 256, 0, stream>>>(
        out, raw, output_count, silu != 0, quantize != 0);
    check_cuda(cudaGetLastError());
}
} // namespace nr
