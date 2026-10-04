// Diagnostic CUDA-core emulation of the pinned OpenDLSS-NR F13 dot schedule.
// Source semantics: tools/reference/port/{src/numerics.js,shaders/numerics.wgsl}
// OpenDLSS-NR is MIT, Copyright (c) 2026 maan. This kernel does not use Tensor Cores.
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include "kernels/numerics/numerics.cuh"
#include "browser_numerics.cuh"

namespace nr {
namespace f13_detail {
using Half = __half;

__device__ __forceinline__ int exponent(Half value, int floor) {
    const int e = int((halfBits(value) >> 10) & 31) - 15;
    return e < floor ? floor : e;
}

// Inputs must be finite E4M3 values decoded exactly to half. A half accumulator
// is aligned with sixteen products, each term is truncated toward zero, and
// the integer total is rounded once to half. Every aligned term fits 15 bits;
// all seventeen terms fit 20 bits, so int32 addition is exact and order-free.
__device__ __forceinline__ Half step16(const Half* a, int a_stride,
                                      const Half* b, int b_stride, Half incoming) {
    const unsigned incoming_bits = halfBits(incoming);
    if ((incoming_bits & 0x7c00u) == 0x7c00u) return incoming;
    int maximum = (incoming_bits & 0x7fffu) ? exponent(incoming, -14) : -21;
    float products[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const Half left = a[i * a_stride], right = b[i * b_stride];
        products[i] = __fmul_rn(halfToFloat(left), halfToFloat(right));
        if ((halfBits(left) & 0x7fffu) && (halfBits(right) & 0x7fffu)) {
            const int product_exponent = exponent(left, -6) + exponent(right, -6);
            if (product_exponent > maximum) maximum = product_exponent;
        }
    }
    const float alignment = __int_as_float((140 - maximum) << 23);
    int units = __float2int_rz(__fmul_rn(halfToFloat(incoming), alignment));
#pragma unroll
    for (int i = 0; i < 16; ++i)
        units += __float2int_rz(__fmul_rn(products[i], alignment));
    const float unit = __int_as_float((maximum + 114) << 23);
    return halfRne(__fmul_rn(float(units), unit));
}

// Separate browser diagnostic: the observed block-0 K32 GEMM retains the
// aligned FP32 result between K16 groups. Its incoming exponent therefore
// comes from FP32, clamped at -14, rather than from a half conversion.
__device__ __forceinline__ float step16_browser(const Half* a,int as,const Half* b,int bs,float incoming) {
    const unsigned bits=__float_as_uint(incoming);
    if((bits&0x7f800000u)==0x7f800000u)return incoming;
    int e=int((bits>>23)&255)-127;
    int maximum=(bits&0x7fffffffu)?(e < -14 ? -14 : e):-21;
    float products[16];
#pragma unroll
    for(int i=0;i<16;++i){
        const Half left=a[i*as],right=b[i*bs];
        products[i]=__fmul_rn(halfToFloat(left),halfToFloat(right));
        if((halfBits(left)&0x7fffu)&&(halfBits(right)&0x7fffu)){
            const int pe=exponent(left,-6)+exponent(right,-6);
            if(pe>maximum)maximum=pe;
        }
    }
    const float alignment=__int_as_float((140-maximum)<<23);
    int units=__float2int_rz(__fmul_rn(incoming,alignment));
#pragma unroll
    for(int i=0;i<16;++i)units+=__float2int_rz(__fmul_rn(products[i],alignment));
    return __fmul_rn(float(units),__int_as_float((maximum+114)<<23));
}

// One CUDA thread owns one output. A/B K16 tiles are shared across 256 outputs;
// only scalar CUDA-core arithmetic performs exponent selection and truncation.
template<bool Browser=false>
static __global__ void gemm_kernel(const Half* a, const Half* weights,
                                   Half* output, Half* raw, const Half* residual,
                                   const Half* scale, int M, int K, int N,
                                   int batches, bool broadcast, bool activation,
                                   int partition, bool quantize) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = int(blockIdx.y) * 16 + ty;
    const int column = int(blockIdx.x) * 16 + tx;
    const int expert = blockIdx.z;
    const size_t input_stride = size_t(K) * (broadcast ? 1 : batches);
    const size_t output_stride = size_t(N) * batches;
    const size_t output_index = size_t(row) * output_stride + size_t(expert) * N + column;
    const bool valid = row < M && column < N;
    using Accumulator=std::conditional_t<Browser,float,Half>;
    Accumulator accumulator;
    if constexpr(Browser){
        accumulator=valid&&residual?__fmul_rn(halfToFloat(residual[output_index]),halfToFloat(scale[expert*N+column])):0.f;
    }else accumulator=valid&&residual?mulHalf(residual[output_index],scale[expert*N+column]):halfRne(0.f);
    Accumulator partition_sum;
    if constexpr(Browser)partition_sum=0.f;else partition_sum=halfRne(0.f);
    __shared__ __align__(32) Half left[16][16];
    __shared__ __align__(32) Half right[16][16];
    for (int start = 0; start < K; start += 16) {
        left[ty][tx] = row < M ? a[size_t(row) * input_stride + (broadcast ? 0 : size_t(expert) * K) + start + tx] : halfRne(0.f);
        right[ty][tx] = column < N ? weights[(size_t(expert) * K + start + ty) * N + column] : halfRne(0.f);
        __syncthreads();
        if (valid) {
            if constexpr(Browser)accumulator=step16_browser(&left[ty][0],1,&right[0][tx],16,accumulator);
            else accumulator=step16(&left[ty][0],1,&right[0][tx],16,accumulator);
            if (partition && ((start + 16) % partition == 0 || start + 16 == K)) {
                // Observed browser partitions retain each FP32 partial and
                // left-fold the totals in FP32. The residual seed belongs
                // only to the first partition; subsequent groups start at 0.
                if constexpr(Browser)partition_sum=start<partition?accumulator:__fadd_rn(partition_sum,accumulator);
                else partition_sum=start<partition?accumulator:addHalf(partition_sum,accumulator);
                if constexpr(Browser)accumulator=0.f;else accumulator=halfRne(0.f);
            }
        }
        __syncthreads();
    }
    if (valid) {
        Half value;
        float unrounded=0.f;
        if constexpr(Browser){
            unrounded=partition?partition_sum:accumulator;
            value=browserHalfRtz(unrounded);
        }else value=partition?partition_sum:accumulator;
        if constexpr(Browser){
            // Observed raw-half and E4 publications are independent. SiLU's
            // table index is the RTZ half, but its E4 table entry comes from
            // the unrounded FP32 polynomial, not from its stored half column.
            const Half raw_value=activation?browserSiluHalf(value):value;
            if(raw)raw[output_index]=raw_value;
            output[output_index]=quantize?(activation?browserSiluE4(value):browserPublishE4(unrounded)):raw_value;
        }else{
            if(activation)value=siluHalf(value);
            if(raw)raw[output_index]=value;
            output[output_index]=quantize?publishE4(value):value;
        }
    }
}
} // namespace f13_detail

// Same row-interleaved expert layout as nr_gemm_blas. M/N tails are supported;
// K and nonzero partition widths must be multiples of sixteen. Half residual
// scaling happens before the first F13 group. No allocations or synchronization.
inline void nr_gemm_f13(const __half* a, const __half* weights, __half* output,
                        __half* raw, const __half* residual, const __half* scale,
                        int M, int K, int N, int batches, int broadcast, int silu,
                        int partition, int quantize, cudaStream_t stream,bool browser=false) {
    if (!a || !weights || !output || M <= 0 || K <= 0 || N <= 0 || K % 16 ||
        batches < 1 || batches > 65535 || M > 65535 * 16 || partition < 0 || partition % 16 ||
        (residual && !scale) || (broadcast != 0 && broadcast != 1) ||
        size_t(M) * N * batches > INT_MAX)
        throw std::invalid_argument("Invalid diagnostic F13 GEMM shape or pointers");
    if(browser)
        f13_detail::gemm_kernel<true><<<dim3((N+15)/16,(M+15)/16,batches),dim3(16,16),0,stream>>>(
            a,weights,output,raw,residual,scale,M,K,N,batches,broadcast!=0,silu!=0,partition,quantize!=0);
    else
        f13_detail::gemm_kernel<false><<<dim3((N+15)/16,(M+15)/16,batches),dim3(16,16),0,stream>>>(
            a,weights,output,raw,residual,scale,M,K,N,batches,broadcast!=0,silu!=0,partition,quantize!=0);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace nr
