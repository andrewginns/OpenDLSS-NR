// Fused SM75 window attention for grouped-half or full-FP32 accumulation.
// Network semantics derived from OpenDLSS-NR (MIT, Copyright (c) 2026 maan).
//
// One CTA per (window, head), four warps per CTA. Q/K/V, scores and normalized
// probabilities stay in shared memory. The caller provides raw half QKV in
// [pixel][head][Q32,K32,V32], priors in [head][natural query][physical key],
// original float learned scales and decoded-E4 half output [pixel][head][32].
//
// The default fuses window_pack/qk/softmax/pv/scatter while preserving FP32 MMA
// -> half rounding after each K16 group. The experimental RoundGroups=false
// variant retains FP32 accumulators through Q*K and P*V, then publishes half
// at the existing final boundaries. Neither emulates Ada FP8 F13 truncation.
// Validate old/new operator outputs on
// SM75, including phases 0..3 and partial edge windows, before enabling it.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <stdexcept>

#include "kernels/numerics/numerics.cuh"

namespace nr_window_detail {

namespace wmma = nvcuda::wmma;
using Half = __half;
using nr::addHalf;
using nr::fmaHalf;
using nr::halfBits;
using nr::halfFromBits;
using nr::halfRne;
using nr::halfToFloat;
using nr::mulHalf;
using nr::publishE4;
using nr::roundF16;
using nr::windowExpWeight;

constexpr unsigned kWarpMask = 0xffffffffu;
constexpr int kWarps = 4;
constexpr int kVectorStride = 40;  // eight half padding values prevent bank aliasing
constexpr int kScoreStride = 72;

__device__ __forceinline__ Half shuffle_half(Half value, int source_lane) {
    const unsigned bits = __shfl_sync(kWarpMask, unsigned(halfBits(value)), source_lane);
    return halfFromBits(static_cast<uint16_t>(bits));
}

// Each lane holds one of the 32 input channels. Match the pinned WGSL's FP32
// pair sum followed by half publication, then the half strides 8,4,2,1 tree.
// All lanes execute all shuffles, including lanes outside each active subtree.
template<bool Browser=false>
__device__ __forceinline__ Half inverse_norm_warp(Half value, int lane) {
    const Half partner = shuffle_half(value, lane ^ 16);
    const float low_square = __fmul_rn(halfToFloat(value), halfToFloat(value));
    const float high_square = halfToFloat(mulHalf(partner, partner));
    const float pair=__fadd_rn(low_square,high_square);
    Half partial;
    if constexpr(Browser)partial=__float2half_rz(pair);else partial=halfRne(pair);
#pragma unroll
    for (int stride = 8; stride > 0; stride >>= 1) {
        const Half other = shuffle_half(partial, (lane + stride) & 31);
        if (lane < stride) partial = addHalf(partial, other);
    }
    if(lane==0){
        if constexpr(Browser)partial=__float2half_rz(__frsqrt_rn(halfToFloat(partial)));
        else partial=halfRne(rsqrtf(halfToFloat(partial)));
    }
    return shuffle_half(partial, 0);
}

// Exact upstream physical-key reduction. Each lane holds two exponentials:
// e0=e[lane], e1=e[lane+32]. Lanes 0..7 produce t[g], lanes 0/1 produce the
// even/odd chains, then lane 0's total is broadcast. Every addition is half.
__device__ __forceinline__ Half softmax_total_warp(Half e0, Half e1, int lane) {
    const Half b0 = addHalf(e0, shuffle_half(e0, (lane + 8) & 31));
    const Half b1 = addHalf(shuffle_half(e0, (lane + 16) & 31),
                            shuffle_half(e0, (lane + 24) & 31));
    const Half b2 = addHalf(e1, shuffle_half(e1, (lane + 8) & 31));
    const Half b3 = addHalf(shuffle_half(e1, (lane + 16) & 31),
                            shuffle_half(e1, (lane + 24) & 31));
    const Half t = addHalf(addHalf(addHalf(b0, b1), b2), b3);
    const Half t2 = shuffle_half(t, (lane + 2) & 31);
    const Half t4 = shuffle_half(t, (lane + 4) & 31);
    const Half t6 = shuffle_half(t, (lane + 6) & 31);
    const Half parity_sum = addHalf(addHalf(addHalf(t, t2), t4), t6);
    const Half even = shuffle_half(parity_sum, 0);
    const Half odd = shuffle_half(parity_sum, 1);
    return addHalf(even, odd);
}

__device__ __forceinline__ int physical_token(int natural) {
    const int x = natural & 7, y = natural >> 3;
    return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3);
}

template<bool RoundGroups = true,bool Browser=false>
__global__ __launch_bounds__(128) void window_kernel(
    const Half* __restrict__ qkv, const Half* __restrict__ prior,
    const float* __restrict__ learned_scale, Half* __restrict__ output,
    int width, int height, int heads, int shift_x, int shift_y, int windows_x) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int head = blockIdx.x % heads;
    const int window = blockIdx.x / heads;
    const int origin_x = (window % windows_x) * 8 - shift_x;
    const int origin_y = (window / windows_x) * 8 - shift_y;
    const Half scale = Browser?__float2half_rz(learned_scale[head]):halfRne(learned_scale[head]);

    // Each WMMA base and tile offset is 32-byte aligned. WMMA half strides are
    // multiples of eight; float accumulator staging stride is a multiple of four.
    __shared__ __align__(32) Half query[64 * kVectorStride];
    __shared__ __align__(32) Half key[64 * kVectorStride];
    __shared__ __align__(32) Half value[64 * kVectorStride];
    __shared__ __align__(32) Half probabilities[64 * kScoreStride];
    __shared__ __align__(32) float accumulator[kWarps][16 * 16];
    float* const scratch = accumulator[warp];

    // Four warps load disjoint natural tokens. All 32 lanes cooperate on a
    // token's norm and issue contiguous channel loads, then store K/V in the
    // physical 4x4-tile order expected by priors and the softmax reduction.
#pragma unroll
    for (int token = warp; token < 64; token += kWarps) {
        const int x = origin_x + (token & 7), y = origin_y + (token >> 3);
        const bool valid = x >= 0 && y >= 0 && x < width && y < height;
        Half q = halfRne(0.0f), k = q, v = q;
        if (valid) {
            const int at = (y * width + x) * heads * 96 + head * 96 + lane;
            q = qkv[at]; k = qkv[at + 32]; v = qkv[at + 64];
        }
        const Half q_norm = inverse_norm_warp<Browser>(q, lane);
        const Half k_norm = inverse_norm_warp<Browser>(k, lane);
        const int physical = physical_token(token);
        // Explicit +0 for outside-field tokens matches window_pack. Valid zero
        // vectors go through 0*infinity -> NaN -> +0 at the E4 publication.
        query[token * kVectorStride + lane] = valid
            ? publishE4(mulHalf(mulHalf(q, q_norm), scale)) : halfRne(0.0f);
        key[physical * kVectorStride + lane] = valid
            ? publishE4(mulHalf(k, k_norm)) : halfRne(0.0f);
        value[physical * kVectorStride + lane] = valid ? publishE4(v) : halfRne(0.0f);
    }
    __syncthreads();

    // Each warp owns all operations for sixteen natural query rows. There are
    // no shared writes across warp-owned query rows after this point.
    const int query_base = warp * 16;
#pragma unroll
    for (int key_base = 0; key_base < 64; key_base += 16) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) {
            const int q = query_base + i / 16;
            const int k = key_base + i % 16;
            scratch[i] = halfToFloat(prior[head * 4096 + q * 64 + k]);
        }
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::load_matrix_sync(acc, scratch, 16, wmma::mem_row_major);
#pragma unroll
        for (int channel = 0; channel < 32; channel += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, Half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, Half, wmma::col_major> b;
            wmma::load_matrix_sync(a, query + query_base * kVectorStride + channel, kVectorStride);
            wmma::load_matrix_sync(b, key + key_base * kVectorStride + channel, kVectorStride);
            wmma::mma_sync(acc, a, b, acc);
            if constexpr (RoundGroups) {
#pragma unroll
                for (int i = 0; i < acc.num_elements; ++i) acc.x[i] = roundF16(acc.x[i]);
            }
        }
        wmma::store_matrix_sync(scratch, acc, 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int i = lane; i < 256; i += 32)
            probabilities[(query_base + i / 16) * kScoreStride + key_base + i % 16] = halfRne(scratch[i]);
        __syncwarp();  // all scratch readers finish before the next prior tile
    }

    // A warp normalizes each of its own query rows. The two exponentials per
    // lane stay in registers, eliminating a separate score/exp buffer.
#pragma unroll
    for (int row = 0; row < 16; ++row) {
        Half* const score = probabilities + (query_base + row) * kScoreStride;
        const Half e0 = windowExpWeight(score[lane]);
        const Half e1 = windowExpWeight(score[lane + 32]);
        const Half total = softmax_total_warp(e0, e1, lane);
        Half reciprocal = halfRne(0.0f);
        if(lane==0){
            if constexpr(Browser)reciprocal=__float2half_rz(__fdiv_rn(1.f,halfToFloat(total)));
            else reciprocal=halfRne(1.0f/halfToFloat(total));
        }
        reciprocal = shuffle_half(reciprocal, 0);
        score[lane] = publishE4(mulHalf(e0, reciprocal));
        score[lane + 32] = publishE4(mulHalf(e1, reciprocal));
    }
    __syncwarp();

    // P*V: four physical-key K16 groups. RoundGroups selects accumulator
    // publication; both variants publish the final half result to E4.
    // Two 16-column output tiles cover the head's 32 channels.
#pragma unroll
    for (int channel_base = 0; channel_base < 32; channel_base += 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int key_base = 0; key_base < 64; key_base += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, Half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, Half, wmma::row_major> b;
            wmma::load_matrix_sync(a, probabilities + query_base * kScoreStride + key_base, kScoreStride);
            wmma::load_matrix_sync(b, value + key_base * kVectorStride + channel_base, kVectorStride);
            wmma::mma_sync(acc, a, b, acc);
            if constexpr (RoundGroups) {
#pragma unroll
                for (int i = 0; i < acc.num_elements; ++i) acc.x[i] = roundF16(acc.x[i]);
            }
        }
        wmma::store_matrix_sync(scratch, acc, 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int i = lane; i < 256; i += 32) {
            const int token = query_base + i / 16;
            const int x = origin_x + (token & 7), y = origin_y + (token >> 3);
            if (x >= 0 && y >= 0 && x < width && y < height) {
                const int c = channel_base + i % 16;
                output[(y * width + x) * heads * 32 + head * 32 + c] = publishE4(halfRne(scratch[i]));
            }
        }
        __syncwarp();
    }
}

}  // namespace nr_window_detail

// Launch only; no allocation, synchronization or stream-capture restriction.
// Caller checks cudaGetLastError and handles errors through its normal API.
inline void nr_window_fast(const __half* qkv, const __half* prior, const float* scales,
                           __half* output, int width, int height, int heads, int phase,
                           cudaStream_t stream, bool round_groups = true,bool browser=false) {
    if (width <= 0 || height <= 0 || heads <= 0 || heads > 16 || phase < 0 || phase > 3)
        throw std::invalid_argument("invalid fused window shape or phase");
    const int sx = phase == 1 || phase == 2 ? 4 : 0;
    const int sy = phase == 1 || phase == 3 ? 4 : 0;
    const int windows_x = (width + sx + 7) / 8;
    const int windows_y = (height + sy + 7) / 8;
    const std::int64_t tasks = std::int64_t(windows_x) * windows_y * heads;
    if (tasks > 2147483647)
        throw std::invalid_argument("fused window grid exceeds one-dimensional CUDA limit");
    if(browser&&round_groups){
        nr_window_detail::window_kernel<true,true><<<static_cast<unsigned>(tasks),128,0,stream>>>(
            qkv,prior,scales,output,width,height,heads,sx,sy,windows_x);
    }else if(browser){
        nr_window_detail::window_kernel<false,true><<<static_cast<unsigned>(tasks),128,0,stream>>>(
            qkv,prior,scales,output,width,height,heads,sx,sy,windows_x);
    }else if (round_groups) {
        nr_window_detail::window_kernel<true><<<static_cast<unsigned>(tasks), 128, 0, stream>>>(
            qkv, prior, scales, output, width, height, heads, sx, sy, windows_x);
    } else {
        nr_window_detail::window_kernel<false><<<static_cast<unsigned>(tasks), 128, 0, stream>>>(
            qkv, prior, scales, output, width, height, heads, sx, sy, windows_x);
    }
}
