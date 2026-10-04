// Exact-F13 helpers. browser-compat-v1 selects compact variant 4; other variants
// are retained internal helpers and have no public runtime policy.
#pragma once
#include "gemm_f13_integer.cuh"
#include <type_traits>

#define NR_F13_VARIANT_HD __host__ __device__ __forceinline__
namespace nr {
namespace f13_variants_detail {
using H = __half;
NR_F13_VARIANT_HD int exponent(float x, int floor) {
    const int e = int((browser_detail::floatBits(x) >> 23) & 255u) - 127;
    return e < floor ? floor : e;
}
NR_F13_VARIANT_HD bool nonfinite(float x) {
    return (browser_detail::floatBits(x) & 0x7f800000u) == 0x7f800000u;
}
NR_F13_VARIANT_HD int start_exponent(float x) {
    return (browser_detail::floatBits(x) & 0x7fffffffu) ? exponent(x, -14) : -21;
}
NR_F13_VARIANT_HD float power(int e) {
#if defined(__CUDA_ARCH__)
    return __int_as_float((e + 127) << 23);
#else
    return std::ldexp(1.f, e);
#endif
}
NR_F13_VARIANT_HD float multiply(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a,b);
#else
    return a*b;
#endif
}
NR_F13_VARIANT_HD int truncate(float x) {
#if defined(__CUDA_ARCH__)
    return __float2int_rz(x);
#else
    return int(std::trunc(x));
#endif
}

// Predecoded exponents share their value array's stride. Values are decoded
// finite E4, so each float product is exact. The aligned sum is the original
// browser F13 arithmetic with repeated half conversions removed.
template<bool Predecoded>
NR_F13_VARIANT_HD float step_float_impl(const float* a, int as, const int* ae,
    const float* b, int bs, const int* be, float incoming) {
    if (nonfinite(incoming)) return incoming;
    int maximum = start_exponent(incoming);
    float products[16];
#pragma unroll
    for (int i=0;i<16;++i) {
        const float left=a[i*as],right=b[i*bs];
        products[i]=multiply(left,right);
        const int e=Predecoded ? ae[i*as]+be[i*bs] : exponent(left,-6)+exponent(right,-6);
        if(left!=0.f&&right!=0.f&&e>maximum)maximum=e;
    }
    const float alignment=power(13-maximum);
    int units=truncate(multiply(incoming,alignment));
#pragma unroll
    for(int i=0;i<16;++i)units+=truncate(multiply(products[i],alignment));
    return multiply(float(units),power(maximum-13));
}
NR_F13_VARIANT_HD float step_float(const float* a,int as,const int* ae,
    const float* b,int bs,const int* be,float incoming) {
    return step_float_impl<true>(a,as,ae,b,bs,be,incoming);
}
NR_F13_VARIANT_HD float step_float_values(const float* a,int as,const float* b,int bs,float incoming) {
    return step_float_impl<false>(a,as,nullptr,b,bs,nullptr,incoming);
}
NR_F13_VARIANT_HD float step_unpacked(const int* a,int as,const int* ae,
    const int* b,int bs,const int* be,float incoming) {
    if(nonfinite(incoming))return incoming;
    int maximum=start_exponent(incoming),products[16],exponents[16];
#pragma unroll
    for(int i=0;i<16;++i){
        products[i]=a[i*as]*b[i*bs];exponents[i]=ae[i*as]+be[i*bs];
        if(products[i]&&exponents[i]>maximum)maximum=exponents[i];
    }
    int units=truncate(multiply(incoming,power(13-maximum)));
#pragma unroll
    for(int i=0;i<16;++i){
        const int product=products[i],shift=7+exponents[i]-maximum;
        const unsigned magnitude=unsigned(product<0?-product:product);
        const int right=-shift<0?0:(-shift>31?31:-shift),left=shift>0?shift:0;
        const unsigned aligned=shift>=0?magnitude<<left:magnitude>>right;
        units+=product<0?-int(aligned):int(aligned);
    }
    return multiply(float(units),power(maximum-13));
}

// Same compact representation as the independently tested fused-window
// prototype: decoded E4 FP32 values leave low mantissa bits available.
NR_F13_VARIANT_HD uint32_t pack_compact(H value) {
    const int e=int((halfBits(value)>>10)&31)-15;
    return browser_detail::floatBits(halfToFloat(value))|uint32_t((e < -6 ? -6 : e)+6);
}
NR_F13_VARIANT_HD float compact_value(uint32_t record) {
    const uint32_t bits=record&~15u;
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;std::memcpy(&value,&bits,sizeof(value));return value;
#endif
}
NR_F13_VARIANT_HD float step_compact(const uint32_t*a,int as,const uint32_t*b,int bs,float incoming) {
    if(nonfinite(incoming))return incoming;
    int maximum=start_exponent(incoming);float products[16];
#pragma unroll
    for(int i=0;i<16;++i){
        const uint32_t left=a[i*as],right=b[i*bs];
        products[i]=multiply(compact_value(left),compact_value(right));
        const int e=int(left&15u)+int(right&15u)-12;
        if(products[i]!=0.f&&e>maximum)maximum=e;
    }
    const float alignment=power(13-maximum);int units=truncate(multiply(incoming,alignment));
#pragma unroll
    for(int i=0;i<16;++i)units+=truncate(multiply(products[i],alignment));
    return multiply(float(units),power(maximum-13));
}

// Variant1: FP32 values + floor exponents. Variant2: separate signed
// mantissas/exponents. Variant3: FP32 values, derive exponents in the step.
template<int Variant>
static __global__ void gemm(const H* a,const H* weights,H* output,H* raw,
    const H* residual,const H* scale,float* diagnostic_wide,
    int M,int K,int N,int experts,bool broadcast,bool silu,int partition,bool quantize) {
    using Value=std::conditional_t<Variant==2,int,std::conditional_t<Variant==4,uint32_t,float>>;
    const int tx=threadIdx.x,ty=threadIdx.y,row=int(blockIdx.y)*16+ty,column=int(blockIdx.x)*16+tx,expert=blockIdx.z;
    const size_t as=size_t(K)*(broadcast?1:experts),os=size_t(N)*experts;
    const size_t at=size_t(row)*os+size_t(expert)*N+column;
    const bool valid=row<M&&column<N;
    float accumulator=valid&&residual?__fmul_rn(halfToFloat(residual[at]),halfToFloat(scale[expert*N+column])):0.f,total=0.f;
    __shared__ Value av[16][16],bv[16][16];
    __shared__ int ae[16][16],be[16][16];
    for(int start=0;start<K;start+=16){
        const H left=row<M?a[size_t(row)*as+(broadcast?0:size_t(expert)*K)+start+tx]:halfRne(0.f);
        const H right=column<N?weights[(size_t(expert)*K+start+ty)*N+column]:halfRne(0.f);
        if constexpr(Variant==4){av[ty][tx]=pack_compact(left);bv[ty][tx]=pack_compact(right);}
        else if constexpr(Variant==2){
            const unsigned ap=f13_integer_detail::pack_operand(left),bp=f13_integer_detail::pack_operand(right);
            av[ty][tx]=f13_integer_detail::mantissa(ap);bv[ty][tx]=f13_integer_detail::mantissa(bp);
            ae[ty][tx]=int(ap>>8)-6;be[ty][tx]=int(bp>>8)-6;
        }else{
            av[ty][tx]=halfToFloat(left);bv[ty][tx]=halfToFloat(right);
            if constexpr(Variant==1){ae[ty][tx]=exponent(av[ty][tx],-6);be[ty][tx]=exponent(bv[ty][tx],-6);}
        }
        __syncthreads();
        if(valid){
            if constexpr(Variant==1)accumulator=step_float(&av[ty][0],1,&ae[ty][0],&bv[0][tx],16,&be[0][tx],accumulator);
            else if constexpr(Variant==2)accumulator=step_unpacked(&av[ty][0],1,&ae[ty][0],&bv[0][tx],16,&be[0][tx],accumulator);
            else if constexpr(Variant==3)accumulator=step_float_values(&av[ty][0],1,&bv[0][tx],16,accumulator);
            else accumulator=step_compact(&av[ty][0],1,&bv[0][tx],16,accumulator);
            if(partition&&((start+16)%partition==0||start+16==K)){total=start<partition?accumulator:__fadd_rn(total,accumulator);accumulator=0.f;}
        }
        __syncthreads();
    }
    if(valid){
        const float value=partition?total:accumulator;
        if(diagnostic_wide)diagnostic_wide[at]=value;
        const H index=browserHalfRtz(value),published=silu?browserSiluHalf(index):index;
        if(raw)raw[at]=published;
        output[at]=quantize?(silu?browserSiluE4(index):browserPublishE4(value)):published;
    }
}
} // namespace f13_variants_detail

inline void nr_gemm_f13_variant(int variant,const __half* a,const __half* weights,__half* output,
    __half* raw,const __half* residual,const __half* scale,int M,int K,int N,int experts,
    int broadcast,int silu,int partition,int quantize,cudaStream_t stream,float* diagnostic_wide=nullptr) {
    if(variant==0){nr_gemm_f13_integer(a,weights,output,raw,residual,scale,M,K,N,experts,broadcast,silu,partition,quantize,stream,diagnostic_wide);return;}
    if(variant<1||variant>4||!a||!weights||!output||M<=0||K<=0||N<=0||K%16||experts<1||experts>65535
        ||M>65535*16||N>INT_MAX-15||partition<0||partition%16||(residual&&!scale)
        ||(broadcast!=0&&broadcast!=1)||(silu!=0&&silu!=1)||(quantize!=0&&quantize!=1))
        throw std::invalid_argument("Invalid F13 variant shape or pointers");
    const size_t os=size_t(N)*experts;
    if(os>INT_MAX||size_t(M)>size_t(INT_MAX)/os||size_t(K)>size_t(INT_MAX)/os||size_t(K)*(broadcast?1:experts)>INT_MAX)
        throw std::invalid_argument("F13 variant tensor exceeds bounded count");
    const size_t input_bytes=size_t(M)*K*(broadcast?1:experts)*sizeof(__half),weights_bytes=size_t(K)*os*sizeof(__half);
    const size_t output_bytes=size_t(M)*os*sizeof(__half),scale_bytes=os*sizeof(__half);
    for(const void*target:{static_cast<const void*>(output),static_cast<const void*>(raw)})
        if(f13_integer_ranges_overlap(target,output_bytes,a,input_bytes)||f13_integer_ranges_overlap(target,output_bytes,weights,weights_bytes)
            ||f13_integer_ranges_overlap(target,output_bytes,scale,scale_bytes)||(target!=residual&&f13_integer_ranges_overlap(target,output_bytes,residual,output_bytes)))
            throw std::invalid_argument("F13 variant output overlaps an input");
    if(f13_integer_ranges_overlap(output,output_bytes,raw,output_bytes))throw std::invalid_argument("F13 variant output overlaps raw");
    if(diagnostic_wide){
        const size_t wide_bytes=size_t(M)*os*sizeof(float);
        if(f13_integer_ranges_overlap(diagnostic_wide,wide_bytes,a,input_bytes)||f13_integer_ranges_overlap(diagnostic_wide,wide_bytes,weights,weights_bytes)
            ||f13_integer_ranges_overlap(diagnostic_wide,wide_bytes,residual,output_bytes)||f13_integer_ranges_overlap(diagnostic_wide,wide_bytes,scale,scale_bytes)
            ||f13_integer_ranges_overlap(diagnostic_wide,wide_bytes,output,output_bytes)||f13_integer_ranges_overlap(diagnostic_wide,wide_bytes,raw,output_bytes))
            throw std::invalid_argument("F13 variant wide output overlaps a tensor");
    }
    const dim3 grid((N+15)/16,(M+15)/16,experts),block(16,16);
    if(variant==1)f13_variants_detail::gemm<1><<<grid,block,0,stream>>>(a,weights,output,raw,residual,scale,diagnostic_wide,M,K,N,experts,broadcast,silu,partition,quantize);
    else if(variant==2)f13_variants_detail::gemm<2><<<grid,block,0,stream>>>(a,weights,output,raw,residual,scale,diagnostic_wide,M,K,N,experts,broadcast,silu,partition,quantize);
    else if(variant==3)f13_variants_detail::gemm<3><<<grid,block,0,stream>>>(a,weights,output,raw,residual,scale,diagnostic_wide,M,K,N,experts,broadcast,silu,partition,quantize);
    else f13_variants_detail::gemm<4><<<grid,block,0,stream>>>(a,weights,output,raw,residual,scale,diagnostic_wide,M,K,N,experts,broadcast,silu,partition,quantize);
    const cudaError_t status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace nr
#undef NR_F13_VARIANT_HD
