// Experimental FP16 Tensor Core GEMM with FP32 output and observed browser
// publications. This approximates F13 truncation; it is not exact F13 emulation.
#pragma once
#include "gemm_blas.cuh"
#include "browser_numerics.cuh"

namespace nr {
namespace browser_blas_detail {
using H=__half;
static __global__ void seed(const H*res,const H*scale,float*scratch,int rows,int n,int stride){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*n)scratch[i]=__fmul_rn(halfToFloat(res[(i/n)*stride+i%n]),halfToFloat(scale[i%n]));
}
static __global__ void partial(const float*scratch,float*sum,int count,bool first){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)sum[i]=first?scratch[i]:__fadd_rn(sum[i],scratch[i]);
}
static __global__ void publish(const float*scratch,H*out,H*raw,int rows,int n,int stride,bool silu,bool quantize){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*n)return;
    const int target=(i/n)*stride+i%n;
    const float value=scratch[i];
    const H index=browserHalfRtz(value),raw_value=silu?browserSiluHalf(index):index;
    if(raw)raw[target]=raw_value;
    out[target]=quantize?(silu?browserSiluE4(index):browserPublishE4(value)):raw_value;
}
} // namespace browser_blas_detail

// The existing score scratch is borrowed only on the same stream. Its storage
// is interpreted as FP32 and bounded row tiles keep large full-resolution
// GEMMs inside it. A/W are FP16, C and residual seeding are FP32; no allocation,
// synchronization, host copy or half result store occurs before publication.
// Partitioned GEMMs use two bounded FP32 zones: the current cuBLAS partial and
// a sequential FP32 total. Their publication follows observed browser behavior.
inline void nr_gemm_fp32browser(const __half*a,const __half*w,__half*out,__half*raw,
    const __half*res,const __half*scale,int M,int K,int N,int batches,int broadcast,
    int silu,int partition,int quantize,cudaStream_t stream,cublasHandle_t blas,
    __half*scratch_half,size_t scratch_half_elements=size_t(32)*1024*1024){
    using namespace blas_detail;
    if(!a||!w||!out||!blas||!scratch_half||M<=0||K<=0||N<=0||batches<=0||
       M%16||K%16||N%16||partition<0||partition%16||(res&&!scale)||
       (broadcast!=0&&broadcast!=1))throw std::invalid_argument("Invalid fp32browser GEMM arguments");
    const size_t as_long=size_t(K)*(broadcast?1:batches),os_long=size_t(N)*batches;
    if(as_long>INT_MAX||os_long>INT_MAX||size_t(M)>size_t(INT_MAX)/os_long)
        throw std::invalid_argument("fp32browser GEMM exceeds bounded strides/count");
    if(scratch_half==a||scratch_half==w||scratch_half==out||scratch_half==res||scratch_half==raw)
        throw std::invalid_argument("fp32browser scratch aliases a tensor");
    const size_t capacity=scratch_half_elements/2;
    const size_t zones=partition?2:1;
    const int tile_rows=int(std::min(size_t(M),capacity/(zones*size_t(N))))/16*16;
    if(tile_rows<16)throw std::invalid_argument("fp32browser FP32 scratch is too small");
    const int as=int(as_long),os=int(os_long),span=partition?std::min(partition,K):K;
    cudaStream_t selected; cublasPointerMode_t mode;
    check_blas(cublasGetStream(blas,&selected));check_blas(cublasGetPointerMode(blas,&mode));
    if(selected!=stream||mode!=CUBLAS_POINTER_MODE_HOST)
        throw std::invalid_argument("fp32browser requires caller stream and host scalars");
    check_blas(cublasSetMathMode(blas,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    float*scratch=reinterpret_cast<float*>(scratch_half);
    float*sum=partition?scratch+size_t(tile_rows)*N:nullptr;
    const float one=1.f,zero=0.f;
    for(int expert=0;expert<batches;++expert)for(int row=0;row<M;row+=tile_rows){
        const int rows=std::min(tile_rows,M-row);
        const unsigned blocks=unsigned((size_t(rows)*N+255)/256);
        const size_t target=size_t(row)*os+size_t(expert)*N;
        __half*out_tile=out+target;__half*raw_tile=raw?raw+target:nullptr;
        if(res){browser_blas_detail::seed<<<blocks,256,0,stream>>>(res+target,scale+size_t(expert)*N,scratch,rows,N,os);check_cuda(cudaGetLastError());}
        for(int start=0;start<K;start+=span){
            const int width=std::min(span,K-start);const bool first=start==0;
            check_blas(cublasGemmEx(blas,CUBLAS_OP_N,CUBLAS_OP_N,N,rows,width,
                &one,w+(size_t(expert)*K+start)*N,CUDA_R_16F,N,
                a+size_t(row)*as+(broadcast?0:size_t(expert)*K)+start,CUDA_R_16F,as,
                first&&res?&one:&zero,scratch,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            if(partition){browser_blas_detail::partial<<<blocks,256,0,stream>>>(scratch,sum,rows*N,first);check_cuda(cudaGetLastError());}
        }
        browser_blas_detail::publish<<<blocks,256,0,stream>>>(partition?sum:scratch,out_tile,raw_tile,rows,N,os,silu!=0,quantize!=0);
        check_cuda(cudaGetLastError());
    }
}
} // namespace nr
