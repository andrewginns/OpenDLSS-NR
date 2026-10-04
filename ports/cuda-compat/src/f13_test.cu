// Reduced tests for the optional CUDA-core F13 diagnostic implementation.
// Primitive answers come from the pinned upstream numerics.bin FDP8 section.
#include "gemm_f13.cuh"
#include "attention_f13.cuh"
#include "../tests/fixtures/upstream_numeric.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
using H = __half;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Random {
    uint32_t state;
    uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
};
uint32_t u32(const unsigned char* p) { return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24; }
std::vector<uint16_t> fixture(const std::string& path) {
    std::ifstream input(path,std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open numeric fixture");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});
    if (bytes.size()<12 || std::memcmp(bytes.data(),"NRNUM001",8)) throw std::runtime_error("Bad fixture magic");
    size_t offset=12;
    for (uint32_t section=0;section<u32(bytes.data()+8);++section) {
        if (offset+8>bytes.size()) throw std::runtime_error("Short fixture header");
        const std::string tag(reinterpret_cast<const char*>(bytes.data()+offset),4);
        const size_t count=u32(bytes.data()+offset+4); offset+=8;
        if (count>bytes.size()-offset) throw std::runtime_error("Short fixture section");
        if (tag=="FDP8") {
            if (count!=32768) throw std::runtime_error("Unexpected FDP8 fixture size");
            std::vector<uint16_t> answer(count/2);
            for(size_t i=0;i<answer.size();++i) answer[i]=uint16_t(bytes[offset+i*2])|(uint16_t(bytes[offset+i*2+1])<<8);
            return answer;
        }
        offset+=count;
    }
    throw std::runtime_error("Missing FDP8 fixture");
}
H decode(unsigned byte) {
    // The model decoder maps both E4 NaN codes to signed zero.
    const float value=(byte&127)==127 ? ((byte&128)?-0.f:0.f) : num::e4m3ToF32(uint8_t(byte));
    return nr::halfRne(value);
}
// Independent nearest-value half oracle; no CUDA rounding intrinsic is used.
H nearest(double value) {
    static const std::vector<double> grid=[] {
        std::vector<double> values(0x7c00);
        for(size_t i=0;i<values.size();++i) values[i]=num::f16ToF32(uint16_t(i));
        return values;
    }();
    const uint16_t sign=std::signbit(value)?0x8000:0;
    if(std::isnan(value)) return nr::halfFromBits(sign|0x7e00);
    const double magnitude=std::abs(value);
    if(magnitude>=65520) return nr::halfFromBits(sign|0x7c00);
    const auto found=std::lower_bound(grid.begin(),grid.end(),magnitude);
    if(found==grid.end()) return nr::halfFromBits(sign|0x7bff);
    const size_t upper=found-grid.begin();
    if(!upper||*found==magnitude) return nr::halfFromBits(sign|uint16_t(upper));
    const size_t lower=upper-1;
    const double down=magnitude-grid[lower],up=grid[upper]-magnitude;
    const size_t chosen=down<up?lower:up<down?upper:(lower%2?upper:lower);
    return nr::halfFromBits(sign|uint16_t(chosen));
}
double number(H value) { return num::f16ToF32(nr::halfBits(value)); }
int exponent(double value,int lower) {
    int power=0; std::frexp(std::abs(value),&power);
    return std::max(power-1,lower);
}
H reference_step(const H* a,int as,const H* b,int bs,H incoming) {
    const double initial=number(incoming);
    if(!std::isfinite(initial)) return incoming;
    int maximum=initial==0?-21:exponent(initial,-14);
    for(int i=0;i<16;++i) {
        const double x=number(a[i*as]),y=number(b[i*bs]);
        if(x!=0&&y!=0) maximum=std::max(maximum,exponent(x,-6)+exponent(y,-6));
    }
    const double alignment=std::ldexp(1.0,13-maximum);
    int64_t units=int64_t(std::trunc(initial*alignment));
    for(int i=0;i<16;++i) units+=int64_t(std::trunc(number(a[i*as])*number(b[i*bs])*alignment));
    return nearest(std::ldexp(double(units),maximum-13));
}
double browser_step(const H*a,int as,const H*b,int bs,double incoming){
    int maximum=incoming==0?-21:exponent(incoming,-14);
    for(int i=0;i<16;++i){const double x=number(a[i*as]),y=number(b[i*bs]);if(x!=0&&y!=0)maximum=std::max(maximum,exponent(x,-6)+exponent(y,-6));}
    const double alignment=std::ldexp(1.,13-maximum);
    int64_t units=int64_t(std::trunc(incoming*alignment));
    for(int i=0;i<16;++i)units+=int64_t(std::trunc(number(a[i*as])*number(b[i*bs])*alignment));
    const double result=std::ldexp(double(units),maximum-13);
    if(double(float(result))!=result)throw std::runtime_error("Browser oracle unexpectedly exceeds exact FP32 range");
    return result;
}
H toward_zero(double value){
    H result=nearest(value);
    if(std::abs(number(result))>std::abs(value))result=nr::halfFromBits(nr::halfBits(result)-1);
    return result;
}
H activation(H input) {
    const double x=number(input),bounded=std::max(-4.0,std::min(4.0,x));
    H inner=nearest(-0.055908203125*std::abs(bounded)+0.447265625);
    H factor=nearest(bounded*number(inner)+0.89453125);
    return nearest(x*number(factor));
}
H publication(H value) { return decode(num::e4m3FromF16Bits(nr::halfBits(value))); }
H browser_publication(double value){
    const bool negative=value<0;
    const double mag=std::min(std::abs(value),448.);
    unsigned best=0;double distance=std::abs(mag-number(decode(0)));
    for(unsigned code=1;code<127;++code){const double d=std::abs(mag-number(decode(code)));if(d<distance||(d==distance&&!(code&1))){best=code;distance=d;}}
    return decode(best|(negative?128:0));
}
float browser_activation(H input){
    const float x=float(number(input)),bounded=std::max(-4.f,std::min(4.f,x));
    const float inner=std::fma(-0.055908203125f,std::abs(bounded),0.447265625f);
    const float polynomial=std::fma(bounded,inner,0.89453125f);
    return x*polynomial;
}
void same(const std::vector<H>& actual,const std::vector<H>& expected,const char* name) {
    if(actual.size()!=expected.size()) throw std::runtime_error("Wrong result size");
    for(size_t i=0;i<actual.size();++i) if(nr::halfBits(actual[i])!=nr::halfBits(expected[i])) {
        std::fprintf(stderr,"%s mismatch at %zu: %04x != %04x\n",name,i,nr::halfBits(actual[i]),nr::halfBits(expected[i]));
        throw std::runtime_error("F13 bit mismatch");
    }
}
template<class T> struct Device {
    T* data=nullptr;
    explicit Device(size_t size) { check(cudaMalloc(&data,size*sizeof(T))); }
    ~Device() { if(data) cudaFree(data); }
    void upload(const std::vector<T>& source) {
        check(cudaMemcpy(data,source.data(),source.size()*sizeof(T),cudaMemcpyHostToDevice));
        // Pageable H2D can return after host staging, before device DMA ends.
        // Tests use a nonblocking compute stream, so order it after this copy.
        check(cudaStreamSynchronize(nullptr));
    }
    void download(std::vector<T>& target) { check(cudaMemcpy(target.data(),data,target.size()*sizeof(T),cudaMemcpyDeviceToHost)); }
};
__global__ void primitive(const H* inputs,H* output,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) output[i]=nr::f13_detail::step16(inputs+i*33,1,inputs+i*33+16,1,inputs[i*33+32]);
}

int matrix_cases(bool gpu,cudaStream_t stream,bool browser=false) {
    Random random{0xc001d00du}; int cases=0;
    for(int experts: {1,2,3}) for(bool broadcast: {false,true})
    for(int partition: {0,32,48}) for(bool residual_enabled: {false,true}) {
        const int M=17,K=64,N=19,as=K*(broadcast?1:experts),os=N*experts;
        std::vector<H> a(M*as),w(experts*K*N),res(M*os),scale(os),expected(M*os),raw(M*os),actual(M*os),expected_raw(M*os);
        for(H& v:a) v=decode((random.next()%89)|((random.next()&1)<<7));
        for(H& v:w) v=decode((random.next()%89)|((random.next()&1)<<7));
        for(H& v:res) v=nearest((int(random.next()%129)-64)/16.0);
        for(int i=0;i<os;++i) scale[i]=nearest(i%2?-.75:1.25);
        const bool silu=residual_enabled,quant=partition!=48;
        for(int row=0;row<M;++row) for(int expert=0;expert<experts;++expert) for(int column=0;column<N;++column) {
            const int at=row*os+expert*N+column;
            H accumulator=residual_enabled?nearest(number(res[at])*number(scale[expert*N+column])):nearest(0);
            double wide=browser&&residual_enabled?number(res[at])*number(scale[expert*N+column]):number(accumulator);
            H total=nearest(0);
            double wide_total=0;
            for(int start=0;start<K;start+=16) {
                const H*left=a.data()+row*as+(broadcast?0:expert*K)+start;
                const H*right=w.data()+(expert*K+start)*N+column;
                if(browser)wide=browser_step(left,1,right,N,wide);
                else accumulator=reference_step(left,1,right,N,accumulator);
                if(partition&&((start+16)%partition==0||start+16==K)) {
                    if(browser)wide_total=start<partition?wide:double(float(wide_total+wide));
                    else total=start<partition?accumulator:nearest(number(total)+number(accumulator));
                    accumulator=nearest(0);
                    wide=0;
                }
            }
            if(partition)wide=browser?wide_total:number(total);
            H value=browser?toward_zero(wide):(partition?total:accumulator);
            if(browser){
                const float activated=silu?browser_activation(value):float(wide);
                if(silu)value=toward_zero(activated);
                expected_raw[at]=value;
                expected[at]=quant?browser_publication(silu&&number(value)==0?0.:activated):value;
            }else{
                if(silu)value=activation(value);
                expected_raw[at]=value;expected[at]=quant?publication(value):value;
            }
        }
        if(gpu) {
            Device<H> da(a.size()),dw(w.size()),dr(res.size()),ds(scale.size()),dout(actual.size()),draw(raw.size());
            da.upload(a);dw.upload(w);dr.upload(res);ds.upload(scale);
            auto launch=[&] { nr::nr_gemm_f13(da.data,dw.data,dout.data,draw.data,
                residual_enabled?dr.data:nullptr,residual_enabled?ds.data:nullptr,
                M,K,N,experts,int(broadcast),int(silu),partition,int(quant),stream,browser); };
            launch();check(cudaStreamSynchronize(stream));dout.download(actual);draw.download(raw);
            same(actual,expected,"GEMM output");same(raw,expected_raw,"GEMM raw");
            if(cases==0) {
                cudaGraph_t graph;cudaGraphExec_t executable;
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();
                check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
                check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
                dout.download(actual);same(actual,expected,"Captured GEMM");
                check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
            }
        }
        ++cases;
    }
    return cases;
}

int browser_witnesses(bool gpu,cudaStream_t stream){
    // Retained K16 precision; final RTZ; distinct raw-half/direct-E4 stores.
    const uint16_t raw_bits[3]={0x1000,0x3c00,0x3cbf};
    const float e4_values[3]={0.f,1.f,1.125f};
    for(int which=0;which<3;++which){
        std::vector<H>a(32,nearest(0)),w(32,nearest(0)),out(1),raw(1);
        a[0]=nearest(which==2?1.125:1);w[0]=nearest(1);
        if(which<2){a[1]=nearest(1./512);w[1]=nearest(which==0?.25:.375);}
        else{a[1]=nearest(.0625);w[1]=nearest(1);a[2]=nearest(-1./512);w[2]=nearest(.125);}
        if(which==0){a[16]=nearest(-1);w[16]=nearest(1);}
        double value=browser_step(a.data(),1,w.data(),1,0);
        value=browser_step(a.data()+16,1,w.data()+16,1,value);
        const std::vector<H>expected_raw{nr::halfFromBits(raw_bits[which])},expected{nearest(e4_values[which])};
        same(std::vector<H>{toward_zero(value)},expected_raw,"Browser literal half witness");
        same(std::vector<H>{browser_publication(value)},expected,"Browser literal E4 witness");
        if(gpu){Device<H>da(32),dw(32),dout(1),draw(1);da.upload(a);dw.upload(w);
            nr::nr_gemm_f13(da.data,dw.data,dout.data,draw.data,nullptr,nullptr,1,32,1,1,0,0,0,1,stream,true);
            check(cudaStreamSynchronize(stream));dout.download(out);draw.download(raw);
            same(raw,expected_raw,"Browser GPU literal half witness");same(out,expected,"Browser GPU literal E4 witness");
        }
    }
    // Independent literal partition witnesses. Narrowing the first partial
    // [1+2^-11,-1] or the running total [1,2^-11,-1] to half loses 2^-11.
    for(int which=0;which<2;++which){
        const int K=which?48:32;
        std::vector<H>a(K,nearest(0)),w(K,nearest(0)),out(1),raw(1);
        a[0]=nearest(1);w[0]=nearest(1);
        const int small=which?16:1,last=which?32:16;
        a[small]=nearest(1./512);w[small]=nearest(.25);
        a[last]=nearest(-1);w[last]=nearest(1);
        double total=0;
        for(int k=0;k<K;k+=16){const double partial=browser_step(a.data()+k,1,w.data()+k,1,0);total=k?double(float(total+partial)):partial;}
        const std::vector<H>expected{nr::halfFromBits(0x1000)};
        same(std::vector<H>{toward_zero(total)},expected,"Browser literal partition witness");
        if(gpu){Device<H>da(K),dw(K),dout(1),draw(1);da.upload(a);dw.upload(w);
            nr::nr_gemm_f13(da.data,dw.data,dout.data,draw.data,nullptr,nullptr,1,K,1,1,0,0,16,0,stream,true);
            check(cudaStreamSynchronize(stream));dout.download(out);draw.download(raw);
            same(raw,expected,"Browser GPU partition raw witness");same(out,expected,"Browser GPU partition output witness");
        }
    }
    return 5;
}

int attention_cases(bool gpu,cudaStream_t stream) {
    Random random{0xa77e11u};int cases=0;
    struct Case {int T,count,heads,first;bool prior,global;};
    for(const Case shape: {Case{64,3,2,1,true,false},Case{64,2,2,0,false,true},Case{128,2,2,0,false,true}}) {
        const int T=shape.T,count=shape.count;
        std::vector<H> q(count*T*32),k(q.size()),v(q.size()),p(count*T*T),prior(shape.heads*4096),reciprocal(count*T);
        std::vector<H> scores(p.size()),expected_scores(p.size()),output(v.size()),expected_output(v.size());
        for(H& x:q)x=decode((random.next()%73)|((random.next()&1)<<7));
        for(H& x:k)x=decode((random.next()%73)|((random.next()&1)<<7));
        for(H& x:v)x=decode((random.next()%89)|((random.next()&1)<<7));
        for(H& x:p)x=decode(random.next()%56);
        for(H& x:prior)x=nearest((int(random.next()%65)-32)/32.0);
        for(H& x:reciprocal)x=nearest((1+random.next()%32)/64.0);
        for(int batch=0;batch<count;++batch) for(int row=0;row<T;++row) {
            for(int column=0;column<T;++column) {
                H sum=shape.prior?prior[((shape.first+batch)%shape.heads)*4096+row*64+column]:nearest(0);
                for(int channel=0;channel<32;channel+=16)
                    sum=reference_step(q.data()+(batch*T+row)*32+channel,1,k.data()+(batch*T+column)*32+channel,1,sum);
                expected_scores[(batch*T+row)*T+column]=sum;
            }
            for(int column=0;column<32;++column) {
                H sum=nearest(0);
                for(int start=0;start<T;start+=16)
                    sum=reference_step(p.data()+(batch*T+row)*T+start,1,v.data()+(batch*T+start)*32+column,32,sum);
                if(shape.global)sum=nearest(number(sum)*number(reciprocal[batch*T+row]));
                expected_output[(batch*T+row)*32+column]=publication(sum);
            }
        }
        if(gpu) {
            Device<H> dq(q.size()),dk(k.size()),dv(v.size()),dp(p.size()),dprior(prior.size()),dr(reciprocal.size()),ds(scores.size()),dout(output.size());
            dq.upload(q);dk.upload(k);dv.upload(v);dp.upload(p);dprior.upload(prior);dr.upload(reciprocal);
            auto launch=[&] {
                nr::nr_qk_f13(dq.data,dk.data,shape.prior?dprior.data:nullptr,ds.data,T,shape.heads,shape.first,count,stream);
                nr::nr_pv_f13(dp.data,dv.data,dout.data,shape.global?dr.data:nullptr,T,count,int(shape.global),stream);
            };
            launch();check(cudaStreamSynchronize(stream));ds.download(scores);dout.download(output);
            same(scores,expected_scores,"QK F13");same(output,expected_output,"PV F13");
            if(cases==0) {
                cudaGraph_t graph;cudaGraphExec_t executable;
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();
                check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
                check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
                ds.download(scores);dout.download(output);
                same(scores,expected_scores,"Captured QK");same(output,expected_output,"Captured PV");
                check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
            }
        }
        ++cases;
    }
    return cases;
}
} // namespace

int main(int argc,char**argv) {
    try {
        bool host_only=false;std::string path="tests/fixtures/numerics.bin";
        for(int i=1;i<argc;++i) {
            if(std::string(argv[i])=="--host-only")host_only=true;
            else if(std::string(argv[i])=="--fixture"&&i+1<argc)path=argv[++i];
            else throw std::runtime_error("Usage: nr-sm75-f13-test [--host-only] [--fixture PATH]");
        }
        const auto bits=fixture(path); Random random{0x85ebca6bu};
        std::vector<H> inputs(bits.size()*33),expected(bits.size()),cpu(bits.size()),actual(bits.size());
        for(size_t row=0;row<bits.size();++row) {
            for(int i=0;i<32;++i)inputs[row*33+i]=decode(random.next()&255);
            uint16_t acc=random.next();if((acc&0x7c00)==0x7c00)acc&=0x7bff;
            inputs[row*33+32]=nr::halfFromBits(acc);expected[row]=nr::halfFromBits(bits[row]);
            cpu[row]=reference_step(inputs.data()+row*33,1,inputs.data()+row*33+16,1,inputs[row*33+32]);
        }
        same(cpu,expected,"CPU vs pinned FDP8");
        cudaStream_t stream=nullptr;
        if(!host_only) {
            cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,0));
            if(device.major!=7||device.minor!=5)throw std::runtime_error("F13 test requires SM75");
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            Device<H> source(inputs.size()),destination(actual.size());source.upload(inputs);
            primitive<<<unsigned((bits.size()+255)/256),256,0,stream>>>(source.data,destination.data,int(bits.size()));
            check(cudaGetLastError());check(cudaStreamSynchronize(stream));destination.download(actual);
            same(actual,expected,"GPU vs pinned FDP8");
        }
        const int cases=matrix_cases(!host_only,stream);
        const int browser_cases=matrix_cases(!host_only,stream,true);
        const int browser_literals=browser_witnesses(!host_only,stream);
        const int attention=attention_cases(!host_only,stream);
        if(stream)check(cudaStreamDestroy(stream));
        std::printf("{\"status\":\"pass\",\"gpu_executed\":%s,\"pinned_fdp8_cases\":%zu,\"matrix_cases\":%d,\"browser_matrix_cases\":%d,\"browser_literal_cases\":%d,\"attention_cases\":%d,\"graph_replay_checked\":%s}\n",
                    host_only?"false":"true",bits.size(),cases,browser_cases,browser_literals,attention,host_only?"false":"true");
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"F13 TEST FAILED: %s\n",error.what());return 1;
    }
}
