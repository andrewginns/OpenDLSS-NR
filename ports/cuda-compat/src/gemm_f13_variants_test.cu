// Standalone exact-browser integer GEMM tests and benchmark.
#include "gemm_f13.cuh"
#include "attention_f13.cuh"
#include "gemm_f13_variants.cuh"
static int selected_variant=1;
static int group_repeats=1;

// Reuse the established independent FP64/exponent oracle and expert fixtures.
// Its browser GPU calls are redirected to this prototype only in this TU;
// the original test, production wrappers and canonical policies are unchanged.
namespace nr {
inline void integer_test_dispatch(const __half*a,const __half*w,__half*out,__half*raw,
    const __half*r,const __half*s,int M,int K,int N,int batches,int broadcast,int silu,
    int partition,int quantize,cudaStream_t stream,bool browser=false) {
    if (!browser) throw std::runtime_error("Integer test called canonical policy");
    nr_gemm_f13_variant(selected_variant,a,w,out,raw,r,s,M,K,N,batches,broadcast,silu,partition,quantize,stream);
}
}
#define nr_gemm_f13 integer_test_dispatch
#define main unused_original_f13_test_main
#include "f13_test.cu"
#undef main
#undef nr_gemm_f13

namespace {
struct GroupRecord { H a[16],b[16];float incoming,expected; };
static_assert(sizeof(GroupRecord)==72,"Independent group record schema");
std::vector<unsigned char> read_bytes(const std::string&path) {
    std::ifstream input(path,std::ios::binary);
    if(!input)throw std::runtime_error("Cannot read integer fixture: "+path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(input)),{});
}
void same_float(float actual,float expected,const char*label,size_t index) {
    if(nr::browser_detail::floatBits(actual)!=nr::browser_detail::floatBits(expected)){
        std::fprintf(stderr,"variant=%d %s[%zu] actual=%.9g/%08x expected=%.9g/%08x\n",selected_variant,label,index,actual,nr::browser_detail::floatBits(actual),expected,nr::browser_detail::floatBits(expected));
        throw std::runtime_error("Integer F13 retained-FP32 mismatch");
    }
}
template<class Packet> float group_cpu(const GroupRecord&record) {
    Packet a[16],b[16];
    for(int k=0;k<16;++k){a[k]=Packet(nr::f13_integer_detail::pack_operand(record.a[k]));b[k]=Packet(nr::f13_integer_detail::pack_operand(record.b[k]));}
    float av[16],bv[16];int am[16],bm[16],ae[16],be[16];
    for(int k=0;k<16;++k){av[k]=float(number(record.a[k]));bv[k]=float(number(record.b[k]));am[k]=nr::f13_integer_detail::mantissa(a[k]);bm[k]=nr::f13_integer_detail::mantissa(b[k]);ae[k]=int(a[k]>>8)-6;be[k]=int(b[k]>>8)-6;}
    if(selected_variant==4){uint32_t ac[16],bc[16];for(int k=0;k<16;++k){ac[k]=nr::f13_variants_detail::pack_compact(record.a[k]);bc[k]=nr::f13_variants_detail::pack_compact(record.b[k]);}return nr::f13_variants_detail::step_compact(ac,1,bc,1,record.incoming);}
    if(selected_variant==1)return nr::f13_variants_detail::step_float(av,1,ae,bv,1,be,record.incoming);
    if(selected_variant==2)return nr::f13_variants_detail::step_unpacked(am,1,ae,bm,1,be,record.incoming);
    if(selected_variant==3)return nr::f13_variants_detail::step_float_values(av,1,bv,1,record.incoming);
    return nr::f13_integer_detail::step16(a,1,b,1,record.incoming);
}
__global__ void group_gpu(const GroupRecord*source,float*output,int count,int variant) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){
        if(variant==4){uint32_t ac[16],bc[16];for(int k=0;k<16;++k){ac[k]=nr::f13_variants_detail::pack_compact(source[i].a[k]);bc[k]=nr::f13_variants_detail::pack_compact(source[i].b[k]);}output[i]=nr::f13_variants_detail::step_compact(ac,1,bc,1,source[i].incoming);return;}
        uint32_t a[16],b[16];
        for(int k=0;k<16;++k){a[k]=nr::f13_integer_detail::pack_operand(source[i].a[k]);b[k]=nr::f13_integer_detail::pack_operand(source[i].b[k]);}
        float av[16],bv[16];int am[16],bm[16],ae[16],be[16];
        for(int k=0;k<16;++k){av[k]=nr::halfToFloat(source[i].a[k]);bv[k]=nr::halfToFloat(source[i].b[k]);am[k]=nr::f13_integer_detail::mantissa(a[k]);bm[k]=nr::f13_integer_detail::mantissa(b[k]);ae[k]=int(a[k]>>8)-6;be[k]=int(b[k]>>8)-6;}
        if(variant==1)output[i]=nr::f13_variants_detail::step_float(av,1,ae,bv,1,be,source[i].incoming);
        else if(variant==2)output[i]=nr::f13_variants_detail::step_unpacked(am,1,ae,bm,1,be,source[i].incoming);
        else if(variant==3)output[i]=nr::f13_variants_detail::step_float_values(av,1,bv,1,source[i].incoming);
        else output[i]=nr::f13_integer_detail::step16(a,1,b,1,source[i].incoming);
    }
}
int group_fixtures(const std::string&directory,bool gpu,cudaStream_t stream) {
    const auto bytes=read_bytes(directory+"/f13-integer-groups.bin");
    if(bytes.size()!=4096*sizeof(GroupRecord))throw std::runtime_error("Wrong group fixture bytes");
    std::vector<GroupRecord>records(4096);std::memcpy(records.data(),bytes.data(),bytes.size());
    for(size_t i=0;i<records.size();++i){
        same_float(group_cpu<uint32_t>(records[i]),records[i].expected,"CPU32 independent groups",i);
        same_float(group_cpu<uint16_t>(records[i]),records[i].expected,"CPU16 independent groups",i);
    }
    if(gpu)for(int repeat=0;repeat<group_repeats;++repeat){
        Device<GroupRecord>input(records.size());Device<float>output(records.size());input.upload(records);
        group_gpu<<<16,256,0,stream>>>(input.data,output.data,int(records.size()),selected_variant);check(cudaGetLastError());check(cudaStreamSynchronize(stream));
        std::vector<float>result(records.size());output.download(result);
        for(size_t i=0;i<records.size();++i)same_float(result[i],records[i].expected,"GPU independent groups",i);
    }
    return int(records.size());
}
template<class T> std::vector<T> consume(const std::vector<unsigned char>&bytes,size_t&offset,size_t count) {
    if(count>(bytes.size()-offset)/sizeof(T))throw std::runtime_error("Short matrix fixture");
    std::vector<T>result(count);std::memcpy(result.data(),bytes.data()+offset,count*sizeof(T));offset+=count*sizeof(T);return result;
}
int full_fixtures(const std::string&directory,bool gpu,cudaStream_t stream) {
    int cases=0;
    for(int fixture=0;fixture<6;++fixture){
        char filename[64];std::snprintf(filename,sizeof(filename),"/f13-integer-gemm-%02d.bin",fixture);
        const auto bytes=read_bytes(directory+filename);if(bytes.size()<16)throw std::runtime_error("Short fixture header");
        const int M=u32(bytes.data()),K=u32(bytes.data()+4),N=u32(bytes.data()+8),partition=u32(bytes.data()+12);
        if(M<1||M>17||K<16||K>4096||K%16||N<1||N>19||partition<0||partition%16)throw std::runtime_error("Unexpected fixture dimensions");
        size_t offset=16;const auto a=consume<H>(bytes,offset,size_t(M)*K),w=consume<H>(bytes,offset,size_t(K)*N);
        const auto res=consume<H>(bytes,offset,size_t(M)*N),scale=consume<H>(bytes,offset,N);
        const auto expected_wide=consume<float>(bytes,offset,size_t(M)*N);
        const auto expected_raw=consume<H>(bytes,offset,size_t(M)*N);
        const auto codes=consume<unsigned char>(bytes,offset,size_t(M)*N);if(offset!=bytes.size())throw std::runtime_error("Trailing fixture bytes");
        std::vector<H>expected(M*N);for(int i=0;i<M*N;++i)expected[i]=decode(codes[i]);
        std::vector<uint32_t>packed_a(a.size()),packed_w(w.size());
        for(size_t i=0;i<a.size();++i)packed_a[i]=nr::f13_integer_detail::pack_operand(a[i]);
        for(size_t i=0;i<w.size();++i)packed_w[i]=nr::f13_integer_detail::pack_operand(w[i]);
        for(int row=0;row<M;++row)for(int col=0;col<N;++col){
            const int at=row*N+col;
            float value=float(number(res[at])*number(scale[col])),total=0;
            for(int k=0;k<K;k+=16){
                GroupRecord group{};group.incoming=value;
                for(int j=0;j<16;++j){group.a[j]=a[row*K+k+j];group.b[j]=w[(k+j)*N+col];}
                value=group_cpu<uint32_t>(group);
                if(partition&&((k+16)%partition==0||k+16==K)){total=k<partition?value:float(total+value);value=0;}
            }
            same_float(partition?total:value,expected_wide[at],"CPU independent matrix",at);
        }
        if(gpu){
            Device<H>da(a.size()),dw(w.size()),dr(res.size()),ds(scale.size()),out(expected.size()+32),raw(expected.size()+32);Device<float>wide(expected.size());
            da.upload(a);dw.upload(w);dr.upload(res);ds.upload(scale);
            std::vector<H>guarded(expected.size()+32,nr::halfFromBits(0x3555));out.upload(guarded);raw.upload(guarded);
            nr::nr_gemm_f13_variant(selected_variant,da.data,dw.data,out.data+16,raw.data+16,dr.data,ds.data,M,K,N,1,0,0,partition,1,stream,wide.data);
            check(cudaStreamSynchronize(stream));std::vector<H>actual(guarded.size()),actual_raw(guarded.size());out.download(actual);raw.download(actual_raw);
            for(int i=0;i<16;++i)if(nr::halfBits(actual[i])!=0x3555||nr::halfBits(actual[expected.size()+16+i])!=0x3555||nr::halfBits(actual_raw[i])!=0x3555||nr::halfBits(actual_raw[expected.size()+16+i])!=0x3555)throw std::runtime_error("Integer output guard changed");
            same(std::vector<H>(actual.begin()+16,actual.end()-16),expected,"GPU independent E4");
            same(std::vector<H>(actual_raw.begin()+16,actual_raw.end()-16),expected_raw,"GPU independent raw");
            std::vector<float>actual_wide(expected.size());wide.download(actual_wide);
            for(size_t i=0;i<actual_wide.size();++i)same_float(actual_wide[i],expected_wide[i],"GPU independent retained FP32",i);
        }
        ++cases;
    }
    return cases;
}

int overlap_guards() {
    H*a=reinterpret_cast<H*>(uintptr_t(0x10000)),*w=reinterpret_cast<H*>(uintptr_t(0x20000));
    H*r=reinterpret_cast<H*>(uintptr_t(0x30000)),*s=reinterpret_cast<H*>(uintptr_t(0x40000));
    int rejected=0;
    for(int which=0;which<7;++which){
        H*out=reinterpret_cast<H*>(uintptr_t(0x50000)),*raw=reinterpret_cast<H*>(uintptr_t(0x60000));
        float*wide=nullptr;
        if(which==0)out=a+1;else if(which==1)out=w+1;else if(which==2)raw=out+1;
        else if(which==3)raw=a+1;else if(which==4)out=s;else if(which==5)wide=reinterpret_cast<float*>(r);else out=r+1;
        try{nr::nr_gemm_f13_variant(selected_variant,a,w,out,raw,r,s,16,16,16,1,0,0,0,1,nullptr,wide);}
        catch(const std::invalid_argument&){++rejected;continue;}
        throw std::runtime_error("Overlap guard reached CUDA");
    }
    return rejected;
}

__global__ void initialize_e4(H*values,size_t count,unsigned salt) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)values[i]=nr::e4m3ToHalf(uint8_t(((i*17+salt)%89)|((i&1)<<7)));
}
template<class F> std::vector<float> timings(F launch,cudaStream_t stream){
    constexpr int count=50;
    for(int i=0;i<20;++i)launch();check(cudaStreamSynchronize(stream));
    cudaEvent_t start[count],end[count];
    for(int i=0;i<count;++i){check(cudaEventCreate(&start[i]));check(cudaEventCreate(&end[i]));}
    for(int i=0;i<count;++i){check(cudaEventRecord(start[i],stream));launch();check(cudaEventRecord(end[i],stream));}
    check(cudaEventSynchronize(end[count-1]));std::vector<float>ms(count);
    for(int i=0;i<count;++i){check(cudaEventElapsedTime(&ms[i],start[i],end[i]));check(cudaEventDestroy(start[i]));check(cudaEventDestroy(end[i]));}
    std::sort(ms.begin(),ms.end());return ms;
}
std::string benchmarks(cudaStream_t stream) {
    const int shapes[][3]={{131072,32,128},{983040,32,96},{131072,128,32}};
    std::string report="[";
    for(int id=0;id<3;++id){
        const int M=shapes[id][0],K=shapes[id][1],N=shapes[id][2];const bool residual=id==2,silu=id==0,quant=id!=1;
        Device<H>a(size_t(M)*K),w(size_t(K)*N),out(size_t(M)*N),res(size_t(M)*N),scale(N);
        auto fill=[&](H*p,size_t n,unsigned salt){initialize_e4<<<unsigned((n+255)/256),256,0,stream>>>(p,n,salt);check(cudaGetLastError());};
        fill(a.data,size_t(M)*K,1);fill(w.data,size_t(K)*N,7);fill(res.data,size_t(M)*N,11);fill(scale.data,N,19);check(cudaStreamSynchronize(stream));
        auto integer=[&]{nr::nr_gemm_f13_variant(selected_variant,a.data,w.data,out.data,nullptr,residual?res.data:nullptr,residual?scale.data:nullptr,M,K,N,1,0,int(silu),0,int(quant),stream);};
        auto original=[&]{nr::nr_gemm_f13(a.data,w.data,out.data,nullptr,residual?res.data:nullptr,residual?scale.data:nullptr,M,K,N,1,0,int(silu),0,int(quant),stream,true);};
        const auto fast=timings(integer,stream),base=timings(original,stream);const double fm=(fast[24]+fast[25])/2.,bm=(base[24]+base[25])/2.;
        char row[1000];std::snprintf(row,sizeof(row),"%s{\"m\":%d,\"k\":%d,\"n\":%d,\"warmups\":20,\"timed\":50,\"timing\":\"CUDA events around uncaptured call\",\"integer_ms\":{\"median\":%.6f,\"p95\":%.6f},\"original_f13_ms\":{\"median\":%.6f,\"p95\":%.6f},\"median_speedup\":%.6f}",id?",":"",M,K,N,fm,fast[47],bm,base[47],bm/fm);report+=row;
    }
    return report+"]";
}
}

int main(int argc,char**argv){try{
    bool host=false,bench=false;int single=-1;std::string directory="tests/fixtures/f13";
    for(int i=1;i<argc;++i){const std::string option=argv[i];if(option=="--host-only")host=true;else if(option=="--benchmark")bench=true;else if(option=="--fixtures"&&i+1<argc)directory=argv[++i];else if(option=="--group-repeats"&&i+1<argc){group_repeats=std::stoi(argv[++i]);if(group_repeats<1||group_repeats>100)throw std::runtime_error("group-repeats must be1..100");}else if(option=="--variant"&&i+1<argc){const std::string value=argv[++i];single=value=="all"?-1:std::stoi(value);if(single<-1||single>4)throw std::runtime_error("Variant must be all,0,1,2,3,4");}else throw std::runtime_error("Usage: nr-sm75-f13-variants-test [--host-only | --benchmark] [--fixtures DIR] [--variant all|0|1|2|3|4] [--group-repeats N]");}
    if(host&&bench)throw std::runtime_error("Host-only mode cannot run GPU benchmark");
    cudaStream_t stream=nullptr;
    if(!host){cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0));if(properties.major!=7||properties.minor!=5)throw std::runtime_error("Requires SM75");check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    std::string report="[";
    const char*names[]={"packed_integer","fp32_predecoded_exponent","unpacked_integer","fp32_values","fp32_compact_exponent"};
    for(int variant=single<0?1:single;variant<=(single<0?4:single);++variant){
        selected_variant=variant;
        const int overlap_cases=overlap_guards();
        const int groups=group_fixtures(directory,!host,stream),full=full_fixtures(directory,!host,stream);
        const int cases=matrix_cases(!host,stream,true),literals=browser_witnesses(!host,stream);
        const std::string benchmark_report=bench?benchmarks(stream):"";
        char row[700];std::snprintf(row,sizeof(row),"%s{\"variant\":%d,\"name\":\"%s\",\"independent_groups\":%d,\"independent_full_matrices\":%d,\"expert_matrix_cases\":%d,\"literal_cases\":%d,\"overlap_guard_cases\":%d",report.size()>1?",":"",variant,names[variant],groups,full,cases,literals,overlap_cases);report+=row;
        if(bench)report+=",\"benchmarks\":"+benchmark_report;report+="}";
    }
    if(stream)check(cudaStreamDestroy(stream));
    std::printf("{\"status\":\"pass\",\"gpu_executed\":%s,\"output_guards_checked\":%s,\"graph_replay_checked\":%s,\"upload_ordering\":\"default stream H2D completed before nonblocking stream use\",\"group_gpu_repeats\":%d,\"runtime_integrated\":false,\"variants\":%s]}\n",host?"false":"true",host?"false":"true",host?"false":"true",host?0:group_repeats,report.c_str());return 0;
}catch(const std::exception&error){std::fprintf(stderr,"F13 VARIANTS FAILED variant=%d: %s\n",selected_variant,error.what());return 1;}}
