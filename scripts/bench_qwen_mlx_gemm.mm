// Reuse the upstream MLX GEMM and its architecture-dependent tile policy.
// Only this benchmark's host/ABI adapter is project code, no new Metal algorithm.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

struct Params {
    int m,n,k,lda,ldb,ldd,tn,tm;
    int64_t stride_a,stride_b,stride_d;
    int swizzle,k_iterations,batch_ndim;
};
static_assert(sizeof(Params)==72);
float sample(size_t index, uint32_t salt) {
    uint32_t x=uint32_t(index)+salt;
    x=(x^(x>>16))*0x7feb352d; x=(x^(x>>15))*0x846ca68b;
    return float((x^(x>>16))>>8)/16777216.f-.5f;
}
int main(int argc,char**argv) { @autoreleasepool {
    if(argc!=3)return 2;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(![device supportsFamily:MTLGPUFamilyApple10])return 2;
    NSError* error=nil;
    NSString* source[2]={
        [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error],
        [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error]};
    if(!source[0]||!source[1]){NSLog(@"%@",error);return 1;}
    NSString* arch=device.architecture.name;
    char devc=[arch characterAtIndex:arch.length-1];
    auto queue=[device newCommandQueue];
    const int shapes[][3]={{257,4096,4096},{1024,4096,4096},{1024,12288,4096},
        {1024,4096,12288},{4096,4096,4096},{4096,12288,4096},{4096,4096,12288}};
    for(const auto& shape:shapes) { @autoreleasepool {
        int m=shape[0],n=shape[1],k=shape[2];
        // Verbatim policy from MLX steel_matmul_regular_axpby_nax.
        int bm=128,bn=128,bk=512,wm=4,wn=4;
        if(devc=='s'||devc=='c'){bk=(k>=8192&&k>(m+n))?64:256;bm=64;wm=2;}
        else if(devc=='d'){if(n>m){bm=128;bn=64;wm=4;wn=2;}else{bm=64;bn=128;wm=2;wn=4;}}
        int tm=(m+bm-1)/bm,tn=(n+bn-1)/bn;
        int swizzle=tm<=3?0:1;
        if(devc=='s'||devc=='c')swizzle=2;
        else if(devc=='d')swizzle=m>n?0:2;
        Params params{m,n,k,k,k,n,tn,tm,int64_t(m)*k,int64_t(n)*k,int64_t(m)*n,swizzle,k/bk,1};
        auto a=[device newBufferWithLength:size_t(m)*k*2 options:MTLResourceStorageModeShared];
        auto b=[device newBufferWithLength:size_t(n)*k*2 options:MTLResourceStorageModeShared];
        auto c=[device newBufferWithLength:size_t(m)*n*2 options:MTLResourceStorageModeShared];
        auto ap=(__fp16*)a.contents,bp=(__fp16*)b.contents,cp=(__fp16*)c.contents;
        for(size_t i=0;i<size_t(m)*k;i++)ap[i]=sample(i,42);
        for(size_t i=0;i<size_t(n)*k;i++)bp[i]=sample(i,1234);
        double ms[2],errs[2],rels[2];
        for(int mode=0;mode<2;mode++) {
            NSString* macros=mode==0?[NSString stringWithFormat:@"#define ROWS %d\n#define COLS %d\n#define INNER %d\n",m,n,k]:
                [NSString stringWithFormat:@"#define OMNI_BM %d\n#define OMNI_BN %d\n#define OMNI_BK %d\n#define OMNI_WM %d\n#define OMNI_WN %d\n#define OMNI_ALIGN_M %d\n#define OMNI_ALIGN_N %d\n#define OMNI_ALIGN_K %d\n",bm,bn,bk,wm,wn,m%bm==0,n%bn==0,k%bk==0];
            auto library=[device newLibraryWithSource:[macros stringByAppendingString:source[mode]] options:[MTLCompileOptions new] error:&error];
            if(!library){NSLog(@"%@",error);return 1;}
            auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:mode==0?@"qwen_tensorops":@"omni_mlx_gemm"] error:&error];
            if(!pipeline){NSLog(@"%@",error);return 1;}
            std::vector<double> times;
            for(int run=0;run<12;run++) {
                auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:b offset:0 atIndex:1];
                if(mode==0){
                    [encoder setBuffer:c offset:0 atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake((n+63)/64,(m+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                }else{
                    [encoder setBuffer:c offset:0 atIndex:2];[encoder setBuffer:c offset:0 atIndex:3];
                    [encoder setBytes:&params length:sizeof(params) atIndex:4];
                    for(int i=5;i<8;i++)[encoder setBytes:&params length:sizeof(params) atIndex:i];
                    int tile=1<<swizzle;
                    [encoder dispatchThreadgroups:MTLSizeMake(tn*tile,(tm+tile-1)/tile,1) threadsPerThreadgroup:MTLSizeMake(32,wn,wm)];
                }
                [encoder endEncoding];[command commit];[command waitUntilCompleted];
                if(command.status==MTLCommandBufferStatusError){NSLog(@"%@",command.error);return 1;}
                if(run>=3)times.push_back(1000*(command.GPUEndTime-command.GPUStartTime));
            }
            std::sort(times.begin(),times.end());ms[mode]=times[times.size()/2];
            double maxerr=0,sqerr=0,sqref=0;
            for(int s=0;s<128;s++){
                int r=s==127?m-1:(s*997)%m,c=s==127?n-1:(s*619)%n;
                double sum=0;for(int i=0;i<k;i++)sum+=double(ap[size_t(r)*k+i])*double(bp[size_t(c)*k+i]);
                double ref=double((__fp16)sum),val=double(cp[size_t(r)*n+c]);
                if(!std::isfinite(val))return 1;
                maxerr=std::max(maxerr,std::abs(val-ref));sqerr+=(val-ref)*(val-ref);sqref+=ref*ref;
            }
            errs[mode]=maxerr;rels[mode]=std::sqrt(sqerr/std::max(sqref,1e-30));
            if(maxerr>.0625||rels[mode]>.001)return 1;
        }
        printf("{\"arch\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"tile\":[%d,%d,%d],\"previous_gpu_ms\":%.6f,\"mlx_gpu_ms\":%.6f,\"speedup\":%.4f,\"previous_cpu_max_abs\":%.8f,\"mlx_cpu_max_abs\":%.8f,\"mlx_cpu_relative_l2\":%.8f}\n",arch.UTF8String,m,n,k,bm,bn,bk,ms[0],ms[1],ms[0]/ms[1],errs[0],errs[1],rels[1]);
        fflush(stdout);
    }}
    return 0;
} }
