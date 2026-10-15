// Standalone Metal runtime compiler smoke test. No Xcode metal tool required.
// clang++ -O2 -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
//   scripts/bench_qwen_tensorops.mm -o artifacts/qwen-image-tensorops/metal-smoke
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) { @autoreleasepool {
    if (argc < 2) return 2;
    int m = argc > 2 ? atoi(argv[2]) : 1024;
    int n = argc > 3 ? atoi(argv[3]) : 4096;
    int k = argc > 4 ? atoi(argv[4]) : 4096;
    int tm = argc > 5 ? atoi(argv[5]) : 64;
    int tn = argc > 6 ? atoi(argv[6]) : 64;
    if (m <= 0 || n <= 0 || k <= 0 || tm <= 0 || tn <= 0 ||
        m > 65535 || n > 12288 || k > 12288 || tm > 256 || tn > 256) {
        fprintf(stderr, "Expected positive bounded matrix and tile dimensions.\n");
        return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyApple10]) {
        fprintf(stderr, "Requires an Apple M5 GPU.\n");
        return 2;
    }
    NSError* error = nil;
    NSString* body = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    if (!body) { NSLog(@"%@", error); return 1; }
    NSString* source = [NSString stringWithFormat:@"#define ROWS %d\n#define COLS %d\n#define INNER %d\n#define TILE_M %d\n#define TILE_N %d\n%@", m,n,k,tm,tn,body];
    // The default language version matches wgpu's MSL passthrough path.
    MTLCompileOptions* options = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { NSLog(@"%@", error); return 1; }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"qwen_tensorops"] error:&error];
    if (!pipeline) { NSLog(@"%@", error); return 1; }
    auto a = [device newBufferWithLength:size_t(m)*k*2 options:MTLResourceStorageModeShared];
    auto b = [device newBufferWithLength:size_t(n)*k*2 options:MTLResourceStorageModeShared];
    auto c = [device newBufferWithLength:size_t(m)*n*2 options:MTLResourceStorageModeShared];
    auto ap = (__fp16*)a.contents;
    auto bp = (__fp16*)b.contents;
    auto cp = (__fp16*)c.contents;
    for (size_t i=0;i<size_t(m)*k;i++) ap[i] = float(int((i*31+11)%127)-63)/64;
    for (size_t i=0;i<size_t(n)*k;i++) bp[i] = float(int((i*17+3)%131)-65)/128;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::vector<double> times;
    for (int run=0;run<12;run++) {
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:c offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((n+tn-1)/tn,(m+tm-1)/tm,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) { NSLog(@"%@",command.error); return 1; }
        if (run>=3) times.push_back((command.GPUEndTime-command.GPUStartTime)*1000);
    }
    double maxerr=0, sqerr=0, sqref=0;
    // Sample rows/columns, explicitly include both last edges.
    for (int s=0;s<128;s++) {
        int row=s==127?m-1:(s*997)%m, col=s==127?n-1:(s*619)%n;
        float sum=0; for (int i=0;i<k;i++) sum+=float(ap[size_t(row)*k+i])*float(bp[size_t(col)*k+i]);
        float ref=float((__fp16)sum), out=cp[size_t(row)*n+col];
        if (!std::isfinite(out)) { fprintf(stderr,"non-finite output\n");return 1; }
        maxerr=std::max(maxerr,std::abs(double(out-ref))); sqerr+=(out-ref)*(out-ref);sqref+=ref*ref;
    }
    std::sort(times.begin(),times.end());
    printf("{\"device\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"tile\":[%d,%d],\"gpu_ms\":%.6f,\"max_abs\":%.8f,\"relative_l2\":%.8f}\n", device.name.UTF8String,m,n,k,tm,tn,times[times.size()/2],maxerr,sqrt(sqerr/std::max(sqref,1e-30)));
    return maxerr>0.125 ? 1 : 0;
} }
