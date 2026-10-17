// GPU-time probe for int8 TensorOps matmul2d. Same launch as bench_qwen_tensorops.mm.
// clang++ -O2 -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
//   scripts/bench_qwen_tensorops_int8.mm -o artifacts/qwen-image-tensorops/int8-bench
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) { @autoreleasepool {
    if (argc < 2) return 2;
    int m = argc > 2 ? atoi(argv[2]) : 256;
    int n = argc > 3 ? atoi(argv[3]) : 4096;
    int k = argc > 4 ? atoi(argv[4]) : 4096;
    int tm = argc > 5 ? atoi(argv[5]) : 64;
    int tn = argc > 6 ? atoi(argv[6]) : 64;
    const char* fn = argc > 7 ? argv[7] : "qwen_tensorops_int8";
    int threads = argc > 8 ? atoi(argv[8]) : 128;
    if (m <= 0 || n <= 0 || k <= 0 || tm <= 0 || tn <= 0 ||
        m > 65535 || n > 16384 || k > 16384 || tm > 256 || tn > 256) {
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
    MTLCompileOptions* options = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { NSLog(@"%@", error); return 1; }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@(fn)] error:&error];
    if (!pipeline) { NSLog(@"%@", error); return 1; }
    auto a = [device newBufferWithLength:size_t(m)*k options:MTLResourceStorageModeShared];
    auto b = [device newBufferWithLength:size_t(n)*k options:MTLResourceStorageModeShared];
    auto c = [device newBufferWithLength:size_t(m)*n*sizeof(int) options:MTLResourceStorageModeShared];
    auto ap = (int8_t*)a.contents;
    auto bp = (int8_t*)b.contents;
    auto cp = (int*)c.contents;
    for (size_t i = 0; i < size_t(m) * k; i++) ap[i] = int8_t(int((i * 31 + 11) % 251) - 125);
    for (size_t i = 0; i < size_t(n) * k; i++) bp[i] = int8_t(int((i * 17 + 3) % 241) - 120);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::vector<double> times;
    for (int run = 0; run < 12; run++) {
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:c offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((n + tn - 1) / tn, (m + tm - 1) / tm, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) { NSLog(@"%@", command.error); return 1; }
        if (run >= 3) times.push_back((command.GPUEndTime - command.GPUStartTime) * 1000);
    }
    double maxerr = 0;
    for (int s = 0; s < 32; s++) {
        int row = s == 31 ? m - 1 : (s * 997) % m;
        int col = s == 31 ? n - 1 : (s * 619) % n;
        int sum = 0;
        for (int i = 0; i < k; i++) sum += int(ap[size_t(row) * k + i]) * int(bp[size_t(col) * k + i]);
        int out = cp[size_t(row) * n + col];
        maxerr = std::max(maxerr, std::abs(double(out - sum)));
    }
    std::sort(times.begin(), times.end());
    double flops = 2.0 * m * n * k;
    double ms = times[times.size() / 2];
    printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"tile\":[%d,%d],\"gpu_ms\":%.4f,\"tflops\":%.2f,\"max_abs\":%.1f}\n",
           m, n, k, tm, tn, ms, flops / ms / 1.0e9, maxerr);
    return maxerr > 0 ? 1 : 0;
} }
