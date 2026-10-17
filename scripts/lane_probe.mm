#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
int main() { @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;
    NSString* body = [NSString stringWithContentsOfFile:@"scripts/tensorops_int8_lane.metal" encoding:NSUTF8StringEncoding error:&error];
    if (!body) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLLibrary> library = [device newLibraryWithSource:body options:[MTLCompileOptions new] error:&error];
    if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"lane_probe"] error:&error];
    if (!pipeline) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLBuffer> out = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLBuffer> a = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    auto ap = (int8_t*)a.contents;
    auto bp = (int8_t*)b.contents;
    for (int i = 0; i < 256; i++) {
        ap[i] = (int8_t)((i % 16) - 8);
        bp[i] = (int8_t)((i / 16) == (i % 16) ? 1 : 0);
    }
    id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBuffer:b offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
        return 1;
    }
    int* p = (int*)out.contents;
    printf("cap a=%d b=%d\n", p[0], p[1]);
    int mismatch = 0;
    for (int i = 0; i < 256; i++) if (p[2 + i] != p[2 + 512 + i]) mismatch++;
    printf("c_vs_a_mismatch %d\n", mismatch);
    for (int lane = 0; lane < 4; lane++) {
        printf("A lane %d:", lane);
        for (int slot = 0; slot < 8; slot++) printf(" %3d", p[2 + lane * 8 + slot]);
        printf("\nC lane %d:", lane);
        for (int slot = 0; slot < 8; slot++) printf(" %3d", p[2 + 512 + lane * 8 + slot]);
        printf("\n");
    }
}}
