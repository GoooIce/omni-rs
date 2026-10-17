// int8 linear: [M,K] @ [N,K]^T, int32 accumulator. Same tile launch as tensorops.metal.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef ROWS
#define ROWS 256
#define COLS 4096
#define INNER 4096
#endif
#ifndef TILE_M
#define TILE_M 64
#define TILE_N 64
#endif

kernel void qwen_tensorops_int8(
    device int8_t* x [[buffer(0)]],
    device int8_t* w [[buffer(1)]],
    device int* y [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    auto a = tensor(x, dextents<int, 2>{INNER, ROWS}, array<int, 2>{1, INNER});
    auto b = tensor(w, dextents<int, 2>{INNER, COLS}, array<int, 2>{1, INNER});
    auto d = tensor(y, dextents<int, 2>{COLS, ROWS}, array<int, 2>{1, COLS});
    auto ta = a.slice(0, int(group.y) * TILE_M);
    auto tb = b.slice(0, int(group.x) * TILE_N);
    auto td = d.slice(int(group.x) * TILE_N, int(group.y) * TILE_M);
    constexpr auto desc = matmul2d_descriptor(TILE_M, TILE_N, int(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto accum = op.get_destination_cooperative_tensor<decltype(ta), decltype(tb), int>();
    op.run(ta, tb, accum);
    accum.store(td);
}
