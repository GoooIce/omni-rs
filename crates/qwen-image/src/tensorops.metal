// FP16 linear, output-major weights: [M,K] @ [N,K]^T.
// MPP tensor dimensions are ordered from fastest to slowest axis.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef ROWS
#define ROWS 1024
#define COLS 4096
#define INNER 4096
#endif
#ifndef TILE_M
#define TILE_M 64
#define TILE_N 64
#endif

kernel void qwen_tensorops(
    device half* x [[buffer(0)]],
    device half* w [[buffer(1)]],
    device half* y [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    auto a = tensor(x, dextents<int, 2>{INNER, ROWS}, array<int, 2>{1, INNER});
    auto b = tensor(w, dextents<int, 2>{INNER, COLS}, array<int, 2>{1, INNER});
    auto d = tensor(y, dextents<int, 2>{COLS, ROWS}, array<int, 2>{1, COLS});
    auto ta = a.slice(0, int(group.y) * TILE_M);
    auto tb = b.slice(0, int(group.x) * TILE_N);
    auto td = d.slice(int(group.x) * TILE_N, int(group.y) * TILE_M);
    constexpr auto desc = matmul2d_descriptor(TILE_M, TILE_N, int(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto accum = op.get_destination_cooperative_tensor<decltype(ta), decltype(tb), float>();
    op.run(ta, tb, accum);
    auto result = op.get_destination_cooperative_tensor<decltype(ta), decltype(tb), half>();
    #pragma clang loop unroll(full)
    for (uint i = 0; i < accum.get_capacity(); ++i) {
        if (accum.is_valid_element(i)) result[i] = half(accum[i]);
    }
    result.store(td);
}
