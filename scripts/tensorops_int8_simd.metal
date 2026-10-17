// CubeCL-style int8 GEMM: one SIMD group owns a 16×32 output tile and steps K by 16.
// The MMA is the same simdgroup matmul2d CubeCL already lowers for FP16 attention.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef ROWS
#define ROWS 256
#define COLS 4096
#define INNER 4096
#endif

kernel void qwen_tensorops_int8_simd(
    device int8_t* x [[buffer(0)]],
    device int8_t* w [[buffer(1)]],
    device int* y [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto full_x = tensor(x, dextents<int, 2>{INNER, ROWS}, array<int, 2>{1, INNER});
    auto full_w = tensor(w, dextents<int, 2>{INNER, COLS}, array<int, 2>{1, INNER});
    auto full_y = tensor(y, dextents<int, 2>{COLS, ROWS}, array<int, 2>{1, COLS});
    auto frag_x = op.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
    auto frag_w = op.get_right_input_cooperative_tensor<int8_t, int8_t, int>();
    auto acc = op.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(frag_x)>,
        metal::remove_addrspace_t<decltype(frag_w)>, int>();
    for (uint i = 0u; i < acc.get_capacity(); ++i) acc[i] = 0;
    int row0 = int(group.y) * 16;
    int col0 = int(group.x) * 32;
    for (int k = 0; k < INNER; k += 16) {
        frag_x.load(full_x.slice(k, row0));
        frag_w.load(full_w.slice(k, col0));
        op.run(frag_x, frag_w, acc);
    }
    acc.store(full_y.slice(col0, row0));
}
