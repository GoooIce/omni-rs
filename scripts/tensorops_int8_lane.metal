#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void lane_probe(
    device int* out [[buffer(0)]],
    device int8_t* src_a [[buffer(1)]],
    device int8_t* src_b [[buffer(2)]],
    uint simd [[thread_index_in_simdgroup]]) {
    auto ta = tensor(src_a, dextents<int, 2>{16, 16}, array<int, 2>{1, 16});
    auto tb = tensor(src_b, dextents<int, 2>{16, 16}, array<int, 2>{1, 16});
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto a = op.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
    auto b = op.get_right_input_cooperative_tensor<int8_t, int8_t, int>();
    auto c = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, int>();
    a.load(ta);
    b.load(tb);
    op.run(a, b, c);
    if (simd == 0u) {
        out[0] = int(a.get_capacity());
        out[1] = int(b.get_capacity());
    }
    for (uint slot = 0u; slot < 8u; ++slot) {
        out[2 + simd * 8u + slot] = int(a[slot]);
        out[2 + 256 + simd * 8u + slot] = int(b[slot]);
        out[2 + 512 + simd * 8u + slot] = c[slot];
    }
}
