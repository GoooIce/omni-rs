// ConvRot int8 GEMM on Metal TensorOps: quantize, unpack, int8 matmul2d, scale.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef ROWS
#define ROWS 256
#define COLS 4096
#define INNER 4096
#endif

kernel void convrot_quantize(
    device float* x [[buffer(0)]],
    device int8_t* q [[buffer(1)]],
    device float* scale [[buffer(2)]],
    device int* sum [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tx [[thread_index_in_threadgroup]]) {
    threadgroup float peaks[256];
    threadgroup int partial[256];
    float peak = 0.0f;
    for (uint col = tx; col < INNER; col += 256u) {
        float value = x[row * INNER + col];
        peak = max(peak, fabs(value));
    }
    peaks[tx] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tx == 0u) {
        float maxp = 0.0f;
        for (uint i = 0u; i < 256u; ++i) maxp = max(maxp, peaks[i]);
        float row_scale = maxp / 127.0f;
        if (!(row_scale > 1.0e-30f)) row_scale = 1.0e-30f;
        scale[row] = row_scale;
        peaks[0] = row_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_scale = peaks[0];
    int acc = 0;
    for (uint col = tx; col < INNER; col += 256u) {
        float value = x[row * INNER + col] / row_scale;
        float magnitude = fabs(value);
        float rounded = floor(magnitude + 0.5f);
        float signed_value = value >= 0.0f ? rounded : -rounded;
        signed_value = clamp(signed_value, -128.0f, 127.0f);
        int code = int(signed_value);
        q[row * INNER + col] = int8_t(code);
        acc += code;
    }
    partial[tx] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tx == 0u) {
        int total = 0;
        for (uint i = 0u; i < 256u; ++i) total += partial[i];
        sum[row] = total;
    }
}

kernel void convrot_unpack(
    device int* packed [[buffer(0)]],
    device int8_t* out [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(COLS) * uint(INNER) / 4u) return;
    int word = packed[index];
    uint base = index * 4u;
    for (uint lane = 0u; lane < 4u; ++lane) {
        int byte = (word >> (lane * 8u)) & 255;
        if (byte >= 128) byte -= 256;
        out[base + lane] = int8_t(byte);
    }
}

kernel void qwen_tensorops_int8(
    device int8_t* x [[buffer(0)]],
    device int8_t* w [[buffer(1)]],
    device int* y [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    auto a = tensor(x, dextents<int, 2>{INNER, ROWS}, array<int, 2>{1, INNER});
    auto b = tensor(w, dextents<int, 2>{INNER, COLS}, array<int, 2>{1, INNER});
    auto d = tensor(y, dextents<int, 2>{COLS, ROWS}, array<int, 2>{1, COLS});
    auto ta = a.slice(0, int(group.y) * 64);
    auto tb = b.slice(0, int(group.x) * 64);
    auto td = d.slice(int(group.x) * 64, int(group.y) * 64);
    constexpr auto desc = matmul2d_descriptor(64, 64, int(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto accum = op.get_destination_cooperative_tensor<decltype(ta), decltype(tb), int>();
    op.run(ta, tb, accum);
    accum.store(td);
}

kernel void convrot_epilogue(
    device int* acc [[buffer(0)]],
    device float* x_scale [[buffer(1)]],
    device int* x_sum [[buffer(2)]],
    device float* w_scale [[buffer(3)]],
    device int* w_zero [[buffer(4)]],
    device float* y [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(COLS) || gid.y >= uint(ROWS)) return;
    int dot = acc[gid.y * uint(COLS) + gid.x];
    int centered = dot - x_sum[gid.y] * w_zero[gid.x];
    y[gid.y * uint(COLS) + gid.x] = float(centered) * x_scale[gid.y] * w_scale[gid.x];
}
