# 解除 int8 矩阵计算内核的冻结

状态：下文的实现禁令由 [ADR-0003](0003-faster-implementation-wins.md) 取代。解除 int8 冻结这一条仍然有效。

[ADR-0001](0001-w4a16-dequant-over-nax-int8.md) 把 int8 与 FP8 的 Neural Accelerator 计算冻成独立专项：MiniCPM-V 单流解码受显存带宽限制，自写 Metal 4 TensorOps 的 int8 内核只帮得上计算受限的 prefill。Qwen-Image-2.1-Turbo 的 Unsloth `INT8-ConvRot` 检查点不是这条路径。它的去噪器线性层是逐行对称 int8，权重处在 group 256 的 Hadamard 旋转基里，并带动态 int8 激活；一次去噪是大矩阵乘，不是单 token 解码。故解除 int8 矩阵计算内核的冻结。int8 matmul 用 Burn/CubeCL 内核实现，不走 Metal TensorOps；只有目标就是 ANE 时才可以使用 Metal TensorOps。MiniCPM-V 继续走 W4A16 融合解量化。FP8 计算内核仍然冻结。已有的 DiT FP16 TensorOps 路径不受这条约束影响。

## Considered Options

- **维持冻结，ConvRot 只在加载点反量化成 FP16（否决）**：权重能对上，但矩阵乘不再是 int8，检查点里的动态激活量化也用不上。
- **int8 与 FP8 计算内核一起解冻（否决）**：这次只打开 int8。FP8 仍要另一套激活量化和 dtype，stock cubecl 在 Metal 上的 MMA 仍只有 F16/BF16/F32。
- **GPU 上用 Metal TensorOps 做 int8 matmul（否决）**：那是 M5 GPU 的 MPP 路径，和已经在用的 FP16 TensorOps 相同入口。int8 内核优先、并且只走 Burn/CubeCL。
- **只解除 int8 矩阵计算内核，实现限定为 Burn/CubeCL（选定）**：ConvRot 在 group 256 Hadamard 旋转之后做 int8 矩阵乘。Metal TensorOps 仅当这次计算必须上 ANE 时才允许。

## Consequences

- Qwen-Image-2.1-Turbo 去噪器可以加载 `Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors`，并在 group 256 Hadamard 旋转后走 int8 matmul。
- 2026-10-17 的测量改写了「GPU 上不用 TensorOps」这一条。M5 上同一形状的 `matmul2d`（`int8 × int8 → int32`）GPU 时间约 0.17–0.94 ms，CubeCL 宽内核的孤立时间约 3.1–8.7 ms。256² 热机去噪从约 15.9 s 降到 1.25 s（整段 2.29 s）。Apple10 上这条点积是 CubeCL 内核，编译成 64×64 的 `matmul2d`，累加仍是 int32，权重仍不在加载点展开成 FP16。`OMNI_CONVROT_TENSOROPS=0` 退回标量 CubeCL 内核。既有 FP16 TensorOps 路径不改。ANE 仍未参与。
- 不把 MiniCPM-V 的官方 int4 检查点改成 int8 计算，也不因此重开 Hy-MT2、Confucius4-R2T2、Breeze-TTS-2 和音乐模型的量化。
- 验收看 int8 矩阵乘相对同一权重的 FP16 反量化参考是否保持约定误差，以及去噪步是否真的走 int8 内核而不是加载点展开。
