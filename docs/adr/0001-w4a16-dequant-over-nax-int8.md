# 量化推理采用 W4A16 融合解量化，暂不实现 NAX int8/fp8 计算内核

MiniCPM-V 的 W4A16 路线仍然是当前检查点的做法。int8 矩阵计算内核的冻结已由 [ADR-0002](0002-int8-matmul-kernels.md) 解除。FP8 以及其他实现不再因本决定被封禁，同一负载上谁更快由 [ADR-0003](0003-faster-implementation-wins.md) 决定。

MiniCPM-V-4.7 35B 的官方 int4 检查点（compressed-tensors W4A16、symmetric、group 128）与 Burn 0.21 的量化张量方案（Q4S + 分块 + 打包存储，matmul 融合器在加载点解量化为 FP16 计算）格式对应，直接加载即获得 int4 显存占用且无需自写内核；而 M5 的 int8/fp8 Neural Accelerator 计算只能通过自写 Metal 4 TensorOps 内核触达，且实测证据（Draw Things、llama.cpp/MLX 的生产路线）表明它只加速计算受限的 prefill，单流解码是显存带宽受限。故当时决定量化推理走 W4A16 融合解量化，NAX 量化计算冻结为独立优化专项（验收指标：prefill 吞吐提升）。

## Considered Options

- **官方 int4 → Burn 量化张量直载（选定）**：无零点 symmetric 分组格式恰好匹配 Burn 方案；存储与带宽收益直接到位。
- **int4 权重上变换 int8 + 激活 int8 + 自写 Metal 4 TensorOps 内核（当时冻结，int8 矩阵乘已由 ADR-0002 解冻）**：macOS 26.1+ 门槛、无 Rust 绑定、仅计算受限场景受益。这条不要求把 MiniCPM-V 的 int4 权重量化成 int8。
- **官方 FP8 检查点（W8A8，含逐 token 动态激活量化）**：作为存储格式被 int4 支配（38 GB vs 18.4 GB）；作为计算格式同样需要自写内核且多一套激活量化机器，与 macOS 27 的 Metal fp8 dtype 无关（stock cubecl 在 Metal 的 MMA 仅 F16/BF16/F32）。

## Consequences

- safetensors 加载器需从仅 F16/BF16/F32 扩展为支持 pack-quantized int4 + bf16 混合精度布局（35B int4 只量化 MoE 专家权重）。
- 未来若启动 NAX 专项，不推翻本路线：存储格式不变，只替换 matmul 内核实现。
