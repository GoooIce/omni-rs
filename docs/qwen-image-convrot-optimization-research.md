# Qwen-Image INT8-ConvRot：下一步优化

调研日期：2026-10-17。目标机器：M5 Max、40 核 GPU、128 GB 统一内存、macOS 27。

2026-10-17 补充：按下面「需要重新决定」里的 `int8 × int8 → int32` `matmul2d` 做了实测，差距够大，已经接到生成路径。GPU 中位时间（预热后 9 次）：256×4096×4096 为 0.171 ms，256×12288×4096 为 0.943 ms，256×4096×12288 为 0.595 ms，抽样点积与 int32 参考一致。同一茶壶请求的热机第二次：去噪 1.38 s，整段 2.47 s（此前 CubeCL 热机去噪中位数 15.9 s、整段约 17 s）。第一次含着色器编译，去噪 3.74 s。`OMNI_CONVROT_TENSOROPS=0` 可退回 CubeCL。
范围是一次完整生成（文本编码 + 8 步去噪 + VAE），权重为 Unsloth
`Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors`。int8 矩阵乘保持
[ADR-0002](adr/0002-int8-matmul-kernels.md)：Burn/CubeCL、i32 累加、f32 缩放收尾。
Metal TensorOps 只在计算必须上 ANE 时才允许；本文不把 TensorOps 写成默认计划。
FP8 计算内核仍冻结，已有的 DiT FP16 TensorOps 路径不改。
本轮没有新的模型测速，下面的「算术」都是用已有单算子时间或形状推出来的，不是新的 profile。

## 结论与实施顺序

**先把 256² 去噪的 16 秒拆开，再决定要不要再改 int8 点积。**
孤立 GEMM 已经约 2.8 TFLOP/s，共享内存分块没有超过它。按调用次数把孤立时间加起来，
七个线性层只解释去噪的大约 10 秒，还剩大约 6 秒没有归属。
在这 6 秒定位之前，再写一版分块内核不会说明整段生成为什么还是 16 秒。

| 顺序 | 具体工作 | 对本模型的意义 | 验收 |
|---|---|---|---|
| 0 | 同进程预热一次后，再跑 3 次 256²。去噪内部分开记 int8 GEMM、Hadamard、注意力、其余 | 12.0 / 16.1 / 15.8 秒三次去噪差了 4 秒，单次不能当基线 | 中位数相对下面的 15.9 秒；各段之和能对上去噪总计 |
| 1 | 哪一段占去噪的大部分，就只动那一段。若仍是 int8 点积，在 `prototypes/int8-gemm` 里减少每字节的移位和符号扩展，数值通过后再端口到 `omni-kernel` | 当前宽内核的孤立吞吐停在约 3 TFLOP/s，分块没有再抬高 | 256×4096×4096 孤立 GEMM 低于 2 ms，且 3 次去噪中位数比 15.9 秒至少少 1 秒 |
| 2 | 跑一次 1024²，同一提示、种子、8 步、CFG 关闭，只记录分阶段时间 | M 从 256 增到 4096，线性层 FLOPs 约 ×16，注意力按 token 平方增长 | 有一次完整报告后再决定要不要动注意力。不拿这次时间去改 256² 的内核 |
| 3 | 去掉 Metal 路径上为 CPU 回退保留的 int8 主机副本 | 约 7 GB 常驻，不在这 16 秒的计算里 | 看 RSS，不把去噪时间当作这项的验收 |

需要另作决定的选项放在文末，不排进上表。

## 已经量到的

提示 `a red ceramic teapot`，256×256，seed 42，8 步，CFG 关闭，release，Metal。
`load_seconds` 在 `elapsed_seconds` 之外，约 20 秒，不含在下面的整段里。

上一版内核（每线程一个输出）热机第二次：文本 0.64 s，去噪 20.61 s，VAE 1.02 s，整段 22.27 s。

8 列、双累加器内核进入 `crates/omni-kernel/src/int8_matmul.rs` 之后：

| 次序 | 文本 | 去噪 | VAE | 整段 |
|---|---:|---:|---:|---:|
| 第 1 次 | 0.557 s | 12.003 s | 0.705 s | 13.266 s |
| 第 2 次 | 0.631 s | 16.059 s | 0.671 s | 17.361 s |
| 第 3 次 | 0.471 s | 15.819 s | 0.722 s | 17.013 s |

第 2、3 次去噪中位数 **15.9 s**，整段约 17 s。去噪仍约占整段的 93%。
第 1 次的 12.0 s 只是一个样本，温度和频率没有锁定，不作为验收基线。
[TensorOps 记录](qwen-image-tensorops.md) 同样写明未锁定温度，并以预热后的中位数验收。

`prototypes/int8-gemm` 里，量化一次、GEMM 在同一次 fused op 内重复，用差值去掉上传和回读。
FLOP 按 `2·M·K·N` 计。逐行量化在这些形状上单独计时约为 0。

| M×K×N | 每线程 1 输出 | 8 列、两组部分和 | 共享内存 16×32，权重转成 `[k_word, column]`，内层不展开 |
|---|---:|---:|---:|
| 256×4096×4096 | 6.01 ms，1.43 TFLOP/s | **3.08 ms，2.79 TFLOP/s** | 3.67 ms，2.34 TFLOP/s |
| 256×4096×12288 | 17.38 ms，1.48 TFLOP/s | **8.74 ms，2.95 TFLOP/s** | 9.82 ms，2.62 TFLOP/s |
| 256×12288×4096 | 16.72 ms，1.54 TFLOP/s | **8.74 ms，2.95 TFLOP/s** | 9.77 ms，2.64 TFLOP/s |

更早一版把 16×64 tile 的 K 循环完全展开，三个大形状都停在 **0.11 TFLOP/s**。
那一版能对上参考值，所以慢来自调度和寄存器，不是算错。展开这条分块不再试。

## 这 16 秒里，算术能解释多少

256² 的 latent token 数 M = (256/16)² = 256。DiT 32 层，宽度 4096，FFN 内维 12288。
每层 7 个 ConvRot 线性层：`to_q/k/v`、`to_out.0`、`gate`、`up`、`down`。
8 步 × 32 层 = 256 次 layer。用上表的孤立时间当每次调用的成本（算术，不是图内计时）：

| 投影 | 每次 | 每层次数 | 8 步合计 |
|---|---:|---:|---:|
| Q、K、V、out | 3.08 ms | 4 | 3.15 s |
| gate、up | 8.74 ms | 2 | 4.47 s |
| down | 8.74 ms | 1 | 2.24 s |
| 合计 | | | **9.9 s** |

15.9 − 9.9 ≈ **6 s** 还没有归属。可能落在图内 GEMM 比孤立计时慢、每次自定义 op 的提交，
或其他算子。Hadamard 和 256² 注意力的 FLOP 对不上这 6 秒，见下，所以不能先假设它们是主项。

Hadamard 与 Unsloth 一样，是旋转后再做线性层，不融进 int8 点积。
在线半边是 `x @ H`，H 为 256 阶、分块沿最后一维。
来源：Unsloth `rotate_convrot_activation` 先 `reshape` 再 `matmul`，`ConvRotLinear.forward` 再调用 `F.linear`
（[diffusion_convrot.py](https://github.com/unslothai/unsloth/blob/6f443b5ccd3f54762aa1b347c874cf3e307951ed/studio/backend/core/inference/diffusion_convrot.py)）。
本仓库对应 `crates/qwen-image/src/convrot.rs` 的 `rotate`：`[rows·groups, 256] @ H`。

按 `2·M·groups·256·256` 估算，一层 7 次旋转约 4.8 GFLOP，8 步 32 层约 **1.24 TFLOP**。
即便这块 FP32 矩阵乘只有 2 TFLOP/s，也大约 0.6 s，填不满那 6 秒。
把 256 点积塞进量化内核会重复同样的 FLOP，调度更差；Unsloth 也没有这么做。

256² 注意力更小。32 头、头维 128 时，一层 QK 与 AV 合计约 2 GFLOP，8 步约 0.55 TFLOP。
1024² 才是 M = 4096，QK 按 M² 增长，大约 ×256，那时候才值得单独看注意力。
现有 FlashAttention 已经在 FP16 路径上量过 1024² 去噪，见
[Rust 注意力记录](qwen-image-rust-attention.md)；那是另一条权重路径，不能当成 ConvRot 的时间。

## 内核为什么停在大约 3 TFLOP/s

权重块只有 16–48 MB。8 列内核把每个权重读了 M 次，共享内存 tile 把重复读降到约 M/16，
孤立时间却更差（3.67 ms 对 3.08 ms）。流量少了还变慢，说明大形状上的天花板在点积指令和累加延迟，
不在再读一遍权重。逐行量化已经单独接近 0，不再是对象。

当前点积把每个 `i32` 里的 4 个字节用移位、掩码再做符号扩展，然后做 i32 乘法
（`crates/omni-kernel/src/int8_matmul.rs` 的 `sext` / `dot4`）。
MSL 的 SIMD-group 矩阵类型列在规范 §2.4：`T` 只能是 `half`、`bfloat`（Metal 3.1 起）或 `float`，
形状只有 8×8。没有 int8 的 `simdgroup_*8x8`。
来源：[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) §2.4。
同一规范 §6.8 写明，新代码应改用 Tensor 与 Metal Performance Primitives，而不是再加 SIMD-group 矩阵。
本仓库的 CubeCL Metal 后端也只发出 `simdgroup_{type}8x8`，且只接受 8×8×8
（`vendor/cubecl-cpp/src/metal/dialect.rs` 的 `compile_wmma_fragment`）。
因此在 ADR 之内，下一步能试的是更短的字节扩展（例如把 `i32` 看成 4 个 `char` 再乘），
仍然是 ALU 上的 i32 点积，不是 `matmul2d`。试完必须同时看孤立 GEMM 和 3 次去噪中位数。

## 1024² 与 2048²

M 分别为 4096 和 16384。线性层 FLOPs 相对 256² 约 ×16 和 ×64。
把上面的 9.9 s 按 M 线性放大，是 **约 160 s 和约 630 s 的算术**，还没加注意力。
这不是预测，只说明 256² 上再挤 1 秒，回答不了 1024² 能不能用。
先按顺序 2 记一次真实去噪，再决定注意力和内核谁先动。

FP16 TensorOps 在另一份记录里，1024×4096×4096 的同步基准是 11.44 ms → 3.10 ms
（[TensorOps 记录](qwen-image-tensorops.md)）。那是 FP16，不是本检查点的 int8，
不能把 3.7 倍套到 ConvRot 上。它只说明：同样的投影形状，一旦允许走 TensorOps，
大 M 的单算子已经量到明显低于现在的 int8 ALU。

## 需要重新决定的选项

这些和 ADR-0002 的选定项冲突。默认计划不包含它们。

| 选项 | 一手来源里它能做什么 | 会越过的决定 |
|---|---|---|
| `matmul2d`，`char × char → int` | 规范 Table 7.3：Metal 4 起支持，累加类型是 `int`。Tech Talk 111432 称 macOS 26.4 增加了 4-bit 与 8-bit 整数 tensor，让量化模型使用 GPU 内的 Neural Accelerators。Neural Accelerator 在每个 shader core 里，和独立 ANE 不是同一块硬件 | ADR-0002 否决「GPU 上用 Metal TensorOps 做 int8 matmul」。现有 FP16 路径是单独的 MSL passthrough，不在 CubeCL 的 `simdgroup` 代码生成里 |
| 加载时把 ConvRot 权重展开成 FP16，再走已有 TensorOps | 现有 FP16 路径已覆盖这 7 个投影，256–65535 行、K/N 为 4096 或 12288 | ADR-0002 否决「只在加载点反量化成 FP16」。动态激活 int8 也不再参与计算 |
| 在 CubeCL 里为上述 `matmul2d` 增加代码生成 | 同上，只是入口从 passthrough shader 换成 Cube 内核 | 仍然是 TensorOps，同一条否决。另外不要改 cargo registry 里的 wgpu |

26.3 的 cooperative tensor 允许在 kernel 内做自定义反量化再送进 matmul
（同一则 Tech Talk）。那也是 TensorOps 路径上的融合，随上表第一行一起决定，不单独立项。

ANE 与这项无关。Tech Talk 把 GPU Neural Accelerator 和 ANE 分开讲；
[TensorOps 记录](qwen-image-tensorops.md) 已写明那次 FP16 实现没有用 ANE。
ADR 允许 TensorOps 的条件是「这次计算必须跑在 ANE 上」，当前去噪不满足。

## 来源

- [ADR-0002](adr/0002-int8-matmul-kernels.md)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) §2.4、§6.8、Table 7.3
- [Tech Talk 111432](https://developer.apple.com/videos/play/tech-talks/111432/) 转录：Neural Accelerator 位于 shader core；26.4 增加 4-bit / 8-bit 整数 tensor
- [Unsloth `diffusion_convrot.py` @ `6f443b5c`](https://github.com/unslothai/unsloth/blob/6f443b5ccd3f54762aa1b347c874cf3e307951ed/studio/backend/core/inference/diffusion_convrot.py)：`convrot_hadamard_v1`，group 256，在线 `x @ H` 与 `F.linear` 分开
- `crates/omni-kernel/src/int8_matmul.rs`、`crates/qwen-image/src/convrot.rs`、`vendor/cubecl-cpp/src/metal/dialect.rs`
- 已有测量记录：[TensorOps](qwen-image-tensorops.md)、[Rust 注意力](qwen-image-rust-attention.md)
