# Qwen-Image：Rust/CubeCL FlashAttention 与 M5 TensorOps

2026-10-12 至 2026-10-13。范围：Makepad 本地图片生成及编辑，FP16，8 步，不量化。

用户的要求是借鉴已有优化思路，在 Rust 中实现。当前注意力算法位于
`crates/qwen-image/src/flash_attention.rs`，使用 CubeCL 的 Rust GPU DSL；
`metal_attention.rs` 负责 Burn Fusion 的张量与执行队列适配。
生产构建不调用 MLX 运行时，不编译或包含 MLX shader。

## 采用的已有思路

采用 FlashAttention 的分块 QK/PV、在线 softmax 和寄存器累积。
M5 的 16×16 cooperative tensor 线程分布、64×32 注意力分块以及寄存器直接读取策略，
参考 Apple MLX 的公开 NAX 实现，重新用 Rust/CubeCL 表达；来源及 MIT 许可证保留在
`crates/qwen-image/THIRD_PARTY.md`。上游提交固定为
`388f227f466ef600243bcfce64181ad0d675c476`。

- [Apple MLX NAX 线程布局和矩阵调用](https://github.com/ml-explore/mlx/blob/388f227f466ef600243bcfce64181ad0d675c476/mlx/backend/metal/kernels/steel/attn/nax.h)
- [Apple MLX 分块注意力与在线 softmax](https://github.com/ml-explore/mlx/blob/388f227f466ef600243bcfce64181ad0d675c476/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h)
- [Apple Metal Performance Primitives 编程指南](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
- [CubeK 0.2.0 Rust 实现](https://docs.rs/crate/cubek-attention/0.2.0/source/src/routines/blackbox_accelerated.rs)

首先实际测试了现有 CubeK 0.2 的分块组合。其 Metal 路径采用 8×8 simdgroup matrix；
扩大分块没有得到收益，部分配置超出本机每线程组 32 KiB 共享内存限制。
这些探索记录在 `sweep-isolated.jsonl`，仅用于性能和资源筛选；其中早期的
`relative_l2=0` 是同一输出的自比较，**不能作为数值正确性证据**。
已修正保留的 `rust-attention-sweep` 工具，使其比较独立的 Burn fallback 输出。
正式数值验收使用后述 CPU 双精度 oracle。

## Rust 内核与编译器支持

- 每个线程组 128 线程、4 个 SIMD 组，处理 64 个查询和一块 32 个 K/V。
- QK 和 PV 通过 M5 16×32×16 TensorOps 计算。输入与输出 FP16，累积、softmax、行最大值和分母 FP32。
- Q/K/V 以 FP16 四元素向量读取，输出留在现有 CubeCL 缓冲中，不读回 CPU、不创建另一套运行时。
- 保留 B/H/L strides；不满足向量读取对齐或 D 不连续时，通过现有 CubeCL GPU copy 转成连续布局。
- 支持 Q/K 尾块、GQA、双 batch、因果偏移及缓存前缀，不通过补齐 Q 再裁剪规避边界。

CubeCL 0.10 缺少 M5 manual MMA 的 Metal 编译支持，因此 vendored
`cubecl-cpp 0.10.0` 中增加了一个受限的编译转换：Rust IR 的明确 RHS 布局、
16×32×16 FP16×FP16+FP32 指令转换成 Apple `mpp::tensor_ops::matmul2d`。
Metal 源码由 Rust 编译器生成，注意力算法、索引、mask、softmax 均在 Rust 内核中。
补丁说明见 `vendor/cubecl-cpp/OMNI_PATCHES.md`。

仅显式 manual MMA 内核生成 MPP include 和线程组上限属性。
没有扩大后端公布的 MMA 特性列表，普通 Burn 算子不会因此自动改用 M5 指令。
新路径只由 DiT 调用；文本塔、视觉塔、VAE、大矩阵既有路径、ASR/TTS 及 ANE 推理策略保持原状。
M5/Metal、FP16、D=128 和 shape 检查不通过时使用原有 Burn 路径。
启动前设置 `OMNI_QWEN_RUST_ATTN=0` 可禁用新注意力。

## 独立内核验证

正式测试 15 组：14 种 shape/layout/mask/数值场景和一组不支持类型的回退检查。
包括非对齐首块、矩形图、512/1024 图、双 batch/GQA、非零因果偏移、Q=1、
D 转置、非对齐 D 切片、对齐 Q/K、尖锐 softmax 和切换长度再次调用。
Q=1 的偏移 mask 单独与 CPU oracle 比较，避免复用旧策略跳过 mask 的行为。

CPU oracle 从已舍入的 FP16 输入进行 f64 softmax 运算，验收阈值沿用
相对 L2 < 0.003、最大绝对误差 < 0.04；所有输出同时检查 finite。
GPU worker panic 会使测试立即失败，并要求错误日志为空，避免失败后复用内存造成假通过。

关闭 profiling、两次预热后取五次中位数，包含提交与同步开销：

| 场景（32 heads，D=128） | 原 Burn | Rust/CubeCL | 比值 |
| --- | ---: | ---: | ---: |
| Q=1024，K=1043 | 4.625 ms | 2.809 ms | 1.65× |
| Q=4096，K=4115 | 39.169 ms | 15.097 ms | 2.59× |
| Q=1024，K=2048 | 6.208 ms | 2.790 ms | 2.23× |

主形状 CPU 相对 L2 约 0.00031；尖锐 softmax 的最大绝对误差 0.00390625。
早期共享内存原型的大形状约 31.7 ms；改成寄存器读取后约 20.1 ms，
四元素向量读取后约 15.1 ms。单独增加线程组上限没有测出明确收益。
这些是单算子数据，不能当作整图加速比。

历史 MLX shader 对照的大形状约 6 ms，当前 Rust 版仍有差距；两轮没有做锁频或功耗控制。
没有宣称已经达到其性能，也没有将 MLX shader 作为隐藏 fallback。

## 完整模型与原生界面验证

官方 FP16 对照：

| 场景 | PSNR |
| --- | ---: |
| 矩形 288×320 | 51.07 dB |
| 单参考图编辑 512² | 51.19 dB |
| 非空负向提示词、CFG=2、512² | 40.81 dB |
| 双参考图编辑、空负向词、CFG=2、256² | 46.52 dB |

均高于既有 35 dB 门槛。DiT 两步组件与官方的相对误差低于 0.3%；
第二步缓存与不缓存完全一致。文本塔仍使用已有计算路径，避免把注意力的舍入变化传播到提示词编码。

Makepad 回归覆盖尺寸预设、seed、非法 CFG 拒绝、负向分支进度、取消与恢复、
PNG 保存、透明参考图、多参考图与参考分辨率、关闭 CFG 后的分支行为及形状切换。
CLI 和界面的同一 CFG 请求逐字节一致；编辑后默认图片完全可复现。
取消恢复 0.058 秒；ASR 文件转录结果为“之前有顾客自己带酒水也没加收钱或者不让喝。”。
9 项 CPU 单元测试通过，WebGPU 仅进行编译回归，没有新增 Web 优化。

## Makepad 整图 A/B

基线为此前的 **Burn 注意力 + 大矩阵 TensorOps** 版本，即
`artifacts/qwen-image-upstream-attention/baseline/r2t2-makepad`，不是已归档的 MLX shader 版本。
本机 M5 Max、40 核 GPU、128 GB，固定提示词、seed=42、FP16、8 步。
每个独立进程依次运行 512²、1024²、512² CFG=2；每种请求一次预热、三次计时。
第一轮先基线后 Rust，第二轮交换顺序。没有同时运行本任务的编译或其他推理。
未锁定频率、功耗或桌面负载，因此保留全部样本与范围。

以下为两轮共六个热启动样本的中位数，排除模型加载和 PNG 保存：

| 场景 | 原 Burn | Rust/CubeCL | 耗时下降 | 全部样本范围：原版 / Rust |
| --- | ---: | ---: | ---: | --- |
| 512² | 4.699 s | 4.127 s | 12.2% | 4.419–5.017 / 3.825–4.308 s |
| 1024² | 27.788 s | 23.301 s | 16.1% | 25.658–34.696 / 20.327–25.872 s |
| 512²，CFG=2 | 10.432 s | 9.727 s | 6.8%，结论不稳定 | 8.927–12.252 / 8.765–10.005 s |

分轮中位数如下，避免合并结果掩盖 CFG 的方向变化：

| 场景 | 第一轮：原版 → Rust | 第二轮：原版 → Rust（实际先跑 Rust） |
| --- | ---: | ---: |
| 512² | 4.462 → 4.010 s | 4.868 → 4.202 s |
| 1024² | 25.926 → 21.091 s | 30.376 → 24.639 s |
| CFG=2 | 8.936 → 9.727 s | 12.181 → 9.046 s |

512² 和 1024² 两轮都改善。CFG 第一轮慢 8.9%，第二轮快 25.7%，
不能据此宣称稳定收益，也未确认退化的原因。
第一轮失败记录保留为 `parity-first-pass-failure.log`。
另做过只运行 CFG 的独立进程诊断，中位数原版 10.138 s、Rust 7.025 s；
这组诊断**不混入**上述整流程样本。
最终性能门槛采用两轮全部六个样本的中位数，仍要求每种场景不比基线慢 2% 以上；
通过这一门槛不等于证明 CFG 稳定加速。

512² 的累计进程 CPU 时间中位数为 0.965 → 0.855 s，1024² 为 2.180 → 2.065 s；
CFG 为 1.735 → 1.740 s。这里是累计 CPU 时间，不是 CPU 占用率。
报告中的 RSS 也不是 GPU 统一内存的峰值。

原始报告为 `benchmark-baseline`、`benchmark-final`、`benchmark-reverse-baseline` 和
`benchmark-reverse-rust` 目录中的 `report.json`；汇总与验收为 `parity-report.json`。
已验证产物安装到 `target/release/r2t2-makepad`、
`target/makepad-native/release/r2t2-makepad` 和 `target/release/qwen-image`。
产物与源码校验值、源码快照记录在 `delivery.json` 和 `source.tar.gz` 中。

## 复现

```sh
cargo build --release --locked --manifest-path crates/qwen-image/Cargo.toml \
  --target-dir target/qwen-image --bin qwen-image --example attention-bench
target/qwen-image/release/examples/attention-bench

cargo build --release --locked --manifest-path prototypes/makepad/Cargo.toml \
  --target-dir target/makepad-image
.venv-qwen-image/bin/python prototypes/makepad/validate-image-guidance.py \
  --binary target/makepad-image/release/r2t2-makepad \
  --output artifacts/qwen-image-rust-attention/benchmark-rerun --benchmark-only

.venv-qwen-image/bin/python scripts/check_qwen_image_rust_attention.py
```

实验、日志和报告存放于 `artifacts/qwen-image-rust-attention/`。
此前 MLX shader 适配源文件已移到 `artifacts/qwen-image-upstream-attention/vendor-mlx/`；
仅供历史对照，详情见 [已归档记录](qwen-image-upstream-attention.md)。
