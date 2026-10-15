# Qwen-Image：M5 TensorOps FP16 大矩阵

2026-10-12，M5 Max 40 核 GPU、128 GB、macOS 27.0.1。

## 实现

在图像 crate 内新增 `metal_tensorops.rs` 和 `tensorops.metal`，以 Burn 自定义 Fusion Operation
调用 Metal Performance Primitives `matmul2d`。沿用 CubeCL 的缓冲、资源生命周期、流水调度、队列与编译缓存；
每次矩阵计算不读回 CPU、不创建第二条 Metal 队列，也不添加 GPU 等待。
非连续输入在原 GPU 上整理；权重直接读取 checkpoint 的 output-major 布局，不预转置或复制权重。

- 仅 DiT block 的 Q/K/V、attention 输出和 FFN gate/up/down 投影显式启用。
- 设备检查要求原生 MSL 路径及 `MTLGPUFamilyApple10`，本机已验证。旧设备使用原 Burn 算子。
- FP16 输入/输出、FP32 累加，关闭 relaxed precision；保留现有中间 FP16 舍入位置。
- 64×64 输出 tile、4 个 SIMD groups / 128 线程，MPP 动态 slice 处理首步前缀和尾部不足一块的行。
- 启用范围：256–65535 行，K/N 为 4096–12288 内 64 的倍数；非适用形状、dtype、后端或权重布局回退。
- 核缓存键包含 M/N/K 和线程数，避免不同长度共用错误的形状特化。

使用的是 M5 **GPU** 的 TensorOps API。独立 ANE 没有参与本次图像推理；未量化。
ASR、TTS、图像文本塔、视觉塔、VAE、attention 选择策略和停止检查保持原实现。

wgpu 29 的 Naga 编译路径有 Metal 3.2 上限，但本次使用的是 MSL passthrough 路径。
后者以系统编译器默认语言版本编译，在本机可使用 TensorOps，因而没有修改 wgpu / CubeCL 的全局语言版本。
也不需要安装 Xcode 的离线 `metal` 编译器：shader 通过系统 Metal runtime 编译。

实现依据：[Apple M5 GPU 讲解](https://developer.apple.com/videos/play/tech-talks/111432/)、
[MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)，
以及本机 SDK 的 `MPPTensorOpsMatMul2d.h` 与 `metal_cooperative_tensor` 实际接口。
SDK 示例注释中的 `get_mask` / `#pragma unroll full` 与当前接口不一致；实际采用 `is_valid_element` 和
`#pragma clang loop unroll(full)`。

## 实测结果

Makepad，固定中文茶壶描述、seed 42、8 步、FP16。每种配置在同一进程预热一次后测 3 次，
取中位数。测量时已结束编译，按旧版、新版顺序运行；未锁定温度、频率或功耗。
界面等待包含 worker 完成与按钮恢复，不含模型加载和 PNG 保存。

| 请求 | 原版等待 | TensorOps 等待 | 加速比 | 每图进程 CPU 时间，原版 → 新版 |
|---|---:|---:|---:|---:|
| 512²，CFG 关闭 | 13.57 s | **4.57 s** | **2.97×** | 2.17 → 0.87 s（−60%） |
| 1024²，CFG 关闭 | 48.28 s | **24.53 s** | **1.97×** | 2.31 → 2.10 s（−9%） |
| 512²，CFG=2 | 21.25 s | **8.35 s** | **2.54×** | 2.27 → 1.52 s（−33%） |

对应去噪阶段中位数：12.73 → 3.76 s、46.27 → 22.44 s、20.37 → 7.44 s。
新版三次界面等待范围分别为 4.567–4.578 s、24.508–24.570 s、8.335–8.392 s。
CPU 数据是 `ps` 的累计进程 CPU 时间差，包含界面和 worker，不等于瞬时 CPU 占用率。
大画布主要改善 GPU 等待，累计 CPU 时间下降较少。本轮没有测能耗或统一内存峰值；
报告中的 `rss_after_bytes` 只是请求完成时的进程 RSS 抽样。

这些结果来自 `benchmark-baseline/` 和 `benchmark-final/`，使用同一验证脚本。
`ui-baseline-full/` 是另一次功能回归，部分时间与构建重叠，**不用于上述性能对比**。

随机矩阵单算子结果，时间包含 Burn 提交和显式同步：

| M × K × N | 原 Burn | TensorOps | 加速比 |
|---|---:|---:|---:|
| 1024 × 4096 × 4096 | 3.16 ms | 1.56 ms | 2.02× |
| 1024 × 4096 × 12288 | 8.89 ms | 3.08 ms | 2.88× |
| 1024 × 12288 × 4096 | 9.24 ms | 3.08 ms | 3.00× |
| 4096 × 4096 × 4096 | 11.44 ms | 3.10 ms | 3.68× |
| 4096 × 4096 × 12288 | 40.64 ms | 11.68 ms | 3.48× |
| 4096 × 12288 × 4096 | 42.09 ms | 14.23 ms | 2.96× |

小形状并非都有上述幅度：257 × 4096 × 4096 的同步基准为 1.70 → 1.56 ms，约 1.09×。
所以最终收益以完整生成对照为准，不能拿最快的单算子倍数代表整张图。

## 数值与原生回归

固定官方 Diffusers revision 与 checkpoint 沿用 [接入记录](qwen-image-21.md)，Python oracle 的文本注意力
使用 eager，并注入与 Rust 相同的初始噪声。FP16 浮点求和顺序改变后不要求与原内核逐位一致。

- 随机矩阵与原 Burn 相对 L2 误差约 0.0025%–0.0034%；独立 CPU FP64 点积样本最大绝对误差约 1.24e−5。
- 两个 32 层 DiT 对照的速度输出相对误差约 0.145%–0.197%；两例 cached / uncached 输出逐位一致。
- 官方完整图片 PSNR：512² 非空负向 CFG **39.70 dB**、256² 双参考图空负向 CFG **48.33 dB**、
  512² 单图编辑 **51.18 dB**、288×320 矩形透明背景请求 **40.08 dB**。矩形案例覆盖每步都存在的 360 行尾块。
- 与上一版同请求默认图片的 PSNR 为 49.98 dB；新版 CLI 与 Makepad 图片逐字节一致；
  多图编辑、CFG 开关切换后重新生成默认请求，也逐字节一致。
- 原生参数校验、进度、保存、取消/恢复、双图编辑、1024² 尺寸切换通过；本次取消延迟约 **0.11 s**。
- 生成后切回 ASR 的官方 WAV 转录仍精确匹配。语音源代码、共享后端源代码与原有语音 CLI 哈希保持不变。
- 9 项图像单元测试、WebGPU 入口编译检查通过。其他平台保留原路径，未做那些设备的速度验收。

## 回退与复现

默认在符合条件的设备上启用。排查或做同一可执行文件 A/B 对照时，在启动前设置：

```sh
OMNI_QWEN_TENSOROPS=0 ./target/release/r2t2-makepad
OMNI_QWEN_TENSOROPS=0 ./target/release/qwen-image generate --prompt '一只红色陶瓷茶壶' --width 512 --height 512
```

矩阵基准同时执行原 Burn linear 与 TensorOps，包含整齐行数、257/1031 行尾块、offset、strided 输入、
row-major 权重回退，以及独立 CPU FP64 点积对照。每个形状预热 3 次、记录 9 次中位数；
时间包含 Burn 提交与同步，不能等同于独立 Metal command 的 GPU 时间。

```sh
cargo run --release --locked --manifest-path crates/qwen-image/Cargo.toml \
  --target-dir target/qwen-image --example tensorops-bench
python3 prototypes/makepad/validate-image-guidance.py \
  --binary target/release/r2t2-makepad --output /tmp/qwen-tensorops-ui --large-benchmark
.venv-qwen-image/bin/python scripts/check_qwen_image_tensorops.py
```

同一原生基准脚本的 `--benchmark-only` 模式依次测试 512²、1024²、512² CFG，各预热一次后采样 3 次。

官方 DiT 对照脚本增加 `--latent-size`，默认仍为 8；本次用 16，保证真实进入新 GEMM 路径。
文生图首步有 275 行，双参考图首步有 319 行，后续目标图像有 256 行。
正负 CFG 分支及完整图片另作端到端检查。具体工件与最终二进制由
`artifacts/qwen-image-tensorops/delivery.json` 记录。

后续仍可试验静态 extents、K 分块同步、Morton 遍历、融合 QKV / gate-up，再以端到端结果决定是否保留。
本轮先交付经验证的大矩阵路径；融合 attention、跨请求条件缓存和 ANE 整块实验是后续独立步骤。
后续 DiT attention 采用 [Rust/CubeCL 实现](qwen-image-rust-attention.md)；早期 [MLX shader 对照实验](qwen-image-upstream-attention.md) 已归档。
