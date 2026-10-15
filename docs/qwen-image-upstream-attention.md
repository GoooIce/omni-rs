# Qwen-Image：MLX 注意力对照实验（已归档）

2026-10-12 更新：用户明确要求借鉴已有思路、使用 Rust 实现，不能直接调用 MLX。
本页保留此前 shader 复用实验的来源与测量，**不再描述当前生产路径**。
当前实现见 [Rust/CubeCL FlashAttention](qwen-image-rust-attention.md)。
实验源代码、二进制与原始数据保存在 `artifacts/`；原版本源码快照是本目录报告所指的版本。

2026-10-12。Makepad 本地版，M5 Max / 128 GB，FP16，8 步；不量化。

## 选择已有实现

本轮按“先搜索、再复用、再实测”的顺序评估：

| 实现 | 调研与测试结论 |
| --- | --- |
| 当前 Burn 0.21 / CubeK 0.2 | 作为基线。对齐图像 attention 已融合；因果与非对齐查询使用显式 GPU matmul/softmax。 |
| CubeK 主线 | 已到 0.3，依赖 CubeCL 0.11 开发提交；不能直接替换当前依赖，整体升级会扩大到共享后端。 |
| MLX `steel_attention_nax` | 有完整 M5 TensorOps 注意力内核、尾块、因果偏移、GQA 和 stride 支持。复用内核，增加 Burn 调用适配。 |
| tract 的 MLX NAX 移植分支 | Issue #2548 说明作者尚缺 M5 实机验证。本项目直接固定 MLX 上游源码，并在本机验证。 |

上游固定为 MLX **`388f227f466ef600243bcfce64181ad0d675c476`**，MIT 许可。
文件、许可证、逐文件 SHA256、展开后的 shader SHA256 与适配清单位于
`artifacts/qwen-image-upstream-attention/vendor-mlx/UPSTREAM.json`。

来源：
- [MLX 原始 attention 内核](https://github.com/ml-explore/mlx/blob/388f227f466ef600243bcfce64181ad0d675c476/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h)
- [MLX 原始调度与参数 ABI](https://github.com/ml-explore/mlx/blob/388f227f466ef600243bcfce64181ad0d675c476/mlx/backend/metal/scaled_dot_product_attention.cpp)
- [CubeK 主线依赖](https://github.com/tracel-ai/cubek/blob/main/Cargo.toml)
- [tract M5 SDPA 实机验证议题](https://github.com/sonos/tract/issues/2548)

## 实际适配范围

只在 **DiT** 中使用 MLX 的 `D=128 / BQ=64 / BK=32 / WM=4 / WN=1` 内核。
输入输出 FP16，内部累积和 softmax 使用 FP32，保持上游实现。
文本塔、视觉塔、VAE 和 ASR/TTS 的运算路径保持原样。

适配层复用当前 CubeCL 缓冲和执行队列，不读回 Q/K/V，不使用另一套推理运行时。
直接传入 B/H/L strides，最后一维不连续时使用现有 Burn contiguous 操作。
没有重新设计 attention 算法，也没有通过补齐查询再裁剪来规避尾块。
源文件展开只处理 include、编译常量及可选参数绑定，原始内核文件完整保留。

M5/Metal、FP16、head_dim 和 shape 检查不通过时回到现有 Burn 路径。
启动时设置 `OMNI_QWEN_MLX_ATTN=0` 可回退到原有注意力实现。
前缀 K/V 的跨步复用和拼接策略没有改变。

## 为什么没有替换文本塔

全局替换的实验通过了独立注意力数值测试，但矩形完整图片与官方输出的 PSNR
只有 **27.01 dB**，低于项目已有的 **35 dB** 验收线。
只替换 DiT 后，同一请求恢复到 **39.18 dB**。
因此正式路径明确从 DiT 调用新内核，保留文本塔原有的 FP16 舍入行为。

多模态整体文本特征有既存的视觉误差累积：原基线相对误差约 3.10%，
全局替换实验约 3.15%；项目此前对这一组合采用 4% 阈值。
这与隔离文本塔的 2% 阈值不同，不能混用，也不能用单层测试替代整图验收。

## 瓶颈与第二优先级实验

在原版 1024² 请求中，CubeCL 记录的 GPU 内核累计时间约 22.99 秒：
融合注意力约 **10.06 秒**，现有 TensorOps GEMM 约 **8.56 秒**。
逐算子计时会改变提交方式；这一结果用于排序，正式延迟另以关闭 profiling 的原生界面测量。

另外复用了同一 MLX 提交的 `steel_gemm_fused_nax`，沿用其设备分块与 swizzle 策略。
独立 Metal GPU 计时得到以下结果，均通过 FP16 输入对应的 CPU 双精度抽样检查：

| M×N×K | 现有 GPU ms | MLX GPU ms | 现有/MLX |
| --- | ---: | ---: | ---: |
| 1024×4096×4096 | 1.183 | 0.646 | 1.83× |
| 1024×12288×4096 | 2.060 | 1.977 | 1.04× |
| 1024×4096×12288 | 1.913 | 2.321 | 0.82× |
| 4096×4096×4096 | 2.825 | 2.702 | 1.05× |
| 4096×12288×4096 | 8.438 | 8.973 | 0.94× |
| 4096×4096×12288 | 12.301 | 9.861 | 1.25× |

这不是 Burn 端到端测试，且存在退化形状，因此没有整体替换现有 GEMM。
后续若继续，应先验证按形状选择已有内核的 Burn 适配和整图收益。

## Makepad 整图对照

本轮重新测试原版二进制与候选二进制。相同中文茶壶提示词、seed 42、8 步、FP16，
每种模式在各自进程预热一次后取三次中位数，原版先运行。没有并行编译或其它本任务 GPU 推理。
未锁定系统桌面负载、温度、GPU 频率或功耗；这些数字使用本轮基线，不能与前一轮绝对时间混算。
时间包含界面 worker 完成与按钮恢复，不含模型加载和 PNG 保存。

| 场景 | 本轮原版 | MLX DiT | 加速比 | 每图 CPU 时间，原版 → 新版 |
| --- | ---: | ---: | ---: | ---: |
| 512²，CFG 关闭 | 5.35 s | **4.94 s** | 1.08× | 0.98 → 0.95 s |
| 1024²，CFG 关闭 | 32.67 s | **20.38 s** | 1.60× | 2.15 → 2.09 s |
| 512²，CFG=2 | 11.26 s | **9.09 s** | 1.24× | 1.78 → 1.66 s |

去噪阶段对应为 4.52 → 4.07 s、29.99 → 17.85 s、10.21 → 8.13 s。
新版整图三次范围为 4.872–5.023 s、20.379–20.476 s、9.016–9.129 s。
CPU 是进程累计 CPU 时间差，非瞬时占用率。CPU 改善较小，本轮主要缩短大图的 GPU 等待。
未测统一内存峰值或能耗；请求结束后的 RSS 不能作为统一内存峰值。

独立注意力测试涵盖 12 种 shape/layout/数值场景及一组不支持类型的回退检查，
包括双 batch、GQA、因果偏移、Q/K 尾块、对齐组合、切片 offset、最后一维不连续、
尖锐 softmax 和切换长度后再次调用。CPU 双精度抽样以已舍入的 FP16 输入计算。
GQA 是上游内核的独立测试范围，正式图像路径只在 DiT 调用它。

正式 CLI 的官方对照：矩形图 **39.18 dB**、单图编辑 **51.19 dB**、
非空负向 CFG **41.07 dB**；DiT 第二步缓存与不缓存的输出完全一致。
9 项 CPU 单元测试、WebGPU 编译检查均通过。

Makepad 完整回归通过：非空/空负向词、双参考图与参考尺寸、非法参数拒绝、
取消与恢复、PNG 保存、切换尺寸、编辑后默认生成完全可复现以及 ASR 文件转录。
双参考图与官方输出 **47.59 dB**；CLI 与界面的相同 CFG 请求逐字节一致。
负向分支取消到恢复按钮约 **0.058 秒**。ASR/TTS 及共享后端源文件未修改。

总验收报告：`artifacts/qwen-image-upstream-attention/parity-report.json`。
来源与可执行文件校验：同目录 `delivery.json`，原版备份在 `baseline/`。

## 复现与产物

```sh
# 用固定提交的 MLX checkout 重建 vendored 源码与校验清单
python3 scripts/import_mlx_attention.py artifacts/qwen-image-upstream-attention/mlx

# 下面的 attention-bench 构建仅适用于归档的 source.tar.gz；当前源码已改成 Rust 内核
cargo build --release --locked --manifest-path crates/qwen-image/Cargo.toml \
  --target-dir target/qwen-image --example attention-bench
target/qwen-image/release/examples/attention-bench

clang++ -O2 -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  scripts/bench_qwen_mlx_gemm.mm -o artifacts/qwen-image-upstream-attention/bench-mlx-gemm
artifacts/qwen-image-upstream-attention/bench-mlx-gemm \
  crates/qwen-image/src/tensorops.metal artifacts/qwen-image-upstream-attention/vendor-mlx/gemm_nax.metal

.venv-qwen-image/bin/python scripts/check_qwen_image_upstream_attention.py
```

原始日志、失败实验、正式组件/图片对照与界面基准保存在
`artifacts/qwen-image-upstream-attention/`。失败实验不作为正式性能或质量结果。
