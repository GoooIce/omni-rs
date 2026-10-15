# Breeze-TTS-2：Metal depth 小算子优化

2026-10-11，Apple M5 Max / 128 GiB。在上一轮 [GPU 常驻和流式播放优化](breeze-tts-streaming-optimization.md)
之上，为 Makepad 本地版的 depth decoder 增加专用 GEMV 与 RMSNorm，并将组合路径设为默认。
没有尝试权重量化；继续使用 Burn 0.21 / CubeCL 0.10 和原来的生成器 FP16、敏感边界与 codec FP32。

## 采用的实现

- GEMV：一个 32 线程 SIMD group 计算一个输出通道，FP32 累加、FP16 输出；直接使用原有
  输出通道连续的权重布局。仅在 depth 的无 bias、FP16、1–4 行、输出宽度不超过 1024 的投影中启用。
  大输出矩阵继续使用 Burn 原算子，没有加载时重排或复制全部权重。
- RMSNorm：将归约、归一化、权重缩放合成一个内核。保留 FP32 归约，以及归一化结果和 gamma
  分别转换 FP16、再相乘的舍入位置。残差相加继续走原来的 FP32 边界。
- 通过 Burn Fusion 的自定义操作注册，输入保持原始维数，不增加 CPU 回读或同步屏障。
  非支持后端、精度、形状或布局自动使用原算子；两个内核均检查运行时的 32 线程 SIMD group 条件。
- 仅修改 TTS depth。文本编码器、主生成器、codec、ASR / ANE 推理和 Web 没有切换到新内核。

实现：[metal_depth.rs](../crates/breeze-tts/src/metal_depth.rs)、
[算子选择](../crates/breeze-tts/src/ops.rs)、[默认加载](../crates/breeze-tts/src/lib.rs)。
`DepthKernelMode` 和 CLI 的 `--depth-kernels stock|gemv|norm|gemv_norm` 保留单项对照能力；
正常 Makepad 使用无需设置参数。

## 完整生成对照

同一进程、同一个已加载模型，原实现与组合内核交错运行，每轮轮换顺序。普通中文和中文声音描述
（CFG 4）各做 6 次，每次 128 帧 / 10.24 秒音频，舍去第 0 次。文本、seed 42、两帧分块、
模型和依赖版本相同。测量期间没有并行编译、模型任务或 UI 验收；系统其他进程未作干预。

| 场景 | 原实现热中位 RTF | 组合热中位 RTF | 配对生成耗时减少 | 配对 CPU 时间减少 | 包间隔 p95 |
|---|---:|---:|---:|---:|---:|
| 普通合成 | 1.395 | 1.271 | 6.5% | 12.2% | 264 → 233 ms |
| 声音描述 | 1.612 | 1.478 | 8.1% | 9.4% | 306 → 283 ms |

百分比按同轮 `candidate / stock` 的五个比值取中位数计算，与分别取中位 RTF 后再相除不同。
普通合成五轮都更快，声音描述四轮更快、一轮慢 2.6%。CPU 是进程累计 user + system 秒数，
包括该请求的结果/WAV 写出和很小的管道采样延迟，不代表瞬时 CPU 占用率降低了同样比例。
macOS CPU 计数使用 Mach timebase 转换，并与本进程 `getrusage` 交叉核对。

组合的首个 PCM 包生成时间中位数为 261 / 340 ms；这是生成首包，设备首声还要等待播放缓冲。
两种场景仍然慢于实时，包间隔也大于两帧音频的 160 ms。**持续 RTF ≤ 0.8 的目标尚未达到。**
固定帧 benchmark 不用于证明长段 EOS 或播放连续性。

原始记录：[交错运行](../artifacts/tts-gemv-opt/confirm)、
[配对统计](../artifacts/tts-gemv-opt/confirm-summary.json)、
[CPU 计数](../artifacts/tts-gemv-opt/confirm.cpu.json)。

这组吞吐数据来自筛选时的组合候选二进制。交付时清理了未采用的展平投影开关，并补充非 32
线程 SIMD group 的 RMS 回退；本机选中内核的算术与适用形状没有改变。最终默认二进制另做了
下述数值和原生设备回归。筛选二进制及其哈希保存在 `artifacts/tts-gemv-opt/screen-implementation/`。

## 筛选与修正

长段同步诊断中，depth 前 64 帧、中段、后 64 帧的中位耗时约为 104.87 / 104.83 / 104.51 ms，
backbone 为 15.68 / 16.51 / 16.92 ms；没有观察到 depth 耗时随长段明显上升。
强制同步会改变融合和调度，因此这里只用于定位主要成本，不与正常吞吐直接比较。
记录：[阶段统计](../artifacts/tts-gemv-opt/long-stage-summary.json)。

按实际三维输入布局做独立微基准：8192 → 1024 的投影部分形状约快 1.6–1.95 倍，
但 GEMV 单项在整模型短测中只有约 1%–5% 改善，且有波动。将大输出投影展平为二维的候选
未显示稳定整体收益，已从生产选择中删除。32 帧短测也不足以决定默认方案，最终采用上面的
128 帧、五轮热运行对照。
记录：[实际布局微基准](../artifacts/tts-gemv-opt/gemv-3d-summary.json)、
[单项筛选](../artifacts/tts-gemv-opt/rank-screen-paired-summary.json)、
[组合筛选](../artifacts/tts-gemv-opt/norm-screen-paired-summary.json)。

最初在进入自定义 GEMV 前把三维张量展平，触发了 Burn 0.21 融合归约的索引越界。
现改为保留操作边界处的原始维数，仅在内核索引中处理行号，并增加归约后接 GEMV 的回归检查。
没有修改共享 Burn 或 ASR 来规避这个问题。失败日志和未采用实现保留在
`artifacts/tts-gemv-opt/`，不计作通过。

## 数值验证

10 项 TTS 单元测试（含两项真实 Metal 内核测试）通过：GEMV 对照 CPU FP64 点积，覆盖非连续
输入、尾部通道不足四个、不同权重布局的回退；RMSNorm 对照独立 FP64 计算与 FP16 舍入，
并检查 FP32 输入回退和不使用专用归约内核时的结果。

组合候选的中文 teacher-forced 检查中，depth 全部 logits 的相对 RMSE 约 0.00155，
低于既有 0.005 门限；文本编码、主生成器、音频 embedding 和后续主生成器 logits 与基线一致。
归约顺序发生变化，不能声称逐位一致；随机生成也可能改变后续码字和完整音频长度。
记录：[中文数值对照](../artifacts/tts-gemv-opt/gemv_norm-chinese-compare.json)。

最终默认实现又检查了中文、带描述英文和参考克隆，共 21 个 teacher-forced 张量，全部通过；
最大的相对 RMSE 为 0.00176。greedy 的 GPU / CPU 采样回退得到相同码字，15,360 个 PCM
样本逐元素一致。记录：[中文](../artifacts/tts-gemv-opt/final-chinese-compare.json)、
[英文](../artifacts/tts-gemv-opt/final-english-compare.json)、
[克隆](../artifacts/tts-gemv-opt/final-clone-compare.json)、
[greedy](../artifacts/tts-gemv-opt/greedy-compare.json)。

## 原生播放与交付

Makepad 实际 Metal 推理、重采样、环形缓冲与 48 kHz CPAL 回调完成了 24 项检查，覆盖模型加载、
中英文、声音描述、克隆、音色设计/复用、重播、保存、生成前/播放中/排空时停止，以及随后 ASR
官方例句识别。七段完整短音频均没有断流、缺样或尾包截断。

额外的长段完整输出 **55.12 秒**音频，播放全部样本，断流、间隙与剩余缓冲为零；生成耗时
88.07 秒，RTF 约 1.60，首个 PCM 在 0.57 秒生成，设备首声在 43.45 秒开始。
它仍需要明显的启动缓冲，不能将连续播放说成低延迟实时生成。停止测试为 54–315 ms，
其中播放/重播/排空时停止均在 59 ms 以内。

该长段与上一轮的输出长度不同，且不是交错配对测量，因此不将两次总耗时之差算作提速比例。
设备验证只将输出幅度置零，不改变系统音量；验证了真实回调时序，没有宣称人工听感评价。
记录：[原生验收](../artifacts/tts-gemv-opt/ui-final/report.json)。

原生单元测试 14 项通过，默认忽略的设备测试由上述实际设备验收覆盖；WebGPU 回退路径编译通过，
没有开展 Web 性能优化。格式检查通过，Clippy 未将已有 ASR/音频风格警告升级为错误。
29 个受保护 ASR / ANE / Web 源文件与冻结 ASR CLI 哈希保持一致。
交付二进制、源码哈希、完整结果及回退版本见[交付索引](../artifacts/tts-gemv-opt/delivery.json)。

## 复现

```sh
CARGO_TARGET_DIR=target/makepad-native cargo build --offline --locked --release \
  --manifest-path prototypes/makepad/Cargo.toml -p r2t2-makepad -p omni-breeze-tts \
  --bin r2t2-makepad --bin breeze-tts --example stream-bench

# 新建输出目录；该脚本记录 macOS 进程 CPU 时间。
python3 artifacts/tts-gemv-opt/run-bench-cpu.py \
  artifacts/tts-gemv-opt/confirm-cases.json /tmp/breeze-depth-compare 6 128
python3 scripts/analyze_breeze_paired.py /tmp/breeze-depth-compare \
  --output /tmp/breeze-depth-summary.json

# 与性能测量分开运行。
CARGO_TARGET_DIR=target/makepad-native cargo test --offline --locked \
  --manifest-path crates/breeze-tts/Cargo.toml --lib -- --include-ignored --test-threads=1
python3 prototypes/makepad/validate-tts.py --device-output-muted --long-stream \
  --output /tmp/breeze-depth-ui
```

完整 depth 图重放和 M5 TensorOps 仍属于后续工作。本轮增加的是小批次向量投影与归一化，
不能将微基准收益当作已完成整帧重放或已接入 GPU Neural Accelerators 的证据。
