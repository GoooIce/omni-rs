# Breeze-TTS-2 本地流式优化调研

调研日期：2026-10-11。目标机器为 Apple M5 Max、128 GiB，当前 macOS 27.0.1、SDK 27.0。
以 Makepad / Rust / Burn 为交付路径，ASR、ANE 和 Web 不在本轮改动范围内。
本轮只做在线资料核对、源码分析和已有测量的重新计算，没有新增模型测速或替换运行程序。

后续用户确认按顺序实施，但明确不试权重量化；实际采用与未采用的候选、验证结果见
[实施记录](breeze-tts-streaming-optimization.md)。以下保留调研时的完整候选作为背景。

建议分两条线推进：缩短过于保守的播放等待；降低每个声学帧的生成成本。
主线优先是 **分阶段测量 → GPU 数据常驻与固定 depth 工作区 → 整帧重放及小批次算子对照**。
量化和弱化 CFG 作为可选实验，不能把上游 GPU 数字当作本机收益。

## 现有数据说明了什么

当前输出两帧一块，即 160 ms。上一轮同一原生进程中，6.0 秒音色设计样本的包间隔中位数为
235 ms、p95 为 257 ms，持续生成赶不上播放。无锁输出和缓冲已消除该样本的断流，
进一步缩短首声必须同时考虑生成速度。下面重新分析的是原有事件，不是新跑的 benchmark：

| 样本 | 音频时长 | 生成耗时 | 首包生成 | 设备首声 | 离线连续播放最早起点 |
|---|---:|---:|---:|---:|---:|
| 普通中文 | 1.76 s | 3.21 s | 0.48 s | 3.22 s | 1.59 s |
| 音色候选 1 | 6.00 s | 9.44 s | 0.61 s | 6.57 s | 3.52 s |
| 音色候选 2 | 5.20 s | 8.46 s | 0.31 s | 7.62 s | 3.34 s |

“离线最早起点”按 `max(包到达时间 - 此包之前累计音频时长)` 计算，假定完整时间线已知，
忽略设备延迟与后续调度抖动。它是理想下限，不能作为在线可实现的承诺。
候选 1/2 的实际启动分别比这个下限晚约 3.05/4.27 秒，说明时长估计和安全余量还有改进空间。
这些样本没有严格隔离不同形状的冷启动与预热，也不用于比较不同音色的性能优劣。

原始数据：[设备事件](../artifacts/tts-playback/ui-final/tts-events.json)、
[重新计算的指标](../artifacts/tts-optimization-research/latency-analysis.json)。

在持续生成 RTF 大于 1 时，固定几百毫秒缓冲无法保障任意长音频连续播放。
对有限长度的一句话，增加启动等待仍然可以避免断流；不能把这两种情况混为一谈。
若平均 RTF 约为 1.6，想得到 0.8 的余量，需要约 2 倍整体提速。
这只是目标差距计算，并非下面任一方案已经获得的收益。

## 当前实现已经具备、以及仍缺少的部分

| 检查项 | 当前状态与意义 |
|---|---|
| Burn 后端 | 固定 0.21.0，CubeCL 0.10.0；Metal feature 与 fusion/autotune 已开启。无需重复建议“打开融合” |
| 帧内 depth KV | 已复用：每个声学帧创建一次 cache，后续码字沿用它；不存在每次把整个码字前缀重算的问题 |
| depth 采样 | 默认 `top_p=1` 时后 15 个 token 留在 GPU；其他配置仍有 CPU 回退 |
| 每帧同步 | 第一码字完整 logits 回读 CPU；随后 16 个码字一起回读；codec 输出 PCM 时另有回读 |
| 回读后的输入 | CPU 码字又逐个建小 tensor，重新查 16 个 embedding 并求和，再给 backbone |
| 随机采样 | 每个 depth 步在 CPU 生成 2048 个 Gumbel 噪声值并上传；选择结果在 GPU 不代表整个采样路径都无主机开销 |
| KV 更新 | Transformer 通过 `Tensor::cat` 追加 K/V，产生随步数变化的形状；depth 每帧重新创建各层缓存 |
| CFG | depth 已按 batch=2 合并；backbone 的条件/无条件分支仍分别调用 |
| codec | 已有流式状态，只解码新帧；不需要再加一次“避免完整前缀重解码”优化 |
| 播放 | 已无锁消费、工作线程 sinc 重采样和自适应缓冲；这部分修复不会降低模型运算量 |

对应源码：[生成循环](../crates/breeze-tts/src/lib.rs)、[采样](../crates/breeze-tts/src/sampling.rs)、
[Transformer](../crates/breeze-tts/src/transformer.rs)、[算子](../crates/breeze-tts/src/ops.rs)、
[播放策略](../prototypes/makepad/src/playback_buffer.rs)。

每个声学帧对应 80 ms 音频，depth 是 12 层、1024 维、FFN 8192 维，要依次调用 15 次，
即每帧 180 次层调用；第一次处理两个位置，其余处理一个新位置。
这使 depth 成为优先测量的对象，但调用次数不能代替耗时占比，当前还没有分阶段 GPU profile。

## 官方与社区可以借鉴什么

### Breeze 官方：固定工作区和整段 depth 重放

官方快速实现预分配 KV、位置与输入输出，并捕获整段码字生成循环。
BreezeRT 还采用逐渐增大的音频块、按播放截止时间调度和增量 codec 状态。
这些机制适合借鉴；其 H100 多并发延迟不能直接外推到本机单会话。
来源：[官方 depth 实现](https://github.com/breezeblue-ai/breeze-tts/blob/58ec70ce5fa4cc361bdebf77ec40d1365da00ab2/models/cudagraph/depth_decoder_graph.py)、
[BreezeRT 技术说明](https://breezeblue.ai/blog/breezert-ultra-low-latency-tts)。

不能把“整帧重放”理解为 15 个码字并行预测：它们有自回归依赖。
可减少的是图构建、资源分配、内核准备、提交与同步，依赖顺序仍要保留。

### MLX 社区：先确认修复是否适用于本项目

找到直接针对 Breeze 的 [Mac streaming 修复](https://github.com/xzf-thu/BreezeTTS2_Mac_Streaming/tree/c6552ad1aea38b64a7f63c466e6098e7ad3da595)。
它主要补帧内 KV、减少逐码回读，另提供整帧编译。源码明确说明只做了小模型数值测试，
未在完整 Breeze 权重和真实 Mac 上测量，不能引用仓库名称作为“已经实时”的证据。
我们前两项已经大体具备，仍可借鉴固定整帧执行和诊断方式。

其参考的 [MLX 编译文档](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
也提醒编译函数要正确管理形状与状态。本项目捕获采样图时必须显式更新 RNG、位置和有效长度，
避免重播同一随机序列、读取上个请求的 KV 或污染停止后的新任务。

### vLLM-Omni：按模型形状优化，而非照搬服务端配置

可以借鉴阶段解耦、减少设备回读、固定缓存，以及 `q_len=1` 的专用 attention。
其 Qwen3-TTS 实践还指出 code predictor 对数值路径敏感；较大融合和较低精度需要完整音频回归。
我们的 codec 已保留因果状态，不应照搬其解码窗口参数或叠加重复前缀解码。
来源：[vLLM-Omni TTS 工程记录](https://vllm.ai/blog/2026-06-23-vllm-omni-tts)。

## Burn 与 M5 的新机会和限制

### Burn 0.22：值得验证，但不是改版本号即可提速

在线确认 Burn **0.22.0 于 2026-10-06 发布**，项目仍是 0.21。
新版提供运行时图捕获/重放和更完整的后端扩展，也大幅改变 `Tensor<B, D>` 等用户 API。
发布说明中的编译提速指 Rust 重建速度，不能算作语音生成提速。
来源：[0.22 release](https://github.com/tracel-ai/burn/releases/tag/v0.22.0)、
[官方说明](https://tracel.ai/blog/release-0.22.0)。

WGPU 路径是软件重放：复用已解析的 pipeline、binding 等，再编码命令，
没有 CUDA 那样的驱动图对象；遍历工作仍随内核数量增长。
因此适合用固定 depth 形状测收益，不能承诺与 CUDA Graph 一样快。
[WGPU 重放源码](https://github.com/tracel-ai/cubecl/blob/4362f783f4d160ca33f466d141245d4e30c67510/crates/cubecl-wgpu/src/compute/graph.rs)
和[上游图重放 benchmark](https://github.com/tracel-ai/cubecl/blob/4362f783f4d160ca33f466d141245d4e30c67510/crates/cubecl-wgpu/examples/graph_bench.rs)。

0.22 的显式 Metal 设备选择仍可走 WGPU 的原生 MSL 路径；另一个独立的 `cubecl-metal`
运行时在所检查提交中没有图录制实现，不能将两者混称为“原生 Metal 图已支持”。
[0.22 设备选择](https://github.com/tracel-ai/burn/blob/v0.22.0/crates/burn-dispatch/src/device.rs)、
[独立 Metal 运行时](https://github.com/tracel-ai/cubecl/blob/4362f783f4d160ca33f466d141245d4e30c67510/crates/cubecl-metal/src/compute/server.rs)。

建议先建独立 benchmark 验证 0.22 + WGPU/MSL 的固定 depth 工作区与重放。
当前 TTS 依赖根项目中带 Burn 泛型的权重接口；直接全仓升级会牵涉 ASR/ANE，
应在收益成立后设计 TTS 版本边界或定向移植，避免为了试图重放先迁移整个项目。

### M5：GPU Neural Accelerators 可以单独研究

Apple 的 MPP / TensorOps 可在 Metal shader 中使用 M5 GPU 的 Neural Accelerators，
这是 GPU 内的矩阵加速单元，与独立 ANE 是不同路径。Apple 建议自定义矩阵内核从
SIMD-group matrix 转向 TensorOps；同时明确单 token 的窄矩阵 decode 往往受内存带宽限制。
来源：[Apple M5 技术讲解](https://developer.apple.com/videos/play/tech-talks/111432)、
[MPP 编程指南](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)。

本地 CubeCL 0.10 的 Metal capability 注册使用 `MetalArchitecture::Metal3`，不能假定
当前 Burn 已使用这条新 TensorOps 路径。后续实验应同时测：

- `M=1/2` 的 decode：优先 GEMV、小批次 GEMM、权重排布和带宽利用；不能盲目填充成大矩阵。
- 文本 prefill 与 codec 的较大矩阵/卷积：更适合验证 MPP 与计算后融合。

MPP 集成必须复用设备与缓冲，避免为每个算子 CPU 回读或跨设备复制。
应先做同形状小实验，再决定是否加 Burn 后端扩展。当前 `xcrun -f metal` 不可用，
离线编译/调试工具需要另行配置；已有运行时 Metal 编译仍可工作，这不是推理被阻塞。

## 建议实施顺序

以下收益均为方向判断，尚未实测。先做能够解释瓶颈的实验，再决定投入较大的路线。

| 顺序 | 方案 | 主要收益 | 成本 / 风险 | 验证后再决定 |
|---|---|---|---|---|
| 0 | 分阶段 CPU/GPU profile 与包间隔记录 | 区分提交、同步、矩阵运算、codec 的成本 | 低；强制同步会干扰原性能 | 保留无 profiler 的端到端对照 |
| 0 | 按语言/音色校准时长和缓冲余量 | 减少多余首声等待 | 低至中；估计偏短会重新断流 | 用留出时间线测零断流与首声，不能用完整未来数据作在线输入 |
| 1 | GPU token、embedding、采样状态常驻 | 减少每帧回读和重复上传 | 中；EOS、RNG 和取消语义敏感 | 先固定码字数值对照，再随机采样完整语音 |
| 1 | depth 固定 KV / mask / 输出工作区 | 减少分配和形状变化，准备图重放 | 中；有效长度和旧数据隔离 | 每帧重置位置；不能把上一声学帧当本帧历史 |
| 2 | 固定整帧 depth 重放 | 降低大量小操作的主机开销 | 中至高；0.22 API 迁移需隔离 | 与同算子、同精度的非重放版比较 |
| 2 | 专用 GEMV、GQA 与相邻算子融合 | 降低小形状访存、启动开销 | 高；融合可能改变舍入 | 按真实形状逐项测量，避免只测大方阵 |
| 3 | 仅线性层的 W8A16，之后再试 W4A16 | 降低权重带宽 | 中至高；精度、内核回退 | 确认实际为量化计算，而非每次展开整块权重 |
| 3 | CFG 性能档、逐渐增大 codec 块 | 减少条件分支/解码调用 | 低至中；描述遵从度或首包延迟变化 | 显式可选，音色设计保留原质量默认 |
| 后续 | M5 TensorOps 后端扩展 | 加速适合的矩阵与卷积形状 | 高；小批次未必受益 | 同形状实测显著胜出再接入 |

### 第一轮需要落到代码上的具体点

1. **保留 frame token tensor，直接生成下一帧 embedding。** 现在 CPU 回读 16 个码字后又逐个上传，
   可以让 codec 和 backbone 接收 GPU 上的码字，报告/保存用的 CPU 副本延后、合批。
   第一码字采样也可移入设备；先争取每帧一次状态检查，再评估与 PCM 输出合并同步。
   EOS 必须在提交有效声学帧前判定，不能把保留 token 送给 codec。
2. **减少采样准备。** 默认每帧 15 份 CPU 噪声可先合批上传，后续换设备 RNG 和专用 top-k/sampling。
   `top_p!=1` 保留正确回退。随机数算法或浮点路径改变后，不能声称同 seed 的 WAV 必然逐样本一致。
3. **先固定 depth，后扩展 backbone。** depth 只需很短的帧内历史，适合固定容量、有效长度和原位写入；
   backbone 按实际 prompt 加帧上限分配，不应一律按 40960 个位置预分配/计算。
   用 `slice_assign` 写法本身不保证消除了复制，需要检查实际分配和别名处理。
4. **预备静态数据。** 各码字的 head 视图、mask、offset、RoPE 切片提前准备。
   固定 embedding 与无偏置 depth 输入投影可以实验性预计算，动态 backbone hidden 仍实时投影；
   此项是本地代码推导，收益与 FP16 舍入差异需测量。

较大融合按层内相邻操作开始：QKV 投影打包、gate/up 打包与 SiLU×up、RMSNorm/残差、
RoPE/KV 写入、支持 GQA 的短 attention。当前 `repeat_dim` 显式扩展 KV heads，可用直接 GQA 避免。
不要把整段 180 次层调用塞成一个巨型 kernel；图重放与算子融合解决不同开销。
现有 FP32 residual/归一化边界，尤其 T5Gemma2 超出 FP16 范围的保护，必须保留或重新验证。

### CFG、量化和流水线的取舍

CFG 的 depth 已合 batch=2，下一处可研究 backbone 两个分支合批。
两条 prompt 的长度可能不同，需要独立 position、有效长度和 mask，不能只拼张量就认为等价。
将 `guidance_scale` 从默认 4 调到 1 可去掉无条件分支，声音描述仍在条件输入中，
但控制强度会变化；从 4 调到 2 仍是双分支，并不会因数值减半就把计算减半。

量化优先 depth/backbone 的线性权重，先 W8A16，再局部 W4A16；codec 与敏感归一化初期保持原精度。
Burn 已有量化张量和矩阵路径，但本地 0.21 的部分策略会反量化回退；
“权重文件变小”不能证明矩阵计算更快。MLX 的量化文件也不能直接交给现有 Burn 加载器。

把 codec 分到另一线程可以重叠主机准备或回读，但同一 GPU 的两个重负载阶段不一定更快。
优先试可控双缓冲、异步 PCM 读取和播放余量驱动的调度；只有 profile 显示空隙时再增加并发。
生成接近实时后，可试首块 1 帧、随后 2/4 帧，并在缓冲不足时提前解码已生成帧。
已有两轮 2/4/8 帧对照未显示足够的端到端提速，不建议把固定大块作为主方案。

## 验收目标与实验边界

建议作为首轮工程目标，均不是已有能力承诺：

| 指标 | 建议目标与口径 |
|---|---|
| 热启动设备首声 | p50 ≤ 0.8 s、p95 ≤ 1.2 s；模型已加载、对应路径已预热、参考编码已缓存 |
| 持续生成 | 8–30 秒音频上的 RTF ≤ 0.8；分别报告含/不含首包预填充的结果 |
| 包间隔 | 两帧 160 ms 分块的 p95 ≤ 160 ms；同时记录 p99 和累计供给不足，平均值不够 |
| 连续播放 | 30–60 秒场景无新增软件队列断流；使用真实设备回调，先静音测时序，再人工试听 |
| 停止 | 目标 ≤ 200 ms；整帧重放后要重新检查提交工作不可取消带来的最坏延迟 |
| CPU | 比较工作线程 CPU 时间、提交次数与 GPU 空隙，不只看平均总进程占用率 |

普通朗读、描述 CFG、克隆、音色设计分别测；覆盖中英文、长句、数字、长停顿、尾包、
冷启动、停止后立即重启。先用固定 teacher-forced codes / greedy 检查数值与 EOS，
再测多 seed 采样、音色一致性、描述遵从度和听感。ASR 回送只能辅助检查可识别性，不能代替音质评价。

每个候选保持相同文本、权重、精度/采样设置、codec 块和播放策略，预热后至少重复 5 轮；
冷启动、编译、模型加载、参考编码分别计时。profiling 运行和最终速度验收分开，
避免把每个阶段强制同步后的耗时当成真实流水线结果。

第一阶段的可交付结果应是一份阶段耗时/提交与回读数量报告，以及固定 depth 工作区的小型对照。
只有这个对照显示主机准备成本明显，才投入整帧重放；若权重访存占主导，优先 GEMV 和量化内核。
较精确的缓冲预测可同时独立推进，但不能为了降低首声指标重新引入周期性断流。

## 证据索引

- 本机上一轮交付：[播放修复记录](breeze-tts.md#后续修复流式播放断续)、[交付索引](../artifacts/tts-playback/delivery.json)。
- 本轮离线重算：[latency-analysis.json](../artifacts/tts-optimization-research/latency-analysis.json)。
- 上游快照：[Git 提交记录](../artifacts/tts-optimization-research/upstream-heads-git.json)、
  [源码下载索引](../artifacts/tts-optimization-research/sources/index.json)、
  [补充源码索引](../artifacts/tts-optimization-research/sources/extra-index.json)。
- 官方 Breeze HEAD 仍是本项目固定的 `58ec70ce5fa4cc361bdebf77ec40d1365da00ab2`。
- 调研保存了公共源码供核对，没有安装 MLX 依赖、下载另一份模型或更新生产依赖。
