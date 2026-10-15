# Breeze-TTS-2 流式优化实施记录

2026-10-11，Apple M5 Max / 128 GiB。本轮按[调研顺序](breeze-tts-optimization-research.md)
先测量、再对照候选，用户明确排除了权重量化。生成网络的 FP16、敏感残差的 FP32 和 codec 的
FP32 精度策略保持不变。ASR / ANE 推理与 Web 不在修改范围内。

## 保留的实现

- 16 路采样码字留在 GPU，直接交给 codec；报告所需的码字在请求结束时一次读取。
  原来每帧回读、给 codec 重新上传码字的往返被移除，EOS 检查仍在主生成器处执行。
- 复用 depth 已查出的音频 embedding，直接构造下一帧输入，保持原来的 FP32 求和与 FP16
  舍入边界，避免每帧重新上传 16 个 token 并查表。
- 每帧 15 份 Gumbel 噪声合批上传。保留 SplitMix64、CPU 双精度噪声计算和原有 RNG 消耗顺序；
  greedy 不消耗随机数，CPU / top-p 回退路径继续可用。
- 复用 depth 的层槽容器，每帧清空历史，不把前一声学帧的 KV 用于下一帧。
- 播放前八个包间隔继续使用保守余量；之后同时参考平均速度、近期速度和较慢包的 p90，
  避免对稳定供给一直附加 25% 的速度罚值。保留最小缓冲和断流后的重新缓冲。
- 同一会话按声音条件和语言保存最近完整结果的时长比例，仅参考文本估计长度在本次
  0.5–2 倍范围内的样本，取较慢值并留 20% 余量；没有可比历史时使用保守估计。
  停止、达到生成上限的结果不参与学习；模型重新加载时清空记录。预测只控制播放启动，
  不截断生成，也不将未来到达时间用作在线输入。

代码：[生成循环](../crates/breeze-tts/src/lib.rs)、[采样](../crates/breeze-tts/src/sampling.rs)、
[codec](../crates/breeze-tts/src/codec.rs)、[播放策略](../prototypes/makepad/src/playback_buffer.rs)、
[原生工作线程](../prototypes/makepad/src/tts_engine.rs)。

## 测量方法与第一轮结果

新增同进程 benchmark，区分第一次运行和五次热运行；普通中文和带声音描述的 CFG 4
各生成 64 帧，即 5.12 秒。固定文本、seed 42、采样设置、两帧分块、同一锁定依赖；
这是固定帧数的吞吐测量，完整 EOS、长句和真实设备播放另行验收。

同步阶段诊断中，原版 depth 每帧约 96–100 ms，占测量总时间约 72%–79%；backbone
约 13 ms / 25 ms，codec 平摊约 5.5 ms。强制同步会改变流水线，不能把这些诊断数字
作为最终端到端吞吐。低层内核日志也包含加载、autotune 和时间戳开销，仅用于定位算子。

第一轮 GPU 常驻与噪声合批的热运行中位数如下：

| 场景 | 基线 RTF | 常驻版本 RTF | 生成耗时减少 |
|---|---:|---:|---:|
| 普通中文 | 1.466 | 1.258 | 14.2% |
| 声音描述 / CFG 4 | 1.883 | 1.427 | 24.2% |

相同十二次生成的进程 CPU 累计时间由 253.92 s 降至 213.58 s，约减少 15.9%。
这里包含模型加载、冷启动和文件输出；它不表示瞬时 CPU 占用率降低了同样的比例。
原始记录在 [baseline-off](../artifacts/tts-streaming-opt/baseline-off-summary.json) 和
[resident](../artifacts/tts-streaming-opt/resident-summary.json)，所有失败/较慢试验也保留。
后续排队大小试验出现运行间速度漂移，因此不从不同时间批次中挑选最快值作为最终收益。

这里的首包时间表示 PCM 生成时间，设备首声还包括缓冲等待。两者分别记录，不能互换。

### 最终源码复测

原生验收进程退出后，分别用冻结基线和交付源码编译的 CLI 顺序重复相同的 6 次 × 64 帧，
各舍去第 0 次冷运行。本轮没有并行启动模型任务或编译，系统其他进程未作干预。

| 场景 | 基线热中位 RTF | 交付热中位 RTF | 生成耗时变化 | 交付包间隔 p95 |
|---|---:|---:|---:|---:|
| 普通中文 | 1.853 | 1.720 | 减少 7.2% | 314 ms |
| 声音描述 / CFG 4 | 2.416 | 2.424 | 增加 0.3%，接近持平 | 441 ms |

这组十二次生成的进程累计 CPU 时间为 351.94 → 338.96 秒，减少 3.7%；总墙钟时间为
142.49 → 138.98 秒。保留数据常驻的结构性改进，但**不能把首轮 14%–24% 提速或
15.9% CPU 降幅当作稳定保证**。普通中文本轮有收益，声音描述尚无稳定收益；两条路径
均未达到 160 ms 的包间隔目标。
记录：[基线复测](../artifacts/tts-streaming-opt/baseline-delivery-summary.json)、
[交付复测与码字/PCM 对照](../artifacts/tts-streaming-opt/chosen-off-summary.json)。

## 验证后没有采用的候选

| 候选 | 本机观察 | 处理 |
|---|---|---|
| 固定 16 位置 depth KV / mask | 两种场景的码字和 PCM 均与常驻版本相同；普通场景接近持平，CFG 变慢 | 未设为默认；实验快照保留 |
| QKV 投影合并 | 普通场景稍慢，CFG 基本持平 | 不纳入生产路径 |
| gate/up 或全部投影合并 | 两种场景均变慢 | 不纳入生产路径 |
| 加载时准备固定输出头矩阵 | 随后的复测没有稳定收益，低于常驻候选的速度 | 撤回该项，保留原有矩阵路径 |
| 短查询 GQA 折叠，避免显式 KV 重复 | CPU 数值测试通过，但实际 Metal 合成变慢 | 不纳入生产路径 |
| WGPU 提交批量 64 / 128 / 256 / 512 | 明显变慢 | 保持现有配置 |
| WGPU 提交批量 8 / 16 | 8 更慢；16 的短测有小幅收益，长测存在漂移 | 不改变共享 ASR / TTS 设备配置 |

完整算子候选在 [operators/source](../artifacts/tts-streaming-opt/operators/source)，
实验二进制在 `artifacts/tts-streaming-opt/{baseline,resident,operators}/`。
这些实验开关只用于所保存的候选二进制，未作为正式 UI 配置保留。

### 图重放验证

在独立目录构建固定提交 `4362f783f4d160ca33f466d141245d4e30c67510` 的 CubeCL 0.11
上游 `graph_bench`，实际后端为 `wgpu<msl>`。没有升级生产 Burn 0.21 或其锁文件。

| 微基准 | 普通提交 | 软件重放 | 相对速度 |
|---|---:|---:|---:|
| 64 元素 × 1000 内核 | 1.93 µs/内核 | 1.61 µs/内核 | 约 1.2× |
| 64 元素 × 5000 内核 | 1.77 µs/内核 | 1.34 µs/内核 | 约 1.3× |
| 1M 元素 × 500 内核 | 7.27 µs/内核 | 7.12 µs/内核 | 约 1.0× |

这是交错运行的底层命令重放测试，**不是完整 depth 或整模型提速**。结果说明该路径可运行，
但不能据此承诺两倍提速。完整整帧捕获仍需要解决固定资源、更新输入/RNG、EOS/停止检查和
Burn 0.22 API 迁移边界；本次没有把微基准当作已接入模型的能力。
记录：[graph-results.log](../artifacts/tts-streaming-opt/graph-results.log)。

## 正确性与播放回归

CPU 回归检查覆盖噪声合批前后的 token 与下一帧 RNG 状态、保留 token、EOS、因果卷积、
流式尾包、时长历史隔离、消费顺序与取消。实验分支另检查固定 KV 重置和 GQA head/mask 顺序。
最终源码的中英文、克隆条件共 21 个 teacher-forced 张量与基线逐元素一致；greedy 的
GPU 采样与 CPU 采样回退也得到相同码字和全部 15,360 个 PCM 样本。
记录：[中文](../artifacts/tts-streaming-opt/chosen-chinese-compare.json)、
[英文](../artifacts/tts-streaming-opt/chosen-english-compare.json)、
[克隆](../artifacts/tts-streaming-opt/chosen-clone-compare.json)、
[greedy](../artifacts/tts-streaming-opt/greedy-compare.json)。

随机合成不能以整段 WAV 哈希相同为验收标准：第一轮普通/描述样本分别在第 34/40 帧首次
采样分叉。精度和随机数算法没有改变，但 GPU 运算组织会影响边界处的浮点选择。
因此保留逐层数值检查、相同码字前缀的 PCM 检查，以及完整语音/停止/播放回归。

原有 12 条到达时间线与新 36 条生成时间线，共 48 条，分别在五种输出采样率下回放，
共 240 组。新策略没有新增断流；所有默认两帧场景无断流。原有一条四帧非默认场景的
0.49 秒间隙仍存在，没有把它计为本次修复成功。48 kHz 下启动等待的中位数减少 0.82 秒；
少量高抖动时间线会等待更久。记录：[buffer-comparison.json](../artifacts/tts-streaming-opt/buffer-comparison.json)。

真实设备回归完成 24 项检查，中英文、描述、参考克隆和两次音色设计的七段完整音频均无
断流。额外的 49.2 秒长段在真实 48 kHz CPAL 回调下播放全部样本，断流、间隙和剩余缓冲
均为零；停止/取消响应为 53–170 ms，随后 ASR 官方样本仍正常识别。
记录：[原生验收](../artifacts/tts-streaming-opt/ui-verified2/report.json)。

长段验收中也发现并修复了时长校准问题：直接把短句的语速用于长句，会过早开始播放；
修正前同一长段出现 4 次断流、累计 9.61 秒间隙。现在校准要求长度可比，并有短句/段落
互不误用的回归测试。失败记录保存在 `ui-delivery`，没有删去或计为通过。
另一次重跑停在验证脚本的按钮时序问题，脚本已改为等待界面空闲。

这次长段生成耗时 109.31 秒，RTF 为 2.22，设备首声等待 79.05 秒；它证明播放连续性，
不能证明低延迟或实时吞吐。动态长度和运行间速度漂移仍需进一步定位，短测结果不能外推
到长段。持续 RTF ≤ 0.8 和热首声约一秒仍未达到。

本轮完成了常驻路径及播放策略交付，并筛选了固定 depth、合并投影、GQA 和底层重放候选。
截至这轮，完整 depth 图重放、专用 GEMV 内核与 M5 TensorOps 尚未接入；它们需要独立收益对照，
不应把本轮的微基准或通用矩阵投影合并称作这些能力已完成。
后续的 GEMV / RMSNorm 实现与整模型对照见[Metal depth 优化](breeze-tts-metal-depth-optimization.md)。

## 复现

```sh
CARGO_TARGET_DIR=target/makepad-native cargo build --offline --locked --release \
  --manifest-path prototypes/makepad/Cargo.toml -p omni-breeze-tts --example stream-bench

target/makepad-native/release/examples/stream-bench \
  artifacts/tts-streaming-opt/cases.json /tmp/breeze-stream-bench 6 64
python3 scripts/analyze_breeze_benchmark.py /tmp/breeze-stream-bench

# 诊断模式会改变同步行为；吞吐验收应移除此变量。
BREEZE_TTS_PROFILE=sync target/makepad-native/release/examples/stream-bench \
  artifacts/tts-streaming-opt/cases.json /tmp/breeze-stream-profile 3 32

python3 prototypes/makepad/validate-tts.py --device-output-muted --long-stream \
  --output /tmp/breeze-native-validation
```

`BREEZE_TTS_PROFILE=host` 保留正常调度并记录阶段主机耗时；默认完全关闭阶段记录。
原生设备验证使用真实 Metal 推理、重采样、环形缓冲和 CPAL 回调，仅输出幅度置零；
不修改系统音量或用户设置。时序通过不等于已经人工评价音色质量。

最终检查：TTS 单元测试 7 项通过，原生单元测试 14 项通过（设备测试 1 项按默认忽略，
另由上述真实设备验收覆盖）；格式检查和 Python 语法检查通过。Clippy 正常通过，但现有
ASR/音频代码有 5 条新 Rust 版本的风格警告，因此不声称 `-D warnings` 通过。
源码、二进制哈希及保护文件审计见
[交付索引](../artifacts/tts-streaming-opt/delivery.json)。
