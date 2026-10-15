# Makepad 本地版流程优化

2026-10-11。本轮只推进 Makepad 本地版。ANE 编译器、原生桥接、矩阵算子、encoder/decoder
实现及权重保持原样；Web 页面和脚本不改。旧 ANE 优化目标仍暂停，历史性能结论仍按各自冻结构建解释。

## 调研与取舍

| 社区策略 | 在本项目中的处理 |
| --- | --- |
| Whisper-Streaming 使用最小处理间隔，减少重复计算，处理速度不足时积累更多输入 | 本地新增 320 ms 均衡档，160 ms 前瞻；保留 160 ms 低延迟档。按完整音频块处理，不丢帧 |
| Whisper-Streaming 用 LocalAgreement 确认前缀，并限制音频上下文长度 | 保留 R2T2 的不可回改协议与 token 回退。本地麦克风另用既有解码结果确认稳定尾字，见下文后续修复；不为此增加解码次数 |
| whisper.cpp 的采集与转写解耦，等待输入；显式设置线程数 | 保持采集独立线程，消费者从 5 ms 定时轮询改为数据/结束通知；不把 C++ CPU 线程数套到 Metal 后端 |
| faster-whisper / Whisper-Streaming 用 VAD 跳过非语音区间 | 本轮尚未接入。VAD 会改变切段、尾音和音频时间线，需要用轻声、短停顿、噪声和长静音单独验证；本轮收益不含 VAD |
| Whisper 社区的量化、注意力融合、模型大小切换 | 本轮未采用；涉及 ANE 推理的实现需另行讨论 |

参考一手实现：[Whisper-Streaming 的处理间隔和确认协议](https://github.com/ufal/whisper_streaming/blob/6da90b44b7e50d79695e68166d2a2c7609c75abb/README.md)、
[whisper.cpp 流式采集](https://github.com/ggml-org/whisper.cpp/blob/d1be6fde11ac6e0407606b4e42fe72d34add8037/examples/stream/stream.cpp)、
[faster-whisper 的 VAD 与测量注意事项](https://github.com/SYSTRAN/faster-whisper/blob/a52b026ac844c02117865e61968547639b60a6f8/README.md)。
源码版本和 SHA-256 保存在 [research/sources.json](../artifacts/native-pipeline/research/sources.json)。
表中适配方案是结合本项目代码作出的选择，不代表上述项目已经验证过 R2T2。

## 当前实现

本地麦克风与 WAV 使用预计算的有理相位 sinc 滤波器：保留原滤波半径、截止频率、归一化、
补零和流式尾部处理，在 48 kHz → 16 kHz 的热路径复用同一组系数，避免逐样本调用 sin/cos。
44100 等采样率使用有限的相位表；系数总数超过 262144 时退回原实现，避免异常采样率导致大分配。
这部分只由本地界面启用，原库重采样器和模型 mel 前端未改。

临时 PCM 仍分段落盘、按顺序读取并清理；数据和结束状态唤醒等待的转写线程，实时音频回调不获取新增锁。
有超时检查作为取消兜底，结束通知后仍核对最后一段数据，手动停止继续取消待处理音频。
本地界面取消无用途的 250 ms 心跳定时器，文本没有变化时不再重排完整转写控件。
逐步全文 JSON 日志仅在设置 `R2T2_TRACE_EVENTS=1` 时启用；验证工具会显式设置此变量。

本地档位依次为：均衡 320/160 ms、低延迟 160/160 ms、快速 960/0 ms、省算力 1280/0 ms。
每块 token 预算依次为 8、4、8、12。均衡档首次音频积累为 480 ms，低延迟档为 320 ms；
这些是调度参数，实际出字延迟还包含模型运行与文字确认时间。停止仍由用户决定，没有新增录音时限。

## 官方 Python 后处理对照

对齐对象是 [R2T2 的 `streaming_transcribe_no_reset`](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/r2t2/r2t2_asr.py)
及其调用的 [Qwen3-ASR 文本工具](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/utils.py)。

| 环节 | 旧行为 | 本地新策略 |
| --- | --- | --- |
| 标点上下文 | 在累计前缀和新文本上统一处理 | 仅处理新生成片段，以原始前一非空白字符判断；包括英文引号 |
| 中文字间空白 | 无条件移除 | 普通流式更新仅在指定/检测为 Chinese 时移除；结束阶段遵循官方独立分支 |
| 过度重复 | 未实现官方清理 | 对齐 Qwen 阈值 20 的字符及短模式重复清理，正常重复保留 |
| 语言元信息 | 简化解析 | 对齐空白、大小写、无标签文本和 `language None` 的解析规则 |
| 已提交文本 | 永久保留 | 继续永久保留；清理结果若会改写已提交前缀，暂不提交该候选 |

新增 `StreamConfig.text_postprocess = "official"`，由本地引擎显式启用。库、CLI 和 Web 默认仍为
`"legacy"`，避免将新的文本策略套到既有 ANE 数值/转写基线。
992 组纯文本参考由固定版本的 Python 函数生成，包括混合文字、引号、标点串、Unicode 空白、
替换字符、停止符、语言标签、重复阈值和结束分支；Rust 逐项比对。
重复清理在 token 回退前执行一次，回退后仅提取已确认文字；专门覆盖再次清理会改变结果的边界。
测试不是通过 Python 重新运行神经网络，不能据此声称两个推理后端的所有输出相同。

仍保留两项明确的产品行为差异：音频恰好落在分块边界时仍完成尾字解码，防止未确认后缀丢失；
手动停止立即取消积压和未确认尾音，不再排空队列。
官方 [WebSocket 服务](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/ws_server.py)
另有 FireRed VAD、动态 token 预算、阈值 5 的循环检测和会话重置策略；本轮不宣称复刻这一整套服务。

## 验证方法

原有本地程序和从修改前源码重新构建的控制程序分别归档，性能对照以后一份为准。
CPU 时间使用 macOS `proc_pid_rusage`，按 Mach timebase 转成秒；每次运行先用
`time.process_time()` 校准单位。100% 表示占用一个 CPU 核心，不是整机所有核心。
模型载入、首次预热、载入前后空闲、三轮热态 WAV 和三轮按时钟到达的 48 kHz 双声道 PCM 分开测量。
性能测量期间不运行构建，调用栈采样另开一轮，不混入计时。

当前 Makepad 的推理后端是既有 Burn Metal，因此本页数据不是 ANE 推理性能。
控制版的[调用栈采样](../artifacts/native-pipeline/control-measure/capture.sample.txt)除音频前处理外，
还出现 CubeCL 的内核提交、张量分配及线程间调度。所用 `cubecl-common 0.10.0` 内部通道有
自旋、让出线程和睡眠的等待策略；本轮未修改这些依赖。降低音频处理频率同时减少了后端提交次数，
但不会消除后端的 CPU 工作；不能把全部收益都归因于重采样。

录音测试音源经过同一采集缓冲、磁盘队列和重采样流程，但没有打开物理麦克风；不能替代实际设备、
房间噪声和系统权限验证。固定中文集同样只覆盖干净朗读音频。

```sh
# 构建与启动
python3 prototypes/makepad/build-native.py
./target/makepad-native/release/r2t2-makepad

# CPU 和实时采集对照；输出必须是尚不存在的目录
python3 prototypes/makepad/measure-native.py --output artifacts/native-repeat
python3 prototypes/makepad/measure-native.py --mode 1 --output artifacts/native-low-latency-repeat

# 独立滤波器对照，不加载模型
rustc --edition 2021 -O prototypes/makepad/bench-resampler.rs -o /tmp/r2t2-resampler-bench
/tmp/r2t2-resampler-bench

# 原生交互、20.22 秒跨窗口输入与 17 秒主动打断
python3 prototypes/makepad/validate-native.py --continuous --output artifacts/native-interaction-repeat
```

测试日志、二进制、源码指纹、原始计数和对照数据均在 `artifacts/native-pipeline/`。

## 本机实测

同一个 6.74 秒公开样本，分别运行三轮；表内 CPU 占用按累计 CPU 秒 / 累计墙钟秒计算。
各版先载入模型并预热，诊断日志统一开启；最终文本与全部增量拼接一致。

| 配置 | 模拟麦克风 CPU 占用 | 三轮 CPU 总时间 | 每次处理事件数 |
| --- | --- | --- | --- |
| 修改前控制，160/160 ms | 116.24% | 24.47 s | 42 |
| 新版本低延迟，160/160 ms | 91.12% | 18.95 s | 42 |
| 新版本均衡，320/160 ms | 65.88% | 13.62 s | 21 |

本轮低延迟与均衡档的 CPU 总时间分别降低 22.55%、44.34%。这是三个顺序短测的结果，
不是随机交替、多设备或长会话结论；单轮波动明显。空闲 CPU 没有测出可靠改善。
文件转写会尽快跑完：均衡档 CPU 总工作时间减少，但平均占用仍约 213%（控制约 203%），
所以不能把表中实时采集的占用降幅用于承诺文件转写的瞬时 CPU 百分比。
完整对照在 [cpu-comparison.json](../artifacts/native-pipeline/cpu-comparison.json)。

独立重采样对照使用 5 秒 PCM、20 ms 分包，排除首轮后统计三轮：48 kHz → 16 kHz
约快 9.58 倍且逐样本完全相同；44.1 kHz → 16 kHz 约快 10.17 倍，最大绝对差
为 2.98e-8。这里仅计重采样函数，不是整个应用的加速倍数。
[原始结果](../artifacts/native-pipeline/resampler-bench.json)。

固定中文集使用 [mandarin-v1.json](eval/mandarin-v1.json) 中全部 96 条音频，共 494.52 秒、
1417 个参考字符。逐文件预热后各测一次，使用既有 CER v1 规范化：NFKC、转小写、移除 Unicode
标点和空白，保留数字、符号和简繁差别。

| Makepad 配置 | 字符错误数 / CER | CPU 总工作时间 | 推理 RTF |
| --- | --- | --- | --- |
| 修改前控制，160/160 ms | 43 / 3.03% | 494.23 s | 0.446 |
| 最终版本均衡，320/160 ms | 39 / 2.75% | 235.26 s | 0.230 |

在此集合上，3 条的字符错误减少、没有条目增加，另 1 条仅标点不同；CPU 总工作时间减少 52.40%。
RTF 为累计模型处理时间 / 音频时长，不包含预热、载入和界面操作。
所有运行的增量拼接与最终文本相同。这是干净中文朗读的小集合，未据此承诺噪声、轻声、其他语言
或长会话的同等精度。[逐条对照和原始报告](../artifacts/native-pipeline/corpus-comparison.json)。

上述数字来自本轮性能优化的冻结构建 `94b81043ff82c60b19e0f4e61262be8a25de3fef030d8ca037e400ca1f396c86`，
详见 `final-balanced/` 和 `final-low-latency/`。中间候选及修正后处理顺序前的测量保留在独立目录，
没有混入上表。此后发布路径已更新为下文的停顿尾字修复构建；上表仍对应此前冻结版本。

最终构建通过 12 次原生运行、10 项界面检查：模型加载期间响应、48 kHz 双声道采集链路、
取消和重启、输入打开失败后重试、WAV 错误恢复、四档切换及中文输入。
20.22 秒模拟录音跨过 16 秒滚动窗口，三遍正文完整、全部增量拼接一致；另一轮在已采集
17.03 秒时手动停止，约 103 ms 返回，已确认音频为 16.80 秒，积压取消。该停止时间是单次实测，
不是上限；仍无自动录音时限。
[交互报告](../artifacts/native-pipeline/native-interaction/native.json)、
[原生截图](../artifacts/native-pipeline/native-interaction/native.png)。

最终核对了 33 份冻结源码及验证工具、29 份受保护的 ANE/模型/Web 文件：源码与测试构建一致，
受保护文件均未改变。发布构建、核心与本地测试及 Clippy 均通过；Clippy 沿用已有风格豁免。
[构建记录](../artifacts/native-pipeline/final-build-checks.json)、
[源码与二进制审计](../artifacts/native-pipeline/final-audit.json)。
真实麦克风权限、硬件与房间噪声仍需实机验证。

## 后续修复：麦克风长停顿不提交尾字

2026-10-11。原生音源结束才会完成尾字解码；麦克风的静音仍是持续输入，固定回退一个 token
可能永远保留最后一个字。用 `aishell-test-0-1228` 重复两次、中间及结尾各加 5 秒静音复现：
“恰恰在风格打法和比赛理念上”的“上”在停顿期间一直未输出，第一句直到音频 10.08 秒继续说话时
才补出，第二句直到整个测试音源在 17.916 秒结束才补出。

本地麦克风启用 `stable_tail_ms = 800`：只有解码已产生停止标记、完整候选持续相同，
且累计处理了至少 800 ms 新音频，才确认原来回退的尾部。候选改变、生成预算耗尽或滚动窗口
更换上下文都会重新计时。以音频采样计时，单纯等待或重复检查不会提前确认。
均衡档每 320 ms 解码一次，因此通常需要候选不变后的 3 次更新（960 ms 新音频）；
这不是从实际说话停止到出字的固定延迟保证。

这是本地麦克风在官方固定回退之上的提交策略，不是 VAD 切段，也不需要额外解码、停止录音、
释放模型或丢弃音频。文件、CLI、Web 的 `stable_tail_ms` 默认仍为 0；ANE 与模型计算实现未改。

同一反例的新版本在音频 4.64 / 13.60 秒已经补出两句末尾的“上”，仍保持录音，
比两段音频各自结束晚约 0.68 秒。降低 20 dB 音量并添加 3 秒开头静音时，两次尾字同样提交，
开头静音无文字；这仍是受控音源验证，不是物理麦克风或房间噪声验收。
新旧运行的增量拼接均与最终文本一致。

复现工具：[validate-pause.py](../prototypes/makepad/validate-pause.py)。
旧版反例：[control-tail](../artifacts/native-pause/control-tail/summary.json)；
修复验证：[fixed](../artifacts/native-pause/fixed/summary.json)、
[quiet](../artifacts/native-pause/quiet/summary.json)。
该次停顿尾字修复构建 SHA-256：`374949854f6af860dd0644f31fd6dbf9f03be65e4b1941b9528608c37686999c`。
后续本地界面的自动加载与目录选择交互见 [Makepad 验证记录](../prototypes/makepad/VALIDATION.md)；
本页性能数字仍对应各自标注的历史构建。

修复前后此反例都产生 56 个事件、处理 17.916 秒音频，未增加解码轮次。
8 项核心单元测试（含稳定候选确认与 992 组 Python 对照）、8 项模型/流式测试、Clippy 和发布构建通过。
另有 12 次原生交互回归通过，覆盖 20.22 秒长流、停止/重启、四档切换和错误恢复；
本次长流在采集 17.09 秒时主动停止，约 85 ms 返回。
[交互报告](../artifacts/native-pause/interaction/native.json)、
[构建和源码审计](../artifacts/native-pause/audit.json)。
`target/release/r2t2-makepad` 与 `target/makepad-native/release/r2t2-makepad` 均已更新；
已启动的旧进程需要重启才使用修复。

```sh
python3 prototypes/makepad/validate-pause.py --output artifacts/pause-repeat
python3 prototypes/makepad/validate-pause.py --gain 0.1 --leading-silence 3 --output artifacts/pause-quiet-repeat
```

## 后续修复：短音频结束时漏掉句尾

2026-10-11。音色复用生成的 1.84 秒 WAV（`artifacts/breeze-design/ui-initial/design-reused.wav`）
在四个本地档位均识别成“你好，欢迎”，漏掉“回来”。已有 `finish()` 会处理残余输入并解除
token 回退，但没有足够的末尾音频上下文。对原 WAV 保持不变，只在最后一次解码中补零，
即可恢复句尾；这与上文的麦克风“已识别尾字尚未提交”是两个问题。

本地 WAV 和采集音源自然结束时启用 `StreamConfig.final_padding_ms = 640`：
在 mel 前端虚拟补足 640 ms 数字静音，已有的尾部零样本会抵扣。这里只检查精确的零，
不以音量阈值判断人声，不截断轻声或替换原始样本。虚拟样本不进入 PCM 缓存、采集时钟或
滚动窗口淘汰逻辑，因此重试不会累计补零，接近 8/16 秒窗口边界时也不会错误地提前淘汰文字上下文。
空输入直接结束；手动停止仍取消积压和未确认文字，不执行补尾。持续录音沿用 800 ms
稳定候选确认，没有自动录音时限，也没有新增持续解码轮次。

固定补零长度的 28 次初始对照中，四档在 160/320 ms 补零后恢复“回来”，480/640/800/1000 ms
还恢复了句号。随后发现官方示例本来已有 500 ms 数字静音，继续固定追加 640 ms 会使
模型省略句号，因此最终版本改为只补不足部分。初始探索记录单独保留在
`artifacts/asr-tail/probe/`、`natural-padding/` 和 `matrix-initial/`，不作为最终构建报告。

这是本地输入结束策略。固定版本官方 Python 的 `finish_streaming_transcribe_no_reset`
不补静音，本次没有把这一差异说成 Python 输入路径的一致实现；文字后处理仍使用此前对齐的
官方规则。库、CLI 和 Web 默认 `final_padding_ms = 0`，历史 ANE 基线不启用新策略。
模型、ANE 算子与权重未改，Breeze-TTS 生成的 WAV 也未修改。

可重复验证（新报告目录必须不存在）：

```sh
CARGO_TARGET_DIR=target/makepad-native cargo build --release --locked \
  --manifest-path prototypes/makepad/Cargo.toml --bin r2t2-makepad --example asr-regression
python3 prototypes/makepad/validate-asr.py --suite matrix --output artifacts/asr-matrix-repeat
python3 prototypes/makepad/validate-asr.py --suite corpus --output artifacts/asr-corpus-repeat
python3 prototypes/makepad/validate-asr-ui.py --output artifacts/asr-ui-repeat
```

实模型运行器与界面共用 `asr.rs` 中的参数策略、WAV 读取和本地重采样。
矩阵覆盖四档、尾包及整块边界、窗口边界、空输入、长静音、轻声、少量固定噪声、暂停续说、
中英文、自动语言、变长分包、缓存开关、取消和重启。语料对照在已冻结的 96 句中文集里
事先每隔四条取一条，共 24 句；对照与候选只相差末尾上下文策略。
验收分开记录逐字差异、去标点字符编辑距离和流式不变量，不用“文本相近”替代增量完整性检查。
UI 验证通过按实时时钟输入的 PCM 经过采集缓冲、落盘队列和重采样，但没有打开物理麦克风，
不能替代权限、实际设备、房间噪声和人工试听验证。

最终验证记录：

| 验证 | 结果 |
| --- | --- |
| [70 次实模型场景运行](../artifacts/asr-tail/matrix-final/report.json) | 所有流式、尾部、时间戳及指定文本断言通过；四档短句均输出“你好，欢迎回来。”，时长仍为 1.84 秒 |
| [24 句中文、48 次新旧对照](../artifacts/asr-tail/corpus-final/report.json) | 去标点文字逐句一致；两侧均 8/412 字符编辑错误（1.94%）；3 句仅新增句号 |
| [12 轮 Makepad 界面验证](../artifacts/asr-tail/ui-final/report.json) | 文件/采集 EOF、四档、48 kHz 双声道、18.68 秒暂停续说、停止和重启通过 |
| 核心测试 | 10 项库单元测试 + 8 项小模型/流式测试通过，包含补零前后的特征等价、缓存与时钟不变性 |
| 原生检查 | 9 项原生测试通过，1 项硬件测试沿用忽略；Clippy 与格式检查通过 |

UI 中 3 次主动停止分别约 59/54/60 ms 返回，最后一轮在采集 17.04 秒时打断，
保留已确认的 16.80 秒文字。这些是单次实测，不是延迟上限。两次暂停期间完整句尾均已显示，
16.5 秒时仍保持录音，最终转写时长与真实输入相符。

“通过”不表示所有语音逐字正确：音色设计候选 1 的“你好/您好”差异仍存在，
没有增加文字替换规则；上述语料也仍有原来的 8 个字符错误。轻声和噪声仅为受控变体，
不是全面噪声/方言精度结论。报告保留旧行为控制组与全部差异。
最终构建 SHA-256：`0722b26c0d74e1174ef8c19b5d253307e1fef1ee97fdbd179bcc6038a818135c`。
已更新 `target/release/r2t2-makepad`；旧进程需重启。交付索引见
[`artifacts/asr-tail/delivery.json`](../artifacts/asr-tail/delivery.json)。

## IMK 系统日志排查

用户报告的 `IMKCFRunLoopWakeUpReliable` 消息在本机统一日志中定位到了
`Carbon.framework/.../HIToolbox.framework`，时间为 2026-10-11 12:23:34，
见[日志来源](../artifacts/native-pause/imk-local-log.json)。它是 macOS 输入法相关层输出的诊断。
[Electron 上游同名问题](https://github.com/electron/electron/issues/45002)被维护者归因于 SDK 交互副作用，
并说明其报告场景中的应用功能不受影响；这条旁证不等于所有 Makepad 输入法故障均可忽略。

本轮受控运行没有复现这条消息，也未确认其在本程序中造成中文输入或窗口故障。
若只有终端日志而输入、候选框和窗口均正常，可以继续使用；若伴随实际输入异常，应按具体输入法
与操作过程排查。当前构建没有针对该系统消息进行屏蔽或禁用中文输入，日志仍可能出现。
