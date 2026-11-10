# 合成旋律的 LiquidCodec 保真评估

`omni_rsi::humming_audio` 提供离线单声部 PCM 的 F0／有声估计、LiquidCodec 编解码往返和对 [#87 音符及时钟指标](humming-evaluation.md) 的调用。输入可以由调用者提供 PCM 和音符标注；`synthetic_melody()` 则固定生成一段 24 kHz 单声部信号，无第三方录音。信号由 A3、C4、E4、A4、A3 五个已知音符、间隔静音、1～3 次谐波和 10 ms 淡入淡出组成；音符真值来自合成配方，不来自 F0 估计器。生成代码按本仓库 Apache-2.0 许可提供，输入不使用第三方作品或数据。

这只是合成信号上的离线测量。它不能证明自然哼唱保真、模型跟唱、实时能力或主观听感，也不能代替 #85 所需的合法真人音频、许可核对、隔离测试和听审。报告没有自动质量通过结论，并显式记录自然哼唱证据不足。

## 测量方式

`estimate_pitch` 在原始采样时钟上输出连续的 `PitchSpan`：每帧包含 F0 Hz、连续 MIDI 音高、有声状态和估计起音，`None` 为无浊音。默认每 240 个采样一帧（10 ms），以帧中心为中心取 2,048 采样（约 85 ms）的分析窗，记录边缘以零补足。当前帧的 RMS 低于 0.01 时判为无声；否则用归一化周期差及抛物线插值估计 70～1,000 Hz 的基频，周期差阈值为 0.15。参数均在报告中保存。

从无声转为有声，或相对于上一个估计起音的音高变化达到 0.75 半音，记为新起音。这只是声学启发式：同音重复、颤音、滑音、呼吸、噪声、复音和弱基频／强谐波都可能造成漏检、误检或八度错误。该方法使用前后样本，不是实时估计；帧步长、分析窗、淡入淡出和检测阈值会影响起止与时值误差。合成输入自身也单独评分，便于区分估计器误差与 codec 后的变化。

`AudioEvaluation` 把估计结果传给 #87 指标，调用者须提供参考音符、两边的采样时钟、记录长度和实际可用的输入前缀。此入口固定为 `Offline`，不能证明因果实时性。`measure_codec` 是同一采样时钟上的便捷入口：先评分输入，再以公开的 `CodecEncoder::encode` 和 `Codec::decode` 往返并评分完整解码结果；编码器、解码器和 PCM 的采样率必须一致。调用者应确保两者来自预期的同一检查点。PCM 不平移、拉伸或动态时间规整；输出不足以覆盖输入窗口会报错。codec 填充的尾部被保留在原始波形和长度里，#87 指标按输入时间窗口求值。

报告包含发声率、召回、精确率及误报，音高误差、音高准确率、chroma 和八度错误，匹配／漏检／额外起音、带符号起音偏差和时值误差，以及旋律走势。音高指标只以双方有声重叠为分母，缺失分母时是 `null`；应与发声召回及起音数量一起解读。这里的采样零点是离线信号起点，并非扬声器的物理出声时间。

## 无权重测试

从仓库根目录运行，完全离线且不访问设备：

```sh
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 \
  cargo test --offline --locked --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features cpu \
  --test humming_audio --test humming_codec --test humming_metrics
```

测试用已知 A4 频率、手算谐波采样、静音、八度错误和 +100 ms 偏移检查估计及计分，并用仓库中来自官方参考的微型 codec fixture 检查帧与解码补零。fixture 权重不是训练好的正式权重，测试通过不表示正式 codec 保真已验证。

## 本地正式权重的可重复入口

需已有 LFM3 的本地 `audio_tokenizer` 目录，内有 `config.json` 和 `model.safetensors`（或索引及其全部分片）。入口只读取权重及生成的内存信号，不录音、下载或访问远程服务。以下命令从仓库根目录运行；`RUN_DIR` 必须是**尚不存在**的目录，其父目录须已存在：

```sh
mkdir -p artifacts/humming-codec-runs
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 \
  cargo run --offline --locked --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features metal --example humming_codec -- \
  /absolute/path/to/LFM3-Omni-2B/audio_tokenizer \
  "$PWD/artifacts/humming-codec-runs/run-001"
```

可将 `metal` 换为 `cpu`，但 CPU 正式权重运行可能很慢。运行时应避免并发 GPU 测试和其他重负载。下面记录了一次 CPU 正式权重的合成样本测量；其他后端和硬件的性能尚未在这里测量。

成功后目录包含 `input-f32.wav`、`codec-frames.json`、`decoded-f32.wav` 和 `report.json`。两个 WAV 为 32 位浮点单声道；解码 WAV 保留全部样本，包括帧补零。JSON 保留两套逐帧 F0／有声／起音、完整 #87 指标、原始音符、时钟、参数、实际帧数与长度。PCM 哈希是依次拼接的 IEEE f32 小端采样字节的 SHA-256，不含 WAV 文件头；参数哈希是报告内 `parameters` 对象以 serde_json 紧凑编码后的 SHA-256。

codec 指纹以版本前缀为域，按文件名排序，将名称长度、名称、文件长度和内容计入 SHA-256；覆盖配置、可选索引和索引实际引用的分片（无索引时是 `model.safetensors`）。运行前后分别核对，若指纹不同则报错。报告另记录所列关键源码与 Cargo.lock 的组合指纹；这不是整个二进制或 GPU 驱动的完整构建证明。模型目录、后端及 f32 精度也在报告中；权重文件在运行期间不得修改。

## 本机 CPU 合成样本实测

2026-11-10 使用本机 LFM3-Omni-2B 的正式 LiquidCodec 权重、CPU NdArray 和 FP32，离线处理上述 24 kHz、75,600 采样（3.15 秒）的合成旋律。编码得到 40 帧，解码得到含补零的 76,800 采样；评分使用原始输入时间窗口，未平移或拉伸。codec 指纹 SHA-256 为 `8cce6cd987da20380300d248a2e8d6b79a59d0360a06842442b9e52df38a05e4`。本机主检出目录中的原始报告位于 `artifacts/audio-rsi-p0/humming-codec-real-20261110/report.json`，与相邻音频、codec 帧一起保留参数、输入和输出的指纹及逐帧结果。`artifacts/` 被 Git 忽略；换机器或克隆仓库不会自带这些本地产物。

| 指标 | 合成输入自身 | codec 解码后 |
| --- | ---: | ---: |
| 输出发声率 | 68.25% | 68.89% |
| 有声召回／精确率 | 100%／100% | 100%／99.08% |
| 有声误报率 | 0% | 2.00% |
| 音高准确率（双方有声时误差不超过 50 cent） | 100% | 96.28% |
| 平均绝对音高误差 | 0.28 cent | 20.03 cent |
| 八度错误率 | 0% | 0% |
| 匹配起音／额外起音 | 5／0 | 5／1 |
| 匹配起音平均绝对偏差 | 0 ms | 0 ms |
| 已匹配音符平均绝对时值误差 | 0 ms | 30 ms |

起音匹配容差为 100 ms；解码后虽匹配全部五个参考起音，另有一个额外起音，且第三个匹配音符的估计时值短了 130 ms。起音偏差和时值均值只覆盖匹配项，不能单独表示完整保真。音高和八度指标也只覆盖双方有声的时间，应结合发声召回与误报阅读。这是一条合成样本和该离线估计器的结果，不能证明真人哼唱、听感或跟唱能力。
