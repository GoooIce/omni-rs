# 哼唱的音符、时钟与客观指标

`omni_rsi::humming` 的标注评估接口不读取音频、不检测 F0、不做转谱，也不启动模型或设备。参考音符和输出的有声、音高、起音需由调用者提供；`pitch_midi_from_hz` 只把已知的基频换成连续 MIDI 音高，不能从波形推断音高。单独的 [`humming_audio` 接口](humming-codec-fidelity.md) 提供离线 PCM 估计和合成旋律 codec 评估，[`humming::training`](humming-training.md) 是另行显式调用的固定前缀训练原语。这里的合成示例和测试只能验证时间与指标计算，不能证明自然哼唱或 LiquidCodec 对真人声音的保真度。[#85](https://github.com/GoooIce/omni-rs/issues/85) 的真人数据、codec、听审和对 #82 的依赖仍须另行完成。

## 输入与时间

- `Clock { sample_rate_hz, origin_ns }` 分别描述输入和实际输出采样时钟。`origin_ns` 是同一单调时间域中采样位置 0 的时间，时钟偏移及采样率由调用者测量和校准。采样率须为 1–1,000,000 Hz，记录长度须为 1–2^53 个采样。两边可以使用不同采样率和起点；接口不自动推定设备缓冲、重采样延迟、时钟漂移或扬声器物理出声时间。要评实际播放，输出时钟必须来自有相应校准的播放记录，生成时间或入队时间不能冒充播放时间。
- 输入评估窗口为输入采样 `[0, input_samples)`。参考 `Note` 使用输入采样位置、右开起止和 0–127 的有限连续 MIDI 音高；音符按时间排序且不能重叠，空隙表示已标注的无声。参考缺失不是静音，应先修复数据。根据需要在输入窗口包含足够的前后静音，以便观察迟到的尾音。
- 输出 `OutputSpan` 使用输出采样位置和右开起止。`Some(pitch_midi)` 表示有声并给出该段 F0 的 MIDI 标注；`None` 表示无声或无浊音。跨度必须连续覆盖整个 `[0, output_samples)`，其绝对时间须覆盖输入窗口；缺失输出不会自动补成静音。音高或使用的输入前缀变化时切段，`note_onset` 只在新音符发音处置真，音高变化不必然是新音符。从无声转为有声时必须标起音。
- 每个输出段的 `input_prefix_end` 是该段开始前已经用于生成该段的输入前缀 `[0, end)`；它须单调、不能超过记录的输入长度。`InputMode::Causal` 另外核对前缀末端在输出段开始时已按两个时钟到达；空前缀始终可用。只有全程如实记录前缀才能支持因果性检查：此接口不能验证调用者的记录、排除未记录的模型输入或证明实时能力。`InputMode::Offline` 允许未来输入，报告中保留模式，不得作为实时结果。

## 指标

所有发声和音高时长以两个时钟映射到同一个绝对时间轴，在输入窗口内以整数时钟单位直接求交，再转换为秒和比例；不做时间扭曲、延迟平移或重采样。报告保留输入/输出时钟、长度、模式及起音容差。

- 发声率是输出有声时长除以窗口长度；发声召回是双方有声交集除以参考有声时长；精确率是交集除以输出有声时长；误报率是参考无声时输出有声的时长除以参考无声时长。同时返回三个原始有声时长。没有对应分母时为 `None`，例如全静音输出的音高准确率不是 100%。
- 音高指标仅在双方有声的交集上按时长加权。报告平均绝对音高误差（cent）、50 cent 内的音高准确率、按 1200 cent 取模后 50 cent 内的 chroma 准确率，以及绝对误差超过 50 cent 但 chroma 仍正确的八度错误率。阈值包含边界；漏唱须结合发声召回解读，不能被音高条件分母掩盖。
- 起音在原始绝对时间上按顺序一对一匹配，偏差必须在 `onset_tolerance_seconds` 内；算法优先匹配最多音符，再最小化绝对偏差之和，**不移动音频或修改时间**。报告每对参考索引、输出采样位置、当时输入前缀、带符号偏差及可用的时值偏差，还报告漏音、额外起音、召回/精确率、平均带符号与平均绝对偏差。正值表示输出比伴奏参考起音迟；这些偏差即相对参考的同步延迟。超出容差的事件保留为漏音/额外起音，不参与偏差均值；须同时报告数量及容差。匹配矩阵最多 16,000,000 个格子，过大时返回错误。
- 时值仅比较已匹配且结束已被记录的音符：输出在录音结束时仍有声，其实际结束未知，不进入时值平均值。报告比较音符数，以便发现截断。
- 旋律走势只比较成功匹配的相邻参考音符对；中间漏唱时不跨过缺失音符。输出音符音高是该音符各有声段按时长加权的平均 MIDI 音高。上下行变化大于半音的一半记为升/降，±0.5 半音内记为平；报告全部参考对数、实际比较对数、正确对数和条件准确率。音高漂移、音符分割和 F0 标注质量都会影响结果。

例如输入 250–750 ms 的 A4 与输出 350–850 ms 的 A4，记录的同步偏差是 **+100 ms**，即使两段的音高和时值完全相同：

```rust
use omni_rsi::humming::{Clock, Evaluation, InputMode, Note, OutputSpan};

let clock = Clock { sample_rate_hz: 1_000, origin_ns: 0 };
let notes = [Note { start: 250, end: 750, pitch_midi: 69.0 }];
let output = [
    OutputSpan { start: 0, end: 350, pitch_midi: None, note_onset: false, input_prefix_end: 0 },
    OutputSpan { start: 350, end: 850, pitch_midi: Some(69.0), note_onset: true, input_prefix_end: 350 },
    OutputSpan { start: 850, end: 2_000, pitch_midi: None, note_onset: false, input_prefix_end: 850 },
];
let report = Evaluation {
    input_clock: clock, input_samples: 2_000, notes: &notes,
    output_clock: clock, output_samples: 2_000, output: &output,
    input_mode: InputMode::Causal, onset_tolerance_seconds: 0.15,
}.evaluate().unwrap();
assert_eq!(report.mean_signed_onset_error_seconds, Some(0.1));
```

从仓库根目录运行无设备合成负例：

```sh
CARGO_TARGET_DIR="$PWD/target/minicpm-v-cpu" CARGO_BUILD_JOBS=2 \
  cargo test --offline --locked --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features cpu --test humming_metrics
```
