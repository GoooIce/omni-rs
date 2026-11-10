# 数字听答目标语音码

此入口为 [数字听答数据计划](number-audio-rsi.md)准备可选目标语音码。先自行准备已核验的 Breeze 答案 WAV 或主动提供的答案 WAV，再显式运行以下本地命令。模型与答案路径按本机位置替换；正式权重运行由协调方低优先级安排：

```sh
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 cargo run --offline --release --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu --example number_audio -- encode-target "$HOME/models/LiquidAI/LFM3-Omni-2B" artifacts/number-audio/answer.wav > artifacts/number-audio/answer-target.json
```

命令成功时输出的 JSON 对象可整体并入计划的对应记录，含 `frames` 和 `target_codec_sha256`，以及原始答案 WAV 的 `target_audio_sha256`、包含 codec 配置和实际加载权重的 `codec_fingerprint_sha256`、模型码本配置摘要、预处理实现摘要和后端。为兼容 #81，未分片模型的 `target_codec_sha256` 是 `model.safetensors` 的内容哈希；分片模型使用 codec 复合指纹。请保留授权来源及原 WAV，便于之后独立核对。

目标 WAV 必须是单声道 24 kHz PCM16 或 float32，非空、有限且不超幅，最长 20 秒、文件不超过 32 MiB。编码前后重新核对 WAV、模型码本配置及 codec 文件；两次编码必须一致。正文需有 4–199 帧且码本值有效，程序仅追加一帧全码本 EOS。中断时不会输出完整结果，但 shell 重定向可能留下空文件，必须检查命令退出状态。此入口不采集音频、不运行 ASR 或 TTS，也不验证语义、来源或声音质量；正式路径标记 `local_liquidcodec_f32`，测试注入路径标记 `caller_supplied_unverified`，两者的 `quality_evidence` 均为 `not_assessed`。

以下 `ignored` 测试由协调方以本地正式权重和事先已有的独立 codec 参考低优先级代跑；它验证编码数值，不证明听答质量：

```sh
LFM3_MODEL="$HOME/models/LiquidAI/LFM3-Omni-2B" LFM3_CODEC_ENCODE_REFERENCE="$PWD/artifacts/number-audio/codec-reference.json" CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 cargo test --offline --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features cpu --test number_target official_codec_target_matches_independent_reference -- --ignored --exact
```
