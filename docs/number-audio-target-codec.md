# 数字听答目标语音码

此入口为 [数字听答数据计划](number-audio-rsi.md)准备可选目标语音码。先准备已核验的 Breeze 答案 WAV 或主动提供的答案 WAV，再运行以下本地命令。请将模型、答案和输出路径替换为本机路径，并选用尚不存在的输出文件。

命令从仓库根目录运行，使用项目共享的 `target/minicpm-v-cpu` 编译目录。首次在独立工作树运行 Cargo 前，请按[开发工作流](agent-workflow.md#compilation-directories)执行 `uv run --offline --no-project python scripts/share-target.py`。命令以离线模式运行，需要提前缓存所需依赖；首次编译和加载模型可能耗时。

```sh
( set -C; CARGO_TARGET_DIR=target CARGO_BUILD_JOBS=2 cargo run --offline --locked --release --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu --example number_audio -- encode-target "$HOME/models/LiquidAI/LFM3-Omni-2B" artifacts/number-audio/answer.wav > artifacts/number-audio/answer-target.json )
```

命令成功时输出的 JSON 对象可整体并入计划的对应记录，含 `frames`、原始答案 WAV 的 `target_audio_sha256`、codec 身份、模型码本配置摘要、预处理实现摘要和后端。正式编码从同一份内存快照计算包含配置、索引（若有）和权重的 `codec_fingerprint_sha256` 并构造编码器；快照总量最多 1 GiB，索引最多 8 MiB。为兼容 #81，未分片模型的 `target_codec_sha256` 仍是 `model.safetensors` 的内容哈希，分片模型使用 codec 复合指纹。训练准入遇到 `codec_fingerprint_sha256` 时会校验当前 codec 的完整复合指纹；该字段一旦提供就必须是有效摘要，`null` 不视为缺省。没有该字段的旧记录继续按原有的 `target_codec_sha256` 准入，未分片旧记录因此不能证明配置一致。请保留授权来源及原 WAV，便于之后独立核对。

目标 WAV 必须是单声道 24 kHz PCM16 或 float32，非空、有限且不超幅，最长 20 秒、文件不超过 32 MiB。编码前后重新核对 WAV、模型码本配置及 codec 文件；两次编码必须一致。正文需有 4–199 帧且码本值有效，程序仅追加一帧全码本 EOS；最多 128 个码本、25,600 个码（含 EOS）。示例中的 `set -C` 防止重定向覆盖已有输出。中断时不会输出完整结果，但重定向可能留下空文件，必须检查命令退出状态。取消会在分块读取和编码前后检查；正在执行的系统文件 I/O 或单次模型运算不能被此标志立即中断，建议使用可靠的本地存储。

此入口不采集音频、不运行 ASR 或 TTS，也不验证语义、授权或声音质量。正式路径初始标记 `local_liquidcodec_f32`，测试注入路径初始标记 `caller_supplied_unverified`，两者的 `quality_evidence` 均为 `not_assessed`。JSON 可以被编辑，这些字符串、文件哈希和帧都不是运行来源或 WAV 与帧对应关系的认证证明；正式验收仍需独立受控重编码和来源、语义及质量核验，不得仅凭这些字段放开门控。

`preparation_sha256` 的 v2 摘要覆盖目标准备、PCM 解码、codec、相关共享张量源码及所列构建依赖信息；它不证明二进制、编译器、硬件或第三方源码的完整身份，不能与旧版摘要直接比较。本次共享解码与训练准入修复也改变训练实现指纹：已有通用语音及音频条件训练检查点不会自动迁移或恢复。请保留原状态，使用新的空状态目录重新开始；旧数据格式仍可按上述兼容规则读取。

若本地已有正式模型权重和独立生成的 codec 参考文件，可运行以下可选的 `ignored` 测试；请将模型和参考文件路径替换为实际路径。该测试验证编码数值，不证明听答质量：

```sh
LFM3_MODEL="$HOME/models/LiquidAI/LFM3-Omni-2B" LFM3_CODEC_ENCODE_REFERENCE="$PWD/artifacts/number-audio/codec-reference.json" CARGO_TARGET_DIR=target CARGO_BUILD_JOBS=2 cargo test --offline --locked --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features cpu --test number_target official_codec_target_matches_independent_reference -- --ignored --exact
```
