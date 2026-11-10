# 固定前缀的 speech-only 训练原语

`omni_rsi::humming::training` 为 #85 提供一个显式调用的 Rust/Burn 内存训练原语。调用者提供已经自行准入的 16 kHz 单声道伴奏**固定前缀**、任务指令和目标语音码；本接口不读取录音、不从文件加载训练数据，也不启动闲时调度。它不接受文本答案，不添加答案 token，也不计算文本损失。原有音频问答训练保持独立。

`FixedPrefix::new` 要求前缀非空、不超过 320,000 个采样、值有限且在 [-1, 1]；指令非空、不超过 2048 字节且不能包含控制 token 起始符。目标包含 5–200 帧，训练前检查模型要求的码本数、词表边界和全码本结束帧。接口不读取 codec 配置或验证目标码的来源；调用者还须确保码值和码本与实际目标 codec 相符。来源授权、旋律/歌手/派生版本划分和样本本身是否适用于训练也由调用者负责；`FixedPrefix` 的构造成功不证明这些事实。

`Trainer` 持有单独的 `SpeechOptimizer` 和显式目标卷积层；其 `step` 用 `OmniModel::prepare_audio` 编码传入的整个切片，再通过既有 `speech_loss_with_media` 计算语音内容损失和结束帧损失，按 `content + 0.01 * end` 更新 LoRA。返回值标识目标 `omni-rsi.humming.fixed-prefix.speech-only.v1` 并报告两项损失及梯度范数。成功返回时，更新后的 LoRA 同时安装在传入模型，供直接推理；错误和已观察的取消不提交本步更新。训练器在更新前保留临时内存副本，若更新中观察到取消则回退优化器和模型 adapter。调用者必须串行使用训练器及其模型，adapter 更换后还须丢弃该模型先前的推理缓存；若取消恰在最后一次检查之后发生，已提交的步骤不会被追溯撤销，单次模型前向/反向内部也不保证立即停止。

该接口只证明计算接收了调用者提供的切片。它不接受或推断采样时间戳，不能证明每个目标帧只依赖实时可见的伴奏；逐帧对齐、固定延迟与实时因果跟唱需要后续更强的时间界面。训练器没有检查点、恢复、候选发布或晋级接口；`weights()` 只供检查当前内存参数，不能当作经过身份绑定的检查点。生产接线前仍需设计绑定模型、数据、任务版本、层、优化器状态及采样规则的独立协议和失败关闭的恢复/门控路径，不能把此目标写入 #79 或文本答案训练状态。

合成测试把仓库已有的小型文本、音频编码器、语音深度解码器和 codec 夹具组合并裁剪到一致的文本宽度；codec 仅为构造模型而加载，测试不运行 codec，也不声称合成语音码来自该 codec。两种合成正弦前缀在同初值和随机种子下产生不同的语音损失及梯度；相同切片配不同未传入的尾部则产生相同结果。另一组 24 层合成夹具执行原有音频问答训练，单独验证原文本与语音目标仍工作。这些只是计算接线与边界测试，不能证明真人哼唱、自然 codec 保真、数据许可、听审、跟唱能力改善或真实闲时调度质量。

本工作树的离线 CPU 测试入口（私有 target 和日志位于被 Git 忽略的 `artifacts/`）：

```sh
CARGO_TARGET_DIR="$PWD/artifacts/humming-train/target" \
CARGO_HOME="$PWD/artifacts/humming-train/cargo-home" \
TMPDIR="$PWD/artifacts/humming-train/tmp" CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true \
  cargo test --offline --locked --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features cpu --test humming_training -- --test-threads=1
```
