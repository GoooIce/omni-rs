# 闲时中文语音训练实验

`omni-rsi speech-watch` 是本地中文语音训练试验。冻结的 LFM3 生成简短中文回答，本项目的 Breeze-TTS-2 朗读回答，LFM3 的 LiquidCodec 将语音编码成训练码；本项目的 Confucius4-R2T2 转写重解码后的音频，筛出可用样本。闲时训练只更新独立 LoRA，候选必须通过实际生成音频的开发集门槛才保存。原始权重不会被覆盖，候选也不会自动加载到 Makepad、API 或原生 LFM3。

[官方模型卡](https://huggingface.co/LiquidAI/LFM3-Omni-2B)列出中文语音。官方中文 TTS 提示在本机能清楚朗读中文，普通语音对话却可能出现中文文字与语音不一致。旧实验在自生成语音上确实更新了参数，但生成语音没有通过当时的验收；更换语音教师和验收器后，需要重新训练和验证，不能据此声称中文能力已提升。历史对照见 [VALIDATION.md](../VALIDATION.md)。

截至 2026-11-09 18:44，本机新数据有 6 条训练、2 条开发和 3 条测试样本通过核验。闲时调度完成四轮后累计 27 次更新，已评估四个训练 epoch，尚无候选通过：第四个 epoch 的一个候选在 89 帧正常结束，但 R2T2 没有识别出中文；另一个候选在 180 帧仍未结束。第四个调度轮次按计划结束，最新检查点哈希正常；调度器当前没有运行。原模型权重未变。

## 流程与边界

1. `training/prepare_chinese_speech.py` 按训练、开发、测试预先分组的问题生成简短中文回答，再使用本仓库 Breeze-TTS-2 合成普通话。24 kHz 源波形经 LFM3 的 LiquidCodec 编码，再重解码为训练目标；保留源音频、目标音频、语音码、种子和模型／可执行文件指纹。也可以通过 `--answers-from` 复用同一 LFM3 检查点先前生成的文字回答。
2. `training/verify_chinese_speech.py` 再次编码 Breeze 源音频并逐帧对照语音码，重解码后逐样本核对目标波形，再用本仓库 R2T2 在自动语言模式下转写。要求完整终止、至少两个汉字、汉字占转写字母数字的比例至少 0.8，且与回答的汉字相似度至少 0.75。R2T2 不提供这里可用的语言置信度；语言标签可能有误，所以不虚构或依赖该概率。转写仍可能误判，不能等同人工真值。
3. 闲时训练使用官方 PyTorch 模型的语音码损失，冻结底座并训练最后一层的秩 4 LoRA。训练样本中的结束帧不参与内容损失。每次更新后保存 LoRA、优化器和数据游标；模型、codec、教师数据、R2T2 模型／可执行文件、训练代码或核验协议的指纹发生变化时拒绝续训。测试集不参与训练或开发集选取。
4. 每个完整 epoch 都分别用原模型和候选回答开发问题，并用 R2T2 转写实际生成的音频。候选必须正常结束、音频足够长、满足中文比例与文本相似度门槛，且至少在一个原模型未通过的问题上通过。通过后只保存待验证的 `candidate-<epoch>.pt`，不会自动改变产品模型。

训练集至少需要 3 条、开发集至少 2 条、测试集至少 1 条核验通过的样本。开发集很小；通过门槛也不足以证明泛化、英文能力或其他 Omni 模态未退化。正式使用前仍需新的封存集、英语和多模态回归，以及 LoRA 导出／加载到 Rust 的验证。Breeze 权重及其输出遵循其[研究／非商业许可](breeze-tts.md)。

## 本机启动

命令从项目根目录运行，使用已有的 `models/Confucius4-R2T2`、`models/Breeze-TTS-2` 和 LFM3 官方检查点。默认语音数据目录为 `artifacts/chinese-speech/synthetic-breeze-r2t2-v1/`；训练使用 Python／PyTorch，R2T2 和 Breeze 的推理使用本项目 Rust 可执行文件，产品推理仍用 Rust／Burn。需要时可用 `--model`、`--python`、`--data`、`--asr` 和 `--asr-binary` 覆盖路径。

```sh
CARGO_TARGET_DIR=target cargo build --release --bin r2t2
CARGO_TARGET_DIR=target/minicpm-v-cpu cargo build --manifest-path crates/omni-rsi/Cargo.toml

target/minicpm-v-cpu/debug/omni-rsi \
  --state artifacts/chinese-speech/idle-r2t2 speech-watch

target/minicpm-v-cpu/debug/omni-rsi \
  --state artifacts/chinese-speech/idle-r2t2 status
```

R2T2 子进程设置 `CUBECL_AUTOTUNE_LEVEL=full`，以精确形状调优键避免已知的注意力缓存分桶冲突；首次遇到形状可能较慢。进程异常、输出不完整、终止被取消、token 触顶或 GPU 线程 panic 时，验收直接失败，不会将样本或候选判为通过。

启动后默认等待 5 分钟无键鼠输入，且要求交流电、热状态和内存正常、足够磁盘空间以及无已知模型进程；信号不可用时不准入。每轮最多 5 次更新、默认 4 轮；再次运行会从完整检查点续训。Ctrl-C 是显式停止，不自动重启。状态、训练日志、哈希寻址的检查点和评估音频在 `--state` 目录下的 `speech/`。

用户返回或同目录的前台任务请求设备时，调度器通知整个训练进程组退出，只恢复上次完整更新。现有 Makepad、API 和所有其他 GPU 应用尚未接入共同设备租约，进程扫描不能覆盖全部程序，因此没有全系统前台延迟保证。模块不会安装登录启动项，也不会自动常驻。

## 准备种子集

以下使用本机 ddust Python 环境，只作离线训练与核验。`target/release/breeze-tts` 需是启用 Metal 的本项目构建；其他位置可通过 `--tts-binary` 指定。首次安装 OpenCC 依赖可能联网，已有缓存时可给 `uv run` 加 `--offline`。

```sh
SPEECH_MODEL_DIR="$HOME/models/LiquidAI/LFM3-Omni-2B"
SPEECH_PYTHON="$HOME/code/ddust/.venv/bin/python"
SPEECH_DATA_DIR=artifacts/chinese-speech/synthetic-breeze-r2t2-v1
SPEECH_ASR_DIR=models/Confucius4-R2T2

uv run --no-project --python "$SPEECH_PYTHON" python \
  crates/omni-rsi/training/prepare_chinese_speech.py --model "$SPEECH_MODEL_DIR" --data "$SPEECH_DATA_DIR"
uv run --no-project --python "$SPEECH_PYTHON" --with opencc-python-reimplemented python \
  crates/omni-rsi/training/verify_chinese_speech.py --model "$SPEECH_MODEL_DIR" --data "$SPEECH_DATA_DIR" --asr "$SPEECH_ASR_DIR"
```

这台机器以前准备过 LFM3 的文字回答；可在第一条命令后加 `--answers-from artifacts/chinese-speech/synthetic-v1` 复用同一检查点的回答，再重新合成和核验语音。历史测试题一旦用于调参，必须排除并另建未接触的测试集。旧 Whisper 标记的数据和旧训练检查点不能作为新协议的结果直接续用；请使用新数据与新状态目录，保留旧记录以供审计。

`training/check_chinese_voice.py --model "$SPEECH_MODEL_DIR" --asr "$SPEECH_ASR_DIR"` 可用同一个 R2T2 检查一条真实的中文文字／语音交错回答；运行时需要上述 OpenCC 依赖。它只检查一个固定示例，不能代表完整回归。
