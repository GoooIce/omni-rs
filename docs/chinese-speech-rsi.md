# 闲时中文语音训练实验

`omni-rsi speech-watch` 是本地中文语音训练试验。它可以使用 Breeze-TTS-2 生成的教师语音，也可以在显式开启后接收用户已经说完的语音。LFM3 的 LiquidCodec 负责编解码，本项目的 Confucius4-R2T2 负责转写；闲时训练只更新独立 LoRA，不覆盖底座。通过开发集门槛的候选可在原生 CLI 的新会话中显式选择，不会自动加载到 Makepad、API 或其他会话。

[官方模型卡](https://huggingface.co/LiquidAI/LFM3-Omni-2B)列出中文语音。官方中文 TTS 提示在本机能清楚朗读中文，普通语音对话却可能出现中文文字与语音不一致。旧实验在自生成语音上确实更新了参数，但生成语音没有通过当时的验收；更换语音教师和验收器后，需要重新训练和验证，不能据此声称中文能力已提升。历史对照见 [VALIDATION.md](../VALIDATION.md)。

截至 2026-11-09 18:44，本机新数据有 6 条训练、2 条开发和 3 条测试样本通过核验。闲时调度完成四轮后累计 27 次更新，已评估四个训练 epoch，尚无候选通过：第四个 epoch 的一个候选在 89 帧正常结束，但 R2T2 没有识别出中文；另一个候选在 180 帧仍未结束。第四个调度轮次按计划结束，最新检查点哈希正常；当时的调度器已停止。原模型权重未变。

2026-11-10 新协议的隔离试验中，以合成中文语音模拟一次主动输入并完成 7 次影子更新。10 条生成开发题上，底座与候选的字符错误率均为 33.3%，均只有 8 条正常结束；候选未通过，没有发布。朗读组的转写全部正确，对话组仍有明显错误。此结果不能代表真人语音输入后的效果。

同日又使用一段不在保留集中的 AISHELL 真人录音及 6 条 Breeze 回放，完成 7 次影子更新。10 条开发题的底座 CER 为 33.3%，候选为 43.6%，正常结束分别为 8 / 10 和 9 / 10；候选同样未通过。R2T2 对这段真人原声的伪标签与参考文字一致，但一条录音不足以代表不同说话人。

## 流程与边界

1. `training/prepare_chinese_speech.py` 按训练、开发、测试预先分组的问题生成简短中文回答，再使用本仓库 Breeze-TTS-2 合成普通话。24 kHz 源波形经 LFM3 的 LiquidCodec 编码，再重解码为训练目标；保留源音频、目标音频、语音码、种子和模型／可执行文件指纹。也可以通过 `--answers-from` 复用同一 LFM3 检查点先前生成的文字回答。
2. `training/verify_chinese_speech.py` 再次编码 Breeze 源音频并逐帧对照语音码，重解码后逐样本核对目标波形，再用本仓库 R2T2 在自动语言模式下转写。要求完整终止、至少两个汉字、汉字占转写字母数字的比例至少 0.8，且与回答的汉字相似度至少 0.75。R2T2 不提供这里可用的语言置信度；语言标签可能有误，所以不虚构或依赖该概率。转写仍可能误判，不能等同人工真值。
3. 用户语音只在 `lfm3-omni --live-pcm --learn-state ...` 显式开启时采集。完整输入必须是 16 kHz 单声道、0.4–15 秒；取消、过长、损坏和重复输入不会成为新样本。样本及来源指纹保存在本地状态目录。处理器先对原声做 R2T2 转写，再对 LiquidCodec 重解码音频做第二次转写并检查一致性。这两个结果来自同一识别器，标签只是伪标签，不是独立真值；样本只能学习朗读自身转写，不会把用户问题当作助手回答。
4. 闲时训练使用官方 PyTorch 模型的语音码损失，冻结底座并训练最后一层的秩 4 LoRA；结束帧以 0.01 的权重参与损失，这是待生成验收验证的实验设置。每次更新后保存 LoRA、优化器和数据游标。一个 epoch 的样本清单固定，新录音只能在下一 epoch 加入；已使用样本丢失或改变、模型／codec／R2T2／代码协议变化时拒绝续训。旧检查点不会被覆盖或自动迁移。
5. 每个完整 epoch 用相同生成参数分别评估原模型与候选。固定开发集至少各有 5 条朗读和对话题，候选全部正常结束且有声音，整体字符错误率须相对降低至少 10%，两个分组都不得退步，基线错误率为零时不能声称相对改善。参考文本及独立保留的 45 条最终测试题见 [chinese-speech-generation-v1.json](eval/chinese-speech-generation-v1.json)，其中 15 条绑定现有 AISHELL 真人录音清单；最终测试不参与候选选择。通过后会生成带模型、权重及开发证据的候选包，原生加载器重新计算门槛并校验指纹。

种子数据合同仍要求至少 3 条训练、2 条损失监测及 1 条保留样本；另有上述固定生成验收题。当前尚无真实候选通过完整开发门槛，最终测试、英语及其他 Omni 回归尚未形成候选报告，因此不能声称效果改善。小规模伪标签和同一 ASR 的复核也不能保证真人语音标签正确。Breeze 权重及其输出遵循其[研究／非商业许可](breeze-tts.md)。

## 在线语音：启用、状态与停止

以下命令使用独立状态目录，保留历史 `idle-r2t2` 检查点。先启动闲时调度器，再将用户主动提供的 16 kHz 单声道 little-endian `f32` PCM 流送入 CLI；项目不会自行启动麦克风。两个命令必须使用同一个状态目录，前台会持有协作锁并等待后台让出设备。

```sh
SPEECH_STATE=artifacts/chinese-speech/online-r2t2
SPEECH_MODEL="$HOME/models/LiquidAI/LFM3-Omni-2B"

target/minicpm-v-cpu/debug/omni-rsi --state "$SPEECH_STATE" \
  speech-watch --online --rounds 1000 --epochs 100 --updates-per-round 1

# 另一个终端：标准输入由调用方提供，不会自动采集麦克风。
target/minicpm-v-cpu/debug/lfm3-omni --model "$SPEECH_MODEL" \
  --live-pcm --live-route stable --live-language Chinese --learn-state "$SPEECH_STATE" --json < input.f32

target/minicpm-v-cpu/debug/omni-rsi --state "$SPEECH_STATE" status
target/minicpm-v-cpu/debug/omni-rsi --state "$SPEECH_STATE" stop
```

`status` 展示排队、处理中、已核验、拒绝和候选包数量；`stop` 请求调度器及其子进程让出设备，随后用 `status` 确认停止。录音保存在该状态目录下的 `inbox`、`processing`、`processed` 或 `rejected`，核验样本保存在 `online-data`；不会上传。队列上限为 128 条。处理器认领时会检查清单和录音的完整性，损坏的录音直接移入拒绝目录，不会加载核验模型。处理异常可恢复，不能确认的音频不会入训。

固定保留集的 15 条真人录音带有原始 PCM 指纹；录音入队后以及训练续跑前会排除完整匹配的保留录音，训练文字也不能与保留题相同。这不能识别裁剪、变速或其他变换后的相似录音，也不能保证按说话人隔离。原始录音需留在状态目录中供续训时复核；缺失或变化时会拒绝续训。

候选经过开发门槛后，可在启动新的 CLI 会话时追加 `--speech-adapter 路径`。加载器核对冻结底座指纹、权重哈希、目标层和固定开发题的逐条结果；不匹配时明确报告并继续使用底座。去掉该参数即在新会话回退。Rust API 的缓存会在适配器切换时失效，不能把旧缓存用于新权重。`omni-rsi rollback` 管理的是图表阅读实验，不会切换语音 LoRA。

开发集合格的候选可以另行运行一次保留集报告；命令会再次验证候选、R2T2、底座、数据划分及 AISHELL 源音频哈希，只把结果写入新目录，不改变候选或训练状态。报告不会自动运行英语及其他 Omni 回归，也不能用于反复调参。

```sh
uv run --offline --no-project --python "$HOME/code/ddust/.venv/bin/python" \
  --with opencc-python-reimplemented python crates/omni-rsi/training/evaluate_speech_candidate.py \
  --model "$SPEECH_MODEL" --data artifacts/chinese-speech/synthetic-breeze-r2t2-v1 \
  --online "$SPEECH_STATE/online-data" --candidate "$SPEECH_STATE/speech/candidates/候选ID" \
  --asr models/Confucius4-R2T2 --asr-binary target/release/r2t2 \
  --output "$SPEECH_STATE/held-out/候选ID"
```

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
