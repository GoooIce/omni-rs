# 闲时中文语音训练实验

`omni-rsi speech-watch` 是本地中文语音训练试验。它可以使用 Breeze-TTS-2 生成的教师语音，也可以在显式开启后接收用户已经说完的语音。LFM3 的 LiquidCodec 负责编解码，本项目的 Confucius4-R2T2 负责转写；闲时训练只更新独立 LoRA，不覆盖底座。通过开发集门槛的候选可在原生 CLI 的新会话中显式选择，不会自动加载到 Makepad、API 或其他会话。

[官方模型卡](https://huggingface.co/LiquidAI/LFM3-Omni-2B)列出中文语音。官方中文 TTS 提示在本机能清楚朗读中文，普通语音对话却可能出现中文文字与语音不一致。旧实验在自生成语音上确实更新了参数，但生成语音没有通过当时的验收；更换语音教师和验收器后，需要重新训练和验证，不能据此声称中文能力已提升。历史对照见 [VALIDATION.md](../VALIDATION.md)。

截至 2026-11-09 18:44，本机新数据有 6 条训练、2 条开发和 3 条测试样本通过核验。闲时调度完成四轮后累计 27 次更新，已评估四个训练 epoch，尚无候选通过：第四个 epoch 的一个候选在 89 帧正常结束，但 R2T2 没有识别出中文；另一个候选在 180 帧仍未结束。第四个调度轮次按计划结束，最新检查点哈希正常；当时的调度器已停止。原模型权重未变。

2026-11-10 新协议的隔离试验中，以合成中文语音模拟一次主动输入并完成 7 次影子更新。10 条生成开发题上，底座与候选的字符错误率均为 33.3%，均只有 8 条正常结束；候选未通过，没有发布。朗读组的转写全部正确，对话组仍有明显错误。此结果不能代表真人语音输入后的效果。

同日又使用一段不在保留集中的 AISHELL 真人录音及 6 条 Breeze 回放，完成 7 次影子更新。10 条开发题的底座 CER 为 33.3%，候选为 43.6%，正常结束分别为 8 / 10 和 9 / 10；候选同样未通过。R2T2 对这段真人原声的伪标签与参考文字一致，但一条录音不足以代表不同说话人。

上述结果来自历史 PyTorch 实验。现在种子数据准备、`speech-watch` 的录音入队、LiquidCodec 编解码、语音损失、LoRA 更新、开发集生成验收与候选发布均走 Rust／Burn；Breeze-TTS 和 R2T2 使用本仓库的 Rust 可执行文件。模型对照脚本保留为开发测试工具。新的 Burn 检查点与历史 PyTorch 检查点分开保存，不自动互相续训。

2026-11-10 的隔离 Rust／Burn 实验以本机 6 条种子完成一轮 Metal 训练，并用固定 10 题与 R2T2 比较：底座 CER 为 60.3%，候选为 87.2%，正常结束分别为 10/10 和 8/10。候选未通过门槛，没有发布；该开发集结果不能代表最终保留集表现。闲时调度的 mock 验证独立进行。

音频条件听答有独立的 Rust／Burn 前台入口和显式 `audio-watch` 闲时入口：输入音频经推理使用的 WAV 解码、降混、重采样及媒体绑定进入上下文；答案文字必需，语音码和结束帧可选，答案不放进输入提示。[训练步骤](../crates/omni-rsi/src/speech_audio.rs)和[独立检查点](../crates/omni-rsi/src/speech_audio_checkpoint.rs)分别处理更新和恢复，检查点保存在 `audio-conditioned-v1` 子目录。该路线不自动运行生成验收或候选发布；规则真值数据与独立评估仍由 [#82](https://github.com/GoooIce/omni-rs/issues/82) 处理。现有 `speech-watch` 仍执行自朗读路线；数值对照和参数更新不代表听答能力已提高。

[语气数据与反事实评估合同](speech-tone-contract.md)是独立的 Rust 准入与已有预测汇总基础设施，保留多人分歧、弃判及逐题来源；它尚未接入训练或模型推理，也不证明真实语气识别能力。

检查点绑定训练源码。更新相关实现后，旧源码生成的检查点会拒绝续训；历史状态不被删除或自动迁移。开发用官方模型对照脚本只生成测试参考，实际训练和优化仍使用 Rust／Burn。

本次修复改变了语音训练和 R2T2 核验实现，因此旧的音频条件及自朗读检查点会因实现指纹不匹配而拒绝续训，不能据此覆盖原状态；请保留原目录并使用新状态目录。R2T2 核验现在检查所有分块的 token 上限：旧版原生核验记录没有保存足以证明逐块完整性的证据，其原生核验指纹也随之变化，需重新核验后才能按当前原生指纹准入；历史 `legacy` 核验协议的指纹保持不变。

模型和 codec 有索引时，模型来源指纹绑定实际加载的索引与分片；同时存在单文件时索引优先。此前错误地绑定单文件的旧分片检查点会被拒绝。没有索引的模型和 codec 在普通文件布局下保留原有文件指纹算法与目标码哈希；codec 的配置或权重若通过符号链接提供，现在也会纳入来源指纹，此前漏计的旧指纹会不匹配。上述实现指纹变更仍会阻止旧检查点直接恢复。音频条件目标的 `target_codec_sha256`：无索引时仍为单个 `audio_tokenizer/model.safetensors` 的 SHA-256；有索引时使用 `omni_rsi::fingerprint::codec_fingerprint` 对 codec 配置、索引和实际分片计算的指纹，旧单文件哈希不能代表分片 codec。

## 音频条件前台训练

训练数据目录须含 `manifest.json`，引用目录内的 WAV。每条记录显式声明 `split: "train"`；开发集、测试集、缺少分集或其他分集会被拒绝。一个最小文本目标清单如下（哈希为示意，需替换成实际文件的 SHA-256）：

```json
{
  "version": 1,
  "records": [{
    "id": "example-1", "split": "train",
    "input_audio": "example.wav", "input_sha256": "<WAV 的 SHA-256>",
    "authorization": {"kind": "licensed-synthetic", "source": "合成来源", "license": "许可标识"},
    "instruction": "把听到的数字加一，用中文回答。", "answer": "十三"
  }]
}
```

主动提供的录音可声明 `kind: "user-provided"`、`source` 和 `consent`。这些字段是调用方的来源和授权声明，加载器无法独立证实授权或答案真值；使用前须由数据提供者确认。语音监督记录还须提供 `frames`（含最后一帧所有码本的 EOS）及上述 `target_codec_sha256`。省略两者即只训练文本。清单原文和每个实际 WAV 字节共同构成数据指纹；数据、媒体、目标、模型、codec、后端或训练实现变化时拒绝恢复。WAV 解码后每条最多 20 秒，所有划分合计最多 1,200 秒（16 kHz 的 f32 PCM 合计最多约 76.8 MB）；加载器还限制记录数和单个文件大小。

输入 WAV 的原始 float 样本须逐声道有限且位于 `[-1, 1]`；不能依靠降混抵消越界值。此前因声道抵消而被接受的越界文件现在会被拒绝，需保留原件并另行提供合规数据。合法波形经降混和重采样后可能出现有限过冲，训练保留解码器输出的实际 PCM，不裁剪或归一化；数据身份仍绑定 WAV 字节，跨划分检查使用实际 PCM。直接调用 `AudioAnswer::new` 或 `new_text_only` 的 PCM 契约仍要求有限且在 `[-1, 1]`，只有经 WAV 校验和解码的内部样本可包含有限过冲。

输入校验、样本装配和闲时调度的源码变更会改变 legacy 与 number-v3 音频训练实现身份，旧音频检查点不会自动迁移，须保留原状态和原程序。v3 数据身份、开发预注册及绑定相关源码的评分身份也可能改变；旧冻结记录和报告不能当作新协议结果。闲时调度接线也改变了 #79 自朗读实现摘要；其旧检查点同样须保留原程序与状态，不会自动迁移。

当前的闲时探针、监督和报告修复再次改变 #79 自朗读及其后代 legacy／v3 音频实现身份；这些路线的旧程序与检查点须一起保留，当前程序拒绝恢复不匹配状态，不会改写或迁移它们。

每个记录代表**完整输入已经结束之后**的一个回答目标，全部 WAV 位于答案之前。它不表达输入进行中不同时间的目标；要训练某个前缀，只能提供该时刻已可用的音频，不能把后续录音放入 WAV。实时分段和带时间戳的因果训练不在此入口内。

显式使用本地已授权数据运行，首次运行要求 `audio-conditioned-v1` 不存在；恢复必须带 `--resume`。`--epochs` 是希望达到的总 epoch，`--max-updates` 是本次调用最多更新数，可在恢复时调整；固定学习率、损失权重、掩码规则和初始化由实现指纹绑定。每次完整更新后保存参数、AdamW 状态和下一条记录的位置；中断时保留最后一个完整检查点。损坏、旧版或不匹配的音频检查点会报错，必须另选状态目录，不自动迁移或覆盖；原有自朗读检查点保持隔离。

```sh
CARGO_TARGET_DIR=target/minicpm-v-cpu cargo run --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu \
  --bin omni-rsi-audio -- --model "$HOME/models/LiquidAI/LFM3-Omni-2B" \
  --data /path/to/authorized-training-data --state /path/to/rsi-state \
  --epochs 2 --max-updates 1
# 继续：使用相同参数，末尾加 --resume
```

前台音频命令持有同一 `--state` 下的 `foreground.lock` 和 `device.lock`。不同 state 目录的 RSI 设备锁彼此独立；未接入这些协作锁的其他应用也不受协调。直接调用 Rust 训练接口的程序须自行持有设备租约。

### 闲时音频条件训练

闲时路线使用同样的已授权本地模型与音频数据，不录音、不自动发布或晋级。`audio-watch` 必须明确选择首次 `--fresh` 或从完整状态 `--resume`；`--number-v3` 选择 v3，省略它选择 legacy。如下示例从项目根目录运行，先按[项目工作流](agent-workflow.md)准备共享 target：

```sh
uv run --no-project python scripts/share-target.py
CARGO_TARGET_DIR=target/minicpm-v-cpu cargo run --offline --locked \
  --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu \
  --bin omni-rsi -- --model "$HOME/models/LiquidAI/LFM3-Omni-2B" \
  --state /path/to/rsi-state audio-watch --data /path/to/authorized-training-data \
  --fresh --epochs 2 --updates-per-round 1 --rounds 4
# 已有完整匹配检查点时，用 --resume 替换 --fresh；v3 还须添加 --number-v3。
```

`--epochs` 是总目标 epoch，`--updates-per-round` 是每次闲时运行最多的更新数，二者均为 1–100；`--rounds` 是本次调度最多完成的轮次（1–1000）。需要配合前台协作的命令须使用同一 `--state` 根目录。legacy 检查点位于 `audio-conditioned-v1/`，v3 位于 `number-audio-v3/audio-conditioned-v1/`；两条路线的首次启动标记分别是 `audio-idle-legacy-started` 和 `audio-idle-v3-started`。新的 v3 预留也在其检查点命名空间中保存 `audio-idle-v3-namespace-started`，使经该命名空间的别名进入的 legacy 前台、调度器和 Rust 训练接口都不能把已预留状态当成 fresh。旧版只在父目录留下标记的情况，按原有 `number-audio-v3` 路径及其规范路径检查；无法枚举其他任意历史符号链接父目录。

调度器复用空闲策略、watchdog 和 RSI 设备租约，并在受监督的子进程组中执行训练。前台到来或请求停止时，先取消并等待整个进程组退出，再释放设备租约；先尝试协作取消，10 秒后强制终止，系统若不能结束进程组则继续等待。每次完整更新的检查点按现有协议保存，后续闲时只从完整检查点继续。首次启动标记在 worker 启动前持久化；若中断时没有完整 pointer，状态会报告错误，不能自动续训或再次按 fresh 使用该状态，程序不清理它。已存在、损坏或实现身份不匹配的检查点也不会被覆盖或迁移。预检仅核验 checkpoint 完整性和实现身份，实际模型、codec 与数据匹配仍由 worker 的 `load(expected)` 在训练前校验。

RSI 自身的设备协调以相同状态目录中的租约为准，音频 worker 的监督只验证 RSI 自身的租约和连接。所有协作租约都要求普通文件，并用新打开描述符的非阻塞标志避免等待 FIFO 等特殊文件；不会改变调用方已有描述符的标志。

worker 的 stdout 和 stderr 分别在运行时限制为 1 MiB 和 64 KiB，超限时取消进程组；输出不写入无界临时文件。Unix 下通过继承的协作租约描述符及监督 socket 连接 worker，监督端消失时请求取消；公开的受监督 Rust 训练入口也会在整个 fresh 或 resume 调用期间监视连接。取消仍须等训练到达已有的可取消边界。非 Unix 失败关闭。这些机制要求受控本地进程及共享状态目录，不防御同账号刻意伪造描述符或进程脱离受监督进程组。

CLI 在启动时包装原有 panic hook，仅对已捕获的空闲探针异常和持设备租约的调度线程异常省去同步诊断；其他线程仍调用原 hook。每个调度器在取得租约之前启动一个探针线程；单次探测超时、异常或失联会取消工作并结束当前调度。同进程内最多允许 32 个探针线程；有已超时且未返回的探针时不再启动新的探针。macOS 进程采样的 `ps` 有超时和输出上限，并回收可终止的进程组；系统调用或进程处于不可中断状态时不能保证它被终止，下一轮保持失败关闭。直接嵌入 `Watchdog` 的程序不会安装 CLI 包装；若其他代码随后替换 hook，上述诊断和让出保证不再成立。

调度器每次成功完成的轮次报告都作为新文件保存在状态目录的 `idle-round-reports/`，不会覆盖已有报告。已经完成验证的成功结果即使与让出请求交错，也会保存并计数；若取消发生在 worker 成功退出和报告、检查点验证之前，则按中断处理，不推断该轮已成功。文件先写入临时名称并同步，再以不可覆盖的方式发布；异常中断可能留下 `.pending-*.tmp`，不能将其当作完整报告。每条报告经有界序列化后连末尾换行最多 1 MiB；超限时记录错误，不创建该报告文件，也不计入完成轮次。训练过程若先前已保存完整检查点，超限错误不会撤销它。序列化缓冲和写盘受此上限约束，生成报告的上游任务本身仍须遵守各自的内存预算。

stdout 和等待提示采用有界、尽力交付：接收端长期不读取时，实时输出可能延迟或丢失，状态文件会保留未确认的报告路径，原始 JSON 仍可从报告目录读取。写入确认只表示操作系统接受了字节，不证明接收端已处理。单次调度最多运行 1,000 轮；报告跨运行持续保留，没有自动累计容量上限或清理，使用者须自行监控空间并在调度停止后按需归档。写入状态或报告的文件系统本身若不能响应，不能保证停止延迟。系统无法结束进程组时仍须等待，不能把租约提前释放当成设备已停。

轮次结果的 `training_status` 为 `partial` 或 `complete`，`quality` 始终为 `not_assessed`；达到 `--rounds` 而未完成 epoch 时状态为 `partial`，显式停止为 `stopped`，失败为 `error`。即使训练状态为 `complete`，也只表示达到配置的训练进度，不代表听答能力、质量验收或晋级资格。

闲时命令不启动麦克风、下载数据、发布候选或更改已运行的会话。前台请求持有 `foreground.lock`；调度器检查该锁，并在工作期间持有 `device.lock`，它本身不持有前台请求锁。

## 流程与边界

1. 已有种子集按训练、开发、测试预先分组，由冻结的 LFM3 和本仓库 Breeze-TTS-2 生成中文文字与语音。24 kHz 源波形经 LFM3 的 LiquidCodec 编码，再重解码为训练目标；记录源音频、目标音频、语音码及模型／可执行文件指纹。
2. 种子集保留此前的核验证据。新的用户录音由 Rust 处理器编码和重解码，再用本仓库 R2T2 自动语言模式转写。要求完整终止、至少两个汉字、汉字占转写字母数字的比例至少 0.8，且与参考的汉字相似度至少 0.75。R2T2 的语言标签可能有误，不把它当作置信度；转写也不能等同人工真值。
3. 用户语音只在 `lfm3-omni --live-pcm --learn-state ...` 显式开启时采集。完整输入必须是 16 kHz 单声道、0.4–15 秒；取消、过长、损坏和重复输入不会成为新样本。样本及来源指纹保存在本地状态目录。处理器先对原声做 R2T2 转写，再对 LiquidCodec 重解码音频做第二次转写并检查一致性。这两个结果来自同一识别器，标签只是伪标签，不是独立真值；样本只能学习朗读自身转写，不会把用户问题当作助手回答。
4. 闲时训练使用 Burn 自动求导和 AdamW 实现语音码损失，冻结底座并训练最后一层的秩 4 LoRA；结束帧以 0.01 的权重参与损失，这是待生成验收验证的实验设置。每次更新后保存 LoRA、优化器和数据游标。一个 epoch 的样本清单固定，新录音只能在下一 epoch 加入；已使用样本丢失或改变、模型／codec／R2T2／代码协议变化时拒绝续训。新旧检查点不会互相覆盖或自动迁移。
5. 每个完整 epoch 用相同生成参数分别评估原模型与候选。固定开发集至少各有 5 条朗读和对话题，候选全部正常结束且有声音，整体字符错误率须相对降低至少 10%，两个分组都不得退步，基线错误率为零时不能声称相对改善。参考文本及独立保留的 45 条最终测试题见 [chinese-speech-generation-v1.json](eval/chinese-speech-generation-v1.json)，其中 15 条绑定现有 AISHELL 真人录音清单；最终测试不参与候选选择。通过后会生成带模型、权重及开发证据的候选包，原生加载器重新计算门槛并校验指纹。

种子数据合同仍要求至少 3 条训练、2 条损失监测及 1 条保留样本；另有上述固定生成验收题。当前尚无真实候选通过完整开发门槛，最终测试、英语及其他 Omni 回归尚未形成候选报告，因此不能声称效果改善。小规模伪标签和同一 ASR 的复核也不能保证真人语音标签正确。Breeze 权重及其输出遵循其[研究／非商业许可](breeze-tts.md)。

## 分开验证调度和语音质量

闲时调度使用 mock 验证：模拟空闲信号和工作负载，检查用户返回、前台设备请求、让出后恢复、无任务等待以及停止；不加载模型，也不根据训练效果判定调度通过。运行：

```sh
CARGO_TARGET_DIR=target/minicpm-v-cpu cargo test --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features native,cpu --bin omni-rsi mock_
```

语音质量使用真实模型单独验证。`speech-once` 是前台、限量的训练入口，不依赖闲时信号；它使用同目录的前台和设备锁，训练完整 epoch 后对固定开发题分别生成基线和候选，保存 R2T2 转写、字符错误率与结束情况。可在独立状态目录执行：

```sh
SPEECH_MODEL="$HOME/models/LiquidAI/LFM3-Omni-2B"
QUALITY_STATE=artifacts/chinese-speech/quality-native-v1
target/minicpm-v-cpu/debug/omni-rsi --model "$SPEECH_MODEL" --state "$QUALITY_STATE" \
  speech-once --data artifacts/chinese-speech/synthetic-breeze-r2t2-v1 \
  --epochs 1 --updates-per-round 6
```

本机种子训练集当前有 6 条，完整 epoch 的报告位于 `$QUALITY_STATE/speech-native/epoch-1.json`。这是开发集比较，不是最终保留集结果；前台执行会占用计算资源，任务中断后可用相同命令和状态目录续跑。历史 PyTorch 对照的候选未通过开发门槛；Rust/Burn 的效果以其独立报告为准，mock 调度通过不能证明语音质量提高。

报告生成后可用同一份固定题集独立汇总 CER、正常结束数量和开发门槛，不加载模型：

```sh
CARGO_TARGET_DIR=target/minicpm-v-cpu cargo run --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features native,cpu --example speech_quality_report -- \
  "$QUALITY_STATE/speech-native/epoch-1.json"
```

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

开发集合格的候选可以另行运行一次 Rust 保留集报告；命令会再次验证候选、R2T2、底座、数据划分及 AISHELL 源音频哈希，只把结果写入新目录，不改变候选或训练状态。报告不会自动运行英语及其他 Omni 回归，也不能用于反复调参。

```sh
target/minicpm-v-cpu/debug/omni-rsi --model "$SPEECH_MODEL" --state "$SPEECH_STATE" \
  speech-evaluate --data artifacts/chinese-speech/synthetic-breeze-r2t2-v1 \
  --online "$SPEECH_STATE/online-data" --candidate "$SPEECH_STATE/speech-native/candidates/候选ID" \
  --asr models/Confucius4-R2T2 --asr-binary target/release/r2t2 \
  --output "$SPEECH_STATE/held-out/候选ID"
```

## 本机启动

命令从项目根目录运行，使用已有的 `models/Confucius4-R2T2` 和 LFM3 官方检查点。默认语音数据目录为 `artifacts/chinese-speech/synthetic-breeze-r2t2-v1/`；训练与在线核验使用 Rust／Burn，本项目 R2T2 可执行文件负责识别。需要时可用 `--model`、`--data`、`--asr` 和 `--asr-binary` 覆盖路径。

```sh
CARGO_TARGET_DIR=target cargo build --release --bin r2t2
CARGO_TARGET_DIR=target/minicpm-v-cpu cargo build --manifest-path crates/omni-rsi/Cargo.toml

target/minicpm-v-cpu/debug/omni-rsi \
  --state artifacts/chinese-speech/idle-r2t2 speech-watch

target/minicpm-v-cpu/debug/omni-rsi \
  --state artifacts/chinese-speech/idle-r2t2 status
```

R2T2 子进程设置 `CUBECL_AUTOTUNE_LEVEL=full`，以精确形状调优键避免已知的注意力缓存分桶冲突；首次遇到形状可能较慢。进程异常、输出不完整、终止被取消、token 触顶或 GPU 线程 panic 时，验收直接失败，不会将样本或候选判为通过。

启动后默认等待 5 分钟无键鼠输入，且要求交流电、热状态和内存正常、足够磁盘空间以及无已知模型进程；信号不可用时不准入。每轮最多 5 次更新、默认 4 轮；再次运行会从完整检查点续训。Ctrl-C 是显式停止，不自动重启。Rust 状态、训练日志、哈希寻址的检查点和评估音频在 `--state` 目录下的 `speech-native/`；历史 `speech/` 保持独立。

用户返回或同目录的前台任务请求设备时，调度器通知整个训练进程组退出，只恢复上次完整更新。现有 Makepad、API 和所有其他 GPU 应用尚未接入共同设备租约，进程扫描不能覆盖全部程序，因此没有全系统前台延迟保证。模块不会安装登录启动项，也不会自动常驻。

## 原生种子集准备

本机已有上述核验后的种子集，可直接供 Rust 训练使用。需要重新准备时，`speech-seed` 使用 LFM3 生成中文回答、Breeze-TTS 合成、LiquidCodec 转码并通过 R2T2 核验；每条样本在独立临时目录完成后原子发布。两个语音可执行文件需是本项目启用 Metal 的构建。

```sh
SPEECH_MODEL="$HOME/models/LiquidAI/LFM3-Omni-2B"
SPEECH_STATE=artifacts/chinese-speech/native-seed-state
SPEECH_DATA=artifacts/chinese-speech/native-seeds-v1

target/minicpm-v-cpu/debug/omni-rsi --model "$SPEECH_MODEL" --state "$SPEECH_STATE" \
  speech-seed --data "$SPEECH_DATA" \
  --tts-model models/Breeze-TTS-2 --tts-binary target/release/breeze-tts \
  --asr models/Confucius4-R2T2 --asr-binary target/release/r2t2
```

这台机器以前准备过 LFM3 的文字回答；可追加 `--answers-from artifacts/chinese-speech/synthetic-breeze-r2t2-v1` 复用同一检查点的回答，再重新合成和核验语音。`--only greeting` 可只处理一条；已完成且指纹一致的样本会保留。新教师使用新数据目录，避免与历史教师指纹混合；用训练命令的 `--data "$SPEECH_DATA"` 选择它。历史测试题一旦用于调参，必须排除并另建未接触的测试集。旧 Whisper 标记的数据和旧训练检查点不能作为新协议的结果直接续用；保留旧记录以供审计。

历史 Python 脚本和模型对照脚本仅供复现或独立比对，不参与原生数据准备或训练。`training/check_chinese_voice.py` 可用于历史单例对照，不能代表完整回归。
