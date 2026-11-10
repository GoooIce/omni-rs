# 数字听答开发集：隔离入口和预注册基础

`number_development` 提供独立的 train/dev 读取及协议冻结基础。v3 可显式进入下述音频条件训练，但还没有接入正式开发集验收或保留集评估，不能构造 `VerifiedDevelopment`、打开保留集或启用候选。文件格式和测试只约束输入与快照，不证明运行、数据真实性、授权或模型质量。旧版 v2 单清单会暴露 test 内容，只能用于已暴露的实验，不能用作盲验收输入。

## v3 数据边界

调用 `DevelopmentData::load` 时，数据根目录布局如下；`layout.v3` 的内容必须严格为 `omni-rsi-number-train-dev-v3\n`。保留集的秘密清单、盐和媒体由保管人另行隔离，不放在这些文件中。

```text
layout.v3
holdout-public.json
train.groups.v3.json     dev.groups.v3.json
train.v3.json            dev.v3.json
train-media/*.wav        dev-media/*.wav
train-target-media/*.wav dev-target-media/*.wav  # 有语音目标时才需要
regression-plan.json     regression-inputs/*.json
```

公开保留元数据必须是盲格式 v2 的 `PublicMetadata`，只包含承诺和不透明分组；读取方在解析时必须读入公开 JSON，故保管人须确保该文件本身没有秘密。入口先检查 train/dev 的纯分组清单与公开保留分组，比较说话人、会话、原录音和派生家族，只有分组互斥时才打开 train/dev 的案例清单和媒体。公开分组应使用可信跨划分登记中的高熵不透明身份；格式和哈希无法证明声明真实，也不能在不读取保留媒体时比较其 PCM。

每个分组清单包含 `version: 3`、`partition: "train"` 或 `"dev"`，以及按案例顺序排列的 `records`。每条仅有 `id` 和 `source`；后者有 `speaker_sha256`、`session_sha256`、`recording_sha256`、`family_sha256` 四个小写 64 位十六进制字段。案例清单包含同样的 `version`、`partition`、`rule: "number-plus-one-v1"` 和 `records`；每条必须有与分组清单相同的 `id`、`source`，以及 `audio_file`、`audio_sha256`、`spoken_number`、`answer` 和 `declaration`。媒体文件名只能是对应划分目录内的简单 WAV 文件名，输入数字为 0–99，答案必须加一。声明需有非空 `source`；`kind` 为 `licensed-synthetic` 或 `licensed-human` 时另需非空 `license`，`user-provided` 时需 `consent` 与匹配数字的 `human_reference`（`method: "human-listening"`、`labeler`、`spoken_number`）。这些都是**未核实声明**。

`licensed-human` 可记录非空 `published_prompt` 等来源参考，不需要虚构用户同意或人工听审；可选 `reference_status` 只能为 `"unverified"`。记录接口及冻结策略始终将参考标记为未验证：出版文本、数字提取、ASR 输出或授权声明本身都不构成独立人工听审，也不能授予门控通过。实际来源、许可和身份仍须在后续真实证据环节核实；归档或再分发时应遵循相应署名和许可要求。

入口校验实际 WAV 字节哈希并解码成 16 kHz PCM；train/dev 不得出现相同 PCM，同一划分中相同 PCM 若数字或答案矛盾则拒绝，同答案重复可加载，但不因此成为独立案例。每条最多 20 秒，train 与 dev 共用 1,200 秒的解码样本上限；每条解码后、保存前递增检查，上限触发后不再打开后续媒体。Unix 下固定文件通过目录描述符及禁止跟随末级符号链接读取，非 Unix 当前失败关闭。

案例可选 `speech_target` 对象；缺失或 `null` 明确表示纯文本目标。对象包含 `frames`、`target_codec_sha256`、`target_audio_sha256`、`codec_fingerprint_sha256`、`model_config_sha256`、`sample_rate`、`encoding`、`quality_evidence`、`runtime_backend` 和 `preparation_sha256`，与本地 `speech_audio_target::EncodedTarget` 的序列化字段一致；另需提供目标 WAV 的简单文件名 `audio_file` 和 `source` 声明。目标文件只从本划分的 `train-target-media` 或 `dev-target-media` 目录读取；其实际字节 SHA-256 必须匹配 `target_audio_sha256`，WAV 须为最长 20 秒的单声道 24 kHz PCM16 或 float32。`source` 必须有 `kind`、`source`，授权的合成或真人语音还需 `license`，用户提供语音还需 `consent`；`licensed-human` 的参考同样只能标记为未验证。这些字段与编码方式、运行后端同样是**未核实声明**，当前只接受 `quality_evidence: "not_assessed"`。

含语音目标的数据必须使用 `DevelopmentData::load_with_model` 并指定实际本地模型目录；无模型的 `load` 会拒绝含目标的数据，而不是丢弃它。入口保留所有帧和目标字段，按实际模型 `num_codebooks`、`codebook_size`、`audio_eos_code` 及 codec `n_q`、`fsq_levels`、采样率检查：总帧数 5–200，每帧宽度匹配码本，内容码在 codec 和模型范围内，只有最后一帧是全码本 EOS。目标模型配置哈希、codec 配置和权重的完整指纹，以及 #82 使用的目标 codec 身份必须与实际文件一致；单文件和索引分片各按原有算法处理。

冻结的每条目标身份是整个目标对象的规范 JSON 哈希，因而绑定实际帧值、目标 WAV 字节身份、文件名和完整来源声明；加载后的记录仍可取回全部 `frames`，而不是只保留哈希。哈希不能证明 frames 确由该 WAV 和 codec 生成，也不能证实来源、许可或真人标注。显式 v3 训练入口将经模型和 codec 校验的目标 frames、EOS 连同规则文本和实际输入 PCM 交给同一个训练步骤；文本目标缺少语音码时只计算文本损失。来源核查和正式验收仍须另行接线。

## 显式 v3 训练

使用 `omni-rsi-audio --number-v3` 才会选择 v3 入口，省略此参数保留 v1/v2 数据加载路径。`--model`、`--data` 和 `--state` 均须显式指定；例如在已准备好本地模型和隔离数据后：

```sh
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 UV_CACHE_DIR="$PWD/artifacts/uv-cache" \
  cargo run --offline --locked --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu \
  --bin omni-rsi-audio -- --number-v3 --model /path/to/local/model --data /path/to/v3-data \
  --state /path/to/state --epochs 1 --max-updates 1
```

入口只加载 train/dev 和公开保留元数据，构造与旧路径相同的 `AudioAnswer`，调用同一个 `train_step`；它不打开保留答案、盐或媒体，不自动发布候选。检查点位于 `state/number-audio-v3/audio-conditioned-v1/`，不复用旧路径的状态；两条路径共用父目录的训练租约。恢复必须显式加 `--resume`，并重新匹配模型、codec、数据和实现身份。`dataset_hash` 绑定实际 train/dev 清单、媒体、规则答案、目标帧和来源、公开保留承诺，以及 v3 加载和训练数据接线实现；完整 codec 与模型配置还由目标身份及底座身份绑定。开发集只参与身份和隔离检查，不用于训练更新。

此次修改了共享的音频训练入口、检查点和示例构造源码，因此 `audio_implementation_hash` 会变化。旧 v1/v2 清单仍可使用，但旧实现生成的检查点不会自动迁移或恢复；显式恢复时会报告实现摘要已变，保留原检查点和指针，可用原二进制与备份继续。训练完成或本地 mock 测试均不表示质量改善、真实来源已核验或许可已确定。

共享检查点源码也纳入数字开发评分的实现摘要，新增只读接口则改变预注册实现源码摘要。旧评分报告和旧冻结协议仍绑定各自生成时的源码；新版本不会把它们自动升级成新版本证据，旧冻结协议的重验会拒绝当前源码摘要不匹配。保留旧文件及其原二进制，后续按新协议重新冻结并取得所需的真实证据。

可由有本地权重的操作者显式运行 `audio_cli` 测试中的 `v3_real_cli_trains_speech_and_text_then_resumes`：设置 `LFM3_MODEL` 和 `LFM3_NUMBER_V3_DATA`，数据至少两条训练记录，首条带目标帧、第二条为纯文本。测试各执行一次更新与恢复，不发布候选；命令需明确选定本机可用后端并传入 `-- --ignored --exact`。假盲集占位的探索样本只能作接线烟雾，不能作为正式保留或质量证据。

## 冻结和重验

`preregister` 接受数据目录、实际模型目录、候选文件、评估程序、后端名称及可选 R2T2/Breeze 位置。它冻结规则、评分和统计实现身份、解码及统计参数、train/dev 分组与案例清单和逐条媒体/PCM 身份、规则答案、声明哈希、确定性交换映射、公开保留元数据指纹，以及回归计划和输入身份。评分源码与评估器文件哈希只标识文件，不证明它们被执行；后端、来源、许可和真人标注仍是声明。

底座身份直接使用训练的 `omni_lfm3::speech_adapter::source_fingerprint`：索引优先于单文件，绑定索引实际指定的分片，也包含 codec 身份。另行冻结与 `fingerprint::codec_fingerprint` 相同域和内容的 codec 指纹。注意 #82 的旧无分片目标 codec 身份使用单个权重文件哈希，不能把两种算法的值直接等同比较；接线时须按各自协议重新计算并验证。候选与评估程序冻结实际文件 SHA-256；提供 R2T2/Breeze 时使用项目现有身份函数。其未提供或未实际运行时，不能把缺失身份当成回归或语音证据。

`regression-plan.json` 要求 `version: 1`、`criterion: "no_degradation"`，以及 `groups` 数组，其中 `chinese`、`english`、`other_omni`、`reading`、`dialogue` 各出现一次，条目包含 `name` 和 `manifest_file`。每个文件必须是 `regression-inputs` 下的简单 JSON 文件名，当前仅解析为对象并冻结字节哈希；不验证内容是否充分，更不执行回归。

目标注册目录必须尚不存在。实现先读入并验证一次，再以新目录原子占位，重新读取和比较输入，写入并同步 `protocol.json`；重复、并发冲突或占位后的中断均不能静默覆盖。`verify_registration` 重读协议并重新计算输入、文件、参数和承诺；不匹配、损坏或伪造通过状态均失败。占位后异常退出可能留下不完整目录，它不会自动恢复为有效注册；需由操作者处理或选择新位置。冻结不是正式验收，真实运行路径未来还须在执行前后重新验证相同快照、原始成对报告及回归结果。

本地文件边界适用于遵循流程的进程。模型、codec、R2T2 和 Breeze 目录必须由操作者选为可信、稳定且与保留秘密分离的工件树；当前会拒绝树中的符号链接及特殊文件。训练现有 `source_fingerprint` 和两种语音工具身份仍使用路径打开，因此预检与打开之间无法阻止同账号主动替换文件；本地进程控制者也能改写声明或删除账本。这些检查不提供对同账号或保管人的防护。现有身份函数在单次哈希期间的取消可能延迟；完成后仍会检查取消状态。
