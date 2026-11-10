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

公开保留元数据必须是盲格式 v2 的 `PublicMetadata`，只包含承诺和不透明分组；读取方在解析时必须读入公开 JSON，故保管人须确保该文件本身没有秘密。入口先检查 train/dev 的输入及目标语音预读分组与公开保留分组，比较说话人、会话、原录音和派生家族，只有分组互斥时才打开 train/dev 的案例清单和媒体。公开分组应使用可信跨划分登记中的高熵不透明身份；格式和哈希无法证明声明真实，也不能在不读取保留媒体时比较其 PCM。预读清单依赖保管人如实登记和隔离文件：若实际案例隐瞒或伪造来源，后续一致性检查不能撤销已发生的案例读取。

每个分组清单包含 `version: 3`、`partition: "train"` 或 `"dev"`，以及按案例顺序排列的 `records`。每条有 `id`、输入的 `source`，有语音目标时还必须有目标的 `target_source`；每个来源组均有 `speaker_sha256`、`session_sha256`、`recording_sha256`、`family_sha256` 四个小写 64 位十六进制字段。无目标时 `target_source` 可缺省或为 `null`。案例清单包含同样的 `version`、`partition`、`rule: "number-plus-one-v1"` 和 `records`；每条必须有与分组清单相同的 `id`、`source`，以及 `audio_file`、`audio_sha256`、`spoken_number`、`answer` 和 `declaration`。有目标时，案例中 `speech_target.source_group` 必须与预读的 `target_source` 完全一致；不一致或缺失时在打开任何媒体前拒绝。媒体文件名只能是对应划分目录内的简单 WAV 文件名，输入数字为 0–99，答案必须加一。声明需有非空 `source`；`kind` 为 `licensed-synthetic` 或 `licensed-human` 时另需非空 `license`，`user-provided` 时需 `consent` 与匹配数字的 `human_reference`（`method: "human-listening"`、`labeler`、`spoken_number`）。这些都是**未核实声明**。

`licensed-human` 可记录非空 `published_prompt` 等来源参考，不需要虚构用户同意或人工听审；可选 `reference_status` 只能为 `"unverified"`。记录接口及冻结策略始终将参考标记为未验证：出版文本、数字提取、ASR 输出或授权声明本身都不构成独立人工听审，也不能授予门控通过。实际来源、许可和身份仍须在后续真实证据环节核实；归档或再分发时应遵循相应署名和许可要求。

入口校验实际 WAV 字节哈希并解码成 16 kHz PCM；train/dev 不得出现相同输入 PCM，同一划分中相同 PCM 若数字或答案矛盾则拒绝，同答案重复可加载，但不因此成为独立案例。输入和目标每个 WAV 均最多 32 MiB、20 秒。一次 v3 训练调用的初始加载、目标预检和重编码共用 128 MiB 实际媒体读取字节及相当于 1,200 秒 16 kHz 样本的预算；每次重读及每次重编码解码都另外计数。仅加载或预注册也是以各自一次调用为单位计数。重复引用即使复用已缓存的目标 PCM 身份，初始加载仍按每次引用保守计费。加载时目标 WAV 为比较身份会分别扫描 24 kHz PCM 和解码 16 kHz PCM，这两步共按一次目标时长计费；由于额外扫描的时长不超过已计费的初始目标时长，整个训练调用的实际解码及目标扫描合计最多相当于 2,400 秒音频时长。读取文件前检查元数据和剩余字节，读取本身不超过剩余额度；解码前按 WAV 头检查时长并预留样本预算，目标 24 kHz 时长按 16 kHz 等价样本向上取整。Unix 下固定文件通过目录描述符及禁止跟随末级符号链接读取，非 Unix 当前失败关闭。

案例可选 `speech_target` 对象；缺失或 `null` 明确表示纯文本目标。对象包含 `frames`、`target_codec_sha256`、`target_audio_sha256`、`codec_fingerprint_sha256`、`model_config_sha256`、`sample_rate`、`encoding`、`quality_evidence`、`runtime_backend` 和 `preparation_sha256`，与本地 `speech_audio_target::EncodedTarget` 的序列化字段一致；另需提供目标 WAV 的简单文件名 `audio_file`、`source` 声明和上述 `source_group`。目标文件只从本划分的 `train-target-media` 或 `dev-target-media` 目录读取；其实际字节 SHA-256 必须匹配 `target_audio_sha256`，WAV 须为最长 20 秒的单声道 24 kHz PCM16 或 float32。train/dev 目标 WAV 的实际字节和规范 24 kHz PCM 均不得重复；规范 PCM 将 PCM16 按 32768 归一化，并把正负零视为相同。目标还会通过与输入完全相同的 `decode_wav` 路径转为 16 kHz PCM，与另一划分的输入双向比较；跨角色的 WAV 字节相同或规范 16 kHz PCM 完全相同均被拒绝。同一目标字节可复用已验证的 PCM 摘要，但每次引用仍计入总字节和时长预算。这些精确比较不能证明变速、裁剪或其他变化后的素材来源独立。`source` 必须有 `kind`、`source`，授权的合成或真人语音还需 `license`，用户提供语音还需 `consent`；`licensed-human` 的参考同样只能标记为未验证。这些字段与编码方式、运行后端同样是**未核实声明**，当前只接受 `quality_evidence: "not_assessed"`。

含语音目标的数据必须使用 `DevelopmentData::load_with_model` 并指定实际本地模型目录；无模型的 `load` 会拒绝含目标的数据，而不是丢弃它。入口保留所有帧和目标字段，按实际模型 `num_codebooks`、`codebook_size`、`audio_eos_code` 及 codec `n_q`、`fsq_levels`、采样率检查：总帧数 5–200，码本数最多 128，每帧宽度匹配码本，内容码在 codec 和模型范围内，只有最后一帧是全码本 EOS。目标制备、数据加载和训练重编码共用对配置字节的纯格式校验，但各阶段仍分别取得和验证自己的文件或 codec 快照；加载器保留目录描述符及 nofollow 边界，训练解码器仍使用重编码捕获的同一份 codec。目标模型配置哈希、codec 配置和权重的完整指纹，以及 #82 使用的目标 codec 身份必须与实际文件一致；单文件和索引分片各按原有算法处理。

冻结的每条目标身份是整个目标对象的规范 JSON 哈希，因而绑定实际帧值、目标 WAV 字节身份、文件名和完整来源声明；加载后的记录仍可取回全部 `frames`，而不是只保留哈希。哈希本身不能证明 frames 确由该 WAV 和 codec 生成，也不能证实来源、许可或真人标注。显式 v3 训练会在新建检查点或恢复更新之前，对 train/dev 的每条语音目标重新读取并校验 WAV 字节，从同一份有界字节快照解码，以实际 codec 的 FP32 编码器逐帧、逐码本复算并核对 EOS；`caller_supplied_unverified` 不可用于该训练入口。模型的 codec 解码器使用与复算相同的配置及权重快照。文本目标缺少语音码时只计算文本损失；来源核查和正式验收仍须另行接线。

## 显式 v3 训练

使用 `omni-rsi-audio --number-v3` 才会选择 v3 入口，省略此参数保留 v1/v2 数据加载路径。`--model`、`--data` 和 `--state` 均须显式指定；例如在已准备好本地模型和隔离数据后：

```sh
uv run --offline --no-project python scripts/share-target.py
CARGO_TARGET_DIR="$PWD/target/minicpm-v-cpu" CARGO_BUILD_JOBS=2 \
  cargo run --offline --locked --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu \
  --bin omni-rsi-audio -- --number-v3 --model /path/to/local/model --data /path/to/v3-data \
  --state /path/to/state --epochs 1 --max-updates 1
```

入口只加载 train/dev 和公开保留元数据，构造与旧路径相同的 `AudioAnswer`，调用同一个 `train_step`；它不打开保留答案、盐或媒体，不自动发布候选。检查点位于 `state/number-audio-v3/audio-conditioned-v1/`，不复用旧路径的状态；两条路径既共用父目录租约，也按实际检查点根目录取得租约，避免嵌套 state 别名竞争。恢复必须显式加 `--resume`，并重新匹配模型、codec、数据和实现身份，重新执行全部目标重编码。`dataset_hash` 绑定实际 train/dev 清单、媒体、规则答案、目标帧和来源、公开保留承诺，以及 v3 加载和训练数据接线实现；完整 codec 与模型配置还由目标身份及底座身份绑定。开发集只参与身份、隔离与重编码检查，不用于训练更新。

目标的 `runtime_backend` 是被数据身份绑定的历史声明，不要求与本次训练后端相同，也不作为制备过程的证明；现场使用训练后端的 FP32 codec 重编码，编码帧必须完全一致，不使用数值容差。每个目标在构造编码器前先检查 4–199 个内容帧及不超过 16,777,216 个 FP32 STFT 特征，WAV 继续受上文的单文件和累计预算限制。重编码器每次处理一个目标；取消在文件读取及目标之间检查，单次解码、编码器构造和编码期间不保证立即中止。模型主体仍按路径加载，前后哈希与树预检适用于受控、稳定的本地工件，不防同账号并发替换或 ABA。重编码只验证当前帧与这份 WAV 和 codec 的一致性，不证明历史生成过程、授权、语音内容是否读对答案或真人参考质量。

旧 v1/v2 训练入口保留原有行为，包括语音目标；它没有目标 WAV 现场重编码，只能用于探索，不能据此声称满足 v3 的语音目标核验。旧清单若没有 `codec_fingerprint_sha256` 扩展仍可使用；该字段若存在必须是合法哈希且对应语音帧，显式 `null` 或无效值会被拒绝。需要清理无意义的旧 `null` 时，应另存清单副本；改变数据不会迁移旧状态。预注册目前可以记录任意候选文件的哈希，但始终标记资格不足，并不验证其训练来源或授予晋级许可；正式开发验收及保留许可仍未接通，legacy 检查点或探索报告不能作为正式通过证据。

音频 legacy 检查点使用协议版本 1 和共享音频实现摘要，v3 使用协议版本 2 和额外绑定 v3 验证、加载源码的摘要。仅修改 v3 专用源码不会改变 legacy 摘要；修改真正共用的训练、精度配置、权重加载或张量实现则会改变两条路径的摘要。共享源码文件按整个文件计入，修改其中任何部分都会保守地改变两者摘要。独立自朗读训练的检查点实现协议不因本次修改而改变。此次摘要协议和共享数值实现覆盖范围发生变化，因此此前实现（包括 `4735758`）生成的音频 v1/v2 和 v3 检查点不会自动迁移或恢复。显式恢复时会拒绝不匹配，保留原检查点和指针；可用原二进制及备份继续旧实验。训练完成或本地测试均不表示质量改善、真实来源已核验或许可已确定。

音频训练摘要还绑定编译时仓库支持的本地 Cargo 路径补丁及其本地依赖：按相对路径排序并计入源码、构建脚本、配置及原生文件的名称和内容，文件增加或删除也会改变摘要；`.git`、顶层 `target` 和 `artifacts` 构建输出不计入，源码树中的符号链接和特殊文件会被拒绝。CPU 路径在已核对的清单和锁文件不变时只计入相关 CPU 补丁及其本地依赖；清单或锁文件改变时保守计入所有补丁，未知的新补丁也计入。Metal 路径计入全部补丁及本地 kernel。构建脚本只用 Rust，不启动嵌套 Cargo、Git 或 Python，也不会强加离线或锁文件选项。已有配置文件被绑定；可见的 Cargo `paths` 和 `[patch]` 配置覆盖会失败关闭，支持的仓库清单以单行本地路径补丁为准，不支持的语法明确报错。

此约定适用于由仓库清单及已检测配置构建的受控本机环境，覆盖仓库入口使用的 ndarray CPU 和 Metal CubeCL；下游另行注入的自定义后端或本地 registry 源码不在保证范围内，不能仅凭此摘要将其当作受绑定训练证据。配置查找使用构建进程的 `PWD` 和清单目录；应从仓库根目录或清单目录构建，并保证 `PWD` 与实际工作目录一致。Cargo 的命令行 `--config` 可改变依赖但不会完整传递给 build script；不要用它覆盖补丁、路径或依赖来源，本机制不能证明这类覆盖后的实际依赖。新增外部 Cargo 配置后应重新构建并核对，不能依赖旧缓存自动发现；用户目录和文件系统根目录不会作为目录监视。编译器、系统库、驱动、硬件，以及同账号并发替换或伪造工件仍不在此摘要的认证范围内。该摘要变更使旧音频状态失败关闭，不会自动迁移；#79 独立自朗读检查点和已有二进制保持原协议。

目标制备的 `preparation_sha256` 也因加入共享格式校验和数值精度源码而改变；它仍是来源声明，旧目标不会因该字段与当前源码不同而被自动改写或视作执行证明。旧目标要进入 v3 训练仍须通过当前配置、WAV、codec 和逐帧重编码检查。单文件及分片 codec 身份算法不变；涉及新训练实现或数据摘要的旧状态和冻结记录需保留原件，不自动迁移。

共享检查点源码也纳入数字开发评分的实现摘要，新增只读接口则改变预注册实现源码摘要。旧评分报告和旧冻结协议仍绑定各自生成时的源码；新版本不会把它们自动升级成新版本证据，旧冻结协议的重验会拒绝当前源码摘要不匹配。保留旧文件及其原二进制，后续按新协议重新冻结并取得所需的真实证据。

可由有本地权重的操作者显式运行 `audio_cli` 测试中的 `v3_real_cli_trains_speech_and_text_then_resumes`：设置 `LFM3_MODEL` 和 `LFM3_NUMBER_V3_DATA`，数据至少两条训练记录，首条带目标帧、第二条为纯文本。测试各执行一次更新与恢复，不发布候选；命令需明确选定本机可用后端并传入 `-- --ignored --exact`。假盲集占位的探索样本只能作接线烟雾，不能作为正式保留或质量证据。

当前盲格式 v2 不支持秘密目标音频：保留案例只有输入音频，公开来源组也只承诺现有输入来源。这里不会读取、比较或声称隔离任何秘密目标音频；若未来要把语音目标用于保留集，必须先设计新的盲封存格式和独立来源、媒体核验流程，正式目标语音验收在此之前仍不可用。

## 冻结和重验

`preregister` 接受数据目录、实际模型目录、候选文件、评估程序、后端名称及可选 R2T2/Breeze 位置。它冻结规则、评分和统计实现身份、解码及统计参数、train/dev 分组与案例清单和逐条媒体/PCM 身份、规则答案、声明哈希、确定性交换映射、公开保留元数据指纹，以及回归计划和输入身份。评分源码与评估器文件哈希只标识文件，不证明它们被执行；后端、来源、许可和真人标注仍是声明。

交换音频必须是答案不同的一对一映射：每条开发案例恰好使用一条不同答案的开发音频，每个音频恰好使用一次，但不要求双向配对；若某答案超过案例总数的一半，或开发集有重复规范 PCM，则拒绝冻结。协议记录不同交换 PCM 的数量，逐条交换通过 `audio_case_id` 关联到 `dev_cases` 的 PCM 摘要和来源组；这些身份仍须独立核实，不能作为通过证据。

成功 `preregister` 或 `verify_registration` 后，`RegisteredDevelopment::swaps()` 返回只读视图；`entries()` 的 `case_id()` 和 `audio_case_id()` 指向同一注册对象的开发案例，视图的 `protocol_sha256()` 是绑定该映射的协议哈希。真实评估必须消费此冻结映射，不能重新挑选交换音频，并应在执行前后重验注册。该视图不提供保留集许可，也不证明评估已执行。开发比较及未完成的证据要求见 [开发证据文档](number-development-evidence.md)。

每条输入及可选目标分别记录 `source_provenance` / `target_provenance`：`created_at`、`teacher_sha256` 和 `verifier_sha256` 缺失或格式无效时为 `null`，并在 `missing` 中列出，状态为 `incomplete`。时间要求带 `T` 和时区的时间戳，教师及验证器身份要求小写 SHA-256；三项齐全时状态仅为 `declared_unverified`，不表示来源、时间或工具已核实。可选工具未提供时对应的冻结身份仍为 `null`，不得用占位值伪造完整性或许可。

底座身份直接使用训练的 `omni_lfm3::speech_adapter::source_fingerprint`：索引优先于单文件，绑定索引实际指定的分片，也包含 codec 身份。另行冻结与 `fingerprint::codec_fingerprint` 相同域和内容的 codec 指纹。注意 #82 的旧无分片目标 codec 身份使用单个权重文件哈希，不能把两种算法的值直接等同比较；接线时须按各自协议重新计算并验证。候选与评估程序冻结实际文件 SHA-256；提供 R2T2/Breeze 时使用项目现有身份函数。其未提供或未实际运行时，不能把缺失身份当成回归或语音证据。

`regression-plan.json` 要求 `version: 1`、`criterion: "no_degradation"`，以及 `groups` 数组，其中 `chinese`、`english`、`other_omni`、`reading`、`dialogue` 各出现一次，条目包含 `name` 和 `manifest_file`。每个文件必须是 `regression-inputs` 下的简单 JSON 文件名，当前仅解析为对象并冻结字节哈希；不验证内容是否充分，更不执行回归。

目标注册目录必须尚不存在。实现先读入并验证一次，再以新目录原子占位，重新读取和比较输入，写入并同步 `protocol.json`；重复、并发冲突或占位后的中断均不能静默覆盖。`verify_registration` 重读协议并重新计算输入、文件、参数和承诺；不匹配、损坏或伪造通过状态均失败。占位后异常退出可能留下不完整目录，它不会自动恢复为有效注册；需由操作者处理或选择新位置。冻结不是正式验收，真实运行路径未来还须在执行前后重新验证相同快照、原始成对报告及回归结果。

这次补充分组、来源和交换规则会改变注册实现与快照身份。旧的纯文本 v3 数据可继续读取；旧语音目标缺少预读目标组时会被拒绝，须由可信来源登记补充真实身份后重新准备。旧注册不会自动迁移或通过重验；请保留原状态并选择新的空目录重新冻结，不要覆盖既有协议。

本地文件边界适用于遵循流程的进程。模型、codec、R2T2 和 Breeze 目录必须由操作者选为可信、稳定且与保留秘密分离的工件树；当前会拒绝树中的符号链接及特殊文件。训练现有 `source_fingerprint` 和两种语音工具身份仍使用路径打开，因此预检与打开之间无法阻止同账号主动替换文件；本地进程控制者也能改写声明或删除账本。这些检查不提供对同账号或保管人的防护。现有身份函数在单次哈希期间的取消可能延迟；完成后仍会检查取消状态。
