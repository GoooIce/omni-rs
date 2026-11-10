# 数字听答开发集：隔离入口和预注册基础

`number_development` 提供独立的 train/dev 读取及协议冻结基础。它尚未接入实际训练或评估，不能构造 `VerifiedDevelopment`、打开保留集或启用候选；返回状态固定为 `insufficient_evidence`。以下文件格式和测试只约束输入与快照，不证明运行、数据真实性、授权或模型质量。旧版 v2 单清单会暴露 test 内容，只能用于已暴露的实验，不能用作盲验收输入。

## v3 数据边界

调用 `DevelopmentData::load` 时，数据根目录布局如下；`layout.v3` 的内容必须严格为 `omni-rsi-number-train-dev-v3\n`。保留集的秘密清单、盐和媒体由保管人另行隔离，不放在这些文件中。

```text
layout.v3
holdout-public.json
train.groups.v3.json     dev.groups.v3.json
train.v3.json            dev.v3.json
train-media/*.wav        dev-media/*.wav
regression-plan.json     regression-inputs/*.json
```

公开保留元数据必须是盲格式 v2 的 `PublicMetadata`，只包含承诺和不透明分组；读取方在解析时必须读入公开 JSON，故保管人须确保该文件本身没有秘密。入口先检查 train/dev 的纯分组清单与公开保留分组，比较说话人、会话、原录音和派生家族，只有分组互斥时才打开 train/dev 的案例清单和媒体。公开分组应使用可信跨划分登记中的高熵不透明身份；格式和哈希无法证明声明真实，也不能在不读取保留媒体时比较其 PCM。

每个分组清单包含 `version: 3`、`partition: "train"` 或 `"dev"`，以及按案例顺序排列的 `records`。每条仅有 `id` 和 `source`；后者有 `speaker_sha256`、`session_sha256`、`recording_sha256`、`family_sha256` 四个小写 64 位十六进制字段。案例清单包含同样的 `version`、`partition`、`rule: "number-plus-one-v1"` 和 `records`；每条必须有与分组清单相同的 `id`、`source`，以及 `audio_file`、`audio_sha256`、`spoken_number`、`answer` 和 `declaration`。媒体文件名只能是对应划分目录内的简单 WAV 文件名，输入数字为 0–99，答案必须加一。声明需有 `source`，并根据 `kind` 为 `licensed-synthetic` 或 `user-provided` 分别提供 `license`，或 `consent` 与匹配数字的 `human_reference`（`method: "human-listening"`、`labeler`、`spoken_number`）。这些都是**未核实声明**。

入口校验实际 WAV 字节哈希并解码成 16 kHz PCM；train/dev 不得出现相同 PCM，同一划分中相同 PCM 若数字或答案矛盾则拒绝，同答案重复可加载，但不因此成为独立案例。每条最多 20 秒，train 与 dev 共用 1,200 秒的解码样本上限；每条解码后、保存前递增检查，上限触发后不再打开后续媒体。Unix 下固定文件通过目录描述符及禁止跟随末级符号链接读取，非 Unix 当前失败关闭。

当前 v3 案例**没有目标语音码字段**，解析器拒绝未声明字段，也没有联合训练入口。后续接线必须显式保留和验证 #82 的目标 frames、EOS 及其 codec 指纹，不能静默丢弃语音目标；目前不得把 v3 表述为已完成联合训练或正式通过。

## 冻结和重验

`preregister` 接受数据目录、实际模型目录、候选文件、评估程序、后端名称及可选 R2T2/Breeze 位置。它冻结规则、评分和统计实现身份、解码及统计参数、train/dev 分组与案例清单和逐条媒体/PCM 身份、规则答案、声明哈希、确定性交换映射、公开保留元数据指纹，以及回归计划和输入身份。评分源码与评估器文件哈希只标识文件，不证明它们被执行；后端、来源、许可和真人标注仍是声明。

底座身份直接使用训练的 `omni_lfm3::speech_adapter::source_fingerprint`：索引优先于单文件，绑定索引实际指定的分片，也包含 codec 身份。另行冻结与 `fingerprint::codec_fingerprint` 相同域和内容的 codec 指纹。注意 #82 的旧无分片目标 codec 身份使用单个权重文件哈希，不能把两种算法的值直接等同比较；接线时须按各自协议重新计算并验证。候选与评估程序冻结实际文件 SHA-256；提供 R2T2/Breeze 时使用项目现有身份函数。其未提供或未实际运行时，不能把缺失身份当成回归或语音证据。

`regression-plan.json` 要求 `version: 1`、`criterion: "no_degradation"`，以及 `groups` 数组，其中 `chinese`、`english`、`other_omni`、`reading`、`dialogue` 各出现一次，条目包含 `name` 和 `manifest_file`。每个文件必须是 `regression-inputs` 下的简单 JSON 文件名，当前仅解析为对象并冻结字节哈希；不验证内容是否充分，更不执行回归。

目标注册目录必须尚不存在。实现先读入并验证一次，再以新目录原子占位，重新读取和比较输入，写入并同步 `protocol.json`；重复、并发冲突或占位后的中断均不能静默覆盖。`verify_registration` 重读协议并重新计算输入、文件、参数和承诺；不匹配、损坏或伪造通过状态均失败。占位后异常退出可能留下不完整目录，它不会自动恢复为有效注册；需由操作者处理或选择新位置。冻结不是正式验收，真实运行路径未来还须在执行前后重新验证相同快照、原始成对报告及回归结果。

本地文件边界适用于遵循流程的进程。模型、codec、R2T2 和 Breeze 目录必须由操作者选为可信、稳定且与保留秘密分离的工件树；当前会拒绝树中的符号链接及特殊文件。训练现有 `source_fingerprint` 和两种语音工具身份仍使用路径打开，因此预检与打开之间无法阻止同账号主动替换文件；本地进程控制者也能改写声明或删除账本。这些检查不提供对同账号或保管人的防护。现有身份函数在单次哈希期间的取消可能延迟；完成后仍会检查取消状态。
