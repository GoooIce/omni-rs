# 数字听答：本地规则数据与开发集对照

这是 #82 的实验入口，不启用候选。程序从输入数字计算中文读数和加一答案；R2T2 对输入声音的转写只能作核查，不能决定答案。当前没有可核验的未见真人数字数据、最终保留集报告和既有能力回归，不能据此关闭 #82。现有 AISHELL 普通句子缺少清晰说话人 ID 与数字片段，不能充当这组数据。

## 数据计划

先在一个本地目录放置主动提供或许可允许的 WAV，按说话人、会话、原录音和增强家族预先分配 `train`、`dev`、`test`，再制作增强。每个集合都必须有记录；开发集至少需要两个答案不同的样本才能执行交换对照。小样本只能验证管线，研究提案建议至少 120 条训练、30 条开发和 50 条独立最终保留样本。合成声音不是真人；不能仅凭不同 seed 就把同一声音标为不同说话人。

计划 JSON 的版本是 1，每条记录包含以下字段；示例中的尖括号是必须替换的占位内容，不是现成的许可、哈希或标注：

```json
{
  "version": 1,
  "records": [{
    "id": "train-12", "split": "train", "input_audio": "train-12.wav", "spoken_number": 12,
    "authorization": {"kind": "licensed-synthetic", "source": "<合成来源>", "license": "<适用许可>"},
    "provenance": {
      "speaker_id": "<预先确定的合成声音>", "session_id": "<会话>",
      "recording_id": "<原录音>", "family_id": "<派生家族>",
      "teacher_sha256": "<64位合成模型和程序指纹>",
      "verifier_sha256": "<64位核查模型和程序指纹>",
      "verified_text": "十二", "created_at": "<来源时间>"
    }
  }]
}
```

计划还需添加分组互不泄漏的 `dev` 和 `test` 记录。`input_audio` 是数据目录内的相对路径。输入为 0–99 的整数；构造器生成固定提示、规则名称、规范中文读数、答案和实际 WAV 字节 SHA-256。若记录是增强版本，`provenance.augmentation_of` 应指向同一集合、说话人、会话、录音和家族中的原始记录。可选语音码 `frames` 必须同时绑定 `target_codec_sha256`，与 #81 一致。

`user-provided` 类型必须有非空 `source`、`consent`，并在 `provenance.human_reference` 写入 `{"method":"human-listening","labeler":"<审核人>","spoken_number":12}`。ASR 不能替代人工参考。当前没有独立的公开真人数据授权类型；不要将公开数据伪写为个人同意。说话人、授权、人工标注和模型指纹都是来源声明，哈希与一致性检查不能证明这些声明的真实性，也不能识别未声明的裁剪、变速或同一真人的不同身份。

输入音频需要自行用已许可的本地材料准备。仓库的 Breeze 可按 [模块说明](breeze-tts.md)合成；构造器不会调用 Breeze，也不会自动执行 R2T2。以下命令均从仓库根目录运行，假设模型位于 `models/`、可执行文件位于 `target/release/`；独立工作树或其他机器须按实际位置替换。例子可分别取得本地程序/模型指纹，或显式运行一次 R2T2；后者输出原始转写及音频哈希，供人工核对并填写计划，不产生算术真值。`build` 不会自动消费或验证这条命令的输出，计划里的 `verified_text` 仍是声明。

```sh
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 cargo run --offline --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu --example number_audio -- fingerprint-tts models/Breeze-TTS-2 target/release/breeze-tts
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 cargo run --offline --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu --example number_audio -- verify-input "$PWD/artifacts/number-audio/data/train-12.wav" models/Confucius4-R2T2 target/release/r2t2
```

在备好完整计划和 WAV 后构造清单。输出 `manifest.json` 已存在时拒绝覆盖；整份数据验证通过后才写入。请勿自动录音或下载未经许可的数据。

```sh
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 cargo run --offline --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu --example number_audio -- build "$PWD/artifacts/number-audio/data" "$PWD/artifacts/number-audio/plan.json"
```

第 2 版清单在准入时重新计算答案，校验输入、来源、所有媒体哈希和分组，检查完全相同的解码 PCM 未跨集合出现，再仅把 `train` 记录交给训练。原 #81 第 1 版通用训练清单不要求数字字段，仍按原授权规则读取。其检查点实现指纹包括加载器源文件，因此此改动后，**之前代码生成的音频训练检查点会因实现指纹变化而拒绝恢复**；旧文件不会被覆盖，需保留原版本恢复或在新状态目录重新训练，不自动迁移。

## 真实模型开发集对照

以下训练和评估需要本机模型权重及资源，应由协调方低优先级串行代跑。`state` 必须是新的状态目录，六次更新仅是预算，不是能力通过证明。完整训练前请先检查许可、计划、输入转写和数据隔离。

```sh
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 cargo run --offline --release --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,metal --bin omni-rsi-audio -- --model "$HOME/models/LiquidAI/LFM3-Omni-2B" --data "$PWD/artifacts/number-audio/data" --state "$PWD/artifacts/number-audio/state" --epochs 1 --max-updates 6
CARGO_TARGET_DIR="$PWD/artifacts/private-target" CARGO_BUILD_JOBS=2 cargo run --offline --release --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,metal --example number_audio -- dev "$HOME/models/LiquidAI/LFM3-Omni-2B" "$PWD/artifacts/number-audio/data" "$PWD/artifacts/number-audio/state" "$PWD/artifacts/number-audio/dev-report.json" models/Confucius4-R2T2 target/release/r2t2
```

省略最后两个 R2T2 参数可只运行文本对照，报告的输出语音 CER 会标为 `not_run`。传入时，输出语音使用相同 seed 42、温度 0.4、重复惩罚 1.2、180 slots，由本机 R2T2 转写并以规则答案计算 CER；原始文本、转写、语言、正常结束和有效性逐题记录。静音没有规则答案，不计入 CER。临时输出 WAV 不保留。文本生成对两组统一使用 FP32、贪心、最多 32 token；每次均重新建立模型上下文，输入是完整音频，不推断流式前缀，也不复用不同录音的声学前缀。两个入口通过相同状态目录的协作锁避免与遵守该约定的 RSI 工作并发；不能保证其他应用不占用设备。

报告绑定数据、底座、codec、候选检查点及运行参数；样本数、正确率、交换与静音、输出语音 CER 和正常结束分别列出。当前文本正确率采用移除标点后的规范中文答案精确匹配，阿拉伯数字或附带解释的回答会保守计为错误，不能当作完整的语义评估。固定中文/英文及其他 Omni 回归、开发门控与独立保留集单次账本尚未完成，后两项由 [#90](https://github.com/GoooIce/omni-rs/issues/90) 处理；当前 `candidate_eligible` 始终为 `false`，即使开发集指标看似良好也保留基座。开发集上任何调参都不能使用最终保留集；真人迁移还需有授权的未见说话人、人工听审和独立参考。
