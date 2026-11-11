# 中文封闭文本回归 v1

本切片使用 [公开题目清单](number-chinese-regression-v1.json) 中的 12 道中文文本题，比较同一 v3 底座与已受控的 LoRA 候选。它是公开开发检查，不是盲测；只覆盖这一组文本，不代表英语、其他 Omni、阅读、对话或自然语音的质量。报告始终是 `insufficient_evidence`、`candidate_eligible=false`，不能签发 `VerifiedDevelopment`、访问正式保留集或启用候选，也没有真人质量证据。

清单版本、题 ID、用户 prompt、评分参考、解码、结束与评分规则在注册前确定。评分参考只进入评分器；推理入口只向模型传入各题的用户文本，不传参考或媒体。解析器拒绝未知或重复字段、重复 ID、完全相同的 prompt、越界输入及 prompt 中逐字出现的参考。计划和五组清单各限 256 KiB，总计最多 1.5 MiB；中文清单最多 128 题，prompt 最多 2048 UTF-8 字节，单次输出最多 4096 字节，报告最多 2 MiB。逐字检查不能识别语义重复或泄漏，出题仍需人工检查。

逐题使用相同的贪心解码选项和独立新 cache。原始输出是 tokenizer 解码后的文本（跳过特殊 token），报告保留每题文本、结束原因和实际停止 token；不是原始 token 序列。只有在注册的 stop token 正常结束，且输出去除首尾空白后与参考完全相同时才计为正确；达到长度限制或其他 token 均不算正常结束。返回 `Cancelled` 即使取消标志未置位也会立即中止，保留未完成诊断；完整报告与复算均不接受取消行。报告同时保存两臂的正确数、正常结束数和差值；复算逐题检查原始输出、题目和小计。

## 准备与执行

使用已有的 train/dev v3 数据、公开承诺、实际底座和受控候选状态。数据根目录的 `regression-plan.json` 必须是现有五组格式（`version: 1`、`criterion: "no_degradation"`），其中 `chinese` 的 `manifest_file` 指向数据根目录 `regression-inputs` 中的中文清单。其余四组的清单也必须存在并会被冻结身份，但此切片不会执行或验收它们。把本仓库的公开清单复制到新开发数据目录中，先确认目标不存在；不要修改旧注册使用的输入。注册将冻结原始文件内容，之后即使只改变空白也需重新注册。

以下命令需要预先由操作者设置 `MODEL`、`DATA`、`STATE`、`REG`、`REPORT` 为本机路径，其中 `REG` 和 `REPORT` 都必须不存在，父目录需已存在。状态中须已有匹配候选的 `number-audio-v3/audio-conditioned-v1/checkpoints`。调用者选定的根目录及报告父目录的祖先可使用可信路径别名（例如 macOS 的 `/var`），但路径不能包含 `..`，所选根或报告本身的末级不能是符号链接。若一个所选路径位于另一所选根之下，该根以内的部分仍不能使用链接，即使它被指定为注册目录或报告路径。程序解析并固定这些路径后，根内部的目录和文件仍逐级不跟随链接；计划、五组清单、注册文件、状态中的锁与候选文件、待复算报告必须是单链接普通文件。调用者须信任所选路径的祖先；别名不是隔离或访问授权机制。这些路径应指向可信的本地开发材料，不要指向保留题、盐或保留音频。本命令需要已经缓存的 locked Rust 依赖；不会自动联网安装依赖。私有 Cargo home 可由操作者提供已有可信缓存；不要求设置 `CARGO_HOME`。

```sh
export CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR="$PWD/artifacts/chinese-regression/target"
cargo build --offline --locked --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features native,cpu --example number_audio
"$CARGO_TARGET_DIR/debug/examples/number_audio" preregister-chinese-v3 "$MODEL" "$DATA" "$STATE" "$REG"
"$CARGO_TARGET_DIR/debug/examples/number_audio" dev-chinese-v3 "$MODEL" "$DATA" "$STATE" "$REG" "$REPORT"
"$CARGO_TARGET_DIR/debug/examples/number_audio" recheck-chinese-v3 "$MODEL" "$DATA" "$STATE" "$REG" "$REPORT"
```

注册与后续执行必须使用完全相同的可执行文件。注册和复核会读取并核对现有模型、候选、数据和可执行文件的身份；实际执行还会加载本机权重并进行 Rust/Burn 推理。没有隐式下载、录音、外部 ASR 或云调用。注册目录内的 `protocol.json` 冻结五组计划与清单、数据、底座、候选和 evaluator；独立的 `chinese-regression-policy.json` 还绑定中文清单、评分与评测源码、实际模型/候选/evaluator、解码参数和无媒体 prompt 规则。中文读取器复用语音复核的严格协议哈希镜像和描述符相对读取，再将计划哈希、文件名和清单哈希与已重验协议逐项比较。输入和策略在执行前后重新核对。

通用注册读取器的路径修复改变了 v3 训练指纹，也改变了中文评测实现与可执行文件身份；**旧 v3 候选检查点及旧注册与新身份不兼容**。旧状态不会自动迁移或覆盖；需保留旧状态，在新的隔离状态中按新身份准备候选，再新建评测注册。#79 的身份源和旧音频训练身份源未改；实际后端图及二进制身份仍需在运行环境中核验。失败或中断留下的不完整目录也不会被复用或覆盖。

`recheck-chinese-v3` 不执行模型：它重新核对当前注册身份，然后对报告逐题重算。它也检查报告结构、重复题和小计；注册策略文件采用规范字节比较，拒绝任何改动。报告是本地文件，不是防伪签名或可信执行证明。程序的注入式测试入口只产生 `injected_unverified`；实际命令产生 `local_rust_burn_unverified`，两者都不能获得资格。对原始本地 JSON 的人为篡改不应被视为执行证明。

执行使用现有协作租约，以目录描述符打开锁与候选，执行前后核对目录及锁的身份，并在关键阶段重新解析选定路径，拒绝观察到的别名重指。预先取消、推理错误、输入变化或在发布前观察到取消会失败，不发布完整报告。开始逐题推理后的错误会作为有界的未完成诊断返回；命令行把它写到标准错误，其中保留预算内已观察的原文、结束原因及注册身份，遇到生成返回的取消也记录实际的 `cancelled`。调用方可以自行保存标准错误；诊断没有完成执行后身份复核，不是正式报告，也不能据部分题数计算完整对照差异。达到总输出预算时，诊断明确标注最后输出被省略。最终发布前仍有并发取消与文件系统同步的边界，若目录同步在发布后失败，报告可能已存在，应先人工检查而不是覆盖。描述符相对打开防止受检路径中的符号链接重定向，但根解析不是原子操作，协作锁、路径复核和报告发布不防御不遵守租约、能并发替换并恢复路径或可执行文件的同账号对抗进程。实际运行前应确保其他进程不会改动输入。真实权重的自动测试标为 `ignored`，必须显式提供路径并选择运行。
