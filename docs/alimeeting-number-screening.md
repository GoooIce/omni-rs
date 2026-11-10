# AliMeeting 训练转写：探索性数字候选筛查

`alimeeting-screen` 只读取调用者明确指定、已确认许可的训练 TextGrid 目录，写出一个新的本地 JSON 报告。它不打开音频、不训练、不裁剪、不划分数据集，也不访问保留集。报告是**转写文本候选**，不是对实际听到数字的人工参考、说话人身份独立性、音频完整性、模型改进或正式验收的证明。调用者必须确认目录属于允许的训练来源；程序不能单凭目录名或哈希证明许可或数据分区。

例如，从仓库根目录运行以下命令，先自行准备一个已取得许可且仅包含训练 TextGrid 的目录，以及不存在的输出文件名；示例路径只是占位，不表示已解包或验证真实 AliMeeting 语料：

```sh
mkdir -p artifacts/alimeeting-screen
CARGO_TARGET_DIR="$PWD/artifacts/alimeeting-target" CARGO_BUILD_JOBS=1 cargo run --locked --offline \
  --manifest-path crates/omni-rsi/Cargo.toml --no-default-features --features native,cpu \
  --bin alimeeting-screen -- --textgrid-dir "$PWD/artifacts/ali-train/Train_Ali_near/textgrid_dir" \
  --output "$PWD/artifacts/alimeeting-screen/candidates-001.json"
```

输出父目录必须已存在，且 Unix 权限位不允许组或其他用户写入；输入和输出路径不允许符号链接、`..` 路径分量，输出必须位于输入树之外。程序递归遍历输入目录，拒绝符号链接、特殊文件、非 TextGrid 普通文件、重复录音名及链接数不是 1 的输入文件；拒绝时不发布报告。输出先写同目录的私有临时文件，再以不覆盖已有目标的方式发布；目标已存在时失败。目录句柄及 `openat`/`O_NOFOLLOW` 用于避免通过目录项跟随链接，目前只支持 macOS 和 Linux。权限位检查不能证明不存在 ACL 或同一用户进程的并发干预；来源哈希记录的是读取到的字节，不是对之后文件状态的证明。

格式依据 [Praat TextGrid 规范](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html)以及 AliMeeting 官方[数据准备脚本](https://raw.githubusercontent.com/yufan-aslp/AliMeeting/main/asr/local/alimeeting_data_prep.sh)和[解析示例](https://raw.githubusercontent.com/yufan-aslp/AliMeeting/main/asr/local/alimeeting_process_textgrid.py)。本实现支持严格的带标签长文本格式、UTF-8（可带 BOM）和带 BOM 的 UTF-16 LE/BE、双引号加倍及多行字符串、区间层和空点层；短格式、中间格式、非空点层、Praat 反斜杠转义及旧版非 Unicode 编码会明确报错，不会静默忽略。要求有限、非负且不超过 24 小时的时间，同一层区间正长、按顺序不重叠，且处于层和文件的时间范围。根据主代理对官方训练文本的结构核验，区间之间允许时间空隙：这比 Praat 手册所述相邻约束更宽，但不代表空隙是静音或已被标注。空区间层可接受。字段或计数缺失、未知尾部、异常编码和不受支持内容都拒绝整次运行。

当前上限：目录深度 16、总条目数 20,000、单文件 8 MiB、总输入 256 MiB、每文件 4,096 个 tier 和 250,000 区间、全局 1,000,000 区间、单字符串 64 KiB、tier 名 256 字节、候选与待审合计 100,000 行、输出 64 MiB；任何上限被触发都返回错误。录音标识取 TextGrid 文件名去扩展名后的 ASCII 字母、数字、`_` 或 `-` 部分，未核验对应音频是否存在；tier 名和区间编号、原文本及时间分别保留。若官方真实文件不满足这些约束，程序会拒绝；合成测试不能证明真实发布版本兼容。

`whole-number-v1` 先按 Unicode NFKC 规范化、只裁去首尾空白和约定标点，再限制为 0–99 的整段数词形式，调用现有答案数字解析器转换；`二零`、`一二`、日期、含上下文的 `一个`、前导零和 100 以上不进入候选。按同一层中的记录顺序，前一个或后一个区间有非空文本时，匹配项进入 `deferred`；即使区间之间有时间空隙也如此，以便审查类似被切成“一”与“个”的上下文。这项保守规则不证明其他间隔没有上下文。作为额外的保守约束，NFKC 把圈号、上标等非常规形式转换成数字，或裁去开头句点可能将 `.1` 误作整数时，也将匹配项放入 `deferred`；这些待审项不赋予 `candidate_number`。即使是独立的阿拉伯数字文本，也不能仅凭文本断定录音实际是整数读法而非逐位读数；匹配及待审都需要独立听审。

JSON 的 `source_files` 保存相对文件名、原始字节数与逐文件 SHA-256；`source_sha256` 对按相对路径排序的文件列表计算摘要：域分隔符 `omni-rsi/alimeeting-textgrid-source/v1\0` 后，逐项依次加入路径 UTF-8 字节长度（u64 小端）、路径字节、文件长度（u64 小端）及逐文件 SHA-256 的小写 ASCII 十六进制。`tool_sha256` 覆盖筛查、解析、安全文件访问、复用数字解析、CLI 源码及 Cargo 清单和锁文件；它只标识编译时嵌入的源码快照，不证明进程实际执行或输入来源。报告排序和内容不依赖输入根路径及目录枚举顺序。

`candidates` 是逐条可复核记录，包含相对来源、录音名、tier 标签、区间索引、起止时间、原文本及候选数字；`deferred` 保存相邻非空区间或数字记法引起的歧义，并按原因分别计数。`distinct_tier_ids` 和 `tier_id_counts` 按原始 tier 标签去重；`distinct_recording_tier_ids` 和 `recording_tier_counts` 则按录音名与 tier 标签的组合去重，以区分不同录音中的同名标签，两组统计都只计 `candidates`，待审项另计。这些来源标签并未核验是否匿名、跨会话是否稳定或是否对应不同真人，组合数不是说话人人数；也未检查跨 tier 音频重叠。报告会原样包含转写和标签，可能是受限数据，应只留在本地忽略目录，不要提交 Git 或把它当作数据许可证明。

来源文件、候选和待审记录各自带 `spk_suffix`：文件名去扩展名后若以 `_SPK` 加至少一个 ASCII 数字结尾，记录该 `SPK数字` 原文，否则标记 `unknown`；不会把不同前导零形式合并。来源级的 `distinct_source_spk_suffixes`、`source_spk_file_counts` 和 `source_unknown_spk_files` 分别统计已识别的不同后缀、每个后缀的文件数和未知命名文件数；候选级的 `distinct_candidate_spk_suffixes`、`candidate_spk_counts` 和 `candidate_unknown_spk_count` 只统计候选；另有独立的 `distinct_deferred_spk_suffixes`、`deferred_spk_counts` 和 `deferred_unknown_spk_count` 统计待审项。这些是由文件名提取的发布方来源标签，尚未证明跨会话对应同一人、不同标签对应不同人或数字实际读法正确；不应称为独立真人数。
