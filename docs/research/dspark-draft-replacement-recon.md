# DSpark prose draft 换代可行性侦察（#51）

2026-10-23。问题来自 #41：prose τ≈2.0 是 6 层 markov draft（389M，5 层，target 层
[3,11,19,27,35] 条件化）的质量上限，工程线上限 ~33-36 tok/s；prose 下一个量级必须
τ≥3。本页回答「有没有可用的更强 draft」，不实现接入。判据（#51 brief）：任一候选
离线 τ̂ ≥ 3（prose 样本）→ 立项；全部 < 3 → #41 收伞。

## 结论

**没有候选达标，#41 按判据收伞。** 逐条：

1. **官方更大 draft 变体：不存在。** OpenBMB 为 35B-A3B 只发布了这一份
   DSpark 草稿（389M、5 层、block 15），「不单独提供服务」，无更大 rank/更深变体。
2. **同词表小型文本 LM：不存在。** 35B 词表 248144 是 4.7 新 tokenizer。
   MiniCPM4/4.1 全系 73448、MiniCPM5-1B 130560——都接不上，换词表=训练新
   embed/lm_head，出界。
3. **MiniCPM-V-4.7-1B（唯一 token 兼容候选，本机已有）：τ̂ 不达标。**
   token-id 与 35B 完全兼容（base 词表 248044 条逐条一致；1B 的 50 个 added
   tokens 与 35B 前 50 个逐 id 一致，35B 多出的 50 个是 `<unit>`/`<|audio|>` 类
   特殊符，1B 文本生成永不触达）。但 teacher-forced 逐位命中率 p=0.45-0.59
   （prose），τ̂(block 15) 最高 **2.46 < 3**；code 上 τ̂(15)=5.81 **低于现役
   草稿的在线 7.69**。
4. **训练路线（EAGLE3/DSpark 更大头）：出界**（本仓无训练设施），落为重启条件。

### 测量数据（1B teacher-forced 对 35B int4 贪心 ground truth）

| 条件 | 生成 token | p（逐位 argmax 命中） | top-2 覆盖 | τ̂(3) | τ̂(15) | 现役在线 τ |
| --- | --- | --- | --- | --- | --- | --- |
| prose 短（「用几句话…」，EOS） | 70 | 0.449 | 0.623 | 1.74 | 1.79 | 1.94–2.25 |
| prose 详（「详细说明…」，420 截断） | 420 | 0.592 | 0.697 | 2.09 | 2.46 | 同上 |
| code（回文函数，EOS） | 303 | 0.848 | 0.927 | 3.05 | 5.81 | 7.69 |

τ̂ 为按 dspark 记账（每周期 bonus+接受前缀）在命中向量上的周期仿真，block ∈
{2,3,4,7,8,15,16} 完整曲线在 `artifacts/draft-recon/*-1b-match.json`（本地，
不入库）。方法：`crates/minicpm-v/tests/draft_recon.rs`（`#[ignore]`，
`ground_truth` 跑 35B 贪心存 ids，`one_b_match` 整序列 `forward_verifying`
取逐行 logits，argmax/top-2 在设备上算、一次读回）。

方法学校注：teacher-forced 假设成立——链式草稿被接受时条件化的正是 ground
truth 前缀，失配即周期结束，无发散路径；prefill 式整段前向与运行时逐位解码在
近平局 argmax 上可能翻转个位数位置（±1-2% p，35B 的 verify_commit 门就是为
这类差设立），不影响量级结论。prose 两个变体 p 相差 0.14（70 对 420 位），
说明 p 对文体敏感，但乐观端 0.59 也只到 τ̂ 2.46。

## 为什么即使 τ̂ 达标，独立 1B 草稿也算不过来（成本模型）

独立草稿每周期要 **L 次串行** 1B 前向（无训练过的 masked/MTP 头就不能像 DSpark
那样一次并行出整块）。1B 文本塔 ~1.0B 参数 BF16 ≈ 2.0-2.4GB/步 ≈ 5-6ms
（400GB/s）。以实测 τ̂ 代入 prose 周期（现 85ms @ block 3）：

| 配置 | 追加 draft 成本/周期 | 净 tok/s | 对比现状 23.3 |
| --- | --- | --- | --- |
| BF16，block 3 | +15-18ms | ~20.6 | 更差 |
| BF16，block 15 | +75-90ms | ~14.7 | 更差 |
| MTP×2 + int4 1B（都不存在，需新资产） | +3-8ms | ~23.7 | 噪声级 |

现役 389M 草稿的结构优势正是侦察独立复现出的 DSpark 论文论点：target 隐状态
条件化 + 并行整块提议，让 1/3 参数的草稿在 code 上达到 p≈0.87（τ 7.69），
超过独立 1B 的 0.848。top-2 树同理：prose 覆盖 0.697 意味着宽度不计成本的
理论上限 ~3.3，而宽度正是带宽账的敌方——#43 的零宽度增益已量过这条路。

## #41 收伞记录与重启条件

prose 归档：**τ≈2.0（p≈0.5）为当前生态下的已证 ceiling**；工程线上限
~33-36 tok/s（发射债归 #52）。重启条件（满足其一再开伞）：

1. OpenBMB 发布更强的 35B-A3B DSpark 草稿（更深/更大 markov rank/更高 τ 卡片数据）。
2. 出现同词表（248144）更强小模型，或愿意为其训练 embed 对齐——届时先用本页
   `draft_recon.rs` 方法量 τ̂，≥3 才立项。
3. 自训 DSpark/EAGLE3 式头（需要训练设施与 35B 激活数据，当前出界）。

## 复现

```text
cd crates/minicpm-v
cargo test --release --features metal --test draft_recon ground_truth -- --ignored --nocapture
cargo test --release --features metal --test draft_recon one_b_match -- --ignored --nocapture
```

需要 `models/MiniCPM-V-4.7-35B-A3B-int4`、`models/MiniCPM-V-4.7-1B` 本地检查点
与 Metal；两次跑串行，GPU 独占。产物在 `artifacts/draft-recon/`。

来源：本地 `models/*/config.json` 与 tokenizer.json 逐条 diff；[MiniCPM-V-4.7-35B-A3B
模型卡](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)（DSpark 配对声明）、
[DSpark 草稿卡](https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark)、
[MiniCPM-V 仓库](https://github.com/openbmb/MiniCPM-V)（4.7 只有 1B/35B-A3B 两个
变体）、[MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B) 与
MiniCPM4-0.5B 的 config.json（vocab 73448）、MiniCPM5-1B-Base config.json
（130560）；接受长度口径沿用 `docs/research/dspark-acceptance-length-reasonable.md`。
