# Kapitsa 的投机机制由哪些步骤组成

2026-10-27。问题（wayfinder #58，地图 #57）：钉死 `nace-ai/kapitsa@d2d149543b4b4e8624e25eca4c6f233a4a20a582`，只从源码回答四件事——KFlash 的树校验怎么构造树（best-first 的形状、行从哪来）；树形怎么按标定接受度与测得的每行成本定大小；自适应校验怎么按请求收预算；DFlash2 / DFlash / MTP / LiLiCorr / DSpark 这些 drafter 在同一接缝上怎么并存与选择。每条机制给出它依赖的输入和它改变的东西。`docs/architecture.md` 只用来定位文件，一切结论钉在源码行上。本文只记机制事实，不做「值不值得搬进 omni-rs」的判断。

**结论：Kapitsa 把投机解码拆成「一个后端无关的宿主预算（`KfBudget`/`KfBatch`/`KfLookup`）+ 两个 GPU 共享内核（建树 `kf_build`、接受 `kf_accept`）+ 各 drafter 自产的候选格（children 表）」。树校验的行不来自任何贪心链，而来自 drafter 一次 8 行块前向产出的 top-K 候选格：转移分在 GPU 上归一成 children 表（每格 = 「slot s-1 选了候选 p 之后，slot s 的 K 个候选各自的 log q」），`kf_build` 用按深度在线标定的路径概率做 best-first 展开，尺寸规则在同一个内核里按 `E(n) − rate·cost[n]` 选活行数 n，全程无宿主往返。标定是每深度一条在线 logistic 回归（α/β，从自己校验的接受记录拟合）；成本是每行数一个在线测得的周期 GPU 毫秒（base + 每行数偏移，宽校验用对数比率模型）；两者按「模型 + drafter + 位宽 + 起草块 + 芯片」键跨请求、跨进程缓存。预算不按请求分：每引擎一份、每周期 write/observe、每请求收尾 save；真正按请求的是 prompt-lookup 预算 `KfLookup`（不读时钟，保证同请求同文本）和链模式的深度门 `DepthGate`。drafter 接缝是引擎构造期的一次性排他选择（`plan_load` → `DFlash2::find` / `NhDrafter::open`，内存放不下就退 MTP 或 plain），运行期按 `lili_ / dfl_ / dory / ngram / MTP` 分派到六个互斥的解码循环；DFlash v1 是 `DFlash2` 类的一个配置位，LiLiCorr 与 DSpark 共享 `NhDrafter` 接口与 `LiliSeq` 状态。**

文内行号都指该提交。涉及文件：`src/kernels/kflash.h`（树描述符 + 建树/接受内核，宿主与 GPU 共享）、`src/model/kflash_budget.{h,cpp}`（预算）、`src/model/kflash_wrap.h`（每请求包装与 prompt-lookup 预算）、`src/model/dflash2.{h,cpp}`、`src/model/nh_drafter.{h,cpp}`、`src/model/lilicorr.{h,cpp}`、`src/model/dspark.{h,cpp}`、`src/kernels/spec_dflash.metal`、`src/kernels/nemotron/nh_tree.metal`、`src/engine/engine.cpp`、`src/model/qwen35.h`。

## 0. 一个树周期的六步

以 Nemotron LiLiCorr 循环为例（`engine.cpp:3083-3212`），DFlash2 循环同构（`engine.cpp:3213-3393`，双缓冲）：

1. 宿主在锁内 `KfBudget::write(KfParams&)`（`kflash_budget.cpp:334-392`）：写出的内容是每深度标定 α/β、每行数成本表、探索强制 `force_n`、允许的行数位集、容量 `cap`、近周期 tokens/ms `rate`。
2. 起草半场（一个 command buffer）：drafter 的 8 行块前向（anchor + 7 个 mask 行）产出每 slot 的 top-K 候选与转移分；接着 GPU 内核把候选格归一成 children 表（`dfl_kf_table` / `nh_dsp_table` / `nh_lili_table`）；最后 `kf_build` 在同一 buffer 里 best-first 建树、选活行数，写树描述符（`nh_drafter.cpp:89-123`、`dflash2.cpp:1828-1855`）。DFlash2 的 GPU 驱动变体 `kf_build_gpu` 连 anchor 都从 GPU 侧 drafter 状态读（`spec_dflash.metal:1612-1627`），宿主不进循环。
3. 校验半场：目标模型按树跑一次多行前向——每行只 attend 历史 + 自己的祖先（Qwen：`qwen35.h:402-435` 的 `set_verify_tree`；Nemotron：`nemotron_h.h:172-181` 的 `set_tree` + `nh_ops.metal` 的 `nh_ssm1_tree` / `nh_attn_partial_tree`），递归层从自己分支的状态跑，MoE 只为活行路由专家（`dflash2.cpp:1900-1919`，`set_verify_row_limit(&kf_live_)`）。
4. `kf_accept`（`kflash.h:270-324`）：从根沿「token 等于该行自己样本」的子节点下降，接受最长一致分支；写回接受路径、下一周期的 pend 行、标定记录（每个「父在接受路径上的活节点」的深度 / log q / 是否命中）和预算观测字 `kKfObs`（活行数、E(n)、容量、建成行数、E(8)/E(16)/E(32)）。
5. 宿主 wait 后：`KfBudget::observe(tree, draft->gpu_ms() + verify->gpu_ms())`（`engine.cpp:3164`、`engine.cpp:3319`）——记录进标定环、成本表更新、速率更新、每 16 周期 Newton 重拟合、每 32 周期宽容量决策（`kflash_budget.cpp:207-296`）。
6. 接受路径的前 m 行喂回目标（`commit_rows` / `commit_rows_tree`），drafter 环按路径行 gather 提交（`nh_drafter.cpp:73-87` `encode_commit_path`；`nh_kf_gather`，`nh_tree.metal:96-104`）。

贪心与采样同一个接受：`samp[r]` 是目标在该行根路径位置上自己的 argmax（T=0）或按位置键均匀数的朴素采样（T>0），树在样本抽出前已由 drafter 固定，所以输出序列与 plain decoding 逐 token 相同（`kflash.h:254-263`）。DFlash2 在 T>0 走精确投机采样（`dflash2.cpp:1880-1897` `encode_accept`；批量侧 `encode_accept_rows`）。

## 1. 树怎么构造：best-first 的形状与行的来源

**行从哪来：候选格，不是链。** 每个块 drafter 的一次起草 pass 产出「slot s 的 K 个候选 + (s-1 的候选 p → s 的候选 c) 转移分」，三个后端各自归一成同一个 children 表布局：

- 表布局（`kflash.h:80-89`）：children-row `(s·K + p)` 存「slot s 的 K 个候选在 slot s-1 选了 p 之后的 log q」；`clq[row][c]` log 概率、`ctok[row][c]` token、`cnext[row][c]` 该子节点自己的 children-row（一阶格：`kf::lattice_next`，`kflash.h:83-89`，S 个 slot 每格 K 路，最后一个 slot 无后继）。
- DFlash2（K=16）：起草块 8 行（或 16 行，`NI_DFLASH_BLOCK`）过 drafter 层，目标 LM 头（或缩头 `NI_DFLASH_HEAD`，默认 65536 行）给每 slot 的 logits，`dfl_top16` 取每 slot top-16（`dflash2.cpp:1806-1818`）；候选选择器头（`sel_w_` + codebook）给转移分，`dfl_lattice` 写 `edges_[7][16][16]`（`dflash2.cpp:1820-1827`）；`dfl_kf_table` 把 `clq[s·16+p][c] = log softmax_k(cand_v[s][c] + E[s][p][c])`（`spec_dflash.metal:1585-1607`）。DFlash v1 没有选择器，edges 恒 0，每提案只来自自己行的 top-16（`dflash2.cpp:711-712`、`dflash2.h:27-32`）。
- DSpark（K 默认 16，`NI_DSPARK_K` 至 32）：6 层因果块 + 目标 LM 头每 slot top-K，Markov 头 `logits_i += W2 W1[token_{i-1}]`（rank 512）对每对 (p, c) 修正；`nh_dsp_table` 在 K 个候选 + 格外概率质量上归一（`nh_tree.metal:6-53`、`dspark.h:1-8`）。
- LiLiCorr（K=8，config 强校验 `topk == 8`，`lilicorr.cpp:306-316`）：5 层双向块，头给转移打分 `ls·(in[s,c]·out[s-1,p])`；`nh_lili_table` 在 slot 的 K 候选上 log softmax，再除以格上的概率质量 `min(1, mass)`，让「K 个候选全 miss 目标 token」的 slot 不被当成必然（`nh_tree.metal:55-88`）。

**best-first 的形状**（`kf_build_impl`，`kflash.h:125-229`，1×32 线程，12 KB threadgroup）：

- 根（row 0）是 anchor `tokens[P]`，分数 0；frontier 初始为根的 K 个子节点。
- 节点分数 = 根路径上 `log p` 之和，`p = sigmoid(α_d + β_d·logit(q))` 是该深度标定后的接受概率（`kflash.h:118-122` `kf_lcal`）。因为每步加的是非正 log p，**子节点分数必不高于父节点**——所以按分数从 frontier 取节点（simd 规约，平局取更早的 frontier 项，`kflash.h:149-165`）时，行按插入序写出且每行父行必在它之前。这给出树描述符的核心不变量：「前 n 行就是 n 行预算下的最优树，对每个 n」（`kflash.h:8-10`）。
- 取到容量 R（= 本周期校验容量）为止；建出的节点数 m 记在 `kKfNodes`，活行 n ≤ m。容量之外的行（到校验缓冲的行容量）填一条永不为活的 filler 链（`kflash.h:215-222`）。`kKfJump` 位图标记「父行不是前一行的行」，校验内核据此重放该行的祖先（`kflash.h:23-24`）。
- 描述符是 336 个 int32 的定长块（`kflash.h:6-30`）：每行的父 / 深度 / 祖先-or-自身位掩码 / log q / 标定路径分，加接受结果 `kKfA`、接受路径 `kKfPath`、标定记录区、`kKfE[n]` 期望 token 表、观测字 `kKfObs`。宿主与 GPU 共享同一布局（`kflash_budget.h` include 之，两份 Metal 库嵌入之）。

尺寸规则也在这同一个内核尾部的 lane 0（`kflash.h:196-212`），见下节。**建树全程在 GPU 上、在起草 pass 的同一 command buffer 里**：批量路径里唯一的宿主决策点在起草 pass 提交并等待之后（`engine.cpp:1474-1492`）。

## 2. 树形怎么定大小：标定接受度 × 测得的每行成本

三个在线量，全部只来自本机本模型本 drafter 的历史周期：

**（a）标定（输入：草稿分布 → 输出：每深度 α/β）。** 记录 = 每个「父节点被接受的已校验活节点」的 `(深度, logit(q), 是否命中)`，由 `kf_accept` 写进描述符（`kflash.h:297-303`），每深度一个 2048 条的环（`kflash_budget.h:113-117`）。拟合是 L2 正则（λ=1，先验约 4 条记录的权重）向恒等映射 (α=0, β=1) 的 logistic 回归，Newton 步从当前值出发，α 夹在 ±8、β 夹在 [0.1, 4]（保住兄弟排序），每 16 个周期对 ≥32 条记录的深度各走 4 步（`kflash_budget.cpp:153-175`、`298-317`）。无记录时 p = q。`KfBatch` 的贪心链记录也喂同一标定（`kflash_budget.cpp:648`）；语法 mask 周期与 masked 批成员的接受不喂（`engine.cpp:3315-3319`、`kflash_budget.cpp:605-607`）。

**（b）成本（输入：测得的周期 GPU 毫秒 → 输出：行数 → ms 表）。** 一个 n 行周期的成本 = 起草半场 command buffer 的 `gpu_ms()` + 校验的（`engine.cpp:3164`、`engine.cpp:3319`）。`KfCosts` 把它拆成「所有周期共动的 base（上下文长度）+ 每行数自己的偏移」：本进程首周期定 base，之后每次观测 base 吃掉残差的一半、该行数偏移走步长 `1/min(n+1,32)` 的运行均值（`kflash_budget.cpp:52-73`）。窄校验之外，宽校验（>8 行按一次宽验证跑）用对数成本模型 `log cost(n) = lb_ + lq_[n]`（base 是当前 8 行窄校验的成本，比率各自在线拟合，`kflash_budget.cpp:229-255`）。探索与保鲜：每个允许行数先各跑 `kExplore=3` 次再让规则接管（缓存表在场时先强制一个表内行数定 base，`kflash_budget.cpp:75-85`、`kflash_budget.cpp:350-374`），每 2048 周期重测最陈旧的行数（`kflash_budget.cpp:262-268`）；`NI_KFLASH_FORCE` 供测试钉死。

**（c）尺寸规则（输入：α/β + 成本表 + 速率 → 输出：活行数 n）。** 建树内核为每个 n 累计 `E(n) = 1 + 前 n−1 行的标定路径概率之和`（「n 行校验期望发出的 token 数」）写入 `kKfE`，然后取 **最大化 `E(n) − rate·cost[n]` 的允许 n**；`rate` 是近周期的 tokens/ms（衰减和，衰变 63/64，满 16 个周期才用，之前用 `E(n)/cost[n]` 比值规则；`kflash_budget.cpp:258-261`、`kflash_budget.cpp:379`）。一行值得不值得，判据是「它加的期望 token 是否够付它在长期速率下的毫秒」。`force_n` 覆盖（探索 / 刷新 / 环境变量），没有允许的 n 落在建成行数内时取全部（`kflash.h:196-212`）。

**允许哪些 n（后端能力决定）：** MoE 目标（Qwen 35B-A3B、Nemotron 全系）每行都付专家钱——允许 1..8 全体，校验只为活行路由专家与跑递归步（`dflash2.cpp:603-612`、架构文档步骤 4）；dense 目标每周期反正把权重读一遍——只允许整 8 行瓦片 8/16/32（`Qwen35::kf_dense_rows` + `NI_KFLASH_ROWS` 限制，`dflash2.cpp:609-631`），>8 的计数作为一次宽验证跑（GPU 按活瓦片跑，一次权重读，`qwen35.h:412-435`）。宽容量本身也是决策：默认每周期以容量 8 跑，仅当近 32 周期的树史（E(8)/E(16)/E(32) 环，容量 8 时每 `kStatsEvery=8` 个周期仍建满树来记 E 统计）说「逐周期在宽计数里挑」按宽成本算出的 tokens/ms 超过 8 行窄校验时才切宽（`eval_wide`，`kflash_budget.cpp:179-205`、`kflash_budget.cpp:269-286`、`kflash_budget.cpp:380-386`）。16 行起草块例外：一次权重读起草整格，只校 8 行等于丢掉一半草稿，直接 `pin_wide()` 钉死容量（实测 69.3 vs 90.6 tok/s，`dflash2.cpp:633-662`、`kflash_budget.h:73-77`）。树宽上限 `kMaxRows=32`；`tree_rows_` = 允许集最宽值，目标在加载期按它分配校验缓冲（`dflash2.cpp:656-667`）。

树开关：默认开（`kf_tree_default`，自 #187）；`NI_KFLASH=0` / `NI_DFLASH_TREE=0` 回链；`NI_DFLASH_CAP=<n>`（F74 深度帽，opt-in）在低于满深时保链砍深，因为树需要满深格（`dflash2.cpp:366-400`、`dflash2.cpp:1454-1461`）。

## 3. 自适应校验怎么按请求收预算

**树预算不分请求，分「模型 + drafter + 位宽 + 起草块 + 芯片」。** `KfBudget` 每引擎一份，构造时给缓存键（Qwen：`<模型>-<drafter>-b<bits>[-blk16]-<芯片>`，`dflash2.cpp:590-598`；Nemotron：`<模型>-<LiLiCorr|DSpark>-<芯片>`，`engine.cpp:429-438`），状态存 `KAPITSA_HOME/kflash/<key>.json`：成本表（带 `cost_of: "cycle GPU ms"` 记账口径戳，不同口径的旧缓存只留标定不留成本，`kflash_budget.cpp:20-23`）、每深度 α/β、宽校验的容量决策。写盘条件：观测满 32 周期且距上次 save 有新周期；请求循环结束时 `save(KfBudget::kSaveEvery=256)`（`engine.cpp:3209-3212`、`engine.cpp:3385-3388`），server 停止时 `Engine::save_kflash` 三份预算各 save 一次（`engine.cpp:582-586`），析构再 save；写法是临时文件 + rename 原子替换（`kflash_budget.cpp:27-45`）。**「按请求收预算」因此收的是：每个请求结束时把共享预算落盘一次（并打一行 summary 日志：周期数、平均行数、按行数的成本、标定参数、期望 vs 实得 token/周期，`kflash_budget.cpp:394-413`）。**

每周期的协议是 `write() → 起草/建树 → wait → observe()`，两者都在 `mu_` 下（服务器 stop 线程可并发 save，`kflash_budget.h:111`）。观测侧除了标定/成本/速率外还维护：宽容量决策（每 32 周期）、期望对账（`epred_/etok_` 记 E(n) 对实得 a+1，只进日志）、最陈旧行数刷新。**不进观测的周期**：语法 mask 下的 DFlash2 周期（语法会拒掉 drafter 本来押对的提案，教会共享标定一个偏低的接受率，`engine.cpp:3315-3319`）。

真正每请求的预算状态有三块：

- `KfWrap`（`engine.cpp:3006-3017`）：请求级计数器（source 名、周期 / token / 行数）+ prompt-lookup 预算的宿主。`kflash_source()`（`engine.cpp:561-570`）给该引擎唯一的源命名。
- `KfLookup`（`kflash_wrap.h:57-120`，opt-in `NI_KFLASH_BUDGET=1`）：prompt lookup（Nemotron Nano、Flint `--spec`、Dory ngram）的行数预算。输入是**按位置的衰减接受计数**（先验 0.5 记 2 个提案，衰变 0.95/提案）和一个**常数行成本 r**（多一行校验 pass 折多少个 decode step：Flint 0.6、Nemotron 0.25，`NI_KFLASH_ROW_COST` 可覆盖，`engine.cpp:522-529`）；取 `argmax_d (1 + Σ_j Π p(j)) / (1 + r·d)`，1% 边际平局取短 pass。关键设计：**选择不读任何时钟**，行的选择是 token 的纯函数，所以同一请求在负载下文本不变（`kflash_wrap.h:50-56`）；每请求各建一份也是为此（`engine.cpp:3009`）。前 16 个提案用源自己的计数，之后每 16 个带提案的周期全量探测一次保鲜，`d=0` 的周期不产生记录（`kflash_wrap.h:63-79`）。成本 ms（c1 与每行 s）只为日志/校准 r 测量。
- `DepthGate`（`dflash2.h:251-296`，opt-in `NI_DFLASH_GATE`，链模式专用）：每请求的滚动边际接受窗（64 周期位窗，贪心/采样各一扇），把链校验深度向下钳到「边际接受率跌破损益平衡 p 的首个位置」；请求开始 `gate_reset()`（`engine.cpp:3231`），park 保留窗口。树模式下它 inert（`dflash2.cpp:617-622`）。

**批量解码的行预算 `KfBatch`**（`kflash_budget.h:149-224`、`engine.cpp:1430-1613`）：2..4 个成员共享一次 8 行校验 pass 时，每成员验证 anchor + 若干草稿。输入：每成员链上每草稿位的**存活率** `Π_j sigmoid(α_d + β_d·logit(c_j))`（c_j = 该位 16 候选中 walk 分数下的最大概率；贪心成员用 `KfBudget` 的树标定——链节点就是同格的树节点，采样成员有 `KfBatch` 自己的精确投机采样标定，`kflash_budget.h:152-158`）；成本 = 按 (成员数 M, 行数 N) 的 `KfCosts` 表。分配规则 `allocate`：每成员先保 anchor（和可选 floor 草稿），然后所有草稿按存活率降序（平局取更早位、更早成员，保证每成员的草稿是链前缀），**每加一行要求 `存活率 ≥ rate·(cost(N+1) − cost(N))`**，凸成本下这是全局最优分配（`kflash_budget.cpp:481-523`）。未测行数用中位数斜率 + 截距的直线补齐（`cost_line`，`kflash_budget.cpp:525-545`）。宿主时机：起草 pass 提交并等待、读出各成员分数后 `plan()`，然后才编码校验 pass——「行序不依赖草稿本身」保住采样成员的精确性（`engine.cpp:1474-1514`、架构文档）。观测把贪心成员的记录转喂 `KfBudget`（`kflash_budget.cpp:588-649`）。`NI_BATCH_KF=0` 退回按 (M) 查实测表：两人各 3 草稿、三人各 1、四人 plain（`engine.cpp:1277-1296`）。

链模式的动态深度（树的替代，同在「按成本收尺寸」轴上）：MoE 目标 `dfl_depth` 按 τ 阈值在 GPU 上砍活行（默认 τ=0.5/0.3，`dflash2.cpp:545-563`）；LiLiCorr / DSpark 链各有 `dyn_ratio_`（0.04 / 0.2，`lilicorr.h:121-125`、`dspark.h:90-97`）。

## 4. drafter 接缝：并存与选择

**接缝的形状：引擎构造期一次性、互斥地选一个源；运行期按指针分派。** `Engine` 持有 `dfl_`（`DFlash2`，含 DFlash v1）、`lili_`（`NhDrafter` = LiLiCorr 或 DSpark）、`spec_`（Qwen MTP 头）、`nh_ngram_` / `flint_ngram_` / `dory_spec_`（prompt lookup / 自投机）或全无（plain）。选择链在 `plan_load`（`engine.cpp:814-860`）+ 构造（`engine.cpp:392-470`）：

1. `--spec 0` 全关；否则按后端：Qwen（`Qwen35`，含子类）找 DFlash2 目录（`DFlash2::find`：模型旁 `dflash2/`、`*-DFlash2` 前缀最长匹配，config 过 hidden/vocab/层数检查）；Nemotron 找 `dflash/`→`lilicorr/`→`dspark/`（`NhDrafter::find`，`nh_drafter.cpp:32-39`）。**Qwen3.8-Flash-Next 不绑块 drafter**（`engine.cpp:840`），走 MTP。
2. auto 下内存 fit 检查：drafter 会把一条满上下文对话挤出预算就整体弃用（`dflash_left_out`，`engine.cpp:828-860`），DFlash2 还要过 `supports()`（目标 embedding / LM 头格式）。
3. 找到 DFlash2 → `spec_ = false`（MTP 让位，`engine.cpp:410`）；没找到 → Qwen 用 MTP（`spec_ = spec_depth > 0 && has_mtp()`，`engine.cpp:392`）。
4. Nemotron 无块 drafter 的 dense 型（非 NVFP4）默认 prompt lookup K=4；Flint 只在 `--spec`/`NI_FLINT_NGRAM` 要时开（`engine.cpp:449-470`）。
5. 找到块 drafter 且 `NI_KFLASH≠0` → 树预算挂上（Nemotron 侧在引擎里建 `kf_budget_`；Qwen 侧在 `DFlash2` 构造里建 `kf_budget_` + `kf_batch_`）。

运行期是六个互斥循环（`engine.cpp:3083` LiLiCorr/DSpark、`3213` DFlash2、`3394` Dory、`3454`/`3510` prompt lookup、`3562` MTP、`3629` plain），全部经 `KfWrap` 记账、`cycle_done` 回调、park/yield_turn 轮转协议。

**同一接缝上的 drafter 差异：**

| drafter | 载体 | 起草 | 树 | 链化（双缓冲） | 批量 |
|---|---|---|---|---|---|
| DFlash2 | `DFlash2`（35B/27B） | 候选格（选择器头 + edges） | 默认，8/16/32 行 | 是（GPU 驱动态，`gpu_state_reset`） | 是（`KfBatch`，实测项） |
| DFlash v1 | 同类，`cfg_.v1`（4B/9B） | 每提案取本行 top-16（edges=0） | 同上 | 同上 | 否（未实测，`batch_measured_` 排除，`engine.cpp:520-521`） |
| MTP | `Qwen35` 自带头（`qwen35_spec.cpp`） | 固定深度链（默认 MoE 1 / dense 4，`qwen35.h:292-299`） | 无 | 否（朴素两 buffer） | 否 |
| LiLiCorr | `NhDrafter` 实现（Bolt / Super-VL） | 双向块 + 头转移格（K=8） | 默认，1..8 行 | 否（`chains()` false；树也不链，`engine.cpp:3154-3160`） | 是（Bolt 24GB，实测项） |
| DSpark | `NhDrafter` 实现（Lightning） | 因果块 + Markov 头修正格（K=16） | 默认，1..8 行 | 链模式可链；树模式从宿主 P 起草 | 否（未实测） |

- DFlash v1 与 DFlash2 共享 `DFlash2` 全部机制，只差选择器头与格深（v1 用 8 行块 / 2048 窗跑，训练是 16 行 / 4096 窗——块大小是运行期选择，`dflash2.h:27-32`）。
- MTP 不进树预算：固定深度由 `--spec-depth` / `default_spec_depth()` 定，循环只做「MTP 连锁起草 depth 个 → depth+1 行校验 → 前缀匹配接受」（`engine.cpp:3562-3628`）；`KfWrap::planned` 对 MTP / drafter 循环保 0（`kflash_wrap.h:133-134`）。
- LiLiCorr 与 DSpark 共享 `NhDrafter` 接口（上下文捕获钩子、环提交、`encode_draft` / `encode_draft_tree` / `encode_commit_path` / `stage_init`）与 `LiliSeq` 状态（`nh_drafter.h:20-93`）；两者都把自己的格写成 `tlq_[S][K][K]` 再走共享 `enc_kf_build`（`lilicorr.cpp:901-913`、`dspark.cpp:734-748`）。
- 批量接缝：`batch_rows` 先问 `KfBatch`（DFlash2，`NI_BATCH_KF`），否则按实测表（LiLiCorr：`NI_BATCH_KF=0` 时 DFlash2 也走表）；16 行起草块无批量（批循环只起草 7-token 链，`engine.cpp:1256-1301`）。

## 5. 机制 × 输入 × 改变的东西

| 机制 | 依赖的输入 | 改变的东西 |
|---|---|---|
| 标定拟合（`KfBudget::fit`） | 草稿分布（每深度 log q）+ 自己校验的接受记录 | α_d/β_d；进而树节点的路径分数与 E(n) |
| 成本表（`KfCosts` / 宽对数模型） | 每周期实测 GPU ms（起草半场 + 校验） | `cost[n]`；进而尺寸规则的惩罚项与宽容量决策 |
| 尺寸规则（`kf_build` 尾部） | α/β、cost 表、衰减速率 rate、允许行数位集、（探索 force_n） | 本周期活行数 n（MoE：1..8；dense：8/16/32）与校验容量 |
| 宽容量决策（`eval_wide`） | 近 256 周期的 E(8)/E(16)/E(32) + 两种 8 行成本 | 下一批周期以容量 8 还是容量 max_rows 编码校验 |
| 接受（`kf_accept`） | 目标每行自己的样本 + 固定的树 | 接受长度 a、bonus/修正 token、pend 路径、标定记录、观测字 |
| 树校验内核（Qwen `set_verify_tree` / Nemotron `nh_*_tree`） | 树描述符 + 活行数 | 每行只看祖先的注意力 / 分支递归态 / 活行专家路由 |
| `KfLookup` | 按位置接受统计（衰减）+ 常数行成本 r | prompt-lookup 每周期校验的草稿数 d（token 的纯函数） |
| `KfBatch` | 每成员链的存活率（标定 × 候选概率）+ (M,N) 成本表 + 速率 | 批量 pass 里每成员的行分配 k[g]（总 ≤8） |
| `DepthGate` / `dfl_depth` τ / `dyn_ratio_` | 链模式的边际接受统计 / walk 置信度 | 链校验的活深（只向下） |
| drafter 选择（`plan_load`） | 目录发现 + 内存 fit + 格式检查 + `--spec` | 该引擎唯一的起草源与六个循环之一的分派 |

## 材料

- 仓库：`https://github.com/nace-ai/kapitsa`，提交 `d2d149543b4b4e8624e25eca4c6f233a4a20a582`（2026-10-27，"Flash-Next runs Sushi's EXL3 packs …"）。克隆于 `/tmp/kapitsa-src`。
- 共享内核与描述符：`src/kernels/kflash.h`（kf_build_impl 125-229；尺寸规则 196-212；kf_accept 270-324；KfParams 65-78；lattice_next 80-89）。
- 预算：`src/model/kflash_budget.h`（机制总注 1-28；KfCosts 43-63；KfBudget 65-147；KfBatch 149-224）；`src/model/kflash_budget.cpp`（KfCosts::observe 52-73；fit 153-175；eval_wide 179-205；observe 207-296；write 334-392；allocate 481-523；cost_line 525-545；plan 547-586；KfBatch::observe 588-649；load/save 415-467、680-717）。
- 每请求包装与 prompt-lookup 预算：`src/model/kflash_wrap.h`（全文 1-147）。
- DFlash2 / DFlash v1：`src/model/dflash2.h`（v1 配置 27-32；块行 85-90；树注 223-240；DepthGate 251-296；树缓冲 403-414）；`src/model/dflash2.cpp`（kf_tree_default 366-400；构造 500-676；set_tree 1454-1461；observe 1463-1465；起草与建树 1774-1864；encode_cycle_verify 1899-1971）。
- Nemotron drafter：`src/model/nh_drafter.h`、`src/model/nh_drafter.cpp`（find/open 32-53；enc_kf_build 89-123；encode_commit_path 73-87）；`src/model/lilicorr.h`（1-8、121-128）；`src/model/dspark.h`（1-8、90-100、108）；`src/kernels/nemotron/nh_tree.metal`（nh_dsp_table 14-53；nh_lili_table 63-88；nh_kf_gather 96-104）；`src/model/nemotron_h.h`（set_tree / live_rows 172-199）。
- 引擎：`src/engine/engine.cpp`（drafter 装载与选择 392-470、814-860；kflash_source 561-570；save_kflash 582-586；batch_rows 1256-1301；batch_cycle 1430-1613；LiLiCorr 循环 3083-3212；DFlash2 循环 3213-3393；Dory 3394-3453；prompt lookup 3454-3561；MTP 3562-3628；plain 3629-3674；KfWrap 装配 3006-3017）。
- Qwen 侧树校验与 MTP：`src/model/qwen35.h`（default_spec_depth 292-299；set_verify_tree / verify_wide / kf_dense_rows 注 402-435）；`src/model/qwen35_spec.cpp`（MTP 装载 201-364）。
- 导览（非证据）：`docs/architecture.md`；`docs/kflash.md`、`docs/kflash-map.md` 未用作结论来源。
