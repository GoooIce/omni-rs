# Kapitsa 的引擎调度与批量解码怎么做

2026-10-27。问题来自 #59（地图 #57）：nace-ai/kapitsa 提交 `d2d149543b4b4e8624e25eca4c6f233a4a20a582` 上，同一提交回答三件事——会话槽与 GPU 轮转锁（先到先得、持锁 30 ms 让位、parking 不留自身在飞任务）与连续合批是什么关系；`run_batch` / `encode_verify_multi` 的多行前向怎么组织；`Engine::draft`（`/v1/draft` 预填充用户下一句）与空闲期助手历史预填充怎么触发、怎么不与在跑请求抢资源。本文只记机制事实，引用均为该提交上的文件与行号（路径相对 kapitsa 仓库根）。

**结论。轮转与合批是并存的两层，轮转同时是合批关闭后的回退路径：票锁 `TurnLock` 是所有 GPU 工作的底座，prefill 永远只按它轮转，批的 runner 自己也持锁并在周期之间让位；连续合批是只发生在「至少两个请求同时在解码、且 `batch_rows(n) > 0`」时的机会性上层，成员在解码周期边界 park 后加入。`NI_BATCH=0`（或成员不足、行数不划算、drafter 未测量、16 行起草块）时 `batch_rows` 返回 0，多请求只剩轮转。`run_batch` 由一个成员线程当 runner 持锁跑全体成员的周期：每成员先起草（7-token 链），再一次 `encode_verify_multi` 前向覆盖所有成员的行——norm、投影、MLP/MoE、LM head 对全部行只跑一次，GDN/Mamba 扫描、注意力、采样第二段按序列分组；总行数不超过 8，每成员行数来自 M4 Pro 上的测量表或 KFlash 批预算逐周期挑选。合批结果与单流逐 token 不逐位相同：行批内核的舍入可翻转近平局，采样请求消耗的随机数流也不同；`NI_BATCH=0` 让多请求回到逐 token 等价的轮转。`draft` 与 `prefill_history` 都只从 `TurnLock::try_lock`（无人持有且无人等待）加 `active_ == 0` 进入，即只在引擎完全空闲时启动；`draft` 在每个 chunk 边界检查新请求并停下（`interrupted`），`prefill_history` 默认拿到锁后跑完整个历史（`NI_HISTORY_YIELD` 默认关，带 DFlash2/LiLiCorr 时永不中途让位）。**

## 0. 请求的一生：ExecutionSession、slot、turn

每个 HTTP 请求经 `ExecutionSession` 起一条自己的工作线程：`src/engine/execution_session.h:9-24` 给出一次性 start/cancel/wait 接口，`src/engine/execution_session.cpp:84-98`（`State::run`）在线程里调用绑定的 work；服务端把 `Engine::chat` 绑成这个 work（`src/server/native_execution.cpp:12-15`）。引擎层的调度全部发生在 `Engine::chat` 内部：请求先在 `src/engine/engine.cpp:2283-2308` 拿 slot，然后持有 GPU turn 走 prefill 与解码。`Engine::chat` 的注释（`src/engine/engine.h:314-321`）写明：在飞请求各占一个 slot、在 prefill chunk 与解码周期之间轮流用 GPU；slot 全忙时先到先得排队；单独一个请求时跑得像只有它一个。

## 1. 会话槽与 GPU 轮转锁

**slot。** `EngineOptions::slots`（`src/engine/engine.h:46-48`）设同时在飞请求数，每个请求一个 slot、自己的 KV cache 与保温的会话状态；不指定时 `auto_slots` 默认 `kAutoSlots = 4`，内存不够降到 2（`src/engine/engine.h:84-93`）。`Slot` 结构在 `src/engine/engine.h:376-408`（`busy`、`arrival`、`decoding`、`mem_wait`、drafter 环 `dseq`/`lseq`、MTP 停靠缓冲 `park_h`、turn 边界 checkpoint `ckpts` 等）。slot 全忙时请求把自己排进 `slot_queue_`（deque，`src/engine/engine.h:575`），只有队首才尝试 `pick_slot`（`src/engine/engine.cpp:2294`），随后在 `turn_.wait_event` 上睡等事件，请求结束释放 slot 时 `turn_.notify()` 唤醒（`src/engine/engine.cpp:2312-2321`）——先到先得。

**票锁。** `TurnLock`（`src/engine/engine.h:259-285`，实现 `src/engine/engine.cpp:137-199`）是 FIFO ticket 锁：`lock` 取号排队（`next_`/`serving_`），`waiting()` 给等待数，`held_ms()` 给持锁时长，`events()/notify()/wait_event` 提供不占锁的事件等待。`try_lock` 只在「无人持有且无人等待」（`next_ == serving_`）时成功（`src/engine/engine.cpp:158-164`），这是空闲期工作（history prefill、draft、SSD 保存）的进入口。

**30 ms 让位。** `Engine::contended`（`src/engine/engine.cpp:201-211`）：有请求在等 turn 且自己已持锁 `NI_TURN_MS`（默认 30 ms）才返回真；`src/engine/engine.h:431-434` 的注释写：30 ms 是 35B 约 19 ms 解码周期的两倍、27B 或 Bolt 的一个周期，每次让位在空闲 GPU 上花约 0.6 ms；`NI_PARK_TEST=1` 强制每步都让（作为 park 路径的精确性检查）。`yield_turn`（`src/engine/engine.cpp:213-217`）就是 unlock+lock：让出者排到所有已等待者之后。prefill 侧另有一条更早的让位点：一个 chunk 很长，chunk 边界直接让位，不等 30 ms（`src/engine/engine.cpp:2695-2703`，`contended(false)` 免去 quantum 检查）。

**parking 不留自身在飞任务。** `yield_turn` 的契约（`src/engine/engine.h:436-437`）：调用者已排空自己的 GPU 工作，模型共享缓冲里不再留它还需要的东西，状态都在自己 slot 里。三个解码循环各自的 park 实现：LiLiCorr 把上个周期保留的行 commit 进 drafter 环并 `commit_and_wait`（`src/engine/engine.cpp:3108-3115`，调用点 3198-3202）；DFlash2 在 verify 之后 park 分支里 `encode_commit` + `commit_and_wait`，并停发下一周期的起草半周期（`src/engine/engine.cpp:3302-3313、3347-3361`）；MTP 路径把下一周期要读的 hidden 行从模型的共享 `hidden_rows()` 缓冲拷进 slot 自己的 `park_h`（`src/engine/engine.cpp:3567-3576`，申请于 3572）。仓库自己的概述（`docs/architecture.md:55-58`）总结为：让位前请求 park，自身无在飞任务、共享缓冲里无它下一步需要的东西，因此只轮转的请求跑的周期和得到的 token 与独占时相同。

## 2. 轮转与连续合批的关系：并存，兼作回退

结构性事实是两层并存：

- **轮转是底座，永远存在。** 批的 runner 也持 turn（见下），且在周期之间当 `contended()` 时让位给等待的请求（`src/engine/engine.cpp:1403-1406`）；prefill 从不合批——`batch_wanted` 只统计 `busy && decoding && mem_wait == none` 的 slot（`src/engine/engine.cpp:1310-1319`），一个在解码的请求旁边一个在 prefill 的请求只会轮转。
- **合批是机会性上层。** 解码周期边界上，`join = batch_wanted()`、`turn = !join && contended()`（LiLiCorr：`src/engine/engine.cpp:3155`；DFlash2：3304-3306）：batch 想要它就 park 后 `join_batch`（3057、3356），否则按 30 ms 规则让 turn。`join_batch`（`src/engine/engine.cpp:1339-1359`）把 job 挂进 `batch_`，第一个加入者成为 runner（`batch_runner_ = true`），成员线程在 `bcv_` 条件变量上、**不持 turn** 睡到自己的 job done/returned 或轮到自己当 runner。runner 的 `run_batch`（1361-1428）循环：`rows = batch_rows(n)`；为 0 时等其它解码请求在周期边界加入（`turn_.wait_event` 最多 1 s，`src/engine/engine.cpp:1366-1375`），等不来则 `batch_backoff_ = now + 2000` 并让所有成员退回各自循环（1376-1378）；KV 不够的成员退回自己的循环（其循环自己去等内存，1381-1388）；成员停止则 `batch_leave`；周期之间 `contended()` 让位。runner 退出时把 runner 交给剩余成员或让最后一个回自己的循环（1408-1427）。仓库概述（`docs/architecture.md:62-81`）与 `src/engine/engine.h:480-488` 的注释同此。

`batch_rows(n)` 返回 0 的全部条件（`src/engine/engine.cpp:1256-1265`）即回退条件：`NI_BATCH=0`；成员数 `< 2` 或 `> 8`；没有 DFlash2 或 LiLiCorr drafter；drafter 未测量（`batch_measured_` 为假且 `NI_BATCH` 未设——默认只有 Qwen3.6 35B-A3B、Qwen3.8 27B 的 DFlash2 与 Bolt 24 GB 的 LiLiCorr 算已测量，`src/engine/engine.cpp:517-521`）；16 行起草块（批循环只会起草 7-token 链，配不上 depth-15 格栅，1263-1265）。另有两个运行期回退：成员没来齐（1 s 超时后 2 s 内不再成批），以及 KV 无余量的成员离开批次。所以对票面问题的回答是：**轮转不是合批专用 的回退路径，而是恒在的底座；`NI_BATCH=0`（及以上条件）把合批层摘掉后，多请求调度就只剩轮转**。单请求独跑时整套批机制不运行，路径就是单流的（`src/engine/engine.h:488`）。

## 3. run_batch / encode_verify_multi 的多行前向

**一次几行、行长怎么定。** 一条 verify 前向的总行数是各成员行数之和，上限 8（`Qwen35::kMaxVerify = 8`，`src/model/qwen35.h:302`；`NemotronH::kMaxRows = 8`，`src/model/nemotron_h.h:149`；`encode_verify_multi` 对超限直接 `fatal`，`src/model/qwen35_v8.cpp:1278`、`src/model/nemotron_h.cpp:2523`）。每成员行数 = 锚点 + 留下的草稿（`1 + k`），来自三处之一（`src/engine/engine.cpp:1256-1308`）：

- 测量表（M4 Pro 实测，注释在 1291-1305）：MoE `{2:4, 3:2, 4:2}`、dense `{2:4, 3:0(轮转), 4:2}`、LiLiCorr `{2:4, 3:2, 4:1}`；成员多于 4 取 1（纯行）；约束 `n * rows > 8` 时砍到 `8 / n`。
- `NI_BATCH_ROWS="n:rows,..."` 覆盖表。
- KFlash 批预算（DFlash2、2-4 名成员，哨兵 `kBatchBudget = 9`）：每周期先跑完各成员起草，再由 `KfBatch::plan` 从草稿校准概率与本机实测批周期成本里挑每成员行数（`src/engine/engine.cpp:1281-1290、1474-1492`）。

**batch_cycle 怎么组织一个周期。** `src/engine/engine.cpp:1430-1613`。两个 command buffer，同单请求的一周期：e1 先做每成员的 staged 行 commit 进 drafter 环与起草 pass（`encode_draft`，批内固定 7-token 链、`tree = false`；采样请求按自身参数采样草稿并保存 walk 分数），e2 是 grouped encoder 上的一次 `encode_verify_multi`，覆盖所有成员的行；随后采样成员的精确投机接受（`encode_accept_rows`，按 pass 产出的 stage-1 候选，1516-1528）。e2 `commit` 后主机读回 `samples()`，逐成员做接受：贪心与 LiLiCorr 成员保留「与前一行自身采样相等」的最长草稿前缀（1564-1570），采样 DFlash2 成员用 GPU 侧精确接受；再经各成员自己的 `handle()` 吞 token、`commit_rows`、`cycle_done`/`trace` 回调。带结构化输出的成员在此路径里由 runner 在 verify 层运行期间算掩码（1496-1511、1543-1549）。

**纯行的流水线变体。** `batch_rows(n) == 1` 时走 `batch_plain_run`（`src/engine/engine.cpp:1619-1732`）：每个 pass 给每成员编码 1 行（`tok_out = true` 让采样直接写进各序列自己的 token 缓冲），主机最多提前编码 8 个 pass（drafter stage 容量），读完一个再排下一个；这仍是 `encode_verify_multi`，只是每成员 `R = 1`。

**哪些跨行共享、哪些按组。** `Qwen35::encode_verify_multi`（契约 `src/model/qwen35.h:318-338`，实现 `src/model/qwen35_v8.cpp:1265-1510`）：行无关部分对全部行只跑一次——norm 进 X8 表（`enc_x8`）、全部线性投影（`enc_qmm8`：qkv/out/o 投影）、MLP 或 MoE（`enc_moe8`/`enc_mlp8`，1447-1448）、final norm + LM head（1465-1466）、采样 top-k 第一段（1490-1494）；按序列分组、在 grouped encoder 里并排跑的是读写各自状态的部分——GDN 卷积窗、递推与 tape（`gdn_prep_verify`、`gdn_step_verify8s`，1318-1358）、各自 KV cache 的写入与注意力（`attn_prep`、`attn8m_partial`、`attn_rows_reduce`，1397-1443）、采样第二段（各成员自己的参数与 RNG，1495-1508）、drafter 捕获（1449-1461）。每组先把该序列的 pending 行折叠（1284-1285），组内行落在共享行缓冲的互斥视图上（`rows()` 视图，1294-1297）。`NemotronH::encode_verify_multi`（契约 `src/model/nemotron_h.h:154-170`，实现 `src/model/nemotron_h.cpp:2508-`）同构：in_proj、注意力投影、MoE、LM head 全行一次；Mamba 扫描（`enc_ssm`，含 merged flush 先折叠各自 pending 行）与各自 KV 注意力按组。

**合批与单流逐 token 的差异从哪来。** 差异是明示的：`docs/architecture.md:82-84`——一起解码的请求可能得到与独跑不同的 token，条件是两个候选落在批 pass 的舍入差之内，且采样请求抽到的随机数不同；`NI_BATCH=0` 可让它们改回轮转。源码里两处来源：其一，行数 ≥2（MoE ≥5）的 verify 走 8 行 MMA 行批内核，小行数走小批内核（`src/model/qwen35.h:300-302`），两族内核浮点舍入不同，近平局可翻转；其二，批内起草是采样链、单流周期是 KFlash 树/格栅（`src/engine/engine.cpp:1263-1265`、`batch_cycle` 的 `encode_draft`），采样请求消耗的随机数流随之不同（采样器本身按「RNG step = 行位置」组织，`src/model/qwen35.h:321-322`、`qwen35_v8.cpp:1500`，与单流一致，故贪心近同而采样流不同）。对拍工具是 `NI_TRACE_TOKENS`（`src/engine/engine.cpp:3018`：把每个投机周期的 token 与 logits 行写出，测试用来对比批与单流输出）；`NI_BATCH_EXACT`（默认开）控制采样成员是否走精确投机接受（1450-1453）。

## 4. Engine::draft 与空闲期 history prefill

**`/v1/draft`（用户下一句预读）。** 端点在 `src/server/main.cpp:1478-1520`：请求体 `{"after": 上一条助手回复原文, "text": 当前草稿}`，直接调 `Engine::draft`，永不切换模型。引擎侧 `src/engine/engine.cpp:4180-4366`：

- 触发与匹配：仅 Qwen3.5 族（4191）；`remember_reply` 只保留最近 32 条以 `stop` 结束的回复（`src/engine/engine.h:577-586`、`src/engine/engine.cpp:4161-4178`），`draft` 从新到旧找 `reply` 与 `after` 相等的那条；找不到答 `no_conversation`。
- CPU 部分（不占 GPU turn）：`draft_prompt` 把「已答请求 + 草稿可读部分」渲染成下一请求的提示（`draft_readable` 截到最后一段空白之前，4223 经 `src/engine/engine.h:226-239`），分词。
- 进入 GPU 的门槛（4237-4239）：`std::try_to_lock` 拿 turn 且 `active_ == 0`，否则答 `busy`——即只在引擎完全空闲时启动。
- 读什么：读到最后一个 token 之前（`end = n - 1`，4233；正请求自己读那个 token 配 MTP 对）；草稿在已预读状态之后则续 live state，被编辑过则先恢复 common prefix 内最新的 turn checkpoint（4247-4264），并在用户回合起点 `turn_at = user_at - 1` 存 checkpoint，使任何后续编辑都能回到同一起点（4268-4302）。
- 不与在跑请求抢：chunk 循环里每个 chunk `commit_and_wait` 落地后检查 `requests_ != seen || turn_.waiting() > 0`，有新请求或有人在等 turn 就停下，slot 保留已读部分，答 `interrupted`（4311-4316、4361）。

**空闲期助手历史预填充（`prefill_history`）。** 服务端在每个回复结束后调 `model_mgr->prefill_after`（`src/server/main.cpp:1249-1252`，四个 API 路径都挂），`src/server/model_manager.h:299-340` 为每个回复起一条自己的线程（列表 `prefill_ths_`，收尾时 join 已完成者；同会话更新的请求会把旧回复的预充标记 `stale`（superseded）；pending 上限 `kMaxPending = 4`，`src/server/model_manager.h:579`）。引擎侧 `src/engine/engine.cpp:3878-4159`：

- 触发条件（3879-3892）：`NI_HISTORY_PREFILL` 默认开；只有 `finish_reason == "stop"` 的回复做（下一条是用户提问，模板会去掉本轮 think 块，live state 与下一请求的渲染不匹配；tool call 后不匹配问题不存在）。
- 空闲门槛（3883-3933）：先睡 `NI_HISTORY_IDLE_MS`（默认 250 ms）；然后最多等 `NI_HISTORY_WAIT_MS`（默认 5000 ms）等一个安静时刻——`QuietGate`（`src/engine/history_gate.h`）判「idle_ms 内无新请求、无在跑请求」，加 `try_lock` 拿到 turn 且 `active_ == 0` 且 `requests_` 未变；等不到就放弃（日志 `history: given up`）。即它绝不排在任何在跑请求前面，只在完全空闲的缝隙里启动。
- 做什么：把会话渲染成下一请求将要的样子（`with_answer` + 占位用户回合，3962-3981），从最深的 turn checkpoint（或 SSD 恢复，含 `NI_HISTORY_ANYSLOT` 借别的 slot，3934-3967、4011-4069）补 prefill，同样少读最后一个 token（3995-3997）；带 drafter 时行进 drafter 环（4085-4094）。
- 启动之后：chunk 循环（4110-4147）默认**不**中途让位——`NI_HISTORY_YIELD` 默认关；开着也仅限无 drafter 的模型（`can_yield`，4104-4108）。中途让位的代价写在注释里（4098-4107）：后来的请求首 token 从 2.7 s 提前到 1.25 s，但被打断会话的下一问只能从缓存读 9206 行中的 5216 行（1.44 s 对 0.06 s）。所以「不抢资源」的分界是：**进入靠空闲门槛 + `try_lock`；进入后默认持有 turn 到跑完**，等待中的新请求按 FIFO 排在它后面。

## 材料

- nace-ai/kapitsa，提交 `d2d149543b4b4e8624e25eca4c6f233a4a20a582`（Flash-Next runs Sushi's EXL3 packs, #722），本地克隆 `/tmp/kapitsa-src`。
- 引擎层：`src/engine/engine.h`（`EngineOptions` 42-73、`TurnLock` 259-285、`Slot` 376-408、`contended`/`yield_turn` 契约 431-437、批量解码段 480-534、`answered_` 577-586）；`src/engine/engine.cpp`（TurnLock 实现 137-199、`contended`/`yield_turn` 201-217、`batch_measured_` 517-521、slot 队列 2283-2325、prefill 让位 2695-2703、`NI_TRACE_TOKENS` 3018、批加入点 3046-3062/3155/3198-3202/3304-3306/3347-3361、MTP park 3567-3576、`batch_rows` 1256-1308、`batch_wanted` 1310-1319、`batch_leave`/`join_batch` 1321-1359、`run_batch` 1361-1428、`batch_cycle` 1430-1613、`batch_plain_run` 1619-1732、`prefill_history` 3878-4159、`remember_reply` 4161-4178、`draft` 4180-4366）。
- 会话外壳：`src/engine/execution_session.h:9-24`、`src/engine/execution_session.cpp:17-99`、`src/engine/execution_session_internal.h`、`src/engine/history_gate.h`、`src/server/native_execution.cpp:12-15`。
- 模型层：`src/model/qwen35.h:300-352`（`VerifyGroup` 契约与 `kMaxVerify`）、`src/model/qwen35_v8.cpp:1265-1510`（`encode_verify_multi`）、`src/model/nemotron_h.h:148-202`、`src/model/nemotron_h.cpp:2508-`。
- 服务端：`src/server/main.cpp:1249-1252、1478-1520`；`src/server/model_manager.h:299-340、579`。
- 仓库自述：`docs/architecture.md:40-103`（轮转与合批的作者概述、`ExecutionSession` 所有权）。
