# Kapitsa 的状态复用与预算账怎么记

2026-10-27。研究票 #60（地图 #57）。问题是：同一提交 `nace-ai/kapitsa@d2d149543b4b4e8624e25eca4c6f233a4a20a582` 上，`prefix_cache.h` 的 SSD 前缀缓存生命周期（何时写盘、失效条件、跨重启怎么命中、与内存前缀缓存的关系）；会话槽内 KV 与递归状态检查点怎么复用；`kflash_budget.h` 的预算用什么测得的量标定，内存压力怎么触发释放（图像缓存、空闲 KV、ANE 资源的释放顺序）。本文只记机制事实，不做值不值得的判断。与本仓对照的两条负载是 35B int4 + DSpark 单流、resin 合批（双请求 + 前缀命中）；模型家族专属的递归状态机制（Dory 的 loop 计数、Flint 的 n-gram 起草）只在它们与通用机制交叉处点名，不展开。

一处路径勘误：票面写的 `src/engine/kflash_budget.h` 在该提交下实际是 `src/model/kflash_budget.h`（实现 `src/model/kflash_budget.cpp`）；`src/engine/` 下只有 `prefix_cache.{h,cpp}` 与 `execution_session.{h,cpp}`。

**事实摘要：SSD 缓存按「命名空间 = 模型文件指纹 + 起草器 + 状态布局」分目录，空闲 10 秒后由后台线程分块写盘，槽被替换前先补写 SSD 缺的部分，限额 min(配额, 可用空间的四分之一)内 LRU 逐出，重启后扫目录重建索引按 token 前缀 + 恢复点深度命中；槽内复用是三选一——提示延伸活状态则续活状态，否则恢复公共前缀内最深的轮次边界检查点（系统块末尾的前缀检查点只增不覆、可跨槽拷贝），换对话时最浅代价逐出；KFlash 预算标定量是 Metal 命令缓冲的 GPU 时间（起草半拍 + 校验拍 `gpu_ms()` 之和）与逐深度的接受记录（logistic 回归拟合 p = sigmoid(αd + βd·logit q)），行数选 E(n) − rate·cost[n] 最大者，表落盘 `kflash/<模型-起草器-芯片>.json`；内存压力释放次序在 `reserve_kv` 里是：本槽无复用先自弃 → 早先请求的图像 embedding 缓存 → 空闲槽 KV（LRU，先落 SSD）→ 等预填的请求的 KV（最新先弃）→ ANE split 面 → 图像编码器权重。**

## 1. SSD 前缀缓存（`src/engine/prefix_cache.h` / `prefix_cache.cpp`）

### 目录、键与命名空间

根目录是 `$KAPITSA_HOME/prefix-cache`（`KAPITSA_HOME` 缺省 `~/.cache/kapitsa`，`src/hub/hub.h` 的 `home_dir`、`src/hub/resolve.cpp:75`；`NI_DISK_CACHE_DIR` 可改，`src/engine/engine_disk.cpp` `disk_start`）。每模型一个命名空间目录，名字是 `about` 串的 128 位哈希（`engine_disk.cpp` `disk_start`）：`about` 含状态格式号 `kStateFormat`（当前 1，状态字节含义一变就 bump，旧命名空间永不再读、14 天不用即删，`engine_disk.cpp:39`）、模型目录指纹 `folder_id`（目录内每个文件的 名字+大小+mtime 各一行，加 `config.json` 的 CRC-32C，`engine_disk.cpp:120`）、模型自己的 `disk_format()` 串、解码方式（DFlash2 起草器位数 / MTP 头 / 无起草器）、起草器目录指纹、每块行数。即换权重、换起草器、改状态布局都会得到新命名空间；旧的不清就占限额，14 天不用或需要腾地方时删。

命名空间内三类文件（`prefix_cache.h` 头注释与 `Kind` 枚举）：`kv/<key>` 一块位置的 KV 行（`kBlockRows = 512` 行/块，35B 上约 10 MB/块，`engine_disk.cpp:44`）；`st/<key>` 一个位置的状态（递归状态 + 起草器上下文环，MTP 时还有头自己的 KV 行与收集的草稿行）；`e/<key>` 一条会话的 entry（token 序列、块键列表、恢复点列表、图像 span+digest 附带记录）。块键按内容链式哈希：前一块键 + 行区间 + 区间 token + 计算这段的请求 id（`disk_plan` 的 `chain`，`engine_disk.cpp:440`），下一轮只加新块新状态（头注释：「a write adds the new blocks and states only」）。每个文件是 4 KiB 头（magic、kind、命名空间、键、尺寸、负载与头的 CRC-32C）加负载，写临时文件再 rename，读写都 `F_NOCACHE`（`prefix_cache.cpp` `write_file`/`read`）。

### 何时写盘

四条路径（`engine_disk.cpp` 头注释与实现）：

1. **写滞后（write-behind）**：一轮结束后服务器空闲 `NI_DISK_CACHE_IDLE_MS`（缺省 10 s），磁盘线程挑「已变且未存」的空闲槽，最近使用者优先（`disk_loop`，`engine_disk.cpp:768` 起）。每块拷贝都要拿 GPU 轮（`turn_.try_lock()` 且 `active_ == 0`），请求一来就让位，已写的留给下次续写；解码循环之间什么都不写。每分钟 `store.measure()` 一次，每小时 `touch()` 一次（刷新命名空间 lock 文件 mtime = 最后使用时间）。
2. **替换前抢救（`disk_leaving`，`engine_disk.cpp:600`）**：一个请求即将用别的对话（或为内存）替换槽里 SSD 还没有的对话时，先当场拷贝 SSD 缺的块与状态，交给后台线程写。拷贝量上限 `capture_max = min(1 GiB, 物理内存/32)`；超限则按「写盘毫秒 vs 重读这些行的预填毫秒」权衡（`disk_wait::decide`，双方都是测得的速度：写速按 ≥8 MB 写平滑估、预填速按 ≥256 行的预填平滑估，`engine_disk.cpp:250`、`disk_note_prefill`），值得等就直接从槽写盘（至多 3 s），否则放弃并留日志。
3. **落盘确认（`disk_confirm` / `#467`）**：槽的 `disk_version` 只在写真正落地后才等于 `version`；被拒（没空间）或写失败的版本保持未存，history prefill 要替换它时放弃而不是丢对话（`disk_save_slot` 的 `finish`/`SaveOutcome`，`engine_disk.cpp:509` 起）。重试按 `NI_DISK_CACHE_RETRY_MS`（缺省 60 s）退避，分「全缓存没地方」与「只有这条太大」两级（`Disk::refused_all` / `refused_version`）。
4. **服务器停止（`save_disk_cache`，`engine_disk.cpp:789`）**：按预算毫秒（缺省 8000）把没存的槽按最近使用顺序写完。

存什么由 `disk_plan`（`engine_disk.cpp:407`）决定：活状态（`live_ok` 且 ≥ `min_pos`）作为 `Snap::kLive` 恢复点；轮次检查点里位置 ≥ `NI_DISK_CACHE_MIN`（缺省 1024，下限 64；「靠近开头的状态不存，重新预填更快」）的作为恢复点，`query` 检查点带 `Snap::kQuery` 旗标；DFlash2 的前缀检查点跳过（它没有起草器环快照，恢复后环无法重建，`engine_disk.cpp:419`）。

### 失效条件

- **换模型/起草器/状态布局**：命名空间整个换掉（见上），不是逐条失效。
- **空间**：限额 `space_limit = min(quota, max(0, avail + held − floor)/4)`，`floor = min(10 GiB, 卷容量/20)`；`held` 算自己文件，所以自己写不缩自己的限额，别人填盘按四分之一折进来（`prefix_cache.h:66`、`prefix_cache.cpp:328`）。`auto` 配额上限 32 GiB。非本地卷（网络盘）不追可用空间：给了尺寸就按尺寸，auto 则不存（`measure()`，`prefix_cache.cpp:544`）。腾地方次序（`make_room`，`prefix_cache.cpp:587`）：先删无引用孤儿块，再整条 entry LRU 逐出（被 pin 的除外）；还不够先删别的命名空间里没人持锁的、最旧的（`remove_others`）。磁盘真满（ENOSPC/EDQUOT）触发 `measure()` 收缩并只报一次（`write_file`）。
- **损坏**：读时 CRC 或头不对 → 删该块并连带删所有需要它的 entry（`drop_damaged`）；开仓时 entry 引用的块缺失 → 删 entry；没 entry 需要的块 → 删块；残留 `.tmp.` 文件 → 删（`open()`，`prefix_cache.cpp:427` 起）。
- **时间**：命名空间 lock 文件 mtime 距今超 `stale_days`（14 天）→ 开仓时整目录删（`open()` 里的他命名空间扫描）。

### 跨重启怎么命中

文件就是持久层，索引是内存的，重启时 `open()` 扫目录重建（entry 逐个整读并校验一致性）。互斥用 `<ns>/lock` 的 flock：一个服务器一个命名空间，另一个等最多 30 s（`lock_wait_s`）；开仓时 `touch()` 刷新使用时间。LRU 顺序也持久：`unpin(used=true)` 时 `utimes` entry 文件（「the order of use survives a restart」，`prefix_cache.cpp:709`）。查找 `lookup(keys, n, below, tag, reusable)`：遍历同 `tag` entry，比 token 前缀（`reusable` 回调可再砍——引擎拿它比图像：前缀里每张图的 span+digest 都得是同一张图才可复用，`disk_lookup` 的 `same_images`），取「恢复点位置 ≤ 前缀长度且最深」者，同深取最近使用。读回（`disk_restore`）状态最大先读，然后 KV 块，至多 4 线程并行读（「SSD 并行读快于单读」），读时不持 GPU 轮；读到的状态字节级等于槽内状态，「a request resumed from it decodes as it would from the slot」。读回后 `slot.turns ≥ 2`（算续聊）、`live_ok = false`（从检查点语义续）。entry 写入时的去重：恢复点集合被新 entry 完全覆盖的旧 entry 删除（`put`）。

### 与内存前缀缓存（槽）的关系

两者是同一套 KV/状态的两级，读取时先算槽自己能给多少：`reuse_start(slot, common, n)`——提示延伸活状态则续活状态，否则公共前缀内最深的检查点（`engine_disk.cpp:327`）；SSD 只在其命中比这更深时才用（`disk_lookup` 返回条件 `h.pos > ram`，`engine_disk.cpp:899`），命中则槽内原状态整个丢弃（`engine.cpp:2345` 清 tokens/检查点），SSD 状态成为槽的活状态。写入方向，槽被替换前 `disk_leaving` 先把 SSD 缺的补上（见上），所以 SSD 是「被逐出的对话」的归宿；正在排队还没写完的拷贝，后续请求若续同一对话可以等它（`disk_lookup` 里对 `unwritten` 队列的等待权衡）。

## 2. 会话槽的保留语义（`src/engine/engine.h` / `engine.cpp`）

先澄清对象：`ExecutionSession`（`src/engine/execution_session.h`）只是单发请求的控制包装——一次性 `start`、幂等 `cancel`、可重复 `wait`，自等待/自启动抛异常，析构先 cancel 再 join（`execution_session.cpp`）；它不持有 KV 与检查点。保留语义在 `Engine::Slot`（`engine.h:376`）：每槽一个 `Seq`（KV 缓存，随对话增长）、DFlash2 时一个起草器上下文环 `DFlashSeq`、`tokens`（活状态已含的行；图像行以内容派生键记入）、图像 span+digest、检查点数组 `ckpts`、SSD 记账（`fills`/`version`/`disk_version`/`live_ok`）与调度字段（`last_used`/`turns`/`busy`/`arrival`/`mem_wait`）。槽内固定放三个检查点（`engine.cpp:484`）：两个轮次边界检查点加一个前缀检查点（`NI_PREFIX_CKPT=0` 关；带 LiLiCorr/KFlash 起草器时禁用——其上下文无法倒回空环，`engine.cpp:472`），每个非前缀检查点配一份当时的起草器环快照。

### 复用起点的三选一（`chat()`，`engine.cpp:2432`）

1. 提示严格延伸槽的 `tokens`（`common == tokens.size()`）→ 从 `start = common` 续活状态。
2. 否则找公共前缀内最深的可用检查点 `model_->restore`；DFlash2 环要么倒回（`rewind`，保留复用前缀仍在窗口内的条目）要么按检查点环快照拷贝；前缀检查点没有环快照，恢复后环从空开始。
3. 新对话拿空槽时可跨槽借前缀检查点：别的空闲槽若有 `prefix` 检查点落在公共前缀内且 ≥ 64 行、比本槽能恢复的更深，就把它的 KV 行 `[0, pos)` 拷过来并恢复其递归状态，存为本槽自己的前缀检查点（`NI_PREFIX_COPY`，`engine.cpp:2443`）。选槽时这个共享前缀本身不构成抢槽理由，只在本来就能选的槽之间平票用（`pick_slot` 的 `has_prefix`，`engine.cpp:2090` 注释：「共享系统提示的检查点不是从对话手里抢槽的理由」）。

恢复后所有 `pos > start` 的检查点作废（`engine.cpp:2492`）。`res.cached_tokens = start` 即命中量。

### 检查点在何处保存、谁被覆盖

- 轮次边界 `boundary`：本请求渲染出的「到此为止的对话」末尾（生成提示以 `<|im_start|>` 起，边界从一次编码顺带得出；`engine.cpp:2190`）。存这里的理由写在注释里：下一轮会把本轮回答的推理剥掉重渲染，所以活状态从上一条 user query 之后就对不上了——只有 query 处的检查点可恢复。
- 前缀边界 `pbound`：系统块末尾（第一个 user turn 起点），要求 ≥ 64 行且是提示的真前缀（`engine.cpp:2223`）。它在读提示经过时保存（`p == pbound` 时 `model_->save`，`engine.cpp:2843`），且从不被轮次边界覆盖（`victim()` 只在非前缀检查点里挑，`engine.cpp:2571`）。
- 轮次边界覆盖谁（`victim()`）：先无效的（`pos < 0`），再 tool response 处存的，再最旧的；`NI_QUERY_CKPT=0`（默认开）时退回「一律最旧」。注释给出动机：tool 循环每步都存检查点，「最旧先覆」会把 query 检查点挤掉，导致下一条 query 重预填整段对话；所以保「最新 query 检查点 + 最新一个检查点」（`engine.cpp:2552`）。
- 请求结束时 `slot.live_ok = (finish == "tool_calls")`（`engine.cpp:3820`）：tool 循环的下一请求按原样续活状态；最终回答后活状态与下一请求的渲染不符（推理被剥），要靠空闲期 history prefill 把渲染后的历史重预填进去，之后 `live_ok = true`、后续直接续（`prefill_history` 尾部，`engine.cpp:4153`；等待 `NI_HISTORY_IDLE_MS` 250 ms 的空闲窗、至多 `NI_HISTORY_WAIT_MS` 5000 ms，见 `engine.h` 注释）。`draft()` 读用户的打字前缀同一套路：先存用户轮起点的检查点，再延活状态（`engine.cpp:4000`，`live_ok` 随之置真；draft 换重时 `engine.cpp:4356` 置回假）。

`drop_slot`（`engine.cpp:1139`）是槽的彻底清空：KV 丢弃、递归状态归零、tokens/图像记录清空、检查点全部 `pos = -1`、`live_ok = false`、`disk_version = ++version`（没有可存的了）。

### 槽被谁逐出（`pick_slot`，`engine.cpp:2076`）

按「本槽实际可用复用深度」选可复用槽；没有 ≥ 64 行的复用时按新对话处理，逐出优先级：空槽 → 一次性请求的槽（`turns ≤ 1`、其请求在本对话请求之后到达、长度 ≤ max(1024, 4×本提示)）→ 续聊中的对话槽，同级内 LRU。即一次性请求（标题、摘要类）永远不挤正在进行的对话；更长的「首轮被短请求插队」的对话也不算一次性。图像请求的前缀比较按 `vision::common_prefix`：前缀里每张图必须同 digest 同 span（`vision/encoder.h:139`）。

（递归状态本身的家族差异——Dory 按 loop 计数作废 KV、Flint 无起草器等——略过，只记它们在通用机制上开的口子：`dory_` 时槽 KV 带 loop 计数、SSD entry 的 `tag` 承载它。）

## 3. KFlash 预算账（`src/model/kflash_budget.h` / `.cpp`）

### 标定量：测的是什么

两个量，都是运行中实测：

- **成本**：一个投机周期的 GPU 时间，毫秒，= 起草半拍的命令缓冲 `gpu_ms()` + 校验拍的 `gpu_ms()`（`engine.cpp:3164` 的 `kf_budget_->observe(nh_->tree_buf(), draft->gpu_ms() + e2->gpu_ms())`；`gpu_ms()` 是 Metal 命令缓冲 `GPUEndTime − GPUStartTime`，`src/metal/gpu.mm:980`）。头注释强调「whichever tool runs the cycles (ni_run, the server)」——谁跑都写同一个缓存；缓存文件里带 `kCostOf = "cycle GPU ms"` 标签，读到用别的方式（只测校验拍、宿主时钟）测出的表就作废重测，标定保留（`kflash_budget.cpp:14`）。
- **接受率**：逐深度 d 的校准 p = sigmoid(αd + βd·logit q)，q 是路径模型（起草器）给节点的概率；由在线 logistic 回归（Newton 步，每攒 16 周期的记录拟合一次，`refit_tick`）从接受记录拟合：每个「父节点被接受才轮到校验」的节点及其接受与否（头注释与 `add_record`；记录环形缓冲每深度 2048 条）。没有任何记录前 α=0、β=1，即 p = q。合批贪心链的节点同样喂给这个标定（`KfBudget::records`，「a chain's node is a tree's node of the same lattice」）。

### 成本表怎么记

`KfCosts`：每个行数 n 一条「基线 + 偏移」。基线由本进程第一个周期设定（有缓存表时由缓存里已有的某个行数先跑一次来设，`force()` 保证）；此后每周期把「实测 − 表值」误差的一半挪给基线（上下文变长抬高所有行数的成本——头注释举例：35B 在 21k token 比在 100 token 每周期贵约 7 ms，与 n 无关）、偏移按最近约 32 次的均值步进（`KfCosts::observe`，`kflash_budget.cpp:53`）。宽校验（16/32 行一拍）用对数比：log cost(n) = log 基线 + 每行数自己的对数比，在线拟合。探索：每个允许行数先跑 `kExplore = 3` 次（宽校验 2 次）表才做主；每 `kRefresh = 2048` 周期强制重测「最久没测的行数」（`KfBudget::observe` 的 refresh 分支）。允许哪些行数由后端决定（dense Qwen 只跑整 8 行 tile：8|16|32；MoE 每行都要拉 expert，任意行数——头注释）。

### 行数决策

`write(KfParams&)`（每周期起草拍前调用）：取允许 n 中最大化 E(n) − rate·cost[n] 者，E(n) = 1 + 前 n−1 个节点校准路径概率之和（一次 n 行校验的期望产出 token），rate 是最近周期的 tokens/ms（衰减和，每周期 ×63/64；不足 16 周期用 E(n)/cost[n]，且基线项相消）。rate 也写进 `KfParams` 给 GPU 树构建器。期望对实际的核对记账在 `observe`（`epred_`/`etok_`，summary 输出）。

### 缓存文件

`$KAPITSA_HOME/kflash/<key>.json`，key = 模型目录名-起草器名-GPU 名（`engine.cpp:432`）。落盘时机：析构（`save()`）、本处观测满 32 周期且有新周期时、引擎在请求之间每 `kSaveEvery = 256` 周期存一次；服务器收到停止信号调 `save_kflash()`（可在别的线程、请求还在跑时调用）。整文件临时写 + rename 原子替换（`write_whole`）。

### KfBatch（同一预算搬到合批校验）

`KfBatch`：合批周期（成员起草拍 + 一次多行校验）的 GPU ms 按（成员数 M，总行数 N）记成本（每 M 一套 `KfCosts`，每 N 探索 3 次、每 64 周期重测最旧）。成员草案 p 的存活率 = ∏j≤p sigmoid(αd + βd·logit cj)（贪心成员用起草器自己的 KfBudget 标定；采样成员走精确投机采样，事件不同，用本预算自己的一套标定）。分配规则 `allocate()`：每个成员先保 anchor（和 `floor` 个草稿行），然后按存活率从高到低分发行，直到某行的期望 < rate × (cost(N+1) − cost(N))；凸成本下这是全体最优分配。常量：最多 4 成员、每周期共 8 行、每成员 7 草稿。语法掩码下的成员接受不作标定记录，但 token 与行数照计入 rate 与成本。

## 4. 内存预算与压力释放次序

### 预算本身

`default_mem_budget()` = Metal `recommendedMaxWorkingSetSize`（macOS 按内存定的推荐工作集）减 4%（至少 512 MB）余量，留给「Neural Engine 服务的缓冲与其他 GPU 用户」；`NI_MEM_BUDGET`/`--mem-budget` 覆盖（`engine.h:95`）。槽的 KV 余量 = 预算 − 已分配（模型、缓冲、各槽状态），`context_fit_` 据此算单对话能装多少位置，并在图像编码器加载后随其权重重算（`engine.cpp:497`、`2377`）。

### `reserve_kv`（KV 增长路径，`engine.cpp:1157`）的释放次序

1. 直接 `model_->reserve` 试；
2. `vision_->release(false)`——**早先请求缓存的图像 embedding**；
3. **空闲槽的 KV**，LRU 先删；删前若本请求尚未开始预填则先 `disk_leaving(*o, 0)` 落 SSD（解码循环之间不写）；
4. **等在 prefill 前的请求的 KV**（`MemWait::before_prefill`），最新到达的先删；
5. `model_->ane_release()`——**ANE split 的面**与重跑检查点 `ane_ck_`（此后 prefill 全在 GPU 上跑）；
6. `vision_->release(true)`——**图像编码器权重**（下一张图再加载）。

此外，请求自己的槽若对提示毫无复用（无活状态可续、公共前缀内无检查点），先弃自己的再动别人的（`engine.cpp:2398`）。`reserve_kv_wait` 在以上不够时按轮事件等待别的请求结束或弃缓存，等待者标 `mem_wait` 让后来者可见；prefill 前的等待者只在「独跑也放不下」时才失败。

### `make_room`（图像 embedding 的暂存申请，`engine.cpp:1743`）的次序

空闲槽 KV（`disk_leaving` + `drop_slot`）→ ANE 释放。不动等待中的请求，也不卸编码器权重（此刻正要用）。`Vision::embeddings` 每次分配把 `make_room` 回调传进去（`engine.cpp:2370`）。

### 图像缓存（`src/vision/encoder.h:86` 的 `Vision`）

两级：按内容 digest 的 embedding 缓存（`NI_IMAGE_CACHE_MB` 缺省 256 MB，LRU，飞行中的请求 pin 住自己那份，`end_request` 归还）与编码器权重本身（首图才加载，纯文本永不加载）。`release(false)` 只清缓存条目；`release(true)` 连权重卸载。另一级是数据 URL 图像的准备结果（SHA-256 记忆，重发的截图不再解码）。

## 与本仓两条负载的对应（事实，不判断）

- **35B int4 + DSpark 单流**：对上的账是 `kflash_budget` 的标定与决策——每周期 GPU ms（两命令缓冲之和）、逐深度接受标定（在线 logistic 回归，先验 p=q）、E(n) − rate·cost[n] 选行数、`<模型-起草器-芯片>.json` 跨进程/跨重启共享成本表。Kapitsa 的行数选择在单流上是「整块或从表里挑」；本仓现在的 DSpark 块长 15 固定、只整块或前 2。
- **resin 合批（双请求 + 前缀命中）**：对上的账是槽的保留语义与两级前缀——双请求即两槽各自活状态；前缀命中走「活状态 / 公共前缀内检查点 / 跨槽前缀检查点拷贝」三选一，`cached_tokens` 即命中口径；`KfBatch` 是合批校验时按（成员数，行数）测成本、按存活率分配行的预算版本；SSD 前缀缓存把「被逐出的对话」按块存续、跨重启按 token 前缀 + 恢复点深度命中，与内存命中是「取更深者」的关系。

## 材料

- `nace-ai/kapitsa` @ `d2d149543b4b4e8624e25eca4c6f233a4a20a582`（Apache-2.0），克隆于 `/tmp/kapitsa-state`。
- SSD 前缀缓存：`src/engine/prefix_cache.h`、`src/engine/prefix_cache.cpp`（store、键、限额、LRU、损坏处理）；`src/engine/engine_disk.cpp`（何时写、`disk_plan`、`disk_leaving`、`disk_save_slot`、`disk_loop`、`save_disk_cache`、`disk_lookup`、`disk_restore`）；`src/engine/disk_saved.h`、`src/engine/disk_wait.h`（#467 落盘判定与等待权衡）。
- 会话槽：`src/engine/engine.h`（`Slot`、`TurnCheckpoint`、`TurnLock`、`make_room`/`reserve_kv` 注释）、`src/engine/engine.cpp`（构造槽与 `context_fit_` :479-503、`drop_slot` :1139、`reserve_kv` :1157、`make_room` :1743、`pick_slot` :2076、`chat()` 前缀复用与检查点保存 :2331-2844、请求收尾 `live_ok` :3820、`prefill_history` :4153、`draft` :4000/:4356）；`src/engine/execution_session.h`、`execution_session.cpp`（单发控制包装，非保留语义载体）；`src/vision/encoder.h`（`Vision` 两级图像缓存、`release(bool)`、`common_prefix`）。
- 预算：`src/model/kflash_budget.h`、`src/model/kflash_budget.cpp`（`KfCosts`、`KfBudget::observe/write/save`、`KfBatch::allocate`、缓存文件与 `kCostOf` 标签）；观测点 `src/engine/engine.cpp:3164`（`observe(tree, draft->gpu_ms() + e2->gpu_ms())`）、:1608（KfBatch）、:3131（`write(KfParams)`）；`Encoder::gpu_ms()` 定义 `src/metal/gpu.h:165`、`src/metal/gpu.mm:980`；`$KAPITSA_HOME` 定义 `src/hub/hub.h:87`、`src/hub/resolve.cpp:75`。
