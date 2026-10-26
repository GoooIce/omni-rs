# 验证记录

记录日期：2026-10-15（历史基线保留原日期）。设备：Apple M5 Max，40 核 GPU，128 GB RAM，macOS 27.0.1。
工具链：rustc/cargo 1.99.0，Burn 0.21.0；依赖版本保存在 Cargo.lock。

## 2026-10-23：DSpark 投机解码首次反超普通解码（分组 MoE 内核）

`crates/minicpm-v` 的 DSpark 路径从“比普通贪心慢 4.6 倍”改到“快 1.5–1.9 倍”。
提示「用几句话说明分页 KV 缓存是什么。」，温度 0，同一检查点（int4 目标 + BF16 草稿）。
本机 GPU 时钟随热状态波动（普通解码同长度两场可差 134→235 ms/token），所以以
**同场交错对比**为准；`MINICPM_DSPARK_TRACE=1` 可复测分相。

| 对照场次 | 普通贪心 | DSpark | 加速比 |
| --- | --- | --- | --- |
| 向量内核后 200 token 交错（普/DS/普） | 188 / 196 ms/token | 85 ms/token（11.8 tok/s） | 2.21× / 2.30× |
| GPU argmax 后慢状态场交错（普/DS/普） | 149 / 139 ms/token | 83 ms/token（12.0 tok/s） | 1.79× / 1.68× |
| 收尾快状态场（普/DS 交错） | 134 / 141 ms/token（历史快基线） | 78–86 ms/token（最高 12.8 tok/s） | 保守 ≥1.72×；深度节流场高达 3.8–4.9×（基线 329/422 ms/token） |
| 向量内核后 200 token 单场 | — | 80 ms/token（12.5 tok/s） | — |
| 200 token 交错三连（普/DS/普） | 134 / 141 ms/token | 86 ms/token（11.6 tok/s） | 1.56× / 1.64× |
| 300 token 第一场 | 235 ms/token | 126 ms/token | 1.87× |
| 300 token 复测 | 166 ms/token | 103 ms/token | 1.61× |
| 200 token 早期场 | 169 ms/token | 113 ms/token | 1.50× |
| DSpark 改前 | — | 381 ms/token（0.44×） | — |

改前每周期 746 ms（propose 148 / verify 582 / accept 8），τ=1.96；向量内核后
200-token 周期约 153 ms（propose 17 / verify 120 / accept 13 / retain 2），
τ≈2.0。DSpark 绝对吞吐从改前 2.6 tok/s 升到 11.8–12.5 tok/s。

### 改动

1. **草稿增量上下文**（`dspark.rs` `DraftState`）：fc/hidden-norm/K/V 按接受 token
   增量投影；K/V 驻留在预分配的 `[1, KV_HEADS, capacity, HEAD_DIM]` 缓冲——接受前缀
   与 propose 的块行都走 `slice_assign`，任何周期都不再拷贝上下文（此前每周期两次
   O(ctx) 的 `Tensor::cat`）。Markov 头 gather+argmax 全程驻留设备，一次读回 15 个
   提案。
2. **批量校验数值模式**（`model.rs` `VerifyNumerics`）：默认 `Batched`（多行 tile、
   批量注意力、prefill 式 MoE，即 vLLM 式整块打分）；`MatchingDecode` 保留逐位一致
   语义，`forward_verifying` 与 `verify_commit` 测试不变。CLI `--dspark-match-decode`
   可退回。
3. **分组 MoE 三内核**（`moe_dense.rs`，cube DSL）：每层 256 个专家在加载期拼接成
   三个原始 u32-words+f32-scales 线性缓冲（`quant.rs` 的 v/h concat 加载器），校验块
   的 MoE 由 gate+up / down / combine 三次启动完成——路由 softmax/topk 全在设备端
   （40 次/层的路由回读同步清零），未触达专家的工作组在读权重前早退。内核内层
   “每 word 单次加载、m 为 comptime 展开的 16 个 Array 累加器”：nibble 解包与 token
   无关，解一次喂全部 token（解包量从 16× 降 1×）。校验前向 582→约 180 ms。
4. **接受相整块 GPU argmax**（`dspark.rs`）：贪心且无惩罚时（DSpark CLI 限定温度 0）
   整块 `[16, vocab]` 在设备上 argmax 一次读回 16 个 id，替代每位置一次 1 MB 行读回
   加主机采样；惩罚路径保留逐行主机采样。
5. **verify 块的向量内核**（`moe_gate_up_vec`/`moe_down_vec`）：校验块恒为 16 token，
   该形状走 `Tensor<Vector<f32, 8>>` 专用内核——x 转置 `[hidden, 16]` 使每列 16 值为
   两个向量读，累加器是两个 8-lane 向量，解包出的权重经 `Vector::cast_from` splat
   后做向量 FMA（标量版每权重 16 次 FMA）；act 用转置 `[E*inter, 16]` 布局衔接
   K1→K2。verify 176→120 ms。标量内核保留给 prefill（m≠16）。后续四个受控实验
   均为中性或负结果并已记录：纯转置不配向量（更慢）；K1 的 k 维 4 路拆分加两
   阶段归约（持平）；K1 双累加器指令级并行（单链 119 对双链 136 ms，寄存器压
   力反超）；**权重 k 主序重排以合并相邻线程的内存读取（A/B 同机同态：行主序
   156–160 ms、k 主序 158 ms——完全持平，证明分组内核不是带宽/合并受限，瓶颈
   在每层的内核发射与主机编排）**。k 主序版保留（设计正确、测试等价）。

### 坑（都已写成测试或注释；常驻清单在 docs/cube-dsl-pitfalls.md，本节留证据）

- cube DLS 的 `if cond { a } else { b }` 不能当值用（求值成 0）——nibble 补码必须
  branchless（`raw - i32::cast_from(raw >= 8) * 16`，同 cubek dequantize 的写法）。
  该 bug 曾让全部专家权重带上 +8×scale 的系统偏移。
- burn 的 Int 张量默认 dtype 是 I64：`Tensor::<B,1,Int>::from_data` 需显式
  `TensorCreationOptions { dtype: Some(DType::I32) }`，否则 i32 内核读进交错垃圾。
- 大尺寸拼接 QFloat 张量过 packed matmul 静默错值（与 3-D batched QFloat 同病），
  故分组内核直读原始线性缓冲，CPU 测试证明两种 concat 与逐专家加载器逐元素一致。
- `TensorData` 的 f32 数据默认上成设备 f16：scales 用显式 F32 options 创建。
- cube 的 `#[unroll]` 循环变量类型必须可推断：字面量范围需要后缀（`0..8u32`）。

### 测试

`verify_commit`（matching 模式逐位一致）、`greedy_parity`、`metal_spot` 全绿；
lib 单元测试 59/59（`moe_dense` 分组内核 9/9，含 16-token 向量路径与 256×512×2048
的 35B 层尺寸、f64
参考加“平方权重必须可分辨”的敏感性断言；`quant` concat 加载器 CPU 对照；既有 int4
测试）。CPU 构建（`--no-default-features --features native,cpu`）编译通过（分组路径
仅 Metal，其余后端回落原路径）。

### 与 vLLM 的对照

vLLM 的 DSpark（GPU）对自身普通解码的加速通常 1.5–3×；本机（M5 Max/Metal，无
vLLM 部署）以同场普通贪心为基线测得 1.6–2.3×（GPU 高频态 2.2–2.3×，低频态
1.7–1.8×；DSpark 绝对值稳定在 80–86 ms/token，波动主要来自普通解码基线），
落在该区间的中段。绝对
tok/s 对照需要用户提供其 vLLM 部署（同模型 int4、同提示、
`num_speculative_tokens 15`）的实测值。verify 内 10 层全注意力随上下文线性增长
属模型固有；接受相 13 ms 为 GPU 队列排空（与 verify 部分重叠）。

## 2026-10-23（晚）：用户提供本地参照与新靶值 100 tok/s

用户本机（同机 M5 Max）LM Studio/MLX 实测：MiniCPM-V-4.7-35B-A3B **BF16 全精度
+ DSpark draft** 挂载，think=on，883 token 生成，**稳态 50 tok/s**（首 token
26.14 s、预填充 1 tok/s）。BF16 每激活参数 2 字节 → ~6 GB/token → 50 tok/s 即
~300 GB/s 有效带宽（M5 Max ~400 GB/s 的 75%），证明该硬件上 MLX 级实现的上限。
用户新靶值：**100 tok/s**。我方现状 12.0–12.8 tok/s（int4，~2 GB/token，有效
~27 GB/s）。差距根源经本轮合并读取实验排除带宽解释后，定位在每层内核发射次数
（40 层 × ~25 启动/层 ≈ 千次/周期）与主机编排间隙——对齐 100 tok/s 需要内核融
合与发射批量化级别的重构（多会话工程）。

## 2026-10-23（夜）：MoE 单内核融合（五连发 → 每层 5 发）与两个瓶颈假设的证伪

按「融合 MoE 五连发作为单内核」的计划，把每层 MoE 的设备端算子从 ~19 次发射
（router matmul + topk + min_dim + lower + mask_fill + softmax + gate/up + down +
combine + 共享专家 3×matmul + silu/mul + 门控 matmul + sigmoid + mul + add，再计
cast/transpose）压到 **5 次**（`moe_dense.rs`）：

1. `moe_router`：一 cube 一 token、一 unit 一专家。unit 算各自 logit，
   `SharedMemory` + `sync_cube` 共享后，每个 unit 冗余推导同一 top-k（严格 `>`
   保最低索引，与 `route()` 稳定排序同规约），renormalized softmax 权重双布局输出
   （`[m,E]` 给 gate/up 触达检查、`[E,16]` 给向量 down 的 lane 权重）；unit 0 顺带
   算共享专家 sigmoid 门。替换 6 连 burn 链（topk 内含排序）。
2. `moe_gate_up[_vec]`：不变（向量版 act 写回转置 `[E*inter,16]`）。
3. `moe_shared_gate_up`：共享专家 SwiGLU，f32 权重（bf16 加载点无损上转）。
4. `moe_down_vec_split` + `moe_reduce_shared`（m=16）：down 向量化保持（一个解包
   权重喂 16 token 的 lane FMA），专家分 ≤16 块并行、路由权重以 `[E,16]` 向量乘入，
   partial 由 reduce 内核求和并叠加门控共享专家——combine、共享 down、门控乘、最终
   加法全部折入。m≠16 走标量 `moe_down_total`（每 (m,h) 线程自带 combine+共享）。

`DenseExperts` 扩展持有 router/共享专家 f32 视图；`forward_dense` 瘦身为一次
`try_dense_moe`（+cast）。测试拆出 `try_dense_experts`（显式权重/门控驱动专家段，
与 router 的 f32 点积噪声隔离）与 `try_dense_router`（router 单测）。

### 本轮坑（均已并入 docs/cube-dsl-pitfalls.md；此处留证据）

- **cube 宏不接受 let 初始化器里的负浮点字面量与字面量间算术**（报
  `f32: From<NativeExpand<f32>>`，`-1.0f32`/`0.0f32 - 1.0`/`f32::NEG_INFINITY`
  全灭）：两行式 `let mut x = 0.0f32; x = x - 3.4e38;` 可用；语句位置的 `-v`/e 记数
  正常。
- **`shared::Linear::weight()` 返回可执行视图 `[input, output]`**（加载时已转置）：
  内核按 checkpoint 式 `[output, input]` 行索引，构造 `DenseExperts` 时必须再
  `.transpose().cast(F32)`。此 bug 曾让真权重 seam 相对 RMSE 0.28（合成测试自建
  张量测不出）；修复后 **real-35B layer0 seam RMSE 0.0147**。
- **中途 `into_data` 读回会污染后续内核**（既有已知坑）：诊断用的多读回 bisect 测试
  给出「共享专家输出恒 0」的假象；以单次前向的 seam 测试为准。
- SharedMemory 需 `let mut`；`sync_cube` 前 diverge（`terminate!`）会挂起。

### 性能（同态交错 A/B，propose 相为热态指示器，200 token）

| 版本 | propose | verify | 备注 |
| --- | --- | --- | --- |
| 旧（3 发 + burn router/共享链） | 21.2/22.1/22.0 | 135.7/136.7/141.8 | 基线 |
| 新（融合，标量 down_total） | 23.3/19.9/21.5 | 144.9/120.6/127.2 | τ 1.942（f32 路由，略升） |
| 新（融合，向量 down_split） | 23.0/23.4/22.2 | 137.7/144.5/138.2 | τ 1.905 |

**初步结论（当晚）**：发射数 -600/verify（~19→5/层）只换来 verify ~5–10%；down
标量/向量分解持平——当时记为「发射间隙假设证伪」。**次日逐内核计数证明该结论是
抽样偏差**：-600 发只占 verify 总发射的 ~10%，-8% 墙钟恰是线性响应；整体
verify 仍是发射受限（见下节构成表，140.8 ms ÷ 5829 发 ≈ 24 µs/发）。

### 测试

lib 60/60（分组内核 10：显式权重专家段 × 9 + router 模型规模对照；router 权重对照
容差 1e-3 = f32+FMA 点积噪声，选择集/零元/门控严格断言）；`verify_commit`、
`greedy_parity`、`metal_spot`、CPU 构建、fmt 全绿。

### 2026-10-24：verify 140 ms 的确切构成（逐内核计数）

新诊断 harness `crates/minicpm-v/tests/verify_profile.rs`（`#[ignore]`，跑法与
聚合命令在其文件头）：只跑 Batched 校验前向（16 token、40 层、上下文 ~30，
verify 后 retain 回滚保持上下文恒定），配合
`CUBECL_DEBUG_OPTION=profile CUBECL_DEBUG_LOG=1` 抓 cubecl 逐内核日志。无
profiling 时 **140.8 ms/前向**，与 trace 的 verify 相完全吻合（harness 有代表性）。
跑两遍取第二遍（第一遍付 autotune/编译）。

**每前向 5829 次内核发射，140.8 ms ÷ 5829 ≈ 24 µs/发——verify 整体发射受限，
GPU 从未跑满。** 按发射数（精确）与 System 计时口径时长（含每发同步、为上界）：

| 构成 | 发射/前向 | 占比(发) | ms/前向(上界) | 占比(时) |
| --- | --- | --- | --- | --- |
| burn 粘合算子（Binop/Scalar/Unary/Cast/Slice/SliceAssign/Reduce/Copy/Mask 等） | ~5244 | 90% | ~352.8 | 57% |
| 融合后 MoE（router+gate_up+shared+down+reduce） | 164 | 2.8% | ~101.3 | 16% |
| MatmulEntry（qkv/o/a/b 等投影） | 250 | 4.3% | 45.2 | 7% |
| GdnScanKernel（30 层线性注意力扫描） | 31 | 0.5% | 97.4 | 16% |
| FullKernel（10 层全注意力） | 96 | 1.6% | 8.0 | 1% |

单层发射序列点名（每层 ~145 发）：MoE 段已是 7 发（融合生效）；其余 ~60 发为
Norm 链（Reduce+ScalarBinop+Unary+Mul+Cast 各 7-9 发 ×2 norm）、gdn 卷积状态维护
（Slice×8 + SliceAssign×12）、投影间 Cast/Binop。粘合算子风暴的两条出路：
**burn-fusion 流式融合**（metal 构建已编译入 burn-cubecl 的 fusion 特性但日志显示
这些链未被融合——需要核实流式上下文为何不吞并 Cast f16↔f32 断链）与**手工融合
每层粘合**（rmsnorm+残差+cast 一发化、gdn 状态 SliceAssign 风暴一发化）。
GdnScanKernel 是最大的单内核线目（30 发、System 计时 ~3.1 ms/发，recurrence 对
token 天然串行）——若其实机时长接近上界，单它就占 verify 一半，需读
`crates/minicpm-v/src/gdn_scan.rs` 的并行化空间。

## 2026-10-24（午后）：burn-fusion 实验全程——确认为死路并回退

按构成表的第 1 杠杆验证 burn-fusion 流式融合。探针（`tests/fusion_probe.rs`，
保留）显示合成 norm 链 409→71 发（-83%）——收益方向成立；但全模型接入后为
**决定性负结果**，已完整回退（131 ms/前向复原，全部测试绿）：

- **接入**：metal 特性补 `burn/fusion`（爆炸半径仅 moe_dense 7 处 + gdn_scan 2 处）；
  新桥接层 `resolve`（排空流保序）/`register_external_tensor`+`Init` op 回流。
  途中修真 bug：router 输出张量曾以一维 `[values]` 形状冒充 rank-2（裸后端不查、
  融合 trace 分析崩）——已按真实 2-D 形状分配（`alloc_f32_2d`，回退后保留）。
- **负结果一**：Reduce/ReduceBroadcasted fuser 的 trace codegen 在本模型流上 panic
  （`GlobalArgsLaunch::resolve_arg` 越界 + ordering 二次崩）——上游 bug。
- **负结果二**：砍 reduce fuser后 **MatmulFuser 把 matmul 发射 250→6290/前向（25×）**。
- **负结果三**：只留 ElementWiseFuser 仍 **285 ms/前向（2× 回归）**：融合流的逐算子
  簿记（IR 构造 + 通道发送，~50µs/op × 5800 op）比省下的发射更贵；且真实流上
  Slice/Cast（2774+1496 发）根本不参与融合——探针的干净合成链会融合，交错流不会。
- **结论**：burn-fusion 0.21 在此工作负载上不可用（CPU 簿记主导 + 两个 fuser 各自
  崩/炸）。100 tok/s 的路径回到：粘合算子手工融合（norm+残差+cast 单内核、gdn 状态
  SliceAssign 风暴单内核）与 GdnScanKernel 优化。未来 burn 升级可用 fusion_probe
  一分钟重测（全 fuser、真实交错流两类都要测）。

## 2026-10-24（晚）：自研粘合融合（norm+残差单内核）——结构兑现、墙钟负结果

不待 burn 升级、自己实现 fusion 的第一刀：`crates/minicpm-v/src/glue.rs` 两个
cube 内核——`gemma_norm`（一行一 cube，SharedMemory 行平方和归约，f16 进出/f32
计算，Gemma `weight+1` + `NormReduction::Divide` 逐位对应 `Norm::forward`）与
`residual_add`（平铺 f32 相加回 f16）——替换四条层 seam 的 burn 链。

**结构精确兑现**：verify 发射 5829 → 4915（-914，与构成表预测一致；norm 链的
Reduce/Cast/Binop/Scalar/Unary 全部塌缩）。**但同二进制交错 A/B（`MINICPM_GLUE`
开关，消二进制/数值/热态配对噪声，冷态两对）显示 glue ON 稳定慢 ~4%**（verify
+8–11 ms、accept +12 ms/周期）：被替换的 10 个轻量 elementwise 在深队列里互相
流水，而融合内核把自身全延迟加进依赖链（GemmaNorm 53 µs/发、ResidualAdd
67 µs/发，System 口径）。**修正认知：24 µs/发的整体均值只对重内核成立**
（MoE A/B 的 -600 发是读几百 MB 的内核）；轻量 elementwise 的边际发射成本远低
——「verify 发射受限」的结论被本实验直接证伪，verify 实为 GdnScan 延迟链主导。

处置：内核与接缝保留、**默认关闭**（`MINICPM_GLUE=1` 选用；lib 单测内
cfg!(test) 强制开启以保覆盖）， seam/lib 61/verify_commit/greedy_parity/
metal_spot 全绿。下一杠杆回到 **GdnScanKernel 本体**（30 发、System 计时
~97–105 ms/前向，真·大设备时间）与 accept 相读回（13–45 ms/周期）。

## 2026-10-24（夜二）：GdnScan 两刀——主延迟链 -75%

对构成表定位的最大单内核 GdnScanKernel（30 层线性注意力扫描，System 计时
97–105 ms/前向）连砍两刀，均为 `gdn_scan.rs` 内的小改动：

1. **并行度**：发射从每 cube 32 线程提到 `v_dim`（128）线程——原先 v 维被
   chunk 循环串成 4 轮，全内核只有 heads×32=1024 线程在跑。内核本身已按
   `CUBE_DIM_X` 参数化，改一行发射参数，chunk 循环自动退化。
   **3.1–3.4 → 1.29 ms/发（-60%），97–105 → 40.2 ms/前向。**
2. **访存**：kv 归约循环原先边读边写衰减态（每元素一次 store 且第二循环依赖
   这些写）；改为只读归约 + 单写更新循环，数学不变。
   **40.2 → 25.1 ms/前向（807 µs/发）。两刀合计 -75%。**

3. **kk 维四分**（`gdn_scan_split_kernel`）：每 v 列由 4 个 unit 分摊 k 维归约，
   部分和走 cube address space、各 unit 冗余求和（省一次屏障），每 token 仅
   两次 sync。串行链 256→64 迭代，线程 4096→16384。**807 → 208 µs/发（再
   -74%），25.1 → 6.7 ms/前向。三刀合计 3.1–3.4 ms → 208 µs/发（16×），
   97–105 → 6.7 ms/前向。**（坑：`SharedMemory::new` 的字面量尺寸需要
   `usize` 后缀，否则宏报 `usize: From<i32>`。）

端到端（半冷态）：verify 119 ms、accept 4.9 ms、整链 **75.5 ms/token
（13.3 tok/s）**；verify 内 GdnScan 占比从 ~70% 降到 ~6%。分裂版改变了 kv/y
的浮点累加顺序，贪心文本随之变化（τ 波动属预期），全部数值门禁仍绿。
注意本机在长时间连续重载后会进入暂态劣化（连 propose 都 ×2.5，空转几分钟
即恢复，`pmset` 无热告警）——性能测量如见全场劣化先空转几分钟再测；测试与
基准不要并行跑（抢 GPU 互相污染）。

测试：gdn 2/2、lib 61/61、seam、verify_commit、greedy_parity、metal_spot、
CPU 构建、fmt 全绿。

### MoeRouter 拆分实验（负结果，已回退）

按「每 unit 2048 串行点积是瓶颈」的假设做了 hidden 四分（SharedMemory 部分
和）+ 选择循环收敛到 unit 0 + taken 掩码三步。同态 profiled 对比判决：
拆分版 ≈ 原版（721 vs 26.3×劣化系数），掩码版反而更差（862 µs/发）。
**教训（已入坑文档）：并行的冗余工作不占墙钟**——原设计 256 unit 冗余推导
top-k 是并行抵消的，改成单 unit 串行没有省任何墙钟，反而把 18k 次比较压到
一条线程上。原设计（并行冗余选择 + 每 unit 独立点积）本就是均衡的。已完整
回退（与原版字节等价），回退后冷却态墙钟 **verify 102.3 ms、65.7 ms/token
（15.2 tok/s）——本日最佳**。

剩余构成（System 口径，扫描已降至 ~7ms）：Binop/Cast/Slice 风暴 ~200ms
（双证伪）、MoE 其余四内核 ~93ms（int4 读出为主）、Matmul 投影 45ms、
propose 相 19-21ms。下一刀候选：MoeGateUpVec/DownVecSplit 的 int4 读出
评估（BW 下限 ~20ms，System 70ms，有 2-3× 空间）或 propose 相剖析。

## 2026-10-24（深夜）：稀疏 token-expert gate_up——MAC 砍 13×、墙钟持平、代码大减

按「每个触达专家为全部 16 token 计算」的冗余假设（分散路由下每专家平均
~1.2 个 token）实施稀疏化：`moe_masks`（每专家一个 u32 token 位图，67µs/发）+
`moe_gate_up_sparse`（位图驱动，只投影路由到的 token）+ down 回归按 (m,h)
天然跳过的 `moe_down_total`。

**结果三分**：① down -8ms System（24.5→16.5，弃 16× m 向量浪费）；②
gate_up 持平（44.6→47.1ms）——**真正的瓶颈是每 word 的 int4 解包 ALU 与
权重读出带宽（各 ~15ms 下限），不是被砍掉的 FMA**；③ 墙钟热态折算持平
（~78-81 ms/tok 非冷态）。保留稀疏版：System 净 -11ms、去掉 transpose 发射、
派发简化为恒定 5 内核单路径（router → masks → gate_up_sparse →
shared_gate_up → down_total），并删除四个不再调度的旧内核
（moe_gate_up/vec、moe_down_vec_split、moe_reduce_shared）与 weights_t 双布局
管道（router 输出、分配、转置全链）。

测试：grouped 10/10、router 1、seam、lib 61/61、verify_commit、greedy_parity、
metal_spot、CPU 构建、fmt 全绿。稀疏化改变近平局 argmax（贪心文本变化，预期）。

剩余 verify（System 口径）：Matmul 投影 44、gate_up_sparse 47（解包/带宽下限
~15+15，理论余量 ~2×）、down_total 16.5、router 21、共享 15——下一刀候选：
gate_up 解包向量化（nibble→f32 用向量查表或位技巧减 ALU）或 propose 相
（21ms）剖析。

## 2026-10-24（夜三）：解包 ALU 与 router 扫描两刀落地；逐内核 System 归因证伪

两刀落地 + 一个方法论反转，全部门绿（grouped 10、router、lib 61/61、
verify_commit、greedy_parity、metal_spot、CPU 检查、doctor、fmt）。

### 方法论反转：CUBECL 逐内核 System 归因不可信

复测 verify profile 时发现昨日与今日的逐内核 System 数字互相矛盾（未改动过的
`moe_shared_gate_up` 15→5.3ms、`moe_down_total` 16.5→4.2ms——后者低于其
~15ms 带宽下限，物理不可能）。机理：批量异步提交下，profiler 把**排队排水
时间记到同步落点所在的内核**，重内核（gate_up）吸收了别人的时间，轻内核记 0。
结论：跨版本/跨场次比较逐内核 System 数字无效；launch 计数仍精确。

替代真相源：新增 `moe_bench_model_scale`（`#[ignore]` 隔离微基准，35B 层形状 +
分散路由 ~110/256 专家，每入口单独发射、尾部一次 sync，~1.4s/场）。可信读数
（同场）：router 270-296µs/发（dot 段 ~230µs 与 top_k 无关，扫描段 65-129µs）、
experts 四连发 1.05ms/层（与 ~320MB/层的**带宽下限吻合，已在地板上**）、
qkv 形状 F16 matmul [16,2048]×[2048,6144] 114-139µs、投影形状
[16,2048]×[2048,2048] 38-45µs。由此得诚实墙钟构成：MoE ~53ms（router 12 +
experts 42，带宽定死）+ 投影 matmul ~8-12ms（System 44ms 是虚的）+ gdn 6.7 +
attention 8 + 粘合发射风暴。模型级墙钟：**119.0 vs 118.6 ms/前向（12 分钟闲置
恢复态），持平**——router/matmul/gate_up 不在关键路径上（发射管线在其执行期间
已在提交后续内核），只有 GdnScan 类串行链是墙钟敏感的。

### 落地的两刀（内核做功确减，墙钟持平）

1. **int4 解包双算术移位 + 每词 scale 提升**（`moe_gate_up_sparse`/
   `moe_down_total`）：`word_bits` 从 mask+修正 6 op/nibble 改为
   `(w << (28-4n)) >> 28` 2 op 符号扩展（Metal 有符号 `>>` 为算术移位，grouped
   测试的 f64 参考含负 nibble 可证伪）；一个 word 的 8 个 nibble 同属一个 128
   宽 scale 组，scale 乘法从每权重提升到每词部分和一次。experts 四连发本已
   带宽封顶，故墙钟不可见，保留（严格减功）。
2. **router top-k 扫描 -inf 掩码**：选中槽写哨兵后下轮纯扫描，去掉 taken-check
   分支风暴（扫描段 129→65µs，整发 360→296µs 隔离口径）。**新坑**：共享
   memory 掩码必须在每轮 `扫描→sync→写→sync` 双屏障——单屏障下快 unit 的写
   与慢 unit 的同轮读竞争，各 unit 选出不同 top-k（router 测试
   "outside weights not zero" 捕获）；tie 规则与求和顺序不变，数值位级等价。

### 今日证伪（封存）

- **转置 router 读**（[hidden,experts] 合并访存）：+200µs/发——跨 unit 合并输给
  每 unit 顺序流式读（L1 行内 32 次复用）。
- **burn F32 细长 matmul 出 logits + topk 内核拆分**：332µs——F32 m=16 matmul
  与串行 dot 一样慢；F16 matmul 快但 logits 精度撑不住 router 测试 1e-3 容差
  且有 τ 风险。
- **router dot 4 路 k-split**（1024 线程）：dot 231→179µs 但 1024 线程的扫描
  屏障 +64µs，净变差。

下一刀候选重排：① 粘合发射风暴（唯一大池，CPU 发射侧；需源级 op 削减或图捕
获，burn-fusion/手写粘合已证伪）；② propose 相（19-21ms）剖析；③ BLOCK/τ 带
宽权衡研究（16-token verify 读 ~8GB MoE 权重、τ≈2 → 4GB/接受 token，是普通
解码 1.7GB 的 2.4×；小 BLOCK 或许在更低 τ 下净赚 tok/s——算法级，需用户知情）。

## 2026-10-25（凌晨）：自适应 draft block——两类内容同时逼近各自最优

### propose 剖析（插桩：draft/logits/markov/读回）

`MINICPM_DSPARK_TRACE=1` 下细分计时（BLOCK=15，63 周期均值）：draft 前向
8.68ms（**CPU 派发受限**：~320 个 burn op × ~27µs 派发费）、logits 1.86（带宽
地板附近）、markov 2.95 CPU + **~6.3ms GPU 串行**（15 行链每行全量重读
127MB 的 markov_w2）、读回 6.67ms（大半是 sync 排队排水）。结论：内核融合在
CPU 派发受限的 propose 相有效（与 GPU 受限的 verify 相相反——解释了粘合融合
在 verify 的负结果）。

### BLOCK/τ 扫描（固定值，200 token，同窗交错）

带宽账成立：verify 触达专家数随 m 近线性，τ 降得慢。**但最优 BLOCK 内容相
关**：

| BLOCK | 中文说明文 τ | tok/s | 代码生成 τ | tok/s |
| --- | --- | --- | --- | --- |
| 15（旧默认）| 2.254 | 13.4 | **7.692** | **52.5** |
| 11 | 2.254 | 14.8 | — | — |
| 7 | 2.029 | 15.7-18.3 | — | — |
| 4 | 1.972 | 17.1-20.3 | — | — |
| 3 | 1.945 | 24.1 | — | — |
| 2 | 1.844 | 24.1 | — | — |

（冷窗最佳值；热窗整体下移但排序不变。τ=7.7 时管线 52 tok/s 证明接收率配合
时硬件早已越过 MLX 50 tok/s 参照。）

### 自适应策略（三轮迭代后定稿）

`BLOCK_LADDER = [3, 8, 16]`，AIMD+滞回：连续 2 周期接收 ≥80% → 升一级；连
续 2 周期 ≤50% → 降一级；起点 3；`MINICPM_DSPARK_BLOCK=N`（1..=31）钉死固
定值做实验。迭代教训：① 块大小**连续变化会不断制造新 autotune shape key**，
每次转档触发数秒运行时调优——阶梯化后一次付清永久缓存；② 纯「两次全收」增
长在 acc≈5/6 的内容上几乎不触发（均衡点过低）；③ 单次低谷收缩会被方差打出
均衡点——收缩也要双连。

终测（同窗交错，冷窗）：中文说明文 **22.5-23.3 tok/s**（旧默认 13.4，
**+69%**）、代码生成 **50.7 tok/s**（固定 15 的 52.5 噪声内）。贪心输出与固
定 15 **逐字节一致**（verify 保护与 BLOCK 无关）。回归门：lib 61、
verify_commit、greedy_parity、metal_spot、CPU 检查、fmt 全绿。

### 后续杠杆（更新）

1. draft 前向融合（RMS norm/rope 单内核化，砍 ~5ms CPU 派发；verify 相无效
   但 propose 相有效）；
2. markov_w2 量化（int8/int4 副本：6.3ms GPU 串行 → 1.6-3.2ms；仅影响
   draft，τ 可测）；
3. 粘合发射风暴（verify 相，需源级 op 削减）；
4. 杂项：BLOCK=5（m=6）疑似 autotune 坏候选（verify 反常慢于 m=8），可删
   `target/autotune` 对应缓存重调。

## 2026-10-25（午）：draft 前向融合——draft 段 -37%、墙钟 -2%

`glue.rs` 重构为两套门控：verify seam 维持 `MINICPM_GLUE=1` 选配（GPU 受限相，
已证伪）；**draft seam 默认开启**（`MINICPM_DRAFT_GLUE=0` 为同二进制杀开关），
含 plain RMS norm（`rms_norm` 内核加 comptime `plus_one`，主模型 Gemma 路径不
变）、split-half rope 单内核（cos/sin 表全宽、前半为独立频率——索引坑）、
residual。dspark 的 propose 接 seam：input/post 终 norm（rank-3）、q/k norm
（project_kv 内 permute 后 reshape 一次拷贝进 (head,pos,dim) 序）、rope×2、
residual×2；~170 发/propose 削掉。

A/B（同二进制交错，两对一致，prose、block avg 3.3）：draft 段 **6.13→3.87ms
（-37%）**；propose 相 12.3→10.6-11.1；总墙钟 6261-6266→6133-6163ms（**-2.1%**，
23.1 tok/s）。增益小于 draft 段的原因：CPU 派发省下后 GPU 的 draft 计算成为
propose 新地板（读回排水不变）。τ 无变化（draft picks 未翻转），贪心文本与融
合前**逐字节一致**。glue 单测扩三链对照（plain RMS/rope/residual vs burn 链）
全绿；lib 61、verify_commit、metal_spot、CPU、fmt 全绿。

剩余杠杆：markov_w2 量化（≤4%）、verify 粘合风暴（大池难啃）、BLOCK=5 的
m=6 autotune 坏候选排查。prose 端 propose 已近 GPU 地板，下一步大头只能在
verify（60ms@block3）或 τ。

## 2026-10-25（晚）：#43 树状推测原型——精确合同成立、τ 无增益、墙钟负（原型口径）

`MINICPM_DSPARK_TREE=1`（默认关，自适应模式专属）落地完整树状推测原型：
- **propose**：markov 头逐层批量（每深度一次 gather + 一次 w2 matmul + 一次
  top-2，设备端链、末尾一次读回）——同节点数下比逐行链少读 w2（深度次 vs
  节点次）；draft 前向长度从块长缩到深度数。
- **verify**：BFS 展平；位置=深度；全注意力层一次 `module::attention` 走显式
  掩码（历史全可见 + 块内仅祖先）；gdn 层逐 token 批量投影 + 每条根→叶路径
  复用现有 causal_conv/try_scan（路径=从根态出发的合法线性序列，精确）；
  `KvCache::copy_selected` + `DecodeState::retain_path` 按路径压缩。
- **accept**：贪心步进器沿树走最长匹配路径 + 修正 token；AIMD 梯级语义映射
  为深度（3/8/16 → 1/2/3 层）。

**结果（prose 200-token ×2 场）**：文本与线性版**逐字节一致**（两场）——贪心
精确合同在树模式下成立；τ=1.972 与线性完全相同（**第二候选从未在首候选失
手处命中**——该内容下宽度零增益）；墙钟 17.8s vs 线性 6.1s：verify 171 vs
60ms，来自每路径 burn-op gdn（~220 发/层）与掩码注意力的**原型惩罚**（换
每深度批段扫描内核可消，预估 -100ms 级）。

**结论**：① 精确树 verify 在 gdn 混合架构上工程可行（祖先掩码 + 路径状态
回放，无需改内核）；② prose 场 τ 增益为零、code 场深度上限反而封顶——
**当前树形状（k=2 全二叉）不满足切换门槛**，默认保持关闭；③ 后续若重启，
前置条件：批段 gdn 内核（墙钟归位）+ 更宽/更深的树形与 τ(B) 曲线实测。
**新地雷**：burn `module::attention` 掩码约定 **true=排除**（与直觉相反；
首版即因此全错——历史被整段屏蔽）。已记 pitfalls。

测试：lib 61、verify_commit、metal_spot、CPU 检查、fmt 全绿；greedy_parity
（1B CPU 路径）不触及本改动。未提交 git 变更（沿用本仓库会话惯例）。

## 2026-10-25（夜）：#44 markov_w2 int8 量化——段 -36%、默认场景 ~0%、默认保持关

`MINICPM_DSPARK_MARKOV_QUANT=1`（默认关，加载期生效）：w2（[248144,256] f16
127MB）量化为 packed i32 words（每词 4 个 +128 偏置 int8，~64MB）+ per-row f32
scale；新 `markov.rs` 单内核一次算 `base + scale·Σq·w1[prev]`（prev 留设备作
id 张量，省 gather 发射），替代每行 gather+matmul+add 三连发。线性链与树模式
两处接入，CPU 后端走原路径。

**A/B（同二进制 env 交错，同窗 ≥2 场）**：
- prose（block≈3.3）：文本逐字节一致×4、τ=1.972 无漂移；markov 段本就
  0.2-0.3ms，总墙钟中性偏正（6271-6344 vs 6361-6474ms）。
- 代码自适应（block≈8.7）：markov 段 1.01-1.31→0.76-0.78（**-35%**），读回
  排水 -0.7~-1.9ms（GPU 节省实锤），总墙钟持平（draft 段 +0.6ms 噪声吸收）。
- 代码固定 BLOCK=16（最长链）：markov 段 2.15→1.38（**-36%**），总墙钟
  4174→4091-4111（-1.5~-2.0%），文本一致、τ 5.882 无漂移。

**判定：未达切换门槛**（-36% < -40%；自适应默认下总收益 ~0%——markov 仅
~2ms/105-125ms 周期，符合 checkpoint 预估的 ≤4% 上限）。剩余地板=串行链每行
~60µs 内核延迟（量化不可及，需树模式或批段内核消除）。默认保持关闭；数据留
档，若未来 BLOCK 上调或树模式重启可复用。

测试：markov 内核单测（vs CPU 镜像 worst<5e-2）、lib 62、verify_commit、
metal_spot、CPU 检查、fmt 全绿。未提交 git 变更。

## 2026-10-25（夜二）：#42 draft 上下文层集合扫描——默认集合全面最优（负结果）

`MINICPM_DSPARK_LAYERS`（5 个 0..39 升序层号，非法回退默认）落地；默认值下
200-token 输出与改动前逐字节一致。4 集合 × 2 提示 × 2 轮交错扫描（τ 两轮
逐集合完全一致，信号确定性极强）：

| 层集合 | prose τ | code τ |
| --- | --- | --- |
| **[3,11,19,27,35]（默认）** | **1.972** | **5.882** |
| [1,5,9,13,17] 前段 | 1.449（-27%）| 2.439（-59%）|
| [23,27,31,35,39] 后段 | 1.479（-25%）| 2.500（-58%）|
| [5,13,21,29,37] 偏移展开 | 1.919（-3%）| 4.762（-19%）|

**结论：默认集合已是训练匹配的最优，换层即大幅负优化**（draft 权重与
[3,11,19,27,35] 的隐状态分布强耦合）。保持默认；env 留作未来换 draft 检查点
时的实验通道。门全绿（lib 62、verify_commit、metal_spot、CPU、fmt）。

## 2026-10-26：#52 切片 1——verify 缝默认开（rope 缝新增）：verify -18~-21%

靶值重锚定（#41 grill 决议）后工程线第一刀。`verify_profile` 构成（block16，
缝关口径）：5829 发/前向，90% 为 burn 粘合逐算子。本刀三件事：

1. **现成 verify 缝（Gemma norm、residual）翻默认开**（`MINICPM_GLUE=0` 杀）。
   2026-10-24 那次「verify 融合更慢」的结论被 `verify_composition` 纠正为抽样
   偏差后未复测——本次同窗交错 A/B（代码提示词、BLOCK=15、200 token）：
   verify 117.7/118.8 → 90.4/96.2 ms/周期（**-21%**），墙钟 -15%。
2. **split-half RoPE 缝上验证路径**（`rope_launch` 拆出 + `rope_verify_metal`
   包装，算术与 draft 缝同源）。连同上项同窗复测：verify 100.3/99.6 →
   81.8/82.2（**-18%**）；周期 ~106ms（τ 5.882 不变，≈55 tok/s 热窗）。
   缝开构成复测：5829 → **2643 发/前向**。剩余大池：elementwise ~1193、
   cast 693、Slice/SliceAssign ~604（gdn 卷积状态维护，切片 2 的目标）。
3. **逐头 q/k norm（256 宽）内核化弃用**：字节门在 prose 第 61 字节抓到
   分叉（code 200 token 仍逐字节同）——内核的 64×4 连续段归约序与 burn
   autotune 为 [m,256] 选的 reduce 序不同（2048 宽实证一致，256 不一致）。
   draft 路径从不需要逐字节一致（draft 漂移只改 τ），故该内核此前未经此门。

字节门：缝开/关、默认翻转两配置 × prose/code 双提示词，贪心输出逐字节相同
（分叉中间态已丢弃）。测量来源：同窗交错 CLI trace（isolated，非 CUBECL
System 归属）。门全绿：lib 71、fmt、CPU check。

### 切片 2（gdn 卷积单发化）：按契约回滚——优化器不容逐算子 f16 精确

按 brief 把 `causal_conv`（padded 组装 + 4 tap + 尾巴/record 写出，每层 ~40
发 × 30 层）单发化。字节门三次拦截：①形状判据把 `[conv_dim,1,kernel]` 的
卷积权重当批次张量验、seam 静默不点火（-79 发构成不变暴露了它）；②修好后
prose 第 61 字节分叉——单测复刻 burn 链做位级对照，CPU `half` 算术（乘积先
舍 f16、加法再舍 f16）与 burn 完全一致，而内核五种写法（逐算子 cast、整句
融合、scratch store、同址输出格 store、to_bits/from_bits 位绕行）全部被
cubecl 0.10 的 SSA 折叠/fma 收缩消掉中间舍入（一次偶然位级全对的调试编译
不可复现、无解释，不作为依据）。按「无法精确复现的单点上报维护者、不得擅
自放宽」回滚，conv 保持 burn 链（回滚后二进制与切片 1 参考逐字节相同，
lib 71/71）。坑位与证据记入 `docs/cube-dsl-pitfalls.md`。剩余池（缝开构成
复测）：elementwise ~883、cast 693、copy/slice ~621 发/前向——切片 3 的
「全链 cast 消除」是下一个量级正确的目标。

## 2026-10-16：W4A16 融合反量化路由修复（ADR-0001 目标路径落地）


T11 收口时生产路径按 ADR-0001 的降级方案走“加载点反量化”。本节记录把融合路由
（int4 驻留、matmul 内核反量化）修通的过程与证据。

### 根因（两层）

1. **m=1 并非 kernel 不可用，而是 autotune 候选展开即崩溃**：块量化操作数
   （PackedU32 存储，tune key 中 `elem_rhs = UInt`）使 `vecmat_plane_parallel` 与
   `vecmat_unit_perpendicular` 两个 GEMV 候选在 kernel 展开（comptime cast）时
   panic（"Cast element count must match if input is not scalar"），大形状（m=90 的
   视觉前缀）另有候选在 blueprint 计划中除零。展开 panic 发生在 dispatch 线程，
   上游行为是线程死亡 + 后续 `CallError`，进程随即崩溃——这正是初版“无 q_transpose”
   结论的真实机制（`q_transpose` 本身只是元数据交换，从未缺失）。
2. **tuner 对候选失败无兜底**：`warmup` 本已收集候选错误并跳过，但线程死亡使
   错误以 panic 形式泄漏。

### 修复（vendor 补丁，均带 LOCAL-CHANGES.md）

- `vendor/burn-cubecl`（唯一改动 `kernel/matmul/tune/base.rs`）：两个 GEMV 候选在
  任一操作数为整数存储（packed 量化）时返回 `PRIORITY_NEVER`。
- `vendor/cubecl-common`（唯一改动 `device/handle/channel.rs` 的 `Task::run`）：
  dispatch 线程逐任务 `catch_unwind`，panic 的候选被丢弃并以 `CallError` 呈报，
  tuner 按既有语义跳过——其余候选（含未来未知的不兼容候选）不再具备杀伤力。

### 验证

- 合成回归测试 `quant::gpu_tests::fused_int4_matmul_covers_expert_shapes_and_gemv`：
  专家形状 [512,2048]/[2048,512] × m∈{21,1}，冷 autotune 下与 f64 主机参考
  max|delta| < 5e-2（实际 0.0000）。
- 35B 真实 seam 测试切换到 `LoadMode::Packed`：全部 seam RMSE 与反量化路径一致
  （mlp0_out 1.6e-2、layer0/1 7.7e-3）。
- E2E：冷 autotune 缓存连续 5 次（文本 ×3、单图 ×2）与官方贪心流逐 token 一致，
  stderr 0 panic。
- 生产 CLI 在 cubecl 后端默认 `LoadMode::Packed`（CPU 参考路径仍为加载点反量化）。

### M5 Max 实测（Metal FP16，Packed 融合路由，对照加载点反量化）

| 指标 | 融合（int4 驻留） | 反量化（T11 记录） |
| --- | --- | --- |
| 权重加载 | 21.5 s | 45–47 s |
| 首 token（文本 21 prompt） | 1.2 s | 1.3 s |
| 解码 | 174 ms/token（空载）；MoE 批处理后同负载复测 528→517（见下文批处理节） | 261 ms/token |
| 峰值 RSS | ≈23.5 GiB（含 mmap 检查点驻留） | ≈23.5 GiB（footprint 73 GB，压缩后） |

## 2026-10-16：MiniCPM-V-4.7 35B-A3B-int4 接入（T9–T11）

`crates/minicpm-v` 第二阶段：compressed-tensors pack-quantized int4 权重加载（T9）、
Qwen3.5-MoE 稀疏专家层（T10）与 35B 端到端（T11）。checkpoint 为
`OpenBMB/MiniCPM-V-4.7-35B-A3B-int4`（pin 与 SHA256 清单见 `scripts/download_minicpm_v35b.py`，
22.65 GB，9 分片）。官方对照：transformers 5.19.0 bf16 CPU，
fixture 由 `scripts/export_minicpm_reference.py --part {quant-fixture,moe-layer,greedy-35b,linear-layer-35b}` 生成。

### 关键语义（与 llm-compressor 源码对齐）

- **int4 打包是“无符号 +8 偏移”而非补码符号扩展**：compressed-tensors `pack_to_int32`
  打包 `q + 8 ∈ [0, 15]`，解包为 `nibble − 8`；4 bit 值在 int32 字内按 LSB 优先排列（每字节低半 nibble 在前）。
  初版按补码 nibble 解包导致专家权重完全错误（残差流逐层放大直至溢出）。
  合成 fixture（`quant-fixture`/`moe-layer`）与真实 checkpoint 的运行时专家权重
  （`linear-layer-35b` 内含 `weight_packed` 原始字节与 transformers 物化结果）双重固定该语义，
  最坏相对误差 ≤0.39%（bf16 量化舍入一步）。
- **分组 scale 布局**为 `[out, in/128]` bf16 行主序；全零权重行的 scale 为非规格化小值，属正常现象。
- 35B 线性注意力 32 值头/16 键头：q/k 经 L2 归一化后按 `[h0,h0,h1,h1,…]` 组间重复
  （`repeat_interleave(dim=2)` 等价实现），与官方 `chunk_gated_delta_rule` 内核输入逐值一致。

### 数值对照（Metal FP16，`tests` 内 `text::layer35b_tests`，需本地 35B 权重与 fixture）

真实 35B 第 0/1 层逐 seam 对照（`artifacts/minicpm-reference/linear-layer-35b.safetensors`，
official bf16 逐阶段转储：embeddings→各层残差输出、conv/q/k/v/g/beta/递推核心/门控归一化、MoE 出入）：

| seam | RMSE |
| --- | --- |
| causal conv（SiLU 后）/ q_in / k_in / v_in / 递推核心 / 门控归一化 / g / beta | ≤3.5e-3 |
| 注意力块输出 / MoE 块输出 / layer0 / layer1 残差 | ≤1.6e-2 |

### 端到端（`artifacts/minicpm-reference/greedy-35b.json`，各 16 token，逐 token 一致）

| 案例 | 结果 |
| --- | --- |
| 文本 "What are the three primary colors? One short sentence." | 16/16 一致 |
| 单图 small.png "Describe this image in one sentence." | 16/16 一致 |

### M5 Max 实测（Metal FP16，release，加载点反量化）

| 指标 | 文本（21 prompt token） | 单图（90 prompt token） |
| --- | --- | --- |
| 权重加载 | 45–47 s | 45–47 s |
| 首 token | 1.3 s | ~4 s |
| 解码 | 261 ms/token | ~261 ms/token |
| 峰值 RSS | 23.4 GiB | 23.5 GiB |

### 已知问题（burn 0.21 / cubecl 0.10 限制，均已在代码内规避并留证）

- **BF16 计算通路在本机构建上损坏**：任意模型以 bf16 精度运行即产生通道错位数据
  （1B 复现：同二进制 f16 正常、bf16 乱码）。全程使用 F16；`Precision` 不再暴露 bf16。
  checkpoint 的 bf16 权重/scale 经 f32 精确换载为 F16（无损表示范围内）。
- **前向中途的 host 上传（`into_data`+`from_data`）会污染后续内核**（与分配布局相关、可复现）：
  MoE 逐专家累加改为全 GPU `slice`/`slice_assign`（保留每层一次路由小张量回读），
  生成路径不再有中途上传。
- burn-cubecl 0.21 无 `q_transpose`：转置权重的融合 int4 matmul（W4A16 驻留）不可用，
  生产 CLI 按 ADR-0001 在加载点反量化；非转置路径的融合算子由 `quant::gpu_tests` 固定。
- 35B 逐 token 解码在当时由路由回读 + 逐 (token, expert) 派发主导；按专家批处理已实现，
  见下文「2026-10-16：MoE 按专家批处理」。

## 2026-10-16：MoE 按专家批处理（保留单层一次路由回读）

`SparseMoeBlock::forward` 双路径重写（`crates/minicpm-v/src/moe.rs`），无任何中途 host 上传：

- **解码（T=1）**：去掉逐对的 slice/add/slice_assign 回程，逐 slot 单行 expert matmul +
  顺序累加；内核序列与旧实现逐比特一致（E2E 贪心 parity 不变），每层 ~81 → ~57 内核。
- **前缀（T>1）**：遮蔽 softmax 在设备端构造再归一化 top-k 权重矩阵
  （`topk(k).min_dim` 阈值 + `lower`/`mask_fill(-inf)` + `softmax`；全局最大恒在 top-k 内，
  与 host 路线代数等价，遮蔽列精确为 0），每个被触专家一次全行批处理（3 matmul）后
  按专家升序加权累加。每层 ~10·T·k → ~9+8·E_touched 内核。NdArray 缺 argtopk/topk 内核，
  全程仅用双后端可移植算子（无 sort/gather/scatter/one_hot/argtopk）。
  已知理论边界：第 k 位 logit 精确并列时 `lower` 严格比较会把并列者一并保留，
  与 host top-k 的索引决胜略有出入；f16 连续 logit 下未在 E2E parity（16/16 ×4）中触发。

实测（同机同负载背靠背，M5 Max，Metal FP16，Packed）：

| 指标 | 基线（逐对派发） | 批处理后 |
| --- | --- | --- |
| 首 token（文本 21） | 1.28–1.42 s | 1.16 s 热（冷 7.2 s，见注） |
| 首 token（单图 90） | 3.44 s | 2.32 s 热（-33%；冷 8.5 s） |
| 解码（ms/token，中位） | 522–542 | 507–526（≈-2%，噪声级） |
| E2E 贪心 parity | 16/16 ×2 | 16/16 ×4（文本/单图 × 冷/热） |

冷启动：新 m=T 形状触发 autotune 候选 Metal 着色器编译（一次性 +5~6 s）；macOS 着色器
缓存命中后各进程均为热值（autotune 结果为进程内缓存）。解码收益受限原因：当前条件下
解码下限由每层路由回读同步（40 次/token）主导，裁剪轻量内核无法越界；历史 174 ms/token
为空载测量，同代码当日负载下复测 ≈528。数值验收：`cargo test --lib` 36/36（含 NdArray
合成 fixture T=12 批处理路径与 Metal 35B seam T=21）。证据与复现命令：
`artifacts/minicpm-moe-batching/`（RESULTS.md、JSON 日志、对照二进制）。

## 2026-10-15：MiniCPM-V-4.7 1B 接入（T1–T8）

`crates/minicpm-v`：SigLIP2 视觉塔、混合注意力解码器（18 线性 + 6 全注意力）、canvas M-RoPE、
ChatML 词表与单图端到端生成，全部 Rust/Burn。构建标识：commit `35253a8`（T7）后含 T8 修补；
rustc 1.99.0，Burn 0.21.0（Metal 关闭 fusion，见下）。复现命令与脚本见
[模块文档](crates/minicpm-v/README.md)；fixture 由
`scripts/export_minicpm_reference.py --part {tokenizer-chat,vision,text-layer,linear-layer,greedy,greedy-image}`
离线生成（transformers 5.19.0 + torch 2.14.1 CPU，PIL 图像后端）。

### 数值对照（CPU FP32，`cargo test --no-default-features --features native,cpu`）

| 对照 | 结果 |
| --- | --- |
| 词表编解码 / ChatML 渲染 / 占位符展开 | 官方快照逐 id/逐字节一致 |
| 视觉塔 8 阶段 × 2 图（small.png 全阶段；切片 large.jpg） | PNG 各阶段 RMSE<2e-3；JPEG（解码器差异 ≤3/255）RMSE<2e-2、cosine>0.999 |
| 全注意力层（合成权重） | cos/sin、q_proj、attention、mlp、layer 输出 RMSE<1e-4 |
| 门控 delta 网络层（合成权重） | prefill、终止递推/卷积状态、3 步缓存解码、重置语义 RMSE<2e-4 |
| canvas M-RoPE 位置表（3 组含分行切片） | 逐值一致 |
| 文本贪心（4 组中英提示词，各 24 token） | 逐 token 一致（`tests/greedy_parity.rs`，~209 s CPU） |
| 单图贪心（3 组：96×64 PNG ×2、900×1400 切片 JPEG 444 图像 token） | input_ids、模态类型、token 流全部一致（`tests/image_parity.rs`，~273 s CPU） |
| Metal FP16 抽查 | small.png 案例逐 token 一致（`tests/metal_spot.rs`，~30 s） |

需要本地权重 `models/MiniCPM-V-4.7-1B` 与 `artifacts/minicpm-reference/`；缺失时上述测试自动跳过。
统一检查入口：`./scripts/check cpu`（CI 无权重时仅运行合成 fixture 部分）、`./scripts/check metal`。

### M5 Max 实测（Metal FP16，release 构建，64 token 生成）

| 场景 | 首 token | 逐步 p50 | 逐步 p95 | 峰值 RSS |
| --- | --- | --- | --- | --- |
| 文本提示（~30 token prompt） | 2655 ms | 26.0 ms | 397 ms | — |
| 单图 96×64（1120 patch + 93 prompt token） | 1287 ms | 26.3 ms | 349 ms | 5.14 GB |
| 单图 900×1400 切片（7 crop、483 prompt token） | 3777 ms | 27.3 ms | 452 ms | — |

复现：`cargo run --release --manifest-path crates/minicpm-v/Cargo.toml -- generate --model models/MiniCPM-V-4.7-1B --image … --prompt … --greedy --json --max-new-tokens 64`。
p95 尖峰来自周期性 GPU 同步/内核缓存，不影响 p50 解码速率（≈38 token/s）。

### 已知问题

- burn-cubecl-fusion 0.21 在本模型解码形状（seq=1 递推图）上代码生成崩溃（index out of bounds），
  Metal 构建已关闭 `burn/fusion`（常规 autotune matmul 保留）；CPU 路径不受影响。
- 官方 PIL 处理器忽略 EXIF 方向并直接丢弃透明通道；本产品按 spec 显式应用 EXIF 方向与白底合成，
  因此 parity fixture 不含 EXIF/alpha 图（该行为由 `image::tests` 固定）。

## 2026-10-14–15：MERT2、SheetSage2、YuE2 音乐链路

已实现 `crates/omni-music` 的原生 Rust/Burn 推理、CLI、三组 HTTP 任务接口和网页工作流。
使用四份固定版本原始权重，包括 YuE2-Vae；下载后校验大小与 SHA256，保留模型许可证和下载清单。
模型版本与真实能力见 [来源核对](docs/music-model-research.md)，使用方法见
[音乐模块文档](crates/omni-music/README.md)。这次音乐界面是本地 Web 页面，未新增 Makepad 音乐页。

完整权重的 Metal FP32 与官方 PyTorch CPU 对照如下。测试输入为自行合成的短钢琴乐句；
SheetSage2 始终使用官方要求的完整 300 秒填充上下文。

| 阶段 | 对照结果 |
| --- | --- |
| MERT2，2 秒 / 50 × 1024 特征 | 最大绝对误差 0.000134945，RMSE 0.0000227845 |
| SheetSage2，2 秒乐句 | 47 个转谱 token 完全一致 |
| YuE2 提示与语义贪心生成 | 349 个提示 token、64 个生成 token 完全一致 |
| YuE2 声学阶段，相同 FP32 初始噪声 / 32 步 midpoint | 潜变量 RMSE 0.000585891，最大误差 0.00669503，余弦相似度 0.999999865 |
| FP32 VAE，32 × 64 潜变量 | 61,376 帧双声道，波形 RMSE 0.000000283427，最大误差 0.00000220537 |
| YuE2 普通 / 分页缓存，FP32 贪心 | 提示与 64 个生成 token 完全一致 |

[数值对照 JSON](artifacts/music-validation/reference-comparison.json)、
[分页对照](artifacts/music-validation/paged-comparison.json)、
[参考程序](scripts/validate_music_reference.py)。随机采样和初始噪声使用 Rust RNG，
不宣称与 PyTorch 相同 seed 下随机输出逐位一致；上表声学对照显式使用同一份噪声。
64-token 数值检查是刻意截断的短片段，不是完整歌曲质量评估。

默认 Metal FP16 的真实浏览器 / HTTP 流程已覆盖：

- 12 秒原创测试录音的 MERT 分析：300 帧、1024 维，特征 JSON 可下载。
- 同录音转谱：130 个 token、26 个事件，FP16 与 FP32 token 一致；ABC 和 MIDI 可下载，独立 MIDI 解析通过。
- 将该 ABC 传入 YuE2，完成 750 帧语义 / 32 步声学 / VAE，生成 29.9987 秒、48 kHz 双声道 WAV。
- 直接输入风格与两行原创中文歌词，先创作 ABC，再生成 59.9987 秒、48 kHz 双声道 WAV；浏览器解码与实际下载成功。
- 上述两条音频都达到人为设置的 30 / 60 秒上限并标记 `truncated`；第二条乐谱末尾也未完整结束，需要补全。
  未据此验证自然结束的完整长歌、音质评价或歌词逐字对齐。原始 VAE 单阶段对照不代表完整歌曲波形逐位一致。
- 创作乐谱途中点击停止，任务进入 `cancelled`，上一份音频和输入保留，未发布半成品；后续生成成功。
- 非法 WAV / 请求限额返回 400，非法枚举按 Axum JSON schema 返回 422。任务记录随服务重启清空，已发布的 JSON / ABC / WAV 仍可下载。

[HTTP 记录](artifacts/music-validation/http-report.json)、
[转谱生成截图](artifacts/music-validation/browser-score-song.png)、
[文字创作截图](artifacts/music-validation/browser-text-song.png)、
[音频样本](artifacts/music-validation/browser-text-song.wav)。
单次调试构建的整项计算分别为 116.869 / 256.789 秒，包含模型加载，未作预热或统计性能基准；不当作加速结论。

音乐 CPU 单元测试 **7 项**、HTTP **22 项**通过；两个 crate 的 Clippy `-D warnings`、rustfmt 与原生 Metal 构建通过。
音乐测试覆盖语法、缺失节拍、非 4/4 乐谱、采样阶段掩码与最短长度、重复窗口、非法参数和部分 ABC 采样覆盖；
HTTP 新增完整音乐文件组原子发布、阻塞进度通道退出，以及正常完成不误置取消标志。
最终原生服务额外通过不规划乐谱的双分支 CFG，以及部分 ABC 选项的完整权重短任务；分别核对规划与音乐截断字段，
检查 48 kHz 双声道 WAV 的 extensible IEEE-float 格式、有限样本、精确帧数和整组发布。
[最终 API 检查](artifacts/music-validation/final-api-v2/report.json)、[可复用检查脚本](scripts/validate_music_api.py)。
ASR/TTS/图像推理源代码与 Makepad 未改动，HTTP 旧生命周期回归继续通过。

当前限制：单个音乐请求依次执行，未实现音乐连续批处理或跨请求 Radix 前缀共享；
分页仅覆盖 YuE 自回归历史。转谱最多 300 秒，无长录音分窗拼接；音乐理解输出表征，无现成风格分类头；
提供 ABC 编辑和 MIDI 导出，无 MIDI 导入或五线谱渲染。CPU 做构建与单元测试，未测完整模型 CPU 性能或 WebGPU 性能。

## 2026-10-14：TTS、ASR、HTTP 生命周期深化

三项架构方案已按确认的推荐落地，公开调用保持兼容。
TTS 的私有 `Synthesis` 统一 prompt/CFG、采样与 RNG、完整帧、codec、尾帧、取消和报告，
串行与合批只分别处理张量计算；串行保留原 FP32 归约、FP16 舍入、profiling 阶段及末帧主干计算。
首音频延迟统一计到首次成功音频回调的**开始前**，回调成功后才累加样本。
尾帧回调内取消时，两条路径现在都保留已交付音频并报告取消；尚未交付的音频丢弃。
`GLOSSARY.md` 已补充 Speech synthesis 与 First audio latency。

ASR 的 `Session` 统一分块、前瞻、EOF、解码轮次、提交与缓存清理；`Transcription` 统一重采样和统计。
`Session::push/finish`、`Transcription::push/finish` 与离线 `TranscriptionBatch` 都推进同一个 async implementation，
整批每轮仍最多 256 个 token。音源 adapter 保留停止后的错误检查，真实采集错误不会被取消覆盖。

HTTP 的真实 `Models<Driver>` 独立管理驻留模型与唯一活动任务类型。一个私有 Driver seam 提供 load/admit/step，
生产执行器与 ScriptedDriver 都走这套任务归属和输出逻辑；driver 不持有 Context，也不发布终态。
加载与准入进度同步可见，计算后再应用拥有独立所有权的进度、PCM 和完成结果。
重复/未知 ID 或类型不符会在结果文件产生前失败并清理整个受影响批次；重复准入也先释放引擎再返回错误。
Jobs 的 FIFO、唯一终态和原子文件发布继续作为同一条生产执行路径接受测试。

### 验证结果

根 CPU **44 项**、TTS CPU **8 项**、TTS 含 Metal **11 项**、HTTP **19 项**、桌面 **24 项**通过；
桌面 1 项物理默认输出设备单元测试仍按原策略忽略，完整模型桌面回归实际运行播放路径。
根 Wasm 与 Makepad Web 检查、API/桌面构建及格式检查通过；API Clippy `-D warnings` 通过。
根和 TTS 的普通 Clippy 通过，保留未触及代码中的既有风格告警；Wasm 也有既有 vendor/原生方法未使用告警。
[CPU 与平台检查](artifacts/architecture-deepening/)、[完整执行记录](artifacts/architecture-deepening/execution.json)。

TTS 用原始完整权重、固定 seed、温度 0.9 生成 **129 帧**，串行与合批 token 完全相同、PCM RMS 均为 **0**；
生成 token 也与改造前保存的基线逐帧一致。
额外覆盖 CPU 深度采样与 nucleus、CFG、分页、host/sync profiling 阶段计数、完整 chunk 和尾帧回调取消、
预取消、非法帧限额、慢回调计时、回调失败隔离、中途加入和重试。
[旧基线对照](artifacts/architecture-deepening/tts-baseline.json)、
[完整权重与生命周期](artifacts/architecture-deepening/tts-long.log)、
[CPU 采样/host](artifacts/architecture-deepening/tts-cpu-nucleus-host.log)、
[分页/sync](artifacts/architecture-deepening/tts-paged-sync.log)。

ASR 普通/分页完整权重的三条转写均与旧基线一致，中途加入及取消隔离通过。
新增 CPU 模型事件对照覆盖空输入、非零 lookahead 前后、精确分块 EOF、尾包、不同分包宽度、
逐事件 delta/统计，以及正常终态后的迟到取消。
[ASR 基线](artifacts/architecture-deepening/asr-baseline.json)、
[普通缓存](artifacts/architecture-deepening/asr-dense.log)、[分页](artifacts/architecture-deepening/asr-paged.log)。

完整模型计时在编译结束后串行运行，设备、精度与前一轮相同。ASR 取第二轮预热结果；
TTS 预热批量 1/2/3 的完整生成长度后计时，不包含模型加载或额外生命周期验证。

| 工作负载 | 串行总耗时 | 合批耗时 | 本组吞吐倍率 |
| --- | ---: | ---: | ---: |
| ASR，三段官方 WAV | 6.396 s | 4.732 s | 1.35× |
| ASR，同输入、开启分页 | 6.161 s | 4.704 s | 1.31× |
| TTS，三条普通/带指令朗读，最多 48 帧 | 19.257 s | 11.702 s | 1.65× |

以上是当前机器的一组测量，不能当作架构改造相对旧版本的加速，也不代表所有输入或浏览器后端。
为排除早先记录与本轮单次计时的差异，保留的改造前 TTS 程序与当前程序又串行做了一组同机对照：
串行 **17.953 → 17.959 s**（+0.035%），
合批 **11.741 → 11.685 s**（-0.471%）。
本组未观察到稳定的回退；仍是单组观测，不作统计显著性或额外加速结论。
ASR 同一程序补测为串行 5.434 s、合批 3.939 s（1.38×），文本仍一致；
两次运行有明显计时波动，因此不把异轮差值当作改造效果。[ASR 补测](artifacts/architecture-deepening/asr-dense-repeat.log)。
[旧/新版计时与程序哈希](artifacts/architecture-deepening/paired-performance.json)。

ASR/TTS 分页与批容量默认策略未调整；图像计算内核没有改动。

### 真实应用

真实 HTTP 覆盖 ASR/TTS 三请求重叠、取消后的幸存请求、部分 WAV、跨模型 FIFO、Hy-MT 翻译及前缀复用、
两条图像请求的独立取消、PNG 下载与退出清理。SIGINT 后 0.349 秒正常退出。
桌面完整模型回归通过加载取消/重试、官方转写、ASR/TTS 翻译、合成和播放分别停止、
保存面板取消后恢复、再次合成期间保存上次完成结果；两份旧结果 WAV 逐字节一致，为 24 kHz、5.84 秒。
测试使用自己启动的隐藏窗口与隔离设置；文件面板回复由测试库提供，没有申请物理麦克风权限。
[HTTP 报告](artifacts/architecture-deepening/http/report.json)、[桌面报告](artifacts/architecture-deepening/native/report.json)。

原生程序两个启动路径已同步，SHA256：`102d07f3e7423a753ba5d86652e801d3020c55bb20ba5568d835f87d9f413bef`。

```sh
cargo test --no-default-features --features native,cpu --lib --tests
cargo test --manifest-path crates/omni-api/Cargo.toml --no-default-features --features cpu --target-dir target/omni-api-cpu
cargo run --manifest-path crates/breeze-tts/Cargo.toml --example batch-bench -- --long --sampled --warm-full --lifecycle
BREEZE_TTS_PROFILE=host cargo run --manifest-path crates/breeze-tts/Cargo.toml --example batch-bench -- --short --sampled --cpu-depth --nucleus --lifecycle
BREEZE_TTS_PROFILE=sync cargo run --manifest-path crates/breeze-tts/Cargo.toml --example batch-bench -- --short --paged --lifecycle
cargo run --example asr-batch-bench
cargo run --example asr-batch-bench -- --paged
python3 scripts/validate_multimodal_batch_api.py --output artifacts/architecture-http-new
python3 prototypes/makepad/validate-lifecycle.py --output artifacts/architecture-native-new
```

## 2026-10-14：ASR/TTS 连续批处理与图像去噪合批

ASR、TTS 和图像均增加了有容量上限的 `try_submit` / `step` 执行器，并接入 HTTP 工作线程。
ASR 合并文本解码器的投影和 MLP，单独维护音源、重采样、编码器及文本提交；每轮全批最多预填充 256 token。
TTS 按帧合并语义头和 16 码本深度解码，主干缓存、采样 RNG、音频 codec 和回调各自独立。
图像按去噪步合并 DiT 投影和 MLP，支持不同尺寸、时间步及 CFG，保留原有图像注意力内核。
新请求能在轮次之间加入；结束、取消或单请求音源/输出错误释放该请求，其他请求继续。
共享前向计算错误会清理受影响的整个批次，真实错误不会被同时到达的取消请求掩盖。

共享分页 GPU 内核已移到 `omni-kernel::paged`，Hy-MT 仍持有自己的前缀树、固定容量池和页面生命周期。
ASR/TTS 使用 `DecodeCache` 私有历史：16 token 一页、几何增长、ASR 后缀回退，以及写入对前次读取的依赖。
即使延迟读回后再回退或扩容，也不会先覆盖仍在使用的 KV。长预填充走矩阵注意力，短查询直接读页表。
语音请求不共享文本前缀缓存；TTS 仅对长主干分页，深度解码和 codec 保留原缓存。

### 完整模型性能

使用本机完整原始权重、Metal FP16，同一进程对比逐条执行与合批，排除模型加载。
以下测量独占测试 GPU，期间没有编译作业。基准使用优化过的 dev 构建（本包 opt-level=1、依赖 2）；
发布版另做桌面功能回归。预热后各测一组，不能外推到所有输入、硬件、并发数或桌面单请求延迟。

| 工作负载 | 串行总耗时 | 合批耗时 | 吞吐倍率 | 对照结果 |
| --- | ---: | ---: | ---: | --- |
| ASR，3 段不同长度的官方 WAV | 5.755 s | 4.326 s | **1.33×** | 每条转写一致 |
| TTS，3 条普通/带指令朗读，最多 48 帧，种子固定、温度 0.9 | 17.918 s | 11.598 s | **1.54×** | 129 帧 token 与 PCM 完全一致 |
| 图像，256×256 与 288×256，含 CFG，8 步 | 5.087 s | 3.501 s | **1.45×** | 两图像素 RMS 0.420 / 0.716（0–255） |
| 图像，512×512 与 544×512，含 CFG，8 步 | 13.818 s | 13.804 s | **1.00×** | 两图像素 RMS 0.393 / 0.836 |

[ASR 记录](artifacts/multimodal-batching/asr-batch-final.jsonl)、
[TTS 完整长度预热记录](artifacts/multimodal-batching/tts-batch-warm.jsonl)、
[小图记录](artifacts/multimodal-batching/image-batch-bench.jsonl)、
[512 像素记录](artifacts/multimodal-batching/image-batch-512.jsonl)。
TTS 预热批量 1/2/3 的完整生成长度，合批首音频约 0.532–0.569 s；吞吐增加不意味着每条请求更早返回。
初期仅预热两帧、或与编译并发的实验日志保留在同目录，不能代替此处最终测量。

TTS 同步分阶段探针显示，短句普通朗读约 78% 的计算时间在深度解码，带引导朗读约 70%；
因此先优化深度解码的合批。最初 FP16 合批与单请求不一致，定位到 GEMV/RMS 只接收 1–4 行，
更大的合批切到了不同数值路径。内核现在支持至 64 行（16 请求 × 2 CFG 分支 × 首步 2 token），
保留原精度与舍入边界；标量 FP64 对照、Metal 及 ANE/Metal 桥接均验证通过。

### 默认策略与分页验证

HTTP 的 ASR/TTS 默认容量为 4，分别由 `--asr-batch-size` / `--tts-batch-size` 调整（1–16）。
图像默认容量为 1；小图或需要并发时可用 `--image-batch-size 2`（允许 1–4）。512 像素测试没有明显收益，
因此没有默认增加图像内存和并发。跨模型任务保持 FIFO，后来的同类任务不能越过排队中的其他模型。
新请求的文本/参考输入预处理仍可能推迟现有请求；离线 ASR 批处理不接收等待真实麦克风的阻塞音源。

ASR/TTS 分页默认关闭，HTTP 分别提供 `--asr-paged-attention` / `--tts-paged-attention`，
库对应 `StreamConfig::paged_attention` / `SynthesisOptions::paged_attention`，TTS CLI 也提供 `synthesize --paged-attention`。
ASR 三请求分页合批 4.328 s，与密集缓存 4.326 s 基本相同；单请求预热差异也很小。
[分页 ASR](artifacts/multimodal-batching/asr-batch-paged.jsonl)通过文本、中途加入和取消隔离验证。
TTS 短分页合批同样通过 token、PCM、中途加入、取消和重试对照，尚无证据说明分页本身改善端到端吞吐。

ASR 分页直接对照官方随机小模型的 mel、编码、prefill 和 3 步 logits；4 组音频长度全部通过既有容差，
解码 argmax 一致。TTS 完整预训练模型用固定音频 token 对照两种历史，续帧 logits 的相对 L2 误差为
0.1275% / 0.1038%，其余记录阶段一致。FP16 注意力舍入仍可能改变随机生成轨迹，未宣称分页与密集缓存逐样本等同。
图像单请求 DiT 和参考图 DiT 均通过既有官方张量容差（相对 RMSE <2%、cosine >0.9998），缓存/无缓存对照 <1%。
图像合批实测文生图，尚未测量多参考图并发的性能。
[ASR 参考](artifacts/multimodal-batching/asr-reference-paged.log)、
[TTS 固定输入](artifacts/multimodal-batching/tts-cache-verify.jsonl)、
[图像官方对照](artifacts/multimodal-batching/image-reference-checks.json)。

### 回归与复现

根 CPU 43 项、共享分页缓存 2 项、Hy-MT 12 项、TTS 普通/Metal 10 项及 ANE 桥接 2 项、
图像 10 项、HTTP 12 项、桌面 24 项通过；桌面 1 项默认音频设备单元测试仍按原策略忽略。
真实 HTTP 验证覆盖 ASR/TTS 三请求重叠、图像两请求重叠、单个取消、幸存请求、有效 WAV/PNG 下载、
跨模型 FIFO、队列背压及退出清理，SIGINT 后约 0.291 s 退出。
桌面发布版通过真实加载取消/重试、官方转写、ASR/TTS 翻译、合成/播放与推理分别停止，
以及新一轮生成期间保存上次已完成 WAV；保存结果为 24 kHz、5.84 s。
[HTTP 报告](artifacts/multimodal-batching/http-final/report.json)、
[桌面报告](artifacts/multimodal-batching/native/report.json)、
[检查与程序哈希](artifacts/multimodal-batching/checks.json)。

根 WebGPU/Wasm 和 Makepad Web 检查、原生发布构建及格式检查通过。API/Hy-MT 的 Clippy `-D warnings` 通过；
根、TTS、图像的普通 Clippy 检查通过，仍有已有风格告警；Wasm 另有原生清理方法未使用和 vendor 条件编译告警。
本轮没有宣称全库严格 lint 清零，也没有请求物理麦克风权限或运行完整 ASR 语料准确率评测。

```sh
cargo test --no-default-features --features native,cpu --lib --tests
cargo run --example asr-batch-bench
cargo run --manifest-path crates/breeze-tts/Cargo.toml --example batch-bench -- --long --sampled --warm-full
cargo run --manifest-path crates/qwen-image/Cargo.toml --example batch-bench -- --512
cargo build --manifest-path crates/omni-api/Cargo.toml --target-dir target/omni-api
python3 scripts/validate_multimodal_batch_api.py --output artifacts/multimodal-http-new
python3 prototypes/makepad/build-native.py
python3 prototypes/makepad/validate-lifecycle.py --output artifacts/multimodal-native-new
```

## 2026-10-14：分页注意力与连续批处理

Hy-MT 已改用固定容量分页 KV 池与 token 前缀树。Metal FP16／FP32 内核直接访问非连续页表，
支持分组查询、逐请求因果位置、共享尾页复制，以及按 256 个位置分段的 FP32 在线 softmax。
多个请求的 token 行在同一次前向中共用投影、MLP 和词表计算；每轮允许新请求加入，完成或取消后释放名额。
HTTP 保持独立终态、队列背压和 FIFO 模型切换；CLI 与桌面单请求也复用这一执行路径。

Hy-MT **12 项**测试通过，包含独立标量参考、8／64／128 维注意力、非连续页面、GQA、
部分尾页、写前复制、8192 token 上下文、缓存淘汰后的活动页面保留、混合 prefill/decode logits、
中途加入、取消隔离、容量不足与重试、独立采样 RNG，以及随机小模型的官方 Transformers 参考。
分页路径直接对照 7 组官方参考，Metal FP16 最大绝对误差 **2.415e-4**，每组 argmax 一致；
Metal FP32 最大误差 **8.941e-8**。参考权重是固定种子生成的随机小模型，随测试收录，无需下载预训练权重。
根 CPU **40 项**、HTTP **12 项**、桌面 **24 项**回归通过（1 项音频设备测试忽略）。
[Hy-MT 测试及误差](artifacts/paged-validation/hy-unit.log)、[根测试](artifacts/paged-validation/root-tests.log)、
[HTTP 测试](artifacts/paged-validation/api-tests.log)、[桌面测试](artifacts/paged-validation/desktop-tests.log)。

完整 **Hy-MT2-7B FP16** 在同一进程比较 4 个贪心翻译请求，关闭跨请求前缀缓存，
预热批量 1／2／3／4 后计时。单独执行总耗时 **2.836 秒**，批处理 **1.335 秒**，
64 个输出 token 的总吞吐从 **22.56** 提升到 **47.94 token/s**，这组工作负载约 **2.12 倍**。
四个译文、输出 token 数和结束原因逐一相同；另外验证长提示词在解码中加入、取消其中一个请求、
其余请求完成、缓存重试一致，以及清除缓存后全部页面可复用。
[完整结果](artifacts/paged-validation/batch.json)、[运行日志](artifacts/paged-validation/batch-bench.log)。
这不是首请求延迟或所有输入的性能保证：未预热缩小后的 2／3 请求形状时，初次批次因编译和 autotune
耗时 4.184 秒，见[首次运行](artifacts/paged-validation/batch-initial.json)。模型加载约 17.7 秒，未计入吞吐。
默认 KV 缓冲容量约 2 GiB；`used_pages` 表示已占用的逻辑页，不表示整池 GPU 缓冲已释放。

真实 HTTP 验证观察到 **4 个同时运行**的翻译；取消一个后保留部分译文，其余三个正常结束。
重复请求复用 33 个前缀 token。8 个已接受任务包含运行与排队请求时发送 SIGINT，
约 **0.174 秒**完成关闭并以 0 退出；重启后缓存为空，译文一致。
[HTTP 报告](artifacts/paged-validation/http-retry/report.json)。
通用 WebGPU 的 FP32 参考路径也通过 logits、前缀复用、分支与取消检查；这是本机 WebGPU 后端验证，
不代表已覆盖所有 GPU 或浏览器平台。[WebGPU 日志](artifacts/paged-validation/webgpu-reference.log)。

最终原生发布版使用完整 ASR、TTS、Hy-MT 权重，实际操作两个页面的翻译按钮通过：
ASR 原文保留，TTS 翻译期间的新编辑不被覆盖，主动采用译文生效；长文本停止只产生一个取消终态，
随后短句翻译成功。测试在自己启动的隐藏窗口中运行，结束后已退出。
[原生报告](artifacts/paged-validation/native/report.json)、[ASR 截图](artifacts/paged-validation/native/asr-translation.png)、
[TTS 截图](artifacts/paged-validation/native/tts-translation.png)。ASR 的实验性 ANE 编译失败后正常回退，
官方 WAV 转写正确；Hy-MT 翻译使用 Metal 分页路径。
原生发布构建、Metal API 构建、Hy-MT／API Clippy 与四个工作区格式检查通过。
已同步更新 `target/makepad-native/release/r2t2-makepad` 和旧启动路径 `target/release/r2t2-makepad`，
两份程序 SHA256 相同，见[检查汇总](artifacts/paged-validation/checks.json)。

```sh
cargo test --manifest-path crates/hy-mt/Cargo.toml --lib
cargo run --manifest-path crates/hy-mt/Cargo.toml --example batch-bench
cargo test --manifest-path crates/omni-api/Cargo.toml --no-default-features --features cpu --target-dir target/omni-api-cpu
cargo build --manifest-path crates/omni-api/Cargo.toml --target-dir target/omni-api
python3 scripts/validate_paged_api.py --output artifacts/paged-validation-new
python3 prototypes/makepad/build-native.py
python3 prototypes/makepad/validate-translation.py --output artifacts/paged-native-validation-new
```

## 2026-10-14：任务生命周期、桌面并发与结果发布

四项深入整改已落地：HTTP 任务统一管理准入、取消、唯一终态和退出等待；桌面共享模型执行名额；
结果文件完整编码后原子发布，下载寿命与任务历史分开；模型加载在张量读取与阶段之间检查取消。
桌面导航和全局停止固定在滚动区域之外。播放、文件对话框和结果导出各自管理状态；
冷翻译的加载与续跑作为同一次已接受操作处理，页面自动加载不会抢占续跑。

根 CPU 测试 **40 项**、HTTP **10 项**、桌面 **24 项**通过（另有 1 项音频设备测试默认忽略）。
HTTP 测试经生产工作循环覆盖队列拒绝不淘汰历史、排队取消先释放输入、错误与取消竞争、panic 恢复、
发布失败清理后重试、关闭等待模型释放；经实际路由检查未发布文件不可见及历史淘汰、重启、手动删除后的下载。
根 Wasm、Makepad Wasm、Metal API 和原生发布构建通过；相关 Rust 格式检查通过。

完整模型在真实浏览器跑通 ASR → 翻译、TTS → WAV 下载、256×256 文生图 → PNG 下载。
再执行 34 个短翻译任务淘汰旧记录，旧 WAV/PNG 仍可下载；服务重启后，从原页面点击旧链接，
两份文件 SHA256 与首次下载一致。合成运行中、另有 3 个排队请求时发送 SIGINT，服务正常退出；
重启后保留的 0.16 秒部分音频可解码为 24 kHz、3840 个样本。
[历史淘汰](artifacts/lifecycle-validation/retention.json)、[退出](artifacts/lifecycle-validation/shutdown.json)、
[浏览器重启下载](artifacts/lifecycle-validation/restart.json)、[音频解码](artifacts/lifecycle-validation/decoded-audio.json)。

原生完整模型实测通过加载取消后重试、跨页编辑、冷翻译续跑优先于新页面自动加载、
TTS 尾音/重播与翻译并行、两种停止相互独立、保存对话框取消后恢复，以及再次生成期间导出旧结果。
旧结果两次保存的 WAV 逐字节相同；810 像素高窗口向下滚动后，全局停止和导航仍可见。
[原生报告](artifacts/lifecycle-validation/native/report.json)、[滚动时停止入口](artifacts/lifecycle-validation/native/scroll-with-global-stop.png)、
[播放与推理并行](artifacts/lifecycle-validation/native/playback-and-translation.png)。
测试使用自己启动的隐藏窗口、隔离设置和真实模型；文件面板只替换系统返回值，回调和导出仍走生产路径，
不覆盖系统文件选择窗口的鼠标/键盘操作。没有请求物理麦克风权限。
补充的完整权重阶段取消实测通过：ASR 在语音编码器第 3/24 层、TTS 在文本编码器第 10/26 层发出停止，
均返回 `load_cancelled`，随后各自重新加载到 ready。[加载取消报告](artifacts/lifecycle-validation/loading/report.json)。
原生 ASR 的实验性 ANE 路径出现一次系统编译失败并回退，转写仍与官方样本预期一致；本轮不作 ANE 性能结论。

Hy-MT Metal FP16 的 7 组独立参考 logits 检查通过，最大绝对误差 `2.984e-4`；
Radix 缓存开关、重复提示、分支和取消后重试也通过。[数值日志](artifacts/lifecycle-validation/hy-metal-reference.log)。
该轮验证时使用 Burn 连续 attention 与前缀 KV 复用，没有新增吞吐结论；分页内核及连续批处理见本页最新记录。

```sh
cargo test --no-default-features --features native,cpu
cargo test --manifest-path crates/omni-api/Cargo.toml --no-default-features --features cpu --target-dir target/omni-api-cpu
cargo test --manifest-path prototypes/makepad/Cargo.toml --target-dir target/makepad-native
python3 prototypes/makepad/build-native.py
python3 prototypes/makepad/validate-lifecycle.py --output artifacts/lifecycle-validation-new
```

## 2026-10-14：架构回归、Hy-MT2、Web API 与前缀复用

五项架构整改已落实：共享张量运算与 KV、Makepad 工作线程生命周期、统一转写运行器、
模型封装与权重布局，以及共享 Metal/ANE 自定义内核桥。完整访问权限下的硬件回归通过：
共享内核 3 项、TTS Metal/ANE 5 项、ASR 独立参考 FP32/FP16 各 4 组、
Qwen 注意力 14 组及 dtype 拒绝检查、TensorOps 9 种尺寸和 3 种布局。
注意力基准的单 query 因果参考遮罩已修正，重新对照通过。
浏览器真实 WebGPU 的 FP32/FP16 各 12 项检查通过，覆盖事件对象、重采样尾包、运行中停止及唯一终态。
硬件日志：[注意力](artifacts/architecture-validation/attention.jsonl)、[TensorOps](artifacts/architecture-validation/tensorops.jsonl)。

Hy-MT2-7B 使用固定 revision `9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223` 的原始权重，
完整 4 个分片已下载并校验。运行时为 Rust/Burn；官方 Transformers 仅用于导出独立数值参考。
两层微型模型保留固定 α 的 NTK RoPE、GQA、旋转后的 Q/K RMSNorm 和 tied embedding，
使用非均匀归一化权重检验运算顺序。7 组 logits 覆盖完整 prefill、分块缓存解码和分支续算：

| 后端 | 精度 | 相对官方 FP32 的最大绝对误差 |
| --- | --- | --- |
| CPU NdArray | FP32 | ≤ 1.20e-7 |
| Metal | FP32 | ≤ 6.00e-8 |
| Metal | FP16 | < 3.00e-4 |
| 原生 WebGPU | FP16 | < 2.50e-4 |

Metal FP16 对照暴露了 Burn 0.21 将注意力残差和后续 RMSNorm 融合时的短行误差；
在残差处通过共享 CustomOp 桥划分融合边界后通过原测试容差，无 CPU 回读或设备同步。
每种后端同时验证缓存开关、相同输入、共享前缀分支和取消后重试。
根项目 CPU 测试 39 项、Hy-MT 采样测试、HTTP 请求/队列测试 3 项通过；
根 Wasm、Makepad 原生发布构建与 Wasm 检查通过。现有上游重复包及 vendor 未使用变量提示仍存在。

完整 7B 模型的三轮热态交替测量，两边先预热，均使用贪心解码同一句中文，33 个 prompt token、17 个输出 token：

| 路径 | 复用 token | prefill 中位数 | 总推理中位数 |
| --- | --- | --- | --- |
| 缓存关闭 | 0 | 94.807 ms | 759.953 ms |
| Radix 缓存命中 | 32 | 38.405 ms | 701.186 ms |

本例 prefill 为 **2.47 倍**，总推理耗时减少 **7.73%**，输出逐字相同。
修改末尾的请求复用 28 个 token，与关闭缓存的译文相同；首次遇到新后缀形状仍有编译开销，
该分支检查不作为提速结论。压缩前缀树另有真实 KV 内容、分裂、LRU 淘汰、预算及活动引用存活测试。
该阶段优化为 Burn 张量的前缀 KV 复用，使用连续 attention 输入；后续分页注意力内核及连续批处理见本页最新记录。
[原始测量](artifacts/hy-mt-validation/radix-bench.jsonl)、[汇总](artifacts/hy-mt-validation/radix-summary.json)。

原生界面用完整 ASR、TTS 和 Hy-MT 模型跑通：转写后单独显示译文、保留 ASR 原文、
翻译期间编辑 TTS 文本、主动应用译文、取消只产生一个终态以及再次翻译。
[原生报告与截图](prototypes/makepad/VALIDATION.md#2026-10-14-hy-mt2-翻译)。
本地 HTTP 服务和真实浏览器跑通 WAV 转写 → 翻译、TTS 翻译 → 合成 → WAV 下载、
文生图 → PNG 下载，以及指定任务取消。测试确认音频 RIFF 内容和 256×256 图片解码；
其他 HTTP 请求不能取消同队列中的无关任务。
[API 报告](artifacts/hy-mt-validation/web-api.json)、[网页截图](artifacts/hy-mt-validation/web-api.png)。
这些是数值与功能回归，短句缓存收益不代表长文、不同语言或持续并发的通用性能结论。

复现新增检查：

```sh
# 仅导出参考需要独立 Python 3.12+ 开发环境；本轮使用 Python 3.12.15。
python3 -m pip install -r scripts/requirements-hy-mt-reference.txt
python3 scripts/hy_mt_reference.py
cargo run --manifest-path crates/hy-mt/Cargo.toml --example verify-reference -- artifacts/hy-mt-reference f16
cargo run --manifest-path crates/hy-mt/Cargo.toml --no-default-features --features native,cpu \
  --example verify-reference -- artifacts/hy-mt-reference f32
cargo run --manifest-path crates/hy-mt/Cargo.toml --no-default-features --features native,webgpu \
  --example verify-reference -- artifacts/hy-mt-reference f16
cargo run --manifest-path crates/hy-mt/Cargo.toml --example radix-bench -- models/Hy-MT2-7B
cargo test --manifest-path crates/omni-api/Cargo.toml --no-default-features --features cpu
python3 prototypes/makepad/validate-translation.py --output artifacts/translation-validation-new
```

## 通用 Burn ANE Backend

实现位于 `crates/burn-ane`，由 `vendor/burn` 的 `ane` feature 导出
`burn::backend::Ane`。模型使用正常 Burn API；图由 Rust 自动生成、编译并执行，未调用旧的
模型专用导出器。以下结果与下一节的历史 Core ML encoder 实验分别记录。

### 2026-10-11：按用户要求收尾

工作树恢复到 `infinite-mask-source`，保留稳定卷积块复用、IOSurface 资源恢复及负无穷遮罩修复。
43 份源码与冻结指纹一致，重新运行的 80 项后端测试全部通过；发布构建通过，4 个可执行文件
与原冻结版本的 SHA-256 全部一致。CLI 配置读取及麦克风参数检查通过，未追加性能实验。
图内行切片实验已有 81 项测试、两组 Clippy 和发布构建，但未测模型性能，已撤回并完整归档。
收尾构建已有模型及跨窗口数值检查、47 份数组位对照和短测；完整集及 30 分钟测试仍属于较早版本，
不能据此宣称该构建已完成全部验收。原实时性能目标未达成，本轮停止继续优化。
[版本选择、测试范围与运行命令](docs/ane-optimization-goal.md#2026-10-11-收尾选择)。

### 2026-10-10：稳定卷积块复用（历史完整集最佳基线）

仅复用 mel 已稳定的完整 100 帧卷积块；全局裁剪值变化、输入缩短、关闭复用或清空会使缓存失效。
完整双向注意力窗口仍按原语义计算，卷积块复用不计入 decoder 的稳定音频前缀。
6 项 CPU 单元、8 项模型/流式回归、ANE 与 Core ML 功能组合 Clippy、Wasm 检查通过。
真实权重的 9 阶段探针中，缓存与完整重算的 ANE 输出逐字节相同，相对 Metal 最大 RMSE 为 1.7919%，通过原 5% 门槛。
后端源码与程序槽版一致，沿用其 74 项后端测试与原生桥接的 192 组缓冲验证。
新增回归覆盖稳定帧边界、8 秒窗口、变化的尾帧、音量变化、输入缩短、关闭复用及清空。

三个交替短音频对照，控制 **RTF 0.8392 / p95 183.27 ms**，候选 **0.7944 / 171.41 ms**，耗时降低 5.34%；
六次正文相同、直接 ANE 检查通过。候选三轮 RTF 均达标，p95 均未达标。
完整 96 条一轮为 **0.7708 / 173.38 ms / CER 3.1052%**，正文全部与 IR 遍历版相同。
相对归档控制耗时降低 5.14%、p95 降低 7.50%；CER 相对 Metal +0.0706 个百分点，精度、直接 ANE、缓存预算通过。
78 / 96 条 RTF、28 / 96 条 p95 达标，完整性能验收仍未通过；热态 0 新图/淘汰/重载失败，8874 次重载。

相同 LRU 策略下各一轮 40 秒输入，控制 **1.0373 / 274.59 ms**，候选 **0.9951 / 268.78 ms**；
结束积压 2.42 / 1.46 秒，停止响应 1.35 / 99.37 ms，RSS 采样峰值 25.75 / 25.42 GB。
共同 233 个提交步骤正文一致，直接 ANE、缓存、final/delta 检查通过。处理跨度不同，不比较累计阶段的等工作量比值。
同一冻结构建的 LRU 30 分钟测量已结束：墙钟 1802.24 秒，**RTF 1.0228 / p95 290.78 ms**，
结束积压 42.74 秒、峰值 43.13 秒；SIGINT 停止响应 105.32 ms，RSS 采样峰值 22.82 GB。
单一 final、文字增量、主动停止和缓存预算检查通过；IOSurface 分配失败触发 1045 次相关回退，
其中 1015 次命中失败缓存，**性能与直接 ANE 验收未通过**。此测试未覆盖物理麦克风或 UI 点击。
完整集使用默认 `prefer-small`，尚无 LRU 完整集验收。
[长测审计及原始事件](artifacts/burn-ane-goal/convolution-prefix-lru-soak30-summary.json)。
后续资源恢复候选已通过 79 项后端测试和两组 Clippy，新增有界重试、临时失败不缓存及旧输出生命周期回归。
真实模型 10 个阶段、20 份数组与冻结的程序槽版逐字节一致，推理无新增回退。
短音频单轮为 0.8067 / 173.36 ms，正文相同，性能门槛未通过。
30 分钟复测已完成：**RTF 1.0174 / p95 293.70 ms**，结束积压 33.07 秒、SIGINT 停止响应 63.97 ms，
RSS 采样峰值 25.68 GB、结束 23.07 GB。76 次真实资源恢复全部成功，矩阵回退、编译失败和失败缓存命中为 0。
直接 ANE、缓存、时长、主动停止及 final/delta 检查通过；与旧长测共同的 10,995 个非 final 步骤正文一致。
保留资源修复，性能仍未通过，底层分配失败仍会出现并触发恢复。
[修复与验证范围](docs/ane-optimization-goal.md#iosurface-临时资源恢复已保留性能仍未达标)。
原生 Softmax 与动态遮罩填充值候选通过 80 项后端回归和真实模型原 5% 数值门槛，
短音频为 **RTF 0.8336 / p95 179.05 ms**，同期控制为 **0.8142 / 175.71 ms**，未显示提速，已归档。
遮罩路径只归一化一次的候选也通过 80 项回归、两组 Clippy 和真实模型原 5% 数值门槛，
最大 encoder / decoder 相对 RMSE 为 1.2508% / 0.9468%；短测 **0.8383 / 195.57 ms**，
同期控制 **0.8158 / 174.41 ms**，未显示提速，已归档。
负无穷遮罩修复已保留，省去行最小值归约，同时通过极值概率用例：80 项回归、两组 Clippy、
真实模型及跨窗口编码器原 5% 数值门槛通过，47 份数组全部与各自冻结控制逐字节一致。
短音频 **RTF 0.8200 / p95 173.75 ms**，同期控制 **0.8180 / 175.32 ms**，各一轮不足以证明
稳定加速，均未通过性能门槛。保留的是数值边界修复；完整集 CER 与本构建长测尚待验证。
[修复验证范围](docs/ane-optimization-goal.md#负无穷遮罩已保留数值修复性能未达标)。
后续 40 秒逐图诊断发现三组较大解码投影/FFN 形状占原生执行约 36.34%；追踪计时不作为性能验收。
同一 MIL 图内的 64 行投影切片实验通过 81 项后端回归、两组 Clippy 和发布构建；
按用户要求收尾后撤回归档，未继续模型或性能测量。
[图内切片实验](docs/ane-optimization-goal.md#图内投影行切片已撤回归档未测性能)。
历史完整集构建为 `convolution-prefix-bin`；源码、二进制指纹及证据在 `artifacts/burn-ane-goal/convolution-prefix-*`。
[完整对照与复现](docs/ane-optimization-goal.md#稳定卷积块复用当前完整集保留版)。

### 2026-10-10：IR 遍历减少分配（归档控制）

常用输入/输出遍历使用借用切片与栈上引用，其他算子直接委托 Burn 上游迭代器，保留全部输入顺序和输出。
73 项后端测试、后端 Clippy 和使用原四项豁免的根 Clippy 通过；原生桥接未改，沿用原 192 组缓冲检查。
row64 / KV256 / 单 token 融合的模型 20 份数组与二进制键版逐字节相同，原 5% 数值门槛通过。
三轮短测 **0.8366 / 183.76 ms**，同期对照 **0.8470 / 185.49 ms**；图规划降低 16.53%，总耗时降低 1.23%。
每轮热态无新图/淘汰、121 次重载、4209 次直接执行，转写相同。

KV128 三轮短测 **0.8166 / 179.32 ms**，模型 20 份数组也与 KV256 逐字节相同；KV64 单轮 0.8663 / 181.80 ms，重载增至 214 次。
40 秒持续输入，KV128 / KV256 分别为 **1.0671 / 308.48 ms** 与 **1.0509 / 309.53 ms**，积压 3.22 / 2.90 秒，
停止响应 9.29 / 33.13 ms。两轮均无错误/淘汰/重载失败、final 唯一、delta 正确，直接 ANE 与原缓存预算满足。
KV128 有 379.09 ms 热态系统编译，全部计入；不同处理跨度的累计阶段数据不作为等量工作提速。
连续输入尚未确认 KV128 优势，完整集继续用 row64 / KV256。一轮 96 条为 **0.8126 / 187.45 ms / CER 3.1052%**，
相对 Metal 的 CER 增加 0.0706 个百分点，96 条转写与共享权重版相同，精度与直接 ANE 检查通过。
与归档控制相比，总耗时降低 0.97%、p95 降低 1.15%；均为单轮，控制非同期复测。
规划 19.79 → 14.21 秒，evaluate 几乎相同，加载路径 25.71 → 29.04 秒。热态无新图/淘汰，8874 次重载且无失败。
52 / 96 条通过 RTF，10 / 96 条通过 p95，性能未通过；三轮完整重复与 30 分钟验收仍待完成。
该阶段完整集构建为 `inline-ir-bin`；源码/二进制/日志保存在 `artifacts/burn-ane-goal/inline-ir-*`。
工作树另在验证可选程序槽策略：74 项后端测试、两组 Clippy、3 项评测脚本测试、20 份模型数组位对照通过。
同一二进制各一轮 40 秒探针，LRU 的 RTF / p95 为 1.0112 / 251.05 ms，原策略为 1.0509 / 303.63 ms；
LRU 短测更慢，KV128 也未进一步改善连续表现，默认仍为原策略。新策略完整集和 30 分钟结果尚缺。
见[策略筛选与证据](docs/ane-optimization-goal.md#程序槽替换策略可选候选默认未改)。

后续 RMSNorm 附加能量通道候选通过 75 项后端测试、两组 Clippy 和新增 48 组数值案例。
完整模型 encoder、prefill 与后续 8 token 的原 5% 数值检查通过，最高分 token 与 Metal 相同，推理无回退。
但完整短音频转写变为“之前有顾客自己带酒水解”，且 RTF 2.1043 / p95 422.00 ms，已否决全批次 L2 版本。
后续只对最多 32 个向量使用 L2 的候选通过 56 组数值案例、75 项后端测试、两组 Clippy、完整模型数值检查及短音频转写。
三个交替 A/B 短测汇总为候选 0.8410 / 183.75 ms，控制 0.8502 / 185.24 ms；收益集中在其中一轮。
LRU 连续各一轮为候选 1.0292 / 256.05 ms，控制 1.0582 / 277.49 ms，共同音频段正文相同，但未超过较早的 LRU 控制。
候选完整 96 条集为 0.8139 / 186.85 ms / CER 3.0346%（43 / 1417），精度与 Metal 相同，但没有稳定性能收益。
只改变一条正文，去掉原先“中新闻网”中多出的“新”。47 条通过 RTF、4 条通过 p95；保留版为 52 / 10。
小批次算术改动也已归档撤回，该阶段保留结果属于冻结的 IR 遍历版，未完成 30 分钟验收。
见[归一化候选](docs/ane-optimization-goal.md#rmsnorm-附加能量通道)。

### 2026-10-10：二进制图键与编号处理（后续候选）

内存缓存键改用 bincode 编码相同的规范化 IR 身份，编号与名称一同生成，排序借用字符串。
71 项后端测试、后端 Clippy 与使用原四项豁免的根 Clippy 通过；原生桥接未变，沿用 192 组缓冲验证。
row64 的完整 encoder、prefill、后续 8 token 的 20 份数组与共享权重版逐字节相同，原 5% 数值门槛通过。
三轮候选短测为 **0.8635 / 186.83 ms**，同期控制为 **0.8711 / 190.76 ms**，规划耗时平均从 283.70 降至 231.07 ms。
同期 40 秒持续输入候选为 **1.1000 / 323.74 ms**，控制为 **1.1012 / 322.77 ms**，均积压 4.18 秒，整体接近。
候选另发生 281.02 ms 的系统编译，全部计入指标；两版提交事件正文相同，停止响应 1.25 / 12.03 ms，均无错误、淘汰或重载失败。
本阶段单独构建未运行完整集 CER 或 30 分钟验收；与 IR 遍历组合后的完整集结果见上节，冻结构建为 `inline-ir-bin`。
证据保存在 `artifacts/burn-ane-goal/binary-plan-*`；[详细验收](docs/ane-optimization-goal.md)。

### 2026-10-10：共享图描述的不可变权重（上一完整集对照）

不同 row/KV 形状共享相同内容的 NSData 权重，图预算按每图元数据加唯一权重计费，原预算和 112 个程序槽不变。
71 项后端测试、192 组原生缓冲区检查、后端 Clippy 和使用原四项豁免的根 Clippy 通过。
encoder、prefill 与后续 8 token 的 20 份模型数组与 host-layout 逐字节相同，原 5% 数值门槛通过。

row64 / KV256 / 单 token 融合，6.74 秒短探针三次汇总为 **0.8556 / 188.02 ms**，正文一致，直接 ANE 检查通过。
40 秒持续输入报告音频 36.32 秒，积压 **3.54 秒**、RTF **1.0774**、p95 **305.06 ms**、停止响应 **22.44 ms**，
RSS 采样峰值 **23.02 GB**。热态 3 次新图、0 次淘汰、1506 次重载且无失败；一个 final、delta 正确，无错误或 token 预算事件。
381 张图共享前为 22.94 GB，共享后计费 8.26 GB，打包缓存 4.08 GB。它们是源数据字节，不是 RSS 上限。
旧 row64 的 407 次新图 / 451 次淘汰已消除，持续实时性能仍未达到目标。

同一候选 row128 的 40 秒探针为 1.1344 / 302.64 ms、积压 4.98 秒，RSS 峰值 **26.06 GB**；
旧 host-layout 对应为 1.1109 / 293.50 ms、4.18 秒、33.73 GB，单轮内存下降不能证明算子提速。
row32 单轮短测因程序重载增加而退步到 0.9975 / 301.87 ms，未采用该配置。
row64 完整 96 条集一轮为 **RTF 0.8206、p95 189.63 ms、CER 3.1052%**（44 / 1417），96 条转写与 host-layout 逐条相同。
相对 Metal 的 CER 劣化为 +0.0706 个百分点，精度和直接 ANE 检查通过；相对上一版，耗时降低 **5.86%**，p95 **增加 1.88%**。
热态无新图/淘汰，8874 次重载且无失败；direct evaluate 从 324.47 降至 297.36 秒，加载路径从 19.92 增至 25.71 秒。
row64 单独复跑原模型数值检查，20 份数组也与 host-layout / row128 逐字节相同。
性能未通过，三轮完整重复和 30 分钟验收未完成。冻结源码、二进制、日志和对照见
`artifacts/burn-ane-goal/shared-descriptor-*` 与 [验收文档](docs/ane-optimization-goal.md)。

### 2026-10-10：主机索引与布局复制（上一轮对照版）

Direct 允许回退时，对已有主机/原生张量执行 Select、Cat 和步长为 1 的 Slice 的位复制，减少 Metal 往返。
67 项后端测试、后端 Clippy、使用原四项豁免的根 Clippy 通过；未修改已有 192 组验证的原生桥接。
完整模型 20 份数组与原生输出保留版逐字节相同，原数值门槛通过。布局复制仍算回退，主要矩阵计算仍为直接 ANE。
row128 / KV256 / 单 token 融合，6.74 秒短探针三次为 **RTF 0.9082 / p95 188.22 ms**，正文一致。
40 秒持续输入处理到 35.68 秒，积压 **4.18 秒**、RTF **1.1109**、p95 **293.50 ms**、停止响应 **33.63 ms**；
一个 final、delta 正确、无错误或 token 预算事件。RSS 采样峰值 **33.73 GB**、期末 **33.27 GB**，比上一版高。
完整 96 条集一轮为 **RTF 0.8716、p95 186.13 ms、CER 3.1052%**（44 / 1417），
比上一版 RTF 降低 3.46%、p95 降低 3.78%；96 条转写逐条相同，CER 相对 Metal +0.0706 个百分点。
精度和直接 ANE 检查通过，性能未通过。热态 0 次新建图/淘汰、6372 次实例重载且无失败；
输入准备从 22.77 降至 6.15 秒，直接 evaluate 仍约 324.47 秒。三轮完整重复和 30 分钟验收尚未完成。
[完整集对照](artifacts/burn-ane-goal/host-layout-corpus-comparison.json)。
row64 的短探针为 0.8596 / 186.93 ms，但连续输入热态 407 次新建图、451 次淘汰，积压 21.62 秒、
p95 2667.19 ms、停止响应 2220.54 ms，因此保留 row128 和原缓存预算。
源码、二进制、测试、逐字节对照与测量保存在 `artifacts/burn-ane-goal/host-layout-*`；
[详细验收](docs/ane-optimization-goal.md)。

随后验证了两项实验并保留独立源码/二进制：同输入的宽投影合并通过 71 项后端测试和 20 份模型位对照，
16,384 通道上限三轮短探针为 **0.9118 / 188.26 ms**，无确认收益；12,288 上限的 40 秒连续输入
为 **1.1170 / 291.80 ms**、积压 **4.34 秒**，采样 RSS 峰值 **23.27 GB**，仅一轮。
固定 64 行的独立投影块通过 75 项后端测试和 20 份模型位对照，但三轮短探针退步为 **0.9590 / 214.71 ms**。
两项实验均未并入当前源码，没有完整 CER 或 30 分钟结果。验证共享权重前曾逐文件恢复并校验 42 份源码及 4 个 release 二进制
与 `host-layout` 控制一致；见 [实验细节](docs/ane-backend-research.md)。

### 2026-10-10：FP16 输出保留原生 surface 视图

Direct FP16 输出按逻辑形状和有效补齐区段跨图保存，仅在公开读取、回退和设备转移时物化主机张量。
按引用计数复用输出缓冲；旧分支和 KV 不会被下次预测覆盖，图缓存清除后仍可读取。图之间仍执行复制。
61 项完整后端测试、192 组原生缓冲区用例、后端 Clippy 通过；根 Clippy 使用原四项豁免。
完整模型 20 份数组与传输合并版逐字节相同，原 5% 数值门槛与推理回退检查通过。

160 ms 分块/前瞻、row128 / KV256 / 单 token 融合：6.74 秒 WAV 三次独立热态探针为
**RTF 0.9290、p95 191.83 ms**，较上一版三轮降低 3.85% / 3.74%，正文一致，性能尚未达标。
40 秒持续输入探针报告音频 34.72 秒、积压 **5.14 秒**、RTF **1.1425**、p95 **300.45 ms**，
停止响应 **50.13 ms**；一个 final、delta 正确、无错误或 token 预算事件。
RSS 采样峰值 **27.03 GB**（上一版 22.11 GB），序列化图/权重仍在原 16 GiB / 4 GiB 预算内，加载 112 个程序。
测量期间 3 次新建图、0 次淘汰、1055 次实例重载且无失败；实际处理跨度与上一版不同。
完整 96 条中文集一轮为 **RTF 0.9028 / p95 193.44 ms**，较传输合并版降低 4.28% / 4.19%。
96 条转写逐条相同；CER **3.1052%**，比 Metal 高 **0.0706 个百分点**，精度和直接 ANE 检查通过，性能未通过。
热态 0 次新建图/淘汰、6372 次实例重载且无失败。三轮完整重复和 30 分钟验收尚未完成。新增物化计时与输入/输出整理等阶段可能重叠，不能直接相加。
源码、二进制、测试、逐字节对照与探针保存在 `artifacts/burn-ane-goal/native-resident-*`；
[详细验收及复现](docs/ane-optimization-goal.md)。

### 2026-10-10：合并逻辑布局转换与 IOSurface 传输

去掉 direct 路径每次调用的输入/输出重排临时缓冲，直接按 IOSurface 步长复制逻辑 FP16 数据。
26 项 direct 回归、27 组独立原生传输测试通过；后者检查真实物理布局、NaN 位模式、负零、奇数步长与边界。
后端 Clippy 和 native `-Wall -Wextra -Werror` 通过，根 Clippy 使用既有四项豁免。
encoder、prefill、后续 8 token 的 20 份 ANE/Metal 数组与 `residency` 版本逐字节相同。

row128 / KV256 / 单 token 融合，同一 6.74 秒 WAV 三次独立进程热态汇总为
**RTF 0.9662、p95 199.27 ms**，转写与直接 ANE 检查通过，性能门槛仍未通过。
缓存保留版单次为 1.0058 / 207.47 ms。40 秒持续输入的新探针处理到 32.64 秒、积压 7.22 秒，
RTF 1.2153、p95 323.93 ms、停止响应 27.57 ms；采样 RSS 峰值约 22.11 GB，仍未实时。
本版完整 96 条中文集已完成一轮：**RTF 0.9432、p95 201.90 ms、CER 3.1052%**（44 / 1417）。
原始转写与缓存版 96 条逐条相同；相对 Metal 为 +0.0706 个百分点，CER / 直接 ANE 检查通过。
完整集耗时比缓存版降低 2.51%、p95 降低 2.58%，性能仍未达标。热态 0 次新建图/描述淘汰，
6372 次实例重载且无失败；prediction 从 376.20 降到 364.37 秒。
证据：[完整集对照](artifacts/burn-ane-goal/logical-io-corpus-comparison.json)。三轮重复和 30 分钟验收尚未完成。
native 复制计时现在包含布局转换，跨版本应比较整体 `prediction_ms` 和完整步骤耗时。
源码、二进制、位对照及日志保存在 `artifacts/burn-ane-goal/logical-io-*`，
原生缓冲区测试命令见 [验收文档](docs/ane-optimization-goal.md)。

此前 SDPA 候选虽通过 58 项后端测试和完整模型数值门槛，短探针三次为
1.0072 / 208.05 ms，未观察到可确认收益；主线恢复原有分解注意力。
该实验的真实 decoder 图均通过重试回到直接 ANE 分解表达，不能宣称已用上融合 SDPA。
实验保留在 `native-sdpa-*`，未做完整 CER 或 30 分钟验收。

扩大每图常量上限到 512 / 1024 MiB 通过模型数值门槛，但单次短探针为
0.9762 / 198.96 ms、0.9917 / 204.54 ms，未采用。纯 LRU 卸载程序的短探针为 0.9723 / 199.52 ms；
40 秒探针积压降低到 6.58 秒，但采样 RSS 峰值增到 29.01 GB，暂保留原先优先卸载小图的策略。
源码已恢复并逐文件校验与 `graph-fusion-source-sha256.json` 一致，当前使用默认 256 MiB / 512 操作。
详细对照及适用范围见 [优化验收](docs/ane-optimization-goal.md)。

### 2026-10-10：编译图缓存与已加载程序分别管理

ANE 编译描述与 I/O 缓冲可以保留在受字节/数量限制的缓存中；为了释放程序槽，
先卸载并销毁模型实例，重用时再从同一描述加载。仍使用 16 GiB 编译图、4 GiB 打包权重、112 个已加载程序的评测预算。
`cached_graphs` 可大于 `loaded_direct_graphs`，不能再把两者混为同一计数。
新增重载、卸载、重载失败和耗时计数；热态 `compile_load_ms` 与 RTF 包含这些工作。

仅调用 `unload` 的首版通过小图，却在 140 程序压力测试失败；失败记录为
`artifacts/burn-ane-goal/residency-backend-tests.log`，对应 native 文件保留为 `residency-unload-only-native.m`。
修正为释放实例后，资源压力测试和新加的 2 槽/4 图循环重载测试通过。
**49 项后端测试**（5 单元、27 通用/Core ML、17 direct）、后端 Clippy 与 native Clang 严格告警检查通过。
根项目 Clippy 在原有四项风格告警豁免下通过。
完整模型 20 份输出与保留版逐字节相同，包含 encoder、prefill、8 个缓存 decoder 步骤；推理无新增回退。
当前 release、源码与指纹保存在 `artifacts/burn-ane-goal/residency-*`。
短探针 row128 / KV256 单次热态 RTF **1.0058**、p95 **207.47 ms**，加载路径从保留版首轮
806.55 ms 降为 247.15 ms；0 次新建图、90 次实例重载、0 次失败，文本和直接 ANE 检查通过。
row64 单次为 0.9736 / 214.85 ms；row32 因字节预算淘汰产生 221 次图重建，退步到 2.7583 / 2651.38 ms，未采用。

相同 row128 / KV256 的 40 秒持续到达输入探针，报告音频 **32.00 秒**、积压 **7.86 秒**，
RTF **1.2399**、p95 **324.55 ms**，停止响应 **12.87 ms**。测量期间 0 次图重建/缓存淘汰、
949 次实例重载且无失败，加载路径 2.98 秒、直接 evaluate 25.77 秒。
与保留版处理音频 18.72 秒、积压 21.14 秒相比有改善，但未实时，且两次处理跨度不同。
314 张缓存图中加载 112 个程序；模型描述约 16.53 GB、权重缓存约 4.08 GB，符合原预算。
RSS 抽样峰值约 26.00 GB（保留版约 9.28 GB），新增描述保留有实际内存成本。
完整 96 条中文集已完成一轮：RTF **0.9675**、p95 **207.26 ms**、CER **3.1052%**（44 / 1417），
与拼接边界版 96 条原始转写逐条相同；相对 Metal 为 +0.0706 个百分点，CER 和直接 ANE 检查通过。
相对拼接边界版的完整集耗时降低 8.83%、p95 降低 8.30%；性能门槛仍未通过。
完整集热态 0 次图新建/描述淘汰、6372 次实例重载、0 次重载失败；加载路径 17.66 秒，直接 evaluate 321.31 秒。
证据：[完整集结果](artifacts/burn-ane-goal/residency-corpus-v1-r1/summary.json)。仅适用于该保存版本，不代表后续注意力实验。
30 分钟连续流式仍未验收，目标未完成。

### 2026-10-10：减少主机复制、token 拼接边界与停止检查（保留对照）

FP16 输入/输出布局转换直接写入目标缓冲，同类型主机输入借用读取，KV 补齐按连续区段复制。
只改复制路径时，20 份完整模型 ANE/Metal 输出与 sigmoid 修复版逐字节相同。
输入 token 拼接前结束已有投影图，防止可变行数进入后续大权重图；硬件回归覆盖多 batch、
不同有效行数、数值与图复用，旧版在缓存断言处失败，新版通过。strict/Core ML 路径保持原边界。

固定 96 条中文集、494.52175 秒、每条先预热再测量一次，160 ms 分块/前瞻、row128 / KV256 / 单 token 融合：

| 版本 | 汇总 RTF | 所有步骤 p95 | CER |
| --- | ---: | ---: | ---: |
| Metal FP16 对照 | 0.3417 | 78.73 ms | 3.0346% |
| sigmoid 修复后保留版 | 1.1952 | 314.68 ms | 3.1052% |
| 当前 ANE | 1.0613 | 226.03 ms | 3.1052% |

当前版 96 条归一化转写与保留版一致，CER 与直接 ANE 门槛通过，性能仍未通过。
图加载路径累计从 92.94 秒降至 55.79 秒；一次完整集结果不等于三次重复验证。
短探针有退步：6.74 秒 WAV 的三次汇总从 1.0766 / 222.43 ms 变为 1.1148 / 227.08 ms。
中间的仅减少复制版本单次为 1.0325 / 211.54 ms，保留此结果而不只报告改善的输入。

当前后端 **47 项测试**（4 单元、27 通用/Core ML、16 direct）、后端 all-targets Clippy、fmt 通过。
根项目 7 项 CPU 模型测试、3 项麦克风队列停止测试、3 项 Python 评测测试通过。
根项目 Clippy 仍仅在豁免下节四类既有告警后通过。release 构建了 r2t2、ane_blocks、stream_eval、stream_soak。
源码快照、二进制与 SHA256 记录在 `artifacts/burn-ane-goal/concat-boundary-*`。

CLI 麦克风直接共享 Session stop handle，停止后丢弃排队 PCM 和重采样尾部，不调用 finish 追加生成。
通过预录 PCM 分别验证计算前、首步计算中和 OS SIGINT 停止，Session 响应为 **19.97 / 42.04 / 16.52 ms**；
均只发一个最终事件，不追加 delta 或 token，文本由全部已提交 delta 还原。
证据：[停止结果](artifacts/burn-ane-goal/concat-stop/summary.json)。本轮未打开物理麦克风验证驱动。

40 秒持续到达输入、32 秒连续路径预热后的探针已跨过 16 秒窗口边界。
停止时报告音频 18.72 秒、积压 21.14 秒，RTF **2.1221**、p95 **938.70 ms**，停止响应 **49.82 ms**。
无错误或 token 预算事件，缓存处于原预算内，但远未持续实时。
585 次热态加载均命中系统编译缓存，加载路径耗时 17.42 秒，直接 evaluate 14.54 秒。
这是计数扣除预热后的测量增量。[原始记录](artifacts/burn-ane-goal/concat-rolling-smoke.jsonl)。
相同二进制改为 KV512 的单次 40 秒探针，报告音频增至 25.76 秒、积压降至 14.10 秒，
RTF 1.5419、p95 354.32 ms，停止响应 36.77 ms；仍未实时，且未做 KV512 完整集 CER。
此对照的截止音频位置不同，只能比较同等墙钟时长下的积压，不能直接比较等量音频的阶段计数。
[KV512 原始记录](artifacts/burn-ane-goal/concat-kv512-rolling-smoke.jsonl)。
**30 分钟测试尚未运行，优化目标未完成。**

### 2026-10-10：规划开销、单 token 融合与 sigmoid 数值修复（保留对照）

验收标准固定为 M5 Max、FP16、160 ms 分块/前瞻、热态 RTF ≤ 0.8、单步 p95 ≤ 160 ms，
并约束中文 CER、直接 ANE 执行、缓存预算、30 分钟连续流式和手动停止。
完整协议与复现命令见 [docs/ane-optimization-goal.md](docs/ane-optimization-goal.md)。

类型化 IR 替代图规划中的 JSON Value 往返，短音频规划时间从约 938 ms 降至 274 ms，
控制数组逐字节相同。实验单 token 融合在 KV256 下减少程序重载，默认仍关闭；
单行 row padding 实测更慢，未采用为默认。

定位旧第 3 个缓存 token 的 5.271% logits 相对 RMSE：直接 native sigmoid 在实际 SiLU
输入上的自身误差约 2.9%。改为有界 `exp(-abs(x))` 分支表达后，同一输入数学对照约 0.051%，
完整模型原失败步骤降至 **0.197%**。原 5% 阈值保留，未加 trace 的 Metal 参考逐字节不变。
Prefill 与 8 个缓存步骤均通过，decoder 误差全低于 1%、最高分 token 全部一致、无新增推理回退。
Encoder 为 1.239%。融合与分离配置的 decoder 数组逐字节相同。这不是长上下文或语料 CER 结论。

该阶段保留的 release，6.74 秒 WAV，每种配置三次独立进程、各完整预热一次：

| 160 ms 分块 / 160 ms 前瞻 | 汇总 RTF | 所有步骤 p95 | 性能门槛 |
| --- | ---: | ---: | --- |
| row128 / KV32，分离配置 | 1.1051 | 230.69 ms | 未通过 |
| row128 / KV256，单 token 融合 | 1.0766 | 222.43 ms | 未通过 |

文本均为 `之前有顾客自己带酒水也没加收钱或者不让喝。`，直接 ANE 检查均通过。
下文更早的 160/320 ms 结果使用修复前 sigmoid，不能当作当前版本速度。

后端 44 项测试通过（2 单元、27 通用/Core ML、15 direct），CPU 模型测试 7 项、Python
评测测试 3 项、fmt 与后端 all-targets Clippy `-D warnings` 通过。根项目严格 Clippy 有既有
风格告警；仅豁免 `manual_is_multiple_of`、`collapsible_if`、`single_range_in_vec_init`、
`chunks_exact_to_as_chunks` 四类后通过，不声称无条件通过。

固定中文集已保存 96 条、494.52175 秒、1417 字符和来源/音频 SHA256；三条 smoke 的
47 字符上，Metal 与 ANE 的 CER 都为 0，但 ANE RTF 1.0291、p95 221.40 ms 未达性能门槛。
随后保留版本完成全部 96 条的一轮对照：Metal 为 43/1417 错误、CER 3.0346%；
ANE 为 44/1417、CER 3.1052%，差 **+0.0706 个百分点**，准确率门槛通过。
完整集 ANE RTF **1.1952**、p95 **314.68 ms**，实时性能门槛未通过；直接 ANE 检查通过。
Metal 同集 RTF 0.3417、p95 78.73 ms。部分 8–12 秒片段热态加载耗时仍达 6.8–9.2 秒。
这是一轮完整集对照；后续复制优化与停止结果见上一节，30 分钟流式尚待验证。原始日志、数组、二进制与指纹保留在
`artifacts/burn-ane-goal/`，[summary.json](artifacts/burn-ane-goal/summary.json) 明确标记目标未完成。

### 2026-10-10：FP16 图间传输（历史）

直接运行时的主机图缓冲、补齐/裁剪、布局转换和原生桥接统一使用 FP16，按 IOSurface
行步幅复制 16 位数据。二维/三维转置改为分块执行，单行跳过转置。没有降低计算精度；
IOSurface 原本就是 FP16，减少的是主机侧的 F32 中间数据、转换及复制。
Core ML 图接口继续使用 F32，张量的公开类型和回退语义保留。

**43 项后端测试通过**：原有 42 项，加一项 FP16 跨图/回退测试，覆盖 batch=2、
1/7/32/33 行、实际主机存储类型、传输字节、两张图连续执行和转回 F32 后的 Sin 回退。
Burn Router 的公开 `into_data()` 会转换到 `FloatElem`，因此测试使用 RunnerClient
检查内部 FP16 存储。Clippy（`-D warnings`）、两处 fmt、原生 Clang 告警检查和 release
构建通过。日志位于 `artifacts/burn-ane-fp16/`。

完整 FP16 检查点复测：encoder、prefill 和两个带 KV decoder 步骤的 ANE/Metal 输出，
共 8 份数组，与上一轮保存的数组**逐字节一致**。相对 RMSE 仍分别为 1.239%、2.826%、
3.311%、2.513%，decoder 最高分 token 一致，四阶段无新增推理回退。第二个缓存步骤
42.93 ms，62 次图命中、0 次新图加载或权重回读。更长任意 token 序列的既有误差失败
记录仍有效；本次没有修改计算图数值表达或放宽 5% 阈值，也不表示完整语料准确率已验证。

同一台 M5 Max、同一段 6.74 秒 WAV、160 ms 前瞻，每次独立进程先完整预热一遍。
修改前的 release 可执行文件已保留并重新测量；测试顺序执行，没有并行构建或其他基准。
两边均用 16 GiB 图预算、112 个加载程序、4 GiB 打包权重、row bucket 128、KV bucket 32。

| 分块 | 修改前耗时 / RTF | FP16 通路耗时 / RTF | 耗时减少 | 每步 p95，前 → 后 |
| --- | ---: | ---: | ---: | ---: |
| 160 ms | 10.501 s / 1.558 | **7.867 s / 1.167** | 25.1% | 315.6 → 243.2 ms |
| 320 ms | 5.283 s / 0.784 | **4.100 s / 0.608** | 22.4% | 316.0 → 264.5 ms |

两组修改前后的所有事件，除 `compute_ms` 外完全相同，分别为 42/21 个事件；最终文本为
`之前有顾客自己带酒水也没加收钱或者不让喝。`，delta 拼接一致，没有 token 上限事件。
这里只测了一段音频的热态吞吐；160 ms 仍未实时，320 ms 在该样本上实时。模型加载和
首次图准备不在计时内，未测长时间麦克风/CER/WER。原生界面默认仍为 Metal。

测量阶段的计数先减去预热；图执行数、缓存命中、重加载和回退次数与修改前完全相同：

| 指标 | 160 ms，前 → 后 | 320 ms，前 → 后 |
| --- | ---: | ---: |
| 图接口输入 + 输出 | 13.276 → **6.638 GB** | 7.007 → **3.504 GB** |
| 主机输入准备 + 输出写回 | 1569.0 → **649.4 ms** | 824.5 → **349.3 ms** |
| 完整预测调用 | 6780.8 → **5236.6 ms** | 3544.2 → **2859.8 ms** |
| 编译/加载路径 | 660.4 → 626.8 ms | 164.0 → 162.0 ms |
| 直接 ANE 调用 | 5770 → 5770 | 3350 → 3350 |
| 图重加载（均命中系统编译缓存） | 113 → 113 | 32 → 32 |
| Burn 回退操作 | 702 → 702 | 369 → 369 |

各次热态测量均为 0 次编译失败、失败缓存命中、资源不足重试及静态权重回读。
回退仍包含 embedding 索引、argmax 和布局操作。图接口字节不包含 IOSurface 对齐空隙，
不代表芯片侧传输量减半。新的原生计时显示，160 ms 的输入复制/evaluate/输出复制分别为
166.8/4385.7/130.6 ms，320 ms 为 91.0/2394.2/70.6 ms；这些都包含在完整预测调用中。
evaluate 包含驱动和调度，不能等同于纯 ANE 计算时间。

另外实测 row bucket 64、24 GiB 图预算、120 个程序的 160 ms 配置：虽然 evaluate
降至 4187.3 ms，但热态有 275 次图淘汰/重加载，加载路径达 9815.1 ms，转写反而耗时
17.576 s（RTF 2.608）。文本正确且无编译失败，仍因性能退步不采用。图字节预算增加
无法抵消有限程序槽下的大权重图反复加载，推荐参数维持 row bucket 128 / 112 个程序。

命令仍使用下节的 320 ms 配置；新旧二进制指纹、参数、阶段差值、数值比较及全部结果见
[summary.json](artifacts/burn-ane-fp16/summary.json)，原始文件为 `before-{160,320}`、
`after-{160,320}`、`parity-after` 和 `tuned-row64-160` 的 JSONL/log。

### 2026-10-10：不变权重复用与 KV 容量分档

直接运行时新增独立的 FP16 权重 LRU，以及默认 128 token 的 KV 容量分档。后者只改变
物理形状，逐次生成有效长度掩码，恢复逻辑输出；不增加录音或上下文长度限制。
库的权重缓存默认 2 GiB，CLI 默认 4 GiB；这是图缓存之外的预算。参数和算法见
[后端说明](crates/burn-ane/README.md#直接-ane-运行时)。

**42 项后端测试通过**：2 项缓存单元、27 项原有后端/缓存测试、13 项直接 ANE 硬件测试。
新增对照覆盖连续变长内容、缓存截断后追加、跨容量边界、多个新 token、GQA、外部/因果
掩码、attention bias、全遮挡行；并断言同档的实际图缓存命中。权重测试覆盖新建转置视图、
替换权重、LRU 淘汰、预算为零和清理。新增大权重图跨当前 token 数复用，以及运行时单 token
选择；后者用未选行 NaN 验证掩码不会污染输出，并检查词表输出未被补齐。
Clippy（`-D warnings`）、fmt 和 release 构建通过。
原始记录在 `artifacts/burn-ane-kv/`。

最终 release、完整 FP16 检查点与 Metal FP16 对照，使用 row bucket 128、KV bucket 32：

| 数值阶段 | 相对 RMSE | 最大绝对误差 | Decoder 最高分 token | 新增推理回退 |
| --- | ---: | ---: | --- | ---: |
| Encoder，32 帧，24 层 | 1.239% | 0.001053 | 不适用 | 0 |
| Decoder，8 token prefill，28 层 | 2.826% | 0.356445 | 一致，151645 | 0 |
| Decoder，第 1 个缓存 token | 3.311% | 0.398438 | 一致，151645 | 0 |
| Decoder，第 2 个缓存 token | 2.513% | 0.328125 | 一致，198 | 0 |

Decoder 使用同一份 Metal embedding 数据，隔离 encoder 误差；初始化仍有 344 个布局等
回退操作。第 1 个缓存 token 需要 30 次新图加载，但不再回读静态权重；第 2 个缓存 token
62 次全部命中，0 次新图加载、0 次权重回读，包含输出回读共 **45.95 ms**。这不是
完整 ASR 单步耗时。记录：`parity-final.{jsonl,log}`，原始张量：`parity-final-arrays/`。

完整流式测试用同一段 6.74 秒 WAV、160 ms 前瞻、完整音频预热一遍，再累计正式事件
的计算耗时。各次运行均顺序执行，没有并行构建或其他基准。ANE 参数为 16 GiB 图缓存、
112 个已加载程序上限、4 GiB 打包权重缓存、row bucket 128、KV bucket 32。

| 路径 / 分块 | 推理总耗时 | RTF | 每步中位数 | 每步 p95 | 事件数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 直接 ANE / 160 ms | 10.829 s | 1.607 | 265.1 ms | 328.7 ms | 42 |
| Metal / 160 ms | 2.265 s | 0.336 | 57.5 ms | 74.7 ms | 42 |
| 直接 ANE / 320 ms | **5.247 s** | **0.778** | 256.9 ms | 312.3 ms | 21 |
| Metal / 320 ms | 1.224 s | 0.182 | 62.6 ms | 77.9 ms | 21 |

四次均输出 `之前有顾客自己带酒水也没加收钱或者不让喝。`，delta 拼接一致且有最终事件，
没有 token 上限事件。320 ms 直接 ANE 在该样本上达到热态实时吞吐，Metal 仍快约 4.3 倍。
这不是冷启动或长期麦克风验证：ANE 两次权重加载分别约 6.76/6.50 s，另有完整预热过程，
均未计入表格。320 ms 的最大单步为 329.8 ms，不能保证每一步都低于分块时间；更大
分块也会增加采集等待。模型深度、权重精度没有降低，尚未做完整语料 CER/WER 评测。

测量阶段单独减去初始化和预热统计：

| 直接 ANE 测量阶段 | 160 ms 分块 | 320 ms 分块 |
| --- | ---: | ---: |
| 直接 ANE 调用 | 5770 | 3350 |
| 图缓存命中 | 5657 | 3318 |
| 淘汰后重新加载 / 系统编译缓存命中 | 113 / 113 | 32 / 32 |
| 编译失败 / 失败缓存命中 / 资源不足重试 | 0 / 0 / 0 | 0 / 0 / 0 |
| 静态权重回读 | 0 B | 0 B |
| 图输入 + 输出字节（当时的 F32 接口） | 13.276 GB | 7.007 GB |
| 编译/加载路径耗时 | 692.8 ms | 160.9 ms |
| 预测调用（含原生 IOSurface 搬运） | 6891.7 ms | 3565.1 ms |
| 主机输入准备 + 输出写回 | 1652.5 ms | 801.0 ms |
| Burn 回退操作 | 702 | 369 |

完整 ASR 的回退来自 embedding 索引、argmax、输入 token 拼接及无矩阵小图；不能把
数值对照中的零回退推广为整个应用全 ANE。下一项主要优化是 FP16 图间传输、减少完整 KV
复制及跨图驻留，随后再评估图内 argmax。内存统计与完整阶段差值见
[summary.json](artifacts/burn-ane-kv/summary.json)，原始文件为
`asr-final-{160,320}.{jsonl,log}` 和 `metal-final-{160,320}.{jsonl,log}`。

复现上述 320 ms 配置：

```sh
cargo run --release --features ane --bin r2t2 -- \
  --backend ane --ane-runtime direct \
  --ane-graph-cache-mib 16384 --ane-loaded-graphs 112 \
  --ane-row-bucket 128 --ane-kv-bucket 32 \
  transcribe --model models/Confucius4-R2T2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav \
  --chunk-ms 320 --lookahead-ms 160 --warmup --json
```

压力测试在 strict 模式下连续执行 140 个不同直接 ANE 模型：本机同时保留 127 个程序后
触发资源不足，13 次释放旧程序并重试全部成功，最终 140 次硬件执行、0 次编译失败、
0 条永久失败缓存、0 次 Burn 回退。默认限制 112 个存活程序，并优先淘汰较小程序，避免
仅增大字节预算触发该限制。记录：`resource-recovery-test.log`。

中间版本的 32 GiB 缓存实验触发上述资源错误，错误曾被永久缓存，热态有 132639 个
操作回退，并花费约 16.47 s 回读输入。虽然最终文本正确，其 21.98 s 转写耗时不作为
全主体 ANE 的性能结果；已修复并保留诊断于 `asr-m5-resource-failure.json`。

中间版本扩展到第 3 个缓存 token 的额外诊断，触发原有相对 RMSE < 5% 的断言：优化开启为
5.271%，两项均关闭为 5.339%，最高分 token 均与 Metal 一致。该诊断保留为失败记录，
没有放宽测试阈值，不能宣称任意序列均低于 5% 误差。文件：
`extended-parity-diagnostic.json`、`parity-release.{jsonl,log}`、`parity-disabled.{jsonl,log}`。
分档会改变 FP16 归约布局，因此也不承诺逐位相同。完整语料的 CER/WER 尚未评测。

### 2026-10-09：直接 ANE 与稳定缓存路径

`AneRuntime::Direct` / `--ane-runtime direct` 已接通通用 Burn IR → MIL → 私有 ANE 运行时。
接口、路线依据与兼容边界见 [调研文档](docs/ane-backend-research.md)。原生默认 Metal 和 WebGPU
路径不变。直接调用成功次数单独统计，不冒用 Core ML 的计划计数。

本轮独立后端 **36 项测试通过**（1 项缓存单元、27 项原有后端/缓存集成、8 项直接 ANE 集成）。
新增测试覆盖短行对齐、多输入/输出、batch、bias 广播、卷积、矩形因果/外部掩码、softcap、
全遮罩行、归一化、RoPE、GQA，以及连续 7/1/3 token 的 KV 追加。
三次 decoder 测试各执行一张完整图，strict 模式下没有 Burn 回退。
原项目 6 项单元与 7 项模型/流式回归通过；后端 Clippy、fmt、Wasm 和 ANE CLI 编译检查通过。
记录：`artifacts/burn-ane-direct/{backend-tests,cpu-regression,wasm-check,native-check}.log`。

真实完整 FP16 检查点，与 Metal FP16 对照：

| 阶段 | 相对 RMSE | 最大绝对误差 | 最高分 token | 推理阶段 Burn 回退 |
| --- | ---: | ---: | --- | ---: |
| Encoder，32 帧，24 层 | 1.239% | 0.001053 | 不适用 | 0 |
| Decoder，8 token prefill，28 层 | 2.826% | 0.356445 | 一致，151645 | 0 |
| Decoder，已有 KV 的单步 | 3.311% | 0.398438 | 一致，151645 | 0 |

Encoder 调用 6 张直接 ANE 图；decoder 两个阶段各调用 28 张。初始化时 344 个权重布局等操作
仍在 Burn 回退后端执行，表中零回退只指随后三个数值推理阶段。Decoder 使用相同 Metal embedding
输入隔离 encoder 误差；并未验证所有 token 序列，也未做整套 CER/WER 评测。
数据：[parity-final.jsonl](artifacts/burn-ane-direct/parity-final.jsonl)。

归一化实验保留了 `LayerNorm(concat(x,-x))` 的实现。小方差单独回归覆盖 1e-5、1e-4、0.01、1、1000
五种输入尺度；最大绝对误差均小于 0.01。本项目生成的“最大幅值缩放”MIL 候选在小方差下曾出现
明显误差；MIL L2Normalize 的 epsilon 在该实验中未产生预期效果。这两种候选未启用，不能据此认定
上游实现或其他设备也有相同问题。投影抵消后再做 Q/K 归一化会放大 FP16 中间舍入，KV 对照同时
约束最大绝对误差 0.06 和相对 RMSE 1.5%，整层输出另按 0.015 的绝对误差检查。

微型基准顺序运行，31 个不同输入，第一项冷态，其余 30 项取统计；NdArray 对照不计入耗时。
FFN 为 32×1024 → 4096 → 1024；Attention block 另含 16 heads、QKV/输出投影和残差。

| 微型场景 | 热态中位数 | 热态 p95 | 首次执行 |
| --- | ---: | ---: | ---: |
| FFN，直接 ANE，首次独立进程 | 0.659 ms | 0.702 ms | 128.90 ms |
| FFN，直接 ANE，系统缓存命中进程 | 0.651 ms | 0.700 ms | 47.04 ms |
| FFN，Core ML，包缓存命中 | 0.751 ms | 0.852 ms | 166.27 ms |
| Attention block，直接 ANE | 0.961 ms | 1.025 ms | 195.91 ms |
| FFN，Metal F32 对照 | 2.852 ms | 5.479 ms | 103.94 ms |
| Attention block，Metal F32 对照 | 2.967 ms | 6.509 ms | 410.48 ms |

微型用例都调用 F32 Burn Tensor API，ANE 内部按 FP16 计算，Metal 按 F32 计算；不可把这组 Metal
数据称为同精度加速倍率。直接 ANE 的两个用例均无 Burn 回退，最大相对 RMSE 小于 0.047%。
直接 ANE FFN 相对同样低精度的 Core ML 路径，热态时间减少约 13%；这是局部收益。
汇总与原始计时：[summary.json](artifacts/burn-ane-direct/summary.json)、`ffn-*.json`、`block-*.json`。

原 Core ML 包缓存改为首次发布后加载稳定路径，以共享锁保护活动包，LRU 不再淘汰正在使用的模型。
独立进程验证：修改前缓存命中的 compile/load 为 129.19 ms，修改后为 133.48 ms，其中计划检查
106.58 ms、模型加载 11.06 ms。**稳定路径修复没有在这个用例中证实明显提速**。
直接 ANE 的系统缓存命中进程则为 15.38 ms 的 native compile/load；总首步仍包含权重读取、
MIL 构建、输入输出处理，不能以 15.38 ms 当作总启动时间。

继续将权重转换改为 half 的批量 FP16 路径，并提前预留 blob 空间。相同完整模型复测中，
prefill 的 MIL/权重构建从 7.115 s 变为 6.813 s，单步从 4.864 s 变为 4.572 s；
仍然远大于约 58/55 ms 的 ANE 预测调用，不能认定瓶颈已解决，也不把两次测量的几个百分点差异
当作稳定加速承诺。三阶段误差和 greedy token 完全相同，没有新增回退，系统编译缓存仍命中。
数据：`parity-simd.jsonl`、`ffn-direct-simd.json`；更新后的 36 项后端测试仍通过。
这进一步支持先复用不变权重和固定容量 KV 图，避免每步重新构建大模型。

完整 160 ms 分块流式路径尚未达到实时。冷态诊断中，首块约 22.65 s，随后多数约 16 s/块，
有一块达到 30.06 s。变化的 KV 长度导致大图反复构建和编译/加载，系统缓存命中仍有模型构建成本。
为停止重复的低效工作，本次诊断提前中止，**没有完成整段转写**，
不报告热态 RTF 或成功转写。开始阶段还并行进行了 CPU/编译检查，所以该记录也不是严格性能对照。
数据：[asr-cold-status.json](artifacts/burn-ane-direct/asr-cold-status.json)、`asr-cold.jsonl`。
这没有给应用增加录音时长限制。KV 容量分档、图复用与 IOSurface 驻留是下一步优先项，
然后再复测完整流式延迟、取消行为与真实语料准确率。

复现新增路径：

```sh
cargo test --manifest-path crates/burn-ane/Cargo.toml -- --test-threads=1
cargo run --release --features ane --example ane_blocks -- --ane-runtime direct --case ffn --strict --steps 31
cargo run --release --features ane --example ane_blocks -- --ane-runtime direct --case block --strict --steps 31
cargo run --release --features ane --example ane_blocks -- --ane-runtime direct --case model --model models/Confucius4-R2T2
```

### 既有 Core ML 混合路径验证

- 独立后端 **27 项测试通过**（26 项集成测试 + 1 项缓存单元测试）：Linear/ReLU、Conv2d、GELU/归约/广播/布局、动态 RHS、
  分支存活张量、静态权重同时作为残差、输入及权重变化、LRU 淘汰、失败回退、严格错误、
  设备转移、废弃图和非有限标量；新增 Attention 掩码/因果对照、动态长度补齐、LayerNorm/RMSNorm
  小方差与大方差、SiLU 饱和、权重视图跨图边界、单 token 投影、缓存损坏恢复与主机驻留回归。
- 原项目 **6 项单元测试 + 7 项模型/流式回归测试通过**。
- `cargo clippy --manifest-path crates/burn-ane/Cargo.toml --all-targets -- -D warnings` 通过。
- Web/Wasm、原生 Makepad 和无 Metal 的 ANE CLI 编译检查通过。

优化前的 `ane_backend` 示例为普通 Burn `Linear(512, 512) + ReLU`，输入 `[1,128,512]`。
与 NdArray 的最大绝对误差为 **0.000586735**；连续 3 个不同输入只编译 1 次、命中缓存 2 次，
未发生 Burn 回退。3 次执行的设备计划合计：ANE 算子 9、其中矩阵乘法 3，Core ML CPU 算子 3。
首次包含 Core ML 编译和执行为 84.7 ms，随后约 1.14 ms/次。这是小图的观测，非完整 ASR 基准，
且测量时有其他构建/回归任务运行。原始数据：`artifacts/burn-ane/burn-example.json`。

优化前的 `ane_metal` 使用同一个通用后端，回退类型为 Metal。FP16 矩阵乘法、ReLU、保留布尔中间
结果及布尔回读均通过；1 次图执行包含 1 个分配到 ANE 的矩阵乘法，2 次显式类型转换由 Metal
完成。数据：`artifacts/burn-ane/metal-example.json`。判断 ANE 成功时必须有矩阵乘法或卷积
被分配到 ANE，常量或布局节点不计入此条件。Core ML 计划是调度证据，并非硬件计数器追踪。

优化前，前 5 个分块分别耗时约 48.5、46.0、45.5、46.6、45.5 秒，仅处理到 0.96 秒
音频后中止。日志：`artifacts/burn-ane/asr.{log,jsonl}`。这些时间包含编译，不能解释成 ANE
芯片相对 Metal 的算力差距。

### 大图融合与数值修复

静态投影改为 `[1,C,1,S]` 的 1×1 卷积，权重转置在编译时折叠，并与其使用者一起跨越图
切分边界。支持完整 Attention、FFN、归一化和 GELU/SiLU 子图；保守验证 token 轴后才补齐
形状，输出立即裁回原长度。动态 KV Attention 在输出处切分，避免后续固定投影重复编译。
输入和图输出可留在主机，发生 Burn 回退时才上传。缓存分别限制成功图/失败记录，磁盘缓存
带逐文件 SHA256 校验，并按系统/硬件/运行时版本隔离。

本轮还发现并修复两项实际数值错误：拆开的 FP16 方差/epsilon 在小方差归一化中放大误差，
以及展开的 Sigmoid 在大幅值输入中丢失饱和行为。现在分别识别 LayerNorm/RMSNorm 和原生
Sigmoid；压力回归覆盖尺度 0.001、1、1000 及 SiLU 的 [-20,20] 输入。

最终代码用相同 FP16 检查点和输入对照 Metal：

| 阶段 | 相对 RMSE | 最大绝对误差 | 最终 logits 的最高分 token |
| --- | ---: | ---: | --- |
| Encoder，32 帧 | 1.251% | 0.001129 | 不适用 |
| Decoder prefill，8 token | 0.528% | 0.066406 | 一致，151645 |
| Decoder 单步，已有 KV 前缀 | 0.195% | 0.050781 | 一致，151645 |

数据：`artifacts/burn-ane-optimized/parity-final.{jsonl,log}`。Decoder 使用相同 Metal embedding
数据隔离 encoder 误差。相同的两步 decoder 验证中，Core ML 输入传输由中间版本约 209.4 MB
降至 12.49 MB；不包括模型初始化和静态权重读取。不能把这当作整体提速倍率。

Encoder 执行 7 个子图；设备计划中 197 个矩阵乘法/卷积分配到 ANE，1 个批量矩阵乘法在 CPU。
仍有归一化、布局等 CPU 操作以及 Slice/RepeatDim 等 Burn 回退，**不是全 ANE**。以上是
`MLComputePlan` 的调度记录，不能替代硬件计数器或按 FLOPs 计算的占比。

中间版本完整 6.74 秒音频已输出 `之前有顾客自己带酒水也没加收钱或者不让喝。`，42 个事件
的 delta 拼接一致，未触发 token 上限。包含首次编译且与构建并行的耗时约 225.8 秒，保留在
`artifacts/burn-ane-optimized/asr-boundaries.{jsonl,log}`；它不是热态性能数据。
更早的 `asr-probe`、`decoder-v3` 记录包含已定位的数值错误，不作为正确性或性能成功结果。

磁盘命中也不代表免除设备加载开销：encoder 的 7 个编译包全部命中时，计划检查/加载仍约
23.1 秒。当前应复用存活的 `AneDevice`；不能据此声称首次启动已得到明显加速。

### 最终热态性能

所有以下性能运行均顺序执行，没有同时构建或运行其他基准。原始数据和汇总位于
`artifacts/burn-ane-optimized/benchmark-summary.json`。

FFN 微基准：32 行、宽度 1024、隐藏层 4096，归一化 + 两次投影 + GELU + 残差；每轮输入
变化，与 NdArray 比较。首轮包含图生成/加载，随后 30 轮计时包含输出回读。

| 路径 | 首轮 | 热态中位数 | 热态 p95 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| ANE，融合图 | 194.10 ms | **0.770 ms** | 0.852 ms | 0.001388 |
| ANE，`--max-ops 8` 人为切成小图 | 217.00 ms | 4.396 ms | 4.692 ms | 0.001286 |
| Metal，F32 | 146.22 ms | 1.655 ms | 2.913 ms | 0.000000119 |

同一后端内，融合 FFN 相对人为切图约快 **5.71 倍**；不是旧版本完整 ASR 的提速倍率。
融合版本 31 次执行只编译 1 次、缓存命中 30 次、没有 Burn 回退。Core ML 仍有部分 CPU
归一化/逐元素操作。微基准 Metal 为 F32，ANE 内部使用较低精度，因此表格不能当成相同精度的
芯片算力比较。原始文件：`ffn-{ane,fragmented,metal}-final.{json,log}`。

独立 Attention block 在 32 行、128 行，以及关闭补齐的 32 行三种探测中，都被 Core ML
全部分配到 CPU；基准中的 ANE 资格断言正确拒绝了这些运行，**没有计作 ANE 性能**。
记录：`block-ane-final.log`、`block-ane-128-final.log`、`block-ane-unpadded-final.log`。
这也说明真实 encoder 大图的调度结果不能推广到所有孤立 block。

完整 ASR：两边都用 FP16、160 ms 分块、160 ms 前瞻，完整音频预热一次，计时不含权重加载
和预热。最终代码两边均输出 `之前有顾客自己带酒水也没加收钱或者不让喝。`，42 个事件的
delta 拼接一致，没有 token 上限事件。

| 路径 | 音频 | 推理总耗时 | RTF | 每步中位数 | 每步 p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 通用 Burn ANE + Core ML CPU + Burn Metal 回退 | 6.74 s | **14.420 s** | **2.139** | 340.1 ms | 477.6 ms |
| Burn Metal | 6.74 s | **2.615 s** | **0.388** | 66.8 ms | 82.7 ms |

当前通用 ANE 混合实现的热态耗时约为 Metal 的 **5.51 倍**，**仍未达到实时**。
这包含编排、同步、传输和回退，不是 ANE 芯片本身慢 5.51 倍。尚需补齐 RoPE/KV 路径中的
Slice/RepeatDim、减少跨图同步，并改善小 Attention/RMSNorm 的设备分配；不应仅为“全部
ANE”而忽略真实端到端延迟。原始数据：`asr-final-warm.{jsonl,log}` 与
`metal-final-warm.{jsonl,log}`；日志末尾的 ANE 计数包含预热和测量两遍。

最终 `ane_metal` 示例中，FP16 矩阵乘法、ReLU、布尔回读通过，1 次图执行、0 次 Burn 回退，
类型转换已留在图内。记录：`metal-example-final.{json,log}`。

默认原生 UI 继续选择 Metal。重新构建后的界面回归通过：正确显示 Metal、6.74 秒样本
转写与预期完全相同、20.22 秒连续模拟麦克风、超过 17 秒后手动打断、取消后重启、输入失败
重试、Unicode 文本输入。测试没有打开物理麦克风。记录：
`artifacts/burn-ane/native-regression/native.json`。停止往返约 85.8 ms；短样本预热后
RTF 约 0.39–0.40，20.22 秒长流 RTF 约 0.50（与其他构建并行，不作为性能提升声明）。

复现：

```sh
cargo test --manifest-path crates/burn-ane/Cargo.toml
cargo test --no-default-features --features native,cpu
cargo run --release --features ane --example ane_backend
cargo run --release --features ane --example ane_metal
cargo run --release --features ane --example ane_blocks -- --case ffn --steps 31
cargo run --release --features ane --example ane_blocks -- --case ffn --steps 31 --max-ops 8
cargo run --release --features ane --example ane_blocks -- --case ffn --backend metal --steps 31
cargo run --release --features ane --example ane_blocks -- --model models/Confucius4-R2T2 --case model
cargo run --release --features ane -- --backend ane transcribe \
  --model models/Confucius4-R2T2 --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav --warmup --json
cargo check --locked --target wasm32-unknown-unknown --no-default-features --features wasm
python3 prototypes/makepad/build-native.py
python3 prototypes/makepad/validate-native.py --continuous --output artifacts/burn-ane/native-regression
```

## 历史：模型专用 ANE + Metal 实验（已退出默认流程）

2026-10-09 在上述 M5 Max 上新增并验证。通过 Core ML 编译图执行完整的 24 层音频
Transformer、末端归一化和输出投影，卷积前端及文本解码器保持 Burn Metal。
这是本模型的混合执行路径，不是通用 Burn ANE Backend。

`MLComputePlan` 在 `.cpuAndNeuralEngine` 配置下将导出子图的 **2336 个算子全部分配给 ANE**，
包含 768 个 attention einsum、146 个 1×1 conv、384 个 softmax。
Core ML 子图禁用 GPU，运行时再次检查有 ANE 分配，模型配置和完整检查点 SHA256 不匹配时拒绝加载。
原始计划：`artifacts/ane-validation/full-plan.json`，导出工具依赖记录在 `scripts/requirements-coreml.txt`。
仅子图的随机输入预测预热约 6.0–6.2 ms，不含 Burn 前端、跨后端传输或文本解码。

真实权重编码对照覆盖 1、8、32、99、100、101、240、674、800、802、1602 帧。
ANE 对 Metal 的 embedding 相对 RMSE 为 **2.32%–4.70%**；所有输出有限，完整缓存前缀逐项一致。
未完成尾窗重新计算可有约 0.06% FP16 差异，不能要求整个重新计算结果逐位相等。
记录：`artifacts/ane-validation/encoder-parity.{json,log}`。这些逐形状计时包含首次编译，不能用于宣称加速倍率。

相同 CLI 参数（160 ms、160 ms 前瞻、FP16、完整音频预热一次）的本次对照：

| 路径 | 音频 | 推理耗时 | RTF | 每步 p95 |
| --- | ---: | ---: | ---: | ---: |
| Metal | 6.74 s | 2.347 s | 0.348 | 77.6 ms |
| ANE + Metal | 6.74 s | 2.307 s | 0.342 | 73.7 ms |
| Metal | 20.22 s | 7.566 s | 0.374 | 81.6 ms |
| ANE + Metal，复测 | 20.22 s | 7.533 s | 0.373 | 80.7 ms |

其中一轮 ANE 长流出现 RTF 0.732 / p95 490.9 ms；当时有并行构建，具体原因未定位。
保留原始结果于 `artifacts/ane-validation/benchmark.json`，复测位于 `recheck-ane-long.{log,jsonl}`。
总体速度与 Metal 接近，不能据此宣称明显提速或更低能耗。所有完整转写的正文、增量拼接一致；
这是单个中文公开样本及其重复音频，未测完整 WER/CER。
ANE 首次模型载入包含系统特化，观察到 21.1 s；后续约 5.1 s。纯 Metal 约 4.2 s。

原生 Makepad 通过真实推理的界面回归：自动选择并显示 `ANE + Metal`、连续 20.22 s 模拟麦克风、
17.09 s 手动停止约 97 ms 返回、停止后重开、采集失败重试、文件错误重试、中文输入。
音源为按时钟发送的 48 kHz 双声道 WAV，未申请或验证物理麦克风权限。
报告：`artifacts/ane-validation/native-ui/native.json`。原生 UI 使用独立 target 目录和系统 Swift runtime rpath。
启用 ARM SHA 指令后的最终可执行文件另通过 5 次界面转写/停止/错误恢复测试，报告在
`artifacts/ane-validation/native-final/native.json`。不匹配的层数和权重指纹均验证为明确拒绝加载，
记录在 `artifacts/ane-validation/rejected-models.json`。新旧原生启动路径指向相同内容的可执行文件。

原有 6 个单元测试、7 个模型/缓存测试通过；根 Wasm 检查和 Makepad Web 发布构建通过。
Web 页面复查仍为 DPR 2、CSS 960×853、画布 1920×1706，COOP/COEP 隔离有效。

```sh
cargo run --release --example verify_coreml -- models/Confucius4-R2T2 \
  artifacts/Confucius4-R2T2-upstream/resources/test.wav
python3 prototypes/makepad/build-native.py
python3 prototypes/makepad/validate-native.py --continuous --expect-ane \
  --output artifacts/ane-validation/native-ui
```

## 完整权重与真实音频

使用 ModelScope 的 `netease-youdao/Confucius4-R2T2` 原始检查点，707 个参数张量，约 4.08 GB。
`model.safetensors` SHA256：
`cc4d5324d386c80f98a8a7b09fbcdcc813ad08a6503fe3a586ebb144ec4610dc`。
其他文件哈希保存在 `models/Confucius4-R2T2/download-manifest.json`。

测试音频来自 [官方仓库 test.wav](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/resources/test.wav)：
16 kHz 单声道、107840 个样本、6.74 秒，SHA256
`b703174ef013bf7332ab74d4b65d568e2981cff2633cb620252bb4ba18ea7f67`。

- 官方 PyTorch FP32 CPU 离线转写：`之前有顾客自己带酒水，也没加收钱或者不让喝。`
- Rust Metal FP16 流式转写：`之前有顾客自己带酒水也没加收钱或者不让喝。`
- 两者正文一致，流式结果少一个逗号。逐块 `delta` 拼接等于最终 `text`，没有触发 token 预算上限。

这里对照的是一条官方样本，不代表在完整语料集上的 WER/CER 已与上游等价。

## M5 性能

默认 Metal/MSL、FP16、160 ms 分块、160 ms 首块前瞻、每步最多 4 token、回退 1 token，
默认 16 秒滚动窗口。每项先用相同音频完整运行一次 `--warmup`。
RTF 为所有流式步骤及最终刷新耗时之和 / 音频时长，排除模型加载、预热和主动等待。
p95 统计每次解码步骤，包括最终刷新；它不是麦克风到字幕的端到端延迟。

| 测试 | 音频 | 推理耗时 | RTF | 每步 p95 |
| --- | ---: | ---: | ---: | ---: |
| 官方样本，缓存开启 | 6.74 s | 2.555 s | 0.379 | 82.4 ms |
| 样本连续重复 3 次，缓存开启 | 20.22 s | 8.219 s | 0.406 | 85.2 ms |
| 同一长流，`--no-cache` | 20.22 s | 12.106 s | 0.599 | 139.8 ms |
| 官方样本，`--realtime` 按时钟输入 | 6.74 s | 2.494 s | 0.370 | 81.7 ms |

这些数据证明本机、该样本、预热后的处理速度满足实时输入要求，不是对所有音频或设备的保证。
短样本模型加载约 4.16 秒；首次内核编译/调优另计。
长流覆盖了 8 秒窗口缓存和超过 16 秒后的滚动清理，最多复用 104 个音频 token。
开启/关闭缓存的 **126 个事件**，`delta` 和 `text` 逐条完全一致。

本机日志：`artifacts/metal-final.{log,jsonl}`、`metal-long-cache.{log,jsonl}`、
`metal-long-uncached.{log,jsonl}`、`metal-realtime.{log,jsonl}`。

复现短样本：

```sh
mkdir -p artifacts
curl -L https://raw.githubusercontent.com/netease-youdao/Confucius4-R2T2/26d55a54ce5670cff9947a167d8ed95d569fd4d9/resources/test.wav -o artifacts/test.wav
python3 scripts/download_model.py
cargo run --release -- transcribe --model models/Confucius4-R2T2 \
  --audio artifacts/test.wav --language Chinese --backend metal --warmup --json
# 加 --realtime 可按输入时钟模拟采集。
```

## 独立数值对照

参考代码来自 Qwen3-ASR commit `7c6daf77a2421100f5fb066495372c00129d39ff`。
使用官方模型类建立固定种子的 2 层、32 维小模型，覆盖 80 ms、1 s、1.01 s、8.02 s 的输入。
显式补上 CPU eager attention 的窗口块掩码，使其与官方 FlashAttention varlen 语义一致。
验证 mel、编码器、文本 prefill 和 3 步 KV 缓存解码。Python 3.12.15、PyTorch 2.14.1、
Transformers 4.57.6；完整环境在 `scripts/requirements-reference.txt`。

| 对照 | 编码器最大绝对误差 | logits 最大绝对误差 | 贪心 token |
| --- | ---: | ---: | --- |
| Rust CPU FP32 / 官方 FP32 | 约 1e-8 | 4.8e-7 | 16/16 一致 |
| Rust Metal FP32 / 官方 FP32 | 约 1e-8 | 3.6e-7 | 16/16 一致 |
| Rust Metal FP16 / 官方 FP16 | 1.53e-5 | 0.001343 | 16/16 一致 |

FP32 网络容差为 `2e-4`；FP16 为 `3e-3`，另要求 logits 相对 L2 误差 < 1% 且 argmax 相同。
FP16 实测最大的 logits 相对 L2 误差约 0.091%。未通过放宽容差掩盖融合误差。
mel 最大误差 `6.914e-5`、最大 RMS `2.925e-6`，容差 `1e-4`；Rust 双精度 FFT 与
`torch.stft` 单精度 FFT 在对数噪声底附近存在舍入差异。
另一个独立 STFT/Slaney 测试在 PyTorch 2.8.0 上覆盖静音、脉冲、双音、噪声和非整 hop 尾部，
5 项通过，最大误差 `1.22e-5`（容差 `2e-5`）。

已复现并绕开 Burn 0.21 Metal 的两处融合数值问题：矩阵乘法后 GELU，以及 FP16 投影、
残差和归一化的组合。GELU 改用 FP32 基础运算实现 erf 近似；残差使用 FP32 累加后转回原精度。
保留 GPU 算子融合，无中间张量 CPU 回读。FP32、FP16 对照均使用清理诊断代码后的最终路径。
参考程序设置 `R2T2_UNFUSED=1` 可以运行非融合 Metal 对照。

本机日志：`artifacts/reference-{cpu,metal-f32,metal-f16}-final.log`。

## 构建与回归

- `cargo test`：13 项通过（包括实际 Metal 融合 GELU 对照）。
- `cargo test --no-default-features --features native,cpu`：12 项通过。
- `cargo check --features microphone`：通过。
- Metal、原生 WebGPU、Wasm 的 release 构建：通过。
- Wasm 使用 `wasm-bindgen 0.2.129` 生成 JS 绑定；浏览器不需要 Python。
- `cargo fmt --all -- --check`、Python 脚本语法检查、Safetensors 分片逐字节重组和 SHA256 校验：通过。
- 回归覆盖任意 PCM 包边界、尾部刷新、完整分块边界结束、空输入、NaN、重复结束、
  FFT/重采样、卷积边界、KV 追加与回退、封闭音频窗口缓存和音量变化导致的缓存失效。

## WebGPU 与测试范围

WebGPU 提供原生 WGSL 后端和浏览器 Wasm/module worker。权重分成最大 32 MiB 的下载块，
逐块校验 SHA256、立即上传，避免整个 4 GB 检查点进入 Wasm 线性内存。
浏览器必须支持 WebGPU，FP16 还需 `shader-f16`；可选 FP32。

原生 WGSL 在本机 M5 实测完整权重识别通过：6.74 秒音频推理 10.013 秒，
预热后 RTF 1.486，p95 297.6 ms；结果与 Metal 完全一致。
日志为 `artifacts/webgpu-native-final.{log,jsonl}`。

浏览器演示页 `web/demo.html` 在 Chromium 中加载全部 707 个张量（725 个下载块），
选择官方 WAV 后连续两次识别成功，正文与 Metal 一致。两次推理分别为 19.40 秒和 19.72 秒，
第二次 RTF 2.926。完整状态和逐块文本记录在 `artifacts/webgpu-demo-final.json`。
原生 WGSL 和浏览器当前均慢于实时输入速度；跨平台后端的目标是可运行，实时路径优先使用 Metal。

默认 M5 路径使用 Metal GPU；通用 ANE 实验路径的范围与限制见上文。Windows/Linux 显卡未实测。
麦克风路径已编译，没有录制用户麦克风；设备权限和实际采集链路仍需在目标环境确认。
没有测量完整语料集 WER/CER、峰值 GPU 内存、麦克风到首字的端到端延迟或多路并发容量。
