# cube DSL 坑与 GPU 调试地雷

写或调 `#[cube]` 内核（以及 Metal 上跑 Burn/cubecl）之前先读这里。单一事实源：
新坑先进本文件，VALIDATION.md 各日期章节只留证据并指向这里。条目格式：
症状 → 原因 → 修法。

## 编译期（宏）

- **`if cond { a } else { b }` 不能当值用**：else 分支静默求值成 0。nibble 补码等
  二选一必须 branchless：`raw - i32::cast_from(raw >= 8) * 16`（cubek dequantize
  同款）。该 bug 曾给全部专家权重加 +8×scale 系统偏移，数值仍"看起来能跑"。
- **let 初始化器里的负浮点字面量与字面量间算术会崩宏**：`-1.0f32`、
  `0.0f32 - 1.0`、`f32::NEG_INFINITY` 全部报
  `f32: From<NativeExpand<f32>> is not satisfied`，且错误 span 指向整个
  `#[cube]` 属性，零局部性。修法：两行式
  `let mut x = 0.0f32; x = x - 3.4e38;`。语句位置的 `-变量`、e 记数字面量正常。
- **burn 的 Int 张量默认 I64**：`Tensor::<B,1,Int>::from_data` 须显式
  `TensorCreationOptions { dtype: Some(DType::I32) }`，否则 i32 内核读进交错垃圾。
- **`TensorData` 的 f32 数据默认上成设备 f16**：scales 等必须用显式 F32 options。
- **`#[unroll]` 字面量范围要后缀**（`0..8u32`）；`Array::new` 取 usize；
  `Array<T>` 类型由首次赋值的字面量推断。
- **签名里写字面量 `Vector<f32, 8>` 会让宏 panic**：用 `<N: Size>` 泛型 + 发射时
  传 `8`。
- **Vector×标量需 splat**：`Vector::<f32, N>::cast_from(scalar) * vec`，直接
  `scalar * vec` 不支持。
- **`SharedMemory` 索引赋值需 `let mut`**；`new` 的字面量尺寸需要 `usize` 后缀
  （写 `new(1024usize)`，否则宏报 `usize: From<i32>`）；`sync_cube()` 前后不得
  有任何分支发散（`terminate!()` 只能出现在 sync 之后的统一路径上），否则挂起。

## 数值与布局

- **`shared::Linear::weight()` 返回 `[input, output]` 可执行视图**（`from_parts`
  加载时已转置过一次）。内核按 checkpoint 式 `[output, input]` 行主序索引时，必须
  再 `.transpose().cast(F32)`（cast 会把视图物化成连续缓冲）。曾因直接传
  `weight()` 让真权重 MoE 输出相对 RMSE 0.28，而合成测试自建张量完全测不出。
- **3-D batched QFloat 与大尺寸拼接 QFloat 过 packed matmul 静默错值**：分组内核
  直读原始线性缓冲（u32 words + f32 scales），不经过 QFloat 张量。
- **测试数据生成在 usize 上做 `% 15 - 7` 会下溢**：先在 i32 里量化。参照里的
  scale 必须走 bf16 round-trip（`bf16::from_f32(...).to_f32()`），与加载器一致。

## GPU 调试地雷

- **中途 `into_data`/读回会污染后续内核**（本 burn-cubecl 构建）：一次前向之后、
  再看任何读回后的数值都不可信。多读回的诊断测试会产出看似合理的假数据（曾把
  「共享专家输出恒 0」的假象当成真 bug 查了数轮）。诊断规则：每个进程只做一次
  前向；读回放最后；优先用 seam fixture（如 `real_35b_layer0_seams`）而不是
  现场拼装的多段对比。
- **机器热态让绝对耗时不可比**（普通解码同长度两场可差 134→422 ms/token）：
  性能结论只认同态交错 A/B，见 [agent-workflow](agent-workflow.md) 的性能测量节。
- **持久 autotune 缓存重放不重验候选可启动性**：`target/autotune/<ver>/<device>/`
  下按 key+md5 直接重放缓存的 `fastest_index`，设备/feature 漂移后无效候选
  （如 `blackbox_accelerated_2_planes` 对不能整除 stage 的 `seq_q`）会
  `panic: Should run when selected by autotune`（tune/local.rs），与本次代码改动
  无关、且崩溃运行不再写缓存。缓解：删对应
  `burn_cubecl-kernel-attention-tune.json.log`（缓存可再重，新鲜调优会淘汰无效
  候选）。真修复方向：给该候选的 priority 加 `seq_q` 整除门槛（同 40f59db 的
  GEMV 补丁模式），或重放失败时降级 fallback 而非 expect。

## 性能直觉坑

- **并行的冗余工作不占墙钟**：256 个 unit 各自冗余推导同一 top-k，墙钟只算
  一遍（并行抵消）；把它"优化"成单 unit 串行执行没有任何收益，反而把工作量
  压到一条依赖链上（MoeRouter 实验：原版均衡，单线程选择版更慢）。判断冗余
  计算是否值得消除，先问它在不在关键路径上。
- **CUBECL profile 的逐内核 System 计时跨版本不可比**：批量异步提交下，profiler
  把排队排水时间记到同步落点所在的内核——重内核吸收别人的时间（gate_up 被记
  83ms/前向），轻内核记 ~0（down_total 被记到低于其带宽下限）。launch 计数仍
  精确；耗时归因用隔离微基准（每入口单独发射、尾部一次 sync，参照
  `moe_dense.rs` 的 `moe_bench_model_scale`）。
- **跨 unit 合并访存不总是赢**：router 权重转置成 [hidden, experts] 让同一
  load 指令内 256 个 unit 读连续地址，反而 +200µs/发——每 unit 顺序流式读时
  L1 行内 32 次复用，转置后每 unit 跨 1KB 步进摧毁流式预取。改访存布局前先
  用隔离基准 A/B。
- **共享 memory 的逐轮掩码要双屏障**：多 unit 冗余执行「扫描→选中→写哨兵掩
  码」时，仅写后一个 sync 不够——快 unit 的写会撞上慢 unit 的同轮读，各 unit
  选出不同结果（数值测试以"outside weights not zero"捕获）。每轮需要
  `扫描 → sync → 写 → sync`。
- **burn `module::attention` 掩码 true=排除**：与"true=可见"直觉相反（shared
  的 causal 构造把未来位置标 true）。树状 verify 首版把祖先/历史标 true，历史
  被整段屏蔽、垃圾位置全被注意——文本早发散。写显式掩码前先对齐极性。
- **内核做功减少 ≠ 墙钟下降**：verify 墙钟是发射管线与串行链的混合；不在关键
  路径上的内核（router/matmul/gate_up）内部提速会被管线重叠吞掉。只有
  GdnScan 类串行依赖链的改动直接兑现墙钟。

## 定位技巧

- **宏错误的二分必须真删代码**：`if false { ... }` 里的代码仍会被宏展开，
  包一层假分支不隔离任何东西。
- 宏报错无局部性时，按内核逐个、按语句块逐段删除后 `cargo check` 二分；
  单次 check 约 15 s（增量缓存）。
- 逐内核耗时/发射计数不用自己写：`CUBECL_DEBUG_OPTION=profile` +
  `CUBECL_DEBUG_LOG=...`（见 workflow 性能测量节）。
