# resin 合批负载的热机基线

2026-10-28。研究票 [#61](https://github.com/GoooIce/omni-rs/issues/61)（地图 [#57](https://github.com/GoooIce/omni-rs/issues/57)）。问题是：在本机（M5 Max，40 核 GPU、128 GB）量出 resin 合批负载的热机基线，给合批票与调度票当赢法对照。基线形状是 M0 的「双请求合批 + 前缀命中」：请求 A 先独跑，请求 B 以 A 的前缀加后缀提交，引擎恢复 A 已发布的前缀块，B 的预填只喂后缀，随后 A、B 的解码行合入同一拍前向，A 退休后 B 独跑尾段。每个回合引擎与直连在同一进程、同一热窗外交错运行，回合 0 是冷 autotune，弃用。

**结论：当前 runner 逐行 batch-1 前向，合批不摊薄每行成本。** 1B 贪心：直连每 token 19.836 ms，引擎独跑每步 20.245 ms（+2.1%），两行合批每步 40.123 ms、折每行 20.062 ms（对直连 1.01×）。35B int4 + DSpark（块 16）：直连每周期 136.0 ms，引擎独跑每周期 156.3 ms（+14.9%），两行合批每周期 308.4 ms、折每行 154.2 ms（1.13×）。两条负载上合批行的每行成本都等于或略高于单流；“两行一拍”只把墙钟翻倍，没有省下共享投影/MLP/LM head 的时间。前缀命中只省了引擎侧的预填喂入（B 只喂 16 个后缀 token，直连/单跑喂 256 个），但当前 runner 不消费引擎的缓存页，B 的模型状态只有后缀——这笔预填节省是记账口径，不是可兑现的赢面。

## 协议

- 代码：worktree 基于 `d93eb49`（main），分支 `research/resin-batch-baseline`。示例 `crates/resin-metal/examples/resin_batch_baseline.rs`，配两处 runner 访问器（`MinicpmRunner::model`、`MinicpmDsparkRunner::{model,draft}`，供同进程 A/B 复用同一份权重）与一处契约防护（DSpark 适配器在 `room == 0` 的 terminal 物理行返回喂入 token，不再空行）。
- 引擎：每回合新建 `EngineCore`，`KvGroupLayout` 16 token 块、`KvRetention::FullContext`、池 2048 块、`continuation_prefill = true`；DSpark 模式 `cache_write_lookahead_tokens = 16`（引擎每行租 `lookahead + 1` 个 token 的页，投机行最多返回 17 个）。贪心模式 lookahead 为 0。
- 形状：前缀 256 token（16 的整数倍），后缀 16 token，A 独跑 16 步后提交 B；`max_output_tokens` 两侧都固定，不设停止符，使步数对齐。
- 交错：回合 0 冷跑弃用，之后引擎/直连按回合奇偶交换先后；每回合取中位，汇总取各回合中位的中位。
- 计时：引擎每步拆 `prepare_step`（调度）、`runner.forward`（模型 + 采样回读）、`complete_step_with`（提交）；直连按 token 回调打点，DSpark 另按周期统计 `PhaseTrace`。**DSpark 直连的逐 token 间隔中位是 0.001 ms**——投机行一次回调吐多个 token，逐 token 间隔没有意义，只看 `cycle_ms = 热机总时间 / 周期数`。

命令（worktree 内 `crates/resin-metal`，`CARGO_TARGET_DIR` 指向 worktree 自己的 target）：

```sh
cargo run --release --features metal --example resin_batch_baseline -- \
    greedy <model-dir> 5 64 256 16

MINICPM_DSPARK_TRACE=1 MINICPM_DSPARK_BLOCK=16 \
cargo run --release --features metal --example resin_batch_baseline -- \
    dspark <model-dir> <draft-dir> 3 128 256 16 16
```

模型：`models/MiniCPM-V-4.7-1B`；`models/MiniCPM-V-4.7-35B-A3B-int4` + `models/MiniCPM-V-4.7-35B-A3B-DSpark`。

## 1B 贪心（5 个测量回合）

| 量 | 中位 |
|---|---:|
| 直连每 token（`direct_decode_p50_ms`） | 19.836 ms |
| 引擎独跑每步 | 20.245 ms |
| 两行合批每步 | 40.123 ms |
| 合批折每行 | 20.062 ms |
| 合批每行 / 直连 | 1.01× |
| 引擎 `prepare + complete` 每步 | 0.019 ms |
| A 预填（256 token，热） | 96.3–99.2 ms |
| B 预填（16 token，命中 256） | 20.6–21.9 ms |

逐回合（测量回合 1–5）：直连 19.887 / 19.836 / 20.256 / 19.743 / 19.620 / 21.090 ms；引擎独跑 19.535 / 19.597 / 20.245 / 20.409 / 20.609 ms；合批每步 40.084 / 40.123 / 40.578 / 40.032 / 41.795 ms。每回合 `cached=256`、`cobatch_n=47`、`tail_n=16`，A/B 各 64 token。

## 35B int4 + DSpark（3 个测量回合，块 16）

| 量 | 中位 |
|---|---:|
| 直连每周期（热机 / 周期数） | 136.0 ms |
| 引擎独跑每周期 | 156.3 ms |
| 两行合批每周期 | 308.4 ms |
| 合批折每行 | 154.2 ms |
| 合批每行 / 直连 | 1.13× |
| A 预填（256 token） | 2515–3192 ms |
| B 预填（16 token，命中 256） | 94.9–130.8 ms |

逐回合（测量回合 1–3）：直连每周期 136.0 / 156.2 / 128.3 ms；引擎独跑 159.008 / 156.317 / 125.725 ms；合批每周期 313.276 / 308.416 / 256.309 ms。引擎 `PhaseTrace` 每解码周期：propose 15.8–19.8 ms、verify 103.2–125.0 ms、accept 6.1–10.7 ms、retain 2.4–2.9 ms；直连 16.4–19.6 / 110.5–121.7 / 6.1–11.5 / 2.8–3.3 ms。两边的相位结构一致，引擎的差主要落在验证拍前后的接缝。

## 读法

1. **合批每行 ≈ 单流。** 两条负载上，两行一拍的墙钟都是一行的两倍左右，折每行时与直连/独跑持平或略高。当前 `MinicpmRunner` / `MinicpmDsparkRunner` 对计划里的每一行各跑一次 batch-1 前向，norm、投影、MLP、LM 头都没有跨行共享——这正是 #55 M1-R 注释里「多请求合批前向需要模型侧多行 verify」的量化面。
2. **前缀命中目前只到引擎记账层。** 引擎恢复了 256 个缓存 token，B 的预填只喂 16 个后缀（`fed 16`），预填时间从约 3.1 s 降到约 0.12 s。但 runner 不读 `ExecutionPlan::cache_groups`，B 的模型状态只由后缀建立，解码不是真正的「前缀 + 后缀」续写。这笔预填节省不能记为赢面，B 的接受长度也不代表真实前缀命中的表现。基线里 B 的 τ 与直连不可比（引擎聚合 τ=2.090 含 A、B 两个请求，直连 τ=2.783 是另一条提示）。
3. **引擎接缝成本。** 1B 上 +2.1%，35B DSpark 上 +14.9%/周期。后者比 #55 M1-R 的 6–10% 高，差异在测量口径：M1-R 比的是整轮墙钟（含预填），这里比的是解码周期本身；且这次的 runner 仍是「在接受读回处阻塞」的同步形状。
4. **冷 autotune。** 新二进制第一次跑时，CubeCL 的 worker 线程可能打出 `cubek-matmul ... attempt to divide by zero`；候选被淘汰后热窗干净。1B 与 35B 的最终测量回合都是 0 次 panic。回合 0 一律弃用。

## 对后续票的含义

- [模型侧多行 verify 合批值不值得做](https://github.com/GoooIce/omni-rs/issues/63)：基线证明当前合批不摊薄任何前向成本；要赢，必须让模型前向对多行共享 norm/投影/MLP/LM head。赢法可直接用本基线：同一双请求负载上，两行合批每行成本低于单流直连 19.8 ms（1B）或 136.0 ms（35B DSpark）。
- [调度机制哪些入选](https://github.com/GoooIce/omni-rs/issues/64)：`prepare + complete` 每步约 0.02 ms，调度本身不是瓶颈；轮转让位/预填充的收益只能来自避免大块阻塞（预填、读回），不能来自调度开销。
