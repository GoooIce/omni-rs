# resin-metal

Resin 引擎栈的 Rust 侧:用 [resin](https://github.com/ekzhang/resin) 的 `EngineCore`(已 vendor 在 `vendor/resin-core`,见其 `LOCAL-CHANGES.md`)驱动本仓的模型,runner 与前向全部留在 Rust/Burn。对应 [#55](https://github.com/GoooIce/omni-rs/issues/55) M0-R。

## 架构映射

| 职责 | 归属 |
| --- | --- |
| 前向、采样、DSpark 提议/验证/接受 | `Runner` 实现(`src/minicpm.rs` / `src/dspark_runner.rs`),复用 `omni-minicpm-v` 的 `TextModel`、`dspark::run_cycle` 与 `dspark_batch::run_cycle_batch` |
| 停止条件、解码循环、连续批处理与 overlap 调度 | vendored `EngineCore`(`submit` / `prepare_step` / `complete_step_with`) |
| 多序列链式 verify 合批 | `ResinConfig.multi_sequence_batch`(默认关)+ `DecodeSession::forward_batched_decode` + `DsparkAdapter::decode_batch`,一个 plan 的 decode 行走一次 `TextModel::forward_verifying_multi` |
| 对话槽检查点 | `ResinConfig.conversation_slots`(默认关)+ `conversation.rs`(每槽 3 个检查点、跨槽前缀借用) |
| SSD 状态缓存 | `ResinConfig.state_cache`(默认关)+ `state_cache.rs`(命名空间、内容链、原子写、额度 LRU) |
| KV 额度与释放次序 | `ResinConfig.kv_budget_mb`(默认 `None`)+ `memory.rs` / `session.rs`(#65 次序) |
| KV/状态 | runner 自持(按引擎 sequence id);引擎页式记账未启用(`cache_layout` 为空) |

全部机制默认关。开关、代码接缝、测量结果与限制见 [docs/kapitsa-57.md](../../docs/kapitsa-57.md)。

- **贪心 runner**:一个 `forward` 处理一个计划行——prefill 分块与单 token decode 走同一个 `TextModel::forward`,采样共用 `generate::Sampler`,因此引擎驱动的 token 流与直连 `generate` 循环按构造逐 token 一致(parity 门:`tests/minicpm_engine_parity.rs`,1B/NdArray,需本地权重)。
- **overlap 接缝**:`prepare_step()` 冻结计划 → runner 在引擎外执行(真实 runner 可在此异步入队 GPU 工作)→ `complete_step_with()` 原子收割;收割前页不可复用。
- **DSpark 块**:decode 行返回 k>1 token(speculative token rows),`cache_write_lookahead_tokens` 预留块写页——propose+verify+accept 恰为一个 `forward`;默认逐行执行,合批由 `multi_sequence_batch` 开启。

## 多序列链式 verify 合批 (#72)

同一 decode plan 的 ≥2 行共用一次目标前向:每个成员照常逐序列 propose,随后一次 `forward_verifying_multi` 验证全部块,再按 plan 顺序逐成员 accept/trim/retain。`DecodeSession` 在调用前把参与槽从 map 临时取出,成功或失败都放回,避免多槽可变借用;prefill 行始终走逐行路径。

- 开关:`ResinConfig { multi_sequence_batch: true, .. }`;实验覆盖 `RESIN_MULTI_SEQUENCE_BATCH=1`。默认关。
- 仅贪心 DSpark:非贪心请求退回逐行;`MinicpmRunner` 不覆写 `decode_batch`(贪心 runner 无合批路径)。
- 行预算:一次多序列 verify 最多 8 行(`dspark_batch::MAX_PLAN_ROWS`)。两个 block 16 链是 34 行,超出契约,runner 自动退回逐行;block 3 的两成员(各 4 行)是已测形状。
- 测量:35B int4 + DSpark、block 3、24 token,合批 43.99 ms/接受 token,同 block 直连 52.05(0.845×,未过 #63 的 0.8× 门),默认保持关。

## 对话槽检查点 (#74)

带 `DecodeRequest.slot` 的请求启用检查点(无 slot key 时行为与以前相同):release 时存轮次边界快照,prefill 结束存只增不覆的前缀快照;每槽最多 3 个(2 轮次 + 1 前缀),后来请求可取任意槽中最深的严格前缀检查点,恢复后只预填后缀。两个约束写进 `session.rs` 模块文档:

- **单块 prefill**:只有整段 prompt 在一行(`context_lengths == 0 && prefill_lengths == 行长度`)才查检查点;续块与前缀缓存命中照旧全量预填。
- **token 对齐**:引擎把最后采样 token 留在 KV 之外,检查点的 token 历史是 `history[..state.length]`,恢复时至少重放最后一个 prompt token(`tokens.len() <= prompt.len() - 1`)。

测量:1B CPU 两轮门通过(第二轮只预填后缀+问题、输出与全量重预填逐 token 相同),但第二轮 TTFT 2275.8/2831.2 = 0.804,未过 0.5× 门,默认保持关。

## SSD 状态缓存 (#75)

`conversation_slots`(#74)的内存检查点可以落盘,跨进程/重启按 token 前缀命中,冷启动只预填后缀。默认关,开关关时无任何缓存读写。

- 开关:`ResinConfig { conversation_slots: true, state_cache: true, .. }`;实验覆盖 `RESIN_STATE_CACHE=1`(仍需 `conversation_slots`)。
- 目录与额度:`RESIN_STATE_CACHE_DIR`(默认 `~/.cache/omni-rs/resin-state`)、`RESIN_STATE_CACHE_QUOTA_MB`(默认 2048,按字节 LRU 逐出)、`RESIN_STATE_CACHE_NAMESPACE`(覆盖调用方传入的命名空间)。
- 命名空间:`state_cache::namespace_for` 对模型目录(文件名+大小+mtime+`config.json` 哈希)、可选起草器目录、状态布局版本和 block 行数取哈希;调用方算好后交给 `StateCache`,runner 构造器 `MinicpmRunner::with_state_cache` / `MinicpmDsparkRunner::with_state_cache` 接收缓存实例,会话不猜路径。
- 布局:`<root>/<namespace>/<token 前缀链哈希>.bin`;magic + 格式版本 + JSON header(tokens/length/rope_delta/pending/张量与偏移)+ 原始张量字节,dtype 原样保留(F16/F32/BF16)。临时文件 + rename 原子写;损坏、截断或异命名空间条目在扫描/读取时忽略。
- 写盘时机:release 后或下一次 `forward` 前的引擎空闲边界,在 runner 线程同步 drain——编码要 `into_data()` 读回设备,故不是独立后台线程。进程退出前可调 `flush_state_cache()`(`release` 已自动 drain)。
- 冷启动门(#66):`scripts/state-cache-cold-start.sh` 起两个进程(`cold-write` / `cold-read`),要求恢复请求 0 预填 token、TTFT ≤ 全量预填 0.5×、首 token 一致;无 1B 权重时打印 `gate=skip` 退出。

## KV 内存额度与释放次序 (#76)

`ResinConfig.kv_budget_mb: Option<usize>` 给 live slot KV + 内存检查点设上限,默认 `None`(无额度,行为与 #75 相同);实验覆盖 `RESIN_KV_BUDGET_MB=<MiB>`。额度只由 runner 侧的字节账驱动,不读设备分配器。

- 字节口径:`DecodeAdapter::slot_bytes` / `checkpoint_bytes`,由适配器按张量形状 × dtype 宽度报告——贪心 runner 是目标 KV + 线性 conv/recurrent,DSpark 另加预分配的草稿 K/V 缓冲。默认实现返回 0,不 opt-in 的适配器不进入账本。
- 释放次序(#65):
  1. 本槽自弃:带 slot key 的整段 prompt 请求没有可用前缀检查点时,先释放该 key 的检查点(同时取消同内容的待写盘队列);
  2. 图像 embedding 缓存:本仓 runner 不持有,跳过并在实验头注明;
  3. 空闲槽 KV:跨 key 的 LRU 检查点;`state_cache` 开时先编码落盘再移出内存,否则直接丢弃;
  4. 待预填 KV:单块 prefill 不跨请求保留,跳过;
  5. ANE split:地图 Out of scope。
- 强制点:检查点插入后、slot 创建后、`release` 后、`set_config` 后。live slot 不淘汰;live 单独超额度时请求照常跑完,`memory_stats()` 的 `over_budget_bytes` 记录差额。每次释放打印层级与字节数(`resin memory: tier=...`)。
- 观测:`MinicpmRunner::memory_stats()` / `MinicpmDsparkRunner::memory_stats()` 返回 budget、live、checkpoint、spills、按层级的丢弃计数与 over-budget 差额。

#66 测量脚本(引擎层,1B CPU,无权重时 `gate=skip`):

- `scripts/state-cache-cold-start.sh`:#75 冷启动(两进程)。
- `scripts/state-reuse-two-turn.sh`:第二轮 prompt = 渲染后的第一轮历史 + 新问题;要求只预填后缀+问题、输出与全量重预填逐 token 相同、TTFT ≤ 0.5×。
- `scripts/state-reuse-quota-batch.sh [--budget-mb N] [--prefix N] [--suffix N]`:双请求合批在强制淘汰的额度下跑完、吞吐 ≥ 无额度 0.9×、被淘汰前缀可从 SSD 重取。
- `tests/minicpm_quota.rs`(`#[ignore]`):1B CPU 额度门。

## 运行

```sh
# 引擎语义测试(stub runner,无权重)
cargo test --manifest-path crates/resin-metal/Cargo.toml \
  --no-default-features --features cpu --target-dir target/resin-metal --lib --tests

# parity 门(1B 权重,CPU)
cargo test --manifest-path crates/resin-metal/Cargo.toml --release \
  --no-default-features --features cpu --target-dir target/resin-metal \
  --test minicpm_engine_parity -- --ignored --nocapture

# Metal 贪心示例
cargo run --manifest-path crates/resin-metal/Cargo.toml --release --features metal \
  --example greedy_engine -- models/MiniCPM-V-4.7-1B "用一句话介绍分页缓存。" 32
```

debug 构建的 NdArray 前向极慢,parity 门请用 `--release`。
