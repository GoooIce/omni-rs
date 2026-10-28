# resin-metal

Resin 引擎栈的 Rust 侧:用 [resin](https://github.com/ekzhang/resin) 的 `EngineCore`(已 vendor 在 `vendor/resin-core`,见其 `LOCAL-CHANGES.md`)驱动本仓的模型,runner 与前向全部留在 Rust/Burn。对应 [#55](https://github.com/GoooIce/omni-rs/issues/55) M0-R。

## 架构映射

| 职责 | 归属 |
| --- | --- |
| 前向、采样、DSpark 提议/验证/接受 | `Runner` 实现(`src/minicpm.rs`),复用 `omni-minicpm-v` 的 `TextModel` 与 `generate::Sampler` |
| 停止条件、解码循环、(未来)连续批处理与 overlap 调度 | vendored `EngineCore`(`submit` / `prepare_step` / `complete_step_with`) |
| KV/状态 | runner 自持(按引擎 sequence id);引擎页式记账未启用(`cache_layout` 为空) |

- **贪心 runner**:一个 `forward` 处理一个计划行——prefill 分块与单 token decode 走同一个 `TextModel::forward`,采样共用 `generate::Sampler`,因此引擎驱动的 token 流与直连 `generate` 循环按构造逐 token 一致(parity 门:`tests/minicpm_engine_parity.rs`,1B/NdArray,需本地权重)。
- **overlap 接缝**:`prepare_step()` 冻结计划 → runner 在引擎外执行(真实 runner 可在此异步入队 GPU 工作)→ `complete_step_with()` 原子收割;收割前页不可复用。
- **DSpark 块**(下一步):decode 行返回 k>1 token(speculative token rows),`cache_write_lookahead_tokens` 预留块写页——propose+verify+accept 恰为一个 `forward`。

## SSD 状态缓存 (#75)

`conversation_slots`(#74)的内存检查点可以落盘,跨进程/重启按 token 前缀命中,冷启动只预填后缀。默认关,开关关时无任何缓存读写。

- 开关:`ResinConfig { conversation_slots: true, state_cache: true, .. }`;实验覆盖 `RESIN_STATE_CACHE=1`(仍需 `conversation_slots`)。
- 目录与额度:`RESIN_STATE_CACHE_DIR`(默认 `~/.cache/omni-rs/resin-state`)、`RESIN_STATE_CACHE_QUOTA_MB`(默认 2048,按字节 LRU 逐出)、`RESIN_STATE_CACHE_NAMESPACE`(覆盖调用方传入的命名空间)。
- 命名空间:`state_cache::namespace_for` 对模型目录(文件名+大小+mtime+`config.json` 哈希)、可选起草器目录、状态布局版本和 block 行数取哈希;调用方算好后交给 `StateCache`,runner 构造器 `MinicpmRunner::with_state_cache` / `MinicpmDsparkRunner::with_state_cache` 接收缓存实例,会话不猜路径。
- 布局:`<root>/<namespace>/<token 前缀链哈希>.bin`;magic + 格式版本 + JSON header(tokens/length/rope_delta/pending/张量与偏移)+ 原始张量字节,dtype 原样保留(F16/F32/BF16)。临时文件 + rename 原子写;损坏、截断或异命名空间条目在扫描/读取时忽略。
- 写盘时机:release 后或下一次 `forward` 前的引擎空闲边界,在 runner 线程同步 drain——编码要 `into_data()` 读回设备,故不是独立后台线程。进程退出前可调 `flush_state_cache()`(`release` 已自动 drain)。
- 冷启动门(#66):`scripts/state-cache-cold-start.sh` 起两个进程(`cold-write` / `cold-read`),要求恢复请求 0 预填 token、TTFT ≤ 全量预填 0.5×、首 token 一致;无 1B 权重时打印 `gate=skip` 退出。

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
