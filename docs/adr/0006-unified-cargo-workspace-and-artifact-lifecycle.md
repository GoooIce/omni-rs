# 统一 Cargo Workspace、构建配置与产物生命周期

2026-11-26：本决定取代 [ADR-0005](0005-burn-dependency-boundaries.md) 中「各模型暂时保持独立 Cargo 项目」的临时条款，继续保留 ADR-0005 关于执行后端、训练与参考测试的依赖边界及按包 `cargo tree` 验收要求。

## 背景

16 个独立 Cargo 项目导致 12 个注册 `target/*` 目录重复编译存储 Burn/CubeCL/WGPU，叠加 macOS 默认 `split-debuginfo = "unpacked"` 产生 49 万余个 `.rcgu.o` 文件（290.8 GiB）、worktree 软链共享 `target/` 造成绝对路径哈希倍增、以及 117 个独立 `tests/*.rs` 集成测试 crate，使仓库总占用膨胀至 521 GiB（`target/` 占 414 GiB）。

## 决定

1. **单一 Cargo Workspace（`resolver = "3"`）**：根 [Cargo.toml](../../Cargo.toml) 纳管全部 16 个包，`default-members = ["."]`；仓库仅保留一份 `Cargo.lock`、一份 `[patch.crates-io]` 与一份根 `[profile.*]`，成员清单不再重复声明 `[workspace]`、`[patch.crates-io]` 或 `[profile.*]`。按包执行 `cargo tree --offline --locked -p <pkg>` 与 `cargo test -p <pkg>`，不通过无过滤 `--workspace` 合并各模型的生产 feature 闭包。
2. **构建配置收口与消除 `.rcgu.o` 膨胀**：根 `[profile.dev]` 统一固定 `debug = "line-tables-only"`、`split-debuginfo = "off"`、`incremental = false`、`opt-level = 1`，并对第三方依赖设置 `[profile.dev.package."*"] opt-level = 2, debug = 0`；完整调试信息隔离在 `--profile debug-full`（`debug = 2`、`split-debuginfo = "packed"`、`incremental = true`）。
3. **worktree 隔离与产物生命周期治理**：不再通过软链把临时 worktree 写入主仓 `target/`，每个 worktree 使用本地 `target/` 并在 `git worktree remove` 时一并回收；`scripts/storage_report.py` 承担只读盘点与受 `artifacts/storage-retention.json` 保护的 `--prune` 清理。
4. **集成测试收拢为每包 Harness**：各 crate 的独立 `tests/*.rs` 折叠为少量 `tests/suite/main.rs`（以及按 `required-features = ["cpu"]` 隔离的 CPU parity harness），降低链接次数与二进制体积，不改变测试过滤方式和覆盖范围。
5. **临时构建目录与持久基线分离**：`target/` 严格限定为可随时重建的临时构建产物；[ADR-0003](0003-faster-implementation-wins.md) 的 A/B 对照基线只保存剥离后的独立可执行文件与元数据于 `artifacts/baselines/`，运行时状态缓存写入 `artifacts/state/`。

## 后果

- 所有日常编译和检查共用一份根 `target/`，删除成员 `Cargo.lock` 与 `crates/*/target` 软链。
- `crates/omni-rsi` 的构建指纹统一读取根 `Cargo.toml` 与根 `Cargo.lock`。
- 已退役的 Web/WASM 残留文件（`models/web/`、`pkg/`、`prototypes/makepad/dist/`、`web/demo*` 等）随本决定一并清理，仅保留 `crates/omni-api/assets/api.html` 服务本地 HTTP 页面。
