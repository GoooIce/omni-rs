# 引擎页接管投机回滚时,模型侧回滚退役

decode history(见 GLOSSARY.md)的投机回滚今天有两套并存的概念实现:

- **模型侧**(`crates/minicpm-v/src/model.rs`):`retain_accepted` / `retain_path` 用 GPU 拷贝(`copy_range` / `copy_selected`)重建接受前缀的 KV,线性层状态取接受步的快照。每个 runner/循环自己调用。
- **引擎侧**(vendored `resin-core`):页式记账(`KvLayout` / `KvGroupPlan` / `publish_prefix` / `restore_prefix`),回滚是释放未接受页的元数据操作,零 GPU 拷贝;投机块写入由 `cache_write_lookahead_tokens` 预留。

## 决定

当 #55 的 M2 给 runner 声明 `cache_layout`(引擎页记账启用)时,**投机回滚归引擎页**:runner 只声明布局并把块写入预留页,接受/释放由 `EngineCore` 的执行事务完成;模型侧 `retain_accepted` / `retain_path` 及其 `copy_range` / `copy_selected` 路径随之退役(先降级为非引擎路径专用,再在无调用方后删除)。

在此之前,两套并存是有意的:runner 尚无 cache 布局(见 `crates/resin-metal/src/dspark_runner.rs` 模块文档),引擎不做页记账,模型侧回滚是唯一实现。**不合并两套实现**——vendor 纪律承诺 resin-core 源码零改动(LOCAL-CHANGES.md),且按 ADR-0003 的精神,换实现需要它自己负载上的实测(引擎页路径的回滚从 20 发 GPU 拷贝降为 0 发,收益要在整链 A/B 里兑现,不能预设)。

## Consequences

- 未来架构评审不再把"KvCache 家族与 resin 页式缓存概念重复"当作待合并项;归属已裁决,方向是引擎页。
- `DecodeState::retain_path` 保留为非引擎路径(CLI dspark 循环)的唯一实现,直到引擎路径实测接管。
- 免权重测试的期望面:引擎页路径落地时,回滚语义经 `engine_core.rs` 的页契约测试钉死,模型侧不再需要新的回滚测试。
- 2026-10-27 已做的清理:`KvCache::append_cache`(零调用方)删除;`retain_accepted` 降级为 `retain_path` 的连续前缀特例薄包装。
