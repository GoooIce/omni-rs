# Burn 与 CubeCL 上的投机解码

2026-10-22。问题是：有没有人在 `burn-rs`/`tracel-ai/burn` 或 `tracel-ai/cubecl`（及其 fork）上实现投机解码，包括草稿模型、校验前向、拒绝采样、树注意力、EAGLE、Medusa、DSpark、MTP。若没有完整的「起草 N 个 token、一次前向校验、回滚被拒 KV」，最近的零件是什么：K 个草稿 token 的一次前向、拒绝时的 KV 追加与回滚、分页 KV、bonus token、贪心 sample-and-match 与拒绝采样。

下面每条回到当时读到的提交和拥有该行为的文件。本文没有改本地解码器，也没有重跑这些仓库的测试。

**结论：官方 `tracel-ai/burn`、`cubecl`、`cubek` 和已归档的 `burn-lm` 没有跑完「起草、一次校验、回滚被拒 KV」的解码器。** 第三方里有人在 Burn 张量上跑完了这条循环，条件比一份通用草稿模型窄得多。`asqrzk/CombsEngine` 用提示里的 n-gram 起草，一次 `decode_all_logits` 校验，再用分页缓存的 `popn` 丢掉被拒后缀；它只在温度 0、没有惩罚、没有滑窗、缓存可分页时打开。`9prodhi/qwen3-moe-rs` 给全注意力和门控 delta 状态做了回滚原语，静态 `Add` 写入的缓存不能回滚。`erlkonig-ai/maryml` 把试探步写进缓存的克隆，拒绝时丢掉克隆。`rhi-zone/sketchpad` 只有主机上的接受函数。`tianhaoz95/snowglobe` 的草稿是填充 token，接受之后再对真缓存重跑一遍。下面第 1–4 节是官方栈和 candle 旁路；第 5 节是这些 Burn 循环。

## 检索里没有的东西

`gh search repos speculative --owner tracel-ai` 没有仓库。`gh search repos --owner tracel-ai` 当时列出的是 burn、cubecl、models、已归档的 burn-lm、burn-onnx、cubek、burn-bench 等，没有名为 cubes、也没有以投机解码为名的仓库。

代码搜索：

| 查询 | 结果 |
|---|---|
| `speculative`，`--repo tracel-ai/burn-lm` | 空 |
| `speculative`，`--repo tracel-ai/burn` | 两份写作说明里的 “Speculative generality / Speculative failures”，见下 |
| `speculative decoding OR eagle OR medusa OR dspark`，`--owner tracel-ai` | 空 |
| `rejection sampling`、`medusa`，`--owner tracel-ai` | 空 |
| `eagle`，`--owner tracel-ai` | ImageNet 标签 “bald eagle”，以及 kokoro 链接里的 `thewh1teagle` |
| `multi-token`，`--owner tracel-ai`（限流后重试） | 空 |
| `KeyValueCache`，`--repo tracel-ai/burn` | 空 |
| `paged OR kv_cache OR speculative`，`--repo tracel-ai/cubek` | 空 |
| `speculative burn`、`speculative cubecl`，`--filename Cargo.toml` | 空 |
| issues：`speculative` 于 burn 与 burn-lm；`speculative decoding` 于 `owner:tracel-ai` | 空 |

`"draft model" user:tracel-ai` 与第一次的 `"multi-token prediction" user:tracel-ai` 返回 HTTP 403，限流。后一条改成 `multi-token` 重试后为空。`draft model` 那条没有重跑成功。

`gh search code "eagle" --language Rust "burn::"` 命中的是 ImageNet 类名旁边的 `use burn::tensor`，没有 EAGLE 头。

`gh search repos "speculative decoding" --language rust --limit 30` 的描述里，把 Burn 写进仓库简介的只有 `sehaxe/burn-dspark`。`abyo-software/abyo-speculate` 提交 `2c122517ae52f53c4830682128196a8b3ad8a3b1` 的根 `Cargo.toml` 依赖 `candle-core` / `candle-nn` / `candle-transformers` 0.8，关键词含 candle 与 medusa。`moderately-ai/lmbrrr` 提交 `b7de000b4ebf4b0791909c70dc4f4d83fafe532a` 的 `Cargo.toml` 依赖 candle 的一个 git fork。`tabbyml/tabby` 提交 `21b29048d7bcf6b94f9f482f2d0fd05efadfd19f` 的根 `Cargo.toml` 工作区成员含 `crates/llama-cpp-server` 与 `crates/ollama-api-bindings`；在该文件里搜 `burn`、`cubecl`、`candle`，只有 llama 那两行。这三份清单不实现 Burn 上的投机解码。

`gh search code "dspark" --filename Cargo.toml` 还列出 pie、FeLLM、hipfire、bloomery、arle、ALICE-LLM、katgpt-rs。打开这些清单的默认分支 `Cargo.toml`（FeLLM 的 `plugins/dspark/Cargo.toml`、hipfire 的 `crates/hipfire-specdecode-dspark/Cargo.toml` 等），依赖项里没有 `burn` 或 `cubecl`。`xynexus/hipfire` 那份 dspark 清单的 description 写的是 scaffold (future work)。katgpt-rs 提交 `69c171e6e2614b459ba6c90250891ae9cb203b7e` 的 `Cargo.toml` 第 409 行唯一的 “burn” 是 “burn-in”。这些仓库不在本文的实现范围内。

CubeCL 提交 `d425faa533d9ce57e160ca470247130e901107c5` 里带 speculative 的是编译器，不是解码。`crates/cubecl-ir/src/interfaces/side_effects.rs` 的 `Speculatability` 区分一条 IR 操作能不能被推测执行。`crates/cubecl-opt/src/analyses/dataflow_solver/pre.rs` 写：CPU 用简单的 anticipated，GPU 应用 speculative availability，并标着 TODO: Actually implement that。`crates/cubecl-ir/src/interfaces/control_flow.rs` 说某个方法可能被推测调用。`crates/cubecl-environment/src/bytes/default_controller.rs` 用 realloc 推测地省掉一次对齐拷贝。Burn 提交 `f0d96d35c05f8801e653a57ff815e7894f5195dd` 的 `.github/instructions/design.instructions.md` 把 “Speculative generality” 写成一条设计禁令：配置项和扩展点要有当前调用者。

## 1. `sehaxe/burn-dspark`

提交 `7d816bcf3a13a3a6d008980d1fc7d5b595b670bd`（2026-07-30，`feat: initial release`，该仓库当时只有这一次提交）。树是 `Cargo.toml`、`README.md`、`src/lib.rs`、`LICENSE`、`.github/workflows/ci.yml`。没有第二份源文件，没有权重，没有基准。

`Cargo.toml` 依赖 `burn`、`burn-tensor`、`burn-nn` 0.21，默认特征 `std`。dev-dependency 用 `burn-ndarray`。文件里没有 `cubecl`，也没有 `burn-cubecl`。CI 是 `cargo check` 与 `cargo test --all-features`。

`src/lib.rs` 的模块说明把推理流程写成 `target_hidden → parallel_proj → draft_logits → Markov → sample → AcceptRatePredictor → verify_budget`。同文件里能调用的推理函数是 `greedy_draft`。它不调用 `AcceptRatePredictor`。仓库里没有名为 verify 的函数，没有目标模型，没有 KV。

`VanillaMarkov`：`prev_id → Embedding(rank) → Linear(vocab)`，无偏置，得到一块 logit 偏置；`apply` 把它加进调用者传入的 logits。`AcceptRatePredictor`：无偏置 `Linear(input_dim, 1)` 再 sigmoid。没有阈值，也没有按这个概率截断块长。

`greedy_draft` 的输入是已经算好的 `base_logits`，形状 `[B, N, block_size, vocab]`，外加 anchor token。函数对 `block_size` 做主机侧循环：每一步取出该位置的 base logits，把已经生成的 id（含 anchor）送进 `markov.bias`，只加上偏置的最后一行，再 `argmax`。返回值丢掉 anchor，留下 `block_size` 个 id。`bias` 的两步是逐位置的 embedding 和逐位置的 linear；加上的是前一个 id 的投影。函数注释写 “conditions on ALL previous positions”。送进 `bias` 的是整段前缀，取用的是最后一行。

`training` 特征下的 `dspark_loss` 是 logits 与一份被注释称为 one-hot 的张量之间的带掩码 MSE。`accept_rate_loss` 是预测接受概率和 0/1 标签的 MSE。测试检查形状、sigmoid 落在 0 到 1、损失有限。`greedy_draft_shape` 用全零 logits，只断言形状 `[1, 1, 4]`。

README 有一句 “60-85% faster generation vs MTP-1 in production DeepSeek-V4 serving”，并写 “Replaces MTP for both training and inference”。这次提交的树里没有计时、没有服务循环，没有 MTP 对照。这句速度是 README 的文字。

## 2. `tracel-ai/burn-lm`

提交 `a53baab057e819d107394fc2b641209a93fa74de`（2026-09-04，`Archival notice`）。README 写仓库已归档，并说会有一个新的 Burn 引擎，细节随后再公布。本次在 tracel-ai 的仓库列表里没有看到那个后续仓库。

`Cargo.lock` 把 `burn` 钉在 `git` `branch=main` 的 `3c6992d33d81c8727f13ebabc6797deca8ede2c5`（2026-06-05，`feat: add missing Device staging and memory cleanup methods (#5048)`），版本号 `0.22.0-pre.1`。`crates/burn-lm-llama/Cargo.toml` 默认依赖带 `ndarray` 的 burn；`test-cuda` 打开 `burn/cuda`、`fusion`、`autotune`，`test-wgpu` 打开 `burn/wgpu`。生成与缓存代码是 Burn 张量，不按后端分叉。

在该树里对 `specul|eagle|medusa|dspark|draft|reject|paged|bonus|rollback` 的检索，唯一命中是 `crates/burn-lm-http/src/schemas/chat_schemas.rs` 的 `accepted_prediction_tokens` 与 `rejected_prediction_tokens`。`crates/burn-lm-http/src/handlers/chat_handlers.rs` 构造 `UsageSchema::default()`。这两个字段留在默认值 0，没有赋值点。

`crates/burn-lm-llama/src/generation/generate.rs` 的 `Llama::generate`：开头 `reset`。第一次前向的 `input_pos` 是 `arange(0..prompt_len)`，整段提示一次送进 `model.forward`。logits 只切最后一行（`seq_len - 1..seq_len`）。之后 `input_pos` 收成一个位置再加一，一步一个 token。温度大于 0 时先做 `temperature_scaled_softmax`，否则把原始 logits 交给采样器。`generation/sampling.rs` 的 `Sampler::Argmax` 是 `logits.argmax(1)`。`TopP` 要求 batch 为 1，把概率拷到主机，用 `WeightedIndex` 抽一个下标。温度不大于 0 时没有强制改成 Argmax；测试 `test_llama3_2_3b_test` 用的是 `Sampler::Argmax` 和温度 0。没有第二个模型，因此也没有「输出必须等于目标自己的采样器」这道检查。没有 bonus：每步写出一个 token。

KV 在 `crates/burn-lm-llama/src/nn/attention/cache.rs`。`AutoregressiveCache` 用 `Tensor::empty` 按 `max_seq_len` 预分配。`append` 用 `slice_assign` 把新 token 写进 `cur_seq_len` 之后，再把 `cur_seq_len` 加上输入长度。`prepare` 在超出容量时按策略丢掉最老的一段：默认 `Shift` 在原缓冲里把尾部挪到开头，`Shrink` 拷进一块新缓冲。`reset` 只把 `cur_seq_len` 置 0，注释写张量不必清。没有把最新的 K 个位置裁掉的函数。

`crates/burn-lm-llama/src/nn/attention/kv_cache.rs` 的 `forward` 对 K 和 V 各 `append` 一次。`crates/burn-lm-llama/src/nn/transformer.rs` 的 `TransformerCache::prepare` 先把 `curr_seq_len` 加上本步 `seq_len`，若总和超过 `max_seq_len` 就让每层丢掉最老的差额；单次 `seq_len` 大于 `max_seq_len` 则返回 `MaxSequenceLengthExceeded`。`mask_attn` 在 `seq_len <= 1` 时返回 `None`。大于 1 时做 `tril_mask([seq_len, curr_seq_len], curr_seq_len - seq_len)`。测试 `test_transformer_cache_should_shrink` 把一次长度为 6 的 `forward_seq` 写进已经有 6 个 token、上限为 8 的缓存，长度停在 8。溢出丢掉的是前缀，不是刚写入的后缀。

`crates/burn-lm-llama/src/nn/attention/mha.rs` 的 `forward_cache` 把本步 K、V 追加进缓存后再做注意力。`seq_len > 1` 才建因果掩码；等于 1 时不加掩码，单个查询看见缓存里的全部键。`test_attention_empty_cache` 对空缓存一次送入长度 3；`test_attention_decoding` 把同一输入切成三次长度 1。两份测试比的是同一组期望值。解码测试没有调用 `PositionalEncodingState::prepare`，三次 `apply` 的起始下标都留在 0。

Burn 提交 `3c6992d33d81c8727f13ebabc6797deca8ede2c5` 的 `crates/burn-nn/src/modules/rope_encoding.rs`，`RotaryEncoding::apply` 的参数 `start` 是序列起始下标，频率表切 `start..start + seq_len`。burn-lm 的 `crates/burn-lm-llama/src/nn/pos_encoding.rs` 把 `index()` 交给 `rope.apply`。`prepare(seq_len)` 把 `next_position` 增加 `seq_len`，因此一块长度为 K 的前向拿到连续的 K 个位置。超出预计算窗口时 `rope.shift`。`reset` 调用 `rope.reset()` 并把计数清零。`test_rope_reset_after_shift` 说明 shift 之后必须靠这次完整 reset 才能回到绝对位置 0。没有按 K 回退位置的函数。

`tril_mask` 的方向在 Burn 提交 `f0d96d35c05f8801e653a57ff815e7894f5195dd` 的 `crates/burn-tensor/src/tensor/api/bool.rs`。`tril_mask` 的文档示例在 offset 0 时上三角为 `true`。`tri_mask` 对下三角用 `lower_scalar`，比较的是 `row - col + offset`。`true` 出现在 `col > row + offset`。burn-lm 把这份 `true` 交给 `mask_fill(..., NEG_INFINITY)`，offset 是 `curr_seq_len - seq_len`。这和下一节 CubeK 的右下角对齐是同一个不等式。`seq_len == 1` 时 burn-lm 直接省掉掩码；按这个不等式，单行查询会保留全部列，两件事重合。

## 3. CubeK 注意力：K 个查询，更长的 KV

Burn 提交 `f0d96d35c05f8801e653a57ff815e7894f5195dd` 的 `Cargo.toml` 把 `cubek` 钉在 `cc23e559c9022a90f0da9e4662b30d889d220f1f`。该提交相对前一版 cubek 的源码改动，是 `crates/burn-cubecl/src/kernel/attention/bounds.rs` 在 `AttentionDims` 里补上 `num_kv_heads: key.meta.shape[1]`。同提交的说明是 `feat(cubecl): use grouped-query-attention (#6127)`。

下面的 CubeK 文件都读自 `cc23e559c9022a90f0da9e4662b30d889d220f1f`。

`crates/cubek-attention/src/forward/definition/base.rs` 的 `AttentionDims` 把 `seq_q` 和 `seq_kv` 分成两个长度。K、V 的形状是 `[batch, num_kv_heads, seq_kv, …]`。注释写 GQA 的比值属于训练好的模型，KV 保持打包，核不在启动前把 KV 展开到查询头数。掩码形状写成 `[batch, seq_q, heads, seq_kv]`（`AttentionProblem` 的字段说明）。`AttentionOptions` 有一个 `causal: bool`。

`crates/cubek-attention/src/components/tile/mask.rs` 的 `LogicalTileMask::should_mask`：

> Bottom-right-aligned causality, matching burn's fallback and the KV-cache decode contract: mask where `col > row + (seq_kv - seq_q)`.

`bounds` 是 `(seq_q, seq_kv)`。没有 bounds 时只处理方阵，退化为 `col > row`。有界时的条件写成无符号比较 `pos.1 + bounds.0 > pos.0 + bounds.1`。

Burn 同提交 `crates/burn-backend/src/backend/ops/modules/attention.rs` 的 `build_causal_mask` 把 offset 设成 `seq_k - seq_q`，注释写右下角对齐，用来处理 `seq_k > seq_q`。掩码条件是 `col > row + offset`。

CubeK 测试 `crates/cubek-attention/tests/attention/forward/masking.rs` 把这件事写成两句：最后一行查询总能看见整段键；短的续写块看见整个前缀，再加上它自己的因果三角形。`causal_decode_single_query_unit` 的形状是 `(seq_q, seq_kv) = (1, 128)`。`causal_cached_prefill_unit` 是 `(4, 128)`。`causal_cached_prefill_blackbox` 是 `(16, 128)`。注释写，若改成左上角对齐，单行解码查询会被错误地限制在第一把键上。

`crates/cubek-attention/src/forward/definition/cost.rs` 的 `visited`：因果掩码丢掉 `j + seq_q > i + seq_kv` 的分数，查询 `i` 访问 `i + seq_kv - seq_q + 1` 把键。注释接着写：这不是把矩阵对半砍；`seq_q = 1` 的解码步仍然访问整个缓存。测试 `a_decode_step_costs_a_full_pass_over_the_cache` 把 `seq_q` 设为 1（其余维度来自 `seq_kv = 16` 的 `cost()`），断言因果与非因果的 `compute_ops` 相等。测试 `a_causal_mask_skips_the_scores_above_the_diagonal` 的注释和断言用的是 8 个查询对 16 把键：访问 100 个分数，方阵全访问是 128。`8 * 16 - 8 * 7 / 2 = 100`。

Burn 同提交 `crates/burn-cubecl/src/kernel/attention/base.rs` 的 `attention`：flash 例程要求加速 stage 的 `seq_q` 整除问题的 `seq_q`，tile 的 head dim 也要整除问题的 head dim。`InvalidConfig` 时退回 `attention_fallback`。注释写 fallback 是多次独立 kernel、把分数物化出来，没有这条整除约束。`flash_attention` 把调用者的 Q、K、V、可选 mask 做成 binding 交给 `launch::launch_ref`，`causal` 来自 `options.is_causal`。输出长度等于查询的 `seq_q`。核的参数里没有页表，也没有缓存游标；K、V 是这次启动的张量。

`crates/burn-flex/ARCHITECTURE.md`（同一 Burn 提交）把注意力分成两条，并写明选择理由。`seq_q * seq_kv <= 256K` 走 naive，物化整张 `[seq_q, seq_kv]`。更大则按 KV 分块做 flash，内存从 `O(seq_q * seq_kv)` 降到 `O(seq_q * TILE_KV)`。文中的 “Why two strategies” 写：基准显示 naive 在典型形状（`seq <= 512`）上快 5–10%，因为两次大的 gemm 比许多小块更摊得开调度；分数矩阵超出 L2 时 flash 更快。阈值 `NAIVE_SCORE_BUDGET` 是 256K 个元素，文中写成 f32 下 1 MB。同文写工作按 `(batch, head)` 用 Rayon 切，小工作量改串行以免调度开销。这是 flex 后端的说明，不是投机解码器，也没有把「多校验几个草稿 token」写成 GPU 上接近免费。

这些文件合在一起，能支撑一次校验前向的注意力形状：查询长度 K，KV 长度是已有前缀加 K，`causal` 打开。它们不包含草稿提案、接受检验、bonus，或 KV 的提交与回滚。

## 4. Starling：循环在 candle CPU，Burn 引擎不跑它

`gjovanov/starling` 提交 `41913822e152bcbb4824fdf47a5a5c519e28c5ac`（2026-10-21）。`apps/burn-server/Cargo.toml` 同时依赖 `burn` 0.20（特征含 wgpu）和 `cubecl` 0.9，并用可选特征拉入 candle：`candle-cpu-ggml` 在 `candle-cpu` 之上再加 `ggml-matmul`。`apps/burn-server/patches/` 下是 vendored 的 `cubecl-cuda-0.9.0`、`cubecl-wgpu-0.9.0`、`cubecl-wgpu-0.10.0-pre.3`、`burn-candle-0.20.1`。投机函数不在这些补丁里。

`apps/burn-server/src/inference/candle_cpu/model.rs` 开头写明 CPU、激活和 KV 用 F32、权重走 Q4 GGUF。`use candle_core_cpu`。`KVCache::update` 用 `Tensor::cat` 把新的 K、V 接到已有缓存后面，`seq_len` 加上新长度。`truncate_to` 的注释写：从末尾截断，保留前 `keep` 个位置，供投机解码回滚试探步。实现是 `narrow(1, 0, keep)` 再 `contiguous`。`keep >= seq_len` 时直接返回。`compact` 是另一方向：丢掉最老的一段，给编码器滑窗用。

`DecoderAttention::forward_with_cache` 在 `seq > 1` 时建掩码：键下标 `j > i + offset` 的位置填负无穷，`offset` 是追加前的 `logical_len`。`seq == 1` 不加这张掩码。

`apps/burn-server/src/inference/candle_cpu/engine.rs` 的 `speculative_decode` 在 `VOXTRAL_SPEC` 解析为大于 0 的整数时被调用。N 是本段新的 adapter token 数，不是一个独立的草稿预算。函数注释引用 Leviathan et al. 2023，并写成本：两次 batched 前向在 CPU 上各约 140 ms，合计约 280 ms；6 个 token 的逐步解码约 540 ms。这些毫秒数是注释里的字，本文没有重测。

代码做的事：

1. 记下 `kv_seq_before`。草稿前向把同一个 `prev_token` 的嵌入 `broadcast_add` 到整段音频上，一次 `decoder_forward`，再对每个位置 `argmax`。`argmax_at` 把该位置 logits `reshape` 成 131072 再在主机上取最大。
2. 每个缓存 `truncate_to(kv_seq_before)`，草稿前向写入的 KV 全部丢掉。
3. 校验前向的文本 id 是 `[prev_token, drafts[0], …, drafts[n-2]]`，再一次 `decoder_forward`，再逐位置 `argmax`。
4. 位置 0 直接采用 `verifies[0]`，注释说它和草稿用的是同一个 `prev_token`，应当相同。其后在 `drafts[k] == verifies[k]` 时继续。第一次不等，把 `verifies` 在该位置的 token 放进结果，并把接受计数加一。注释写：这个位置的 KV 用的是已经接受的前一个 token，所以是对的。
5. `truncate_to(kv_seq_before + accepted)`。
6. 若还没写满这 N 个音频位置，剩下的位置逐个 `decoder_forward`，用 `lm_head_argmax`。`lm_head_argmax` 同样是主机上对 131072 维取最大。

全部相等时接受数停在 N，不再多采一个 token。遇到 EOS 在写完已接受的 token 后返回，不做 fallback。没有温度，没有草稿概率，没有 `min(1, p_target/p_draft)`。

同一函数后部的注释写：曾在被拒绝的位置上试过 Jacobi / lookahead（batched 再校验），硬提交（接受数不超过 3，约四分之一的流量）每一轮只多一个位置，而一轮大约 180 ms，逐步一步大约 120 ms；`MAX_JACOBI=3` 的测量是 P50 多 13 ms、P90 多 70 ms、均值不变。`SpecPhases` 仍留着 `jacobi_iters` 字段，这段循环只做逐步 fallback，不跑 Jacobi。

同文件的非投机 `VOXTRAL_BATCH` 路径把同一个 `prev_token` 广播到所有位置，一次前向后逐位置 argmax，注释写这更快，但会在一次 commit 里产生重复 token。逐步路径的注释写它才是正确的自回归。

`apps/burn-server/src/inference/cuda_engine.rs` 的 `commit` 注释是 “Full re-encode + re-decode per commit (stateless, proven correct)”，调用 `transcribe_resident`。`bf16_engine.rs` 的 `commit` 调用 `transcribe`。两个文件里都没有 `speculative_decode`、草稿、校验或 KV 回滚。

`apps/burn-server/docs/pending-optimizations.md` 的 Option E “Speculative Pad-Skip” 是一段未实现的设想：用编码器输出上的小分类器预测 pad，预测为 pad 就跳过 26 层解码器。同文 “Attempted and Rejected” 记录了一次已放弃的做法：把同一个 `prev_token` 广播到 N 个位置、`chunk_size=6` 的一次前向；文中写质量垮了，因为位置 i+1 必须看见位置 i 生成的 token。Burn 的 CUDA / wgpu 引擎源码里没有这条已放弃的循环。

## 5. 在 Burn 张量上跑完校验的第三方循环

这些循环都在 Burn 张量上，都不把「多一个查询」当成和单 token 解码同一条可以随意加长的数值路径。

`asqrzk/CombsEngine` 提交 `a80a947a2bb0fb5e27a721907233fcc70d0dd2f0` 的 `engine/core/combs-runtime/src/spec.rs` 用历史里重复的 n-gram 起草，没有第二个模型。`engine/core/combs-runtime/src/step.rs` 的 `submit_speculative` 把 `[next, draft…]` 一次送进 `decode_all_logits`。注释写第 `i` 行的 argmax 就是贪心在该位置会写下的 token。接受最长相等前缀，修正取第一次不等那一行；全部相等时取第 `k` 行。`cache.popn` 只丢掉未接受的草稿。弹出的行数对不上就丢弃整个会话，不在分叉的缓存上继续。`SPEC_DRAFT` 是 6。全部拒绝后冷却若干步。打开条件写在 `begin_generation`：温度不大于 0，重复、频率、存在惩罚为空或为不起作用的值，没有 logit bias、logprobs、约束，模型实现了 `decode_all_logits`，注意力没有滑窗，缓存 `pages_used()` 有值。`engine/core/combs-models/src/traits.rs` 把 `decode_all_logits` 定义成返回 `[1, n, vocab]`，没有实现的模型进不了这条路径。

`9prodhi/qwen3-moe-rs` 提交 `c62bd60bc9c2f99fe6cd1c37aa1ad8f92f5602e9` 的 `src/cache.rs`：`KVCache::rewind` 把 `filled` 收短，注释写这只对 `slice_assign` 的静态缓冲安全；`update_static` 用 `select_assign(Add)`，回滚会在同一列上累加。`GdnStateCache::snapshot` / `restore` 克隆递推状态和卷积历史，注释写这与 `init_static` 不能一起用。这次检索没有在该仓库里找到起草和接受循环，只有这两处回滚原语。

`erlkonig-ai/maryml` 提交 `578c3b3182d155aad9c587f99c4b0dc9594e098e` 的 `src/models/inkling/mtp.rs` 把 MTP 头写成草稿。文件开头写：任何草稿都改不了正文，只改变接受率；被拒的草稿若把 K、V 留在缓存里，表现是接受率慢慢下降，而不是一条报错。`mtp_block_step` 的注释要求试探位置使用缓存的克隆，拒绝时丢掉克隆。

`rhi-zone/sketchpad` 提交 `6a7ad5391cdcff0011eaae5d3ef0016f2614c537` 的 `crates/sketchpad-core/src/speculative.rs` 是主机上的 `verify_speculative`。贪心分支用目标行的 argmax 对比草稿 token。默认起草长度 4。全部接受时，bonus 取的是最后一行，也就是已经用来核对最后一个草稿的那一行。这个文件没有模型前向，也没有 KV。

`tianhaoz95/snowglobe` 提交 `95c9aefd77ea9ddee03852683ad1a1cab9fe6cfa` 的 `engine/src/model/speculative.rs` 用填充 token 当草稿。校验写在隔离的序列上，接受之后把保留下来的 token 再 `forward_all` 一次，写回主序列。没有就地回滚。

## 和「起草 N 个、一次校验、回滚被拒 KV」相比

典型循环是：草稿模型提出 N 个 token，目标模型用一次前向在同一前缀上给出这 N 个位置的分布，按拒绝采样或贪心相等接受最长前缀，再从目标分布写出一个修正或 bonus，KV 只保留这段接受的后缀之前的缓存。上面的源码和这个循环有五处承担重量的差别。

1. **没有「草稿模型产 N 个 token，目标一次前向校验」。** `greedy_draft` 吃调用者已经准备好的并行 logits，在主机上逐步加前一个 token 的 Markov 偏置再 argmax。burn-lm 的多 token 前向用来吃整段提示，logits 只留下最后一行。CubeK 可以一次算 K 个查询对 `seq_kv = 前缀 + K` 的键，输出长度是 K，调用者自己准备 K 和 V。Starling 的两次前向是同一个 candle 解码器：草稿遍把同一个文本嵌入广播到每个音频帧，校验遍才把草稿 token 错一位嵌进去。

2. **官方 Burn 的 KV 不会回滚被拒后缀。** burn-lm 往预分配的稠密缓冲里 `slice_assign`。超出 `max_seq_len` 时丢掉最老的 token。`reset` 把长度和 RoPE 窗口整个清掉。能回滚的是第 5 节的第三方：CombsEngine 的 `popn`，qwen3-moe-rs 的 `rewind` 加递推状态快照，maryml 丢掉克隆，snowglobe 不回滚而是重跑接受前缀。Starling 的 candle 缓存才有后缀回滚，而且做两次：草稿前向的 KV 先全部丢掉，校验前向之后再裁到接受前缀；实现是把前缀 `narrow` 出来再 `contiguous`。burn-lm 和 CubeK 都没有分页 KV。CubeK 的 K、V 是本次启动的绑定，页表不在核里。

3. **接受规则不是拒绝采样，也没有「全接受再追加 bonus」。** `AcceptRatePredictor` 不进入 `greedy_draft`。burn-lm 每步采样一个 token。Starling 比较的是两边的 argmax。不匹配时留下校验遍在该位置的 token，然后对剩余音频位置继续逐 token 前向，这一轮不在第一次拒绝处结束。全部匹配时接受数就是这 N 个音频位置，不再多写一个。

4. **没有第二套采样器要去对齐。** burn-lm 在温度大于 0 时先 softmax，再 Argmax，或把概率拷到 CPU 做 top-p；温度不大于 0 时 Argmax 直接作用在 logits 上。Starling 在主机上对写死的 131072 维取 argmax，函数注释把位置 0 写成应当与草稿一致。Burn CUDA 引擎的 `commit` 注释写的是每次无状态地整段重编码、重解码。

5. **没有「GPU 上多校验几个 token 几乎免费」的实现。** CubeK `visited` 写明因果掩码不是对半砍，`seq_q = 1` 仍然扫过整个缓存；8 对 16 的测试访问 100/128 个分数。右下角对齐使长度为 K 的续写看见前缀加自己的三角，计算量按这个三角形计，而不是按 K 次互不相关的解码。flash 路径还有一条启动约束：stage 的 `seq_q` 必须整除问题的 `seq_q`，否则 Burn 退回把分数物化出来的 fallback。burn-flex 的说明把「物化整张分数」写成 `seq <= 512` 时更省调度，把分块 flash 写成分数矩阵超出 L2 才更快。Starling 里「两次 batched 前向比逐步便宜」以及「Jacobi 重校验比逐步更贵」写在 candle CPU 的函数注释里。

## 材料

- `sehaxe/burn-dspark` `7d816bcf3a13a3a6d008980d1fc7d5b595b670bd`：`README.md`，`Cargo.toml`，`src/lib.rs`，`.github/workflows/ci.yml`。
- `tracel-ai/burn-lm` `a53baab057e819d107394fc2b641209a93fa74de`：`README.md`，`Cargo.toml`，`Cargo.lock`，`crates/burn-lm-llama/src/generation/generate.rs`，`crates/burn-lm-llama/src/generation/sampling.rs`，`crates/burn-lm-llama/src/nn/attention/cache.rs`，`crates/burn-lm-llama/src/nn/attention/kv_cache.rs`，`crates/burn-lm-llama/src/nn/attention/mha.rs`，`crates/burn-lm-llama/src/nn/transformer.rs`，`crates/burn-lm-llama/src/nn/pos_encoding.rs`，`crates/burn-lm-llama/src/nn/llama/inference.rs`，`crates/burn-lm-http/src/schemas/chat_schemas.rs`，`crates/burn-lm-http/src/handlers/chat_handlers.rs`。
- Burn `3c6992d33d81c8727f13ebabc6797deca8ede2c5`：`crates/burn-nn/src/modules/rope_encoding.rs` 的 `RotaryEncoding::apply`。这是 burn-lm 那份 lock 钉住的 burn。
- Burn `f0d96d35c05f8801e653a57ff815e7894f5195dd`：`Cargo.toml`（cubek rev `cc23e559c9022a90f0da9e4662b30d889d220f1f`），`crates/burn-cubecl/src/kernel/attention/base.rs`，`crates/burn-cubecl/src/kernel/attention/bounds.rs`，`crates/burn-backend/src/backend/ops/modules/attention.rs`，`crates/burn-tensor/src/tensor/api/bool.rs`，`crates/burn-flex/ARCHITECTURE.md`，`.github/instructions/design.instructions.md`。
- CubeK `cc23e559c9022a90f0da9e4662b30d889d220f1f`：`crates/cubek-attention/src/forward/definition/base.rs`，`crates/cubek-attention/src/forward/definition/cost.rs`，`crates/cubek-attention/src/components/tile/mask.rs`，`crates/cubek-attention/tests/attention/forward/masking.rs`。
- CubeCL `d425faa533d9ce57e160ca470247130e901107c5`：`crates/cubecl-ir/src/interfaces/side_effects.rs`，`crates/cubecl-ir/src/interfaces/control_flow.rs`，`crates/cubecl-opt/src/analyses/dataflow_solver/pre.rs`，`crates/cubecl-environment/src/bytes/default_controller.rs`。
- `gjovanov/starling` `41913822e152bcbb4824fdf47a5a5c519e28c5ac`：`apps/burn-server/Cargo.toml`，`apps/burn-server/src/inference/candle_cpu/engine.rs`，`apps/burn-server/src/inference/candle_cpu/model.rs`，`apps/burn-server/src/inference/cuda_engine.rs`，`apps/burn-server/src/inference/bf16_engine.rs`，`apps/burn-server/docs/pending-optimizations.md`。
- `asqrzk/CombsEngine` `a80a947a2bb0fb5e27a721907233fcc70d0dd2f0`：`engine/core/combs-runtime/src/spec.rs`，`engine/core/combs-runtime/src/step.rs`，`engine/core/combs-models/src/traits.rs`。
- `9prodhi/qwen3-moe-rs` `c62bd60bc9c2f99fe6cd1c37aa1ad8f92f5602e9`：`src/cache.rs`。
- `erlkonig-ai/maryml` `578c3b3182d155aad9c587f99c4b0dc9594e098e`：`src/models/inkling/mtp.rs`。
- `rhi-zone/sketchpad` `6a7ad5391cdcff0011eaae5d3ef0016f2614c537`：`crates/sketchpad-core/src/speculative.rs`。
- `tianhaoz95/snowglobe` `95c9aefd77ea9ddee03852683ad1a1cab9fe6cfa`：`engine/src/model/speculative.rs`。
- 排除项：`abyo-software/abyo-speculate` `2c122517ae52f53c4830682128196a8b3ad8a3b1` 根 `Cargo.toml`；`moderately-ai/lmbrrr` `b7de000b4ebf4b0791909c70dc4f4d83fafe532a` 根 `Cargo.toml`；`tabbyml/tabby` `21b29048d7bcf6b94f9f482f2d0fd05efadfd19f` 根 `Cargo.toml`；katgpt-rs `69c171e6e2614b459ba6c90250891ae9cb203b7e` 根 `Cargo.toml`。
