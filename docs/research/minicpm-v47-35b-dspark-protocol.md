# MiniCPM-V-4.7-35B-A3B-DSpark 一次前向怎么出 token

2026-10-21。材料是 OpenBMB 这份草稿的模型卡、`config.json`、`model.safetensors` 头，模型卡点名的论文 arXiv:2607.05147v1，以及模型卡点名的两套推理实现：vLLM 的 `method: dspark` 与 llama.cpp 的 `--spec-type draft-dspark`。ModelScope 与 Hugging Face 上的 README、`config.json` 逐字节相同，下面把它们当成同一份发布。论文没有出现 MiniCPM。推理实现的提交是阅读当日的主干，模型卡没有钉提交号。

**结论：这是一份 5 层 BF16 草稿，一次骨干前向提出 15 个 token，条件来自配对目标文本塔第 3、11、19、27、35 层（从 0 数）结束之后的隐藏状态。** 置信度估计的是「前面都已接受」时这一位还能被接受的概率，用来决定送去校验的前缀有多长；模型卡里的 vLLM 命令把自适应校验关掉，那条命令因此校验满 15 个。接受几个 token 由目标分布上的从左到右检验决定：第一次不接受就丢掉后缀，该位置改由目标写出；15 个都接受时，目标再写一个 bonus。论文把写进上下文的序列定义成目标分布的样本。官方写明的配对检查点是未量化的 [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)。本仓库的 `models/MiniCPM-V-4.7-35B-A3B-int4` 不在这句配对里。声明的许可是 Apache-2.0。

## 1. 身份、配对和许可

| 项 | 来源所写 |
|---|---|
| 草稿 | ModelScope / Hugging Face `OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark`，架构 `Qwen3DSparkModel`，`model_type` `qwen3` |
| 论文 | [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)，v1，2026-07-06。HTML：<https://arxiv.org/html/2607.05147> |
| 配对目标 | 「trained for exact pairing with [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) and its tokenizer」 |
| 草稿参数 | 389,223,169，BF16，一个 `model.safetensors` |
| 许可 | ModelScope 元数据 `License` 与 Hugging Face 模型卡 YAML 都是 `apache-2.0` |

模型卡原文把目标列成 [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)。该仓库的卡是 `license: apache-2.0`，架构 `MiniCPMV4_7ForConditionalGeneration`，safetensors 参数量 BF16 35,212,875,824。文本配置 `text_config` 是 `qwen3_5_moe_text`：40 层，隐藏宽度 2048，词表 248144，16 个 query 头、2 个 KV 头、`head_dim` 256，256 个专家、每 token 8 个，MoE 中间宽度 512。`layer_types` 里 30 层是 `linear_attention`、10 层是 `full_attention`。草稿点名的 3、11、19、27、35 都落在 `full_attention` 上；同配置里其余的 full attention 下标是 7、15、23、31、39。草稿卡没有写选择规则。

草稿仓库的文件只有 `.gitattributes`、`README.md`、`config.json`、`model.safetensors`，没有 `LICENSE` 文件。README 正文没有再写许可名，只附了「AS IS」免责声明。ModelScope 元数据与 Hugging Face 卡的 YAML 把许可写成 `apache-2.0`。GGUF 仓库的 YAML 同样是 `apache-2.0`，`base_model` 是这份 BF16 草稿，关系是 `quantized`。

同一句配对也出现在目标模型卡的投机解码小节：草稿是为 [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) 训练的。本仓库 `models/MiniCPM-V-4.7-35B-A3B-int4` 的卡写明自己是该 BF16 检查点的 int4（W4A16，compressed-tensors，llm-compressor）量化，`base_model` 仍是未量化仓库。DSpark 模型卡没有把这份 int4 写成配对目标。GGUF 草稿卡的 llama.cpp 示例另用 [openbmb/MiniCPM-V-4.7-35B-A3B-gguf](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) 里的 `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` 当目标，并写「the draft must be paired with the matching MiniCPM-V 4.7 35B-A3B target」。那是另一份 GGUF，不是本仓库这个 W4A16 目录。

论文的实验目标是 Qwen3-4B、8B、14B 和 Gemma4-12B，部署目标是 DeepSeek-V4-Flash 与 V4-Pro 的 preview。它开放的检查点也是这两份 DeepSeek 草稿和训练库 [DeepSpec](https://github.com/deepseek-ai/DeepSpec)。论文没有把 MiniCPM-V-4.7 列进目标或检查点。

## 2. 草稿有几层，一次前向出几个 token

模型卡的规格表：

| 项 | 值 |
|---|---|
| Draft layers | 5 |
| Draft parameters | 389,223,169 |
| Draft tokens per forward pass | 15 |
| Target hidden-state layers | `[3, 11, 19, 27, 35]` |
| Checkpoint precision | BF16 |

`config.json` 与表一致：`num_hidden_layers` 5，`block_size` 15，`dflash_config.target_layer_ids` 就是上面五个数，`num_target_layers` 40。`layer_types` 五层都是 `full_attention`。`sliding_window` 为空，`use_sliding_window` 为 false。隐藏宽度 2048，MLP 中间宽度 6144，8 个 query 头、2 个 KV 头、`head_dim` 256。这是稠密层，不是目标的 MoE。

权重头共 62 个张量，元数据 `format: pt`，dtype 全部 BF16。`layers.0` 到 `layers.4` 各有 input / post-attention RMSNorm、Qwen3 式的 q/k/v/o 与 q/k norm，以及 gate/up/down。另外有 `fc.weight` `[2048, 10240]`、`hidden_norm.weight` `[2048]`、`norm.weight` `[2048]`、`markov_head.markov_w1.weight` 与 `markov_w2.weight` 各 `[248144, 256]`、`confidence_head.proj.weight` `[1, 2304]` 和 `bias` `[1]`。没有 `embed_tokens`，没有 `lm_head`，也没有 `d2t`。`10240 = 5 × 2048`。`2304 = 2048 + 256`。`draft_vocab_size` 与 `vocab_size` 都是 248144，和目标文本词表相同。

论文把生成拆成两段（[§3.1](https://arxiv.org/html/2607.05147)）。并行骨干一次前向产出块内全部位置的隐藏状态和基础 logits。相对 DFlash，它把 anchor 自己当成第一个预测位置：γ 个输入（1 个 anchor 加 γ−1 个 mask）给出 γ 个草稿 logits。顺序段再在这些 logits 上加只依赖前一个已采样 token 的低秩转移，从左到右采样。默认实例是 Markov 头，秩 r = 256：\(W_1\in\mathbb{R}^{V\times r}\) 做嵌入查找，\(W_2\in\mathbb{R}^{r\times V}\) 投回词表（式 (5)）。检查点的 `markov_head_type` 是 `vanilla`。两个权重都存成 `[V, r]`；vLLM 把 `w1` 用作 `Embedding(V, r)`，把 `w2` 用作 `Linear(r, V)` 的权重，也就是论文里 \(W_2\) 的转置存放。权重里没有 RNN 头。

这份检查点的 γ 是 15，三处一致：模型卡的「Draft tokens per forward pass」、`block_size`、以及 vLLM 命令里的 `num_speculative_tokens: 15`。vLLM 的 `DSparkSpeculator` 在 `sample_from_anchor` 为真时，每个请求的查询长度等于 `num_speculative_tokens`，位置 0 是上一轮目标写出的 anchor，其余是噪声；15 个位置全部采样。该字段在配置里缺省，实现默认是真。llama.cpp 同一默认下把可起草的个数钳到 `block_size`，输入是 `[上一 token, mask, …]`，从第 0 个槽读草稿。`mask_token_id` 是 248076。配对目标的 `tokenizer.json` 里，这个 id 的字符串是 `<|audio_pad|>`。仓库文件列表里没有单独的 `mask_embedding.pt`，mask 位置用的是目标嵌入表的这一行。`bos` / `eos` / `pad` 分别是 248045、248046、248044，在同一份词表里是 `<|im_start|>`、`<|im_end|>`、`<|endoftext|>`。本仓库 int4 目录里的 `tokenizer.json` 与该 Hugging Face 词表逐字节相同；词表相同并不使 int4 权重成为配对目标。

论文写草稿共享目标的嵌入层和语言模型头，两者冻结，训练只更新骨干、顺序块和置信度头（[§3.3](https://arxiv.org/html/2607.05147)）。vLLM 的 `Qwen3DSparkForCausalLM.load_weights` 写明：省略的 `embed_tokens` / `lm_head` 由加载过程改用目标的。llama.cpp 的 DFlash / DSpark 文档要求转换时用 `--target-model-dir` 继承目标的分词器和词嵌入。

块内注意力：论文写块内位置彼此双向可见，并看见注入的目标上下文（[§2.2](https://arxiv.org/html/2607.05147)）。配置没有 `is_causal`，`dflash_config` 里也没有 `causal`。vLLM 对这种草稿把 `full_attention` 层解析成非因果；llama.cpp 只在元数据把因果标成真时才打开因果注意力，转换器在配置没写该字段时不写入。

上下文怎么注入，论文的式 (2)(3) 是：把选定层的隐藏状态拼接，乘 \(W_c\in\mathbb{R}^{d\times md}\)，再做 RMSNorm，得到 \(H_{\mathrm{ctx}}\)；每一草稿层用自己的 \(W^K,W^V\) 同时投影 \(H_{\mathrm{ctx}}\) 和草稿块，并在序列维拼到 K/V 上。`fc.weight` 的形状与 \(W_c\) 一致，`hidden_norm` 是其后的 RMSNorm。目标的 16 个 query 头不出现在草稿权重里，因为投影发生在草稿自己的 2048 维空间。

论文里 DeepSeek-V4 的生产草稿是三层 MoE、滑动窗口 128、最大块长在部署一节写成 γ = 5。那不是这份 MiniCPM 检查点。论文离线实验把 DSpark 和 DFlash 都设成 5 层，默认块长与 Eagle3 的 TTT 视野对齐为 7，再把「提案长度 γ 再加一个 anchor token」扫过 {4, 8, 12, 16}，并单独报告 γ = 7 与 γ = 15。那些数字属于 Qwen3-4B 上的消融。这份检查点的 15 来自它自己的模型卡和 `block_size`，论文没有把该消融说成 MiniCPM 的配置。

## 3. 要读目标的哪些层

模型卡与 `dflash_config.target_layer_ids` 写的是 `[3, 11, 19, 27, 35]`，`num_target_layers` 是 40。目标 `text_config.num_hidden_layers` 也是 40，所以上界落在文本层内部。

两套被模型卡点名的实现把这五个数解释成同一处激活：

- vLLM `get_eagle3_aux_layers_from_config` 对 `dflash_config.target_layer_ids` 的每个下标加 1，注释写这是在换 DFlash 的辅助层编号。Qwen3-Next 的前向在第 `layer_idx` 层算完之后，用 `layer_idx + 1` 去对这个编号，存下 `hidden_states + residual`。MiniCPM-V 4.7 在 vLLM 里的文本塔是 `Qwen3_5ForCausalLM`，其解码器继承这段前向。配置里的 3 因此是第 3 层（从 0 数）结束之后的残差，采集编号是 4。
- llama.cpp 的 DFlash 转换同样写成 `extract_layer_ids = [i + 1 for i in target_layer_ids]`，运行时读取这些层的输入。`qwen35moe` 图在进入第 `il` 层之前把当时的残差记为该层输入。配对目标的 `text_config.model_type` 是 `qwen3_5_moe_text`；llama.cpp 的 MiniCPM-V 4.7 文档把文本塔写成 `qwen3_5_text` 或 `qwen3_5_moe_text`。输入到第 4 层的残差，就是第 3 层的输出。

五处都按这个规则：读的是第 3、11、19、27、35 层的输出，也就是第 4、12、20、28、36 层的输入。最后一层的下标是 39。配置没有要求读第 39 层，也没有要求读最终 RMSNorm 之后的状态。论文只说目标层集合 \(\{l_1,\ldots,l_m\}\) 的隐藏状态，没有给这份检查点单独编号；编号以模型卡和 `config.json` 为准。

## 4. 置信度在推理时做什么

训练目标，模型卡写成 `CE + L1 + confidence loss`，2M 条序列、6 个 epoch、AdamW。响应由 MiniCPM-V-4.7-35B-A3B 从视觉和文本混合提示生成。配置里有 `confidence_head_alpha: 1.0`、`enable_confidence_head: true`、`confidence_head_with_markov: true`。论文的默认组合是 \(\alpha_{\mathrm{ce}}=0.1\)、\(\alpha_{\mathrm{tv}}=0.9\)、\(\alpha_{\mathrm{conf}}=1.0\)，L1 项是草稿分布和目标分布的全词表 L1，置信度标签是 \(c_k^{*}=1-\tfrac12\|p_k^{d}-p_k^{t}\|_1\)（[§3.3](https://arxiv.org/html/2607.05147)，式 (8)–(12)）。论文的数据是 Open-PerfectBlend 的 130 万条提示、10 个 epoch、非思考模式。模型卡没有写 0.1 / 0.9 这两个系数，也没有写自己用了 Open-PerfectBlend。`confidence_head_alpha: 1.0` 与论文的 \(\alpha_{\mathrm{conf}}\) 数值相同，配置没有说明这个字段就是那个系数。

推理时，论文把 \(c_k\) 定义成：给定块内前面的 token 都已被接受，第 \(k\) 个草稿 token 仍能通过目标校验的条件概率（[§3.2.1](https://arxiv.org/html/2607.05147)）。形式是 sigmoid 后的线性投影，输入是骨干隐藏状态拼上前一个草稿 token 的 Markov 嵌入。前缀活下来的概率是这些条件概率的连乘。它不替换接受检验。接受检验仍是目标分布和草稿分布之间的拒绝采样。

检查点的投影有 bias，形状 `[1]`。论文的式 (7) 只有 \(w^{\top}[h_k; W_1[x_{k-1}]]\)，没有 bias 项。

置信度进入推理的方式是截断待校验前缀，而不是给 token 打接受分：

- 论文的硬件感知调度器用连乘生存概率和一份预先测好的吞吐曲线，挑选每个请求送进目标的前缀长度（算法 1，[§3.2.2](https://arxiv.org/html/2607.05147)）。为了保持目标分布，截断决定不能依赖还没被接纳的未来 token。附录 A 给了一个反例：若回头看完全部连乘再决定第一位进不进校验，输出分布会离开目标。论文在生产系统里改成用两步之前的置信度决定预算，并去掉提前停止；它把这写成 DeepSeek-V4 的部署，不是这份 MiniCPM 卡上的命令。
- 模型卡的 vLLM 命令写的是 `"enable_adaptive_verification": false`。vLLM 文档写：该开关默认关；设为 false 时每个请求校验整段。`DSparkSpeculator` 只在开关为真且置信度头存在时才计算并使用 \(c_k\)。因此这条命令不读取置信度来缩短 15。
- llama.cpp 用 `--spec-draft-conf-min P` 在第一个预测接受率低于 P 的位置截断；默认 P 是 0，表示关闭。GGUF 卡的 `llama-cli` / `llama-server` 示例没有传这个参数。

模型卡的接受长度表没有写评测时自适应校验是开还是关，也没有写温度和硬件。不能从用法示例反推那张表。

## 5. 校验一步如何决定接受多少 token

论文的规则在 [§2.1](https://arxiv.org/html/2607.05147)：草稿提出 \(x_1,\ldots,x_\gamma\)。目标一次前向算出每个位置的 \(p_k^{t}\)。位置 \(k\) 以 \(\min(1, p_k^{t}(x_k)/p_k^{d}(x_k))\) 接受。检验从左到右，第一次拒绝丢掉 \(x_{k+1},\ldots,x_\gamma\)。引言写：接受与目标分布一致的最长前缀，再追加一个 bonus token；因为接受规则精确保持目标分布，加速不带来质量损失。图 1 的例子是 E、F 被接受，G 被拒绝，目标写出修正 token G*。全部草稿都被接受时，同一次目标前向在块尾再采样一个 bonus，它成为下一轮的 anchor。论文脚注 4 写明：除非另说，接受长度和接受率都包含目标生成的 bonus token。离线实验的采样温度是 1.0。

模型卡把接受长度定义成完成 token 总数除以投机校验步数，评测用自然 EOS 和 `max_new_tokens=8192`。它没有使用 bonus 这个词。完成 token 是写进完成文本的 token，论文则明确把 bonus 算进 \(\tau\)。

模型卡点名的实现，在各自默认参数下，比论文的「用完整草稿分布做比率」更窄：

- vLLM 命令没有写 `draft_sample_method` 和 `rejection_sample_method`。阅读到的 vLLM（`f4c07f326c7c7c56dbb86857404e26a6ef7c065f`，2026-10-21）里，前者默认 `greedy`：草稿取 argmax，拒绝采样时把草稿分布当成该 token 上的 one-hot。后者默认 `standard`：仍按 \(p/q\) 做概率拒绝采样。one-hot 时 \(q(x)=1\)，接受概率变成 \(p^{t}(x)\)。命令没有打开论文里的概率草稿采样；vLLM 把那一档叫做 `probabilistic`，需要显式设置。
- llama.cpp（`ea9e3446b5c84a7f972cbbdea9df615fc4ff9159`，2026-10-21）的 `draft-dspark` 在草稿侧用 top-k 10 后取第一名，并不把草稿分布写入 `result_q`。server 只在温度大于 0 且 `result_q` 非空时调用 \(\min(1, p/q)\) 那条路径。DSpark 这条路径因此走到 sample-and-match：目标采样器在该位置抽出的 token 与草稿相同才继续，第一次不同就停，停住的位置使用目标抽出的 token；草稿全部相同，再从目标多抽一个 token。`speculative-simple` 示例直接调用的也是这条匹配函数。温度为 0 时，server 的注释写两种检验重合，因为此时目标分布也是点质量。

GGUF 卡写：「Draft quantization only affects acceptance length (never the output)」。这句话的主语是草稿量化。它同时要求草稿配上匹配的 MiniCPM-V 4.7 35B-A3B 目标。它没有写目标量化不改变输出。

校验前的长度还可以被置信度截短，见上一节。模型卡给出的两条命令都没有打开这道截断，所以它们把起草出的 15 个位置送进上述检验。论文的调度器可以在检验之前就少送一些。少送的决定改变的是校验预算，不改变「从左到右、第一次失败即止」这一条。

## 6. 接受之后的 token 相对目标单独解码

论文的关系是分布相等，写在引言、[§2.1](https://arxiv.org/html/2607.05147)、[§3.2.2](https://arxiv.org/html/2607.05147) 和附录 A：接受规则保持目标分布，调度若破坏非预期性就会离开该分布。一个校验步写回的是「被接受的草稿前缀，加上目标在失败位置写出的修正，或者在全接受时写出的 bonus」。下一轮的 anchor 就是这个由目标决定的最后一个 token。

这不是「与某一次不带草稿的解码逐 token 相同」的定义。论文离线温度是 1.0。在该温度下，两次独立的目标采样不必逐 token 相同；投机路径要保持的是同一次目标采样分布。GGUF 卡那句「不改变输出」说的是草稿量化不改由目标决定的输出，接受长度可以变。

温度为 0、草稿也取 argmax 时，llama.cpp 的匹配检验接受的是与目标 argmax 相同的前缀，失败位和 bonus 都是目标 argmax。这时写出的序列与目标贪心解码相同。模型卡没有声明评测或用法示例的温度。论文没有把这份 MiniCPM 检查点的输出定义成贪心逐 token 相等。

## 7. 模型卡报告的接受长度

模型卡的定义见第 5 节。表内数字：

| 域 | 接受长度 |
|---|---|
| MMMU | 5.098 |
| MathVista | 6.126 |
| CharXiv_reasoning_val | 5.479 |
| OCRBench | 5.407 |
| Math | 6.687 |
| Code | 7.387 |
| General | 3.501 |

表把前四项标为 Vision，后三项标为 Text。模型卡没有给出温度、硬件、是否打开自适应校验、以及分子是否按论文脚注包含 bonus。论文表 1 的数字来自 Qwen3 与 Gemma4，不能填到这张表上。

## 8. 接到当前推理

本仓库已有的 35B 检查点是 `models/MiniCPM-V-4.7-35B-A3B-int4`。图像改写默认加载它，去噪观察用的 `MiniCpm35bProjection` 读的是该检查点最后一层、宽度 2048 的序列。DSpark 要的是另一份 BF16 草稿，以及配对目标在第 3、11、19、27、35 层之后的残差，不是最后一层，也不是最终范数之后的状态。`crates/minicpm-v` 在这条笔记所在的提交里没有草稿模型、没有 15 宽的块、也没有拒绝采样。

若以后按模型卡接这条协议，目标侧要在文本塔上述五层之后留出隐藏状态，草稿侧是一次 5 层前向加块内 Markov 采样，嵌入和输出头用目标的。置信度只在打开自适应校验或 llama.cpp 的 `--spec-draft-conf-min` 时参与决定送检长度。int4 目录的卡自己把 `base_model` 指回未量化的 35B-A3B，DSpark 卡的配对句也指向那里。在 OpenBMB 把 int4 写进配对句之前，这份 W4A16 权重仍然只是本地的目标量化，不是已经配对的草稿目标。

## 材料

- 模型卡与文件列表：<https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark>，接口 <https://modelscope.cn/api/v1/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark>。README 修订 `41ef3c3298202aa043f6b05ef4605c6884eda70f`，`config.json` 与权重修订 `e118b3a245dfcbfe1e50fa59c744d9394eaa20a7`。`config.json` 的 sha256 是 `a603178db396e6a18e95d3151156ad817315e4135f99426253ac469629182096`，README 的 sha256 是 `c99dd2052fad0be461802f2e2e6177a7d09c386a6703aaf18a86619dd67e905f`。权重 sha256 是 `211eda7b962795ba727b0d7db072d8c474dc3b94075f85f123fe363caf5e9679`，778,452,858 字节。
- 同一发布的 Hugging Face 仓库：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark>，`sha` `55249d3e69315892ca07bc956228b404f2526f9a`。README 与 `config.json` 的 sha256 与 ModelScope 相同。参数量接口报告 BF16 389,223,169。本文没有再下载 Hugging Face 上的权重字节。
- 配对目标：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B>，`sha` `896c2d454672476a7b45dcc5e3f0c700a0218577`。`config.json` 与 `tokenizer.json` 取自该仓库。`tokenizer.json` 的 sha256 是 `e4608c5d7b3f4f30ba83be16a82cdac856b2e7191d40f0a08cbacc76c2a926ee`，与本仓库 int4 目录中的词表文件相同。
- GGUF 卡：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf>。
- 论文 v1：<https://export.arxiv.org/api/query?id_list=2607.05147>，HTML <https://arxiv.org/html/2607.05147>。
- vLLM `f4c07f326c7c7c56dbb86857404e26a6ef7c065f`：`vllm/model_executor/models/qwen3_dspark.py`，`vllm/model_executor/models/qwen3_dflash.py`，`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`，`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`，`vllm/config/speculative.py`，`vllm/model_executor/models/qwen3_next.py`，`vllm/model_executor/models/interfaces.py`，`vllm/model_executor/models/minicpmv4_6.py`，`docs/features/speculative_decoding/adaptive_verification.md`。
- llama.cpp `ea9e3446b5c84a7f972cbbdea9df615fc4ff9159`：`docs/speculative.md`，`conversion/qwen.py`，`common/speculative.cpp`，`common/sampling.cpp`，`src/models/qwen35moe.cpp`，`tools/server/server-context.cpp`，`docs/multimodal/minicpmv4.7.md`。
