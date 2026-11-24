# 这次 MiniCPM DSpark 测量算不算合理

2026-10-22。问题是：BF16 的 MiniCPM-V-4.7-35B-A3B-DSpark 草稿，配本地 int4 目标 `models/MiniCPM-V-4.7-35B-A3B-int4`，一条贪心提示「用几句话说明分页 KV 缓存是什么。」，停在该检查点 `generation_config.json` 的 EOS，`max_new_tokens=8192`。接受长度按完成 token 除以投机校验步。测到目标单独 497 token、74.6 s；整块 15 全校验 382 token、168 步、接受长度 2.274、258 s；更早一次只校验前 2 个草稿 token 时 385 token、203 步、1.897、78.4 s。三次都在 EOS 前结束。两条投机完成和目标单独完成有共同前缀，然后分开。

下面五条各自回到论文、模型卡，或拥有该行为的实现。分支 `research/minicpm-dspark-protocol` 上的 `docs/research/minicpm-v47-35b-dspark-protocol.md` 只用来找到已经钉住的提交，不作为证据。本地代码只用来说明这次循环在计什么。本文没有重跑这三次解码。

**结论：除法定义和模型卡一致，分子包含目标写出的修正或 bonus，2.274 不是把 bonus 漏掉了。这个幅度在模型卡整张表之外，也低于论文里每一格 DSpark。同一 int4 目标、温度 0、sample-and-match 下，完成文本应当等于该目标自己的贪心；497 与 382、385 分开，这一条不符合来源。BF16 草稿配这份 W4A16 int4 不是写明的配对。整块 15 比目标单独更慢，对不上论文的服务加速；按论文自己的耗时公式，一步校验远贵于一次解码时就会变慢。vLLM 把 batch 1 的额外 token 写成接近免费，这次的一步大约是一次目标 token 的 10 倍，不在那句成本里。**

错位有两处。一是校验结果：同一目标写出的 token 不一致。二是配对：发表的接受长度不是这个 int4 目标上的数。指标定义本身对齐。

## 测量在计什么

本地 int4 目录的 `generation_config.json` 把 `eos_token_id` 写成 `[248044, 248046]`。`ModelDirectory::eos_ids` 读的就是这个文件里的数组（`crates/minicpm-v/src/config.rs`）。当前 `dspark::generate` 在温度 0 下跑：CLI 在温度不是 0 时直接拒绝 DSpark（`crates/minicpm-v/src/bin/minicpm-v.rs`）。采样器在温度为 0 时取 argmax；这次命令的默认 `repeat_penalty` 是 1.0，`presence_penalty` 写成 0（`generate.rs` 的 `Sampler::sample`，`minicpm-v.rs` 的选项）。

接受长度是 `output_tokens / cycles`（`dspark.rs` 的 `SpecStats::acceptance_length`）。每个循环加一步。写进完成序列的草稿 token，以及修正或 bonus，都加进 `output_tokens`。停住的 EOS 在推入完成序列之前就返回，所以分子不含这个 EOS id。382 / 168 = 2.2738，385 / 203 = 1.8966，与题目里的 2.274、1.897 一致。

当前循环在第一个草稿 token 与目标 argmax 相同之后，对整个 `proposals` 做一次 `forward_verifying`，再从左到右比后面的位置；全部相同且没有 EOS 时，用块尾那一行 logits 再采一个 token（`dspark.rs` 里 `first == proposals[0]` 之后的循环）。当前树里没有「只校验前 2 个」的开关。只校验 2 个的那次是题目记录的更早运行，本文没有从旧版本把那次循环重读出来。

## 1. 定义

论文没有把接受长度写成「完成 token 除以校验步」。[§2.1](https://arxiv.org/html/2607.05147v1) 把 \(\tau\) 定义成每个 cycle 接受的 token 数，单 token 延迟是 \((T_{\mathrm{draft}}+T_{\mathrm{verify}})/\tau\)。引言写：接受与目标分布一致的最长前缀，再追加一个 bonus token。

[§4.1](https://arxiv.org/html/2607.05147v1) 写：报告每个 decoding round 的 accepted length \((\tau)\)。脚注 4 的原文是：

> For clarity, unless otherwise stated, all reported metrics for accepted length and acceptance rate include the target-generated bonus token.

脚注 1 把 anchor 和 bonus 当成同一个东西：上一轮目标模型写出的最后一个 token。图 1 的说明里，E、F 被接受，G 被拒绝，目标写出修正 token G\* 来结束这一轮。G\* 就是这一轮目标写出的最后一个 token，也就是脚注 1 里的 bonus。所以论文报告的 \(\tau\) 含这个目标写出的 token，不论它是拒绝后的修正，还是整块都接受之后多出来的那一个。

模型卡用的是另一句，Hugging Face 与 ModelScope 正文相同。Evaluation 节：

> Acceptance length is defined as the total number of completion tokens divided by the total number of speculative verification steps. Evaluation uses natural EOS termination and `max_new_tokens=8192`.

来源：[openbmb/MiniCPM-V-4.7-35B-A3B-DSpark](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark) 的 README，以及 ModelScope `OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark` 接口里的 `ReadMeContent`。两边正文一致；Hugging Face 多一段 YAML 头。这句没有出现 bonus 这个词。完成 token 是写进完成文本的 token。目标写出的修正或 bonus 若在完成文本里，就在分子里。

这次 2.274 用的是模型卡这句除法。本地计数把修正和 bonus 放进分子，把停住的 EOS 留在分子外面。它和论文脚注 4 指的是同一类量：每步写回上下文的 token，包含目标写的那一个。它没有把 \(\tau\) 算成「只计被接受的草稿、不含 bonus」。

## 2. 发表的幅度

模型卡的表，域和接受长度：

| 域 | 接受长度 |
|---|---|
| MMMU | 5.098 |
| MathVista | 6.126 |
| CharXiv_reasoning_val | 5.479 |
| OCRBench | 5.407 |
| Math | 6.687 |
| Code | 7.387 |
| General | 3.501 |

同一张卡的规格：草稿每次前向 15 个 token，检查点精度 BF16，目标写成 [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)。Evaluation 节没有写温度、硬件、自适应校验开或关、以及目标权重是 BF16 还是某种量化。卡上的 vLLM 命令是另一段：`--model openbmb/MiniCPM-V-4.7-35B-A3B`，`tensor-parallel-size 2`，`max-num-seqs 1`，`num_speculative_tokens` 15，`enable_adaptive_verification` false。这是用法，卡没有把这行命令标成上表的评测条件，也没有给出 tok/s。

`config.json`（同一仓库 `raw/main/config.json`）里 `block_size` 是 15，`dflash_config.target_layer_ids` 是 `[3, 11, 19, 27, 35]`。文件里没有 `sample_from_anchor` 字段。

论文 [Table 1](https://arxiv.org/html/2607.05147v1)（[§4.1](https://arxiv.org/html/2607.05147v1)–[§4.2](https://arxiv.org/html/2607.05147v1)）的目标是 Qwen3-4B、8B、14B 和 Gemma4-12B。块长与 Eagle3 的 TTT 视野对齐为 7。温度 1.0。离线评测关掉 confidence scheduler，固定块长。DSpark 各格里，我读到的最低是 Gemma4-12B 的 Arena-Hard 2.92，最高是 Qwen3-14B 的 GSM8K 6.21。Qwen3-4B 上正文给出的域均值是 math 5.57、code 5.12、chat 3.49。这张表没有 MiniCPM，没有 int4，也没有端到端加速比。论文正文没有出现 MiniCPM。

[§4.3.2](https://arxiv.org/html/2607.05147v1) 在 Qwen3-4B、5 层草稿上把提案长度扫过 {4, 8, 12, 16}，并写 \(\gamma=15\) 时 DSpark 相对 DFlash 的接受长度增益：math 30%、code 26%、chat 22%。这里没有给出 \(\gamma=15\) 的绝对 \(\tau\)。延迟只在 Figure 4 最右栏：batch size 128，上下文长度 {512, 1024, 2048, 4096} 的算术平均，草稿从 4 加到 16 使整轮延迟比 DFlash 多 0.2% 到 1.3%。没有写出 GPU 型号。

2.274、块长 15，在模型卡的 3.501–7.387 之外。离最近的 General 3.501 差 1.23，大约低 35%。表内 General 到 Code 大约差一倍，所以这是偏低一截，不是差一个数量级。一条通用中文提示可以对不齐一项基准的均值，模型卡没有给单条提示的下界。论文的 DSpark 最低格 2.92 也高于 2.274，而那格是另一族模型、块长 7、温度 1.0。论文和模型卡都没有一个数是「BF16 草稿 + 这份 W4A16 int4 + 块长 15」的接受长度。

## 3. 贪心下应不应该逐 token 相同

论文保证的是分布，不是「和另一次独立解码逐 token 相同」。引言和 [§2.1](https://arxiv.org/html/2607.05147v1)：接受规则精确保持目标分布，校验是从左到右的拒绝采样，位置 \(k\) 以 \(\min(1, p_k^{t}(x_k)/p_k^{d}(x_k))\) 接受，第一次拒绝丢掉后缀。附录 A 写：送去校验的长度不能依赖还没采样的未来 token，否则输出分布离开目标。离线实验的温度是 1.0（[§4.1](https://arxiv.org/html/2607.05147v1)）。在这个温度下，两次独立的目标采样不必逐 token 相同。

温度为 0 时，目标分布是 argmax 上的点质量。保持这个分布，写出来的序列就是该目标的贪心序列。条件是：每个位置用来比较的 logits，就是这个目标在同一前缀上自己会算出的 logits；拒绝处和整块接受后的 bonus 都从这份 logits 取 argmax。

vLLM 在提交 `f4c07f326c7c7c56dbb86857404e26a6ef7c065f` 里把这件事写成代码。`vllm/config/speculative.py`：`rejection_sample_method` 默认 `standard`，`draft_sample_method` 默认 `greedy`。greedy 取 argmax，拒绝采样时把草稿概率当成该 token 上的 one-hot（同文件字段说明；`rejection_sampler_utils.py` 里无草稿 logits 时 `draft_log_prob = 0`）。`enable_adaptive_verification` 默认 false；文档 `docs/features/speculative_decoding/adaptive_verification.md` 写：设为 false 时每个请求校验整段。

同一提交的 `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`：`temp == 0` 走 greedy。注释和赋值是：接受当且仅当草稿 token 等于目标 argmax；拒绝时把目标 argmax 写入采样结果，不再重采样。整块都接受时，bonus 位置直接用目标 logits。`vllm/v1/worker/gpu/sample/gumbel.py` 的 `gumbel_noised_argmax`：`temp == 0` 时不加 Gumbel 噪声，返回 logits 的 argmax。因此在温度 0、非 synthetic、非 FLy 的默认路径上，写出的每个 token 都是该目标在对应前缀上的 argmax。

`sample_from_anchor` 管的是草稿怎么排查询，不改变上面的接受检验。`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` 的模块说明：为真时每个请求的查询长度等于 `num_speculative_tokens`，位置 0 是 anchor，每个位置都是预测。实现在配置缺这个字段时默认真（同文件 `__init__`）。MiniCPM 草稿的 `config.json` 没有这个字段，所以这条默认会打开。它不决定输出是否等于目标贪心。

llama.cpp 提交 `ea9e3446b5c84a7f972cbbdea9df615fc4ff9159`：

- `docs/speculative.md` 的 DSpark 节：草稿须用 `--target-model-dir` 继承目标的分词器和词嵌入；`--spec-draft-n-max` 钳到训练块长；`--spec-draft-conf-min` 默认 0，表示不按置信度截断。
- `common/speculative.cpp` 的 DSpark 草稿：采样器是 top-k 10，取候选数组的第一名；这段不把草稿分布写入 `result_q`。`sample_from_anchor` 默认真，anchor-first 时起草个数钳到 `block_size`。
- `tools/server/server-context.cpp`：`use_spec_rejection` 仅在温度大于 0 时为真。校验处若 `spec_draft_q` 为空，调用 `common_sampler_sample_and_accept_n`，也就是 sample-and-match。DSpark 草稿不填 `result_q`，所以即使温度大于 0，这条路径也进 sample-and-match。
- `common/sampling.cpp` 的 `common_sampler_sample_and_accept_n`：对每个草稿位置用目标采样器抽一个 token，抽到的和草稿相同才继续，第一次不同就停，停住的位置留下目标抽到的 token；草稿全部相同，再从目标多抽一个。温度为 0 且采样链以 temperature 结尾时，初始化改用 greedy sampler（同文件，`temp == 0` 的分支）。
- 同文档 Backend Sampling：随机采样在 CPU 与 backend 上可以抽出不同 token；要逐 token 对齐时用 greedy。

这些条件加在一起：目标温度 0，检验是 sample-and-match 或 vLLM 的 greedy argmax 匹配，修正和 bonus 来自同一目标在同一前缀上的 argmax，输出就是这个目标单独贪心的序列。草稿差，只会缩短被接受的前缀。GGUF 卡把这件事说成草稿侧的量化（见下一节）。它没有说「草稿不准，目标就会写出另一串 token」。

这次两条投机完成都和同一 int4 目标的单独贪心分开。块长 15 和只校验前 2 个都分开，所以分叉不依赖「是否把 15 个都送进前向」。这不符合上面的条件。

### 配对

DSpark 卡第一句：trained for exact pairing with [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) and its tokenizer。目标卡的 Speculative decoding 节复述：草稿是为该仓库训练的（[openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) README，Speculative decoding (DSpark)）。vLLM 示例的 `--model` 也是这个未量化仓库。

int4 卡把自己写成该 BF16 检查点的 int4（W4A16，compressed-tensors，llm-compressor），`base_model` 是 `openbmb/MiniCPM-V-4.7-35B-A3B`，`base_model_relation` 是 `quantized`（[openbmb/MiniCPM-V-4.7-35B-A3B-int4](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-int4) README 开头）。目标卡的权重表把 int4 列在模型权重下，DSpark 头只列 BF16 和 GGUF。表的句子是「每个模型配有 BF16 和 GGUF 的 DSpark 头」。没有一句话把这份 W4A16 目录写成 DSpark 的配对目标，也没有给 int4 单独的接受长度。

GGUF 卡的 llama.cpp 示例用 [openbmb/MiniCPM-V-4.7-35B-A3B-gguf](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) 里的 `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` 当目标，草稿是 `dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf`（[openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf) README，Usage with llama.cpp）。紧接着的两句是：draft 必须配上匹配的 MiniCPM-V 4.7 35B-A3B 目标；Draft quantization only affects acceptance length (never the output)。主语是草稿量化。输出仍由那个目标决定。这是一份官方的量化搭配，目标是 Q4_K_M GGUF，草稿是 Q8_0 GGUF。它不是「BF16 草稿 + compressed-tensors int4」，也没有把「换一个更粗的目标量化」写成缩短接受长度、同时保持目标自己 token 不变的办法。

换目标量化会改变目标自己相对 BF16 会写的 token。在同一个 int4 目标上，来源要求投机路径仍然写出这个 int4 目标的贪心。这次测量比的就是同一个 int4 目录。分叉不能用「官方允许用 int4 来换更短的接受长度」来解释。来源里没有这句许可。

DeepSpec 的训练掩码不定义接受长度，也不定义贪心是否逐 token 相同。`deepspec/modeling/dspark/common.py` 提交 `005e03b81cec38b7da6399833d609ee89a2587f2`：`create_dspark_attention_mask` 让同一块内的草稿位置互相可见，上下文只看到 anchor 之前；`create_position_ids` 是 anchor 位置加上 `0 .. block_size-1`。这是训练时的块布局。

## 4. 速度

把题目里的时间除一下，只描述这次测量：目标单独 74.6 / 497 ≈ 0.150 秒/token。整块 15 是 258 / 168 ≈ 1.54 秒/步，约为一格目标 token 的 10 倍；382 token 用了 258 秒，约为单独解码吞吐的 0.22 倍。只校验前 2 个时 78.4 / 203 ≈ 0.39 秒/步，约为 2.6 格目标 token；385 token 用了 78.4 秒，仍慢于 497 token 的 74.6 秒。把校验从 15 收到 2，墙钟从 258 秒降到 78.4 秒。

论文的服务数字是另一件事。[摘要](https://arxiv.org/html/2607.05147v1)和 [§5.4](https://arxiv.org/html/2607.05147v1)：DeepSeek-V4-Flash 与 V4-Pro 的 preview，对照生产基线 MTP-1，最大草稿长度 \(\gamma=5\)。在对齐的吞吐上，每用户生成速度加快 60%–85%（Flash）和 57%–78%（Pro）。Flash 在 80 tok/s/user 的 SLA 上总吞吐高 51%；120 tok/s/user 那个 661% 被正文写成前沿被拉开，而不是相对一个已经吃满的基线的典型倍数。Pro 在 35 tok/s/user 上总吞吐高 52%。中等并发下，调度器把每请求的校验预算从 MTP-1 的固定 2 个 token 放到大约 4–6 个（少于 200 并发的 Flash、少于 150 并发的 Pro）。正文没有给出 GPU 型号，也没有给出「相对不用投机的目标单独解码」的加速比。[§4](https://arxiv.org/html/2607.05147v1) 的离线表报的是 \(\tau\)，不是 tok/s。

[式 (1)](https://arxiv.org/html/2607.05147v1)（[§2.1](https://arxiv.org/html/2607.05147v1)）把每 token 延迟写成 \((T_{\mathrm{draft}}+T_{\mathrm{verify}})/\tau\)。\(\tau\approx 2.27\) 时，一步草稿加校验若贵于大约 2.27 次单 token 解码，总时间就长于目标单独解码。引言里系统侧的句子是：轻负载下多校验一个 token 几乎不花钱；重负载下，高拒绝风险的 token 占用本可以给别的请求的 batch 容量。[§3.2](https://arxiv.org/html/2607.05147v1) 写：无差别校验整个草稿块可以拉低系统吞吐，尤其在高并发。[§5.4](https://arxiv.org/html/2607.05147v1) 的限制写：接受率本来就低的请求，草稿侧生成整块的成本收不回来。生产部署用的最大块是 \(\gamma=5\)，再由调度器缩到大约 4–6，不是固定校验 15。

vLLM 的自适应校验文档（同一提交，`docs/features/speculative_decoding/adaptive_verification.md` 开头）写：投机解码用更多计算换更少的解码步；batch size 1 时这笔交换合算，GPU 处于访存瓶颈，多出来的草稿 token 接近免费；batch size 256 时草稿 token 和真实 token 抢同一份计算，拒绝够多就会掉吞吐。模型卡的 vLLM 命令把 `max-num-seqs` 设为 1，并关掉自适应、校验满 15。命令没有附速度。

因此：单流上整块 15 慢于目标单独贪心，是式 (1) 在「一步校验约为 10 次单 token」时的结果。论文把「几乎不花钱」放在轻负载，把「整块校验变慢」主要放在高并发。这次测量是单条请求，步长时间却远贵于一格 token，不在「batch 1 额外 token 接近免费」那句里。论文没有声称 MiniCPM、int4、块长 15、batch 1 仍然应当快过目标单独解码。60%–85% 也不是这组 258 秒对 74.6 秒。

目标卡把设备上的 **13+ tokens/s、iPhone 17 Pro Max、约 3 GB** 写成 LES、逐层量化和 DSpark 合在一起的结果（int4 卡与 BF16 目标卡的 On-Device 节都有这句）。那不是「15 token 校验对目标单独贪心」的对照，也不是这次测量。

## 5. 错位在哪里

来源要求的结果，和这次测量对不上的地方是：

1. **校验结果。** 温度 0、sample-and-match（llama.cpp）或 greedy argmax 匹配（vLLM），写出的序列是同一个目标的贪心序列。草稿差只缩短接受前缀。这次 497、382、385 有共同前缀然后分开，块长 15 和只校验前 2 个都分开。错位在写出的 token，不在「完成 token / 校验步」这个除法。
2. **配对。** 发表的接受长度属于未量化的 MiniCPM-V-4.7-35B-A3B，或论文里的 Qwen3 / Gemma4。GGUF 示例是 Q4_K_M 目标加 Q8_0 草稿，并且只把草稿量化说成改变接受长度、不改变输出。BF16 草稿加这份 W4A16 int4 没有被写成官方配置，也没有被写成一种「缩短接受长度、目标自己的 token 保持不变」的已知办法。2.274 因此不能当成模型卡 3.501–7.387 的复现；它也没有一份发表的 int4 数字可以对照。

速度上的 258 秒对 74.6 秒，用论文的式 (1) 和这次测到的步长可以算出来。它不是论文报告的服务加速，也不是 vLLM 对 batch 1「额外 token 接近免费」的那一档。

## 材料

- 论文 v1，2026-07-06：<https://arxiv.org/abs/2607.05147v1>，HTML <https://arxiv.org/html/2607.05147v1>。引用的是引言、§2.1、图 1、脚注 1 与脚注 4、§3.2、§4.1–§4.3.2、Table 1、§5.4、附录 A。
- 草稿卡：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark>，`config.json` 同上仓库 `raw/main/config.json`。ModelScope 接口 <https://www.modelscope.cn/api/v1/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark> 的 `ReadMeContent` 与 Hugging Face 正文一致。
- 目标卡：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B>。int4 卡：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-int4>。GGUF 草稿卡：<https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf>。
- vLLM `f4c07f326c7c7c56dbb86857404e26a6ef7c065f`：`vllm/config/speculative.py`，`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`，`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`，`vllm/v1/worker/gpu/sample/gumbel.py`，`docs/features/speculative_decoding/adaptive_verification.md`。
- llama.cpp `ea9e3446b5c84a7f972cbbdea9df615fc4ff9159`：`docs/speculative.md`，`common/speculative.cpp`，`common/sampling.cpp`，`tools/server/server-context.cpp`。
- DeepSpec `005e03b81cec38b7da6399833d609ee89a2587f2`：`deepspec/modeling/dspark/common.py` 的 `create_dspark_attention_mask` 与 `create_position_ids`。
- 本地测量循环：`crates/minicpm-v/src/dspark.rs`，`crates/minicpm-v/src/bin/minicpm-v.rs`，`crates/minicpm-v/src/generate.rs`，`crates/minicpm-v/src/config.rs`，以及 `models/MiniCPM-V-4.7-35B-A3B-int4/generation_config.json`。
