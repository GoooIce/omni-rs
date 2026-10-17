# 零阶训练：哪些方法适合当前 Makepad

2026-10-17。范围是当前 Makepad 本地版已经在跑的推理模型，以及只靠函数值、不靠反向传播的优化方法。对照的是论文、作者维护的算法页和本仓库源码。没有改推理，也没有在本机跑训练。

**结论：先只在 Hy-MT2-7B 上做低维连续前缀，用 CMA-ES 在随机子空间里优化，权重保持冻结。** 这一路和现有前向推理同内存。优化器状态是 \(d\) 维均值、步长和 \(d\times d\) 协方差；\(d=500\) 时协方差有 25 万个数，相对 7B 权重仍然很小。用户改过的译文就能当监督。全参数 MeZO、Sparse-MeZO、LOZO、ZO-Adam、对角 Hessian、张量列车适配器、进化策略种群、有限差分、前向梯度（JVP）和 Forward-Forward 都不适合作为第一刀。状态表征强化学习、RLHF、DPO 和 GRPO 也不适合作为第一刀：前三者要在策略或编码器上做反向，GRPO 还要为一题采样一组完整输出再对策略求对数概率梯度。语音、图像和音乐侧目前也没有对得上的监督和预算。

## 当前运行时实际允许什么

Makepad 是推理界面。模型加载和推理共用一个执行名额，停止靠原子标志，当前运算结束后返回。见 [Makepad 说明](../prototypes/makepad/README.md)。

桌面端依赖的 Burn 是 0.21，特性是 `std` 与 `fusion`；macOS 上是 Metal / ANE，其他原生平台是 WebGPU。[`prototypes/makepad/Cargo.toml`](../prototypes/makepad/Cargo.toml) 没有打开 `autodiff` 或 `train`。本仓库 Burn 0.21 的说明把 Autodiff 定义成包住另一个后端的装饰器，用法是 `backward()`。[vendor/burn/README.md](../vendor/burn/README.md) 在供应商源码里搜不到 `jvp` 或 forward-mode。Burn 0.21.0 的 `burn-autodiff` README 原句是 “For now only first order reverse mode autodiff is supported.” 见 [该标签下的 README](https://github.com/tracel-ai/burn/blob/v0.21.0/crates/burn-autodiff/README.md)。因此现成能力是前向求值和采样，没有可以打开的前向模式自动微分。

权重从 safetensors 读入后变成设备上的 FP16 / FP32 张量再做前向。翻译路径每步已经把词表 logits 读回主机再采样，见 [`crates/hy-mt/src/scheduler.rs`](../crates/hy-mt/src/scheduler.rs) 与 [`crates/hy-mt/src/sampling.rs`](../crates/hy-mt/src/sampling.rs)。交叉熵可以加在这份已经读回的向量上，不必为了损失再做一次反向。

常驻或按需加载的模型是：

| 模型 | 本地事实 | 用户能给出的信号 |
| --- | --- | --- |
| Confucius4-R2T2 | 检查点约 4.08 GB，流式转写 | 改过的转写 |
| Hy-MT2-7B | 7B 翻译，分页 KV，Radix 前缀缓存 | 改过的译文 |
| Breeze-TTS-2 | 文本 26 层 / 1152 维，首路语音 28 层 / 2048 维；克隆走参考编码器 | 参考 WAV、声音描述、试听选择 |
| MERT2 / SheetSage2 / YuE2-3B / VAE | 存储元素数见下 | ABC、风格、歌词 |
| Qwen-Image-2.1-Turbo | FP16，固定 8 步去噪；1024² 约 51 s | 提示词和参考图 |

音乐模型的存储元素数来自 [音乐模型调研](music-model-research.md)：SheetSage2 适配器与解码器 57,174,553（F32），MERT2 632,429,312（F32），YuE2-3B 3,630,684,224（BF16），VAE 132,616,130（F32）。图像耗时来自 [根目录 README](../README.md)。TTS 宽度来自 [Breeze-TTS-2 接入说明](breeze-tts.md)。

Radix 前缀缓存按文本 token 前缀索引 KV 页，见 [GLOSSARY.md](../GLOSSARY.md)。连续前缀是嵌入，不是词表 token。它要单独保存，不能指望现有文本前缀树直接命中。

## 方法对照

下表的“每步前向”按论文算法计数。一次前向在这里指整段被优化的模型计算；扩散模型的一步去噪不是一次前向。

| 方法 | 机制 | 每步前向 | 额外状态 | 论文里的可训规模 | 要不要改网络或要 JVP | 对当前桌面 |
| --- | --- | --- | --- | --- | --- | --- |
| SPSA / ZO-SGD | 沿同一个随机方向做两次函数值差分，估计梯度 | 2，与维数无关 | 经典写法要存整条扰动 | 任意维，理论速率随维数变差 | 否 | 只适合套在很小的侧车参数上 |
| MeZO | 用随机种子就地重放扰动，把 ZO-SGD 的内存压到推理水平 | 2（\(n\)-SPSA 为 \(2n\)，默认 \(n=1\)） | 推理内存；按矩阵扰动时再加最大矩阵 | 350M–66B；也测了 LoRA 和 prefix | 否 | 全参数不适合交互；prefix 可作备选 |
| HiZOO / HiZOO-L | 用第三次前向估计对角 Hessian，当作预条件 | 3 | 全对角是 \(O(d)\)；HiZOO-L 比 MeZO 多不到 10% 内存 | 同样到 66B | 否 | 全参数的对角状态放不下；步数收益换不来桌面交互 |
| AdaZeta | 张量列车适配器缩小可训维数，再按 epoch 增加查询次数 | 每查询 2 次，查询数随 epoch 增加 | Llama-2-7B 的 SST-2 峰值 14 GB | 适配器参数比普通矩阵少 80 倍以上 | 要在每层注意力和 FFN 后插入新模块 | 工程量大，留到前缀方案不够用时 |
| ZO-Adam | 用随机梯度估计替换 Adam 的梯度，保留矩估计 | 2（查询数 \(q=1\)） | 全参数 OPT-13B、MultiRC、batch 4：峰值 158 GB、2 张 A100；同表 ZO-SGD 为 64 GB、1 张 | 全参数微调，20,000 步对一阶 625 步 | 否 | 矩状态比推理内存更大 |
| Sparse-MeZO | 只扰动幅度落在稀疏掩码里的权重，掩码在前向时现算 | 2 | 高效实现与 MeZO 同量级；LLaMA-7B MultiRC 上 14.6 GB，全参数微调 158.6 GB | 已有浮点权重的子集；稀疏度 0.5–0.8，文中 0.8 通常更好 | 否 | 仍是整模型前向，步数仍是数千 |
| LOZO | 用低秩扰动矩阵做两点差分，同一子空间连用若干步 | 2 | 存储与 MeZO 相同；动量只留低秩因子 | 更新的是全参数矩阵，估计秩为 \(r\)；实验到 OPT-66B | 否 | 步数大约减半，仍是大模型监督微调 |
| BBT | 冻结模型，CMA-ES 只优化随机投影后的连续 prompt | 每个候选 1 次 | 子空间默认 \(d=500\)；状态含 \(d\times d\) 协方差。Table 2 种群 20，消融用 \(\lambda=4+3\log d\) | 默认预算 8,000 次调用；prompt 长 50 | 输入侧拼一段嵌入 | **第一选择** |
| BBTv2 | 每一层拼一段 deep prompt，逐层轮流做 CMA-ES | 每个候选 1 次 | 每层一个 \(d=500\) 的 CMA-ES；RoBERTa-large 为 24 层 × 50 token | 同样 8,000 次、种群 20；CPM-2 上可训参数表记为 4.8K | 每层前向都要拼 prompt | 输入侧前缀不够时的下一档 |
| 前向梯度 | 一次前向模式 JVP 得到无偏的方向梯度 | 1 次带切向量的前向 | 切向量与最大激活；MeZO 认为要存 \(z\)，内存略高于两倍推理 | 论文示例是较小网络 | 每个算子都要 JVP | 当前 Burn 路径没有这个算子 |
| Forward-Forward | 正样本、负样本各一次前向，每层用活动平方和当 goodness | 2，但是逐层局部目标 | 要改每一层的目标 | MNIST 一级的小网络 | 要改网络结构 | 排除 |
| OpenAI ES | 一批扰动参数各自做完整前向，只回传标量回报 | 种群大小次 | 论文实现扩到一千个以上并行 worker | MuJoCo / Atari 策略 | 否，但要并行多份前向 | 排除 |
| 有限差分 | 每个坐标单独扰动 | \(2p\) | 小 | 低维 | 否 | 排除 |

## 为什么先做 Hy-MT 的低维前缀

翻译是这里唯一同时满足四点的任务：模型是解码器式语言模型，前向已经吐出 logits，用户改译文就得到成对监督，而且一次翻译是有界的文本前向，不是一段歌或一张 2K 图。

Black-Box Tuning 把连续 prompt 写成 \(\mathbf{p}=\mathbf{Az}+\mathbf{p}_0\)。\(\mathbf{z}\in\mathbb{R}^{d}\) 由 CMA-ES 更新，\(\mathbf{A}\) 冻结。默认 \(d=500\)，搜索盒 \([-5,5]^{d}\)，prompt 长度 50，预算 8,000 次推理 API。损失可以是交叉熵、hinge 或负准确率，所以目标不必可微。原文实验是 RoBERTa-large 的分类，并写明生成式模型留作后续。见 [Sun et al. 2022, arXiv:2201.03514](https://arxiv.org/abs/2201.03514)。

BBTv2 补上了这一档。它在每一层隐藏状态前拼 prompt，依靠残差把优化拆成逐层子问题，每层仍用 \(d=500\) 的 CMA-ES，种群 20，预算仍是 8,000 次调用。同一预算下它覆盖 encoder-only、decoder-only 和 encoder-decoder：GPT-2 large 的 SST-2 从 BBT 的 75.53 到 BBTv2 的 83.72；约 11B 的中文 CPM-2 上，BBTv2 用表中的 4.8K 可训参数，ChnSent 86.4、LCQMC 59.1，对照的全参数微调是 86.1 和 58.8。任务仍是带模板的分类或匹配，不是开放翻译。见 [Sun et al. 2022, arXiv:2205.11200](https://arxiv.org/abs/2205.11200)。

这和桌面约束对齐的方式：

1. 基础权重只读。落盘的是 \(\mathbf{z}\) 和生成 \(\mathbf{A}\) 的种子，或者投影后的前缀张量。BBT 的用户侧内存与模型规模脱钩。
2. 每次查询占现有的那一个推理名额，沿用现有停止标志。8,000 次 7B 前向是后台任务，不能插进一次打字或一次转写。
3. 损失用用户改正后的目标 token 对已读回的 logits 做交叉熵。样本是会话里累计的句对，不是论文里的 16-shot 分类集。论文没有测开放机器翻译，这一步是外推。
4. 子空间维数有现成的消融。BBT 在 \(\{100,200,500,1000\}\) 里看到 \(d=100\) 盖不住好解、\(d=1000\) 泛化变差。BBTv2 看到 \(d>100\) 后收益变平。第一版固定 \(d=100\) 或 \(d=500\)，不要扫全参数。
5. 输入侧前缀只改嵌入拼接。BBTv2 的逐层 prompt 要动每一层前向，作为第二步，而不是第一刀。

CMA-ES 维护多元正态：均值 \(\mathbf{m}\in\mathbb{R}^{d}\)、步长 \(\sigma\) 和协方差 \(\mathbf{C}\in\mathbb{R}^{d\times d}\)。BBT 写明这三项都在迭代中更新。默认实验表的种群是 20；子空间消融才把种群设为 \(\lambda=4+3\log d\)。\(d=500\) 时每一代大约二十次完整翻译前向。见 [Sun et al. 2022, arXiv:2201.03514](https://arxiv.org/abs/2201.03514) §3.2 与 Table 2。

## 同一条前缀上，MeZO 是备选优化器

MeZO 把 SPSA 估计做成就地更新。随机种子 \(s\) 重放 \(z\sim\mathcal{N}(0,I)\)，所以不必保存整条 \(z\)。一步是：加 \(\epsilon z\)、前向、减 \(2\epsilon z\)、前向、加回 \(\epsilon z\)，再用 \((\ell_+-\ell_-)/(2\epsilon)\) 沿 \(z\) 下降。默认一次估计用一个方向。按矩阵而不是按标量扰动时，额外内存等于最大权重矩阵。见 [Malladi et al. 2023, arXiv:2305.17333](https://arxiv.org/abs/2305.17333) 的 Algorithm 1 与 Definition 1。

论文里的预算说明它不能直接拿来做交互式全参数训练：

- 预备实验写明：MeZO 只有在使用 prompt 时才优化得动。下文实验都带 prompt。见该文 §3 与 Appendix A。
- RoBERTa-large（350M）跑 100,000 步，Adam 微调跑 1,000 步。
- OPT-13B / 30B / 66B 跑 20,000 步，对照微调是 5 个 epoch（文中按 625 步计）。Appendix F.6 对 OPT-30B 的墙钟拆开写：MeZO 步数是标准微调的 32 倍，每步快 7.74 倍，GPU 数少 8 倍，总 GPU 时大约一半。这是 A100 上的实现对比，论文同时写明墙钟优势取决于实现。
- OPT-13B 上，MeZO 在 11 项里有 7 项落在 Adam 微调的 1% 以内。生成任务 SQuAD、DROP 用 teacher forcing 训练、贪心解码评估，MeZO 能超过零样本、上下文学习和线性探测。
- 全参数、LoRA、prefix 三者性能接近，收敛速度也接近。理论把减速因子放在损失 Hessian 的局部有效秩 \(r\) 上，而不是参数个数 \(d\)（Assumption 1、Theorem 1）。因此把 MeZO 从全参数换成 prefix，**不会**按论文结论把 20,000 步收成几百步。prefix 的价值是扰动和检查点都落在小张量上。
- 不可微目标成立：分类直接优化准确率，SQuAD 的 prefix 版优化 F1。用户的“改对了 / 没改对”可以做成标量，不必可微。

Spall 维护的 SPSA 页说明，同时扰动每步只要两次目标函数测量，有限差分要 \(2p\) 次；在其条件下，同样迭代次数能达到同样统计精度。见 [JHU APL SPSA](https://www.jhuapl.edu/SPSA/)。MeZO 注明原始 SPSA 扰动的是 \(1/z\)，他们改成高斯 \(z\)，与后续理论工作一致。

若前缀已经存在，MeZO 比 CMA-ES 好写：没有种群和协方差，两次前向得到一次更新。它适合损失是编辑距离、偏好这类不可微标量的时候。已知预算仍应按论文的万步量级规划，直到本机小样本曲线证明可以提前停。全参数就地扰动还会改掉界面正在用的那份 FP16 权重，和“检查点只读、失败可回退”冲突。前缀侧车没有这个问题。

HiZOO 在 MeZO 的两次前向之外再加一次未扰动前向来估计对角 Hessian，默认采样数 \(n=1\)。摘要报告 SST-2 上相对 MeZO 有 8 倍步数收益，66B 上绝对精度最多高 5.1 个点。全对角是 \(O(d)\)，7B 再存一份对角就等于再放一份参数。HiZOO-L 用秩 1 因子把增量压到比 MeZO 多不到 10% 内存。见 [Zhao et al., arXiv:2402.15173](https://arxiv.org/abs/2402.15173)。步数更少，但三次全模型前向加上额外状态，仍然不是桌面第一刀。

AdaZeta 针对 MeZO-LoRA 在 Llama-2-7B 上的发散。张量列车适配器接在每个块的注意力和 FFN 之后，层归一化冻结，可训参数减少 80 倍以上；查询次数按 epoch 次线性增加，中等模型上查询数固定为 1。Llama-2-7B 的 SST-2 峰值内存 14 GB，相对全参数微调大约 8 倍下降。一条曲线上，MeZO-LoRA 约 6,000 步到达损失 0.4，AdaZeta 不到 1,000 步。见 [Yang et al. 2024, arXiv:2406.18060](https://arxiv.org/abs/2406.18060) 与 Table 2。它证明“可训维数”在他们的分析里仍然进入收敛界，这和 MeZO 的有效秩观点并列，两篇都只覆盖带提示的分类和抽取式问答。张量列车要新的收缩内核，不该先于一段普通前缀。

后补的三篇全参数变体也没有把第一选择从低维前缀上挪开。

[ZO-Bench](https://arxiv.org/abs/2402.11592) 在全参数 OPT-13B、MultiRC、batch 4、平均上下文 400 上测峰值内存：ZO-SGD 64 GB、1 张 A100；Forward-Grad 134 GB、2 张；ZO-Adam 158 GB、2 张；FO-Adam 245 GB、4 张。ZO 跑 20,000 步，一阶跑 625 步。ZO-Adam 的矩估计把零阶的内存优势吃掉，还高于同表的 FO-SGD（148 GB）。查询预算消融是另一组设置：LoRA 微调 OPT-1.3B 的 SST-2 上，\(q>500\) 时 Forward-Grad 比 ZO-SGD 高 1 到 2 个点，计算随 \(q\) 线性增加。

[Sparse-MeZO](https://arxiv.org/abs/2402.15751) 把扰动限制在动态稀疏掩码上。1-bit 存掩码会多占内存，所以掩码在前向时现算。LLaMA-7B 的 MultiRC 上，高效实现 14.6 GB，全参数微调 158.6 GB。收敛图的一个例子是约 5,000 步到 70% 准确率，MeZO 要 17,500 步；文中接着给出 RTE 约 3.5 倍、BoolQ 约 3 倍加速。稀疏度从 0.5 到 0.8 有收益，多数任务上 0.8 更好。这仍然是对整份浮点权重做两次前向，而且要写回被掩住的坐标。

[LOZO](https://arxiv.org/abs/2410.07698) 用低秩因子 \(U,V\) 构造扰动，使梯度估计保持低秩，存储开销与 MeZO 相同。OPT-66B 的 WIC 上，达到同一训练损失大约用 MeZO 一半的 epoch。更新对象仍是全参数矩阵，实验协议仍是 13B–66B 的有标签微调。

## 明确排除

**前向梯度。** Baydin 等人定义 \(g(\theta)=(\nabla f(\theta)\cdot v)\,v\)，由一次前向模式的 Jacobian-vector product 得到，证明无偏。他们估计前向模式代价大约是原函数的 1–3 倍，反向模式大约 5–10 倍，并在一些网络上看到相对反向传播最多约 2 倍的墙钟加速。见 [Baydin et al. 2022, arXiv:2202.08587](https://arxiv.org/abs/2202.08587)。这是自动微分的前向模式，不是零阶。MeZO 第 2 节写明：同一估计若用 JVP 实现，必须存下 \(z\)，内存略高于两倍推理，而且既有 JVP 训练的经验效果有限。ZO-Bench 把这件事量到了 OPT-13B：全参数 Forward-Grad 峰值 134 GB，要 2 张 A100。Burn 0.21 只有一阶反向模式，当前 Metal / ANE 内核也没有切向量版本。

**Forward-Forward。** Hinton 用两次前向替换反向：正数据提高每层 goodness，负数据降低它。goodness 可以是整流活动的平方和。论文的对象是皮层模型和低功耗模拟硬件，并写明在所测小问题上比反向传播慢、泛化也差一点，电力充足时不会取代反向传播。文中同时指出，靠扰动权重再关联回报的强化学习，学习率要随被扰动变量数成反比，无法和反向传播在数百万到数十亿参数上竞争。见 [Hinton 2022, arXiv:2212.13345](https://arxiv.org/abs/2212.13345)。接入现有 Transformer、扩散和 codec 都要重写层次目标。

**进化策略种群。** Salimans 等人把 ES 用作强化学习的黑箱替代。官方实现用公共随机数，worker 之间只传标量，实验扩到一千个以上并行 worker；摘要报告 3D 人形行走约 10 分钟、多数 Atari 约一小时。见 [Salimans et al. 2017, arXiv:1703.03864](https://arxiv.org/abs/1703.03864) 与 [官方实现说明](https://github.com/openai/evolution-strategies-starter)。桌面只有一个推理名额，一次候选就是一次 7B 或 3B 前向。

**有限差分。** SPSA 页给出的对比是：维数 \(p\) 时，坐标逐个扰动每步 \(2p\) 次测量。对 7B 或即便对 \(d=500\) 的前缀，这都比同时扰动或 CMA-ES 多一个到数个数量级的前向。

**打开 Burn Autodiff 做普通微调。** MeZO 在 OPT-13B 上测得全参数 Adam 的峰值内存约是推理的 12 倍，PEFT 仍约 6 倍，因为激活仍然要留下来。这正是当前统一内存和单名额调度躲过的成本。零阶前缀不走这条路。

## 其他模型为什么先不动

**转写。** 用户改正文字可以变成字错误率这类标量，MeZO 已表明 F1 / 准确率能当目标。但一次估计是两次完整声学前向，会占住麦克风正在用的那个名额。没有打开的零阶论文在流式 ASR 上做过实验。若翻译前缀跑通，再把同一优化器放到离线的“已保存纠正”上，而不是塞进实时会话。

**语音合成。** 参考编码器、声音描述和“使用此音色”已经在不改权重的情况下换音色，见 [Breeze-TTS-2 接入说明](breeze-tts.md)。试听喜欢与否是极稀疏的标量。MeZO 的不可微实验仍使用每类 512 例或 1,000 条 SQuAD。用零阶去动 28 层语音主干，对不上现有产品信号。

**音乐。** YuE2-3B 一次生成是长序列前向。SheetSage2 的适配器是这组里最小的可训面（约 5.7×10⁷ 个存储元素），但损失会是 ABC 或 MIDI 的结构化距离，打开的论文都没有这个目标。

**图像。** 一次生成是固定 8 步去噪。零阶的一步至少是两条完整轨迹。README 记录 1024² 约 51 秒，默认 2K 更长。两条轨迹才能得到一个标量，而界面上没有成对的“目标图”。

## 建议的最小接口

只加在翻译前向旁边，不改检查点布局：

- 一份可有可无的输入侧连续前缀，默认关闭。关闭时数值与现在一致。
- 优化状态是子空间向量、投影种子和 CMA-ES 的分布参数。停止或失败丢弃未提交的候选，已提交前缀才替换会话里的那一份。
- 监督来自用户明确保存的句对。每查询一次完整前向，查询之间检查现有取消标志。
- 训练任务与转写、合成、出图互斥，因为它们共用一个执行名额。

层数、隐藏维和 prompt 的具体形状以该检查点的 `config.json` 为准。仓库内没有把 Hy-MT2-7B 的隐藏维写进文档，本文不假设一个数。

## 表征强化学习

这个词在一手文献里指的是状态表征学习：从高维观测学一个低维状态，再在这个状态上学策略或价值函数。它不是零阶优化，也不是给冻结语言模型加一段前缀。

Lesort 等人把状态表征学习定义成：学到的特征维数低、随时间变化、并且受智能体动作影响，用来刻画动作在环境里造成的变化。形式是 \(s_t=\phi(o_{1:t})\)，可选地再吃进动作和奖励。综述覆盖的是与环境交互的方法，实验主要在仿真控制里，少量在真实机器人上。前向模型的做法是用 \(s_t\) 和 \(a_t\) 预测下一状态，预测误差经转移模型和编码器反向传播。见 [Lesort et al. 2018, arXiv:1802.04181](https://arxiv.org/abs/1802.04181) 摘要、§2.1 与 “Learning a forward model”。

Echchahed 与 Castro 的综述把范围限定在无模型、在线的深度强化学习。问题仍是马尔可夫决策过程 \((\mathcal{S},\mathcal{A},P,R,\gamma)\)，策略 \(\pi\) 把状态映到动作分布，目标是折扣回报。他们把表征方法分成六类：度量、辅助任务、数据增强、对比、非对比、注意力。编码器 \(\phi\) 的产物交给策略或价值网络。文中点名的价值方法是 DQN，策略梯度方法是 REINFORCE，以及 actor-critic。见 [Echchahed and Castro 2025, arXiv:2506.17518](https://arxiv.org/abs/2506.17518) §2 与 Table 2。

Botteghi 等人的综述讨论的是同一件事的无监督版本：先从观测流里学低维状态，再用它学策略。应用对象是动力系统和控制，例如摆。见 [Botteghi et al., arXiv:2208.14226](https://arxiv.org/abs/2208.14226)。

中文综述《基于表征学习的离线强化学习方法研究综述》把设定收成离线：策略只从历史经验学习，不再在线与环境交互。表征把离线数据压成低维向量，再拿这些向量训练离线强化学习模型。文中的应用是工业、推荐系统和智能驾驶。见 [王雪松、王荣荣、程玉虎，《自动化学报》](https://aas.net.cn/cn/article/doi/10.16383/j.aas.c230546)。离线只是去掉在线采样，价值函数或策略网络的训练还在。

这和当前桌面对不上，有三条独立的原因：

1. Makepad 没有 \((s,a,r,s')\)。转写、翻译、合成、出图都是一次前向任务。用户改正译文是句对或一个标量，不是动作改变环境之后的下一状态。
2. 上面打开的表征方法都把误差送回编码器或策略。Lesort 对前向模型写了反向传播。Echchahed 与 Castro 依赖 DQN、REINFORCE 或 actor-critic。当前 Burn 0.21 桌面路径没有启用自动微分。
3. 已有的“表征”是冻结检查点的一部分：Breeze 的参考编码器、MERT 的 1,024 维帧特征、图像的 VAE 隐变量。它们在推理时被读取，不是用奖励现学的状态编码器。

若把“表征”理解成语言模型对齐里的策略，而不是控制里的状态编码器，打开的三篇仍然要梯度：

- InstructGPT 是三步：监督微调、奖励模型、再用 PPO 对奖励模型优化策略。模型规模实验是 1.3B、6B、175B。见 [Ouyang et al. 2022, arXiv:2203.02155](https://arxiv.org/abs/2203.02155) Figure 2 与 §3。
- DPO 去掉显式奖励模型和强化学习循环，把同一偏好目标写成分类损失，提高偏好回答相对落选回答的对数概率。损失仍然穿过策略的对数概率。见 [Rafailov et al. 2023, arXiv:2305.18290](https://arxiv.org/abs/2305.18290) 摘要与 Figure 1。
- GRPO 去掉与策略同规模的价值网络，对同一题从旧策略采样一组输出 \(\{o_1,\ldots,o_G\}\)，用组内奖励的均值和标准差做基线，再最大化带重要性比率的策略目标。DeepSeekMath 7B 上，GSM8K 从 82.9% 到 88.2%，MATH 从 46.8% 到 51.7%。见 [Shao et al. 2024, arXiv:2402.03300](https://arxiv.org/abs/2402.03300) §4.1.1。一组完整生成再加对数概率梯度，占满当前这一个推理名额，也写不进只读权重。

用户改过的译文可以当成偏好对。能在现有前向上使用它的做法，仍是上一节的低维前缀加 CMA-ES：模型冻结，每个候选一次前向，损失是交叉熵或一个标量。那不是状态表征学习，也不需要 PPO 的价值网络。

## 引用

- Spall, J. C. 维护的 SPSA 说明，Johns Hopkins APL。<https://www.jhuapl.edu/SPSA/>。同时扰动每步两次测量、有限差分每步 \(2p\) 次，以及局部收敛指向 Spall, 1992。
- Sun, Tianxiang, et al. Black-Box Tuning for Language-Model-as-a-Service. arXiv:2201.03514. <https://arxiv.org/abs/2201.03514>
- Sun, Tianxiang, et al. BBTv2: Towards a Gradient-Free Future with Large Language Models. arXiv:2205.11200. <https://arxiv.org/abs/2205.11200>。代码：<https://github.com/txsun1997/Black-Box-Tuning>
- Malladi, Sadhika, et al. Fine-Tuning Language Models with Just Forward Passes. arXiv:2305.17333. <https://arxiv.org/abs/2305.17333>。代码：<https://github.com/princeton-nlp/MeZO>
- Zhao, Yanjun, et al. Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed Zeroth-Order Optimizer. arXiv:2402.15173. <https://arxiv.org/abs/2402.15173>。代码：<https://github.com/Yanjun-Zhao/HiZOO>
- Yang, Yifan, et al. AdaZeta: Adaptive Zeroth-Order Tensor-Train Adaption for Memory-Efficient Large Language Models Fine-Tuning. arXiv:2406.18060. <https://arxiv.org/abs/2406.18060>
- Baydin, Atılım Güneş, et al. Gradients without Backpropagation. arXiv:2202.08587. <https://arxiv.org/abs/2202.08587>
- Hinton, Geoffrey. The Forward-Forward Algorithm: Some Preliminary Investigations. arXiv:2212.13345. <https://arxiv.org/abs/2212.13345>
- Salimans, Tim, et al. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864. <https://arxiv.org/abs/1703.03864>
- Zhang, Yihua, et al. Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark. arXiv:2402.11592. <https://arxiv.org/abs/2402.11592>
- Liu, Yong, et al. Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning. arXiv:2402.15751. <https://arxiv.org/abs/2402.15751>
- Chen, Yiming, et al. Enhancing Zeroth-Order Fine-Tuning for Language Models with Low-Rank Structures. arXiv:2410.07698. <https://arxiv.org/abs/2410.07698>
- Burn 0.21.0 autodiff README. <https://github.com/tracel-ai/burn/blob/v0.21.0/crates/burn-autodiff/README.md>
- Lesort, Timothée, et al. State Representation Learning for Control: An Overview. arXiv:1802.04181. <https://arxiv.org/abs/1802.04181>
- Echchahed, Ayoub, and Pablo Samuel Castro. A Survey of State Representation Learning for Deep Reinforcement Learning. arXiv:2506.17518. <https://arxiv.org/abs/2506.17518>
- Botteghi, Nicolò, Mannes Poel, and Christoph Brune. Unsupervised Representation Learning in Deep Reinforcement Learning: A Review. arXiv:2208.14226. <https://arxiv.org/abs/2208.14226>
- 王雪松、王荣荣、程玉虎. 基于表征学习的离线强化学习方法研究综述. 自动化学报. <https://aas.net.cn/cn/article/doi/10.16383/j.aas.c230546>
- Ouyang, Long, et al. Training language models to follow instructions with human feedback. arXiv:2203.02155. <https://arxiv.org/abs/2203.02155>
- Rafailov, Rafael, et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290. <https://arxiv.org/abs/2305.18290>
- Shao, Zhihong, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300. <https://arxiv.org/abs/2402.03300>

MeZO 把“收敛随参数维数变慢”的经典下界指向 Duchi, Jordan, Wainwright, Wibisono, IEEE Trans. Inf. Theory, 2015，以及 Nesterov and Spokoiny, Foundations of Computational Mathematics, 17:527–566, 2017。本文没有打开这两篇期刊原文，只转述 MeZO 对它们的对照，不用它们的定理常数。Nesterov 的讨论稿 CORE 2011/1 在 <https://dial.uclouvain.be/pr/boreal/object/boreal:69205>，本轮没有逐条核对其中的定理。

## 没有打开的来源

- Nesterov & Spokoiny 2017 的 FoCM 期刊 PDF。按题名在 arXiv 上检索时返回的不是这篇。CORE 2011/1 的条目能定位，本轮没有重读定理。
- Duchi et al. 2015 的 IEEE 原文。
- Spall 1992 的 IEEE 排版页。作者站点提供了 PDF：<https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf>。本轮没有重读引理。两次测量这一操作事实来自作者维护的 SPSA 页。
- Salimans et al. 2017 的 PDF 正文。arXiv HTML/PDF 转换没有给出正文，种群与通信事实来自 arXiv API 摘要和官方仓库 README。
- DPZero（arXiv:2310.09639）只在检索结果里见过摘要，没有打开正文。
- QuZO（arXiv:2502.12346）只作为 Sparse-MeZO 参考文献里的一条出现，没有打开。因此本文不声称量化权重上的零阶微调不存在，只说明上面已打开的微调论文没有 INT4、INT8 或 FP8 权重实验。Sparse-MeZO 里的 1-bit 讨论的是掩码存储，而且该方案被放弃。
