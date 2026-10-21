# DeepInfer：变长校验、步间重叠、设备上采样

2026-10-21。材料只包括 [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) 在 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` 的文档和源码：README、[doc/inference_deployment_qa.md](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md) 的 Speculative decoding 与 Asynchronous scheduling，以及这两处指向的 DFlash、调度器和采样器。没有采用博客或二手综述。下面的步骤按 Python 参考实现和 C++ 调度来写，可以脱离 SM100、昇腾和 TileLang 复述。源码没有给出的公式，文中会直接说明。

**结论：依赖 DeepSeek-V4 DFlash 权重的，是草稿层前向、可选的 Markov 修正、置信度头，以及它们写入设备池的草稿分布。** 置信度的前缀累乘、按吞吐曲线挑草稿前缀、CPU 与 GPU 的交错、拒绝判定和回读清单，都不读这些权重。变长校验决定这一步送去验证的草稿前缀有多长。设备上的拒绝采样决定这条前缀里实际收下哪些 token。调度记的是置信度的前缀累乘，拒绝采样记的是残差估计，两套数各走各的。

## 开关

| 开关 | 源码里的默认 | 作用 |
|---|---|---|
| `DEEPINFER_USE_NEXTN_DECODE` | 5。设为 0 关闭投机解码 | 每个请求一次最多生成、也最多保存这么多个草稿 token（[README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)、[environ.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/environ.py)） |
| `DEEPINFER_ENABLE_MTP_VARLEN` | 关。README 写明要设为 1 才按置信度调度验证长度 | 打开且存在解码图时，总预算和各请求长度按下面的接受率来选。曲线为空时，每个请求验满现有草稿（[README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)、[environ.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/environ.py)） |
| `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | NORMAL 与 COMPLETION 引擎在参数初始化时默认写成 1；PROMPT 默认 0。README 称异步调度默认开启 | 把上一步的后处理留到下一步，和这一步的前向叠在一起（[arg_utils.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/arg_utils.py)、[engine_type.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/engine_type.py)、[README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)） |

部署问答里的生产描述是：草稿一次生成 5 个，再按硬件吞吐曲线和每个草稿 token 的接受率选出一段前缀来验证。开源仓库里变长开关默认是关的；下面第 1 节是这个开关打开之后的步骤。问答给出的幅度是 EP64/DP64 上的测量：动态长度把每步验证的草稿从 5 降到大约 3.2–4.4，平均接受长度仍在 3.2 附近，大约 60 路并发时估计吞吐高 4–7%。这些数字只出现在问答里，源码没有重算（[inference_deployment_qa.md](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)）。

## 1. 变长校验

调度用的接受率，是置信度沿草稿位置的累乘。设备上拒绝采样另有一份接受概率，留在第 3 节。

1. **依赖 DFlash 权重。** 草稿层一次前向产出最多 `nextn` 个位置的 hidden。输入的第 0 列是本步已经确定的 token，其余列是检查点里的 noise token。这些层读的主缓存，由主模型指定层的特征经 DFlash 投影写成，并且写在主模型前向里，当时还没有本步的接受结果。随后的逐位置循环不再跑草稿层：每个位置先做输出归一化，用主模型的 lm head 得到 logits；检查点里的 `dspark_markov_rank` 大于 0 时，再用 DFlash 的 Markov 表，按「上一个已采样 token」对 logits 做一阶修正。本仓库只从检查点读取这个秩，没有另写一个数字（[dflash.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py) `forward_drafts`、`sample_drafts`、`update_main_cache`、[hugging_face/config.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/hugging_face/config.py) `_build_dflash_cfg`）。
2. **依赖 DFlash 权重。** 同一位置上，置信度头把「归一化前的 hidden」投成一个标量。秩大于 0 时，输入再拼上当前条件 token 的 Markov 向量；否则只有 hidden。标量除以该位置的温度，夹到 [-60, 60]，再做 sigmoid。温度来自 `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE`，默认 1.0；可以写成与 `nextn` 等长的一串，按草稿位置取用。小于 `1e-5` 的温度会被抬到这个下限。HF DFlash 固定带这个头。包装函数的说明写成 sigmoid(投影 / 温度)；夹紧出现在参考实现和设备实现里。注释把结果称为每个草稿 token 各自的接受率，值域写成 [0, 1]。sigmoid 的开区间是 (0, 1)。本步发出的 token 含停止符时，该请求下一轮的置信度整行写成 0（[dflash.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py) `ConfidenceHead`、[tilelang_ops/dflash.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/dflash.py) `confidence_head_fused_ref`、[heads.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/heads.py) `AcceptRatePredictor`、[sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler.py)）。
3. **不依赖 DFlash 权重。** 记账时把独立置信度做成前缀累乘。位置 j 的调度接受率是位置 0 到 j 的置信度之积，初值是 1。注释写明：模型返回的是每个 token 独立的置信度，使用前要累乘。累乘结果放进 `accept_rates`，上一轮的累乘结果挪到 `prev_accept_rates`。没有置信度张量时不做累乘；调度若看到空列表，有草稿就每个位置填 1.0，没有草稿就填 0.0（[scheduler.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp) `process_MTPInfo`、[sequence.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/sequence.hpp) `MtpState`、[mtp_varlen.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp) `push_rates`）。
4. **不依赖 DFlash 权重。** 每个请求先占 1 个必验位置，留给本步要新采样的目标 token。额外名额才分给草稿。解码调度总会走进这段选择；变长开关关掉时不会去测曲线，曲线为空，每个请求的查询长度等于 1 加它现有的草稿个数，整段草稿都验证。变长开着但没有解码图时，曲线同样为空，结果相同（[mtp_varlen.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp) `schedule`、[scheduler.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp) `_schedule_decode`、[worker/base.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/base.py) `warm_up_model`）。
5. **不依赖 DFlash 权重。** 有曲线时，先把全部 (接受率, 请求号) 按接受率从高到低排；接受率相同则请求号更小者优先。源码要求后面位置的置信度不能反过来决定前面那个草稿是否被验证。累乘满足这一点：位置 i 的分数只含位置 0 到 i 的置信度。置信度落在 (0, 1) 时，同一请求内的累乘严格下降，所以排进来的位置只能是该请求草稿的前缀。源码没有把「边际收益」写成导数。它从「只保留每个请求那 1 个必验位置」出发，按这个顺序逐个加入草稿位置：期望 token 数加上该位置的累乘接受率，批大小加 1。耗时取「不小于当前批大小的曲线点」的基线毫秒，加上非负斜率乘该点的图尺寸，再加上采样耗时估计。吞吐是期望 token 数除以这个毫秒数。取得分最高的总预算。斜率是近期前向残差的两层中位数，没有足够样本时为 0，并且不允许为负。采样耗时优先用相同请求数最近若干步的中位数（窗口 9，至少 5 个样本），否则用全局中位数，再没有就当 0。曲线本身是启动时对各解码图尺寸计时得到的 (批大小, 毫秒) 列表，批大小严格递增；也可以读缓存。源码没有给出另一套总预算公式（[mtp_varlen.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp)、[model_runner.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/model_runner.py) `profile_decode_perf`）。
6. **不依赖 DFlash 权重。** 总预算和各请求长度可以读不同一轮。配成对时，总预算读 `prev_accept_rates`（上上轮），各请求长度读 `accept_rates`（上一轮）。异步且最新一次前向还没把接受率送回 CPU 时，总预算改读 `accept_rates`，注释仍把它当作上上轮；各请求长度也读这份，于是比配成对时更旧一轮。配对与否看 `schedule_cnt` 与 `process_cnt` 是否相等，二者最多差 1。各请求的查询长度等于 1，再加上它有多少个位置落在「总预算减请求数」这个名额里。送进模型的是草稿序列开头的这么多个 token，再加末尾一个新采样位置。无重叠时，输入从草稿向量开头拷 `seqlen_q - 1` 个（[mtp_varlen.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp)、[zos.cpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/zos.cpp) `prepare_token_inputs`）。
7. **不依赖 DFlash 权重。** 异步并且变长打开时，CPU 在 `schedule(N+1)` 还拿不到第 N 步的置信度。CPU 留下的总数是总预算；各请求长度随后在设备上重算，写回页锁定的主机缓冲，采样前同步这块回读，再改 CPU 侧元数据。重算的参考实现是：按重映射取上一步留在设备上的置信度，新请求整行当 0；沿草稿维累乘；所有位置按累乘值从高到低排，平局时扁平下标更小者优先（同一请求里就是更靠前的位置）；名额仍是总预算减请求数；查询长度等于 1 加该请求的入选个数。部署问答把设备上的这一步概括成累乘、全局取前若干名、以及把上一步的行散射到这一步。问答说 CPU 只固定总预算；CPU 先前按旧接受率填的各请求长度，会在这次重写里被盖掉（[inference_deployment_qa.md](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)、[schedule_mtp_varlen.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py) `compute_seqlens_q_by_confidence_rates_ref`、[worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py) `_correct_seqlens_qk`）。有重叠时，继续着的请求从设备上的上一轮结果取 token：位置 0 是上一步采样出的 token，位置 j>0 是上一步预测矩阵的第 j-1 列，并且只取 j 小于新查询长度的那些列，所以仍是预测序列的前缀（[gather_input_ids_zos.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/gather_input_ids_zos.py)）。

CPU 排序的键是「接受率、请求号」，不含位置。同一请求里出现相等接受率时，源码没有规定这些位置在 CPU 排序中的先后。它最后用的也是入选个数，校验时按这个长度取前缀。全 0 的行会出现这种并列；GPU 参考实现的平局规则会先留下更靠前的位置。源码没有再写一条公式，证明 CPU 与 GPU 在并列时选出同一个下标集合。

## 2. 步间重叠

这一节的顺序不依赖 DFlash 权重。变长打开时，顺序里会插入第 1 节第 7 步的长度重写；那个重写读的是上一步已经留在设备上的数。

没有重叠时，解码第 N 步是一条直线：调度 N、前向 N、采样 N、后处理 N。调度和后处理是 CPU 逻辑。后处理之前要同步，才能拿到采样 N 的结果，于是 GPU 在后处理 N 和调度 N+1 期间空着。异步调度要去掉这段空档（[inference_deployment_qa.md](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)）。

重叠打开时，一次解码在代码里的交错是：

1. **CPU，调度本步。** 选出请求，写出临时的各请求长度和总 token 预算，把每个请求的在途长度加上 `nextn + 1`。这一步发生在本步前向入队之前。上一步的采样完成事件还没有被等待，上一步的采样这时还可以留在设备上跑（[llm_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/llm_engine.py) `step`、[scheduler.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp) `start_inflight_step`）。
2. **GPU，前向之前。** 上一步有真实请求，并且和本步至少有一个相同请求时：把重映射和临时长度拷到设备。主机上的键长度在 `nextn > 0` 时先减去了查询长度。设备上若变长打开，按第 1 节第 7 步重写查询长度，再把新的查询长度拷回页锁定缓冲并记下事件。`nextn > 0` 时，键长度先加回查询长度；重映射能对上上一步的请求，再减去 `nextn - 已接受草稿数`。拷入、修正、查询长度的拷出都排在当前流上。两步没有相同请求时，整段修正跳过。变长关闭时没有查询长度的回读（[worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py) `_correct_seqlens_qk`、[compute_seqlens_k_zos.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/compute_seqlens_k_zos.py)、[zos.cpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/zos.cpp) `prepare_zos_inputs`）。
3. **GPU，启动本步主模型前向。** 启动路径不允许同步（[model_runner.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/model_runner.py) `execute_main_model`）。
4. **CPU，与本步前向重叠。** 后处理上一步：只等待上一步的采样完成事件，然后做 MTP 记账、把接受结果写回请求、构造输出。该事件记在上一步采样末尾，早于本步的长度修正和前向，所以这次等待不会把本步前向等完（[worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py) `_post_process_step`）。
5. **CPU。** 若第 2 步记了长度回读事件，等到它完成，再用回读的查询长度改本步的 CPU 元数据。事件位于前向之前，等待返回时前向可以已经在跑（[worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py)、[zos.cpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/zos.cpp) `update_metadata_for_sampling_zos`）。
6. **GPU，启动本步采样。** 含对刚选出的草稿前缀做拒绝采样，以及再跑一轮 DFlash 得到下一轮草稿。启动路径不允许同步。需要给主机的张量在这一步以非阻塞方式开始回读，此刻还不消费（[model_runner.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/model_runner.py) `execute_sample`、[sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler.py)）。
7. **CPU，采样已入队之后。** 跑采样前挂钩：释放推迟对象、按引擎保存 KV、推进结构化输出、结束已完成请求，并做一次垃圾回收。源码没有把这个挂钩写成纯 CPU。随后把采样完成事件记到当前流上，位置在采样工作之后（[worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py)、[llm_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/llm_engine.py) `pre_sampler_sync`）。
8. **CPU，本步后处理不在这里做。** 解码且重叠打开时，本步状态留到下一步的第 4 步。预填充，以及关掉重叠的解码，是当场后处理（[worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py) `execute_model_step`）。

下一步的调度紧接在第 7 步之后、下一步前向入队之前，因此它和本步仍在设备上的采样重叠。后处理在结束在途计数时，若只有这一笔在途就清零；若下一笔已经加上，就留下下一笔的 `nextn + 1`。同一段后处理里，缓存长度先按本步查询长度前进，再改成实际收下的 token 数（[sequence.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/sequence.hpp) `SequenceScheduleState`、[scheduler.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp) `process_outputs`）。

部署问答把重叠写成：GPU 跑前向 N 和采样 N 时，CPU 同时跑调度 N+1、启动前向 N+1、后处理 N−1、启动采样 N+1。代码里的相位是：后处理 N−1 夹在前向 N 与采样 N 之间；调度 N+1 在采样 N 入队之后、前向 N+1 入队之前。问答的目标是 GPU 中间不空出来；逐步顺序以 worker 的这 8 步为准。

## 3. 设备上采样

采样在设备上做完决定，再只把调度和输出要用的张量读回。完整词表上的 logits 不读回。

设备上的决定，针对每个请求已经排好的查询行：除最后一行外都是待验证的草稿，最后一行是新采样（[tilelang_ops/sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/sampler.py) `prepare_verification_metadata_ref`）。

1. **不依赖 DFlash 权重；logits 来自这一步的主模型。** 设备上做温度、先 top-p 再 top-k、log-softmax，并从目标分布抽出一个 token。`DEEPINFER_SAMPLER_FORCE_TOPK` 默认为 0，表示在完整词表上做。设为正数时，先收成这么宽的候选，再把池里的草稿对数概率对齐到这组候选上。源码没有规定这个宽度的具体取值（[sampler_utils.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler_utils.py) `process_sampling`、[fused_sample.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/fused_sample.py) `fused_sample_ref`）。
2. **判定不读权重；它读的是上一步 DFlash 写入设备池的草稿 token 和对数概率。** 验证行上，取目标与草稿在该 token 上的对数概率之差，上截为 0，再指数化，和一个均匀随机数比较。比较通过则留下草稿 token，否则从残差权重里重采样。草稿 token 编号为负时直接拒绝。新采样行保留第 1 步抽出的 token。残差权重是 `max(目标概率 - alpha × 草稿概率, 0)`，`alpha = exp(-5e-5)`。注释说这个惩罚是为了避免拒绝采样采不出 token，并给出加了惩罚之后输出分布与目标分布的偏差界。参考实现同时给出验证行上的接受概率估计：`max(1 - 残差之和, 0)`；新采样行的估计是 1。这份估计与置信度做绝对差，再在批内求和，作为统计量。第 1 节的调度接受率仍是置信度的前缀累乘（[fused_sample.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/fused_sample.py)、[common.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/common.py)）。
3. **不依赖 DFlash 权重。** 同一请求里，第一次拒绝挡住它后面的草稿行。生效的接受只保留「验证行、比较通过、且前面没有拒绝」。每个请求的最终行是第一行被挡住的位置；草稿全部通过时，最终行就是那行新采样。可选的 golden token 会改写该行的 token；若提供了 golden，接受与否改成「golden 是否等于草稿 token」（[tilelang_ops/sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/sampler.py) `get_accept_result_ref`、`finalize_verification_ref`）。
4. **依赖 DFlash 权重。** 最终行上的 token 作为下一轮草稿的条件，再跑一遍 DFlash，得到下一轮的草稿 token、对数概率和置信度。本步发出的 token 里若含有该请求的停止符，下一轮置信度整行写成 0。这一步发生在回读之前（[sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler.py)、[dflash.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py) `decode_forward`）。

留在设备上的有两处。

- MTP 信息池，按调度槽位存放：每个请求 `nextn` 行的草稿 token 编号、对数概率、置信度；开启 force top-k 时还有候选的词表下标。0 号槽是填充，写入时跳过。对数概率只进这个池，不放进回给调度器的预测对象。下一步的拒绝采样从池里读上一轮的草稿分布。池的写和后续的读在同一条流上，靠流的顺序保证先写后读（[mtp_info_pool.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/mtp_info_pool.py)、[sequence.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/sequence.py) `MTPPredictInfo`、[sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler.py)）。
- 给下一步长度修正用的设备副本：本步采样出的目标 token，形状是每个请求一个；本步预测的草稿 token；每个请求已接受的草稿个数，范围是 0 到 `nextn`，实现上取最终行在该请求内的偏移；本步的置信度。这些副本不随回读被搬走（[sequence.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/sequence.py) `SamplerOutput`、[sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler.py)）。

读回主机的，都是非阻塞拷贝，要等采样完成事件同步之后才能读。重叠打开时，这次同步在下一步的后处理。本步返回时，这些拷贝还停在未同步的状态（[sequence.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/sequence.py) `sampler_epilogue`、[worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py)）。

- 每个请求最终行的 token 编号和对数概率；若请求了 top logprobs，还有对应的编号和对数概率；若打开了较小概率之和，还有这个和。
- 打包验证行上的结果：接受的草稿编号，未接受的位置写成 -1；对应的对数概率，未接受的位置写成 0。top logprobs 和较小概率之和同样只在打开时回读。
- 草稿 token 编号和置信度，供 CPU 做第 1 节的累乘和下一轮调度。对数概率不在这份预测结果里。
- 每个请求在打包缓冲里的起始偏移。
- 按投机步统计的生成个数、送去验证的个数，以及 `|接受概率估计 - 置信度|` 的批内和。这三个数在后处理函数里才变成主机上的列表和标量。
- 预填充且打开了 prompt logprobs 时，另有一份 prompt 对数概率。
- 变长打开时，修正后的各请求查询长度单独回读。它在本步采样启动前同步，不和上面的采样结果绑在同一次等待里。

## 哪些步骤要 V4 的 DFlash 权重

| 步骤 | 是否依赖 DFlash 权重 |
|---|---|
| 草稿层前向、可选的 Markov 修正、置信度头 | 依赖。置信度头和 Markov 表在 DFlash 检查点里；草稿 logits 的底子是主模型 lm head |
| 置信度的前缀累乘、空列表时填 1 或 0 | 不依赖 |
| 按吞吐曲线选择总预算，再把名额收成每个请求的草稿前缀长度 | 不依赖。曲线和耗时估计来自运行时测量 |
| 异步时在设备上用上一轮置信度重算各请求长度 | 不依赖。读的是已经留在设备上的数 |
| CPU 与 GPU 的 8 步交错、在途计数、键长度修正 | 不依赖 |
| 主模型 logits 上的温度、top-p、top-k 和抽样 | 不依赖 DFlash。用的是主模型这一步的 logits |
| 用池里的草稿分布做拒绝采样，并在第一次拒绝处截断 | 判定本身不读权重。池中的草稿分布来自上一轮 DFlash |
| 把草稿对数概率和置信度留在设备池与设备副本里，只回读 token、置信度、对数概率和计数 | 不依赖。写入池的那一次草稿采样依赖 DFlash |
