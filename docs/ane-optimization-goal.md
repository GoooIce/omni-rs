# Burn ANE 实时优化验收

## 2026-10-11 收尾选择

按用户“采用之前的最佳结果收尾”的要求，停止继续优化，工作树恢复到
`infinite-mask-source`：保留完整中文集已有收益的稳定卷积块复用，以及后续通过数值验证的
IOSurface 有界资源恢复、负无穷注意力遮罩修复。图内投影行切片实验已撤回，源码、发布二进制和
测试日志归档到 `internal-row-slices-*`；该实验未测模型性能，不作为选用依据。

恢复后的 43 份源码与冻结指纹一致，重新运行的 **80 项后端测试全部通过**。
发布构建通过，`target/release/r2t2` 及三个评测程序的 SHA-256 全部与此前冻结版本一致；
CLI 模型配置读取和麦克风参数检查通过。冻结的可复现程序位于
`artifacts/burn-ane-goal/infinite-mask-bin/`，日常 `target/release` 入口也已恢复到该版本。
此前该版本通过两组 Clippy、真实模型及跨窗口数值检查，47 份输出数组与各自控制逐字节一致。
原生界面仍默认使用 Metal，Web 使用 WebGPU。

| 证据所属版本与范围 | RTF | 单步 p95 | 结论 |
| --- | --- | --- | --- |
| 卷积块复用版，96 条中文集一轮，prefer-small | 0.7708 | 173.38 ms | CER 3.1052%，相对 Metal +0.0706 个百分点；28/96 条同时通过两项性能门槛 |
| 资源恢复版，30 分钟连续 PCM，LRU | 1.0174 | 293.70 ms | 76 次资源恢复成功、无矩阵回退；SIGINT 停止响应 63.97 ms，结束积压 33.07 秒 |
| 本次选用的负无穷遮罩版，6.74 秒短测一轮，prefer-small | 0.8200 | 173.75 ms | 数值修复通过；完整集 CER、三轮性能与本构建的 30 分钟长测未做 |

前两行分别是历史完整集最佳性能基线和资源恢复长测，**不能当作收尾构建的完整验收结果**。
原实时目标未达成；本轮按用户要求收尾，不再追加性能实验。以下未通过的标准保留为验收缺口。
连续 PCM / SIGINT 记录也不代表物理麦克风或界面停止按钮的验证。

证据：[收尾状态](../artifacts/burn-ane-goal/closeout-status.json)、
[源码核验](../artifacts/burn-ane-goal/current-source-audit.json)、
[构建及指纹复核](../artifacts/burn-ane-goal/closeout-verification.json)、
[收尾回归](../artifacts/burn-ane-goal/closeout-tests.log)、
[发布构建](../artifacts/burn-ane-goal/closeout-build.log)、
[完整集基线](../artifacts/burn-ane-goal/convolution-prefix-row64-corpus-v1-r1/summary.json)、
[资源恢复长测](../artifacts/burn-ane-goal/surface-recovery-lru-soak30-summary.json)、
[收尾版本短测](../artifacts/burn-ane-goal/infinite-mask-short-comparison.json)。

从仓库根目录使用冻结版本接入麦克风，手动 Ctrl-C 停止；分块与前瞻均为 160 ms：

```sh
artifacts/burn-ane-goal/infinite-mask-bin/r2t2 \
  --backend ane --precision f16 --ane-runtime direct \
  --ane-graph-cache-mib 16384 --ane-weight-cache-mib 4096 \
  --ane-max-cached-graphs 1024 --ane-loaded-graphs 112 \
  --ane-row-bucket 64 --ane-kv-bucket 256 --ane-fuse-single-token \
  --ane-program-cache-policy prefer-small \
  transcribe --model models/Confucius4-R2T2 --microphone \
  --chunk-ms 160 --lookahead-ms 160
```

该命令显式使用保留的默认程序槽策略；LRU 的长测结果单独记录，不混用性能结论。

## 原优化目标

目标设置于 2026-10-10。以下项目全部通过才算达成；短音频探针不替代语料和长流测试。

| 项目 | 固定标准 |
| --- | --- |
| 设备、模型 | 本机 M5 Max，完整 Confucius4-R2T2，FP16 |
| 流式参数 | 160 ms chunk，160 ms lookahead |
| 热态性能 | RTF ≤ 0.8，单步 p95 ≤ 160 ms |
| 执行路径 | 主要矩阵计算直接 ANE；单独记录布局、索引、argmax 等 Burn 回退 |
| 缓存预算 | 序列化图 16 GiB，打包权重 4 GiB，已加载程序 112；不是 RSS 上限 |
| 准确率 | 固定中文集 CER 相对同模型 Metal FP16 劣化 ≤ 0.2 个百分点 |
| 连续流式 | 30 分钟连续输入，记录积压、内存、程序重载、错误与停止响应 |
| 数值问题 | 保留原 5% 相对 RMSE 门槛，定位第 3 个缓存 token 的失败 |

RTF = 计算总秒数 / 音频秒数；p95 使用 nearest-rank。计入最后一次 `finish`，不删离群步骤。
模型加载、首次编译与热态分开，热态中实际发生的程序重载计入耗时。
CER = 总编辑距离 / 总参考字符数，不能平均各句 CER；0.2 个百分点是绝对 CER 差 ≤ 0.002。
单步 p95 只描述计算耗时，不等于从说话到显示文字的端到端延迟；后者还包含采集、前瞻、排队和文字提交等待。
RTF 0.8 留出约 20% 的处理时间余量。优先达到上述稳定实时标准，再考虑更低延迟；Web 当前以 WebGPU 兼容可用为主。

## 实验记录

以下按实验阶段保留原始结果；收尾采用版本及验收范围以上方“收尾选择”为准。

### 图内投影行切片（已撤回归档，未测性能）

使用已冻结的负无穷遮罩版，从同一固定中文集依次连接前 40 秒 PCM，采用 LRU / row64 / KV256 /
单 token 融合、160 / 160 ms，并先完整预热一次。开启逐图追踪的诊断共记录 52,882 次图执行，
热态为 26,441 次、196 类输入形状/算子数/常量数分组，无热态编译或淘汰，1,085 次程序重载。
热态原生 evaluate 为 30.58 秒，输入复制 2.94 秒，加载路径 4.34 秒；这些计时存在包含关系。
追踪会增加开销，本轮不能作为 RTF / p95 性能验收。

其中 87 个算子、7 个常量、64 / 128 / 192 行的三组共占原生 evaluate 的 **36.34%**，
累计加载 1.62 秒；每次 evaluate 均值分别为 1.22 / 1.46 / 3.28 ms。
按输入维度与模型结构推断，主要来自解码器的投影、FFN 和归一化；这种签名分组会合并不同层的权重，
不能将分组计时冒充单个算子的耗时。

候选将静态权重线性投影在 MIL 内切成最多 64 行的片段，复用同一个权重值，输出也在 MIL 内拼接。
偏置在拼接后应用，跨批次输入先按既有规范化布局展平。没有新增主机拼接、逐片图调度或缓存预算。
此前归档的图外行块实验会多次执行图并在主机拼接，本轮单独验证图内表达。
新增 7 组实际硬件用例覆盖 1、64、65、128、129、192 行及跨批次尾块，包含偏置与连续两次投影；
均对照 FP32 累加参考，并检查仍只有一次图编译/执行。
控制版和候选均通过新用例；候选通过 **81 项后端测试**和两组 Clippy。
发布构建通过；用户要求收尾后，该实验撤回归档，未继续真实模型、跨窗口数值、转写和完整集 CER 测量。
源码与二进制冻结为 `internal-row-slices-source/` / `internal-row-slices-bin/`，
不宣称提速。工作树恢复为此前保留的 `infinite-mask-source`。

证据：[输入与来源](../artifacts/burn-ane-goal/infinite-mask-profile40-input.json)、
[完整图分组](../artifacts/burn-ane-goal/infinite-mask-profile40-summary.json)、
[诊断结论](../artifacts/burn-ane-goal/infinite-mask-profile40-findings.json)、
[候选状态](../artifacts/burn-ane-goal/internal-row-slices-status.json)、
[81 项回归](../artifacts/burn-ane-goal/internal-row-slices-tests.log)。
逐图汇总可用 [analyze-graph-costs.py](../artifacts/burn-ane-goal/analyze-graph-costs.py) 从原日志重算，
脚本检查分组累计与正式阶段统计一致、单个末尾 final 及全部文字增量，拒绝覆盖现有报告：

```sh
python3 artifacts/burn-ane-goal/analyze-graph-costs.py \
  artifacts/burn-ane-goal/infinite-mask-profile40.log \
  --events artifacts/burn-ane-goal/infinite-mask-profile40.jsonl \
  --output artifacts/infinite-mask-profile40-repeat-summary.json
```

### 长流性能差距

以最近资源恢复版的 30 分钟长测为基准，在相同工作量下达到 RTF 0.8 仍需将总计算耗时降低约
21.4%，p95 从 293.70 ms 降至 160 ms 则需降低约 45.5%。原生 ANE evaluate 占本轮总计算时间
71.67%，图编译/加载路径占 9.98%。若以后重新启动优化，可从这两部分继续分析。
这些是单轮耗时分解及距目标的差距，不代表预测收益；evaluate 属于 prediction，重载属于加载路径，
嵌套计时不能重复相加。[原始统计换算](../artifacts/burn-ane-goal/surface-recovery-long-stage-costs.json)。

### 负无穷遮罩（已保留数值修复，性能未达标）

两个动态行最小值候选均未在短测显示提速，继续用 `-inf` 填充被遮罩项，去掉额外的行最小值归约，
并恢复所有路径的 max/sub/exp/sum/div 表达。取指数后显式清零被遮罩项，覆盖全遮罩行；
非空有效行的分母至少为 1，全遮罩行则以零除以 1。原来的有限值填充会在极值概率用例中失败，
不能只为恢复速度而撤销回归检查。

本机 Direct ANE 编译执行通过：10 组概率配置、80 项后端回归和两组 Clippy 通过，未放宽 0.003
概率误差门槛。源码和二进制冻结为 `infinite-mask-source/` / `infinite-mask-bin/`。
真实模型 encoder、prefill、8 个缓存 token 的原 5% 相对 RMSE 门槛通过，最大 encoder / decoder
误差为 1.2389% / 0.9468%；20 份 ANE / Metal 数组全部与资源恢复控制逐字节一致，推理无新增回退。
跨窗口的 9 阶段编码器探针（32 至 1001 帧，含重复 320 帧及 8 秒边界）也通过；缓存与完整重算
完全一致，相对 Metal 最大 RMSE 1.7919%，27 份数组全部与冻结的卷积块控制逐字节一致。
探针累计实际复用 8 个卷积块，原始数组的有限性、误差及指纹均独立复核。

6.74 秒短音频候选为 **RTF 0.8200 / p95 173.75 ms**，随后同机资源恢复控制为
**0.8180 / 175.32 ms**。本轮候选总耗时增加 0.24%、p95 降低 0.89%；各一轮不足以证明稳定
性能变化，均未达到门槛。两次正文及 73 个 token 一致，各有 4209 次直接 ANE 调用、121 次重载，
无热态新图、淘汰或失败。保留的是遮罩边界数值修复，不能将其称为已验证的性能提升。
测量结束后复核了 43 份源码和 4 个二进制的指纹。

完整集 CER、三轮性能对照及本构建的 30 分钟长测尚未完成；有限探针的逐字节一致不能替代这些验收。
跨窗口探针在 6.74 秒后循环录制 PCM，不能作为物理麦克风或 UI 停止验证。
资源恢复实现保留，当前已验证的长流对照仍为 `surface-recovery-bin`。
证据：[候选状态](../artifacts/burn-ane-goal/infinite-mask-status.json)、
[概率边界](../artifacts/burn-ane-goal/infinite-mask-probability-test.log)、
[完整后端回归](../artifacts/burn-ane-goal/infinite-mask-tests.log)、
[真实模型与数组复核](../artifacts/burn-ane-goal/infinite-mask-parity-summary.json)、
[跨窗口数值复核](../artifacts/burn-ane-goal/infinite-mask-cache-parity-summary.json)、
[短音频同期对照](../artifacts/burn-ane-goal/infinite-mask-short-comparison.json)。

复现数值探针与短音频（输出目录使用新路径，保留既有测量）：

```sh
artifacts/burn-ane-goal/infinite-mask-bin/ane_blocks --model models/Confucius4-R2T2 \
  --ane-runtime direct --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --model-steps 8 --collect-failures --dump-dir artifacts/infinite-mask-model-repeat
artifacts/burn-ane-goal/infinite-mask-bin/ane_blocks --model models/Confucius4-R2T2 \
  --ane-runtime direct --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --encoder-cache-probe --dump-dir artifacts/infinite-mask-cache-repeat
python3 scripts/evaluate_streaming.py --backend ane \
  --binary artifacts/burn-ane-goal/infinite-mask-bin/r2t2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav --repeats 1 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token --output artifacts/infinite-mask-short-repeat
```

### 遮罩 Softmax 单次归一化（已归档，短测未提速）

原生 Softmax 候选在短音频中没有提速，继续去掉遮罩路径的重复归一化。
有遮罩时仍用当前行最小分数填充，减去最大值后取指数，清除被遮罩的项，再只归一化一次；
非空有效行包含一个 `exp(0) = 1`，分母下限设为 1，全遮罩行输出精确为零。
无任何遮罩时保留 MIL 原生 `softmax`。极值概率回归及其 0.003 门槛、资源恢复实现均保留。

80 项后端测试及两组 Clippy 通过，源码与二进制冻结为 `masked-softmax-source/` / `masked-softmax-bin/`。
真实模型 10 个阶段通过原 5% 数值门槛，从导出数组独立重算一致；encoder 最大相对 RMSE
为 1.2508%，decoder 最大为 0.9468%，decoder top1 均与 Metal 相同，推理无新增回退。
10 份 Metal 数组与控制完全一致；ANE 只有 encoder 数组变化，9 份 decoder 数组逐字节相同。
6.74 秒短音频候选为 **RTF 0.8383 / p95 195.57 ms**，随后同机资源恢复控制为
**0.8158 / 174.41 ms**，候选本轮耗时增加 2.75%、p95 增加 12.13%，未显示提速。
两次正文相同，均生成 73 个 token；各有 4209 次直接 ANE 调用、121 次重载，无热态新图、
淘汰或失败。这是各一次筛选，不构成稳定效果估计，两版都未通过性能门槛。
候选归档，未运行完整集 CER 或长流，不能继承资源恢复版的完整结论。
证据：[候选状态](../artifacts/burn-ane-goal/masked-softmax-status.json)、
[80 项回归](../artifacts/burn-ane-goal/masked-softmax-tests.log)、
[真实模型与数组复核](../artifacts/burn-ane-goal/masked-softmax-parity-summary.json)、
[短音频同期对照](../artifacts/burn-ane-goal/masked-softmax-short-comparison.json)。

### 原生 Softmax 与遮罩归一化（已归档，短测未提速）

资源恢复通过长测后，继续试验 MIL 原生 `softmax`，替代注意力中的 max/sub/exp/sum/div 展开。
首先加入独立 f64 概率参考：5 组分数分布、3 种 query 遮罩、KV 容量 0 / 128，共 10 个配置；
V 使用基向量，使输出直接对应注意力概率。绝对误差门槛为 0.003，全遮罩行要求精确为零。

原实现及最初仅替换 Softmax 的候选都在全部有效分数为 -65504、部分键被遮罩时失败：
三个有效键输出零，参考值应各为 1/3。普通分布、饱和分布和零分数的前 6 个配置通过。
固定负值填充改为当前行的最小分数后，再清除被遮罩的概率并重新归一化，10 个配置全部通过；
新增交替 +65504 / -65504 分布，覆盖被遮罩项反而拥有最大分数的情况。
分母只在有效概率总和为零时替换为 1，保留小的正数质量；没有放宽误差门槛。
这些实验定位了失败表达与可通过的替代路径，尚未隔离原生编译器内部的具体原因。

候选通过 80 项后端测试（17 单元、27 通用/Core ML、36 Direct）及两组 Clippy，
保留之前的全部资源恢复测试与实现。发布构建及真实模型 encoder、prefill、8 个缓存 token 的
原 5% 相对 RMSE 门槛通过：encoder 最大 1.2366%，decoder 最大 0.5463%；推理无回退，
decoder top1 均与 Metal 相同。10 份 Metal 数组与控制完全一致，10 份 ANE 数组均有变化。

6.74 秒短音频候选为 **RTF 0.8336 / p95 179.05 ms**，随后同机资源恢复控制为
**0.8142 / 175.71 ms**；候选本轮耗时增加 2.38%、p95 增加 1.90%，未显示提速。
两次正文相同，均生成 73 个 token；各有 4209 次直接 ANE 调用、121 次重载，无热态新图、
淘汰或失败。这是各一次筛选，不构成稳定效果估计；两版都未通过性能门槛。
未运行该算术候选的完整集 CER、三轮对照或 30 分钟长测。
源码及二进制冻结为 `native-softmax-source/` / `native-softmax-bin/`，归档后继续上述单次归一化候选。
当前已验证的长流对照仍为 `surface-recovery-bin`。
证据：[候选状态](../artifacts/burn-ane-goal/native-softmax-status.json)、
[原实现边界失败](../artifacts/burn-ane-goal/native-softmax-control-probability-diagnostic.log)、
[最初候选失败](../artifacts/burn-ane-goal/native-softmax-probability-test.log)、
[10 个配置通过](../artifacts/burn-ane-goal/native-softmax-row-min-probability-test.log)、
[80 项回归](../artifacts/burn-ane-goal/native-softmax-tests.log)、
[真实模型数值](../artifacts/burn-ane-goal/native-softmax-parity-summary.json)、
[短音频同期对照](../artifacts/burn-ane-goal/native-softmax-short-comparison.json)。

### IOSurface 临时资源恢复（已保留，性能仍未达标）

上一轮 30 分钟长测发现临时分配失败被写入“不支持图”的缓存。本次在通用 Burn ANE 后端中
将 `IOSurface allocation failed` 与布局错误、超出 4 GiB 的非法请求区分；临时分配失败不再缓存。
程序槽不足继续卸载实例；IOSurface 压力则优先淘汰最久未用的已卸载 Direct 图，释放其 I/O 和 request，
没有已卸载图时才选其他 Direct 图。编译、重载及原生输出分配均有界恢复，单次最多重试 8 次；
没有可释放的图时立即按调用者的回退策略结束。实际张量独立持有输出引用，图淘汰不会使旧值失效。
恢复失败仍明确计入回退或错误，不能算作直接 ANE 成功。

新增 `surface_resource_retries`（总资源重试的子集）与 `resource_recovery_ms`。
后者在编译/重载时与 `compile_load_ms` 重叠，不能重复相加；失败预测尝试也保留计时和分配统计。
算子数学表达、原生桥接及缓存预算均未改动。

79 项后端测试（17 单元、27 通用/Core ML、35 Direct）和后端、根项目 ANE/Core ML 组合 Clippy 通过。
测试注入只存在于测试构建，覆盖有/无已卸载图的成功恢复、淘汰后旧输出读取、无可释放图、持续失败
最多 8 次重试，以及资源恢复后相同图重新执行 ANE。严格 FP16 测试验证输出租用；允许回退的用例使用
NdArray 支持的 F32。首轮误用 NdArray FP16 回退的测试失败日志保留，未改变生产精度或错误策略来迁就测试。
真实权重 encoder、prefill 和后续 8 token 均通过原 5% 相对 RMSE 门槛，最大为 **1.2389%**；
20 份 ANE / Metal 输出数组与冻结的程序槽版数值对照逐字节一致，推理无新增回退。
6.74 秒短音频单轮为 **RTF 0.8067 / p95 173.36 ms**，正文与卷积块版相同，均生成 73 个 token；
4209 次直接 ANE 调用、121 次重载，无热态新图、淘汰、失败或资源重试。短测性能门槛未通过，
只有一轮且控制为归档数据，不将微小差异解释为确定的性能变化。
相同 LRU / row64 / KV256 / 单 token 融合、32 秒预热后的 **30 分钟复测已完成**。
墙钟 **1800.25 秒**，报告音频 **1767.04 秒**；驱动在测量满 1800 秒后发送 SIGINT，
停止响应 **63.97 ms**。停止前已执行但未提交的 143.78 ms 计算仍计入 RTF。
**RTF 1.0174 / p95 293.70 ms**，结束积压 **33.07 秒**、峰值积压 33.65 秒，性能门槛仍未通过。
RSS 采样峰值 **25.68 GB**（约第 30 秒采样），结束为 23.07 GB；不能把后期较低采样当作全程峰值。

首个真实恢复采样在 692.71 秒，前一个 662.64 秒采样尚无资源重试。热态累计 **76 次 IOSurface
资源恢复、76 次缓存淘汰，全部成功**；无矩阵回退、编译失败、失败缓存项或命中、程序重载失败。
热态新建图 192 张、重载 48,925 次；释放资源计时 53.73 ms，相关重编译和重载开销也保留在测量中。
直接 ANE、缓存预算、时长、主动停止、单一末尾 final 和全部文字增量检查通过。
与旧长测共同的 **10,995 个非 final 步骤正文全部一致**。两轮报告音频跨度不同，RTF 的 0.52% 降幅和
p95 的 1.00% 增幅不能解释为确定的性能变化；这次保留的是已验证的资源恢复修复。
系统分配失败的底层原因尚未证明消除。此测试是录制 PCM 与软件信号，不包含物理麦克风或 UI 点击，
也不能替代最终配置的三轮完整集 CER/性能验收。
证据：[完整复测审计](../artifacts/burn-ane-goal/surface-recovery-lru-soak30-summary.json)、
[两轮对照](../artifacts/burn-ane-goal/surface-recovery-soak30-comparison.json)、
[首次真实恢复前后采样](../artifacts/burn-ane-goal/surface-recovery-first-live-recovery.json)。
资源修复源码 / 二进制冻结为 `surface-recovery-source/` / `surface-recovery-bin/`；
最近一次完整集对照仍为 `convolution-prefix-bin`，其 CER 记录不冒充新构建的完整集验收。
证据：[候选状态](../artifacts/burn-ane-goal/surface-recovery-status.json)、
[修复前失败](../artifacts/burn-ane-goal/surface-recovery-before-native-lease-test.log)、
[完整后端测试](../artifacts/burn-ane-goal/surface-recovery-tests.log)、
[后端 Clippy](../artifacts/burn-ane-goal/surface-recovery-clippy.log)、
[根项目 Clippy](../artifacts/burn-ane-goal/surface-recovery-root-clippy.log)、
[模型数值对照](../artifacts/burn-ane-goal/surface-recovery-parity-summary.json)、
[短音频筛选](../artifacts/burn-ane-goal/surface-recovery-short-comparison.json)。

复现回归、真实模型数值和长流输入（预热结束后计时，满 30 分钟按 Ctrl-C）：

```sh
cargo test --offline --manifest-path crates/burn-ane/Cargo.toml -- --test-threads=1
artifacts/burn-ane-goal/surface-recovery-bin/ane_blocks --model models/Confucius4-R2T2 \
  --ane-runtime direct --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --model-steps 8 --collect-failures --dump-dir artifacts/surface-recovery-parity-repeat
artifacts/burn-ane-goal/surface-recovery-bin/stream_soak --backend ane \
  --manifest docs/eval/mandarin-v1.json --seconds 1860 --warmup-seconds 32 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token --program-cache-policy lru
```

本轮自动信号驱动为 [run-surface-recovery-soak30.py](../artifacts/burn-ane-goal/run-surface-recovery-soak30.py)，
会核验源码和二进制指纹，并拒绝覆盖已有测量。
独立审计脚本 [audit-soak30.py](../artifacts/burn-ane-goal/audit-soak30.py) 从原始事件重算 RTF / p95，
计入停止时取消的计算，逐条检查文字增量，并分别记录时长、SIGINT、缓存预算、直接 ANE 和性能门槛。
已用旧长测验证其结果与原报告一致，保留失败门槛；脚本拒绝把仍在运行的测量当成完整结果。
进程结束后可复核（输出路径必须尚不存在，下面使用另一个文件名保留原审计）：

```sh
python3 artifacts/burn-ane-goal/audit-soak30.py \
  artifacts/burn-ane-goal/surface-recovery-lru-soak30.jsonl \
  --output artifacts/surface-recovery-soak30-audit-repeat.json
```

### 稳定卷积块复用（当前完整集保留版）

将卷积、投影及块内位置编码与双向 Transformer 计算分开。每个 100 帧卷积块只在全部 mel 帧稳定后缓存，
后续调用复用该块；双向注意力仍计算当前整个窗口，decoder 的 `reused_tokens` 仍只包含完整且稳定的注意力窗口。
新增 `reused_convolution_blocks` 单独记录实际复用块数，不计入 decoder KV 的稳定前缀。
全局 mel 裁剪值变化、输入缩短、显式清空或关闭复用会清空缓存；完整窗口完成后释放对应卷积缓存，
只保留当前未完成窗口的数据。此优化适用于 Burn 模型层，后端仍负责原来的矩阵计算。

6 项 CPU 单元与 8 项模型/流式回归通过，包括新增的一秒边界、8 秒窗口切换、变化的临时尾帧、
输入缩短、音量变化、关闭复用和显式清空对照。ANE 与 Core ML 功能组合的 Clippy、Wasm 编译检查通过。
为此新增 `ane_blocks --encoder-cache-probe`，按多个流式长度比较缓存 ANE、完整重算 ANE 和 Metal 输出；
该探针只验证数值，不作为计时结果。9 个阶段（32 至 1001 帧，含 320 帧重复）中，缓存与完整重算的 ANE
输出全部逐字节一致，相对 Metal 最大 RMSE 为 1.7919%，通过原 5% 门槛；直接 ANE 检查通过。
三个阶段累计复用 8 个卷积块，证明实际覆盖缓存路径。32 帧的输出也与冻结的程序槽版逐字节相同。
输入 PCM 在 6.74 秒结束后循环用于长帧探针，不能用它替代 CER 或连续流式验收。

6.74 秒短音频单轮筛选为 **RTF 0.7971 / p95 172.29 ms**，正文和生成 token 数与控制相同。
热态无新图/淘汰、121 次重载、4209 次直接执行；native evaluate 为 4066.83 ms。
三个交替 A/B 复测已完成，采用相同的 `prefer-small`、row64 / KV256、160 / 160 ms 与各自完整音频预热。
控制版合并 **RTF 0.8392 / p95 183.27 ms**，候选 **0.7944 / 171.41 ms**；
总耗时降低 **5.34%**、p95 降低 **6.47%**。候选三轮 RTF 为 0.7958 / 0.7940 / 0.7933，
均通过 0.8 门槛；三轮 p95 为 171.41 / 169.90 / 171.26 ms，均未通过 160 ms 门槛。
六次正文和 73 个生成 token 相同，直接 ANE 检查通过；均无热态新图、淘汰或重载失败，
每轮 121 次重载、4209 次直接执行。native evaluate 平均从 4309.19 降至 4054.51 ms。
这证明了该短音频三轮对照的收益，不能据此认定稳定实时。

相同 LRU / row64 / KV256 / 单 token 融合、32 秒预热后的 40 秒连续输入各一轮：

| 构建 | 报告音频 | RTF | p95 | 结束积压 | 停止响应 | RSS 采样峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 程序槽版控制，本轮复测 | 37.44 s | 1.0373 | 274.59 ms | 2.42 s | 1.35 ms | 25.75 GB |
| 稳定卷积块候选 | 38.40 s | 0.9951 | 268.78 ms | 1.46 s | 99.37 ms | 25.42 GB |

候选 RTF 降低 4.07%、p95 降低 2.12%；共同的 233 个提交步骤正文一致，均生成 482 个 token。
处理跨度不同，不将累计阶段耗时解释为等工作量加速比。候选 / 控制分别热态新图 6 / 5 张、
重载 1044 / 1113 次，均无淘汰、重载失败或推理错误。候选新图的系统编译 117.84 ms 与停止时
已执行但未提交的计算都保留在计时中。两版直接 ANE、缓存预算、单一 final 和 delta 重建检查通过。
这是录制 PCM 按实时节奏输入、定时触发停止句柄的测试，尚未覆盖物理麦克风和 30 分钟验收。
原测量驱动完成候选后因旧指纹文件键名差异退出；仅补跑未启动的控制版，两个模型进程均正常结束。
完整 96 条集一轮已结束，采用与归档控制相同的 `prefer-small`：
**RTF 0.7708 / p95 173.38 ms / CER 3.1052%**（44 / 1417），96 条正文全部与 IR 遍历控制相同。
相对 Metal 的 CER 增加 0.0706 个百分点，精度、直接 ANE 和原缓存预算检查通过。
相对归档控制，总耗时降低 **5.14%**、p95 降低 **7.50%**；两版各一轮，控制非同期复测。
78 / 96 条通过 RTF 门槛、28 / 96 条通过 p95，两项同时通过的也是 28 条；控制分别为 52 / 10 / 10。
因此汇总 RTF 达标不代表性能验收通过。native evaluate 从 297.39 降至 281.92 秒，
prediction 从 329.36 降至 312.79 秒，加载路径从 29.04 降至 26.39 秒。
热态仍无新图、淘汰或重载失败，8874 次重载、304191 次直接 ANE 执行，与控制相同。
保留此次缓存优化，当前完整集构建改为 `convolution-prefix-bin`；先前所有结果继续保存。

同一冻结构建的 LRU 30 分钟连续输入已结束：排除 32 秒预热后，测量墙钟为 **1802.24 秒**，
通过外部控制器发送 SIGINT 主动停止。**RTF 1.0228 / p95 290.78 ms**，报告音频 1759.36 秒，
结束积压 **42.74 秒**、峰值积压 43.13 秒；停止响应 **105.32 ms**。
停止前已执行但未提交的 143.79 ms 计算仍计入 RTF。RSS 采样峰值 **22.82 GB**，结束为 22.57 GB。
只有一个 final 且位于最后，全部文字增量重建正确；时长、主动停止及缓存预算检查通过。
这是录制 PCM 按实时节奏输入与软件信号停止，尚未验证物理麦克风和 UI 按钮。
完整集的 `prefer-small` 结果与 LRU 长测分别记录，尚无 LRU 完整集验收，三轮完整重复仍未完成。

**本轮长测未通过性能和直接 ANE 验收。** 首次含 `IOSurface allocation failed` 的采样在测量
691.95 秒，前一个 661.78 秒采样正常；热态累计 **1045 次相关回退，其中 1015 次命中失败缓存**，
留下 30 个失败缓存项、1 次编译失败，资源重试为 0。源码确认目前只把 ANE 程序槽不足视为
临时资源错误，IOSurface 分配失败会被写入失败缓存，导致相同图持续回退。
卸载程序本身仍保留 I/O 缓冲和 request，不能假定卸载就会释放这种资源。
源图缓存约 8.26 GB、打包权重约 4.08 GB，仍在既有预算内；单次系统内存压力查询报告
free percentage 84%，没有换入/换出。这些计数尚不能证明分配失败的根因。
后续资源分类与有界恢复已单独完成长流复测，结果见上节；Softmax 算子实验另行记录。
保留真实原生输出租用的回归在修复前因 `IOSurface allocation failed` 失败。
证据：[完整长测审计](../artifacts/burn-ane-goal/convolution-prefix-lru-soak30-summary.json)、
[失败时间范围与源码定位](../artifacts/burn-ane-goal/convolution-prefix-soak30-surface-failure.json)、
[修复前回归](../artifacts/burn-ane-goal/surface-recovery-before-native-lease-test.log)。
完整集保留源码 / 二进制为 `convolution-prefix-source/` / `convolution-prefix-bin/`；工作树另有上述资源恢复候选，
其模型与长测证据单独记录。RMSNorm 实验已撤回。
证据：[CPU 回归](../artifacts/burn-ane-goal/convolution-prefix-cpu-tests.log)、
[真实权重探针](../artifacts/burn-ane-goal/convolution-prefix-cache-parity-summary.json)、
[短音频筛选](../artifacts/burn-ane-goal/convolution-prefix-short-screen/summary.json)、
[三组交替对照](../artifacts/burn-ane-goal/convolution-prefix-short-comparison.json)、
[连续输入对照](../artifacts/burn-ane-goal/convolution-prefix-rolling-comparison.json)、
[完整集对照](../artifacts/burn-ane-goal/convolution-prefix-row64-corpus-comparison.json)。

复现完整集及连续输入（长测在预热结束后计时，满 30 分钟按 Ctrl-C）：

```sh
python3 scripts/evaluate_streaming.py --backend ane \
  --batch-binary artifacts/burn-ane-goal/convolution-prefix-bin/stream_eval \
  --manifest docs/eval/mandarin-v1.json --repeats 1 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --compare artifacts/burn-ane-goal/metal-corpus-v1-r1/summary.json \
  --output artifacts/convolution-prefix-corpus-repeat
artifacts/burn-ane-goal/convolution-prefix-bin/stream_soak --backend ane \
  --manifest docs/eval/mandarin-v1.json --seconds 1860 --warmup-seconds 32 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token --program-cache-policy lru
```

### RMSNorm 附加能量通道

全批次 L2 版本被实际转写筛选否决；小批次版本通过完整集精度检查，但没有稳定性能收益。
两版均归档，工作树已恢复原 LayerNorm 实现。

原实现计算 `LayerNorm(concat(x,-x))`，中间通道数为 `2C`。新候选将每个向量置于独立批次，
附加一个 `sqrt(C * epsilon)` 常量通道及对齐用零通道，执行 epsilon 为零的 L2 normalization，
截取原通道并乘 `sqrt(C)`。数学上分母为 `sqrt(sum(x²) + C * epsilon)`，不依赖原生 epsilon 属性。
例如 C=2048 时，归一化输入从 4096 降为 2080 个通道；额外的布局转换与整体速度仍需实测。
L2 跨最后三个维度的语义来自 [Apple MIL 定义](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.normalization.l2_norm)，
附加能量的表达是本项目推导，不能据此认定硬件一定更快。

48 组回归覆盖宽度 1、2、7、31、32、33、64、128、512、1536、2048，以及零、微小、大幅值输入和批次布局。
最大归一化绝对误差 **0.0041031**，最大后续投影误差 **0.0007498**，分别低于 0.006 / 0.01 门槛。
原生单通道 L2 批次出现部分元素未经归一化，候选先复制成两通道再归一化，保持 RMS 数学定义；
早期分散常量能量和单通道表达的失败日志全部保留。
75 项后端测试和两组 Clippy 通过，原生桥接未改。完整权重 row64 / KV256 / 单 token 融合下，
encoder、prefill 和后续 8 token 通过原 5% 相对 RMSE 门槛，decoder 最高分 token 全部与 Metal 相同，推理无回退。
Encoder 相对 RMSE 为 1.2389%，prefill 为 0.2335%，后续 8 token 最大为 0.4734%；
Metal 对照的 10 份数组逐字节相同，ANE decoder 因算术表达改变而变化，不能沿用旧构建的 CER。
这不足以证明真实 ASR 正确：6.74 秒完整转写单轮为 **RTF 2.1043 / p95 422.00 ms**，
正文变成“之前有顾客自己带酒水解”，而对照为“之前有顾客自己带酒水也没加收钱或者不让喝。”，不采用此版本。
热态 0 次新图/淘汰、121 次重载，直接 ANE 检查通过；`direct_evaluate` 为 12.81 秒，旧探针为 4.30 秒。
生成 token 数分别为 55 / 73，不能将累计时间作为等工作量比值。prefill 数值探针的 prediction 也从约 63 增至 216 ms，
而后续单 token 约 32 ms，给出了进一步限定小批次路径的线索，并非热态速度结论。
失败源码与二进制冻结为 `rms-padding-source/` / `rms-padding-bin/`，不修改完整集保留构建。

后续小批次候选只在归一化向量数 ≤ 32 时使用 L2，大批次保留原 LayerNorm；
常量通过 MIL fill 构造，避免改变打包权重 blob，使不同归一化路径仍可共享同一组权重。
新增大批次布局案例后，56 组数值案例、75 项后端测试与两组 Clippy 通过；
完整权重 encoder / prefill / 后续 8 token 通过原 5% 门槛且最高分 token 与 Metal 相同，推理无回退。
该版 prefill 相对 RMSE 为 0.4620%，后续 8 token 最大为 0.6215%，Metal 的 10 份参考数组保持相同。
短音频转写恢复完整。三个交替 A/B 对照，控制 **RTF 0.8502 / p95 185.24 ms**，候选 **0.8410 / 183.75 ms**，
总耗时降低 1.07%、p95 降低 0.80%。原生 evaluate 平均 4298.20 / 4266.08 ms，候选主要收益集中在第二轮，
第一和第三轮接近控制，尚不能认定稳定收益。六次转写相同，均 0 次新图/淘汰、121 次重载、4209 次直接执行。
同一 LRU / row64 / KV256 / 单 token 融合配置、32 秒预热后的 40 秒连续输入各一轮：

| 构建 | 报告音频 | RTF | p95 | 积压 | 停止响应 | RSS 采样峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 程序槽版控制，本轮复测 | 36.96 s | 1.0582 | 277.49 ms | 2.90 s | 21.78 ms | 25.82 GB |
| 小批次 RMS 候选 | 37.76 s | 1.0292 | 256.05 ms | 2.10 s | 3.01 ms | 24.26 GB |

本轮候选 RTF 降低 2.74%、p95 降低 7.73%；但更早的 LRU 控制曾测得 1.0112 / 251.05 ms，不能认定稳定提速。
共同的 230 个提交步骤正文相同、均生成 474 个 token；总处理跨度不同，不直接比较累计阶段耗时。
控制 / 候选热态新图 4 / 5 张、重载 1108 / 1116 次，均 0 次系统编译、淘汰、重载失败及推理错误。
只有一个 final、delta 正确，直接 ANE 与原缓存预算检查通过；RSS 仅为单轮采样。
候选完整 96 条集一轮已结束，采用与完整集保留对照相同的 `prefer-small`。
结果为 **RTF 0.8139 / p95 186.85 ms / CER 3.0346%**（43 / 1417），CER 与 Metal 相同。
相对 IR 遍历保留版，95 条正文相同；`aishell-test-2-1129` 去掉了“中新闻网”中多出的“新”，其余正文不变。
RTF 增加 0.16%、p95 降低 0.32%，未构成实际提速；47 / 96 条通过 RTF 门槛、4 / 96 条通过 p95，
保留版为 52 / 96 和 10 / 96。各一轮且控制为归档结果，不能据此判断细小变化的因果。
候选 native evaluate 为 296.89 秒，控制 297.39 秒；加载路径为 29.44 / 29.04 秒。
候选热态无新图、淘汰或重载失败，8874 次重载、304135 次直接 ANE 执行，直接 ANE 检查通过。
精度改善没有带来稳定性能收益，因此不保留这次算术改动；该阶段的完整集基线仍是 `inline-ir-bin`。
仍缺性能达标、三轮完整重复及 30 分钟验收，默认槽策略不变。
证据：[数值案例](../artifacts/burn-ane-goal/rms-padding-widths-summary.json)、
[模型对照](../artifacts/burn-ane-goal/rms-padding-parity-summary.json)、
[失败转写筛选](../artifacts/burn-ane-goal/rms-padding-short-screen/summary.json)、
[小批次数值记录](../artifacts/burn-ane-goal/rms-small-batch-widths-summary.json)、
[小批次模型对照](../artifacts/burn-ane-goal/rms-small-batch-parity-summary.json)、
[三个交替 A/B 短测](../artifacts/burn-ane-goal/rms-small-batch-short-comparison.json)、
[LRU 连续对照](../artifacts/burn-ane-goal/rms-small-batch-rolling-comparison.json)、
[完整 96 条对照](../artifacts/burn-ane-goal/rms-small-batch-row64-corpus-comparison.json)。

复现小批次候选的原始模型对照和完整语料检查：

```sh
artifacts/burn-ane-goal/rms-small-batch-bin/ane_blocks \
  --model models/Confucius4-R2T2 --ane-runtime direct \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token --model-steps 8 \
  --collect-failures --dump-dir artifacts/rms-small-batch-parity-repeat
python3 scripts/evaluate_streaming.py --backend ane \
  --batch-binary artifacts/burn-ane-goal/rms-small-batch-bin/stream_eval \
  --manifest docs/eval/mandarin-v1.json --repeats 1 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --compare artifacts/burn-ane-goal/metal-corpus-v1-r1/summary.json \
  --output artifacts/rms-small-batch-corpus-repeat
```

### 程序槽替换策略（可选候选，默认未改）

新增纯 LRU 和优先卸载相同共享权重 blob 程序两种策略，描述、I/O 缓冲及预算不变。
相同权重仅用于选择卸载对象，不表示两张计算图等价。74 项后端测试、两组 Clippy、3 项评测脚本测试通过；
两槽压力回归检查重载、数值和资源释放，模型 20 份数组与 IR 遍历版逐字节相同。
112 槽的小模型对照没有触发替换；替换路径由压力回归和真实 ASR 测量覆盖。

同一冻结二进制、row64 / KV256 / 单 token 融合，每项各一轮：

| 策略 | 6.74 秒短测 RTF / p95 | 40 秒连续 RTF / p95 | 连续积压 | 热态重载 |
| --- | ---: | ---: | ---: | ---: |
| 原小图优先 | 0.8456 / 184.41 ms | 1.0509 / 303.63 ms | 2.90 s | 1522 |
| 同权重优先 | 0.8594 / 183.75 ms | 1.0333 / 306.83 ms | 2.42 s | 1228 |
| LRU | 0.8779 / 188.74 ms | 1.0112 / 251.05 ms | 1.62 s | 1124 |

LRU 连续 p95 降低 17.32%，但短测更慢，尚不修改默认配置。三轮连续探针共同的 230 个提交步骤正文相同。
继续用 LRU 筛选 KV128：RTF **1.0236**、p95 **274.90 ms**、积压 **2.10 s**、停止响应 **4.76 ms**，
分别比 KV256 慢 1.23% 和 9.50%，未采用；两者共同的 235 个提交步骤正文相同。
KV128 热态 8 张新图、1135 次重载；所有连续探针均无淘汰、重载失败或推理错误，直接 ANE 与缓存预算检查通过。
计入实际编译、重载和取消计算；各项处理的音频跨度不同，不用累计阶段时间直接推导等工作量收益。
这些是单轮筛选，尚无新策略完整集 CER、三轮复测或 30 分钟验收。

冻结构建为 `program-policy-bin/`，此策略筛选的完整集对照为 `inline-ir-bin/`。
证据：[短测对照](../artifacts/burn-ane-goal/program-policy-short-comparison.json)、
[连续对照](../artifacts/burn-ane-goal/program-policy-rolling-comparison.json)、
[LRU / KV128 对照](../artifacts/burn-ane-goal/program-policy-lru-kv128-comparison.json)。

### IR 输入/输出遍历（归档控制）

常用算子改为借用切片或栈上的最多 5 个张量引用，减少 Burn IR 迭代器的堆分配。
未覆盖算子直接委托上游迭代器，不额外装箱，也不假定只有一个输出；用于规划、形状分档、降低和回退路径。
新增测试以上游接口为对照，检查可选 bias/mask 的全部组合、重复输入、变长拼接、Drop、自定义及普通多输出算子。
73 项后端测试、后端 Clippy 和使用原四项豁免的根 Clippy 通过。原生桥接未改，沿用原 192 组缓冲验证。
row64 / KV256 / 单 token 融合的 encoder、prefill、后续 8 token 共 20 份数组与二进制键版逐字节相同，原 5% 数值门槛通过。

同一 6.74 秒音频、两版各三轮独立进程热测：

| 构建 | RTF | p95 | 平均图规划 |
| --- | ---: | ---: | ---: |
| 二进制键对照版，本轮复测 | 0.8470 | 185.49 ms | 227.10 ms |
| 栈上遍历候选 | 0.8366 | 183.76 ms | 189.56 ms |

规划时间降低 **16.53%**，总耗时降低 **1.23%**、p95 降低 **0.94%**。所有转写相同，直接 ANE 检查通过；
每轮热态均为 0 次新图/淘汰、121 次重载、4209 次直接执行。短测性能仍未达标。

同一候选另筛选 KV128 与 KV64。KV128 单轮 0.8162 / 178.04 ms，三轮复测 **0.8166 / 179.32 ms**，
重载仍为 121 次，20 份模型数组也与 KV256 逐字节相同。KV64 单轮 **0.8663 / 181.80 ms**，
重载增至 214 次、加载路径耗时 702.50 ms，未采用。

32 秒连续预热、40 秒按实时节奏到达的录制 PCM，各配置一轮：

| KV 分档 | 报告音频 | RTF | p95 | 积压 | 停止响应 | RSS 采样峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 36.64 s | 1.0671 | 308.48 ms | 3.22 s | 9.29 ms | 23.14 GB |
| 256 | 36.96 s | 1.0509 | 309.53 ms | 2.90 s | 33.13 ms | 25.93 GB |

均无推理错误、淘汰或重载失败，只有一个 final、delta 正确，直接 ANE 与原缓存预算检查通过。
KV128 / KV256 的热态新图为 6 / 3 次，程序重载 1592 / 1522 次，加载路径 5723.94 / 4786.54 ms。
KV128 的系统编译耗时 379.09 ms，KV256 为 0，全部计入结果。共同的 228 个提交步骤正文相同。
处理跨度不同，不用累计阶段时间直接推导相同工作量的提速；RSS 也只记录单轮采样值。
本轮未确认 KV128 的连续输入收益，完整集验收继续采用 **row64 / KV256 / 单 token 融合**。

完整 96 条集一轮为 **RTF 0.8126 / p95 187.45 ms / CER 3.1052%**（44 / 1417）。
相对 Metal 的 CER 增加 **0.0706 个百分点**，精度与直接 ANE 检查通过；96 条原始转写与共享权重版相同。
与归档共享权重版相比，总耗时降低 **0.97%**、p95 降低 **1.15%**；两版各一轮，控制并非同期重测。
这包含二进制图键与栈上遍历两项改动。规划从 **19.79 降到 14.21 秒**，direct evaluate 几乎相同（297.36 / 297.39 秒），
prediction 从 327.43 增到 329.36 秒，加载路径从 **25.71 增到 29.04 秒**；不能把规划收益当作各阶段都加速。
两版均热态 0 次新图/淘汰、8874 次重载且无失败、304191 次直接执行。
只有 52 / 96 条满足 RTF 门槛，10 / 96 条满足 p95 门槛；性能仍未通过。

该阶段构建冻结在 `inline-ir-source/` 和 `inline-ir-bin/`。后续可切换的程序槽策略另行验证，
默认策略未改，不能用此完整集结果替代后续策略的验收。此轮连续 p95 仍约 309 ms、RTF > 1；
三轮完整重复与 30 分钟验收尚未完成，不能称为稳定实时。
证据：[后端测试](../artifacts/burn-ane-goal/inline-ir-tests.log)、
[模型位对照](../artifacts/burn-ane-goal/inline-ir-parity-byte-comparison.json)、
[三轮同期对照](../artifacts/burn-ane-goal/inline-ir-short-comparison.json)、
[KV 三轮对照](../artifacts/burn-ane-goal/inline-ir-kv128-short-comparison.json)、
[连续输入对照](../artifacts/burn-ane-goal/inline-ir-kv-rolling-comparison.json)、
[完整集对照](../artifacts/burn-ane-goal/inline-ir-row64-corpus-comparison.json)。

### 图规划的二进制键（后续候选，完整集待验）

内存图键改用 bincode 编码相同的规范化 IR、权重身份、存活输出和辅助掩码形状；同时在创建名称时保存编号，
排序时借用名称，省去解析和字符串复制。有限标量限制、FP16 图、缓存预算和模型磁盘缓存格式保持原有设定。
71 项后端测试、两组 Clippy 检查通过；原生桥接未变，沿用共享权重版的 192 组缓冲验证。
row64 / KV256 / 单 token 融合下，encoder、prefill 与后续 8 token 的 20 份数组与共享权重版逐字节相同，原数值门槛通过。

6.74 秒音频三轮：候选 **RTF 0.8635 / p95 186.83 ms**，同期重新测量的控制版为 **0.8711 / 190.76 ms**。
图规划平均从 **283.70 降至 231.07 ms**，总耗时降低 **0.88%**、p95 降低 **2.06%**。
两版各轮均无新图或淘汰，均重载 121 次、直接执行 4209 次，正文相同。
此前控制版三轮为 0.8556 / 188.02 ms；候选相对该旧测量并未更快，因此不能把阶段计时的改善直接等同于稳定的整体提速。

同期 40 秒连续输入均报告 35.68 秒、积压 4.18 秒。候选为 **1.1000 / 323.74 ms**，控制为 **1.1012 / 322.77 ms**，
整体接近。图规划从 1605.66 降至 1277.19 ms（**20.46%**）；候选 3 张新图均触发系统编译，耗时 281.02 ms，
控制的 3 张新图命中系统缓存。全部实测耗时均保留，未排除这部分编译。
两版均无淘汰、重载失败或推理错误，提交事件的时间点与正文逐条相同；停止响应为 **1.25 / 12.03 ms**，只有一个 final、delta 正确。
RSS 采样峰值分别为 **20.32 / 23.78 GB**，仅单轮，不将差值归因于键格式。

候选源码与二进制保存在 `binary-plan-source/` / `binary-plan-bin/`，当前工作树已继续上节的 IR 遍历优化；
已完成完整语料评测的保留构建仍是下一节的 `shared-descriptor-bin`。
本阶段单独构建尚无完整集 CER 或 30 分钟结果；与 IR 遍历组合后的验证见上节。
证据：[后端测试](../artifacts/burn-ane-goal/binary-plan-tests.log)、
[模型位对照](../artifacts/burn-ane-goal/binary-plan-parity-byte-comparison.json)、
[同期短测对照](../artifacts/burn-ane-goal/binary-plan-short-comparison.json)、
[同期连续输入对照](../artifacts/burn-ane-goal/binary-plan-rolling-comparison.json)。

### 相同描述权重共享（上一完整集对照）

不同 row/KV 形状的 Direct 图若包含完全相同的打包权重，现在共享一个不可变、独立持有内存的 NSData。
按完整 blob 内容（含布局头和偏移）匹配；图缓存按各图 MIL 文本加唯一权重分配计费，仍使用原 16 GiB 上限。
4 GiB 打包缓存、112 个已加载程序和 1024 个描述的限制不变。另保留未共享时的累计字节统计，
这两种源文件字节口径均不包含系统运行时副本和 IOSurface，不能作为 RSS 上限。

71 项后端测试、192 组原生缓冲区检查、后端 Clippy 和使用原四项豁免的根 Clippy 通过。
64 KiB 预算压力测试验证三个输入形状共用同一权重，替换权重时释放全部旧描述持有者，重载后数值正确；
低于单图大小的预算不保留描述权重。encoder、prefill 与后续 8 token 的 20 份模型数组与 host-layout 逐字节相同，
原 5% 数值门槛通过。row128 与 row64 两种配置都通过同一检查，均为 20 / 20 份数组逐字节相同。
当前保留共享权重实现，性能验收采用 row64 / KV256 / 单 token 融合；host-layout 完整源码和二进制保留作对照。
源码冻结在 `shared-descriptor-source-sha256.json`，二进制为 `shared-descriptor-bin/`；
[后端测试](../artifacts/burn-ane-goal/shared-descriptor-budget-tests.log)、
[原生检查](../artifacts/burn-ane-goal/shared-descriptor-buffer-tests.log)、
[模型位对照](../artifacts/burn-ane-goal/shared-descriptor-parity-byte-comparison.json)。

KV256 / 单 token 融合，6.74 秒音频的单次配置筛选结果：row128 为 **0.9137 / 188.99 ms**，
row64 为 **0.8761 / 189.87 ms**，row32 为 **0.9975 / 301.87 ms**；正文一致，直接 ANE 检查通过。
三种配置热态均无新图和淘汰，但程序重载分别为 90 / 121 / 365 次，row32 未采用。
row64 另外三次独立重复为 0.8570 / 187.07 ms、0.8554 / 189.86 ms、0.8545 / 187.65 ms，
汇总 **RTF 0.8556 / p95 188.02 ms**。短探针仍未达标。

| 候选连续输入 40 秒 | 报告音频 | RTF | p95 | 停止时积压 | 停止响应 | RSS 采样峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| row128 | 34.88 s | 1.1344 | 302.64 ms | 4.98 s | 2.43 ms | 26.06 GB |
| row64 | 36.32 s | 1.0774 | 305.06 ms | 3.54 s | 22.44 ms | 23.02 GB |

两轮均为 32 秒连续预热、160 ms chunk / lookahead，均只有一个 final，delta 正确，无错误或 token 预算事件。
各有 3 次热态新图、0 次淘汰和 0 次重载失败；实例重载分别为 1058 / 1506 次。
row64 期末 381 张图的未共享序列化大小为 **22.94 GB**，共享后计费 **8.26 GB**，
打包缓存 **4.08 GB**，加载程序 112 个。旧 row64 连续探针的 407 次新建图 / 451 次淘汰已消除。
同参数 row128 的旧版 RSS 峰值为 33.73 GB，此处为 26.06 GB；其单轮速度略慢，不能把内存下降当作计算加速。
各轮实际处理音频不同；另保留等长 34.88 秒已提交步骤的诊断对照，不代替包含取消计算的完整流式指标。
完整 96 条集一轮为 **RTF 0.8206、p95 189.63 ms、CER 3.1052%**（44 / 1417），
96 条原始转写与 host-layout 逐条相同。CER 相对 Metal **+0.0706 个百分点**，精度和直接 ANE 检查通过。
相比 host-layout / row128，整体耗时降低 **5.86%**，p95 **增加 1.88%**；这包含共享权重与 row64 配置两项变化。
热态 0 次新图/淘汰，8874 次实例重载、0 次重载失败。direct evaluate 从 **324.47 降至 297.36 秒**，
prediction 从 **356.24 降至 327.43 秒**，加载路径却从 **19.92 增至 25.71 秒**，实例重载次数增加。
图规划仍需 19.79 秒，原生输入准备/复制 24.87 秒；阶段之间存在包含关系，不能简单求和。
性能门槛未通过：96 条中 45 条的 RTF ≤ 0.8，只有 2 条的 p95 ≤ 160 ms。
三轮完整重复与 30 分钟持续流验收尚未完成，目标保持进行中。
证据：[三轮短探针](../artifacts/burn-ane-goal/shared-descriptor-row64-repeat-timing/summary.json)、
[row64 连续输入](../artifacts/burn-ane-goal/shared-descriptor-row64-rolling-summary.json)、
[row128 连续输入](../artifacts/burn-ane-goal/shared-descriptor-row128-rolling-summary.json)、
[等长步骤对照](../artifacts/burn-ane-goal/shared-descriptor-rolling-common-prefix.json)、
[完整集对照](../artifacts/burn-ane-goal/shared-descriptor-row64-corpus-comparison.json)、
[row64 模型位对照](../artifacts/burn-ane-goal/shared-descriptor-row64-parity-byte-comparison.json)。

### 主机索引与布局复制（上一轮对照版）

Direct 允许回退时，已持有的 FP16/F32 Select、Cat、步长为 1 的 Slice 直接复制元素位模式，
减少先上传 Metal 再回读的往返。它们仍计入回退；矩阵计算、严格模式和缓存预算不变。
67 项后端测试、后端 Clippy 和使用原四项豁免的根 Clippy 通过；原生桥接未变，沿用 192 组缓冲区验证。
完整模型 20 份数组与原生输出保留版逐字节相同，原 5% 数值门槛通过。

row128 / KV256 / 单 token 融合，6.74 秒 WAV 三次独立热态探针为 **RTF 0.9082、p95 188.22 ms**，
比上一版三轮降低约 2.24% / 1.88%，正文一致，直接 ANE 检查通过。首轮输入准备从 313.27 降至 83.21 ms，
649 次主机复制耗时 6.53 ms；这项时间嵌套于回退调度，不能重复求和。
40 秒持续输入报告音频 **35.68 秒**、积压 **4.18 秒**、RTF **1.1109**、p95 **293.50 ms**，
停止响应 **33.63 ms**；一个 final、delta 正确、无错误或 token 预算事件。
采样 RSS 峰值升至 **33.73 GB**、期末 **33.27 GB**；序列化缓存预算与 112 个加载程序上限仍满足。
热态 3 次新建图、0 次淘汰、1072 次实例重载且无失败。实际处理跨度不同，不用累计阶段计数推导等量音频收益。
完整 96 条集一轮为 **RTF 0.8716、p95 186.13 ms、CER 3.1052%**（44 / 1417）。
相对原生输出保留版，RTF 降低 **3.46%**、p95 降低 **3.78%**，96 条原始转写逐条相同；
CER 相对 Metal **+0.0706 个百分点**，精度与直接 ANE 检查通过，性能未通过。
热态 0 次新建图/淘汰、6372 次实例重载且无失败；输入准备从 **22.77 降至 6.15 秒**，
43190 次主机复制共 449.13 ms，direct evaluate 仍约 324.47 秒，主要收益来自减少回退设备往返。
三轮完整重复和 30 分钟验收尚未完成。[完整集对照](../artifacts/burn-ane-goal/host-layout-corpus-comparison.json)。

row64 单次短探针改善到 0.8596 / 186.93 ms，但同样的连续输入产生 **407 次新建图、451 次淘汰**，
积压 **21.62 秒**，RTF **2.2971**、p95 **2667.19 ms**，停止响应 **2220.54 ms**。
因此这一阶段保留 row128；共享权重后重新验证 row64 的结果见上节。
证据：[三轮短探针](../artifacts/burn-ane-goal/host-layout-timing/summary.json)、
[连续流](../artifacts/burn-ane-goal/host-layout-rolling-summary.json)、
[row64 连续流](../artifacts/burn-ane-goal/host-layout-row64-rolling-summary.json)、
[模型位对照](../artifacts/burn-ane-goal/host-layout-parity-byte-comparison.json)。
该对照的冻结源码及二进制指纹为 `host-layout-source-sha256.json` / `host-layout-binary-sha256.json`。

### 宽投影与独立行小块实验，未采用

同输入的 Q/K/V、gate/up 和部分词表投影尝试合为一个宽卷积。71 项后端测试通过，
16,384 通道上限的 20 份模型数组逐字节相同；三轮短探针为 **0.9118 / 188.26 ms**，没有确认收益。
12,288 通道上限单轮为 0.9014 / 185.25 ms，4,096 上限单轮为 0.9053 / 186.13 ms。
12,288 上限的 40 秒连续输入为 **1.1170 / 291.80 ms**、积压 **4.34 秒**，采样 RSS 峰值 **23.27 GB**；
速度未改善，较低内存峰值只有单轮证据。

另一实验将已证明各行独立的投影图固定为 64 行，较长输入重复调用同一个程序，注意力与 token 选择保留完整图。
75 项后端测试和 20 份模型数组对照通过，但三轮短探针退步为 **0.9590 / 214.71 ms**。
每轮额外拆分 560 张图，物化次数增至 4857，额外调用与回读抵消了补齐计算节省。
两项实验均已归档，验证共享权重前曾将 42 份源码和 release 二进制恢复并校验为 `host-layout`；
没有将实验结果当作完整 CER 或 30 分钟验收。
细节与复现参数见 [研究记录](ane-backend-research.md)，源码、二进制和日志分别保存在
`artifacts/burn-ane-goal/parallel-projection-*` 与 `artifacts/burn-ane-goal/row-tile-*`。

### 原生输出跨图保留

Direct FP16 输出保留为有所有权的 IOSurface 视图。下一张图按逻辑有效区段复制，省去 Rust 中的完整回读、
trim、TensorData 存储和重新补齐；公开读取、回退或设备转移仍按需物化，其他 dtype 保留原有边界转换。
每个图输出槽有可复用缓冲池，只重写没有存活张量引用的缓冲；图淘汰后张量仍独立拥有 surface。
这仍有 surface 到 surface 的复制，不是零复制。缓存预算和 FP16 计算未改变。

61 项后端测试（5 单元、27 通用/Core ML、29 direct）、192 组原生缓冲区用例、后端 Clippy 通过；
根 Clippy 使用原四项豁免。新增回归覆盖旧分支、多输出、缓冲稳定复用、缓存淘汰、程序重载，以及 KV 追加/回滚/扩容。
原生测试独立验证补齐空洞清零、位模式、不同补齐轴和系统物理 reshape。
候选源码冻结为 `native-resident-source-sha256.json`。encoder、prefill、后续 8 token 的 **20 份输出与传输合并版逐字节相同**，
原 5% 数值门槛通过，推理无新增回退。

同一 6.74 秒 WAV、三次独立进程完整预热后，RTF **0.9290**、p95 **191.83 ms**，
相比传输合并版三轮的 0.9662 / 199.27 ms，分别降低 **3.85% / 3.74%**。正文一致，直接 ANE 检查通过，性能门槛未通过。
首轮输出整理从 209.70 降到 22.48 ms；整体 prediction 为 5045.68 ms（控制版 5033.63 ms），
直接 evaluate 为 4606.74 ms（控制版 4466.42 ms），不能将回读减少等同于所有阶段都更快。
热态仍有 2132 次输出缓冲新建、13805 次复用，说明仅两个缓冲的微型循环不能代表跨层 KV 的真实工作集。

32 秒预热、40 秒持续到达输入的单次探针：报告音频 **34.72 秒**，积压 **5.14 秒**（控制版 7.22 秒），
RTF **1.1425**、p95 **300.45 ms**、停止响应 **50.13 ms**。停止后只发一个 final、delta 可还原正文、无错误或 token 预算事件。
采样 RSS 峰值 **27.03 GB**、期末 **24.36 GB**，高于传输合并控制版的 22.11 / 19.12 GB。
3 次新建图、0 次描述淘汰、1055 次实例重载、0 次重载失败；一部分输入超过预热范围，不声称全程没有新图。
期末图描述约 16.529 GB、打包权重约 4.076 GB、加载 112 个程序，仍在原序列化缓存预算内；它不是 RSS 上限。
期末有 76 个原生张量视图、补齐后 payload 约 37.74 MB；该值不含图池和对齐，不能解释全部进程内存。
两次连续探针处理音频跨度不同，不直接比较累计阶段计数作为等量音频速度比。

完整 96 条集已完成一轮：**RTF 0.9028、p95 193.44 ms、CER 3.1052%**（44 / 1417）。
与传输合并控制版逐条转写相同，CER 相对 Metal **+0.0706 个百分点**，精度和直接 ANE 检查通过。
完整集耗时降低 **4.28%**、p95 降低 **4.19%**；性能门槛未通过。
热态 0 次图新建/淘汰、6372 次实例重载且无失败；prediction 从 364.37 降到 355.46 秒，
输入准备从 26.07 降到 22.77 秒、输出整理从 13.36 降到 2.14 秒。
[完整集对照与阶段统计](../artifacts/burn-ane-goal/native-resident-corpus-comparison.json)。
三轮完整集和 30 分钟验收仍待完成。
证据：[三轮短探针](../artifacts/burn-ane-goal/native-resident-timing/summary.json)、
[连续流](../artifacts/burn-ane-goal/native-resident-rolling-summary.json)、
[模型位对照](../artifacts/burn-ane-goal/native-resident-parity-byte-comparison.json)。
细节、限制与证据见 [研究记录](ane-backend-research.md)。


### 传输时同步完成布局转换

直接在逻辑 FP16 张量与有步长的 IOSurface 之间复制，去掉每次调用用于输入/输出重排的两个中间缓冲。
保留编译器删除无用输入后的原偏移；按批次和小块完成低秩矩阵的转置，连续数据使用整块复制。
attention 保持原有分解表达；此前整块 SDPA 实验未观察到可确认的端到端收益，源码、失败日志与二进制另行保留，
详见 [SDPA 研究记录](ane-backend-research.md)。

26 项直接 ANE 回归和 27 组原生缓冲区测试通过；后者独立检查物理写入与逻辑读出，覆盖 NaN 位模式、
负零、奇数步长、批次、不同物理布局与输出边界。后端 Clippy、native 严格告警检查通过，根 Clippy 使用原四项豁免。
完整 encoder、prefill 和后续 8 token 的 **20 份输出均与编译描述缓存版逐字节相同**。
证据：[原生测试](../artifacts/burn-ane-goal/logical-io-native-tests.log)、
[直接 ANE 测试](../artifacts/burn-ane-goal/logical-io-direct-tests.log)、
[完整模型数组](../artifacts/burn-ane-goal/logical-io-parity-byte-comparison.json)。

同一 6.74 秒 WAV、row128 / KV256 / 单 token 融合，三次独立进程预热后的汇总
RTF **0.9662**、步骤 p95 **199.27 ms**；正文一致，直接 ANE 检查通过，实时门槛仍未通过。
编译描述缓存版单次为 1.0058 / 207.47 ms，不能将这个短探针的改善直接推广到完整集或长流。
第一轮 `prediction_ms` 从 5320.07 降至 5033.63 ms，直接执行次数仍为 4209，实例重载仍为 90。
native 输入/输出计时现包含布局转换，字段数值与旧版口径不同；跨版本比较应看 `prediction_ms` 和完整步骤耗时。
证据：[三次短探针](../artifacts/burn-ane-goal/logical-io-timing/summary.json)。
40 秒持续输入、32 秒连续预热后，报告音频 **32.64 秒**，停止积压 **7.22 秒**，
RTF **1.2153**、p95 **323.93 ms**、停止响应 **27.57 ms**。仍未实时。
测量期间 1 次新建图、0 次描述淘汰、1017 次实例重载、0 次重载失败；有一部分输入超过预热的 32 秒范围。
采样 RSS 峰值 **22.11 GB**，期末缓存约 16.529 GB、加载 112 个程序、打包权重约 4.076 GB，仍在原缓存预算内。
停止后只发一个最终事件，delta 可还原最终文本，无错误或 token 预算事件。
证据：[连续流汇总](../artifacts/burn-ane-goal/logical-io-rolling-summary.json)。
完整 96 条中文集已完成一轮：**RTF 0.9432、p95 201.90 ms、CER 3.1052%**（44 / 1417），
96 条原始转写与编译描述缓存版逐条相同。相对 Metal 为 **+0.0706 个百分点**，CER 和直接 ANE 检查通过。
相对缓存版完整集，耗时降低 **2.51%**、p95 降低 **2.58%**；仍未达到实时门槛。
热态 0 次新建图/描述淘汰、6372 次实例重载、0 次重载失败；加载路径 17.45 秒，直接 evaluate 323.00 秒，
整体 prediction 从 376.20 降到 364.37 秒。输入准备 26.07 秒、输出整理 13.36 秒，传输仍有优化空间。
证据：[完整集](../artifacts/burn-ane-goal/logical-io-corpus-v1-r1/summary.json)、
[逐条对照与阶段统计](../artifacts/burn-ane-goal/logical-io-corpus-comparison.json)。
以上仅一轮完整测量，三轮重复及 30 分钟验收仍待完成。

原生缓冲区测试可单独复现，不执行 ANE 模型：

```sh
clang -fobjc-arc -fblocks -O2 -Wall -Wextra -Werror \
  -framework Foundation -framework IOSurface \
  crates/burn-ane/native/DirectTests.m -o /tmp/burn-ane-transfer-tests
/tmp/burn-ane-transfer-tests
```

### 扩大融合图的对照，未采用

在传输优化之上，分别把每图常量上限改为 512 / 1024 MiB，操作上限仍为 512。
常量上限按 FP32 等价元素字节计，只控制图切分；总缓存仍为 16 GiB / 4 GiB / 112 个加载程序。
两种配置均通过 encoder、prefill、后续 8 token 的原 5% 数值门槛，推理无新增回退或编译失败。
19 / 20 份数组与保留版逐字节相同，仅 encoder ANE 数组改变（相对 Metal RMSE 1.25815%）；decoder 全部不变。

同一 6.74 秒短探针，row128 / KV256 / 单 token 融合：

| 每图常量上限 | 测量次数 | RTF | p95 | 直接 ANE 调用 | evaluate 累计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 MiB，传输优化 | 3 | 0.9662 | 199.27 ms | 4209（首轮） | 4466.42 ms（首轮） |
| 512 MiB | 1 | 0.9762 | 198.96 ms | 3383 | 4463.99 ms |
| 1024 MiB | 1 | 0.9917 | 204.54 ms | 3012 | 4473.88 ms |

各配置正文相同、直接 ANE 检查通过、热态没有图重建或缓存淘汰，均有 90 次实例重载。
调用减少，但 evaluate 基本未变；准备的输入/输出合计从首轮 10.10 GB 增至 10.46 / 10.64 GB，
整体 `prediction_ms` 从 5033.63 增至 5076.01 / 5166.10 ms。此短探针没有显示收益，默认保持 256 MiB。
没有对更大图配置做完整 CER 或 30 分钟验收。
证据：[512 MiB](../artifacts/burn-ane-goal/graph512-timing/summary.json)、
[1024 MiB](../artifacts/burn-ane-goal/graph1024-timing/summary.json)。

保留诊断参数：CLI `--ane-max-constant-mib` / `--ane-max-ops`，examples 和评测脚本
`--max-constant-mib` / `--max-ops`。默认 256 MiB / 512 操作，未更改缓存总预算。
例如复现 512 MiB 单次探针：

```sh
python3 scripts/evaluate_streaming.py --backend ane \
  --binary artifacts/burn-ane-goal/graph-fusion-bin/r2t2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav \
  --repeats 1 --row-bucket 128 --kv-bucket 256 --fuse-single-token \
  --max-constant-mib 512 --output artifacts/graph512-recheck
```

### 程序槽使用纯 LRU 的对照，未采用

在相同传输实现和缓存预算下，将已加载程序的选择改为纯 LRU，保留源码和二进制作对照。
26 项 direct 测试与 Clippy 通过。短探针单次 RTF **0.9723**、p95 **199.52 ms**，
重载从传输版首轮 90 次增为 100 次，加载路径从 260.39 增为 308.32 ms，正文不变。

相同 32 秒预热、40 秒到达输入的单次连续探针，报告音频从 32.64 增到 **33.28 秒**，
停止积压从 7.22 降到 **6.58 秒**，RTF **1.1888**、p95 **308.10 ms**，停止响应 **7.92 ms**。
851 次实例重载、0 次重载失败、3 次新建图、0 次描述淘汰；无错误或 token 预算事件。
采样 RSS 峰值从约 **22.11 GB 增到 29.01 GB**。两次实际处理跨度不同，累计阶段计数不作为等量音频速度比。
连续探针有改善，但短探针略慢、内存峰值更高，且均未实时；暂保留原先优先卸载小图的策略。
这些均为单次连续测量，未完成纯 LRU 的完整 CER 或 30 分钟验收。
证据：[短探针](../artifacts/burn-ane-goal/lru-slots-timing/summary.json)、
[连续流汇总](../artifacts/burn-ane-goal/lru-slots-rolling-summary.json)。

### 编译描述与程序实例分开缓存

连续流的热态重载会重新构建完整图描述、复制权重和分配 I/O 缓冲。现在将编译图缓存与
ANE 程序槽分别管理：释放程序槽时保留编译描述和缓冲，下次重新创建并加载模型实例。
仍限制最多 112 个已加载程序，编译图（含卸载的图）仍受原 16 GiB / 1024 项预算约束，
打包权重仍为 4 GiB；不把新保留的描述排除在字节预算之外。

首版仅调用 native `unload`，小图测试通过，但 140 程序压力测试出现资源不足；
释放整个模型实例后解决。保留失败日志和 native 源码，不能把首次小图成功视为资源释放已验证。
新的硬件回归用 2 个程序槽循环执行 4 张变输入、多输出图：只构建 4 次，随后重载 8 次，
输出与 NdArray 一致，无 Burn 回退。完整后端 **49 项测试**（5 单元、27 通用/Core ML、17 direct）、
后端 all-targets Clippy 和 native Clang `-Wall -Wextra -Werror` 通过。
根项目 Clippy 在原有四类告警豁免下通过。

完整模型 encoder、prefill、后续 8 token 的 **20 份 ANE/Metal 输出逐字节不变**，原 5% 数值阈值不变。
证据：[数组对照](../artifacts/burn-ane-goal/residency-parity-byte-comparison.json)、
[数值记录](../artifacts/burn-ane-goal/residency-parity.jsonl)、
[完整后端测试](../artifacts/burn-ane-goal/residency-final-tests.log)。

6.74 秒探针先完整预热后单次测量，KV256 / 单 token 融合：

| 行分档 | 热态 RTF | 步骤 p95 | 热态新建图 / 描述淘汰 | 结论 |
| --- | ---: | ---: | ---: | --- |
| row128 | 1.0058 | 207.47 ms | 0 / 0 | 仍未达标 |
| row64 | 0.9736 | 214.85 ms | 0 / 0 | p95 未改善，未采用为默认 |
| row32 | 2.7583 | 2651.38 ms | 221 / 221 | 字节预算淘汰导致明显退步 |

三次转写正文相同，直接 ANE 检查均通过。row128 热态有 90 次实例重载、0 次失败，
加载路径 247.15 ms；拼接边界保留版同配置首轮为 806.55 ms。
保留版短探针三轮汇总 1.1148 / 227.08 ms，新版此处只有一次，尚未完成重复性能验证。
证据：[row128](../artifacts/burn-ane-goal/residency-timing/summary.json)、
[row64](../artifacts/burn-ane-goal/residency-row64-timing/summary.json)、
[row32 失败方向](../artifacts/burn-ane-goal/residency-row32-timing/summary.json)。

相同 row128 / KV256，32 秒连续预热后、40 秒持续到达输入的单次探针：
报告音频从保留版的 18.72 秒增到 **32.00 秒**，停止时积压从 21.14 秒降到 **7.86 秒**；
RTF **1.2399**、p95 **324.55 ms**，仍未实时。停止响应 **12.87 ms**，无错误或 token 预算事件。
测量期间 **0 次图重建、0 次编译图淘汰、949 次实例重载、0 次重载失败**，
加载路径耗时 2.98 秒，直接 ANE evaluate 25.77 秒。
32 秒处理跨度与旧版的 18.72 秒不同，阶段累计耗时不能作为等量音频的直接速度比。
期末保留 314 张图、加载 112 个程序；模型描述约 16.53 GB、打包权重约 4.08 GB，仍在原预算内。
RSS 抽样峰值约 **26.00 GB**，高于保留版约 9.28 GB；保留描述增加了实际内存占用，不能只报告加载提速。
证据：[连续流原始事件](../artifacts/burn-ane-goal/residency-rolling-smoke.jsonl)。

该缓存版本已独立完成 **全部 96 条、每条一次测量**（先完整预热，独立 Session）：

| row128 / KV256 / 单 token 融合 | 汇总 RTF | 所有步骤 p95 | CER |
| --- | ---: | ---: | ---: |
| 拼接边界保留版 | 1.0613 | 226.03 ms | 3.1052% |
| 编译描述缓存版 | **0.9675** | **207.26 ms** | **3.1052%** |

96 条原始转写逐条相同，44 / 1417 错误；相对 Metal FP16 的 CER 差仍为 **+0.0706 个百分点**，
准确率和直接 ANE 检查通过。RTF 降低 8.83%、p95 降低 8.30%，但实时性能目标未通过。
完整集测量阶段 0 次图新建、0 次描述淘汰、6372 次实例重载、0 次重载失败；
加载路径从 55.79 秒降至 **17.66 秒**，直接 evaluate 为 **321.31 秒**。
完整集汇总 RTF 低于 1 不能替代前述仍积压的连续流测试。
证据：[完整集结果](../artifacts/burn-ane-goal/residency-corpus-v1-r1/summary.json)。
此结果仅对应保存的 `residency-bin`，不覆盖后续注意力实验。三轮完整集性能复测及 30 分钟连续流式尚未完成，目标保持进行中。

### 减少复制与调整拼接边界（保留对照）

该版本减少了主机输入/输出的中间复制，并在 token 拼接前结束含投影权重的图，
使后续 transformer 图能按行数分档复用。使用 row128 / KV256 / 单 token 融合，
全部 96 条固定中文集的一轮热态测量：

| 完整固定集，160 ms / 160 ms | 汇总 RTF | 所有步骤 p95 | CER |
| --- | ---: | ---: | ---: |
| Metal FP16 对照 | 0.3417 | 78.73 ms | 3.0346% |
| 修复 sigmoid 后的 ANE 保留版本 | 1.1952 | 314.68 ms | 3.1052% |
| 当前 ANE，减少复制和调整拼接边界 | 1.0613 | 226.03 ms | 3.1052% |

当前版相对 ANE 保留版本的汇总计算耗时下降 **11.2%**，p95 下降 **28.2%**。
96 条归一化转写全部相同，CER 相对 Metal 增加 **0.0706 个百分点**，准确率和直接 ANE 检查通过。
性能门槛仍未通过；这是一轮完整集结果，尚未完成三轮性能复测。
证据：[当前完整集](../artifacts/burn-ane-goal/concat-corpus-v1-r1/summary.json)。

收益并不均匀。8.14 秒诊断片段的 RTF 从 1.853 降至 1.096；但 6.74 秒上游 WAV
三次独立进程测量从 **1.0766 / 222.43 ms** 退步到 **1.1148 / 227.08 ms**。
只减少复制、尚未调整拼接边界的中间版单次为 **1.0325 / 211.54 ms**。
采用当前边界基于完整集改善，不能宣称每段音频都更快。
证据：[当前短探针三次](../artifacts/burn-ane-goal/concat-timing/summary.json)、
[仅减少复制单次](../artifacts/burn-ane-goal/host-copy-timing/summary.json)。

减少复制后的 20 份模型 ANE/Metal 输出与 sigmoid 修复版逐字节相同。
拼接边界修改另外通过覆盖变长、多 batch 投影的硬件回归测试，以及上述完整集 CER 检查；
这不等于证明所有长度的浮点输出逐字节不变。
当前后端 **47 项测试**（4 单元、27 通用/Core ML、16 direct）及 all-targets Clippy 通过。
根项目 7 项 CPU 模型测试、3 项麦克风队列停止测试、3 项 Python 评测测试通过。
根项目 Clippy 仍需下文列出的四类既有告警豁免。

40 秒连续输入探针只处理到 18.72 秒，RTF **2.1221**、p95 **938.70 ms**，
结束时积压 **21.14 秒**；30 分钟验收尚未执行。短片段的独立 Session 性能不能代表长流。
详见文末的连续输入和停止结果。

### 历史优化对照

修复 direct ANE sigmoid 后、减少复制和调整拼接边界之前，在同一段 6.74 秒 WAV 上进行三次独立进程热态测量：

| 保留版本，160 ms / 160 ms | 三次汇总 RTF | 所有步骤 p95 | 实时目标 |
| --- | ---: | ---: | --- |
| row128 / KV32，默认分离 | 1.1051 | 230.69 ms | 未通过 |
| row128 / KV256，单 token 融合 | 1.0766 | 222.43 ms | 未通过 |

每次先完整预热，含最终 `finish`，文本一致；两种配置的直接 ANE 执行检查均通过。
证据：[默认配置](../artifacts/burn-ane-goal/stable-default-timing/summary.json)、
[融合配置](../artifacts/burn-ane-goal/stable-fused-timing/summary.json)。融合默认仍关闭。
以下更早的速度试验使用修复前的 sigmoid，只用于保留优化对照，不代表当前版本性能。

初始 6.74 秒上游 WAV 单次热态 RTF **1.167**、p95 **243.19 ms**，见
[FP16 对照](../artifacts/burn-ane-fp16/summary.json)。该 WAV 没有语料级 CER 结论。

类型化 IR 遍历替代图规划中的 JSON Value 改写与反序列化，保留标量、选项、权重身份、输出活性和动态掩码规则。
首轮热态 RTF **1.063**、p95 **226.24 ms**，图规划累计 **273.80 ms**（此前约 937.96 ms）。
5770 次直接 ANE 调用、113 次重载、702 个 Burn 回退操作与基线相同，没有重新读取常量权重。
完整模型 encoder、prefill、后续三个缓存 token 的 10 份 ANE/Metal 数组与优化前逐字节一致。
证据：[测量](../artifacts/burn-ane-goal/typed-ir-timing/summary.json)、[数组对照](../artifacts/burn-ane-goal/parity-byte-comparison.json)。

单 token 共用 row bucket 的初始试验变慢：row64 RTF 1.960、row128 RTF 1.492。
发现 `valid == 1` 时整段切片没有终止补齐，导致词表投影也被补齐；修复此情况后 row64 仍为 RTF 1.970。
跨编译图后的独立单行投影仍可能被补齐，该方案没有改善整机性能。
`--ane-pad-single-row` 默认关闭，不将未验证的试验当作推荐配置。

另加可选 `--ane-fuse-single-token`：只有当前 token 数为 1 时合并投影、KV attention 与 FFN，
多 token prefill 继续使用分离图。KV32 时热态仍有 193 次重载，RTF 2.768；KV256 时重载降至 53 次，
直接 ANE 调用从 5770 降至 4172，RTF **1.036**、p95 **212.45 ms**。这是修复 sigmoid 前的单次测量，
仍未达标，且完整 KV 的补齐传输从约 6.64 GB 增到 10.28 GB，不能只按调用次数判断提速。

## 固定中文集

AISHELL-1 test 的 96 条完整音频，从三个分片各取 32 条，按
`SHA256("omni-r2t2-ane-mandarin-v1:分片号:行号")` 排序抽取，在推理前固定，不按识别结果筛选。
实际生成 **96 条、494.52175 秒、1417 个参考字符**。保存人工参考、源行号、时长与 WAV SHA256。
镜像未提供原始文件名或说话人 ID，`source_audio_path` 为 null；不声称覆盖了特定数量的说话人。
固定清单：[mandarin-v1.json](eval/mandarin-v1.json)。重复准备已验证输出清单不变。

AISHELL-1 是安静室内录制的 16 kHz 普通话朗读语料；本集不能代表噪声、方言或自由对话。
[OpenSLR 原始说明与许可](https://www.openslr.org/33/)。下载使用 AudioBench 发布的
[AISHELL-1 test 镜像](https://huggingface.co/datasets/AudioLLMs/aishell_1_zh_test)，固定版本
`e73dc7f096ecea5c8e5c722446c8d3b4014fdd18`，三个 Parquet 文件校验发布的 SHA256。

CER 归一化 v1：NFKC、英文字母小写、去空白和 Unicode 标点。保留数字、符号、简繁差异，不逐句纠错。
记录非最终步骤达到生成 token 预算的次数；最终步骤仍达到预算则拒绝该测量，防止不完整转写被当成完整结果。

```sh
python3 -m venv .venv-eval
.venv-eval/bin/python -m pip install pyarrow==21.0.0
.venv-eval/bin/python scripts/prepare_mandarin_eval.py
```

生成 `docs/eval/mandarin-v1.json` 与 `artifacts/eval-corpus/mandarin-v1/*.wav`。
脚本拒绝用不同内容覆盖已有固定清单。

## 复现评测

```sh
cargo build --release --features ane --bin r2t2 --example ane_blocks --example stream_eval

python3 scripts/evaluate_streaming.py --backend metal \
  --batch-binary target/release/examples/stream_eval \
  --manifest docs/eval/mandarin-v1.json --repeats 3 \
  --output artifacts/eval-metal-v1

python3 scripts/evaluate_streaming.py --backend ane \
  --batch-binary target/release/examples/stream_eval \
  --manifest docs/eval/mandarin-v1.json --repeats 3 \
  --row-bucket 128 --kv-bucket 256 --fuse-single-token \
  --compare artifacts/eval-metal-v1/summary.json \
  --output artifacts/eval-ane-v1
```

两个后端顺序运行，测量时不并行构建或跑另一后端。batch 模式共享模型和编译缓存，每条音频使用独立 Session，先完整热身一次再测量。
独立进程探针使用 `--audio FILE.wav`，不传 `--batch-binary`；无人工参考时不输出 CER。
`--case-id ID` 是诊断子集，不能当成 96 条完整验收。
以上明确启用当前评测的单 token 融合；库默认仍关闭。分离图对照使用 `--kv-bucket 32` 并去掉 `--fuse-single-token`。
三条跨分片 smoke 音频的 Metal 与 ANE 文本均匹配人工参考（47 字符，CER 均为 0）；
ANE RTF 1.0291、p95 221.40 ms，仍不达性能门槛。这只是评测流程检查，不能代替完整语料。
记录：[Metal smoke](../artifacts/burn-ane-goal/metal-corpus-smoke/summary.json)、
[ANE smoke](../artifacts/burn-ane-goal/ane-corpus-smoke/summary.json)。

修复 sigmoid 的保留版本已完成 **全部 96 条、每条 1 次测量**（各先完整热身，独立 Session）：

| 后端 | 编辑错误 / 参考字符 | CER | 汇总 RTF | 所有步骤 p95 |
| --- | ---: | ---: | ---: | ---: |
| Metal FP16 | 43 / 1417 | 3.0346% | 0.3417 | 78.73 ms |
| Direct ANE，row128 / KV256 / 单 token 融合 | 44 / 1417 | 3.1052% | 1.1952 | 314.68 ms |

ANE CER 增加 **0.0706 个百分点**，通过 ≤ 0.2 个百分点的固定集门槛，直接 ANE 执行检查也通过。
仅一条归一化转写不同：参考中的“中新网”，Metal 输出“中新网”，ANE 输出“中新闻网”；
两边对人名均有同一个字的错误。[逐句差异](../artifacts/burn-ane-goal/corpus-v1-r1-differences.json)。
这是一轮完整集对照，尚未做该集合的三轮性能复测，且不能推广为其他语料的准确率保证。
证据：[Metal 完整集](../artifacts/burn-ane-goal/metal-corpus-v1-r1/summary.json)、
[ANE 完整集](../artifacts/burn-ane-goal/ane-corpus-v1-r1/summary.json)。

该保留版本的性能门槛未通过。部分 8–12 秒片段 RTF 达 1.89–1.92，热态仍有 165–226 次图重载，
加载路径耗时 6.8–9.2 秒；不能用短音频探针的 1.077 / 222 ms 代表整个固定集。
当前版已独立复测全部 96 条，结果见本轮进展；保留旧版二进制和原始记录供比较。

结果保留二进制、模型配置/权重、清单和音频指纹，命令、原始事件、阶段计数、逐句 CER 与各项通过标志。
`complete` 仅代表请求的测量执行完毕，不等于 `performance_pass`、`direct_ane_pass`、`cer_pass` 通过，也不包含长流和停止验收。

## 数值诊断

原失败用例是 8 token prefill 加第 3 个缓存 token，最终 logits 相对 RMSE **5.271%**，greedy token 相同。
这实际只有 11 个 token，不能当作长上下文验证。扩展到后续 8 步，9 次 decoder 调用最高分 token 全部一致，原第 3 步仍失败。
第 3 步 hidden state 误差第 1 层为 1.72%、第 28 层为 3.84%，最终 logits 为 5.27%；尚不能据此归因于某个算子。
[逐层汇总](../artifacts/burn-ane-goal/parity-layer-summary.json)保留全部 261 个阶段。
逐层诊断的前四次 decoder 最终输出与原始控制逐字节相同。

层内 trace 的 ANE 最终 logits 与原控制逐字节相同，Metal 会因增加中间输出而发生变化。
因此不能拿 trace 模式中降到 4.934% 的误差冒充原用例已经修复。
对每个后端**各自捕获的 gate 输入**做 FP64 SiLU、再舍入 FP16，发现 ANE 激活的相对误差
2.85%–2.98%，Metal 为约 0.034%–0.035%。例如输入 0.5，ANE 输出约 0.296387，参考为 0.311279。
[独立数学对照](../artifacts/burn-ane-goal/silu-fp64-diagnostic.json)把投影之前的误差排除在外。
新增 sigmoid/SiLU 中心区间及饱和测试在旧实现失败，改用有界 `exp(-abs(x))` 表达式后通过。
正半轴输出 `1 / (1 + e)`，负半轴输出 `e / (1 + e)`，其中 `e = exp(-abs(x))`；仍在直接 ANE 图内执行。

完整模型复测已通过：原第 3 个缓存 token 的相对 RMSE 从 **5.271% 降到 0.197%**，
不含 trace 的 Metal 参考与旧控制逐字节相同，原 5% 断言保持不变。
8 个缓存步骤的相对 RMSE 为 0.323%、0.598%、0.197%、0.248%、0.401%、0.947%、0.504%、0.811%，
prefill 为 0.412%；最高分 token 全部一致，推理阶段新增 Burn 回退为 0。Encoder 仍为 1.239%。
单 token 融合的 decoder 输出与修复后的默认分离配置逐字节相同。
同输入 SiLU 数学对照误差降至约 0.051%，参见
[完整数值记录](../artifacts/burn-ane-goal/stable-parity.jsonl)、
[融合数值记录](../artifacts/burn-ane-goal/stable-fused-parity.jsonl)、
[激活对照](../artifacts/burn-ane-goal/stable-silu-fp64.json)。这不替代真实语料 CER 或长上下文验证。

sigmoid 修复阶段后端 44 项测试（2 项缓存单元、27 项通用/Core ML、15 项直接 ANE 硬件）通过，
根项目 7 项 CPU 模型测试、3 项 Python 评测测试及两处 fmt 通过。
后端 all-targets Clippy `-D warnings` 通过；根项目严格 Clippy 仍有既有风格告警，
仅在命令行豁免 `manual_is_multiple_of`、`collapsible_if`、`single_range_in_vec_init`、
`chunks_exact_to_as_chunks` 四类后通过，未修改相关模型代码来掩盖告警。

```sh
target/release/examples/ane_blocks --ane-runtime direct --case decoder \
  --model models/Confucius4-R2T2 --row-bucket 128 --kv-bucket 32 \
  --model-steps 8 --layer-trace --collect-failures \
  --dump-dir artifacts/parity-layer-arrays

target/release/examples/ane_blocks --ane-runtime direct --case decoder \
  --model models/Confucius4-R2T2 --row-bucket 128 --kv-bucket 32 \
  --model-steps 3 --activation-trace --collect-failures \
  --dump-dir artifacts/parity-activation-arrays

python3 scripts/analyze_activation_trace.py artifacts/parity-activation-arrays \
  --output artifacts/silu-fp64.json
```

`--collect-failures` 只把断言推迟到全部数组写出后，失败仍返回非零退出码。默认仍在首次失败处终止。
新增中间输出可能改变融合，诊断耗时不作为性能结果。

## 连续输入与停止验证工具

`stream_soak` 将固定清单内的完整音频按顺序连接、循环为 20 ms PCM 包，按单调时钟模拟输入到达，
始终使用同一个 Session，覆盖 16 秒音频窗口滚动。音源为预录音频，不能据此声称已验证物理麦克风驱动。
落后时保留全部待处理音频，以到达时间与已提交样本数计算积压；没有用丢帧掩盖吞吐不足。
输出逐步事件、每 30 秒的 ANE 计数/RSS 抽样、积压、最终状态和停止响应。
进程 RSS 不包含所有统一内存分配，抽样峰值也不等于瞬时最大值。

```sh
cargo build --release --features ane,microphone --example stream_soak

# 30 分钟持续到达输入，然后通过 Session stop handle 发出测试停止请求。
# --seconds 是测试输入长度，不是产品会话时限。
target/release/examples/stream_soak --backend ane \
  --manifest docs/eval/mandarin-v1.json --seconds 1800 --warmup-seconds 32 \
  --row-bucket 128 --kv-bucket 256 --fuse-single-token \
  --stop-after-ms 1800000 > artifacts/ane-soak.jsonl 2> artifacts/ane-soak.log
```

也可不传 `--stop-after-ms`，用 Ctrl-C 在任意输入/推理阶段发送停止请求。
停止不刷新队列、不追加最终 token 预算，只保留已提交文字；正在执行的硬件调用需返回到中断检查点。
`cancelled_compute_ms` 记录未形成提交事件的在途计算，计入 RTF；`stop_response_ms` 从实际停止请求计到终止 Session。
`summary.status` 为 `interrupted` 仅说明中断生效，不能代替 30 分钟、性能、准确率等验收。
已完成当前 release 的三种停止检查：

| 停止时机 | Session 响应时间 | 结果 |
| --- | ---: | --- |
| 音频计算开始前，通过 stop handle | 19.97 ms | 通过 |
| 第一个计算步骤执行中，通过 stop handle | 42.04 ms | 通过 |
| 收到首个非最终事件后发送 OS SIGINT | 16.52 ms | 通过 |

均只产生一个最终事件，最终 delta 为空、不追加生成 token，全部 delta 可还原最终文本。
CLI 麦克风现在直接使用 Session 的 stop handle，停止后丢弃未处理队列和重采样尾部；
队列停止的三个单元测试通过。上述硬件检查使用预录 PCM，尚未在本轮实测物理麦克风驱动。
证据：[停止检查](../artifacts/burn-ane-goal/concat-stop/summary.json)。复现：

```sh
python3 scripts/verify_stream_stop.py \
  --binary target/release/examples/stream_soak \
  --manifest docs/eval/mandarin-v1.json --output artifacts/ane-stop-check
```

当前 row128 / KV256 / 单 token 融合版本已完成 40 秒持续到达输入探针，先用相同连续路径预热 32 秒音频。
在计时 40 秒发出 stop handle 请求，**49.82 ms** 后 Session 结束。
已提交 18.86 秒 PCM，已报告处理结果覆盖 18.72 秒音频，积压 21.14 秒、峰值 21.28 秒，
RTF **2.1221**、p95 **938.70 ms**。已跨过一次 16 秒滚动边界，无错误或 token 预算事件，
但吞吐严重不足，不能把成功中断解释为稳定实时。
测量阶段 585 次图加载全部命中系统编译缓存，加载路径仍耗时 **17.42 秒**，
直接 ANE evaluate 为 **14.54 秒**；这里已扣除预热计数，不能把日志末尾累计值当作测量增量。
结束时保留 112 个程序、序列化模型约 12.27 GB、打包权重约 4.08 GB，处于固定预算内。
进程 RSS 抽样峰值约 9.28 GB，不代表全部统一内存占用。
证据：[连续探针原始事件](../artifacts/burn-ane-goal/concat-rolling-smoke.jsonl)。

同一二进制、相同输入/预热/40 秒停止方案，把 KV 分档改为 512 后的单次诊断：

| KV 分档 | 报告音频秒数 | 停止时积压 | 汇总 RTF | 步骤 p95 | 停止响应 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 18.72 | 21.14 秒 | 2.1221 | 938.70 ms | 49.82 ms |
| 512 | 25.76 | 14.10 秒 | 1.5419 | 354.32 ms | 36.77 ms |

KV512 的 537 次热态图加载仍均命中系统编译缓存，加载路径耗时 7.73 秒。
两次停止在不同音频位置，不能将阶段计数直接当成处理等量音频的对照；它们比较的是同等墙钟时长的积压。
KV512 仍未实时，且尚未完成该配置的完整固定集 CER 复测，保持为诊断参数，不替换已验证配置。
两个探针都只有 40 秒，不能代替 30 分钟连续运行。
证据：[KV512 原始事件](../artifacts/burn-ane-goal/concat-kv512-rolling-smoke.jsonl)。

拼接边界保留版已通过固定中文集 CER 和上述停止检查；新版描述缓存的复测见本轮进展。
性能及 30 分钟连续运行尚未通过验收，目标保持进行中。
