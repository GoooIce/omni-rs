# Burn ANE 后端调研与实现路线

## 2026-10-10 补充：IR 遍历减少分配与 KV 分档复测

`ir::inputs/outputs` 对常用算子使用借用切片和最多 5 个栈上引用，其他算子保存上游 opaque iterator，
不会增加第二层 Box。可选输入按原顺序压紧，变长输入/多输出借用原切片；未覆盖的多输出操作仍由上游完整返回。
编译规划、分档、MIL 降低和 Runner 回退统一使用该入口，数学表达、图键和原生 ABI 不变。
上游 API 对照测试覆盖可选 bias/mask、重复引用、变长 Cat、Drop、Custom、多输出 reduction；
73 项后端测试、两组 Clippy 和模型 20 份数组位对照通过。原生桥接未改，沿用原 192 组缓冲验证。

row64 / KV256 的三轮短测为 **RTF 0.8366 / p95 183.76 ms**，同期二进制键控制为 **0.8470 / 185.49 ms**。
图规划平均 227.10 → 189.56 ms（降低 16.53%），总耗时改善 1.23%，p95 改善 0.94%。这仍是小幅整体收益。
同一候选 KV128 三轮为 **0.8166 / 179.32 ms**，121 次重载，20 份数组与 KV256 逐字节相同；
KV64 单轮降为 **0.8663 / 181.80 ms**，214 次重载抵消了部分计算节省。

40 秒连续输入中，KV128 为 **1.0671 / 308.48 ms**，积压 3.22 秒；KV256 为 **1.0509 / 309.53 ms**，积压 2.90 秒。
两轮均无错误/淘汰/重载失败，停止响应 9.29 / 33.13 ms，原预算满足，共同提交步骤正文相同。
KV128 有 6 张热态新图、1592 次重载和 379.09 ms 系统编译，KV256 为 3 / 1522 / 0 ms；全部耗时保留。
处理音频跨度不同，各仅一轮，不能将短测收益外推到长流，也不能用累计阶段数据当作等量工作提速。
当前完整集验收仍选 KV256。一轮完整 96 条集为 **RTF 0.8126 / p95 187.45 ms / CER 3.1052%**，
相对 Metal 的 CER 增加 0.0706 个百分点，96 条原始转写与共享权重版相同。
与归档共享权重版相比，总耗时下降 0.97%、p95 下降 1.15%；这是二进制图键与栈上遍历的组合结果，控制非同期测量。
规划 19.79 → 14.21 秒，evaluate 297.36 → 297.39 秒，加载路径 25.71 → 29.04 秒。
均无热态新图/淘汰，均 8874 次重载且无失败；52 / 96 条 RTF、10 / 96 条 p95 达到门槛，性能未通过。
三轮完整重复与 30 分钟验收尚未完成。[完整集对照](../artifacts/burn-ane-goal/inline-ir-row64-corpus-comparison.json)。

后续应同时检查程序驻留和较慢步骤的图执行：KV128 减少补齐计算，但本轮加载路径升至 5.72 秒，
KV256 为 4.79 秒；保留大权重程序的淘汰策略在更多形状下可能需要调整。
后续源码已加入可切换的同权重程序优先卸载与纯 LRU 对照，默认仍为原策略；74 项后端测试和两组 Clippy 通过。
同一二进制各一轮 40 秒连续探针：原策略 1.0509 / 303.63 ms，同权重优先 1.0333 / 306.83 ms，
LRU 1.0112 / 251.05 ms。LRU 连续 p95 降低 17.32%，但 6.74 秒短测更慢，尚无完整集或 30 分钟验收。
LRU / KV128 为 1.0236 / 274.90 ms，比 LRU / KV256 更慢，未采用。共同音频段正文一致、预算和直接 ANE 检查通过。
详见[策略筛选证据](ane-optimization-goal.md#程序槽替换策略可选候选默认未改)。
源码、二进制和完整记录保存在 `artifacts/burn-ane-goal/inline-ir-*`；[详细验收](ane-optimization-goal.md)。

复现选定配置的短测、连续输入和完整集：

```sh
python3 scripts/evaluate_streaming.py --backend ane \
  --binary artifacts/burn-ane-goal/inline-ir-bin/r2t2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav \
  --repeats 3 --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --output artifacts/inline-ir-recheck

artifacts/burn-ane-goal/inline-ir-bin/stream_soak --backend ane \
  --manifest docs/eval/mandarin-v1.json --seconds 40 --warmup-seconds 32 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token --stop-after-ms 40000

python3 scripts/evaluate_streaming.py --backend ane \
  --batch-binary artifacts/burn-ane-goal/inline-ir-bin/stream_eval \
  --manifest docs/eval/mandarin-v1.json --repeats 1 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --compare artifacts/burn-ane-goal/metal-corpus-v1-r1/summary.json \
  --output artifacts/inline-ir-corpus-recheck
```

## 2026-10-10 补充：减少图规划中的文本处理

在共享权重实现之上，内存缓存键从 JSON 改为 bincode 2.0.1（使用已有锁文件中的版本），编码的身份字段不变。
编号随 `tNNNNNN` 名称一起生成，排序借用名称；模型持久缓存仍按生成内容散列。
Burn 0.21 的相关 IR 使用带判别值的派生序列化，保留算子、类型、形状、标量、常量身份和输出选择；
非有限标量继续走既有回退，不借键格式变化扩大算子数值支持范围。

71 项后端测试与两组 Clippy 通过，原生桥接未变。row64 / KV256 / 单 token 融合下，模型 20 份数组与共享权重版逐字节相同。
三轮短测对照：候选 0.8635 / 186.83 ms，同期控制 0.8711 / 190.76 ms；图规划平均从 283.70 降至 231.07 ms，
总耗时下降 0.88%。旧控制三轮曾达到 0.8556 / 188.02 ms，仍需区分可重复的规划阶段改善和整个系统的波动。
同期 40 秒持续输入的 RTF 为 1.1000 / 1.1012，p95 为 323.74 / 322.77 ms，积压均为 4.18 秒，整体接近。
图规划降低 20.46%；候选同时发生 281.02 ms 系统编译，控制命中系统缓存，该时间完整计入结果。
提交事件正文逐条相同，均无推理错误、淘汰和重载失败；完整 CER 与 30 分钟验收尚未进行。
源码、二进制、原始事件及对照保存在 `artifacts/burn-ane-goal/binary-plan-*`；[详细记录](ane-optimization-goal.md)。

本地 Burn 0.21 `operation.rs` 的 `OperationIr::inputs/outputs` 虽返回 `impl Iterator`，底层各类算子实际创建
`Box<dyn Iterator>`；规划中的存活分析、名字生成与形状分析多次调用它们。
上节候选已用栈上的固定输入引用和变长切片迭代器减少分配，将未覆盖算子委托回上游接口，
并验证可选 bias/mask、输入顺序、Drop、变长拼接和多输出操作的等价性；收益按上节独立测量记录。

复现本轮短测：

```sh
python3 scripts/evaluate_streaming.py --backend ane \
  --binary artifacts/burn-ane-goal/binary-plan-bin/r2t2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav \
  --repeats 3 --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --output artifacts/binary-plan-recheck
```

## 2026-10-10 补充：不同形状的描述共享不可变权重

此前 row64 在 40 秒连续输入中发生 407 次图新建、451 次缓存淘汰，积压 21.62 秒。
多个 row/KV 形状包含相同的静态权重，却各自创建一份 NSData，序列化源数据因此超出 16 GiB 预算。
本轮保留原缓存上限，先消除这些相同内容的分配。

Rust 权重池按完整打包 blob 的 SHA256 匹配，包括头、布局与偏移；弱引用索引不延长权重生命。
NSData 复制并拥有输入 bytes，每张图用 Arc 持有共享对象，原生描述收到同一个不可变 NSData。
源 Vec 可以在编译后销毁；模型实例卸载、重载和描述淘汰按原生命周期运行。
新权重、不同 blob 布局和内容变化分别创建对象；失败、未缓存的超大图和清空缓存均释放对应所有权。

图缓存的计费改为各图 MIL 文本之和加唯一存活权重数据，16 GiB 图预算、4 GiB 打包预算、
112 个加载程序和 1024 个描述的限制不变。`cached_model_bytes` 报告此计费量，
`unshared_cached_model_bytes` 保留各图独立大小之和，`shared_model_weight_bytes` 与
`unshared_model_weight_bytes` 单列共享前后的权重部分，另记录共享命中、未命中和存活 blob 数。
这些字段不计算系统运行时副本和 IOSurface；预算不是进程 RSS 上限。
编译时保留候选权重的所有权并按所需元数据预留空间；命中的图已在相同不可变预算下获准驻留，
重新入缓存时无需遍历全部描述计费。原独立大小仍用于小图优先卸载策略。

71 项后端测试通过，包括跨形状/换权重/重载、64 KiB 压力下释放全部共享持有者，以及超预算执行后的释放。
192 组原生缓冲检查、后端 Clippy、使用原四项豁免的根 Clippy 通过。
encoder、prefill、后续 8 token 的 20 份数组与 host-layout 逐字节相同，原 5% 数值门槛未变。
源码、二进制与证据均保存在 `artifacts/burn-ane-goal/shared-descriptor-*`。

row64 的三轮 6.74 秒短探针为 **0.8556 / 188.02 ms**；40 秒连续输入为 **1.0774 / 305.06 ms**，
积压 **3.54 秒**，停止响应 **22.44 ms**，RSS 峰值 **23.02 GB**。热态 3 次新图、0 次淘汰、
1506 次实例重载且无失败，缓存抖动已消除，实时性能仍未通过。
381 张驻留图独立计算需 22.94 GB，实际保留的共享源数据及元数据为 8.26 GB。
row32 虽然无重编译，程序重载却增加，单次短探针退步到 0.9975 / 301.87 ms；不采用该配置。
同一候选 row128 的 40 秒探针为 1.1344 / 302.64 ms，RSS 峰值 26.06 GB，旧 host-layout 为
1.1109 / 293.50 ms、33.73 GB。这些都是单次连续测量，不能把内存节省直接推断为算子提速。
完整 96 条集已完成一轮：**RTF 0.8206、p95 189.63 ms、CER 3.1052%**，96 条转写与 host-layout 逐条相同，
相对 Metal 的 CER 劣化仍为 +0.0706 个百分点。相比 host-layout / row128，耗时降低 5.86%，p95 增加 1.88%；
精度和直接 ANE 检查通过，性能未通过。热态 0 次新图/淘汰、8874 次实例重载且无失败。
direct evaluate 从 324.47 降至 297.36 秒，prediction 从 356.24 降至 327.43 秒；加载路径则从 19.92 增至 25.71 秒。
当前保留此实现，性能验收使用 row64 / KV256 / 单 token 融合；row64 的模型 20 份数组也通过原数值门槛并与 row128 对照逐字节相同。
三轮完整重复和 30 分钟验收尚未完成。详细结果见 [验收记录](ane-optimization-goal.md)。

此阶段之后验证了紧凑二进制缓存键，结果见上节；图规划与程序驻留开销仍值得继续检查。
另一方向是仅在原生表面布局和有效区段一致时直接绑定输入，需要独立证明 padding、不可变租用和旧 KV 的生命周期安全。
输入直接绑定尚未实现，不计入本轮收益。

复现 row64 短测与连续输入：

```sh
python3 scripts/evaluate_streaming.py --backend ane \
  --binary artifacts/burn-ane-goal/shared-descriptor-bin/r2t2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav \
  --repeats 3 --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --output artifacts/shared-descriptor-recheck

artifacts/burn-ane-goal/shared-descriptor-bin/stream_soak --backend ane \
  --manifest docs/eval/mandarin-v1.json --seconds 40 --warmup-seconds 32 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token --stop-after-ms 40000

python3 scripts/evaluate_streaming.py --backend ane \
  --batch-binary artifacts/burn-ane-goal/shared-descriptor-bin/stream_eval \
  --manifest docs/eval/mandarin-v1.json --repeats 1 \
  --row-bucket 64 --kv-bucket 256 --fuse-single-token \
  --compare artifacts/burn-ane-goal/metal-corpus-v1-r1/summary.json \
  --output artifacts/shared-descriptor-corpus-recheck
```

## 2026-10-10 补充：整块注意力算子的验证方向

Apple 的 [MIL iOS18 SDPA 定义](https://apple.github.io/coremltools/docs/_modules/coremltools/converters/mil/mil/ops/defs/iOS18/transformers.html)
提供 `scaled_dot_product_attention(query, key, value, attn_mask)`，支持 FP16/FP32 数据以及布尔/浮点遮罩。
其布尔 true 表示参与注意力，与 Burn 的“true 排除”相反；该算子固定使用 head dimension 的倒数平方根缩放，
没有直接的 causal/softcap 参数。因果约束需要合并进遮罩；自定义缩放、softcap 和其他未覆盖情况保留现有分解实现。
API 定义不证明私有直接 ANE 编译器支持它，更不证明更快，必须用本机严格硬件测试和真实模型对照验证。

本机实验中，直接布尔遮罩编译失败，转换成 FP16 加性遮罩（参与位置为 0，排除位置为负无穷）后，
矩形注意力、广播、不同 value 宽度与全遮罩行通过。全遮罩行在输出处显式选择零，避免把 NaN 乘零当成清零。
验证长度包括 Q=1/7/32、K=5/32/128/256/257；单查询用例同时检查实际有效 key 的输出。
这不意味着所有图布局可用：合并多个动态遮罩仍编译失败，KV 追加与 GQA 的一个既有回归也失败。
将 mask 求和、逐次 select 或沿 head 展开均未解决前一种失败，不能把单个成功形状推广为完整支持。

保存的 SDPA 候选实现只在默认缩放、无 softcap/额外 bias、匹配 batch/head、至多一个遮罩时尝试 SDPA。
其余情况保留现有直接 ANE 分解表达。系统若拒绝含 SDPA 的候选图，重新生成分解表达并编译一次，
缓存最终成功的 ANE 图；资源耗尽仍走原程序槽恢复逻辑。统计单列 `direct_sdpa_compile_retries`
和 `direct_sdpa_graphs_compiled`，重试耗时计入编译加载，不隐藏为 GPU 回退。
完整后端 58 项测试通过（5 单元、27 通用/Core ML、26 direct），后端 Clippy 通过。
完整 encoder、prefill 和后续 8 token 均通过原 5% 数值门槛；20 份 ANE/Metal 数组中 19 份与保留版逐字节相同，
仅 encoder 的 ANE 数组改变，相对 Metal RMSE 从 1.23895% 变为 1.25084%。
真实模型只成功编译 1 种 SDPA 图，另有 26 次分解表达重试；decoder 的 9 份 ANE 数组均未改变。

同一 6.74 秒音频、row128 / KV256 / 单 token 融合，三次独立进程完整预热后的汇总
RTF 为 **1.0072**、p95 **208.05 ms**，正文均一致、直接 ANE 检查通过。
编译描述缓存保留版单次为 1.0058 / 207.47 ms；重复次数不同，不能把小幅差别解释为确定退步，
但未观察到可确认收益。因此主线恢复原有分解注意力实现，SDPA 代码、二进制和失败日志留作研究对照。
未对该实验做完整集 CER 或 30 分钟验收，也不把它计为已采用的性能优化。

保存的候选源码在 `artifacts/burn-ane-goal/native-sdpa-source/`；最初的
`native-sdpa-experiment.patch` 仅保留失败实验，不是当前候选补丁。
证据：[布尔遮罩失败](../artifacts/burn-ane-goal/native-sdpa-bool-shape-tests.log)、
[单因果遮罩通过](../artifacts/burn-ane-goal/native-sdpa-causal-only-test.log)、
[KV 布局失败](../artifacts/burn-ane-goal/native-sdpa-kv-position-test.log)、
[候选完整测试](../artifacts/burn-ane-goal/native-sdpa-final-tests.log)、
[模型对照](../artifacts/burn-ane-goal/native-sdpa-parity-byte-comparison.json)、
[三次流式测量](../artifacts/burn-ane-goal/native-sdpa-timing/summary.json)。
保存的源码与二进制一致；随后 Clippy 仅去掉一个多余的 `&mask` 借用，未改变计算。

后续的固定性能、CER、长流与数值诊断验收见 [实时优化目标](ane-optimization-goal.md)。

## 2026-10-10 补充：传输合并后的后续方向

已把逻辑布局转换与 IOSurface 复制合并，去掉两个每调用临时缓冲，完整模型 20 份数组逐字节不变。
6.74 秒音频三轮热态为 RTF 0.9662 / p95 199.27 ms；这仍只是短探针。
更大融合图在同一输入上减少调用，但 evaluate 时间基本不变、I/O 量增加，未采用为默认。
测量、参数和失败方向均保留在 [验收文档](ane-optimization-goal.md)。

本轮已实现原生输出视图，短探针、连续输入和完整语料均有测量。FP16 输出保留 IOSurface 所有权，图之间按有效区段复制，
只有公开读取、Burn 回退、常量读取或设备转移时才物化 TensorData。其他 dtype 仍在图边界执行原有转换，
包括 BF16 舍入与整数/布尔转换；Core ML 路径保持主机存储。

实现约束与验证：

- 视图保存原始形状及补齐区段。比如旧 KV 的 prefix 位于 256 容量内，新 token 位于物理位置 256；
  下一轮复制时把新 token 放到实际有效 prefix 末尾，清零目标空洞。不同补齐轴和系统报告的物理 reshape 均按实际步长处理。
- 输出使用引用计数租用。缓存图只有在没有活跃张量引用某缓冲时才能重新写入；否则选择另一个空闲缓冲或新建一个。
  每个输出槽除活跃数据外只保留一个额外空闲缓冲。改变输出绑定时重建 ANE request，重建时间计入预测。
- 张量仅持有 surface，不持有模型、编译描述或图缓存。卸载程序或淘汰整张图后，尚未消费的分支和 KV 仍能正确读取。
- 输入先按视图复制到图的原输入缓冲，没有直接别名绑定，也没有绕过 padding 语义；这一步不能称为零复制。
- 增加原生输入/输出、物化、缓冲分配/复用及存活张量字节统计。原生存活字节是补齐后的 FP16 payload，
  不包括 IOSurface 对齐与图池中空闲缓冲，不能代替进程 RSS。图和权重缓存预算、112 个程序槽不变。

61 项后端测试通过（5 单元、27 通用/Core ML、29 direct）。新增三项硬件回归验证变输入、多输出、
旧分支在图复用/淘汰后读取、两个缓冲稳定循环复用，以及 GQA KV 追加、回滚、跨容量和程序重载后的旧值。
独立 native 测试通过原 27 组逻辑读写及 165 组 surface 复制：使用含 NaN/负零的位模式，给源 padding 填入毒值，
交叉验证不同补齐轴、奇数步长、物理 reshape 和目标清零。后端 Clippy 通过；根 Clippy 使用原四项豁免。
证据：[完整后端测试](../artifacts/burn-ane-goal/native-resident-backend-tests.log)、
[原生传输测试](../artifacts/burn-ane-goal/native-resident-buffer-tests.log)。

新增测试中，一个 batch=2 的注意力加投影图在系统 ANE 编译阶段被拒绝（尚未执行传输）；失败源码和日志保存为
`native-resident-batch2-lifetime-tests.*`。最终 KV 硬件回归采用既有可编译的 batch=1 GQA 形状；
多 batch 的原生视图复制由独立测试覆盖。未据此扩大直接 ANE 的图支持范围。
发布版已通过完整 encoder、prefill、后续 8 token 的原 5% 数值门槛；**20 / 20 份数组与传输合并控制版逐字节相同**，
推理无新增回退。[逐字节对照](../artifacts/burn-ane-goal/native-resident-parity-byte-comparison.json)。
同一 6.74 秒 WAV 三轮热态为 **0.9290 / 191.83 ms**；40 秒连续输入积压从 7.22 降到 **5.14 秒**，
p95 **300.45 ms**，RSS 采样峰值从 22.11 增到 **27.03 GB**。仍未实时。
本版完整 96 条集一轮为 **RTF 0.9028 / p95 193.44 ms**，较传输合并版降低 4.28% / 4.19%，
转写逐条相同；CER 3.1052%，比 Metal 高 0.0706 个百分点，精度通过、性能未通过。
短探针每轮仍新建 2132 个原生输出缓冲，后续可针对跨层工作集和真实内存占用进一步检查池的保留策略。

## 2026-10-10 补充：主机索引与布局复制

在 Direct + 允许回退时，若 Select、Cat 或步长为 1 的 Slice 的输入均由当前 State 持有，
直接按 FP16/F32 元素位模式复制。已有原生视图先按需物化，不主动读取仅在回退设备上的张量来尝试此路径。
目标是让 embedding 查表和音频/文本 token 拼接减少 GPU 上传及随后回读，不改变矩阵计算或 dtype。
非支持的 dtype、索引和切片继续交给原后端；严格模式在这个分支前仍然拒绝回退。

这些操作继续计入 `fallback_operations` 和原有回退原因，并单列 `host_copy_operations`、
`host_copy_output_bytes`、`host_copy_ms`。主机复制不算 ANE 矩阵加速；细分时间包含在 `fallback_dispatch_ms` 中。
67 项后端测试通过（9 单元、27 通用/Core ML、31 direct），包括独立 NdArray 对照、FP16 特殊位模式、
重复索引/多维拼接/切片、严格模式，以及索引和复制后两张 ANE 图之间零回退设备上传的硬件回归。
后端 Clippy 与使用原四项豁免的根 Clippy 通过。

此阶段源码冻结于 `host-layout-source-sha256.json`，测试日志为 `host-layout-backend-tests.log`。
20 份完整模型数组与原生输出保留版逐字节相同，原数值门槛通过。
row128 / KV256 / 单 token 融合，短探针三次为 **RTF 0.9082、p95 188.22 ms**（上一版 0.9290 / 191.83 ms）。
首轮输入准备从 313.27 降至 83.21 ms，649 次主机复制耗时 6.53 ms。
40 秒连续输入处理到 35.68 秒，积压 **4.18 秒**、p95 **293.50 ms**、停止响应 **33.63 ms**，
无错误或 token 预算事件；采样 RSS 峰值升到 **33.73 GB**。完整 96 条集一轮为
**RTF 0.8716、p95 186.13 ms、CER 3.1052%**；96 条转写逐条不变，比 Metal 多 0.0706 个百分点，精度通过。
相对原生输出保留版，完整集 RTF 降低 3.46%、p95 降低 3.78%。输入准备从 22.77 降至 6.15 秒，
直接 evaluate 仍约 324.47 秒；尚未达到实时目标。三轮完整重复与 30 分钟验收尚未完成。
row64 单次短探针为 0.8596 / 186.93 ms，但连续输入发生 407 次新建图、451 次淘汰，积压 21.62 秒，
p95 2667.19 ms；没有采用。短音频中的行补齐收益不能代表长流中的图缓存代价。
证据见 [当前验收记录](ane-optimization-goal.md) 和 `artifacts/burn-ane-goal/host-layout-*`。

## 2026-10-10 实验：同输入的静态投影合并，未采用

归档实现提供 `AneOptions::parallel_projection_channels`：在每张已切分的图内，
对同输入、无 bias、权重不可变的二维 Linear/Matmul 按通道拼接打包权重，生成一个更宽的 1×1 卷积，
再按通道切回各个输出。FP16 权重逐字节拼接，沿用不可变权重缓存；图切分和缓存预算不变。
候选主要覆盖 Q/K/V、门控 FFN 的 gate/up，以及图内相邻的词表分片。输出通道和超过配置上限时分组，
bias、动态权重和其他维度保持原表达；不把已丢弃的分支加入计算。

归档候选二进制的 CLI 诊断参数是 `--ane-parallel-projection-channels`，examples/评测脚本使用
`--parallel-projection-channels`，默认均为 0。`direct_parallel_projection_groups_compiled` 和
`direct_parallel_projections_compiled` 只统计成功编译的合并组/原投影数，不将重载计作新编译。
71 项后端测试通过（9 单元、27 通用/Core ML、35 direct），新增硬件回归覆盖 Q/K/V、门控 FFN、
权重替换、转置视图、通道上限、缓存重载，以及丢弃分支/bias/动态权重的排除。后端 Clippy 和使用原四项豁免的根 Clippy 通过。
16,384 通道上限的完整模型对照 **20/20 数组逐字节相同**，没有新增编译失败或推理回退；
编译计数确认 299 个原投影合成 124 组宽卷积。
同一短探针三轮为 **RTF 0.9118、p95 188.26 ms**，未合并控制为 0.9082 / 188.22 ms，
没有确认收益。12,288 上限单轮为 0.9014 / 185.25 ms，4,096 上限单轮为 0.9053 / 186.13 ms，
均在目前的轮间波动范围附近。12,288 上限的 40 秒连续输入为 RTF **1.1170**、p95 **291.80 ms**、
积压 **4.34 秒**、停止响应 **67.65 ms**，无错误，未确认提速。采样 RSS 峰值 **23.27 GB** 低于控制版 33.73 GB，
但只有一轮连续测量，未验证长期内存改善。实现已归档，没有完整 CER 或 30 分钟结果，未并入当前源码。
证据：`parallel-projection-source-sha256.json`、`parallel-projection-binary-sha256.json`、
`parallel-projection-parity-byte-comparison.json` 和 `parallel-projection*-timing/summary.json`。

## 2026-10-10 实验：固定尺寸的独立行投影块，未采用

row64 在短音频中减少补齐计算，但长流增加了带权图的形状种类，超出固定描述缓存预算。
归档实现的 `projection_tile_rows` 在既有 token 轴证明上，只对没有注意力、没有 token 选择、
全部输出保留 token 轴且 I/O 秩不超过 4 的图，将物理行数限制为固定块尺寸；较长输入重复调用同一个程序。
图、权重和程序槽预算不变。注意力与最后一个 token 的选择仍按原来的完整图执行。

原生输入通过零有效长度的区段跳过块前后数据，保留原 surface 的容量和已有空洞；
另一轴存在补齐时按需物化。每次调用仍通过有引用计数保护的输出池，避免覆盖之前较小调用保存的结果。
多个块的输出目前在主机按原逻辑轴拼接，仍是 FP16 位复制；它会增加部分输出回读，并非已确认的收益。
`row_tiled_graph_executions` / `row_tile_calls` 分别记录拆分图和内部调用，直接 ANE 调用总数包含这些内部调用。

75 项后端测试通过（10 单元、27 通用/Core ML、38 direct），覆盖跨空洞/轴的位复制、块尾、
图程序复用、旧输出生命周期、从原生卷积输出的内轴切块，以及注意力/token 选择排除。
后端 Clippy 和使用原四项豁免的根 Clippy 通过。完整模型 20/20 数组逐字节相同，原 5% 数值门槛通过。
该模型探针 prefill 只有 8 个 token，验证的是缩小补齐后的程序；跨多个块由硬件回归及真实流式探针覆盖。
64 行块、row128 / KV256 / 单 token 融合、关闭通道合并，三轮短探针为 **RTF 0.9590、p95 214.71 ms**，
比控制版退步 **5.59% / 14.08%**，正文与直接 ANE 检查通过。每轮拆分 560 张逻辑图为 1120 次内部调用，
原生物化次数增到 4857，首轮 evaluate 4867.64 ms、输出整理 115.58 ms；额外调用和回读抵消了计算节省。
没有对这项退步配置继续做完整语料或连续输入验收。

实验源码/二进制指纹为 `row-tile-source-sha256.json` / `row-tile-binary-sha256.json`，
数值对照 `row-tile-parity-byte-comparison.json`，测量 `row-tile64-timing/summary.json`。
归档二进制的 CLI `--ane-projection-tile-rows`，examples/评测脚本 `--projection-tile-rows`；
验证共享权重前，源码和 release 二进制曾恢复为通过完整语料验证的 `host-layout`；两项实验参数不在当前源码中。

复现这两项实验需使用相应归档脚本和二进制；以下两条顺序运行：

```sh
python3 artifacts/burn-ane-goal/parallel-projection-source/scripts/evaluate_streaming.py \
  --backend ane --binary artifacts/burn-ane-goal/parallel-projection-bin/r2t2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav --repeats 3 \
  --row-bucket 128 --kv-bucket 256 --fuse-single-token \
  --parallel-projection-channels 16384 --output artifacts/parallel-projection-recheck

python3 artifacts/burn-ane-goal/row-tile-source/scripts/evaluate_streaming.py \
  --backend ane --binary artifacts/burn-ane-goal/row-tile-bin/r2t2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav --repeats 3 \
  --row-bucket 128 --kv-bucket 256 --fuse-single-token \
  --projection-tile-rows 64 --output artifacts/row-tile-recheck
```

调研日期：2026 年 10 月 9 日。目标设备：Apple M5 Max，128 GB RAM，macOS 27.0.1。

2026 年 10 月 10 日继续实现了不可变权重复用、KV 容量分档和 FP16 图间传输。实现细节见文末，
最新测量在 `artifacts/burn-ane-fp16/`，前两轮在 `artifacts/burn-ane-kv/` 和 `artifacts/burn-ane-direct/`。

Confucius4-R2T2 的神经网络主体有进一步完整部署到 ANE 的实现路线。当前通用 Burn 后端的瓶颈包含计算图切分、设备回退、缓冲区搬运和设备特化，并不能据此认定 ANE 硬件不适合这个模型。后续仍以 Burn 的通用算子接口为入口，优先优化本机 M5；其他平台继续使用现有 Metal 或 WebGPU 路径。

“模型主体全 ANE”指 Transformer 等数值计算在 ANE 上执行。麦克风采集、流式调度、tokenizer、状态管理和界面仍需要主机程序。计算计划包含 ANE、无 Burn 回退、全部数值计算在 ANE，是不同的验证标准。

## 当前实现与基线

调研开始时，`crates/burn-ane` 捕获 Burn `OperationIr`，由 Rust 生成 Core ML NeuralNetwork protobuf，再通过 Swift 的公开 Core ML 接口执行。它不是 Burn 已发布的官方后端，也不依赖模型专用的 Python 导出器。[Burn 上游后端列表](https://github.com/tracel-ai/burn#supported-backends)尚未列出 ANE。本轮新增的直接 ANE 路径见文末。

最终已有基准见 [验证记录](../VALIDATION.md) 和 `artifacts/burn-ane-optimized/benchmark-summary.json`：

| 场景 | 结果 | 解释 |
| --- | --- | --- |
| 融合 FFN，32 行，1024 → 4096 → 1024 | 热态中位数 0.770 ms | 包含输出回读；Core ML 图内仍有 CPU 操作 |
| 相同 FFN，人为限制每图 8 个操作 | 热态中位数 4.396 ms | 融合约快 5.71 倍，不能推广为完整 ASR 提速 |
| 完整 6.74 秒音频，ANE 混合路径 | 14.420 s，RTF 2.139 | 双方都使用 FP16，完整音频预热后计时 |
| 同一音频，Burn Metal | 2.615 s，RTF 0.388 | 转写与 ANE 混合路径一致 |
| Encoder 的 7 个图全部命中磁盘缓存 | 计划检查与加载仍约 23.1 s | 磁盘命中不等于命中系统设备特化缓存 |

ANE 混合路径耗时约为 Metal 的 5.51 倍，包含编排、同步和回退，不能解释为芯片算力差距。独立 Attention block 曾被 Core ML 全部分配到 CPU；真实 encoder 的较大图又能将大多数矩阵运算分配到 ANE，说明形状与图表达会影响调度。

历史模型专用的 MLProgram encoder 实验中，Core ML 计划将 2336 个算子全部分配到 ANE；这是优化布局和图格式的参考，不是当前通用后端的结果，也不能替代硬件执行追踪。

## 三条可用路线

| 路线 | 已有能力与证据 | 对项目的适用性 |
| --- | --- | --- |
| Core ML MLProgram 与 MIL | 公开格式；支持 stateful 模型；已有纯 Rust IR 编译器 | 升级当前 NeuralNetwork 格式的可行参考，仍由 Core ML 调度设备 |
| 直接 ANE 运行时 | 私有 `_ANEInMemoryModel` 等接口编译 MIL，IOSurface 传递数据；已有 Rust 多层 Qwen3 实现 | 最值得验证模型主体完整使用 ANE，需要处理系统版本兼容和编译限制 |
| Core AI | macOS 27 新公开框架；提供状态、缓存、异步执行与内存布局接口 | 可对照验证新运行时，不能直接当作 ANE-only 开关 |

### 公开 Core ML 与纯 Rust MIL

[RLX 的 CoreML 后端](https://github.com/MIT-RLX/rlx/tree/b76d43f835f98cee019795e176ac80c2cadd5699/crates/backends/rlx-coreml)已实现纯 Rust 的 IR → MIL MLProgram → `.mlpackage`，仅执行层需要 Apple 框架。其 MIL 编译、权重 blob 和包序列化可作为通用 Burn 编译器的设计参考。RLX 的 `Device::Ane` 仍通过 Core ML，不能由设备名称推断所有运算都运行在 ANE。

Apple 从 iOS 18 和 macOS 15 开始为 **MLProgram** 提供 stateful 模型：状态可以在预测间保留并在图内更新，不必作为普通输入输出反复复制。此能力适合 KV 缓存，但单独使用 state 不保证 ANE 分配。[Apple Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)

### 直接 ANE 与 Rust 实现

[maderix/ANE](https://github.com/maderix/ANE)提供私有接口、MIL 编译和 IOSurface 的研究基础。项目明确定位为研究原型，其训练示例仍包含 CPU 运算，不能仅根据“pure ANE”描述认定整个流程都在 ANE。

[PMetal 的多层内核](https://github.com/Epistates/pmetal/blob/795f360b4c35327987d4ad39f4182862ff80b51a/crates/pmetal-metal/src/ane/extend.rs)更接近本项目的 Qwen3 decoder：将 RMSNorm、Q/K 归一化、RoPE、GQA、SiLU FFN 和残差编入多层 MIL 程序，KV 保存在 IOSurface 中。旧缓存与当前新增 token 分开计算 attention，再共同归一化，避免为追加少量 token 复制整个 KV。其主机程序仍更新 KV 增量并管理缓冲区。

PMetal 提供真实 Qwen3 检查点的硬件测试，包括 FP16 greedy 对照和有损 INT8 对照；这些是上游测试代码，不是本项目已经复现的结果。[文本生成实现](https://github.com/Epistates/pmetal/blob/795f360b4c35327987d4ad39f4182862ff80b51a/crates/pmetal-metal/src/ane/lm.rs)、[硬件测试](https://github.com/Epistates/pmetal/blob/795f360b4c35327987d4ad39f4182862ff80b51a/crates/pmetal-metal/tests/ane_lm.rs)

私有接口需隔离在后端内部。PMetal 已记录 macOS 27 的请求参数变化，并按版本处理，说明不能直接假设较旧的 M4 研究桥接代码在本机可用。[PMetal 运行时](https://github.com/Epistates/pmetal/blob/795f360b4c35327987d4ad39f4182862ff80b51a/crates/pmetal-metal/src/ane/runtime.rs)

其他参考包括 [Rustane](https://github.com/ncdrone/rustane) 与 [ane-infer](https://github.com/thebasedcapital/ane-infer)。它们包含 ANE 与 Metal 或 CPU 的混合路径，不能把其整体速度或训练吞吐作为本项目完整 ANE 推理的预期速度。

### macOS 27 的 Core AI

[Core AI](https://developer.apple.com/documentation/coreai)提供 `AIModel`、`InferenceFunction`、`NDArray`、`ComputeStream` 和 `AIModelCache`。其显式缓存、预分配输出、状态和异步流水有助于控制启动与推理开销。[Apple WWDC26 介绍](https://developer.apple.com/videos/play/wwdc2026/324/)

本机已运行接口检查：`SpecializationOptions(preferredComputeUnitKind: .neuralEngine)` 的 `allowedComputeUnitKinds` 仍包含 CPU、GPU、ANE；`availableKinds` 也包含三者。官方说明明确允许不兼容运算回退，因此此设置是优先 ANE，并不强制全 ANE。[设备偏好接口](https://developer.apple.com/documentation/coreai/specializationoptions/init(preferredcomputeunitkind:))

Core AI 还能显式查询和准备设备特化缓存；`expectFrequentReshapes` 可减少不断变化的输入形状触发的特化工作。这些收益需对本模型测量。[特化与缓存](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching)

官方建图教程使用 Python 创建图并保存 `.aimodel`，部分建图 API 仍在调整。Rust 直接生成这一新格式需要额外适配，现阶段不能将其当成当前 Core ML 模型的即插即用替换。[Core AI 建图教程](https://apple.github.io/coreai-torch/main/coreai-core/tutorials/construct-a-graph.html)

## 直接对应当前瓶颈的优化

### 稳定模型路径与特化缓存

Apple 明确说明：Core ML 的设备特化缓存与 `.mlmodelc` 的完整文件系统路径关联。即使模型内容相同，换到新的临时路径仍可能重新特化。[Apple 模型加载与缓存](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html#why-use-a-compiled-model)

调研时的 `native/Graph.swift` 会在命中磁盘缓存后，复制到 UUID 命名的临时目录，再执行计划检查和加载。这是“缓存命中但加载仍慢”的明确嫌疑。改进应保留稳定内容寻址路径，以锁或存活引用保护活动模型，避免 LRU 清理正在使用的包；分别计时编译、计划检查和加载，再用独立进程验证。

### ANE 布局与大图融合

Apple 建议 Transformer 采用 `[B,C,1,S]` 布局、将 Linear 表达成 1×1 卷积，并减少转置和 reshape 的内存搬运。短序列可能受权重读取带宽限制，因此单纯增加算术吞吐不一定改善单 token 延迟。[Apple Transformer ANE 优化](https://machinelearning.apple.com/research/neural-engine-transformers)

继续将优化落实到 Burn IR 编译层：保留中间张量的物理布局，融合投影、归一化、RoPE、Attention、FFN，减少图边界。MIL 只是更合适的表达层，不能保证任意大图、任意形状都能被 ANE 编译器接受。

### RMSNorm 与数值稳定性

两种可对照的表达方式：

- [ANEMLL](https://github.com/Anemll/Anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/models/llama_model.py#L94)构造 `concat(x, -x)`，使均值为零，再执行 LayerNorm 并截取前半段，得到 RMSNorm；代价是更大的中间张量。
- PMetal 先按最大幅值缩放，再计算平方和，并同步缩放 epsilon，避免 FP16 溢出；实际编译与误差仍需覆盖小方差、大幅值和真实模型输入。

当前后端已修复过 FP16 小方差归一化与展开 Sigmoid 的数值问题。新路径必须保留这些压力验证，不能仅用随机均匀输入通过就判断完整模型正确。

### KV 驻留与输出传输

KV 应按会话持久保存，仅写入新增位置；图间复用激活缓冲区。固定形状与容量分桶用于减少编译次数，不应变成麦克风录音时长限制。长流须保留现有上下文更新与手动停止语义，取消后正确重置状态。

在 greedy 路径中可将词表分块归约和 argmax 放入图内，只回传少量候选，而不是完整 logits。ANEMLL 已有图内 argmax 的实现说明；具体索引精度、并列值行为和 ANE 分配需要验证。[ANEMLL](https://github.com/Anemll/Anemll)

### 量化

先完成 FP16 正确性与图融合，再评估按输出通道的 INT8 权重。权重量化减少权重体积，不等价于激活也使用 INT8，不能直接套用其他项目的 W8A8 卷积吞吐倍率。完整 ASR 需用相同音频、分块、前瞻和预热条件比较延迟，并增加真实语料的 CER 或 WER 评估。

## 实施顺序与验收

1. 修复稳定路径缓存，记录独立进程的冷态与缓存加载时间。
2. 保留 Burn `OperationIr` 入口，增加 MIL 生成与直接 ANE 运行时，先验证常见算子及数值压力输入。
3. 使用真实权重验证完整 decoder 层，包含已有 KV、RoPE、GQA、RMSNorm 与 FFN；再扩展到多层融合和 encoder。
4. 优化 KV 与激活驻留、图内输出归约，然后评估 INT8 权重。

验收分开记录：数值误差与 greedy token、Burn 回退、Core ML 内部 CPU 分配、直接 ANE 执行、冷启动、热态 RTF、每步 p95、传输字节和长流取消。已有 Core ML `--ane-strict` 只禁止 Burn 回退，不能作为全部数值计算都在 ANE 的证明。

代码实现与后续实测更新到 [验证记录](../VALIDATION.md)。本文中的外部项目吞吐、源码路径与候选优化不构成本模型的性能承诺。

## 本轮落地

已实现稳定路径与活动模型锁保护，增加分阶段计时；独立进程的微型 FFN 复测未显示加载时间明显改善，主要开销仍在计划检查。不能把这一修复记为已证实的性能提升。

已增加可选的通用 Burn IR → MIL → 直接 ANE 路径，使用一个打包权重文件和按真实步幅读写的 IOSurface，并完成短序列、归一化、Attention、RoPE/GQA/KV 的硬件对照。实现入口是 `AneRuntime::Direct` / `--ane-runtime direct`，不是模型专用导出器。

本机测试没有支持直接照搬所有上游数值表达：按最大幅值缩放的 RMSNorm 和 MIL L2Normalize 在小方差输入上出现明显误差。当前使用对称拼接后的原生 LayerNorm，并保留小方差与大幅值压力测试。不同表达在本机的结果记录在验证文档，不能推广成其他设备或上游项目的结论。

后续自行推导了在输入中附加 `sqrt(C * epsilon)` 能量通道、原生 L2 后乘 `sqrt(C)` 的表达，
避开 epsilon 属性。48 组数值案例、75 项后端测试和真实模型 encoder / prefill / 8 token 检查通过，
但 6.74 秒转写出现遗漏，RTF 2.1043 / p95 422.00 ms，故全批次版本不采用。
小型数值探针不足以证明完整转写正确。后续小批次路径通过完整 96 条精度检查，CER 3.0346% 与 Metal 相同，
但 RTF 0.8139 / p95 186.85 ms 没有稳定改善，已归档撤回，工作树恢复原 LayerNorm。
详见[完整证据与限制](ane-optimization-goal.md#rmsnorm-附加能量通道)。

随后保留了模型层的稳定卷积块复用：缓存与完整重算的 9 个真实 encoder 阶段输出逐字节相同，
完整 96 条一轮为 RTF 0.7708 / p95 173.38 ms / CER 3.1052%，正文全部与 IR 遍历控制相同。
相对归档控制耗时降低 5.14%、p95 降低 7.50%；精度通过，但只有 78 条 RTF、28 条 p95 达标。
该改动减少前端重复计算，双向注意力与 decoder 稳定前缀语义不变。LRU 40 秒连续探针为
0.9951 / 268.78 ms、积压 1.46 秒。随后完成的 LRU 30 分钟测量为 **1.0228 / 290.78 ms**，
结束积压 42.74 秒、SIGINT 停止响应 105.32 ms，缓存预算和 final/delta 检查通过。
IOSurface 分配失败触发 1045 次相关回退，其中 1015 次命中失败缓存；性能与直接 ANE 验收未通过。
源码已确认临时分配失败被错误缓存，分配失败的根因仍待定位。
有界资源恢复修复现已通过 79 项后端测试、两组 Clippy 和真实模型 20 份数组位对照。
30 分钟复测中 76 次资源恢复全部成功，无矩阵回退、编译失败或失败缓存命中；修复已保留。
复测为 **RTF 1.0174 / p95 293.70 ms**、积压 33.07 秒、停止响应 63.97 ms，性能仍未达标。
与旧长测共同的 10,995 个非 final 步骤正文全部一致，但没有新构建的完整集 CER 结论。
[资源恢复复测](../artifacts/burn-ane-goal/surface-recovery-lru-soak30-summary.json)。
此结果来自录制 PCM 与软件停止，[完整长测审计](../artifacts/burn-ane-goal/convolution-prefix-lru-soak30-summary.json)。
源码、对照与参数范围见[稳定卷积块验收](ane-optimization-goal.md#稳定卷积块复用当前完整集保留版)。

2026-10-09 的真实完整检查点中，24 层 encoder、28 层 decoder prefill 和带 KV 单步均已通过直接 ANE 对照；这三个数值阶段没有新增 Burn 回退。当时的完整流式诊断仍受变化形状引发的 MIL/权重重建及模型加载拖累，尚未实时。后续改进与新结果见下文。

后续优先级调整为：批量 FP16 权重打包与不变权重复用 → KV 容量分档和跨图驻留 → 完整流式热态复测 → 图内 argmax → INT8。初始化的权重布局、embedding 索引和主机调度仍单独统计。不能把单次 decoder 的零回退推广成整个应用不需要 CPU/GPU，也不能把局部微型基准推广成实时 ASR。

### 2026-10-10：权重复用与 KV 分档

不变权重现在按原始张量身份和视图变换缓存转置后的 FP16 数据。换一组激活形状只需
重建图描述和组装权重 blob，命中时不再从 Metal 读取、折叠视图或重新转换权重。
该缓存有独立的字节预算和 LRU 淘汰，权重替换、重复创建转置视图、预算为零及缓存清理
均有实际执行对照。

KV 专门化在通用 IR 中追踪序列轴，通过静态容量消除逐 token 的形状变化。使用
`[历史有效值, 历史空槽, 当前 token]` 的物理布局，掩码按每次调用的真实逻辑位置生成；
保存到 Burn 的结果恢复成 `[历史有效值, 当前 token]`。因此固定容量不会把当前 token
错误地插入历史空槽，也不会把容量边界当作录音停止条件。GQA 的 reshape/repeat、
prefix slice、截断后追加、外部 mask、attention bias 与因果 mask 都保持有效长度语义。
无法证明安全的消费者保留原始图，不进行冒险的尺寸替换。

进一步的完整流式诊断发现：当前 token 数也随音频与文本前缀增长，仍会产生多个大权重图。
因此直接运行时默认在 KV 追加前、Attention 后切图，让带权重的投影/FFN 使用独立 token
分档，Attention 使用无权重的小图。RoPE 的通道切片与拼接保留序列轴；单 token 不补齐。
最后一个 hidden token 使用运行时选择掩码取得，避免补齐完整词表输出。硬件测试验证了
不同选择位置共用图，且未选行的 NaN 不污染结果。

这一步尚未实现跨图 IOSurface KV 驻留。容量扩档仍需要编译，图缓存不足仍可能淘汰；
本轮中间版本的完整音频诊断中，8 GiB 图缓存产生大量淘汰，故增加独立的图字节/数量配置和
初始化、预热、测量三个阶段的统计。不能仅凭单步热态宣称整段语音实时。

随后的实测还发现独立的程序资源限制：本机保留 127 个私有 ANE 程序后，再加载返回
Code 54（资源不足、可重试）。把字节缓存增至 32 GiB 无法解除这个限制；旧处理还会
把临时错误写进失败缓存，造成后续永久回退。现在同时限制已加载程序数，优先释放小图，
并在资源不足时释放旧程序后有限重试，不重新编译或缓存临时失败。140 个不同模型的
strict 硬件压力测试完成 13 次恢复，全部计算成功且无 Burn 回退。该上限只代表本机实测。

Prefill 在 GQA 扩展前切分后，可与带缓存的调用共享投影图；调大 token 分档能减少大图
数量，代价是额外的 padding 计算。M5 配置需要同时权衡该代价、字节预算与程序数量，
不能只一味扩大图缓存。下一阶段仍以 KV 驻留、减少主机搬运和图内输出归约为重点。

### 权重复用阶段的完整流式复测

完整 FP16 检查点、同一段 6.74 秒 WAV、160 ms 前瞻，完整音频预热一次后计时。
直接 ANE 使用 16 GiB 图预算、112 个已加载程序、128 token 当前行分档、32 token KV
分档和 4 GiB 打包权重预算。各次运行顺序执行，没有同时编译或跑其他基准。

| 分块 | 直接 ANE 耗时 / RTF | 同配置 Metal 耗时 / RTF | ANE 每步 p95 |
| --- | ---: | ---: | ---: |
| 160 ms | 10.829 s / 1.607 | 2.265 s / 0.336 | 328.7 ms |
| 320 ms | 5.247 s / 0.778 | 1.224 s / 0.182 | 312.3 ms |

两种分块的最终文本均为 `之前有顾客自己带酒水也没加收钱或者不让喝。`，与 Metal 一致，
delta 拼接等于最终文本，未触发 token 上限。320 ms 配置实现了这段样本的热态实时吞吐，
但 Metal 仍快约 4.3 倍；更大分块还会增加采集等待。权重加载和首轮编译/加载不在计时内，
不能推广成冷启动、长流或其他语料的性能与准确率保证。

320 ms 测量阶段发生 3350 次直接 ANE 调用，0 次编译失败、0 次资源不足重试、0 次
静态权重回读；小图仍有 32 次淘汰后加载，全部命中系统编译缓存，总加载路径约 161 ms。
还有 369 个 Burn 回退操作，包含 embedding 索引、argmax、输入拼接和非矩阵小图。
因此这里的“直接 ANE”是后端路径名称，不代表整个应用没有 CPU/GPU 运算。

160 ms 测量的图输入/输出合计约 **13.28 GB**（该版本接口为 F32 数据），仅预测调用约
6.89 s，主机输入准备与输出写回另占约 1.65 s，规划约 1.08 s。预测计时也包含原生桥接的
IOSurface 读写，并非纯 ANE 计算时间。相应的图加载约 0.69 s，不再是此前逐块十几秒的
大权重重建问题。下一步优先让主机/IOSurface 边界直接传 FP16，减少完整 KV 输入输出的
复制，并研究 KV 跨图驻留；随后处理图内 argmax，再评估 INT8。

完整模型数值复测覆盖 encoder、prefill 和连续两个带 KV token：相对 RMSE 分别为
1.239%、2.826%、3.311%、2.513%，decoder 最高分 token 均一致，四阶段无推理回退。
第二个缓存 token 用时 45.95 ms，62 次图命中、0 次新图加载和权重回读。额外更长的
任意 token 序列诊断曾越过 5% 误差门槛，保留为失败记录；未放宽阈值，未做 CER/WER 评测。
42 项后端测试、Clippy、fmt、原生桥接告警检查和 release 构建通过。

完整原始数据、阶段差值与源码指纹在 `artifacts/burn-ane-kv/summary.json`，
参数及命令见 [后端实测配置](../crates/burn-ane/README.md#m5-max-实测配置)，
详细对照与限制见 [验证记录](../VALIDATION.md#2026-10-10不变权重复用与-kv-容量分档)。

### FP16 图间传输与本轮结果

原实现每张图将 FP16 主机张量展开为 F32，在 IOSurface 输入处又转成 FP16；输出反向
重复这一流程。现在从补齐、KV 裁剪、布局转换到原生行复制都保留 FP16 数据，并采用
分块转置及单行快速路径。F32 公共张量和回退时按需转换，Core ML 路径保留 F32 接口。
这减少主机侧中间缓冲和复制，没有进一步降低 ANE 的计算精度。

保留并重新运行修改前的 release，以同一 FP16 检查点、6.74 秒 WAV、160 ms 前瞻比较。
每组独立进程完整预热一次，各次顺序执行。参数继续使用 16 GiB 图预算、112 个程序、
row bucket 128、KV bucket 32、4 GiB 打包权重缓存：

| 分块 | 修改前 | FP16 通路 | 耗时减少 | 新 RTF |
| --- | ---: | ---: | ---: | ---: |
| 160 ms | 10.501 s | **7.867 s** | 25.1% | 1.167 |
| 320 ms | 5.283 s | **4.100 s** | 22.4% | 0.608 |

除计时字段外，修改前后的全部流式事件一致。完整模型 encoder、prefill、两个带 KV
decoder 步骤的输出与上一轮保存的数组逐字节一致，没有新增推理回退。43 项后端测试、
Clippy、fmt、原生桥接告警检查和 release 构建通过。320 ms 仍只是该样本的热态实时吞吐；
160 ms 未达到实时，冷启动、长流与 CER/WER 尚未验证，已有更长序列误差失败记录未消除。

160 ms 的图接口输入/输出从 13.276 GB 降为 6.638 GB，主机输入准备和输出写回从
1569.0 ms 降为 649.4 ms。IOSurface 本来就使用 FP16，不能把接口字节变化当作芯片传输量
减半。新增分段计时中，原生输入复制、同步 evaluate 和输出复制为 166.8、4385.7、130.6 ms；
evaluate 包括驱动/调度，是主要剩余开销。图规划还有约 938 ms，热态重加载约 627 ms。

进一步试验 row bucket 64、24 GiB 图预算和 120 个程序，只把 evaluate 减至 4187.3 ms，
却使热态大图重加载增加到 275 次、加载路径 9815.1 ms，总耗时退步到 17.576 s。该配置
不采用；保留 row bucket 128 和 112 个程序。后续融合或更细分档应先控制已加载程序数量，
区分大算子计算和多次提交的成本，再验证 KV 增量传输、跨图驻留与输出归约。

结果、源码与二进制指纹、阶段计数、被拒绝的配置均保存在
`artifacts/burn-ane-fp16/summary.json`。可复现命令见
[后端配置](../crates/burn-ane/README.md#m5-max-实测配置)，详细限制见
[验证记录](../VALIDATION.md#2026-10-10fp16-图间传输)。
