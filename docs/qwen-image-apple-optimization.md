# Qwen-Image-2.1-Turbo：Metal / ANE 优化调研与生成参数

调研日期：2026-10-12。目标机器：M5 Max、40 核 GPU、128 GB 统一内存、macOS 27.0.1。
范围为 Makepad 本地版、Burn 0.21 / CubeCL 0.10 / CubeK 0.2、FP16，不进行量化。
本文把已经交付的参数支持、现有实现的观测、待验证的优化分开记录。没有把 Qwen 图像推理切到 ANE。

## 结论与实施顺序

**优先验证 M5 Metal TensorOps 大矩阵乘法，再改进融合注意力与条件缓存；ANE 先做整块基准。**
当前没有本模型在 ANE 上的端到端测量，不能断言 ANE 或 Metal+ANE 比 Metal 快。
Apple 的 Stable Diffusion 实现也按模型和硬件选择执行设备，并指出 `ALL` 不保证最快；
其 SDXL 的 Mac 路径推荐 GPU，不能拿手机 UNet 的数字预测本机 7B DiT 的收益。
来源：[Apple ml-stable-diffusion](https://github.com/apple-aiml-research/ml-stable-diffusion)。

| 顺序 | 具体工作 | 对本项目的意义 | 验收条件 |
|---|---|---|---|
| 0 | 固定 512² / 1024²、文生图 / 单图 / 双图、CFG 关闭 / 开启；记录冷启动、预热、峰值内存、CPU 时间 | 防止只优化一个算子或用编译耗时解释推理速度 | 同一请求、同一权重、同一初始噪声，每种预热后至少 3 次；端到端中位数与分阶段耗时 |
| 1 | Metal TensorOps FP16 GEMM；先覆盖 DiT 的 QKV、输出投影、gate/up/down | 32 层 × 8 步重复大矩阵计算，M5 专用硬件有明确 API 入口 | 相同矩阵形状对照现有内核；数值通过后才接整步，最终以完整生成时间判断 |
| 2 | 正确支持尾块、变长 K/V、因果偏移的融合 SDPA；完善调优键和候选适用性检查 | 消除当前安全 fallback 的大中间矩阵及部分重复 K/V | 同进程切换长度、CFG 两分支、多参考图、缓存/无缓存对照，不能只看图片像不像 |
| 3 | 有上限的跨请求文本 / 参考图特征缓存；缓存常量布局与 RoPE；减少频繁 CPU→GPU mask 构造 | 换种子、改输出尺寸、同图反复编辑时减少重复准备 | 模型、文件内容、精度、分辨率、提示词变化都正确失效；停止不留下半成品缓存 |
| 4 | Metal 复用编码状态和缓冲，合并短算子，逐项减少同步点 | 降低 CPU 提交、调度与临时张量成本 | 必须保留停止响应；去掉同步点后重跑融合索引回归 |
| 5 | ANE 编译完整 FFN / Transformer block，或单独的视觉塔 / VAE 候选 | 寻找低能耗或真实端到端提速的分区 | 包含布局、共享内存同步、编译和回退成本；同时报告速度与能耗，整块胜出才讨论接入 |

这份调研中的收益原先均为待测假设。后续已实现并验证第一项的 M5 TensorOps GEMM，
接入范围、测量和复现命令见 [TensorOps 实施记录](qwen-image-tensorops.md)。
随后先做了 [MLX 对照实验（已归档）](qwen-image-upstream-attention.md)，当前按用户要求在 [Rust/CubeCL 中实现注意力](qwen-image-rust-attention.md)。
跨请求条件缓存及 ANE 整块实验仍待验证。

## M5 Metal：最值得先验证的缺口

M5 每个 GPU shader core 内的 **Neural Accelerators** 与独立 **Apple Neural Engine** 是不同执行单元。
Apple 明确建议自定义 `simdgroup_matrix` 内核迁移到 TensorOps；演示通过 GPU trace 区分了普通矩阵指令
与 Neural Accelerator 执行。TensorOps 支持在 shader 内用已有 buffer、extent 和 stride 构造 tensor，
也能把激活函数保留在 cooperative tensor 中处理，避免写回后再读入。
这些能力可用于 Rust/Burn 自定义算子，无需把应用改写成 Swift 或把模型改成 Core ML。
来源：[M5 / A19 GPU 技术讲解及 transcript，13:41、15:11、17:17、19:29、28 分钟附近](https://developer.apple.com/videos/play/tech-talks/111432/)。

本地源码证据：

- `vendor/cubecl-wgpu/src/backend/metal.rs` 使用 `MetalArchitecture::Metal3` 注册矩阵能力。
- 已解析的 `cubecl-cpp-0.10.0/src/metal/dialect.rs` 生成 `metal_simdgroup_matrix`、
  `simdgroup_load`、`simdgroup_multiply_accumulate`，未生成 TensorOps。
- 调研时 `crates/qwen-image/src/ops.rs::Linear::forward` 调用通用 Burn linear；
  后续在图像 crate 中为 DiT 单独注册了 TensorOps Fusion Operation。

因此可确认通用 Burn Metal 内核当时尚未主动采用 TensorOps；**实际 Neural Accelerator 利用率仍须 GPU trace 测量**。
不能把 Apple 的演示加速倍数直接写成本项目预期加速倍数。

建议首个原型固定矩阵形状，不先修改整个 Burn 调度器：

| 算子 | 主要矩阵形状，T 为当前查询 token 数 |
|---|---|
| Q、K、V 投影 | 当前是三次 `[T,4096] × [4096,4096]`；合并才是 `[T,4096] × [4096,12288]` |
| attention 输出投影 | `[T,4096] × [4096,4096]` |
| FFN gate、up | 各 `[T,4096] × [4096,12288]` |
| FFN down | `[T,12288] × [12288,4096]` |

512²、1024² 输出分别有 1024、4096 个目标 latent token；首步还有文本与参考图前缀。
测试应包含转置权重、非连续 stride、尾块和 FP16 输入，保留必要的 FP32 累加。
融合 gate/up、SiLU、乘法以及输出残差时，必须对照原有 FP16 舍入位置。
Apple 建议按形状调 tile、控制寄存器压力、调整 K 维同步与 threadgroup 遍历顺序；仅替换 API 名称并不保证最佳吞吐。

Burn 接入可借鉴本仓库 `crates/breeze-tts/src/metal_depth.rs` 的自定义 Fusion Operation 注册方式，
在图像 crate 中试验，保持张量在同一设备与队列。TTS 的 GEMV 是小矩阵策略，不能把其内核直接用于这里的大 GEMM。
新 Metal 内核需有设备/系统能力检测，不支持时使用现有 Burn 路径，其他平台继续保留 WebGPU。

## 注意力：先修正确性边界，再扩大融合覆盖

现有实现已经启用 Burn fusion，且对齐的图像 attention 使用融合内核；并非所有 attention 都在走拆分实现。
但因果查询和未对齐 256 行的查询目前显式走 GPU matmul + softmax。这是此前真实崩溃及数值回归后保留的安全路径。
详细证据见 [接入记录](qwen-image-21.md) 和 `artifacts/qwen-image-edit/parity-report.json`。

主要改进点：

1. 调优缓存必须区分候选要求的整除条件、mask 类型和具体布局；命中缓存后也检查内核能否处理当前形状。
2. attention 内核应正确处理最后不足一 tile 的行与列，支持 `q_len != kv_len`、因果偏移和参考图的块状前缀。
3. 支持真正的 GQA，避免把 8 个 K/V heads 实体重复成 32 个 heads。
4. 在 tile 内做稳定 softmax，避免将完整 QKᵀ 矩阵写入统一内存。

Apple 的 MLX `fast.scaled_dot_product_attention` 提供 MHA / GQA / MQA、mask，并明确要求 GQA 的 K/V
不要提前铺成 Q 的 heads，softmax 使用 FP32。这适合作为语义和独立性能对照，不能未经测量就整体替换 Burn 后端。
来源：[MLX SDPA 文档](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)。

已否决的方案是“只给 Q 补齐到 256 行再裁剪”：此前 CPU 测试通过，但实际 GPU DiT 缓存对照误差明显扩大。
下一次验证必须覆盖 head_dim 128 / 72、长短文本、非正方形透明参考图、不同 CFG 提示词长度，以及先运行一种尺寸再切另一种尺寸。

## 缓存、统一内存与 CPU 提交

已有的有效利用：模型权重在 GPU 常驻；每次生成首步提取文本/参考图前缀 K/V，余下 7 步只更新目标图像；
潜变量与参考 latent 在 GPU 上连接，最终解码完成才读像素。本次 CFG 复用同一份视觉塔输出和 VAE 参考 latent。

仍存在的成本：

- 跨请求尚不复用文本或参考图编码。每次换种子仍重新运行文本编码器，同图编辑重新运行视觉塔和 VAE encoder。
- `ops.rs::attention` 在 CPU 构造因果布尔 mask，再上传；布局、位置等常量也有重复构造。
- `src/weights.rs` 对 mmap 权重逐张量转换为 TensorData，CubeCL 使用 `queue.write_buffer` 等上传路径。
  **物理统一内存不等于应用层零拷贝**，共享内存不能自动消除转换、临时副本、布局重排或同步。
- 文本 embedding 后的同步用于绕过既有融合索引错误；DiT 周期同步也承担进度与取消检查。移除前须验证用途。

跨请求缓存建议从最后一次条件开始，设置明确字节预算，而不是无上限缓存整个 DiT K/V。
FP16 DiT 的每个前缀 token 在 32 层保存 K、V，大约需要 `2 × 32 × 4096 × 2 = 524288` 字节；
一张 1024² 参考图约有 4096 个 DiT latent token，仅这部分缓存就约 2 GiB，CFG 两分支约翻倍。
视觉特征/文本特征缓存比保留所有前缀 K/V 更容易控制内存。此处为张量尺寸估算，不是峰值内存实测。

## ANE：按块适配，避免频繁往返

Apple 的 Transformer 优化建议采用 `[B,C,1,S]` 布局、用 1×1 卷积表达 linear、分块 attention，并减少 reshape/transpose
造成的实际拷贝；它强调跨执行引擎的传递成本。这些原则可借鉴，但文中 distilbert / iPhone 数字不代表当前 M5 DiT。
来源：[Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)。

对本项目的推论：

- 每个 linear 去 ANE、norm/SiLU/RoPE 回 Metal，会把同一 block 切成很多串行边界。应比较完整 FFN 或完整 block，
  把大部分中间结果留在同一执行单元，复用现有 FP16 共享张量桥接。
- 单张图的一步依赖前一步潜变量；Metal 和 ANE 不会因同时开启就自动并行处理相邻采样步。
  可以实验正向/负向分支并行，但两者会竞争统一内存带宽，仍需等待两边结果才能更新潜变量。
- 优先测固定形状 512² / 1024²，以及编辑的前缀长度。不能误称 ANE 完全不支持动态形状：Core ML 提供
  `EnumeratedShapes`、有限范围形状及 reshape frequency 提示；灵活形状与编译优化需要权衡。
  来源：[Core ML flexible inputs](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)。
- 若采用 Core ML 导出原型，使用 `MLComputePlan.deviceUsage` 检查算子预计落到哪个设备，再用实际 Instruments trace 核验。
  配置中允许 ANE 不能证明每层确实在 ANE 上运行。
  来源：[MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan)。

ANE 候选评估需同时记录首次编译、预热后单块、端到端延迟和能耗。短时推理更快与长时间更省电是不同目标。
VAE/视觉塔只执行一次，迁移它们即使成功，对大画布的 8 步 DiT 总耗时也可能影响有限。
现有 ASR/TTS 的 ANE / Metal 路由不随本次图像参数变更。

## 降低生成计算量的选项

Turbo 已经是 8 步蒸馏模型，配置中含专用 sigma 序列。官方明确说明只改 `num_inference_steps` 不覆盖它；
可以传入自定义 sigma 实验，但其它采样序列没有在此 checkpoint 上评估。因此没有增加会静默无效的步数滑块，
也没有给 4 步生成、采样器切换或通用 img2img strength 作保证。
来源：[Qwen Turbo 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)。

当前可以直接调整输出尺寸、参考图分辨率及是否启用 CFG。参考分辨率降低会同时减少视觉塔、VLM、首步 DiT 的参考 token 数，
也减少后续 attention 读取的前缀 K/V，但可能丢失编辑细节。参考分辨率不改变输出画布尺寸。

大画布出现内存峰值时，再研究 VAE 分块、重叠区域融合和分阶段释放。分块主要解决峰值内存，
不保证更快，也可能引入接缝或数值差异；batch slicing 对当前 batch=1 无直接收益。
当前 128 GB 机器优先让权重常驻，避免为了省内存每步反复装卸 DiT。先确认瓶颈再引入 offload。

TeaCache 根据 timestep 特征变化复用近似输出，属于有损推理策略；它与本模型已有的精确前缀 KV 缓存是不同机制。
其作者提供 FLUX 等模型的适配和迁移方式，但不能据此推断 Qwen-Image-2.1-Turbo 8 步一定受益。
如继续研究，需要重新标定阈值并统计文字、透明背景、多参考图一致性，优先级低于数值等价的矩阵/attention 优化。
来源：[TeaCache 作者仓库](https://github.com/ali-vilab/TeaCache)。

## 本次交付的参数

Makepad “图像生成 → 更多参数”新增：

| 参数 | 默认与行为 |
|---|---|
| 负向引导开关 | 默认关闭，保持 Turbo 官方 CFG=1 单分支路径；收起参数后仍显示开关状态 |
| 负向提示词 | 文生图与 1–4 图编辑共用；开启引导才生效。留空表示无条件分支 |
| 引导强度 | 开启时初始填写 2.0，作为试验起点，不是作者推荐值；接受大于 1、最多 20 的有限数值。20 是本地输入上限 |
| 参考图分辨率 | 默认“自动”，沿用输出面积并限制在约 288²–1024²；可填 288–1024 内 32 的倍数，仅编辑生效 |
| 尺寸预设 | 本地便捷预设：1024²、768×1024、1024×768；也可直接填写宽高 |
| 换种子 | 生成并显示新的 u64 种子，后续可以复制复现；不隐式改变每次点击的种子 |

独立 CLI 对应 `--negative-prompt`、`--true-cfg-scale`（别名 `--cfg-scale`），原有 `--reference-resolution` 保留。
仅提供负向提示词但 CFG=1 时，CLI 明确提示其不生效；CFG>1 但未提供负向提示词时，也按官方语义关闭 CFG 并提示。
使用无条件引导可传 `--negative-prompt '' --true-cfg-scale 2`。

实现跟随固定版本的 [官方 QwenImage21Pipeline](https://github.com/huggingface/diffusers/blob/883d51ef02cd7972eeaf158c652d229f3459f419/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)：
`negative + scale × (positive - negative)`，保留 FP16 三次运算边界；正负文本分别编码、各有独立 KV cache；
两分支都看到相同参考图。没有额外添加 CFG rescale、裁剪或自行修改采样序列。

开启 CFG 每步都多执行一次 DiT，负向提示词不是免费的文本过滤，也不保证对蒸馏 Turbo 的所有提示词改善效果。

## 验证与复现

Python 仅作 oracle；运行时仍是 Rust/Burn。对照环境、权重版本、固定噪声和 eager 文本注意力设置沿用
[既有数值验证](qwen-image-21.md#数值验证)。同一初始噪声的首轮结果：

- 512² 文生图、非空负向词、CFG=2：与官方完整 8 步输出 PSNR **41.12 dB**。
- 256² 双参考图，含非正方形 RGBA，空负向词、CFG=2、参考分辨率 288：PSNR **47.05 dB**。
- 关闭引导后重复原请求，UI 输出逐像素一致；正负分支缓存均不会带到下一请求。
- 使用保存的上一版 CLI 执行同一请求，关闭引导的图片与新版逐像素一致；开启引导的 CLI 与 Makepad 图片也一致。

最终原生界面测试中，512² 茶壶、seed 42、每种模式各取 3 次预热后运行：

| 模式 | 端到端等待中位数 | 观测范围 | 去噪阶段中位数 |
|---|---:|---:|---:|
| CFG 关闭 | 11.3 s | 10.6–14.6 s | 10.53 s |
| CFG=2，含负向词 | 25.6 s | 22.0–28.2 s | 24.59 s |

端到端等待包含界面通知，不含权重加载和 PNG 保存。测试未锁定系统温度/功耗，运行有波动；
这些是本机该请求的观测，不是所有分辨率的性能承诺，也不是 Metal 与 ANE 的对比。
负向分支期间停止的本次观测延迟为 0.17 s；首次内核编译期间仍需等当前操作结束。
9 项图像单元测试、16 项原生单元测试（另有 1 项原有 ignored）通过；WebGPU 入口编译通过，未做该平台性能验收。

验证脚本：`prototypes/makepad/validate-image-guidance.py`，覆盖参数校验、预设、换种子、开关提示、
负向分支取消/恢复、两分支进度、保存、双参考图、恢复默认路径及 ASR 回归。
官方完整管线脚本 `scripts/reference_qwen_image_edit.py pipeline` 现支持负向参数，也可不传参考图验证文生图。
工件保存于 `artifacts/qwen-image-guidance/`；最终二进制、源代码清单与结果以该目录 `delivery.json` 为准。
可用 `.venv-qwen-image/bin/python scripts/check_qwen_image_guidance.py` 重新计算图像误差与上述计时中位数。

本次网页摘录保存于 `artifacts/qwen-image-guidance/research/`。Apple 文档与演示覆盖的硬件/模型各不相同；
文中的本项目性能假设必须通过上面的原型与端到端测试验证。
