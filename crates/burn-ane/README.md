# burn-ane

Burn 0.21 的实验性 Apple Neural Engine **推理后端**。模型直接使用 Burn 的 Tensor、
`nn::Linear`、卷积等 API，后端自动从算子中间表示构建计算图，不要求模型专用导出器。

本机支持 Apple Silicon + macOS 15+，构建需要 Rust 1.92+ 和 Xcode Command Line Tools。
默认运行时使用 Rust、静态链接的 Swift 桥接与系统 Core ML；可显式选择实验性的直接 ANE 运行时。
两条路径均不依赖 Python 或 coremltools。
非 Apple Silicon/macOS 平台只提供明确的回退/错误路径，不具备 ANE 执行能力。

## 使用

仓库的 `vendor/burn` 是上游 Burn 0.21.0 的小型补丁：增加 `ane` feature 与后端导出，
共享 Metal 张量另使用本仓库对 `burn-router`、`burn-fusion`、`cubecl-runtime` 和
`cubecl-wgpu` 的小型补丁，其他依赖来自 crates.io。这些扩展尚未发布或合并上游。

`metal-shared` feature 在 Apple Silicon 上为 `Ane<burn_wgpu::Metal>` 启用
IOSurface / Metal 共享存储。调用方 workspace 必须包含本仓库对应的四条
`[patch.crates-io]`，版本为 Burn 0.21.0、CubeCL 0.10.0、wgpu 29.0.4。
原始包来源、SHA-256 和逐文件补丁见
[vendor 清单](../../artifacts/apple-shared-ane/vendor-patches/manifest.json)。

设置 `AneOptions { runtime: AneRuntime::Direct, ane_first: true, ..Default::default() }`
可关闭小图、单行矩阵的成本路由，让受支持子图优先尝试 ANE；失败时按 `fallback`
策略处理。FP32 公共张量类型不再是共享 Metal 路径排除 ANE 的依据，Direct 的实际
计算和表面数据为 FP16。独立库默认调度不变，Makepad 现采用成本路由及按阶段保持 Metal 的延迟策略；`OMNI_ANE_FIRST=1` 可恢复原诊断策略。
共享边界及验证说明见 [接入记录](../../docs/macos-unified-memory-research.md#fp16-共享张量与-ane-优先接入)。

`AneDevice::fallback_scope()` 将一个依赖连续的阶段提交给回退后端。它先完成已有 ANE 依赖，随后不额外等待回退后端的 GPU 队列。
作用域仅影响当前线程和设备，支持嵌套及异常退出恢复；严格 `FallbackPolicy::Error` 会拒绝它。
Makepad 设置 `prefer_fallback_phases = true`，模型才会自动采用整段路由提示；独立库
默认关闭，`ane_first` 和严格模式仍优先。
`AneClient::fallback_float()` 则把浮点输入的惰性句柄交给自定义回退内核，再注册其输出。
这使 `Ane<Metal>` 可以继续调用原生 Metal GEMV/RMS；已有 ANE 输出通过共享表面转换成 Metal 句柄，
不需要 CPU 数组中转。跨会话输入和非浮点输入会被拒绝。

非 ANE 优先模式中，明显不适合卸载的小图在构造、分桶和序列化 ANE 图之前便回退。
`early_cost_policy_skips`、`fallback_scope_operations`、`custom_fallback_operations` 和
`shared_metal_synchronization_ms` 用于验证路径命中及等待成本。共享内存上的真实数据依赖仍需同步，
优化通过减少设备切换降低等待次数。

```toml
[dependencies]
burn = { path = "vendor/burn", default-features = false, features = ["std", "ane"] }
```

```rust
use burn::{backend::{Ane, AneDevice}, nn::LinearConfig, tensor::Tensor};

type B = Ane; // 默认 NdArray 回退；可改为 Ane<burn::backend::Metal>
let device = AneDevice::default();
let layer = LinearConfig::new(512, 512).init::<B>(&device);
let x = Tensor::<B, 3>::ones([1, 128, 512], &device);
let result = layer.forward(x).into_data();
println!("{:?}", device.stats());
device.clear_cache();
```

只使用独立 crate 时，可直接依赖路径 `crates/burn-ane`，从 `burn_ane` 导入相同类型。
其他主程序的 `build.rs` 需补充系统 Swift 动态库搜索路径（依赖的 link-arg 不会传递）：

```rust
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
```

本项目的可复现实机验证：

```sh
cargo run --release --features ane --example ane_backend
cargo run --release --features ane -- --backend ane doctor
cargo run --release --features ane --example ane_metal
cargo run --release --features ane --example ane_blocks -- --case ffn --backend ane --steps 31
cargo run --release --features ane --example ane_blocks -- --case ffn --backend metal --steps 31
cargo test --manifest-path crates/burn-ane/Cargo.toml
cargo clippy --manifest-path crates/burn-ane/Cargo.toml --all-targets -- -D warnings
```

`ane_backend` 用普通 Burn Linear + ReLU，与 NdArray 对照，并断言真实 Core ML 计划含 ANE
分配、不同输入命中缓存、没有 Burn 回退。`doctor` 验证 ANE 图和 Metal FP16 张量的衔接，`ane_metal` 进一步验证 ReLU 和布尔结果回读。
macOS ARM64 上的硬件测试需要系统允许使用 ANE，不能用纯 CPU 虚拟机代替。

## 执行方式

`Ane<B> = BackendRouter<AneChannel<B>>`，实现 Burn Backend 张量接口。
自定义 `RunnerClient` 捕获 `OperationIr`，在读取张量、同步、遇到不支持的算子或达到图长度
限制时编译子图。编译器直接在 Rust 中生成 Apple 公布格式的 NeuralNetwork protobuf。
Swift 用 `MLModel.compileModel` 编译，使用 `.cpuAndNeuralEngine` 加载和执行。

仅在 `MLComputePlan` 报告至少一个矩阵乘法或卷积优先分配到 ANE 时执行该 Core ML 图；否则记录原因并
交给 `B`。Core ML 的 CPU 算子和 Burn 回退算子分别统计。设备计划不等于硬件计数器追踪，
`ane_compute_operations_planned` 只统计矩阵乘法/卷积，不把常量和布局操作当作计算加速证明。
不能据此宣称所有计算都在 ANE。ANE/FP16 是近似计算，F32 张量接口不意味着 FP32 累加精度。
量化尚不支持；训练、自动微分和所有 Burn 算子的 ANE 覆盖尚未验证。

当前进入图编译器的算子：

| 类别 | 支持的计算 |
| --- | --- |
| 矩阵/模块 | Matmul（含动态 RHS）、Linear、静态权重 Conv2d、Attention |
| 逐元素 | Add/Sub/Mul/Div 及标量形式、Abs、Exp/Log/Sqrt/Recip、Tanh、Erf、PowfScalar、Clamp |
| 布局 | Reshape、SwapDims、Permute、Expand、Cat、F16/F32 Cast |
| 归约/掩码 | MeanDim/SumDim/MaxDim/MinDim、LowerElem/LowerEqualElem、MaskFill/MaskWhere |
| 组合函数 | 归一化、普通及多项式 GELU、FFN、Attention/Transformer block 子图 |

进入编译器并不保证系统将每一种形状都分配到 ANE；小图尤其可能被分配到 CPU。
当前只编译包含矩阵乘法或卷积的子图，支持非空、最多 5 维的 F16/F32 张量与图内布尔掩码。
索引、随机数、整型计算、尚未支持的算子通过 `B` 执行。
NaN/Inf 标量走回退路径。动态 Conv2d 权重等未实现的组合也会明确回退。
图内 F16/F32 Cast 保留在 Core ML 图中；ANE 使用较低精度计算，不能用 Cast 要求 ANE
执行 FP32 累加或保证每个中间值与 CPU 舍入逐位一致。图输出恢复声明的 Burn dtype。

静态 Linear/Matmul 转为 `[1,C,1,S]` 的 1×1 卷积；权重的转置、reshape 和类型转换在编译时
折叠，避免每个解码步重新传输词表投影权重。Attention 合并 batch/head 维，使用两次批量矩阵乘法。
支持 scale、softcap、布尔掩码、加性 bias 和底部对齐的矩形因果掩码。遮罩使用有限下界及
显式清零分子，避免 Inf 常量导致 ANE 拒绝编译，并让全遮罩行输出零。

编译器识别沿最后一维的 LayerNorm/RMSNorm 表达式，分别使用 Core ML LayerNormalization 和
等价的 L2Normalize 缩放。这样避免把平方、方差和 epsilon 拆成 FP16 中间结果时，小方差输入
出现明显误差。归一化的仿射参数继续参与同一子图的融合。
Burn 展开的 `exp(-log(1 + exp(-x)))` 会被识别为原生 Sigmoid，避免 FP16 中间值破坏
大幅值输入的饱和行为；SiLU/SwiGLU 可与相邻投影留在同一计算图。

## 缓存、错误与生命周期

`AneOptions` 控制 `FallbackPolicy::{Allow, Error}`、缓存容量、图操作数和常量内存上限。
`Error` 禁止 Burn 回退，在下一次异步读取或 `sync` 返回 `ExecutionError`；该设备会话进入
错误状态，需新建设备恢复。它不禁止 Core ML 子图内混合 CPU 调度，后者见独立统计计数。

缓存键包含规范化算子、形状、dtype、标量、存活输出和静态权重身份，不包含普通输入的张量 ID。
因此新输入可复用图，替换权重会重新编译。默认驻留上限 256 张图、2 GiB 序列化模型大小；
这不是 Core ML 实际内存占用的硬上限。失败记录有独立的 512 项缓存，不会挤掉已编译模型。
默认每图最多 512 个有效操作，Drop 不占融合预算；到达常量预算时提前切分。

`row_bucket` 默认 32，Self-Attention 至少按 64 分档。编译器跟踪 token 轴，证明归约、
reshape、广播等不混合 token 后才填充。Attention 使用随有效长度变化的运行时遮罩，输出
裁回原长度；不会增加音频等待。跨注意力、外部掩码、token 轴归约等未证明安全的图保持原形状。
设置 `row_bucket: 0` 可关闭。卷积前端在进入线性层前切分，避免阻止后续 Transformer 分桶。
带外部遮罩或不同 Q/K 长度的 Attention 在输出处切分，避免增长中的 KV 缓存让后续 FFN/
投影反复编译；普通无掩码 Self-Attention 仍与前后投影融合。因常量预算切图时，权重视图跟随
使用它的矩阵乘法，避免把词表的大矩阵当作动态图输出逐步传输。

`cache_dir: Some(path)` 开启磁盘缓存，键为包含权重的完整模型 protobuf SHA256，并按系统
版本、硬件型号、运行时版本隔离。默认磁盘预算 8 GiB，按最近使用时间淘汰。模型在首次加载前
发布到稳定路径，此后直接复用；跨进程共享锁保护活动模型，淘汰时跳过它们，所以活动包总量
可能暂时超过磁盘预算。缓存包含逐文件 SHA256 清单，在交给系统加载器前校验；损坏缓存会重新编译，
目录不可写时继续推理。`clear_cache` 释放内存中的模型和
临时编译目录；磁盘缓存保留，原始张量仍可使用。
复用同一个 `AneDevice`，结束会话时调用 `clear_cache`；Burn 0.21 的 Router 客户端注册表
按设备全局持有客户端，不宜为每个输入新建设备。

预测成功后才替换输出句柄；失败时原输入可安全用于回退。后端跟踪 Burn 的 ReadOnly/ReadWrite
及 Drop，消除废弃图并保留仍被引用的中间结果。每个设备串行执行，模型缓存有界。
Core ML 图边界保留主机张量；Direct FP16 输出保留有所有权的 IOSurface 视图，后续图按逻辑有效区段复制。
公开读取或实际回退才按需物化/上传；允许回退时，已有张量的 Select、Cat、步长为 1 的 Slice 可直接在主机复制。
这些布局操作继续计入回退。尚未实现 Metal/ANE 零拷贝。

## 计时与设备分配

`device.stats()` 分开记录规划、权重读取、模型生成、编译/加载、输入读取、预测、输出写入和
回退调度耗时，以及缓存命中/淘汰、传输字节数和回退原因。`fallback_dispatch_ms` 只统计提交，
异步 GPU 执行时间可能落在后续同步/读取中；`prediction_ms` 包括 Swift 与 Core ML 的缓冲区复制。
这些是主机计时，不是 ANE 芯片级 profiler。

`planned_operations_by_device` 按算子类型列出 `ane:*`、`cpu:*`。只有 Core ML 调度证据，
不能把“无 Burn 回退”称为“全部由 ANE 执行”。`min_compute_rows` 默认 1，仍尝试单 token 图；
可手动调高来试验小任务回退策略。CLI 的 `--ane-strict` 仅禁止 Burn 回退，不能强制 Core ML
把 CPU 算子搬到 ANE。

```sh
# CLI 默认开启 ~/Library/Caches/burn-ane 缓存；诊断日志写 stderr。
cargo run --release --features ane -- --backend ane --ane-trace transcribe \
  --model models/Confucius4-R2T2 --audio artifacts/test.wav --json

# 同一模型块：分别测整图与人为切成小图的调度开销；首项冷启动，其余热态。
cargo run --release --features ane --example ane_blocks -- --case ffn --max-ops 512
cargo run --release --features ane --example ane_blocks -- --case ffn --max-ops 8
```

`ane_blocks` 每轮输入不同，与 NdArray 比较数值，计时包含输出回读；数值对照不计入耗时。
比较 Metal 与 ANE 时请顺序运行，避免同时构建/测试，并分别报告冷态与热态。
`--case block` 可探测 Attention block，但本机最终版本的 32/128 行独立 block 被 Core ML
全部安排到 CPU，示例会拒绝把这种结果当成 ANE 基准。真实模型的较大 encoder 子图与这些
独立小图的分配不同；支持图编译不代表每个组合/形状都有 ANE 调度。
`ane_blocks --model models/Confucius4-R2T2 --case encoder` 和 `--case decoder` 还可检查真实
检查点的 encoder embedding、decoder prefill/单步 logits 与 Metal 的数值及 greedy token 一致性。

模型专用 `coreml` feature 和旧导出脚本不参与此后端实现。

## 直接 ANE 运行时

`AneOptions { runtime: AneRuntime::Direct, ..Default::default() }` 改用 Rust 的 Burn IR → MIL
编译器，以及隔离在 `native/Direct.m` 的 `_ANEInMemoryModel` 私有接口。
它直接提交 ANE 程序，不经 Core ML 的 CPU/ANE 调度。默认仍为 `AneRuntime::CoreMl`；
直接路径失败时遵守同一 `FallbackPolicy`，不会暗中改走 Core ML。
私有接口目前只在上述 M5 Max / macOS 27.0.1 实测，不承诺其他系统版本兼容。

```sh
cargo run --release --features ane --example ane_blocks -- \
  --ane-runtime direct --case ffn --strict --steps 31
cargo run --release --features ane --example ane_blocks -- \
  --ane-runtime direct --case block --strict --steps 31
cargo run --release --features ane --example ane_blocks -- \
  --ane-runtime direct --model models/Confucius4-R2T2 --case model
cargo run --release --features ane -- --backend ane --ane-runtime direct transcribe \
  --model models/Confucius4-R2T2 --audio speech.wav --json
```

直接编译器维持低维投影的 `[B,C,1,S]` 物理布局，折叠权重视图，用同一个权重 blob
打包所有常量，避免 macOS 27 多权重文件的模型标识问题。权重 FP16 转换使用 half 的批量
转换接口，并提前预留 blob 空间。按输出递归生成代码，归一化和 Sigmoid 融合后的废弃分支
不进入 MIL。新增 Slice（步长 1）和 RepeatDim，覆盖 RoPE、
GQA 和 KV 拼接。支持带外部掩码/已有 KV 的整层融合；开启下述分档优化时，默认将
动态 Attention 与大权重图分开，以减少形状变化引起的重复加载。

RMSNorm 使用 `LayerNorm(concat(x,-x))` 后取前半段，避免 FP16 平方溢出与小方差损失。
代价是归一化的中间通道加倍。其他候选表达在本机小方差压力测试中出现明显误差，未启用。
测试以相同 FP16 可表示权重/输入对照 NdArray FP32 累加，并单独检查归一化后的 KV 误差。

直接路径用有界 `e = exp(-abs(x))`、正负分支分别 `1/(1+e)` 与 `e/(1+e)` 实现 sigmoid，
避免本机 native sigmoid 在 SiLU 中的中心区间偏差。真实 decoder 第 3 个缓存 token 的 logits
相对 RMSE 从 5.271% 降至 0.197%，原容差未放宽；计算仍在直接 ANE 图内。
Core ML 编译器的原生 Sigmoid 路径保留。详见[数值诊断](../../docs/ane-optimization-goal.md#数值诊断)。

IOSurface 由模型报告的尺寸、行/平面/batch 步幅分配和读写，处理短序列的 64 字节行对齐。
图的多输入、多输出按名称排序绑定。未验证的 depth 布局明确拒绝；异常被隔离在原生桥接中。
编译源目录由跨进程锁保护；加载后删除源副本并释放 descriptor 的权重副本。

直接路径的图间缓冲区现在使用 FP16：长度补齐、KV 裁剪和物理布局转换直接复制 16 位元素，
原生桥接按 IOSurface 的实际行步幅批量复制。FP16 模型张量跨图保持 FP16，避免反复展开
成 F32 再转换回来。二维/三维布局转置分块执行，单行不做转置；F32 公共张量仍按声明类型
读回或提供给回退后端。Burn Router 的公开 `into_data()` 会转换为后端 `FloatElem`，
不能用该接口返回的 F32 类型推断内部图间存储类型。
IOSurface 原本就是 FP16；该改动减少主机侧中间缓冲和转换，不降低模型计算精度。

`input_bytes` / `output_bytes` 按实际图接口类型计数，直接路径为 2 字节/元素，Core ML
路径仍为 4 字节/元素，不包含 IOSurface 的对齐空隙。`prediction_ms` 包含 Rust 布局处理
和完整原生调用；新增 `direct_input_copy_ms`、`direct_evaluate_ms`、`direct_output_copy_ms`
分别记录原生输入复制、同步 evaluate 调用及输出复制。evaluate 时间仍包含驱动与调度，
不是硬件计数器测得的纯 ANE 计算时间。

`direct_ane_graph_executions` 统计直接 ANE 调用成功次数，`direct_system_cache_hits` 统计
系统 aned 的持久编译缓存命中。直接路径的 Core ML 计划计数为零，不能把它当作未执行。
`cache_dir` / `max_disk_cache_bytes` 只管理 Core ML 包；aned 的系统缓存由系统管理。
应用内图缓存仍受 `max_cached_graphs` / `max_cached_bytes` 限制。

直接路径支持 KV 容量分档（`AneOptions::kv_bucket`，默认 128，0 关闭）。编译器保守追踪
prefix slice、KV concat、GQA reshape/repeat 到 Attention 的序列轴。历史区段按容量补齐，
当前 token 保留独立区段；运行时掩码排除空槽，并按原逻辑位置处理因果关系。输出裁回
有效长度，因此缓存截断、重写和跨档追加保留 Burn 语义。未证明安全的图使用原始形状。
容量会随上下文增长，不是录音时限。同一档内且当前 token 数相同时可以复用已加载的图。

启用 row padding 时，默认把带缓存的 Attention 单独放入不含模型权重的图，保留投影、
归一化、RoPE 和 FFN 的融合。不同当前 token 数可共用分档后的大权重图，各层还可共用
相同形状的 Attention 程序。单 token 在直接路径中保持原尺寸；选择最后一个 token
使用运行时 one-hot 掩码和归约，词表投影只算一行，避免补齐整个词表输出。
`separate_cached_attention = false`（CLI `--ane-no-separate-attention`，示例
`--no-separate-attention`）可保留完整块融合用于对照。关闭 row padding 也会关闭此切分。
Prefill 在 GQA 扩展前也切图，使投影与已有 KV 的调用共用输出形状。允许回退时，输入 token
拼接前结束已有投影图，再作为一次布局复制执行并单独计数，使后续 transformer 图可按行数分档；
strict 模式保留融合，避免新增强制回退。

实验选项 `fuse_single_token_attention`（CLI `--ane-fuse-single-token`）仅在当前 token 数为 1 时
合并投影、KV attention 与 FFN；与 `kv_bucket = 256` 搭配降低了短音频热态重载次数，
但增加完整 KV 的补齐传输，尚未达性能目标，默认关闭。
`pad_single_row`（CLI `--ane-pad-single-row`）允许单 token 投影使用 row bucket，实测更慢，默认关闭。
图规划现使用类型化 IR 修改形状、规范化张量 ID，避免 JSON Value 中转；缓存 key 仍包含权重身份和有效输出。
主机同类型张量可借用读取，输入补齐和布局转换直接写入合并缓冲，输出转换直接写入调用方缓冲；
KV 区段按连续片段复制，减少中间分配和逐 token 复制。数值类型仍为 FP16。

不可变权重的转置及 FP16 打包结果由独立 LRU 缓存共享，key 包含原始 TensorId、类型、
形状及完整视图变换；替换权重会重新打包，新建同一种转置视图仍能复用。库默认
`max_cached_weight_bytes = 2 GiB`，CLI 为 4 GiB；该预算额外计入内存，0 关闭。
`clear_cache()` 同时清空图和打包权重，保留张量值。`weight_cache_hits/misses/evictions`、
`cached_weight_bytes`、`kv_bucketed_graph_executions` 可用于验证实际复用。

Direct 图描述还会按完整 blob 内容共享不可变的 NSData 权重，减少不同 row/KV 形状的重复分配。
该对象独立拥有字节，图持有强引用，内容索引只持有弱引用；卸载程序保留描述，淘汰最后一个描述后释放权重。
图预算按每图 MIL 文本加唯一权重计费，`cached_model_bytes` 报告这个数，
`unshared_cached_model_bytes` 保留各图独立大小之和。`shared_model_weight_bytes`、
`unshared_model_weight_bytes` 与 `shared_model_weight_blobs` 分别报告共享后的权重、共享前的累计权重和存活对象数，
`model_weight_share_hits/misses` 统计编译时的复用。打包权重缓存仍另行计费；这些字段不含系统运行时副本和 I/O 缓冲。
后端 71 项测试覆盖内容变化、不同形状、重载、预算压力和清空；192 组原生缓冲检查通过。

图的字节预算与 ANE 已加载程序资源是两个限制。本机在同时加载到第 128 个程序时出现
资源不足；这是本机实测，不是 Apple 公布的跨设备上限。`max_loaded_direct_graphs` 默认
112。编译图缓存和已加载程序现在分别管理：释放程序槽时优先选小于 16 MiB 的小图，再按 LRU，
保留其编译描述和 I/O 缓冲，释放模型实例，下次用同一描述重新加载。仅调用 `unload` 在本机
无法释放全部实例资源，140 程序压力测试曾失败；释放实例后的版本通过。保留的编译图仍受
`max_cached_graphs` 和 `max_cached_bytes` 限制，包含已卸载的图，未增加原来的缓存预算。
加载错误 Code 54 标记为临时资源不足，释放旧实例后有限重试，也不写入永久失败缓存。

`program_cache_policy` 可显式选择 `ProgramCachePolicy::{PreferSmall, Lru, ReplaceWeightShapes}`
（类型位于 `burn::backend::ane`）。默认 `PreferSmall` 保留上述策略；`Lru` 只看最近使用顺序。
`ReplaceWeightShapes` 在槽位不足时优先卸载与待加载图共享同一个不可变权重 blob 的程序，没有匹配时使用原策略。
该偏好不认定两张图等价，不替换图或权重数据，不影响编译描述和张量寿命；所有预算仍生效。
CLI 使用 `--ane-program-cache-policy prefer-small|lru|replace-weight-shapes`，示例与评测脚本使用
`--program-cache-policy`。`direct_same_weight_unloads` 统计该偏好选中的卸载次数，包含在 `direct_program_unloads` 中。
两个槽、两组权重和四张图的回归已验证数值、预算及旧形状重载。真实 ASR 各一轮 40 秒探针中，
LRU / KV256 为 RTF 1.0112 / p95 251.05 ms，原策略为 1.0509 / 303.63 ms；但 LRU 短音频更慢，
LRU / KV128 也未进一步改善连续性能。尚无新策略完整集和 30 分钟验收，默认不启用新偏好。
其他重载失败会丢弃该缓存项，尝试正常编译路径恢复。
`resource_retries` 记录恢复次数。硬件压力测试连续执行 140 个不同模型，13 次恢复成功。
`IOSurface allocation failed` 也按临时资源错误处理，不写入不支持图的失败缓存。
它需要释放缓存图持有的 I/O，恢复时优先淘汰已卸载的 Direct 图，再选其他 Direct 图；
仍被张量引用的输出独立保留。编译、重载和原生输出分配最多重试 8 次，无可释放图时立即按回退策略结束。
`surface_resource_retries` 是总资源重试的子集，`resource_recovery_ms` 记录释放资源耗时，
在编译/重载时与 `compile_load_ms` 重叠。79 项后端测试、真实模型位对照通过；30 分钟实测
76 次资源恢复全部成功，无矩阵回退或失败缓存命中。此修复保留，底层分配压力仍存在。
Direct 注意力使用负无穷遮罩，并在归一化前显式清零被遮罩的指数值；全遮罩行输出精确为零。
有限负值填充在极值概率用例中会失败，现有修复覆盖该边界。80 项后端测试、真实模型和跨窗口
数值对照通过；短测未证明稳定性能收益。2026-10-11 按用户要求以该保留版本收尾，
图内投影行切片实验已撤回归档；原实时性能目标未达成。
版本指纹、验证范围和运行命令见[收尾记录](../../docs/ane-optimization-goal.md#2026-10-11-收尾选择)。
CLI 提供 `--ane-graph-cache-mib`（默认 8192）、`--ane-max-cached-graphs`（默认 1024）、
`--ane-loaded-graphs`（默认 112）、`--ane-row-bucket`（默认 32）。初始化、预热和正式测量
的统计分别输出，便于确认热态是否仍在编译、淘汰或因资源不足而回退。
`cached_graphs` 包含卸载的编译图，`loaded_direct_graphs` 才是当前占用程序槽的数量。
`direct_program_reloads/unloads`、`direct_reload_ms/unload_ms` 和 `direct_reload_failures`
分别记录实例重载、释放及失败；`cache_evictions` 只表示整个编译图被丢弃。
`compile_load_ms` 仍计入重载及释放开销，热态 RTF 不排除这些时间。

CLI 可用 `--ane-kv-bucket 128 --ane-weight-cache-mib 4096` 调整；基准示例对应
`--kv-bucket 128 --weight-cache-mib 4096 --model-steps 2`。Direct FP16 输出现在保留有所有权的 IOSurface 视图，
跨图按有效区段复制，公开读取、回退或设备转移时再物化；其他 dtype 保持图边界转换。
输出缓冲仅在没有活跃张量引用时复用，图淘汰后旧值仍可读取。它不是零复制，首次遇到新容量或新 token 数仍可能编译。
`resident_native_tensors/bytes` 记录存活视图和补齐后 payload，不含图池和对齐；
`native_input_tensors/bytes`、`native_output_allocations/reuses`、`native_materializations` 记录原生输入、分配与回读。
`native_materialization_ms` 是细分计时，可能包含在输入准备、输出整理或回退阶段中，不应与这些阶段简单相加。完整精度、性能和当前限制见 [VALIDATION.md](../../VALIDATION.md)，
路线依据与来源见 [调研文档](../../docs/ane-backend-research.md)。

### M5 Max 实测配置

本机 128 GB 内存配置下，以下参数已完成完整 FP16 检查点的流式转写：

```sh
cargo run --release --features ane --bin r2t2 -- \
  --backend ane --ane-runtime direct \
  --ane-graph-cache-mib 16384 --ane-loaded-graphs 112 \
  --ane-row-bucket 64 --ane-kv-bucket 256 --ane-fuse-single-token \
  transcribe --model models/Confucius4-R2T2 \
  --audio artifacts/Confucius4-R2T2-upstream/resources/test.wav \
  --chunk-ms 160 --lookahead-ms 160 --warmup --json
```

以下历史阶段采用 row128 / KV256 / 单 token 融合；最后的共享权重阶段改用上面的 row64 配置。

2026-10-10 拼接边界保留版在固定 96 条中文集的一轮热态测量为 RTF **1.0613**、p95 **226.03 ms**，
CER **3.1052%**，相对 Metal 增加 **0.0706 个百分点**；精度和直接 ANE 执行检查通过，性能未达标。
相对减少复制和修改拼接边界之前的版本，完整集 RTF 下降 11.2%、p95 下降 28.2%。
6.74 秒短样本三次汇总却从 RTF 1.0766 / p95 222.43 ms 退步为 **1.1148 / 227.08 ms**，
此边界的收益依赖输入。该保留版的 40 秒连续输入探针积压约 21 秒。
随后分开管理编译描述与程序实例，同配置短探针单次为 **1.0058 / 207.47 ms**；
完整 96 条的一轮复测为 **0.9675 / 207.26 ms**，转写逐条不变、CER 门槛通过，性能门槛未通过。
40 秒连续输入的新探针处理到 32 秒，积压约 8 秒，尚未实时；保留描述使 RSS 抽样峰值升至约 26 GB。
随后直接在逻辑 FP16 张量与 IOSurface 之间完成布局转换和复制，去掉两个每调用临时缓冲。
26 项 direct 测试与 27 组原生传输测试通过，完整模型 20 份输出逐字节不变。
同配置短探针三次为 **0.9662 / 199.27 ms**；40 秒持续输入探针积压 **7.22 秒**，停止响应 27.57 ms。
这一传输版本的完整 96 条中文集一轮为 **RTF 0.9432、p95 201.90 ms**，
CER 3.1052%，原始转写与缓存版逐条相同。性能尚未达标，三轮重复和 30 分钟验收尚未完成。
native `direct_input_copy_ms` / `direct_output_copy_ms` 现包含布局转换，跨版本应看整体 `prediction_ms`。
原生输出保留版通过 61 项后端测试和 192 组原生缓冲区用例，20 份模型输出逐字节不变。
短探针三轮为 **0.9290 / 191.83 ms**；40 秒连续输入积压 **5.14 秒**、p95 **300.45 ms**、停止响应 **50.13 ms**，
RSS 采样峰值 **27.03 GB**。完整 96 条中文集一轮为 **0.9028 / 193.44 ms**，转写逐条相同，
CER 3.1052%，比 Metal 高 0.0706 个百分点，精度通过、性能仍未达标。证据保存在 `artifacts/burn-ane-goal/native-resident-*`。
随后进一步让已有张量的 Select、Cat、步长为 1 的 Slice 在主机按元素位模式复制，减少 Metal 往返。
它们仍计入回退，严格模式保持拒绝；新增 `host_copy_operations` / `host_copy_output_bytes` / `host_copy_ms`，
后者包含于回退调度时间。67 项后端测试通过，20 份模型数组逐字节不变。
短探针三轮为 **0.9082 / 188.22 ms**；40 秒连续输入积压 **4.18 秒**、p95 **293.50 ms**，
RSS 采样峰值 **33.73 GB**，仍未实时。完整 96 条集一轮为 **0.8716 / 186.13 ms**，
CER 3.1052%，转写逐条不变，精度通过、性能未通过；证据保存在 `artifacts/burn-ane-goal/host-layout-*`。
该阶段 row64 在短样本更快，但长流缓存反复淘汰、积压 21.62 秒，因此当时保留 row128。
共享权重阶段减少了不同形状描述中的重复分配，原预算不变；71 项后端测试、192 组原生缓冲检查通过，row64 / row128 的 20 份模型数组均逐字节不变。
row64 三轮短探针为 **0.8556 / 188.02 ms**；40 秒连续输入为 **1.0774 / 305.06 ms**，
积压 **3.54 秒**、停止响应 **22.44 ms**、RSS 峰值 **23.02 GB**，热态 3 次新图、0 次淘汰，缓存抖动已消除。
381 张图共享前需 22.94 GB，共享后源文本和权重计费 8.26 GB。完整 96 条集一轮为 **0.8206 / 189.63 ms**，
CER 3.1052%，96 条转写逐条不变；相比上一版 row128，耗时降低 **5.86%**、p95 **增加 1.88%**。
当前实测配置采用上面的 row64；性能仍未通过，三轮完整重复与 30 分钟验收尚未完成。
证据保存在 `artifacts/burn-ane-goal/shared-descriptor-*`。

其后的二进制图键与栈上 IR 遍历保留原图身份和数学表达，73 项后端测试、两组 Clippy 与模型 20 份数组位对照通过。
row64 / KV256 三轮短测 **0.8366 / 183.76 ms**；40 秒连续输入 **1.0509 / 309.53 ms**，积压 2.90 秒、停止响应 33.13 ms。
完整 96 条集一轮 **0.8126 / 187.45 ms / CER 3.1052%**，96 条转写与共享权重版相同，CER 相对 Metal +0.0706 个百分点。
规划 19.79 → 14.21 秒，加载路径 25.71 → 29.04 秒；总耗时降低 0.97%，p95 降低 1.15%，控制为归档单轮。
精度和直接 ANE 检查通过，性能仍未通过；52 / 96 条 RTF、10 / 96 条 p95 达标。三轮完整重复和 30 分钟验收未完成。
该阶段完整集构建为 `artifacts/burn-ane-goal/inline-ir-bin`，后续程序槽策略另行测量，不能混用结论。

曾试验小批次 RMSNorm：最多 32 个向量使用附加常量能量的原生 L2，大批次保留原 LayerNorm。
常量用 MIL fill 表达，不改变共享权重 blob。75 项后端测试、56 组数值案例及模型 encoder / prefill / 8 token
对照通过；三个交替 A/B 短测平均耗时降低 1.07%，收益集中在一轮。LRU 连续单轮 p95 为 256.05 ms，
同期控制为 277.49 ms，但较早控制曾为 251.05 ms。完整 96 条一轮为 0.8139 / 186.85 ms / CER 3.0346%，
精度与 Metal 相同，但没有稳定性能收益。该算术改动也已归档撤回，工作树恢复原 LayerNorm。
全批次 L2 候选曾在数值探针通过后出现真实转写遗漏和明显变慢，已归档否决。
实现、失败对照及复现命令见[归一化实验](../../docs/ane-optimization-goal.md#rmsnorm-附加能量通道)。

最新保留版在 Burn 模型层缓存稳定的完整卷积块，直接后端数学表达与通过 74 项测试的程序槽版相同。
14 项 CPU/模型回归、两种根功能组合 Clippy、Wasm 检查通过，9 个真实 encoder 阶段的缓存与重算输出逐字节相同。
三个交替短测为 **0.7944 / 171.41 ms**，控制为 **0.8392 / 183.27 ms**，所有正文相同。
完整 96 条一轮 **0.7708 / 173.38 ms / CER 3.1052%**，相对归档 IR 遍历版耗时降低 5.14%、p95 降低 7.50%。
96 条正文全部相同，精度、直接 ANE 与缓存预算通过；78 条通过 RTF、28 条通过 p95，性能仍未达标。
当前完整集保留构建为 `artifacts/burn-ane-goal/convolution-prefix-bin`，完整集采用 `prefer-small`。
另用 LRU 的 40 秒连续输入为 **0.9951 / 268.78 ms**，积压 1.46 秒。
同一冻结构建的 30 分钟 LRU 测量为 **1.0228 / 290.78 ms**，结束积压 42.74 秒，
SIGINT 停止响应 105.32 ms。IOSurface 分配失败触发 1045 次相关回退，其中 1015 次命中失败缓存；
性能和直接 ANE 验收未通过；后续资源恢复修复的 30 分钟复测为 **1.0174 / 293.70 ms**、
积压 33.07 秒、停止响应 63.97 ms，76 次恢复成功，矩阵回退为 0。修复已保留，性能仍未达标，三轮完整重复尚缺。

每图融合诊断可用 `--ane-max-constant-mib` / `--ane-max-ops`，默认 256 MiB / 512 操作。
512 / 1024 MiB 的短探针对照未显示收益，未调整默认值；这两个参数不扩大总缓存预算。
此前 320 ms 分块的 4.100 秒 / RTF 0.608 使用修复前 sigmoid，作为历史结果保留，当前版本尚未复测。
完整标准、固定中文集和限制见[优化验收](../../docs/ane-optimization-goal.md)。该历史阶段的原生界面默认 Metal；
当前共享后端及阶段调度见[Makepad 实测记录](../../docs/macos-unified-memory-research.md#混合后端的延迟调度优化)。

图缓存预算 16 GiB，加上 CLI 默认的 4 GiB 打包权重预算，适合这台大内存设备；这不是
进程 RSS 上限，也不包括原始权重和原生运行时分配。新进程还需要权重加载和图编译/加载；
`--warmup` 明确排除了这些首次使用开销。完整 ASR 仍含 embedding 索引、argmax 和
布局复制的 Burn 回退，不应加 `--ane-strict` 来运行这个完整模型命令。

格式依据：[Apple Core ML 模型格式](https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html)。
接口依据：[Burn 0.21 Router](https://github.com/tracel-ai/burn/tree/v0.21.0/crates/burn-router)。
布局参考：[Apple 的 Transformer ANE 优化](https://machinelearning.apple.com/research/neural-engine-transformers)。
