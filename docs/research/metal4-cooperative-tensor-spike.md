# Metal 4 cooperative-tensor 评估 spike（#52 备选路线）

2026-10-23。#52 三切片后 burn 侧保值弹药见底（-21% → 0 → -1%，周期 ~100-119ms
距 ≤98ms 继续线仍差一刀），按止损预案转本 spike：评估手写 Metal 4 张量路径
能否收回发射债并把 verify 压向带宽地板。纯侦察：不改默认、不写生产内核。

## 结论

**GO，量级正确。** 四个先决问题全部就地验证：

1. **机器**：本机 M5 Max（40 GPU 核），`system_profiler` 报 **Metal Support:
   Metal 4**，macOS 27.0.1。
2. **API 面**（SDK 头文件一手证据，`MacOSX27.sdk/.../Metal.framework/Headers`）：
   - `MTL4MachineLearningCommandEncoder`：`setPipelineState` +
     `setArgumentTable` + **`dispatchNetworkWithIntermediatesHeap:`**——一次
     dispatch 跑整个网络，中间张量由 Metal 从堆里管（`intermediatesHeapSize`
     可查）。这正是消灭发射风暴的图式提交模型（当前 verify 每前向 ~2600 发）。
   - `MTL4MachineLearningPipelineDescriptor` 接受 **`MTL4FunctionDescriptor`
     （MSL 函数）**——不是只能喂 Core ML 模型，自定义网络可直接写。
   - `MTLTensor`（`MTLTensor.h`）：rank ≤16 张量资源，数据类型原生含
     **Int4**（macOS 26.4+）、**FP8 E4M3/E5M2、FP4 E2M1、Int2**（27.0）——
     35B int4 专家权重是一等公民。compute encoder 另有 `copyFromTensor`
     张量切片/reshape。
3. **MSL 编程模型**（官方样例 "Running inline ML operations in a shader with
   Metal 4"，经文档 JSON 提取）：普通 compute kernel 直接声明
   `tensor<device half, dextents<int, 2>>` 参数；`slice<静态块>()` 切块；
   `matmul2d<descriptor, execution_simdgroups<4>>` 在**协作张量**（neural
   accelerator 单元）上做矩阵乘，**逐元素后处理直接写在协作张量上**再
   `store()`——norm→matmul→激活→残差的逐层链原子齐了，无双舍入陷阱的
   f16 精度由我们自己定义（与 cube DSL 相比少了 SSA 折叠层）。
4. **Rust 通道**：仓库已锁定的 `objc2-metal 0.3.2`（burn-ane 在用）生成绑定
   里就有 `MTL4MachineLearningCommandEncoder`/`MTL4*` 全家（cargo feature
   选通）。工具链：CLT SDK 头文件齐；离线 `metal` CLI 需完整 Xcode——运行时
   MTL4Compiler 从源码编译 MSL 不需要它（待 M0 实证，列为开放项）。

## 带宽地板模型（判据算术，锚点全部来自本仓实测）

每次 verify 前向的必读字节（int4 专家 1.57MB/个，top-8 路由，40 层）：

| 构成 | block 16（代码 τ≈5.9） | block 3（prose τ≈2.0） |
| --- | --- | --- |
| MoE 专家（触达并集期望） | ~102 专家/层 ≈ 6.4GB | ~24 专家/层 ≈ 1.5GB |
| 稠密投影+共享专家（f16） | ~2GB | ~2GB |
| KV/gdn 状态/杂项 | <0.2GB | <0.2GB |
| 合计 | ~8.5GB | ~3.5GB |
| 理论读时 @400GB/s | ~21ms | ~9ms |
| 现实测 verify | 82–105ms | ~60–70ms |

现 verify 的有效带宽仅 ~90–130GB/s——墙钟大头是发射/串行化，不是字节，这正是
Metal 4（一次 dispatch + 协作张量吞吐）能动的两部分。参照系：MLX BF16 实测
50 tok/s ≈ 全周期 ~140GB/s 有效带宽——**若 Metal 4 路径达到 MLX 同等效率**，
int4 字节数减半给：

- **代码类**：verify ~40–55ms → 周期 ~55–70ms → τ5.9 下 **84–107 tok/s**。
  100 目标在乐观沿（需要 MLX 级效率 + propose 侧同步融合）。
- **prose**：verify ~12–20ms → 周期 ~25–35ms → τ2 下 **55–80 tok/s**——
  τ 天花板不动，但周期地板塌了，prose 也大幅受益（#41 归档里的
  「工程上限 ~33-36」按 wgpu 发射模型算的，本模型下不成立）。

不确定性（诚实标注）：专家并集/路由命中率、协作张量对 int4+分组的实际吞吐、
MLX 级效率能否达成、KV 随上下文增长。M1 微基准是裁决点。

## 数值契约（M2 前必须维护者裁决）

协作张量 matmul 的累加序与我们内核不同 → **对现二进制的字节一致门在此路线上
不可能保持**。建议替代门：冻结参考（当前二进制的 PROSE/CODE 贪心 token 流作
为固定参考）+ τ 偏差 ≤±5% + 抽样质量对照；与 MLX/vLLM 对照的口径一致。
切片 2 的教训（f16 逐算子复现的脆弱性）在此路线下天然消解——精度由我们的
MSL 逐语句定义，没有第三方优化器插手。

## 集成架构选项与最大风险

- **A 全原生解码**：文本解码整链搬到 MTL4（权重直载 MTLBuffer），burn/wgpu
  只留视觉/预填充或同步替换。无互操作税，但是解码器全重写（月级）。
- **B 混合**：burn 保留，仅 verify 前向经 metal_bridge 式缓冲共享交给 MTL4。
  风险：cubecl/wgpu 的缓冲所有权跨运行时导出（wgpu-hal 原生 MTLBuffer 导出
  或权重双驻留——20GB 双驻留不可接受）→ 需零拷贝通道验证，列为 M1 附加项。
- 推荐 **A**（架构上干净，金属桥已证明 unsafe 借用模式可扩展）。

## 项目判据（建议）

- **M0（1-2 会话）**：一个 MSL ML kernel（norm+matmul2d+silu 链）经
  objc2-metal 跑通，CPU 镜像对照正确，孤立基准。失败即止损。
- **M1（2-3 会话，裁决点）**：MoE 分组 int4 matmul 上协作张量 vs 现行
  moe_fused 同形状孤立 A/B。**≥1.5× → 项目继续；<1.2× → 证伪归档**。
- **M2+**：逐层融合链 + ML encoder 整网 dispatch；KV/注意力；数值门换轨。

## 来源

- 本机 `system_profiler SPDisplaysDataType`；CLT `MacOSX27.sdk` 的
  `MTL4MachineLearningCommandEncoder.h`、`MTL4MachineLearningPipeline.h`、
  `MTL4Compiler.h`、`MTLTensor.h`。
- `~/.cargo/registry/.../objc2-metal-0.3.2/src/generated/`（MTL4 符号在场）。
- [WWDC25-262 Combine Metal 4 machine learning and graphics](https://developer.apple.com/videos/play/wwdc2025/262)；
  [Running inline ML operations in a shader with Metal 4（样例）](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
  （MSL 代码经文档 JSON 提取，逐段引用在上文）；
  [WWDC26-330 Optimize custom ML operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330)；
  [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)。
- 带宽/字节锚点：本仓 VALIDATION（MoE 320MB/层带宽口径、τ/周期实测）、
  MLX 参照（用户实测 50 tok/s BF16）。


## 追记（2026-10-26 夜）：M1 边界与运行时/SDK 版本错位

M0 之后进入 M1（MoE 形状裁决），撞上并完整刻画了一个运行时缺陷：

- **约定已锁定位级正确**（对 CPU 镜像 worst=0）：A [k,m] K 内在直用、B [n,k]
  N 内在需一次转置、C [n,m] 读回即行主序；切片模板序 A dim0 必须 dynamic。
- **边界**：任何父张量维度超过分块（M/N）或 4×分块（K），`matmul2d` **静默
  零写**（不是算错）。六路假说全部实验排除：网格覆盖、静态 K 字面量、文档
  的 M/N/K 常量配方、packed strides（官方验证：rank>0 必须 strides）、设备
  分配张量 + copyFromTensor（官方验证：strides 必须 nil；同边界）、协作目标
  run() 变体。`threadExecutionWidth`=32 实测，128 线程组正确。
- **根因线索**：运行时 MPP **旧于 SDK 头文件**——SDK 文档的 `static_slice`
  在本机编译器不存在（M0 即撞）；文档配方瞄准更新版本的库。裸层探测：
  `metal_tensor`/`metal_cooperative_tensor`/`metal_numeric` 可 include，
  `metal_matmul` 不存在；而 cooperative_tensor 的构造需要 MPP 私有
  `__operand_layout`——**没有绕开 MPP 前端的公开出口**（矩阵乘真身在
  Apple 预编译 metallib 里）。
- **裁决**：M1 判据（≥1.5×/＜1.2×）无法在 macOS 27.0.1 上裁出
  （`softwareupdate --list`：无更新）。项目挂起，重启条件：带新版 MPP 运行时
  的 OS 点更新，或 Apple 修复该边界。M0 成果（crates/mtl4 管道 + 位级数值
  验证）保留，OS 更新后 M1 从 `tests/m1_moe_shapes.rs` 的边界地图直接续跑。
