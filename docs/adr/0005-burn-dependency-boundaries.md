# 按执行后端、训练和参考测试划分 Burn 依赖

2026-11-12：模型代码继续统一使用 Burn。NdArray 是 Burn 的 CPU 执行后端，`Autodiff<B>` 是叠加在执行后端上的自动求导；把 import 改成 `burn::…` 不会减少这两套实现。生产推理按所选设备编译，训练和参考测试显式声明额外能力，以减少无用后端和重复编译，同时保留数值对照。

## 依赖边界

- 默认推理使用所选 GPU 后端。`metal`、`webgpu` 不隐式启用 `cpu` 或 `training`。生产 CPU 路径由 `cpu` feature 显式启用；NdArray 数值参考放在 `dev-dependencies`，跨 crate 的参考测试也要通过开发依赖或测试目标的 `required-features` 请求 CPU 支持。
- 具体 Autodiff 实现由 `training` feature 启用。RSI 默认包含 `training`，两个训练 CLI 要求 `native,training`，保持日常使用方式；库与数据筛查工具可以只编译 `native,metal`。`AutodiffBackend` trait 和泛型训练 API 可以留在正常库接口中。
- 实验性 `burn-ane` 的 NdArray 回退是有理由的生产例外，保留其现有行为；Makepad 现有的 ANE 功能仍可能带入它。NdArray 本身也会传递依赖 `burn-autodiff`，因此看到包名不能推断调用方在训练。新增例外必须注明具体消费者、用途和验收方式。
- 直接引用 Burn、CubeCL 和 WGPU 时关闭默认 features，声明所需能力，并在独立 Cargo 项目间对齐兼容版本。当前分别固定为 Burn `0.21.0`、CubeCL `0.10.0`、WGPU `29.0.4`；相同用途的 features 应一致，原始 GPU、融合推理、训练和参考测试之间的差异可以保留。无意接入另一个本地模型的默认 features 也会扩大编译图，应显式转发所需能力。
- macOS 的 CubeCL WGPU 默认只请求 Metal、WGSL 及必要运行时能力。MoltenVK 通过 `cubecl-wgpu/vulkan-portability` 显式选择，SPIR-V 路径同时启用它；其它平台保留原有后端能力。

## 落地与验收

> 状态更新（2026-11-26）：下文「各模型暂时保持独立 Cargo 项目」的临时条款已由 [ADR-0006](0006-unified-cargo-workspace-and-artifact-lifecycle.md) 取代，仓库改为单一 Cargo Workspace（`resolver = "3"`）；本决定关于默认 GPU 推理、CPU/训练显式 feature 与按包依赖图检查的规则继续生效。

各模型原保持独立 Cargo 项目时的依赖约束如下：依赖或补丁变更要更新锁文件；兼容的公共库版本对齐。每个消费者都必须解析到仓库内相应的 vendored 补丁，补丁说明写在对应 `vendor/*/LOCAL-CHANGES.md`。

`./scripts/check quick` 检查清单里的版本、features、别名和平台依赖边界；`./scripts/check metal` 另外检查当前平台的正常/构建依赖图与补丁来源，再运行本机 GPU 验收。参考测试允许编译 NdArray 和 Autodiff，不能拿测试图代替生产图，也不能通过删掉或跳过数值断言来缩小依赖。后端迁移要覆盖真实生产装配、独立参考、变长输入和零 panic；原生界面受影响时运行对应 harness。远程 CI 保持停用，CPU 只作按需本地诊断。

这项决定约束默认依赖的用途，不取代 [ADR-0003](0003-faster-implementation-wins.md) 的同负载择快原则。通过验收的其它加速方案仍可显式接入。依赖图缩小只说明后续少编译哪些包；既有缓存、macOS 调试对象和多个构建目录的占用要单独测量，不能据此声称已回收磁盘。
