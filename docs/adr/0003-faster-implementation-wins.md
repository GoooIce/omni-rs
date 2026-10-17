# 同一负载上谁更快，谁就是当前实现

实现方案不再按后端或数值格式封禁。CubeCL、Metal TensorOps、Apple Neural Engine、加载点反量化、FP8，以及其他模型上的量化，都可以参加同一项计算的比较。当前实现是该项负载上热机实测更快的那个。2026-10-17 的 ConvRot 去噪因此走 Metal `matmul2d`（`int8 × int8 → int32`）：同一茶壶请求的热机去噪从约 15.9 秒降到约 1.25–1.38 秒。`OMNI_CONVROT_TENSOROPS=0` 退回标量 CubeCL，那是一条退路。

## Considered Options

- **按后端预先封禁（[ADR-0002](0002-int8-matmul-kernels.md) 原决定，撤销）**：GPU 上只用 CubeCL，TensorOps 只留给目标就是 Apple Neural Engine 的计算，FP8 冻结，去噪必须走 int8 内核。
- **把这次测赢的 `matmul2d` 写成唯一允许的 GPU 路径（否决）**：那是另写一份禁令。
- **同一负载上择快（选定）**：方案都准入。当前路径是热机更快的那个。

## Consequences

- [ADR-0001](0001-w4a16-dequant-over-nax-int8.md) 和 ADR-0002 里的「FP8 冻结」「GPU 禁用 TensorOps」「TensorOps 仅用于 Apple Neural Engine」「必须走 int8、不得在加载点展开」「不重开其他模型的量化」不再是禁令。MiniCPM-V 现在仍加载官方 int4 W4A16，因为还没有一次测量显示别的做法在它自己的负载上更快。
- 比较发生在同一项负载、同一输入上。语音长流的实时因子、1024² 出图的 50.8 秒、256² 去噪的 1.25 秒，各记各的。
- 内核时间用来解释。换当前路径看的是该负载的热机时间。这次换路时，抽样点积与 int32 参考一致。
- 校表台的抽屉不再表示有一把冻结的刀。
