# 其他模型的量化检查点

本仓库不把 MiniCPM-V 35B 的 W4A16 融合反量化推广到 Hy-MT2-7B、Confucius4-R2T2、Breeze-TTS-2、MERT-v2-FullSong、SheetSage2、YuE2-3B 和 YuE2-Vae。Qwen-Image-2.1-Turbo 的 GGUF 与 FP8 也不做。

Qwen-Image-2.1-Turbo 去噪器的 Unsloth INT8 不在这份拒绝里。

## 为什么不做

2026-10-17 在 ModelScope 上按固定顺序查过：官方量化优先，没有再找 AWQ，然后才接受 Unsloth 或 NVIDIA 发布的量化。这些来源都没有，就不自行把全精度权重量化。

现有加速只吃一种检查点：compressed-tensors 的对称、group 128、打包 int4，加载成 Burn 的 Q4S，权重驻留，matmul 在内核里反量化成 FP16。MiniCPM-V 35B 的官方 int4 已经是这条路径。

查到的结果对不上这条路径：

- Hy-MT2-7B 有官方 FP8（`quant_algo: FP8`）和官方 GGUF（Q4_K_M / Q6_K / Q8_0，依赖腾讯 STQ 内核）。没有 AWQ。`FlagRelease/HY-MT2-7B-nvidia-FlagOS` 与全精度同体积，发布者也不是 NVIDIA。
- Confucius4-R2T2 有官方 GGUF（Q4_K_M、Q8_0、f16，另有 mmproj）。没有 AWQ、FP8、Unsloth 或 NVIDIA 量化。
- Qwen-Image-2.1-Turbo 的官方仓库只有全精度。没有 AWQ，也没有 NVIDIA 组织的量化。Unsloth 另有 GGUF 和 FP8；这两份不做。
- Breeze-TTS-2、MERT-v2-FullSong、SheetSage2、YuE2-3B、YuE2-Vae 在上述来源里没有量化检查点。社区 GGUF 和 MLX 量化不算。

GGUF 的 k-quant 和 FP8 都进不了现在的 Q4S 融合 matmul。这些数值格式不再被封禁；要换上它们，得是该模型自己的负载上测得更快。[ADR-0003](../docs/adr/0003-faster-implementation-wins.md) 记了这条。下面这些模型目前没有这样的测量，也没有被接受的量化检查点。

## 先前请求

- #18：把 MiniCPM-V 的 int4 量化加速推广到其他模型。决定只保留 Qwen-Image-2.1-Turbo 的 Unsloth INT8，其余记录不做。
