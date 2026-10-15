# MiniCPM-V-4.7 (omni-minicpm-v)

[MiniCPM-V-4.7](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) 的本地多模态理解：单图 + 文字提示，
流式文本输出，全部推理在 Rust/Burn 中完成（无 Python 运行时）。默认面向 Apple Silicon 的
Metal GPU（FP16 存储、FP32 归一化与递推状态）；`--precision f32` 提供 CPU 参考路径。

架构实现：SigLIP2 视觉塔（NaViT 变分辨率 patch、第 6 层后 2×2 窗口注意力合并、16× 混合压缩）+
Qwen3.5 风格混合解码器（每 4 层 1 层全注意力：GQA 8Q/2KV、head_dim 256、部分旋转 0.25、
sigmoid 输出门；其余 18 层为门控 delta 网络线性注意力，FP32 递推状态）+ 交错 [11,11,10]
canvas M-RoPE + ChatML 模板（24.8 万词表）。

## 下载与检查

```sh
python3 scripts/download_minicpm_v.py          # ModelScope 默认，--source huggingface 备选
cargo run --release --manifest-path crates/minicpm-v/Cargo.toml -- doctor
```

下载脚本钉住上游 revision（ModelScope `19a6e190…`、HF `b0918c74…`），逐文件校验大小与 SHA256，
写入 `download-manifest.json`；`doctor` 校验目录完整性并做后端 matmul 探针。

## 生成

```sh
# 文本
cargo run --release --manifest-path crates/minicpm-v/Cargo.toml -- generate \
  --model models/MiniCPM-V-4.7-1B --prompt '介绍一下杭州' --greedy

# 单图（PNG/JPEG/WebP；EXIF 方向与透明通道白底合成）
cargo run --release --manifest-path crates/minicpm-v/Cargo.toml -- generate \
  --model models/MiniCPM-V-4.7-1B --image photo.jpg --prompt '描述这张图' --greedy

# 流式 JSON Lines
… --generate --prompt '…' --json
```

`--json` 每 token 输出 `{"delta","text","token_id","step_ms"}`，结束时输出
`{"event":"done","finish_reason","compute_ms","first_token_ms",…}`。`finish_reason` 为
`stop`（双 eos）/`length`（达到 `--max-new-tokens`）/`cancelled`（Ctrl-C，保留已出文本）。

采样参数与翻译模块同一语义：`--temperature`（`--greedy` 等价 0）、`--top-k`、`--top-p`、
`--repeat-penalty`、`--seed`。CPU 使用 `--no-default-features --features native,cpu -- --precision f32`。

## 数值对照

- 逐阶段官方转储（视觉 8 阶段 × 2 图、全注意力/线性注意力合成权重层、canvas 位置表、
  ChatML/词表快照）在单元与集成测试中固定；fixture 由 `scripts/export_minicpm_reference.py`
  离线生成（transformers 5.19，PIL 图像后端）。
- 端到端：文本 4 组、单图 3 组（含切片大图）贪心 token 流与官方 CPU fp32 逐 token 一致
  （`tests/greedy_parity.rs`、`tests/image_parity.rs`，需本地权重与
  `artifacts/minicpm-reference/`，缺失时自动跳过）；Metal FP16 抽查
  （`tests/metal_spot.rs`）在同一 fixture 上逐 token 一致。
- 实测数字与复现命令见仓库根 [VALIDATION.md](../../VALIDATION.md) 的 MiniCPM-V 小节。

## 已知限制

- 单轮单图；多轮 KV 复用、多图与视频未实现（视觉塔与 canvas 位置表已支持多 crop 切片）。
- Metal 构建关闭了 Burn fusion（burn-cubecl-fusion 0.21 在本模型解码形状上代码生成崩溃）；
  常规 autotune matmul 保留。
- MTP 头不加载、不执行（检查点亦不含相应张量）。
- 线性注意力 prefill 为逐步递推（与官方逐层语义一致，prefill 吞吐未优化）。
