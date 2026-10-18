# MiniCPM-V-4.7 (omni-minicpm-v)

[MiniCPM-V-4.7](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) 与
[MiniCPM-V-4.7-35B-A3B-int4](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-int4) 的本地多模态理解：
单图 + 文字提示，流式文本输出，全部推理在 Rust/Burn 中完成（无 Python 运行时）。默认面向
Apple Silicon 的 Metal GPU（FP16 存储、FP32 归一化与递推状态）；`--precision f32` 提供 CPU 参考路径。

架构实现：SigLIP2 视觉塔（NaViT 变分辨率 patch、第 6 层后 2×2 窗口注意力合并、16× 混合压缩）+
Qwen3.5 风格混合解码器（每 4 层 1 层全注意力：GQA 8Q/2KV、head_dim 256、部分旋转 0.25、
sigmoid 输出门；其余 18 层为门控 delta 网络线性注意力，FP32 递推状态）+ 交错 [11,11,10]
canvas M-RoPE + ChatML 模板（24.8 万词表）。

## 下载与检查

```sh
python3 scripts/download_minicpm_v.py          # 1B：ModelScope 默认，--source huggingface 备选
python3 scripts/download_minicpm_v35b.py       # 35B-A3B-int4（22.65 GB，9 分片）
cargo run --release --manifest-path crates/minicpm-v/Cargo.toml -- doctor
```

下载脚本钉住上游 revision（ModelScope `19a6e190…`、HF `b0918c74…`），逐文件校验大小与 SHA256，
写入 `download-manifest.json`；`doctor` 校验目录完整性并做后端 matmul 探针。

上游语义（量化打包布局、kernel 行为、模型代码）先读 `.venv-minicpm` 里的权威源码
（transformers 5.19、compressed-tensors、flash-linear-attention 均已安装），
不要从行为反推——例如 int4 打包是 `pack_to_int32` 的 +8 偏移无符号 nibble，
见 `compressed_tensors/compressors/pack_quantized/helpers.py`。

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

## 公共 API（作为库使用）

- `multimodal::Multimodal::build_prompt`：单图 + 单条用户文本（无 system 消息）。
- `multimodal::Multimodal::build_prompt_multi`：完整消息列表（可带 system）+ 按序多图；
  消息中第 i 个 `Part::Image` 消费 `images[i]`，展开为 `<image_id>i</image_id>` 前缀的
  pad 占位；`enable_thinking` 透传给 ChatML 生成提示（`Some(false)` 生成空 think 块）。
- `generate::Options::presence_penalty`（默认 0.0，默认时行为逐比特不变）：对采样 history
  中已出现 token 的 logit 减去该常数（OpenAI/vLLM 语义）。注意本实现的 history **包含
  prompt token**，而 vLLM 只罚已生成 token。

## 数值对照

- 逐阶段官方转储（视觉 8 阶段 × 2 图、全注意力/线性注意力合成权重层、canvas 位置表、
  ChatML/词表快照）在单元与集成测试中固定；fixture 由 `scripts/export_minicpm_reference.py`
  离线生成（transformers 5.19，PIL 图像后端）。
- 端到端：文本 4 组、单图 3 组（含切片大图）贪心 token 流与官方 CPU fp32 逐 token 一致
  （`tests/greedy_parity.rs`、`tests/image_parity.rs`，需本地权重与
  `artifacts/minicpm-reference/`，缺失时自动跳过）；Metal FP16 抽查
  （`tests/metal_spot.rs`）在同一 fixture 上逐 token 一致。
- 35B int4：真实第 0/1 层逐 seam（conv/递推核心/门控归一化/MoE/残差）对照 official bf16 转储
  （`text::layer35b_tests`，需 35B 权重）；`greedy-35b.json` 文本与单图案例逐 token 一致。
  int4 语义按 compressed-tensors `pack_to_int32`（无符号 +8 偏移 nibble）实现，
  由合成 fixture 与运行时专家权重双重固定。
- 实测数字与复现命令见仓库根 [VALIDATION.md](../../VALIDATION.md) 的 MiniCPM-V 小节。

## 已知限制

- 单轮单图 CLI；库内 `build_prompt_multi` 已支持完整消息列表 + 按序多图，
  多轮 KV 复用与视频未实现（视觉塔与 canvas 位置表已支持多 crop 切片）。
- Metal 构建关闭了 Burn fusion（burn-cubecl-fusion 0.21 在本模型解码形状上代码生成崩溃）；
  常规 autotune matmul 保留。
- MTP 头不加载、不执行（检查点亦不含相应张量）。
- 线性注意力 prefill 为逐步递推（与官方逐层语义一致，prefill 吞吐未优化）。
- 融合 W4A16 路由由 Checkpoint 加载按 Backend 选定：裸 Cube 保持 int4 驻留、内核内反量化；
  NdArray 与 Fusion 包装在加载点反量化（NdArray 不能持有 sub-byte 张量，Fusion 上打包矩阵乘会在
  dtype 上失败）。vendor 补丁排除对块量化操作数展开崩溃的 GEMV autotune 候选（burn-cubecl），
  并在 dispatch 线程兜底候选 panic（cubecl-common），见各 vendor `LOCAL-CHANGES.md` 与 VALIDATION。
- BF16 计算通路在本机构建上损坏，全程 F16。
- MoE 前向保留每层一次路由小张量回读，专家累加全 GPU；不做中途 host 上传
  （该模式在本 burn 构建上会污染后续内核，见 VALIDATION）。
