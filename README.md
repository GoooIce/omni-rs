# omni-rs

<p align="center">
  <img src="assets/readme/hero.svg" width="100%" alt="omni-rs：纯 Rust / Burn 实现的多模态本地推理套件，覆盖语音转写、翻译、语音合成、图像理解、文生图与音乐生成，推理全程在本机 GPU 上">
</p>

纯 Rust / [Burn 0.21.0](https://github.com/tracel-ai/burn/tree/v0.21.0) 实现的本地多模态推理套件（核心 crate `omni-r2t2`）：流式语音识别、翻译、语音合成、图像理解、文生图与图像编辑、音乐理解与歌曲生成，全部运行在这台 Mac 的 GPU 上。推理代码不调用 Python、PyTorch、llama.cpp 或远程服务；默认面向 Apple Silicon 的 Metal，另提供实验性 ANE 后端、WGPU/WGSL 原生后端和浏览器 WebGPU 接口。

每个模块都已用官方完整权重在本机 M5 Max 上完成推理，并与官方 PyTorch / transformers 结果做数值对照；测试条件、性能和范围见 [VALIDATION.md](VALIDATION.md)。

## 能力总览

| 能力 | 模型 | 入口 | 本机实测（M5 Max） |
| --- | --- | --- | --- |
| 流式语音识别 | [Confucius4-R2T2](https://modelscope.cn/models/netease-youdao/Confucius4-R2T2) | CLI · Makepad · Web | RTF 0.379，每步 p95 82.4 ms |
| 翻译 | [Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B) | CLI · Makepad · API | RadixAttention 前缀缓存 + 连续批处理 |
| 语音合成 | Breeze-TTS-2 | Makepad | 流式边生成边播放，音色设计与声音克隆 |
| 图像理解 | [MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) | CLI · Makepad | ≈26 ms/token（Metal FP16，贪心对齐官方） |
| 文生图 · 图像编辑 | Qwen-Image-2.1-Turbo | CLI · Makepad | 8 步生成，1024² ≈51 s，默认 2K |
| 音乐理解 · 歌曲生成 | MERT2 → SheetSage2 → YuE2 | Makepad · Web | 48 kHz 双声道 WAV |

## 本机实测

<p align="center">
  <img src="assets/readme/proof-board.png" width="100%" alt="Makepad 原生「图像生成」工作台截图，右侧是六张由本仓库 Rust/Burn 实现本地生成的图像：毛笔书法「知音」、扁平插画机器人、写实摄影茶壶和三张 RGBA 透明背景贴纸吉祥物，左下是 RTF、解码速度与生成耗时的实测数据">
</p>

左侧是 Makepad 原生「图像生成」工作台（单图 1024²：文本 0.73 s + 生成 50.8 s + 解码 2.5 s）；右侧生成图全部由本仓库 `crates/qwen-image` 的 Rust/Burn 实现在本机生成，没有经过任何云服务——其中三张透明贴纸用 CFG 2.5 负向引导让模型原生 alpha 干净地清空背景。提示词、seed 与实验记录见 [assets/readme/source/gen-prompts.txt](assets/readme/source/gen-prompts.txt)。

## 快速开始

需要 Rust 1.92+。如果 Cargo 不在 PATH，先加载 `~/.cargo/env`。

```sh
git clone https://github.com/GoooIce/omni-rs && cd omni-rs
python3 scripts/download_model.py         # Confucius4-R2T2，约 4.08 GB，默认 ModelScope
cargo run --release -- doctor
cargo run --release -- transcribe \
  --model models/Confucius4-R2T2 --audio speech.wav \
  --language Chinese --chunk-ms 160 --lookahead-ms 160 --warmup --realtime
```

也可以启动网页工作台或原生 Makepad 应用，转写、翻译、合成、文生图与音乐都在里面：

```sh
cargo run --release --manifest-path crates/omni-api/Cargo.toml -- --port 8848
# 打开 http://127.0.0.1:8848
python3 prototypes/makepad/build-native.py && ./target/release/r2t2-makepad
```

## 流水线

<p align="center">
  <img src="assets/readme/pipeline.svg" width="100%" alt="四条本地推理流水线：语音链（WAV 或麦克风 → Confucius4-R2T2 转写 → Hy-MT2-7B 翻译 → Breeze-TTS-2 合成 → 边生成边播放）、音乐链（录音 → MERT2 → SheetSage2 → YuE2 → 48 kHz 双声道）、图像链（提示词或参考图 → Qwen-Image-2.1-Turbo → 可透明 PNG）、理解链（图片加提问 → MiniCPM-V-4.7 → 逐 token 文字）；四条链共用同一 Rust/Burn 运行时，后端覆盖 Metal、ANE、WebGPU、CPU 与 Wasm">
</p>

## 模块

### 流式语音识别 · Confucius4-R2T2

[Confucius4-R2T2](https://modelscope.cn/models/netease-youdao/Confucius4-R2T2) 的 Rust 流式实现。输入支持 WAV（整数/浮点、多声道，取声道平均并重采样到 16 kHz）、16 kHz PCM 管道、可选麦克风和浏览器 module worker；`--realtime` 按音频时钟喂入，不加时尽快处理便于测速。

官方 6.74 秒样本实测 **RTF 0.379、每步 p95 82.4 ms**；20.22 秒长流 RTF 0.406。历史数值使用纯 Metal FP16（`--backend metal`），排除模型加载与预热。

```sh
# 输出 JSON Lines；每条含 delta、累计 text、compute_ms、audio_seconds 和 is_final
cargo run --release -- transcribe \
  --model models/Confucius4-R2T2 --audio speech.wav --json --warmup

# 管道输入格式：16 kHz、单声道、signed 16-bit little endian
ffmpeg -i speech.wav -f s16le -ac 1 -ar 16000 - | \
  cargo run --release -- transcribe --model models/Confucius4-R2T2 --stdin-pcm

# 系统默认麦克风；Ctrl-C 停止识别，保留已提交文字
cargo run --release --features microphone -- transcribe \
  --model models/Confucius4-R2T2 --microphone
```

首次使用麦克风需要操作系统授予录音权限；麦克风特性在 Linux 上需要系统 ALSA 开发库。推理跟不上输入时明确报错，不静默丢弃音频。手动停止会丢弃尚未处理的录音和重采样尾部，并在当前计算的中断检查点结束。`--warmup` 用同一 WAV 做一次不打印结果的预热，再报告 RTF 和每步耗时的 p95（RTF = 推理耗时 / 音频时长，小于 1 表示平均处理快于输入）；首次编译/调优可能明显慢于预热后的运行。

可配置项包括 `chunk_ms`（80–2000，10 ms 的倍数）、`lookahead_ms`、`language auto` 自动识别、热词/背景 `context`、token 预算与滚动音频窗口等，CLI 另暴露 `--text-postprocess legacy|official`、`--stable-tail-ms`、`--final-padding-ms`（默认保持既有解码行为）。语义见 [Rust 接口](#rust-接口)。

下载默认使用 ModelScope，可加 `--source huggingface`；下载脚本记录 SHA256，`--revision` 可固定上游版本。已有模型可直接指定本地目录，至少应包含 `config.json`、`tokenizer.json`、`model.safetensors`，也支持 `model.safetensors.index.json` 和分片权重。

### 翻译 · Hy-MT2-7B

[腾讯 Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B) 已接入 Rust/Burn。Makepad 的 ASR、TTS 页面都能直接翻译：ASR 保留原始转写并单独显示译文；TTS 点击「用译文朗读」将译文填入朗读文本。两个入口共用一份翻译模型和前缀缓存，首次翻译自动载入模型。目标语言填写完整名称，如 `English`、`Chinese`、`Japanese`。

```sh
python3 scripts/download_hy_mt.py
cargo run --release --manifest-path crates/hy-mt/Cargo.toml -- \
  --text '今天天气很好，我们一起去公园散步吧。' --target-language English --json
```

下载固定到上游 `9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223`，校验文件大小及权重 SHA256；原始权重约 16.1 GB，采用 Apache-2.0。默认 Metal FP16，支持 `--precision f32` 对照；CPU 用 `--no-default-features --features native,cpu`，通用 GPU 用 `native,webgpu`。运行时直接读取原始 safetensors，正确处理固定 α 的 NTK RoPE、旋转后的 Q/K RMSNorm、GQA 和官方结束 token。默认温度 0.7、top-k 20、top-p 0.6、重复惩罚 1.05，`--greedy` 用于可重复验证；默认最多输出 512 token、上下文 8192 token，达到输出上限会明确返回 `finish_reason: "length"`。

分页 KV 管理（`crates/hy-mt/src/pages.rs`）参考 [SGLang RadixAttention](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) 与 [PagedAttention](https://arxiv.org/abs/2309.06180) 的分页存储思路：每页 16 token，完整页面跨请求共享、写入共享尾页前复制，LRU 淘汰只释放缓存引用；Metal FP16/FP32 通过 Rust/CubeCL 内核直接读取非连续页表（支持 GQA、逐请求因果位置和分段 FP32 softmax），CPU 与 WebGPU 使用 Burn gather/attention 参考路径，不依赖 Python、SGLang 服务或 CUDA。翻译器提供 `try_submit` / `step` 连续批处理：同一轮的所有请求共用投影和 MLP 矩阵计算，各自保留位置、采样 RNG 和停止条件。默认物理池 16384 token（7B FP16 的 32 层 KV 约 2 GiB），其中前缀缓存默认最多 4096 token；`--no-cache` 关闭跨请求前缀缓存，`--cache-tokens` 调整缓存预算（二者不关闭分页注意力）。缓存不跨模型加载或进程复用，目前为单设备执行。

### 语音合成 · Breeze-TTS-2

[Breeze-TTS-2 本地流式实现](docs/breeze-tts.md)接入同一个 Makepad 界面，在「语言」工作台右栏点击「载入模型」准备语音合成。支持中文/英文、音色设计、参考声音克隆、边生成边播放、手动停止和 WAV 保存。

```sh
python3 scripts/download_breeze.py
python3 prototypes/makepad/build-native.py
./target/release/r2t2-makepad
```

「音色设计」支持描述示例、切换候选、重复试听，并可直接选用设计音色朗读新文字；也支持选择参考 WAV 并填写对应文字克隆声音。模型权重及本地输出采用研究/非商业许可。TTS 当前是原生 Metal/WGPU 入口，实时性能和实现范围见[接入说明](docs/breeze-tts.md)。

### 图像理解 · MiniCPM-V-4.7

[OpenBMB MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) 的本地多模态理解：SigLIP2 视觉塔 + 混合注意力解码器（门控 delta 网络线性注意力 + GQA 全注意力）+ canvas M-RoPE，单图 + 文字提示流式生成。贪心输出与官方 transformers ≥5.18 逐 token 对齐（文本与图像 fixture 集），Metal FP16 实测解码 ≈26 ms/token。

```sh
python3 scripts/download_minicpm_v.py
cargo run --release --manifest-path crates/minicpm-v/Cargo.toml -- generate \
  --model models/MiniCPM-V-4.7-1B --image photo.jpg --prompt '描述这张图' --greedy
```

使用方法、参数与已知限制见[模块文档](crates/minicpm-v/README.md)，数值对照与实测见 [VALIDATION.md](VALIDATION.md)。

### 文生图与图像编辑 · Qwen-Image-2.1-Turbo

[Qwen-Image-2.1-Turbo](docs/qwen-image-21.md) 通过 `crates/qwen-image` 在 Rust/Burn 0.21 内运行视觉编码器、文本编码器、扩散主干和 RGBA VAE，运行时无需 Python、Diffusers、Core ML 导出模型或网络服务。支持中文/英文描述、8 步文生图、1–4 张参考图的指令编辑、透明 PNG、尺寸/种子设置、进度、停止、预览和保存。

```sh
python3 scripts/download_qwen_image.py
./target/release/qwen-image generate --prompt "清晨的山谷，写实摄影" \
  --aspect-ratio 16:9 --output valley.png

# 原生界面：Makepad「创作」工作台右栏「载入模型」，目录缺失时可打开选择框
python3 prototypes/makepad/build-native.py && ./target/release/r2t2-makepad
```

默认 2048×2048、seed 42。CLI 与 Makepad 共用[官方七组尺寸预设](https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo#supported-aspect-ratios)（最长边 2752），如 `--aspect-ratio 16:9`；也可自定义 256–2752 的宽高（总像素不超过 4,300,800），非 32 倍数按[官方推理代码](https://github.com/huggingface/diffusers/blob/883d51ef02cd7972eeaf158c652d229f3459f419/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L640)向下对齐，不做生成后缩放。文生图最大文本 4096 token，图片编辑最多 8192 文本与视觉 token；使用模型内保存的 8 个 sigma，不提供任意步数控件。

可选的[官方提示词改写流程](docs/qwen-image-rewrite.md)：`--rewrite` 在生成前按上游 [prompt_rewrite](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite) 契约扩写提示词并决策画布比例——文生图输出 `positive_prompt` + `wh_ratio`（映射到七预设），编辑输出 `positive_prompt` + `ratio_follow`（跟随参考图比例）；显式 `--aspect-ratio/--width/--height` 一律优先。改写后端是本仓库的本地 MiniCPM-V（默认 35B-A3B-int4，约 100–190 s/次；`--rewrite-model` 可换 1B 快速迭代），官方系统提示词内嵌并带归属，严格 JSON 解析失败时回退原始回答并警告。

图片编辑：在「图像生成」页面点「参考图…」添加 PNG、JPEG 或 WebP（最多 4 张），按钮变为「编辑图像」后填写修改要求（多图可用「图1」「图2」描述关系），点击「清除」恢复文生图。「更多参数」提供负向提示词 / CFG——默认保留 8 步、CFG=1 的快速配置，负向引导需主动开启，会增加去噪计算。想要透明背景时，透明写进描述之外再开负向引导效果最好：`--negative-prompt "opaque white background, solid background" --true-cfg-scale 2.5`；纯提示词的透明输出不稳定（本机 73 个 seed 对照约 1/20 全透明，加负向引导后 3/3 全透明，见[生成记录](assets/readme/source/gen-prompts.txt)）。预览合成棋盘底色，保存的是原始 RGBA PNG。所有阶段都可停止：停止不会把未完成潜变量当作成品输出，已完成的上一张图片仍可预览和保存。

本机图像路径使用 Metal FP16，无量化；完整权重约 32.5 GB，采用 Qwen Research License。官方数值对照与实现范围见[接入说明](docs/qwen-image-21.md)；M5 TensorOps、融合注意力、条件缓存及 ANE 分区的调研与实测见 [Apple 平台优化调研](docs/qwen-image-apple-optimization.md)、[TensorOps 实施记录](docs/qwen-image-tensorops.md)和 [Rust 注意力记录](docs/qwen-image-rust-attention.md)。

### 音乐理解 · 转谱 · 歌曲生成

已接入 **MERT2 → SheetSage2 → YuE2** 本地音乐工作流：导入 WAV 后提取音乐特征、生成 ABC/MIDI，检查或编辑乐谱，再填写风格和歌词生成 **48 kHz 双声道 WAV**；也支持不提供乐谱，由 YuE2 先创作乐谱再生成歌曲。Makepad「创作」工作台的音乐区和 Web 页面均提供这条流水线，Makepad 直接调用 Rust/Burn 音乐模块，无需启动 Web 服务。

```sh
uv run --with huggingface-hub==0.36.2 python scripts/download_music.py   # 四份权重约 10.55 GB
cargo run --release --manifest-path crates/omni-api/Cargo.toml -- --port 8848
```

下载的是用户提供的 YuE 仓库当前版本 YuE2，以及对应的 MERT2、SheetSage2 和必需的音频 VAE；固定版本、校验大小和 SHA256。MERT 输出特征向量；SheetSage2 从录音转谱；YuE2 接收 ABC、风格和歌词生成音频。转谱最长 5 分钟，生成长度上限会明确标记截断。Makepad 支持试听、停止、ABC/MIDI/JSON/WAV 导出。音乐模型默认位于 `models/`，可展开「参考音乐与乐谱」选择父目录，或设置 `MUSIC_MODELS_DIR`。使用方法、原生推理阶段、检查结果与权重许可见 [音乐模块文档](crates/omni-music/README.md)。

## 本地 Web API

```sh
cargo run --release --manifest-path crates/omni-api/Cargo.toml -- --port 8848
# 打开 http://127.0.0.1:8848
```

网页提供 ASR → 翻译、文本翻译、TTS 文本 → 翻译 → 合成，以及文生图和音乐工作流，计算使用本机 Rust/Burn 服务。服务默认监听本机回环地址，模型按需载入；使用 `--asr-model`、`--tts-model`、`--translation-model`、`--image-model` 指定目录，`--music-models` 指定四份音乐模型的父目录。原有 `web/demo.html` 和 Makepad WebGPU 入口继续提供浏览器内 ASR。

| 请求 | 输入 |
| --- | --- |
| `POST /v1/translations` | JSON：`text`、`target_language`，可选 `options` 对应翻译采样设置 |
| `POST /v1/audio/transcriptions` | 请求体为 WAV 字节，`Content-Type: audio/wav`；可选查询参数 `language`、`context` |
| `POST /v1/audio/speech` | JSON：`text`，可选 `instruction`、`seed`、`max_frames`、`guidance_scale` |
| `POST /v1/images/generations` | JSON：`prompt`，可选 `width`、`height`、`seed`、`negative_prompt`、`true_cfg_scale` |
| `POST /v1/music/analyses` | WAV 字节，返回 MERT2 音乐特征 |
| `POST /v1/music/scores` | WAV 字节，可选 `melody_only=true/false`；返回音乐事件、MIDI 与有效 ABC |
| `POST /v1/music/generations` | JSON：`style`、`lyrics`，可选 `abc`、`cot`、`seed` 和生成限额 |
| `GET /v1/jobs/{id}` | 查询 `queued/running/done/cancelled/error`、进度和结果 |
| `DELETE /v1/jobs/{id}` | 取消指定任务，等待终态后读取已完成的部分结果 |
| `GET /v1/files/{name}` | 下载已发布的 WAV、PNG、音乐 JSON、ABC 或 MIDI |
| `GET /v1/models`、`GET /health` | 支持的模型和服务状态 |

```sh
curl -s http://127.0.0.1:8848/v1/translations \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，世界！","target_language":"English"}'
# 返回 HTTP 202 和任务 id；用 GET /v1/jobs/<id> 读取译文和 cached_tokens
```

推理共用 FIFO 调度：同一模型的请求可连续合批（翻译、ASR、TTS 默认各最多 4 个活动请求，图像 1–4），不同模型之间按 FIFO 等待。取消逐请求生效，在当前轮 GPU 计算完成后处理，同批其他请求继续执行。音频和图片编码完成并原子发布后，任务才进入终态；取消时可保留已生成的部分音频，真实错误仍返回 `error`。生成文件保存在 `artifacts/api-output/published`（可用 `--output` 更改根目录），结果 URL 与任务记录分开：只要保留文件并使用同一输出根目录，历史淘汰或服务重启后仍可下载。保留最近 32 个任务，活动任务不会被淘汰。Ctrl-C 会关闭接收、取消运行和排队请求，并等待工作线程清理模型后退出。

翻译用 `--max-batch-size`、`--max-step-tokens`、`--kv-pool-tokens` 调整并发、轮次预算和物理 KV 池；ASR、TTS 分别用 `--asr-batch-size`、`--tts-batch-size`（1–16），图像用 `--image-batch-size`（1–4，默认 1）。等待通道长度为 4，通道满返回 503。逐请求计时包含轮间等待，不能相加作为 GPU 独占计算时间。

HTTP 接口当前提供 WAV 转写、无参考音频的 TTS 和文生图；原生界面保留麦克风、声音克隆及参考图编辑功能。ASR 的 `StreamConfig::paged_attention` 和 TTS 的 `SynthesisOptions::paged_attention` 提供可选 Metal 分页历史（HTTP 对应 `--asr-paged-attention`、`--tts-paged-attention`，默认关闭）：复用 `omni-kernel` 的分页读写内核，独立持有 KV，支持增长、ASR 回退和延迟 GPU 读回，不复用翻译的文本前缀缓存。基准与默认策略见 [VALIDATION.md](VALIDATION.md)。

## 运行后端

当前机器为 M5 Max（40 核 GPU、128 GB 内存）。默认使用 Burn 的 `metal` 特性：CubeCL/MSL、GPU FP16 参数和矩阵计算、FP32 归一化及残差累加、算子融合与 Burn attention。`--precision f32` 用于精度对照；GPU 若不支持 FP16，加载前会返回精度不支持错误。默认配置不会硬编码其他平台的 GPU 设备名称。

### Apple Neural Engine（实验）

通用 [`burn-ane`](crates/burn-ane/README.md) 后端接入 Burn 0.21 的 Backend/Router 算子中间表示：Rust 自动把支持的子图生成 Core ML 模型，Swift 桥接负责编译、检查设备计划和执行。普通 Burn Tensor / `nn::Linear` / 卷积模型可以切换后端，无需模型专用 Python 导出或 `.mlpackage`。需要 Apple Silicon、macOS 15+、Rust 1.92+ 和 Xcode Command Line Tools；权重与 ANE 计算使用 FP16，不保证与 FP32 逐位一致。

```sh
# 普通 Burn Linear + ReLU；检查 ANE 调度、与 NdArray 的数值误差、编译缓存
cargo run --release --features ane --example ane_backend

# ANE 后端 + Metal 回退，验证 FP16 矩阵乘法
cargo run --release --features ane -- --backend ane doctor

# 完整模型仍走同一份泛型 Burn 实现。属于实验路径，首次子图编译较慢
cargo run --release --features ane -- --backend ane transcribe \
  --model models/Confucius4-R2T2 --audio speech.wav --json

# 独立后端的算子、缓存、权重替换、设备切换、生命周期与错误处理测试
cargo test --manifest-path crates/burn-ane/Cargo.toml
```

支持的子图检查到至少一个矩阵乘法或卷积算子由 Core ML 分配给 ANE 才会执行；无 ANE 分配、未支持的算子或编译失败会按原因回退到 `B`，`FallbackPolicy::Error` 可禁止 Burn 回退。提供 FFN/Attention 大图融合、1×1 卷积投影、安全的长度补齐、权重视图折叠、主机张量驻留，以及有容量限制的内存/磁盘缓存。CLI 编译缓存默认在 `~/Library/Caches/burn-ane`；`--ane-trace` 输出分阶段计时及设备计划（系统调度依据，不是硬件计数器测量），`--ane-strict` 禁止回退。该后端仍用于推理原型：直接 ANE 路径执行主要矩阵计算，embedding 索引、argmax 和部分布局操作仍需回退。

2026-10-11 按优化目标收尾，选用 `infinite-mask-source` / `infinite-mask-bin`，保留卷积块复用、资源恢复及负无穷注意力遮罩修复：修复 sigmoid 数值误差后 logits 相对 RMSE 从 5.271% 降至 0.197%；固定 96 条中文集热态一轮 **RTF 0.7708、p95 173.38 ms、CER 3.1052%**（Metal 3.0346%，差 +0.0706 个百分点），96 条正文全部相同。但 LRU 策略 40 秒连续输入 RTF 0.9951、30 分钟复测 RTF ≈1.017、结束积压 33–43 秒，**长流性能验收未通过**。原生界面默认继续使用经过验证的 Metal，Web 使用 WebGPU。版本选择、验收标准与各版本对照见[优化目标](docs/ane-optimization-goal.md)、[验证报告](VALIDATION.md)，原始结果在 `artifacts/burn-ane-goal/`（历史结果在 `artifacts/burn-ane-fp16/` 和 `artifacts/burn-ane-kv/`）。`--ane-runtime direct` 可验证同一 Burn 模型通过 MIL 和私有 ANE 接口直接执行，见[后端说明](crates/burn-ane/README.md#直接-ane-运行时)与[技术调研](docs/ane-backend-research.md)。

在其他 Rust 可执行项目使用此后端时，主程序 `build.rs` 需要在 macOS 输出 `cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift`；本仓库已配置。之前的模型专用 Core ML 导出实验保留在 `scripts/export_coreml_encoder.py`、`coreml` feature 及 `verify_coreml` 示例中，仅供对照，默认不编译、不自动检测导出。

### WebGPU / CPU / 浏览器

```sh
# WGPU/WGSL：Windows/Linux 等原生平台由 wgpu 选择可用的图形 API
cargo run --release --no-default-features --features native,webgpu -- \
  --backend webgpu transcribe --model models/Confucius4-R2T2 --audio speech.wav

# FP32 CPU 对照；用于正确性测试，不用于承诺实时性能
cargo run --release --no-default-features --features native,cpu -- \
  --backend cpu transcribe --model models/Confucius4-R2T2 --audio speech.wav
```

浏览器使用异步 GPU 初始化和回读，权重按块上传，避免完整检查点占用 Wasm 线性内存；分片加载会校验长度与 SHA256。Wasm release 构建及 Chromium 中加载完整权重、连续两次识别已实测：

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
cargo build --release --target wasm32-unknown-unknown --no-default-features --features wasm
wasm-bindgen target/wasm32-unknown-unknown/release/omni_r2t2.wasm --target web --out-dir pkg
python3 scripts/pack_web.py models/Confucius4-R2T2 models/web
```

将 `pkg/`、`web/load-model.js` 和 `models/web/` 放在可通过 HTTPS 或 localhost 访问的同源路径。本仓库附带可直接选择音频文件的演示页：在仓库根目录运行 `python3 -m http.server 8765`，打开 `http://localhost:8765/web/demo.html`，加载模型后开始识别。在 module worker 中使用 [web/load-model.js](web/load-model.js)：

```js
import { loadModel } from "./web/load-model.js";
const model = await loadModel("/models/web/manifest.json");
const session = model.createTranscription(
  JSON.stringify({ language: "Chinese", chunk_ms: 160 }), 48000,
);
for await (const pcm of mono48kSource) {
  for (const event of await session.push(pcm)) postMessage(event);
}
for (const event of await session.finishEvents()) postMessage(event);
session.free();
model.free();
```

输入是创建转写时指定采样率的单声道 `Float32Array`；每次 `push` 必须 await。Rust runner 负责重采样和尾音刷新，事件包含累计 `total_compute_ms`、`event_count` 和 `rtf`。原有 `createSession(config)` 仍接受 16 kHz PCM，`finish()` 保留单个终态事件的兼容接口。浏览器仍需足够容纳模型的 GPU 内存，FP16 需支持 `shader-f16`，对应传入 `loadModel(url, { precision: "f32" })`。WebGPU 的默认 160 ms 配置在本机仍慢于实时；[Makepad 原型](prototypes/makepad/README.md)增加了默认 960 ms 的 Web 快速档，受控 20 秒采集测试 RTF 约 0.75，文字更新间隔更长。Windows/Linux 硬件未实测。

## Rust 接口

```rust,ignore
use std::sync::Arc;
use omni_r2t2::{Model, Precision, StreamConfig, Transcription};
use burn::backend::wgpu::{Metal, WgpuDevice, graphics, init_setup};

let device = WgpuDevice::default();
init_setup::<graphics::Metal>(&device, Default::default());
let model = Arc::new(Model::<Metal>::from_directory(
    "models/Confucius4-R2T2", Precision::F16, &device,
)?);
let mut run = Transcription::new(model, StreamConfig::default(), 48000)?;
for event in run.push(&mono48k).await? {
    print!("{}", event.event.delta);
}
for event in run.finish().await? {
    print!("{}", event.event.delta);
}
let statistics = run.statistics();
```

一个模型可供多个会话共享；PCM、音频缓存和 KV 缓存属于各自会话，会话内串行调用。`Transcription` 统一采样率转换、结束/打断、增量一致性校验及统计；原生 `input::transcribe` 通过 `WavSource`、`PcmSource` 或 `Microphone` 驱动同一 runner，支持实时节拍和事件回调。需要直接控制 16 kHz 解码时仍可使用 `Session`。`delta` 只包含新提交的文本，拼接全部 `delta` 等于最终 `text`；`finish` 即使恰好位于完整分块边界也会解码一次，避免丢掉尚未提交的尾部。重复结束或结束后输入返回错误；输入样本应是归一化 PCM，NaN/Inf 会被拒绝；推理出错后应丢弃当前会话。

手动打断与自然结束分开处理：先用 `Session::stop_handle()` 得到可跨线程置位的原子标志，在任意时刻写入 `true`；正在执行的 `push` / `finish` 会在 GPU 回读边界退出，`push` 若已提交部分块则返回这些事件，否则返回 `Error::Interrupted`。等异步调用退出后调用 `session.interrupt()`，得到保留已提交文本、不解码积压的最终事件。Web 对应 `session.stopHandle().stop()`；取消句柄应事先保存，且必须等异步调用返回后再 `interrupt()` / `free()`。正常音源结束调用 runner 的 `finish()`（Web 为 `finishEvents()`）刷新尾音，并消费全部返回事件。

`Model::load` 接受 `WeightSource`，字段不再公开可写；`config()`、`precision()` 提供元数据，`diagnostics()` 为数值验证提供只读访问。

## 实现要点

- ASR、Breeze TTS 和 Qwen Image 共用 `tensor` 模块的 Linear、Norm、GELU、注意力、RoPE、Embedding 和 KV 缓存；数值差异由显式策略保留，包括 ASR 的 FP16 余量及不同归一化舍入顺序。
- `weights::CheckpointLayout` 集中张量名称、形状和加载阶段，内存 fixture 与生产加载使用同一接口。
- `crates/omni-kernel` 统一自定义 GPU 核的 Fusion 注册、后端检查和 ANE 回退交接；核体与模型形状守卫留在各模型中，Qwen 的基准程序从公开 `kernels` 接口调用实际生产核。
- Makepad 的 `engine_harness` 统一命令互锁、加载、取消、通知及 panic 恢复；恢复时清理模型状态，允许重新加载。
- 400 点周期 Hann STFT、160 sample hop、128-bin Slaney log-mel；FFT 和重采样用 Rust 实现。非整 hop 的最终尾部补不足 10 ms 的零，短至一个样本的输入也可以处理。
- 官方音频网络：3 层二维 stride-2 卷积，按 100 个 mel 帧分块，正弦位置编码，24 层 1024 维双向 Transformer，投影到 2048 维；卷积尾块遵循整段输入的 padding 宽度。
- 官方文本网络：28 层、2048 维、16 个 query heads / 8 个 KV heads、head_dim 128、Q/K RMSNorm、RoPE 和 SwiGLU；支持 tied/untied embedding，按名称与形状检查 Safetensors。
- GELU 使用 FP32 erf 近似，残差使用 FP32 累加，以避开 Burn 0.21 Metal 的已复现融合数值问题；不引入中间张量的 CPU 回读，保留 FP16 主矩阵计算和算子融合。
- 8 秒音频注意力窗口完全封闭、且 STFT 不再受未来输入影响后才缓存；log-mel 的全局峰值变化会使缓存失效，新的音频会改变尚未封闭窗口的旧特征。
- 文本解码只保留不变音频前缀前面的 KV，变动音频之后的文本 KV 每次重建；自回归生成复用 KV，prefill 分块，词表仅投影最后一个位置，每 token 回读一个整数。
- 默认保留 16 秒 PCM，每次移除最早的 8 秒，同时按提交音频时刻移除对应提示词，已发出的文本继续保留；沿用上游按块关联文本的思路，不提供逐词时间对齐。
- 待提交 token 做 UTF-8 安全回退；`|` 表示本轮等待边界；已提交前缀不回写。

当前范围是单路会话的贪心解码，可通过 API 建立多个会话；没有 VAD 自动分句、词级时间戳、量化权重或 WebSocket 服务。使用麦克风时应由调用方或用户确定一句话的结束。

## 验证

统一入口会检查所有独立 Cargo 项目，并按步骤保存可恢复的日志：

```sh
./scripts/check quick   # 约束、验证器反例和所有子项目格式
./scripts/check cpu     # 再运行 CPU Clippy 与测试
./scripts/check metal   # Apple Silicon 上的 Metal 与独立注意力数值对照
./scripts/check native  # 本机权重与桌面环境下的 ASR、生命周期和 TTS 回归
./scripts/check status  # 查看结果与恢复命令
```

检查范围、环境要求和中断恢复见 [开发工作流](docs/agent-workflow.md)。下面保留平台与参考模型的专项检查。

```sh
cargo fmt --all -- --check
cargo test
cargo test --no-default-features --features native,cpu
cargo check --no-default-features --features native,webgpu
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
cargo run --release -- doctor
```

无 Cargo 依赖时也能运行音频前端测试：

```sh
rustc --edition 2024 --test src/audio.rs -O -o /tmp/r2t2-audio-tests
/tmp/r2t2-audio-tests
rustc --edition 2024 -O examples/frontend_dump.rs -o /tmp/r2t2-frontend-dump
python3 scripts/verify_frontend.py /tmp/r2t2-frontend-dump
python3 scripts/test_pack_web.py
```

`tests/model_parity.rs` 包含小模型 KV 追加/回退对照、音频缓存失效、卷积边界、浏览器参数分片、任意 PCM 分包和尾部刷新测试。需要独立核对时，在 Python 3.12+ 的开发环境运行 `python3 -m pip install -r scripts/requirements-reference.txt`，并将官方 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 源码根目录加入 `PYTHONPATH`（推理本身不需要这些依赖）：

```sh
python3 scripts/export_reference.py artifacts/reference
cargo run --release --no-default-features --features native,cpu \
  --example verify_reference -- artifacts/reference
cargo run --release --example verify_reference -- artifacts/reference f32
python3 scripts/export_reference.py artifacts/reference-f16 --precision f16
cargo run --release --example verify_reference -- artifacts/reference-f16 f16
```

参考导出使用官方类与随机小尺寸权重，对照 mel、编码器输出、prefill logits 和 3 步缓存解码 logits；CPU eager 参考显式加入与官方 FlashAttention varlen 相同的块掩码。真实权重的官方转写对照：

```sh
python3 scripts/download_model.py --reference-files
python3 scripts/transcribe_reference.py models/Confucius4-R2T2 speech.wav
```

FP32/FP16、真实语音、长流缓存一致性和浏览器结果已记录在 [VALIDATION.md](VALIDATION.md)。这些是回归与可运行性验证，尚未进行完整语料集的 WER/CER 评测。Makepad 本地界面、运行命令及选型验证见 [prototypes/makepad](prototypes/makepad/README.md)；CPU、流式调度和官方 Python 后处理对照见[本地流程优化](docs/native-pipeline-optimization.md)。

## 许可

本仓库 Rust 代码使用 [Apache-2.0](LICENSE)；模型权重遵循其官方模型许可证，未包含在仓库中——Hy-MT2-7B 为 Apache-2.0，Breeze-TTS-2 权重及本地输出为研究/非商业许可，Qwen-Image-2.1 为 Qwen Research License，其余模型见各模块文档与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

源码参考：[R2T2 流式实现](https://github.com/netease-youdao/Confucius4-R2T2/blob/master/r2t2/r2t2_asr.py)、[Qwen3-ASR 网络](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)、[官方模型配置](https://huggingface.co/netease-youdao/Confucius4-R2T2/blob/main/config.json)。
