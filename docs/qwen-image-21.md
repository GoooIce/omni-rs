# Qwen-Image-2.1-Turbo 本地接入

通过 `crates/qwen-image` 在 Rust/Burn 0.21 内运行视觉编码器、文本编码器、扩散主干和 RGBA VAE。
Makepad 新增“图像生成”页面：自动加载、进度、中文/英文描述、尺寸、种子、停止、预览及 PNG 保存。
运行时无需 Python、Diffusers、Core ML 导出模型或网络服务。Python 只用于下载与开发阶段的官方对照。

已接通文生图与文字指令驱动的图片编辑，支持 1–4 张参考图片、透明通道和多图条件。
保留其他原生平台的 WebGPU 后端编译入口，本次验收平台为 M5 Max / macOS，未新增浏览器界面。

## 运行

```sh
python3 scripts/download_qwen_image.py
python3 prototypes/makepad/build-native.py
./target/release/r2t2-makepad
```

右上角选择“图像生成”，首次自动加载 `models/Qwen-Image-2.1-Turbo`。本机已下载完整权重。
模型不存在时弹出文件夹选择框；请选择包含 `model_index.json`、`text_encoder/`、`transformer/`、
`vae/`、`processor/` 和 `scheduler/` 的根目录。成功加载后记住路径；取消选择后可以重试。

默认 2048×2048、seed 42。Makepad 与 CLI 共用
[官方七组尺寸预设](https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo#supported-aspect-ratios)：

| 比例 | 宽 × 高 |
| --- | --- |
| 1:1 | 2048 × 2048 |
| 4:3 | 2400 × 1792 |
| 3:4 | 1792 × 2400 |
| 3:2 | 2528 × 1696 |
| 2:3 | 1696 × 2528 |
| 16:9 | 2752 × 1536 |
| 9:16 | 1536 × 2752 |

这些是官方给定的具体像素值，不按名义比例重新计算。CLI 可用 `--aspect-ratio 16:9`，
不能同时传 `--width` / `--height`。也可以继续手填小尺寸以缩短生成时间。
本地自定义宽高范围为 256–2752，实际输出总像素不超过 4,300,800。
这两个资源限制分别取官方预设的最长边和最大面积，不代表模型架构的硬上限。
非 32 倍数按照
[官方推理代码](https://github.com/huggingface/diffusers/blob/883d51ef02cd7972eeaf158c652d229f3459f419/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L640)
向下对齐。例如官方快速开始的 1680×2512 实际输出为 1664×2496。
界面输入时提示实际尺寸，点击生成后回填实际宽高；CLI 输出对齐提示。
模型分配、预览、PNG 和结果报告均使用对齐后的尺寸，不做生成后的缩放。
大图的 VAE 通道归一化按空间行分块，将每块 FP32 临时张量限制在 2 GiB 内。
每个像素仍按原公式在通道维归一化，输入输出保持 FP16，归一化保留 FP32 计算。
这避免官方 2K 尺寸下整张特征图转成 FP32 时超过当前 CubeCL/Metal 运行时的单块缓冲上限。

```sh
./target/release/qwen-image generate --prompt "清晨的山谷，写实摄影" \
  --aspect-ratio 16:9 --output valley.png
```

2026-10-13 尺寸验收：11 项 CPU 测试通过，包括归一化分块与原公式的逐像素对照；
七个原生预设按钮与在线官方模型卡逐项一致。已实际生成 CLI 的 2528×1696 和
Makepad 的 2752×1536 RGBA PNG，并检查原生预览与保存。
513×543 请求输出 512×512，CLI 与界面结果均和此前 512² 结果逐字节一致。
原生负向提示词、双参考图编辑、取消恢复和切回 ASR 的回归通过；
默认图、CFG 图和编辑图与上一版逐字节一致，取消恢复约 0.054 秒。
大图日志须无 GPU worker panic；首次触发缓冲分配失败的输出已作废并保留在
`artifacts/qwen-image-sizes/first-attempt/`，不作为验收结果。
所有尺寸参数均经过检查，但完整 2K 出图只覆盖上述两组；本轮不作性能对比。
报告、日志、截图与源码快照位于 `artifacts/qwen-image-sizes/`。

使用模型内保存的 8 个 sigma，不提供任意步数控件。文生图最大文本长度 4096 token；
图片编辑最多 8192 个文本与视觉 token。
输入框在生成时仍可编辑，修改影响下一次请求。GPU 工作在线程内运行；加载、文本层、生成层、
采样步和 VAE 阶段均检查停止。当前 GPU 内核与首次算子编译需完成后才退出检查点。
停止不把未完成潜变量当作成品输出，已完成的上一张图片仍可预览和保存。

透明背景写进描述，例如 `isolated character, transparent background`。预览会合成棋盘底色，
保存的是原始 RGBA PNG，alpha 不被底色覆盖；是否生成透明背景仍取决于模型输出。
PNG 在后台完整写入临时文件后再替换目标文件。

### 图片编辑

在同一页面点击“选择参考图…”，选择 PNG、JPEG 或 WebP。可同时选择最多 4 张；
缩略图显示第一张，旁边显示数量。选择后按钮变成“编辑图像”，在输入框填写修改要求即可，
例如“把茶壶改成蓝色，保持木桌、光照和构图不变”。多图时可以用“图1”“图2”描述图片之间的关系，
顺序为原生文件选择器返回的顺序。点击“清除”恢复文生图，之前的结果仍可保存。

参考图首次使用时才加载视觉塔，并显示可停止的进度；后续复用已加载的权重。
读取图片和制作缩略图在工作线程完成，支持中文路径、JPEG EXIF 方向、RGBA 透明通道；
选择损坏文件或取消选择不会丢失之前的参考图。生成中锁定参考图选择，文字和种子修改用于下一次请求。
保存时拒绝覆盖本次生成所用的参考原图，包括通过符号链接指向同一文件的情况。

界面把每张参考图的目标面积设为输出画布面积，限制在约 288²–1024²，再保持纵横比四舍五入到 32 的倍数。
因此把输出调为 512×512 也会降低参考图的处理成本。“更多参数”可单独填写参考图分辨率，
或保留“自动”。CLI 可用 `--reference-resolution` 单独控制此面积；
默认 1024，与官方 `output_resolution=1024` 的参考图缩放策略相同。可选 256–1024 内 32 的倍数。
界面使用 288² 下限，避免小幅长宽取整让普通非正方形图片落到 VL 处理器的 256² 最小面积之下。
解码前限制边长 16384、解码分配 256 MiB；处理后的参考边长不超过 4096，极端纵横比给出明确错误。
采用参考条件生成完整的新图；当前没有局部蒙版绘制、笔刷或强制像素锁定，细节保留程度取决于模型。

设置文件：`~/Library/Application Support/omni-r2t2/qwen-image.json`。
`QWEN_IMAGE_MODEL_DIR`、`QWEN_IMAGE_SETTINGS_PATH` 分别覆盖模型与设置目录。
ASR、TTS、图像工作线程共享同一个 Burn GPU 设备注册；任务运行时禁用页面切换。
图像使用 Metal FP16，不量化、不改动已验收的语音 ANE/Metal 策略。
M5 的 DiT 大矩阵默认使用 TensorOps；兼容回退、性能和数值验收见 [实施记录](qwen-image-tensorops.md)。
DiT 注意力使用 Rust/CubeCL FlashAttention 与 M5 TensorOps，范围及回退开关见 [Rust 注意力记录](qwen-image-rust-attention.md)。

独立 CLI：

```sh
python3 scripts/build_qwen_image.py
./target/release/qwen-image generate \
  --prompt '一只红色陶瓷茶壶放在木桌上，清晨的柔和光线，写实摄影。' \
  --width 1024 --height 1024 --seed 42 --output teapot.png

./target/release/qwen-image generate \
  --image teapot.png \
  --prompt '把茶壶改成蓝色，保持木桌、光照和构图不变。' \
  --reference-resolution 512 --width 512 --height 512 --seed 42 --output blue-teapot.png
```

重复传入 `--image` 可进行多图编辑；输出路径必须与参考图片不同。

### 负向提示词与更多参数

点击“更多参数”，开启“负向引导”，填写不希望出现的内容并设置强度（大于 1、最多 20）。
留空负向提示词时使用官方的无条件分支。默认关闭，对应 Turbo 推荐的 CFG=1；
开启时初始强度 2.0 供试验，每步多算一次 DiT，会增加耗时，也可能改变画面风格。
收起参数后仍显示开关状态，输入以点击生成时的快照为准。

```sh
./target/release/qwen-image generate \
  --prompt '一只红色陶瓷茶壶放在木桌上，清晨的柔和光线，写实摄影。' \
  --negative-prompt '文字、水印、模糊' --true-cfg-scale 2 \
  --width 512 --height 512 --seed 42 --output guided.png
```

负向词与强度须配合；CLI 会提示未生效的组合。正负分支使用独立文本特征与 KV cache，
图片编辑时共用视觉塔特征和 VAE 参考 latent；文生图、单图及多图都支持。
提供七组官方比例尺寸和“换种子”，随机种子会显示在输入框中以便复用。
详细调研、行为及验证见 [Metal / ANE 优化与参数说明](qwen-image-apple-optimization.md)。

`Ctrl-C` 请求停止；不会保存未完成图像。`--f32` 是数值诊断模式，正常使用 FP16 即可。
非 macOS 原生构建使用 `--no-default-features --features webgpu`，运行时传 `--backend webgpu`。
权重和中间张量较大，需要足够 GPU/统一内存；本机验收为 128 GB M5 Max。
首轮包含按形状编译和调优 GPU 算子的成本，耗时与后续运行有差异。

## 实现与上游对齐

| 阶段 | 实现 |
|---|---|
| 视觉 | Qwen3-VL 27 层视觉塔，1152 维、16 heads，16×16 patch、双帧复制、2×2 合并；第 8/16/24 层的 DeepStack 输出注入文本层 0/1/2 |
| 图片预处理 | 匹配 Pillow 的 RGBA 预乘 alpha Lanczos 与双遍整型舍入；视觉输入在白底合成，VAE 保留原四通道；参考图潜变量取高斯均值 |
| 文本 | Qwen3-VL 文本主干，36 层、4096 维、32 Q / 8 KV heads；图片占位符替换为视觉特征，使用三轴 interleaved mRoPE |
| 文本特征 | 官方原始模板，去除 14 个系统 token，取最后一层 **最终 RMSNorm 之前** 的输出 |
| DiT | 32 层、4096 维，单流文本/图像，零中心 RMSNorm、AdaLN、SwiGLU、三轴 RoPE |
| 注意力缓存 | 文本与参考图前缀使用零时刻调制；每个参考图块内部双向、文字因果且可读之前所有块；首步缓存每层 K/V，其余 7 步只计算目标图像查询 |
| 调度 | FlowMatch Euler；sigma：1、0.978453、0.95418、0.926626、0.89508、0.845148、0.704534、0.414568，最后降至 0 |
| VAE | 64 维潜变量、16 倍空间缩放、RGBA 四通道，包含单帧时间残差 shortcut 的准确通道重排 |
| 精度 | FP16 权重与矩阵计算，归一化统计、Euler 更新及 DiT RoPE 使用 FP32 中间计算 |
| 权重 | safetensors 只读 mmap；词表分片上传；支持 Diffusers 的单文件和分片文件名 |

种子采用稳定的每请求 SplitMix64 + Box–Muller，不依赖前一请求的 RNG 状态。
与 PyTorch 的 RNG 不同，相同数字种子不保证跨框架相同图像；数值对照显式提供相同初始噪声。

Burn 0.21 在变长词表分片 gather/cat 紧接首层 RMS 归约时会触发融合索引异常。
本实现只在拼接后同步一次，保留后续 Transformer 算子融合，没有修改语音后端或关闭全局融合。

编辑验收还发现 Burn/CubeK 0.21 注意力调优键会把不同序列长度归入同一桶，而部分候选内核要求
查询行数整除 stage 大小。在参考图从 512² 切换到 288² 时，324 行查询复用之前缓存的内核会触发
`Stage seq_q must divide problem seq_q`。图像模块对有因果 mask 或查询行数未对齐 256 的情况，
显式使用 GPU 矩阵乘法和 softmax；对齐图片块保留融合注意力。查询、键值和掩码均不改形状，
没有转到 CPU，也没有修改共享语音后端或全局调优缓存。单元测试对照独立标量 softmax 检查
普通注意力、GQA 分组以及带偏移的因果掩码。

曾验证过补齐查询行的方案：CPU 等价测试通过，但原生 GPU 的双参考图 DiT 缓存数值回归失败，
因此没有交付该方案。`ui-delivery/` 保留最初尺寸崩溃记录，`ui-validated/` 是中间补齐方案的界面结果；
最终交付以 `delivery.json` 所列报告和二进制校验值为准。

## 数值验证

固定模型版本：ModelScope `7015dbe91fd9c8efcb95fdf5b784b59349eca6d7`；
Hugging Face 镜像 `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`。
下载脚本逐文件校验 SHA256，支持 `.part` 续传，全部通过才写完成清单。
Diffusers 固定提交 `883d51ef02cd7972eeaf158c652d229f3459f419`，Transformers 5.17.0，PyTorch 2.9.1。
Python 参考环境为独立的 `.venv-qwen-image`，没有修改 ASR/TTS 的对照环境。

| 对照项 | 相对 RMSE | 余弦相似度 |
|---|---:|---:|
| 文本特征 | 0.5430% | 0.9999853 |
| DiT 首步 | 0.4060% | 0.9999918 |
| DiT 第二步（使用前缀缓存） | 0.6568% | 0.9999798 |
| VAE 编码 | 0.0776% | 0.9999997 |
| VAE 解码 | 0.4953% | 0.9999883 |

Rust 缓存与完整重算第二步的相对 RMSE 为 0.2126%，受不同矩阵形状的 FP16 舍入影响。
512×512 茶壶样例完整 8 步、相同初始噪声的原生结果与官方结果 **PSNR 48.14 dB**，
RGBA 像素 RMSE 为 0.9994 / 255；不是逐像素完全相同。已人工检查两张图片。
详细数据与图像位于 `artifacts/qwen-image-21/parity-report.json`、`teapot-512.png` 和 `reference-pipeline-512/official.png`。

### MPS 参考路径问题

本机 PyTorch 2.9.1 的 MPS 隐式因果 SDPA 在长度 39 的 Qwen 文本输入上与 CPU 结果不一致，
连第一位置应只读取 V[0] 的结果也不一致。该参考输出的注意力相对误差约 32.9%，不能作为 Rust 正确性标准。
用同一 Q/K/V 进行 CPU 交叉检查后，改用 **官方 Transformers eager 注意力实现**作为文本 oracle；
保留原始 `reference-text/expected-mps-sdpa.json` 供复现。生成主干与 VAE 仍使用官方 Diffusers MPS 路径。
这是一项本机版本组合的观测，不代表所有 PyTorch 或 MPS 版本都有此问题。

### 图片编辑数值对照（2026-10-12）

新增 `scripts/reference_qwen_image_edit.py` 和 `scripts/check_qwen_image_edit.py`，
沿用上述固定官方版本与 FP16，不在应用运行时调用 Python。

| 对照项 | 结果 |
|---|---|
| 不透明图片、非正方形透明图片的预处理 | 视觉 patch 与 VAE RGBA 输入均逐值一致 |
| 单图视觉塔最终特征 | 相对 RMSE 0.5144%，余弦 0.9999868 |
| 多模态 token 与三轴位置 | 与官方逐值一致 |
| 使用同一官方视觉特征的文本主干 | 相对 RMSE 0.6508%，余弦 0.9999788 |
| 视觉塔与文本主干串联 | 相对 RMSE 3.1031%，余弦 0.9995184；FP16 视觉舍入通过 36 层文字计算累积 |
| 双参考图 DiT 首步 / 缓存第二步 | 相对 RMSE 0.1871% / 0.1641% |
| 双参考图缓存与完整重算第二步 | 相对 RMSE 0.0782% |
| 512×512 红茶壶改蓝，完整 8 步 | 与官方 PNG 的 PSNR **51.17 dB**，RGBA 像素 RMSE **0.7044 / 255** |
| 256×256 双参考图编辑，完整 8 步 | 与官方 PNG 的 PSNR **49.95 dB**，RGBA 像素 RMSE **0.8114 / 255** |

完整样例使用相同的初始噪声、编辑指令及 512² 参考面积；已检查原图、Rust 结果与官方结果。
成品为 `artifacts/qwen-image-edit/edited-512-final.png` 与 `edited-multi-256-final.png`，
对照结果为 `parity-report.json`。最终 CLI 单次加载约 33.47 s，512 编辑约 20.59 s，
其中参考处理 3.03 s、文本 1.27 s、生成 15.49 s、解码 0.80 s；GPU 算子缓存此前已有预热。
最初新形状运行约 56.07 s，包含编译/调优成本；这些是单次运行记录，不是稳定性能保证。
文生图回归与上一交付版本的 512 茶壶结果 PSNR 为 54.29 dB、像素 RMSE 0.4919 / 255，
反映显式 GPU 注意力路径与自动调优路径的 FP16 舍入差异。

```sh
.venv-qwen-image/bin/python scripts/reference_qwen_image_edit.py pipeline \
  --image artifacts/qwen-image-21/teapot-512.png --reference-resolution 512 \
  --output artifacts/qwen-image-edit/reference-pipeline-512
./target/release/qwen-image generate \
  --image artifacts/qwen-image-21/teapot-512.png --reference-resolution 512 \
  --prompt '把茶壶改成蓝色，保持木桌、光照和构图不变。' \
  --width 512 --height 512 --seed 42 --output artifacts/qwen-image-edit/edited-512-final.png
```

`vision`、`multimodal`、`dit_edit` 子命令生成组件输入与 oracle，Rust `trace` 读取同一输入。
`multimodal_text` trace 使用 oracle 的视觉特征，单独测文本计算，避免把两阶段舍入混在一起。
`crates/qwen-image/tests/fixtures/` 保存 Pillow 12.3.0 的 alpha 缩放、JPEG EXIF 和 WebP 黄金样本。
原生 UI 验证入口为 `prototypes/makepad/validate-image-edit.py`，运行报告和截图保存在编辑 artifact 目录。

最终原生验收见 `artifacts/qwen-image-edit/ui-final-safe/report.json` 与 `edited-result.png`。
覆盖取消选图、损坏文件、Unicode 路径、RGBA 预览、首次视觉塔加载停止、实际单图编辑、
请求参数快照、拒绝覆盖原图、双图非正方形 RGBA 的 256 画布生成、生成停止后保存旧结果、
恢复文生图，以及切回 ASR 的固定 WAV 全文回归。512×512 单图编辑预热后本次为 **17.6 s**：
参考图 2.65 s、文本 1.10 s、生成 13.17 s、解码 0.64 s。停止响应约 **58 ms**，
编辑过程中修改输入并读回约 **2.8 ms**；这是本机单次观察，首次编译和大尺寸内核的停止等待会更长。
新增图像 crate 8 项单元测试通过，Makepad 16 项通过、1 项原有忽略，独立 WebGPU feature 编译检查通过。
`tts-regression/report.json` 的实际设备静音回调回归也通过：加载与路径记忆、播放中停止、重播停止、
生成结束后的排空停止、重新播放及 ASR 转写。既有 ASR/TTS 推理与 ANE 后端源码、独立 CLI 均保留。
最终程序安装到 `target/release/r2t2-makepad`、`target/makepad-native/release/r2t2-makepad` 和
`target/release/qwen-image`。本次源码、校验值和验证索引保存在 `artifacts/qwen-image-edit/delivery.json`，
接入前的原生程序与源码备份在同目录 `baseline/` 中。

复现组件与完整图像对照：

```sh
for part in text dit vae; do
  .venv-qwen-image/bin/python scripts/reference_qwen_image.py "$part" \
    --output "artifacts/qwen-image-21/reference-$part"
  ./target/release/qwen-image trace --component "$part" \
    --input "artifacts/qwen-image-21/reference-$part/input.json" \
    --output "artifacts/qwen-image-21/reference-$part/rust.json"
done
.venv-qwen-image/bin/python scripts/reference_qwen_image.py pipeline \
  --output artifacts/qwen-image-21/reference-pipeline-512
./target/release/qwen-image generate \
  --prompt 'A red ceramic teapot on a wooden table, soft morning light, realistic photograph.' \
  --width 512 --height 512 --seed 42 --output artifacts/qwen-image-21/teapot-512.png
.venv-qwen-image/bin/python scripts/check_qwen_image.py
```

原生界面验证脚本 `prototypes/makepad/validate-image.py` 使用真实模型与独立隐藏窗口，
仅用测试专用 native panel shim 替代文件选择回调，不注入用户桌面输入。
覆盖模型缺失、取消选择、加载进度、路径记忆、无效请求、请求快照、实际出图、PNG 保存、
推理停止、保留上一张结果，以及切回 ASR 转写。运行结果以 `artifacts/qwen-image-21/` 下的报告为准。

2026-10-12 本机已通过上述真实界面流程，报告为 `ui-final/report.json`，截图为 `ui-final/result.png`。
1024×1024 中文茶壶提示词：文本 **0.73 s**、8 步生成 **50.78 s**、VAE 解码 **2.48 s**，合计 **54.0 s**；
本次停止响应 **0.115 s**，生成期间修改输入并读回控件耗时约 **3.5 ms**。
这是 GPU 算子缓存已有预热的单次观察，不是多个样本的性能保证。模型加载另计 **32.7 s**。
此前同尺寸新形状冷运行约 **189.8 s**，其中 VAE 首次编译/调优与解码合计 **105.8 s**；
不能把冷运行与预热后数据当成相同条件比较。
512×512 英文完整对照、1024×1024 英文机器人及中文茶壶均已真实出图。
机器人样例请求透明背景但模型仍输出了几乎不透明的背景（alpha 216–255）；
程序保留其真实 alpha，不做强制抠图或伪造透明度。
此段为最初接入时的性能记录；当前支持的官方 2K 预设与尺寸规则见本文开头。

文生图首次交付时，图像 crate 的 3 项单元测试、Makepad 的 16 项单元测试通过（另有 1 项原有忽略项）；
独立 `webgpu` feature 编译检查通过。TTS 的实际设备静音回调测试也通过：加载、播放中停止、
重播停止、生成结束后的排空停止、再次重播，以及切回 ASR；详见 `tts-regression-final/report.json`。
47 个已有语音推理/ANE 后端相关源文件与接入前校验值一致，原独立 ASR/TTS CLI 二进制保留。
首次 ASR 回归包含一次原有 ANE 编译失败后的 Metal 回退，转写与固定预期全文一致；
本轮未调整该加速策略，也不把这次含编译的 ASR 耗时作为性能结论。

## Prompt Rewrite（官方提示词改写）

官方 `prompt_rewrite/` 流程（Qwen3.5-VL 9B 微调的 PE-T2I / PE-I2I enhancer）已按上游
commit `ab96caa` 移植到 `crates/qwen-image/src/rewrite/`：系统提示词资产逐字节内嵌、
任务 profile 与采样默认值、消息构造（edit 图片 `<image1>`… 按序在前）、严格 JSON 解析
与比例决策、CLI `generate --rewrite` 接线均已落地并有无模型单元测试。本地改写后端
（MiniCPM-V 实现 `Rewriter` trait）尚未接入，当前 `--rewrite` 会在加载生成模型前清晰
报错，不静默透传；不带 `--rewrite` 行为不变。详见 [docs/qwen-image-rewrite.md](qwen-image-rewrite.md)。

## 来源与许可

- [ModelScope 模型](https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo)
- [官方权重与使用说明](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
- [Qwen-Image-2.1 官方仓库](https://github.com/QwenLM/Qwen-Image-2.1)
- [固定 Diffusers pipeline](https://github.com/huggingface/diffusers/blob/883d51ef02cd7972eeaf158c652d229f3459f419/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- [固定 Diffusers DiT](https://github.com/huggingface/diffusers/blob/883d51ef02cd7972eeaf158c652d229f3459f419/src/diffusers/models/transformers/transformer_qwenimage21.py)
- [固定 Diffusers VAE](https://github.com/huggingface/diffusers/blob/883d51ef02cd7972eeaf158c652d229f3459f419/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py)

权重采用 **Qwen Research License**，限研究与非商业用途，详见模型目录中的 `LICENSE`；代码许可与权重许可独立。
