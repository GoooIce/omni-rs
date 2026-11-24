# Sol-H3-Spark 这一套权重的硬盘和显存

2026-10-22。问题是：按 [NVlabs/Sana](https://github.com/NVlabs/Sana) 分支 `sol-engine` 上 [Sol-H3-Spark README](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/Sol-H3-Spark/README.md) 把这一套权重下下来，硬盘最多要多少、最少要多少；显存最多要多少、最少要多少。

下面的树是 `sol-engine` 的 `9f0af774e7be7a82cb429de7428e678e8157c750`（2026-10-20，`fix(streamrender-h3): use continuous T in runtime diffusion config`）。README 正文最后一次提交是 `757d9022456e989ca0a9c3cb0dff644d682680a2`（2026-09-10）。权重清单是同一次树里的 [`configs/checkpoints.json`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/configs/checkpoints.json)，字段 `metadata_verified` 写着 `2026-09-10`。下载器是同树的 [`download_checkpoints.py`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/download_checkpoints.py)。文件字节来自这些钉死修订上的 Hugging Face tree API（`size`）。1 GiB = 1024³ 字节。下载器若在本地再留一份 Hub 缓存或临时文件，清单和 README 都没写字节数，下面不计入。

**结论：硬盘最多 256,719,089,824 字节（239.088284 GiB），是三条任务加上离线 INT8 缓存权重，同一路径只加一次。硬盘最少 141,762,206,092 字节（132.026343 GiB），是默认 T2VA 推理要打开的文件，不含离线 Gemma 和 connector。显存只有一个写明的运行数：项目页上的 Sol-H3 驻留 116.9 GiB，统一内存池 119.68 GiB，采样时剩余 2.8 GiB。没有更小的官方运行数字。FL2VA / Ref2VA 要多驻留 H3 视频 VAE，页面和加载代码都没有给出这份额外显存，所以上限没有写死。页面上的 133.4 GiB 和 142.4 GiB 是放不下的对照，不是这套要下载的配置。**

## 这一套列了什么

README 的默认配方是固定的：672×384×124 的草稿，再做成 1344×768、121 帧。分辨率写在 [`configs/default.json`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/configs/default.json)。没有按分辨率再拆一套检查点。Stage2 的 `BatchSplitAdapter` 把 `max_batch_size` 写成 1，像素形状的 batch 也是 1（[`runtime/stage2_ops/models.py`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/runtime/stage2_ops/models.py)）。

[`docs/setup.md`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/docs/setup.md) 把任务收成三行：

| 任务 | H3 分区 | 输入 VAE | 草稿 LoRA |
| --- | --- | --- | --- |
| `t2va` | `transformer/` | 不加载 | FastH3 VSA DataFree |
| `fl2va` | `transformer/` | 原生 H3 VAE | 同一份 VSA DataFree |
| `ref2va` | `transformer_ref/` | 原生 H3 VAE | LightX2V Ref2VA four-step v0.1 |

W8A8 FP8 是加载时从 BF16 DiT 转出来的，清单里没有单独的 FP8 DiT 文件。Stage1 在 BF16 LoRA 合并之后调用 `convert_model_to_fp8`，并要求 312 个 linear（[`runtime/stage1.py`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/runtime/stage1.py)）。FastVideo `3d8ac9d14bd697a89ede8f170cbfbca012a9edcc` 的 `fastvideo/layers/quantization/fp8_config.py` 把原 `weight` 从参数里弹出，留下 float8 权重和每个 linear 一个 float32 标量 scale。

离线的 INT8 Gemma 和 INT8 connector 只用来做一次通用 prompt cache。`checkpoints.json` 给它们标了 `offline_only`。推理时 Stage2 收据把 `stage2_gemma_loaded` 和 `stage2_connector_loaded` 写成 false。cache 没有公开下载地址，`external.prompt_cache.public_source` 是 null。

同一张模型卡上还有别的文件。下载器的 `allow_patterns` 和 `filename` 不选它们，所以不进下面的总和。其中包括：MiniMax 仓库里 `FL2VA/`、`Ref2VA/` 两套重复树，以及 `text_encoder/model-*.safetensors`（清单写明原生 BF16 Qwen 排除在外，这 14 片合计 66,714,912,872 字节）；Comfy-Org 卡上除 NVFP4 以外的 Qwen 和 DiT；FastVideo LoRA 卡上的 dense 与两份 synthetic；lightx2v 卡上除 `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` 以外的 LoRA；upscaler 卡上的 fp16（690,592,672 字节）和 fp32 `.pth`（1,381,251,539 字节），作者 README 说这三份是同一套 3D 卷积的不同精度，钉死的是 bf16；LTX-2.5 卡上的 distilled 全量变压器、NVFP4、非 Conv 视频 VAE、BF16 Gemma、空间/时间 upscaler。清单里的文件没有「同一份权重再存一份 safetensors 和一份 bin」。`transformer/` 和 `transformer_ref/` 是两条任务的两个目录，分片字节数相同，上限里各算一次。

## 硬盘

单文件的字节数，`checkpoints.json` 和对应修订的 tree API 一致。快照目录的字节数只加 `allow_patterns` 匹配到的文件。`huggingface-hub==1.29.0` 写在 [`requirements/download.txt`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/requirements/download.txt)。模式末尾的 `/` 会补成 `*`，`*` 按该版本的 fnmatch 匹配。

### 最多

三条任务各跑一次进同一个输出目录，再加 `--include-offline`。共享文件只留一份。

| 成分 | 字节 | GiB | 来源 |
| --- | ---: | ---: | --- |
| MiniMax-H3 三条任务的并集 | 143,616,796,453 | 133.754 | 修订 `9bfb6693f2cf6de171db46d1aa586f67d773a1da`，[tree API](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3/tree/9bfb6693f2cf6de171db46d1aa586f67d773a1da?recursive=true) |
| NVFP4 Qwen | 15,687,142,551 | 14.610 | `Comfy-Org/MiniMax-H3` @ `0543966fbdce5ba05709a8f2031c94bdba629b4a`，`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| LTX-2.5 dev BF16 变压器 | 42,018,190,584 | 39.133 | `Lightricks/LTX-2.5` @ `5e6e71018ee1756ed329b697a7b4aedc934dfce9`，`diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| distilled LoRA 450 | 8,899,889,568 | 8.289 | 同修订，`loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` |
| Conv 视频 VAE | 1,452,269,922 | 1.353 | 同修订，`vae/ltx-2.5-video-vae-conv-bf16.safetensors` |
| LTX 音频 VAE | 364,866,540 | 0.340 | 同修订，`vae/ltx-2.5-audio-vae-bf16.safetensors` |
| H3 latent upscaler BF16 | 690,592,992 | 0.643 | `LBH-123-AI/Minimax_h3_latent_Upscaler` @ `13ccf95d85d120bdbc92c05b1247a6e147bf54bf` |
| H3-to-LTX adapter | 389,542,136 | 0.363 | `model.safetensors` 389,541,112 加 `config.json` 1,024；修订 `cfcd8a7cc36c135142287584728f7fe362df0867` |
| VSA DataFree LoRA | 5,339,117,712 | 4.972 | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` @ `f509e629374cac104e7f62daecce6d1488a3041d`，`vsa-datafree/adapter_model.safetensors` |
| Ref2VA four-step LoRA | 1,383,677,768 | 1.289 | `lightx2v/Minimax-h3-Turbo` @ `2f015e66b37c585cea9dc4ae6f1850ea8788e742` |
| 离线 INT8 Gemma | 15,372,969,374 | 14.317 | 同 LTX 修订，`text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| 离线 INT8 connector 源 | 21,504,034,224 | 20.027 | 同修订，`diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` |
| 合计 | 256,719,089,824 | 239.088284 | 上表相加 |

MiniMax 并集的构成：

- `transformer/` 16 个匹配文件 66,280,569,250 字节。其中 14 个 `diffusion_pytorch_model-*-of-00014.safetensors` 合计 66,280,504,216 字节，加上 `config.json` 546 和 index 64,488。
- `transformer_ref/` 同样 16 个文件、同样 66,280,569,250 字节。tree 里对应分片的 `size` 与 `transformer/` 逐项相同。
- `vae/` 里尚未计入上面两项的 4 个文件 10,415,633,116 字节。目录一共 5 个文件、10,415,635,127 字节；其中 `config.json` 2,011 字节已经在下一条里。三片权重 10,415,558,888，index 74,228。
- 三条任务都下载的公共匹配文件 640,024,837 字节。其中 `audio_vae/diffusion_pytorch_model.safetensors` 605,429,340 字节，其余 25 个小文件 34,595,497 字节，含 `vae/config.json`。原生 Qwen 的 14 片 safetensors 不匹配 `text_encoder/*.json`、`*.txt`、`*.jinja`、`*.model`。

这四行相加：66,280,569,250 + 66,280,569,250 + 10,415,633,116 + 640,024,837 = 143,616,796,453。

`audio_vae` 这一片每条任务都在公共模式里，T2VA 也下。视频 VAE 的权重片只在 `fl2va` 和 `ref2va`。

### 最少

默认任务是 `t2va`，也是文件最少的一条：不取 `transformer_ref/`，不取视频 VAE 权重片，不取 Ref2VA LoRA，不取两份离线 INT8。Stage1 对 T2VA 把视频 VAE 的加载器换成直接报错（`stage1.py` 的 `forbidden_video_vae`）。

| 成分 | 字节 |
| --- | ---: |
| MiniMax-H3 的 T2VA 匹配文件 | 66,920,594,087 |
| 上表里从 Qwen 到 adapter 的 7 行 | 69,502,494,293 |
| VSA DataFree LoRA | 5,339,117,712 |
| 合计 | 141,762,206,092（132.026343 GiB） |

另外两条任务的推理下载，仍不含离线权重：`fl2va` 152,177,839,208 字节（141.726657 GiB），`ref2va` 148,222,399,264 字节（138.042866 GiB）。Ref2VA 的 LoRA 更小，但多了 `transformer_ref` 不会少掉的视频 VAE，所以比 T2VA 大。

推理还要一个本地 prompt cache。[`runtime/prompt_cache.py`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/runtime/prompt_cache.py) 把视频上下文写成 `[1, 1024, 4096]`、音频写成 `[1, 1024, 2048]`，都是 BF16，张量本体 8,388,608 + 4,194,304 = 12,582,912 字节。加载器拒绝大于 `64 * 1024²` = 67,108,864 字节的文件。`.pt` 的精确体积来源没写。这个文件不是权重下载。

机器上还没有这份 cache 时，`setup.md` 的第一次路径要带 `--include-offline`，再跑 `runtime.cache_builder`。两份离线文件合计 36,877,003,598 字节（34.344386 GiB）。T2VA 推理集加上它们是 178,639,209,690 字节（166.370729 GiB）。cache 生成之后，推理进程不加载这两份；缺的是 cache 文件，不是这两份权重。所以上面的最少不把它们算进去。

## 显存

README 写：在一块 GB10/SM121 上跑，这块 GPU 的全部内存要可用。它没有写字节数。[`docs/validation.md`](https://github.com/NVlabs/Sana/blob/9f0af774e7be7a82cb429de7428e678e8157c750/models/minimax_h3/Sol-H3-Spark/docs/validation.md) 写，输入编码器和参考激活还要额外内存，能不能放下不能从纯文本的驻留结果推出来。README 也写，FL2VA / Ref2VA 的额外参考内存不在 T2VA 驻留结果里。

写明 GiB 的是 README 链到的项目页。源文件在同仓库 `page` 分支 `33bfd79c500aee1ff3cb3e24068fe6728761a0e6`（2026-10-19）的 [`Sol-Engine/Sol-H3-Spark/index.html`](https://github.com/NVlabs/Sana/blob/33bfd79c500aee1ff3cb3e24068fe6728761a0e6/Sol-Engine/Sol-H3-Spark/index.html)，线上是 <https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/>。2026-10-22 读到的线上 HTML 与这份 blob 都是 422,982 字节，下面的 GiB 字符串两边都在。

页面把 DGX Spark 的 CPU 和 GPU 统一内存写成一个 119.68 GiB 的池。三条柱：

| 柱 | 页面上的总数 | 页面自己的话 |
| --- | --- | --- |
| BF16 checkpoint weights alone，MiniMax-H3 official | 133.4 GiB，超出池 13.7 GiB | 官方全量 BF16 管线放不进这块统一内存 |
| Naive two-stage pipeline | 142.4 GiB，超出池 22.7 GiB | 原文是 “Our estimated naive two-stage configuration” |
| Latent transfer + cached Stage 2 prompt，Sol-H3 | 116.9 GiB，采样时剩余 2.8 GiB | “Request-boundary footprint 116.9 GiB fits within the 119.68 GiB Spark pool” |

119.68 − 116.9 = 2.78，页面把剩余印成 2.8 GiB。Sol-H3 这一柱的分段，`title` 属性里写的是：

| 分段 | 页面上的 GiB |
| --- | ---: |
| NVFP4 Qwen | 14.6 |
| FP8 H3 DiT | 23.4 |
| H3 latent upscaler | 0.6 |
| VAE Adapter | 0.4 |
| LTX refiner | 35.4 |
| LTX VAE decoder | 0.8 |
| Runtime, allocator and shared UMA overhead | 41.7 |
| 合计 | 116.9 |

这一柱没有 H3 视频 VAE，也没有 Gemma / connector。这和 T2VA 的加载方式同一侧：视频 VAE 不物化，在线推理不加载 Gemma 和 connector。页面没有给 480p、512p、更大 batch，或 FL2VA / Ref2VA 另一组 GiB。同一页另有一句：Stage 1 可以配 384p、480p 或 512p 草稿。Spark 包的 `default.json` 把草稿固定在 672×384、输出固定在 1344×768。

所以写明的运行占用就是 116.9 GiB，而且已经含 41.7 GiB 的 runtime / allocator / 统一内存开销。没有第二份更小的官方运行数字。代码里也没有更低精度的开关：Qwen 是这份 NVFP4，H3 linear 转到 tensorwise FP8，LTX 用 BF16 且 `quantization=None`、`OffloadMode.NONE`。

FL2VA / Ref2VA 的显存上限来源没写。能从权重文件读到的、只在这两条任务里多出来的视频 VAE 张量是 FP32：decoder 9,694,121,984 字节（9.028 GiB），encoder 721,342,144 字节（0.672 GiB），另有 `quant_conv` 9,408 与 `post_quant_conv` 2,400。这是 safetensors 头里的张量字节，不是页面给的驻留 GiB。加载代码允许这两条任务物化视频 VAE，并禁止用它解码生成结果；它没有写物化之后留下 encoder、decoder 还是整个模块。把 9.028 GiB 加进 116.9 GiB 会填一个来源没写的洞，这里不加。

133.4 GiB 和 142.4 GiB 是页面用来说明「放不下」的对照。133.4 的分段是 Qwen 62.1、H3 DiT 61.7、H3 audio VAE 0.6、H3 VAE decoder 9.0。142.4 那一柱页面自己标了 estimated，而且里面还有在线 Gemma 和 connector。这两条都不是 `download_checkpoints.py` 的运行配置。

### 和文件字节对得上的分段

这些是文件或张量字节除以 1024³ 之后，落到页面印出来的一位小数上。它们说明页面有几格就是权重大小四舍五入，不是另一套实测。

| 页面 | 文件或张量 | 字节 | 除以 1024³ |
| --- | --- | ---: | ---: |
| 官方 Qwen 62.1 | `text_encoder/model-*.safetensors` 14 片，下载器不取 | 66,714,912,872 | 62.133 |
| 官方 H3 DiT 61.7 | `transformer/` 14 片权重 | 66,280,504,216 | 61.729 |
| 官方 VAE decoder 9.0 | `vae/` 里名字含 `decoder` 的张量，dtype `F32` | 9,694,121,984 | 9.028 |
| 官方 audio 0.6 | `audio_vae/diffusion_pytorch_model.safetensors` 的张量，dtype `F32` | 605,306,340 | 0.564 |
| NVFP4 Qwen 14.6 | 钉死的 NVFP4 文件 | 15,687,142,551 | 14.610 |
| upscaler 0.6 | bf16 文件的张量 | 690,560,432 | 0.643 |
| adapter 0.4 | 模型卡写 194,759,504 个参数；safetensors 头里 BF16 载荷 389,519,008 | 389,519,008 | 0.363 |
| naive 柱的 Gemma 14.3 | 钉死的 INT8 Gemma 文件 | 15,372,969,374 | 14.317 |

NVFP4 文件的头是 231,408 字节（8 字节长度加 231,400 字节 JSON）。张量载荷 15,686,911,143 字节：`U8` 12,189,715,279，`F8_E4M3` 1,523,712,000，`I8` 777,912,320，`BF16` 1,194,962,400，`F32` 609,144。`runtime/qwen.py` 按每条记录的 `nbytes` 拷到 `cuda:0`，并要求权重一直留在 CUDA 上。这 15,686,911,143 字节是权重复制量。Comfy 还会不会再做一份视图，源码没有写成一个常数。

### 和文件字节对不上的分段

这些格子不能从钉死文件的字节数直接得到。LTX-2.5 是 gated 仓库，2026-10-22 对 safetensors 头的 Range 请求返回 401，所以下面只有 tree API 的文件 `size`，没有张量载荷。

| 页面 | 钉死文件 | 文件 GiB |
| --- | ---: | ---: |
| FP8 H3 DiT 23.4 | 14 片 BF16 分片 66,280,504,216 字节，加载后才转 FP8 | 61.729，转完不是这个数 |
| LTX refiner 35.4 | dev BF16 变压器 42,018,190,584 字节 | 39.133 |
| LTX VAE decoder 0.8 | Conv 视频 VAE 1,452,269,922 字节 | 1.353 |
| naive 柱的 connector 1.9 | 离线 INT8 文件 21,504,034,224 字节 | 20.027 |
| runtime 41.7 | 没有对应权重文件 | 页面写的是 runtime、allocator 和共享统一内存开销 |

connector 这一格差得最大：页面 naive 柱写 1.9 GiB，下载器取的是整份 20.027 GiB 的 INT8 dev 变压器。推理不加载这份文件。

FP8 之后的 DiT 权重可以按头和加载代码加出来，这是权重字节，不是页面的 23.4，也不是实测显存。14 片头里的张量载荷是 BF16 66,211,541,504 字节加 FP32 68,888,576 字节。FastVideo 同一修订的 `param_names_mapping` 把 `attn.to_out.0` 映到 `attn.to_out`，把 `ff.net.0.proj` 映到 `ff.fc_in`，把 `ff.net.2` 映到 `ff.fc_out`。按这个映射，312 个 linear 权重都在，形状各 52 份：

- `to_q` / `to_k` / `to_v`：`[7168, 5376]`
- `to_out`：`[5376, 7168]`
- `ff.fc_in`：`[28672, 5376]`
- `ff.fc_out`：`[5376, 14336]`

这 312 份原为 BF16，合计 40,076,574,720 字节。`convert_model_to_fp8` 用 tensor 粒度，权重变成 20,038,287,360 字节的 float8，再加 312 个 float32 scale，1,248 字节。原权重被弹出。

`runtime/stage1_ops/lookup.py` 在这之前拆掉 50 个 `adaln_proj.linear`、`norm_out.linear` 和 `time_embedder` 的两层。这些张量合计 26,142,079,488 字节、106 个。T2VA 的查找表行数是代码里断言的 `[1, 2, 2, 2]`，共 7 行。adaln 权重形状 `[96768, 2688]`，`norm_out.linear` 是 `[10752, 2688]`，输出按 BF16 存：`(50 × 96768 + 10752) × 7 × 2` = 67,888,128 字节。其余留在原 dtype 的张量是 61,775,872 字节、220 个。

T2VA 上 DiT 权重因此是：

20,038,287,360 + 1,248 + 61,775,872 + 67,888,128 = 20,167,952,608 字节（18.783 GiB）。

页面的 23.4 GiB 比这个和大 4,957,606,074 字节左右。激活、KV、算子工作区和分配器不在这个加法里。差出来的部分来源没有拆开，这里不把它算进 23.4，也不把 18.783 GiB 说成实测显存。Ref2VA 还会建第二套 timestep 查找表；行数取决于条件 timestep 是否已经在集合里，源码没有把那份额外字节写成常数。

VSA LoRA 文件是 5,339,117,712 字节。合并之后 `stage1.py` 去掉 wrapper 和 A/B。它不另占一份与 BF16 底模并行的驻留权重。distilled LoRA 450 随 LTX `from_checkpoint` 传入；LTX 加载器是否把它折进 39.133 GiB 的底模，gated 源码这次没有读到头，页面的 35.4 又小于这个文件，所以不加。

H3 音频 VAE 的 605,306,340 字节 FP32 张量，Stage1 在 T2VA 也保持驻留（`stage1.py` 写 “DiT/audio VAE resident”）。Sol-H3 那一柱没有单独的 audio 分段。官方柱把 audio 标成 0.6 GiB，和这片张量的 0.564 GiB 对得上。不把 0.564 GiB 再加进 116.9。

## 来源没有写死的数

- FL2VA / Ref2VA 相对 116.9 GiB 多出来的显存。已知的只是视频 VAE 的 FP32 张量字节，以及 validation 说的「额外内存不能从 T2VA 结果推」。
- prompt cache 的 `.pt` 精确字节。只有张量本体 12,582,912 字节，和不超过 67,108,864 字节的拒绝线。
- 116.9 GiB 里 23.4、35.4、0.8、41.7 四格的原始分配日志。23.4 与上面的权重加法不相等；35.4 和 0.8 小于对应文件；41.7 页面写的是运行时开销。
- LTX safetensors 的张量级字节。树 API 给了文件 `size`，头的 Range 请求是 401。
- Hugging Face 下载器在 `local_dir` 之外的缓存和临时空间。
