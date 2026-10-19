# Breeze-TTS-2 本地流式接入

Breeze-TTS-2 通过独立的 `omni-breeze-tts` crate 在 Rust/Burn 中执行，接入已有 Makepad 本地版。
支持中文和英文文本、音色设计、参考音频声音克隆、流式 PCM、手动停止和保存 WAV。
macOS 使用 Metal；其他原生平台选择 WGPU。Python 仅用于开发时的官方数值对照。

## 运行

在仓库根目录执行，已有完整模型时可跳过下载：

```sh
python3 scripts/download_breeze.py
python3 prototypes/makepad/build-native.py
./target/release/r2t2-makepad
```

进入右上角“语音合成”。第一次打开该页面会自动载入 `models/Breeze-TTS-2`，显示各阶段进度。
模型缺失时选择包含 `config.json`、`tokenizer.json`、分片权重和 `audio_tokenizer` 的目录。
也可以直接选择该目录中的模型文件。成功加载后才保存路径，取消选择后仍可重试。

输入朗读文本后点击“合成并播放”。声音描述可以填写“年轻女性，声音温柔，语速自然”，
留空时计算量较小。输入框在合成期间可以编辑，修改在下一次合成生效。
“停止”会让输出回调静音，并在当前推理检查点结束生成；已经生成的部分可以保存。
文本编码与提示词预填充逐层检查取消，声学生成逐 token 检查。正在执行的 GPU 内核或首次
内核编译不能强行打断，冷启动取消可能比预热后慢。
WAV 为 24 kHz、单声道、16 位 PCM。没有音频输出设备时仍可合成并保存。

克隆声音时，点击“选择参考 WAV…”，在“参考文字”中填写这段音频实际说出的内容，
再输入要朗读的新文字。建议使用 3–10 秒清晰人声。可以同时填写声音描述；“清除参考”恢复普通合成。
参考编码器在首次克隆时加载，同一 PCM 的编码在工作线程内缓存，修改目标文字或描述不必重复编码。
参考音频和参考文字必须成对填写，文件读取失败会显示错误，不会退回普通声音继续合成。
首段与总生成耗时包含参考音频的准备时间；第一次克隆还包含参考编码器加载和 GPU 算子编译。

### 音色设计与试听

在语音合成页面选择“音色设计”，填写试听文字和想要的声音，点击“生成并试听”。
可以从温柔女声、沉稳男声、清朗青年、英文旁白示例开始修改描述，指定年龄感、音高、音色和表达方式。
描述语言建议与试听文字一致。此模式使用官方 `tts_instruction` 模板和 CFG 4，无需参考录音。
切换到设计模式后，之前在朗读模式选择的参考音频会保留，但不会参与音色设计。

“换个候选”会改变随机采样种子，再按当前描述和试听文字生成。
“再次试听”直接播放已生成的 PCM，便于比较和重复听取；不会重新运行生成网络。
“停止”可以中断生成、试听或选用音色时的参考编码。已生成的部分仍可保存 WAV；
只有正常到达模型 EOS 的完整设计结果才可点击“使用此音色朗读”。

选用音色后，程序将实际生成的试听 PCM 送入参考编码器，缓存码字，并自动切回“朗读 / 参考克隆”。
参考原文取自那次生成请求的快照，显示为只读；生成期间修改输入框不会改变它。
修改目标文字后即可继续合成，也可以填写声音描述进一步调整表达。
试验新的音色候选不会替换已经选用的音色，直到再次点击“使用此音色朗读”。

已选用音色保留在当前会话内。跨会话复用时保存试听 WAV，并在下次作为参考音频载入，填写对应试听原文。
同一描述或随机种子不能保证换一段文字仍有完全相同的音色；需要持续使用某个设计结果时，使用参考音色复用流程。
音色遵从程度和相似度仍需试听判断。

macOS 的 TTS 路径保存于 `~/Library/Application Support/omni-r2t2/breeze-tts.json`。
`BREEZE_TTS_MODEL_DIR` 和 `BREEZE_TTS_SETTINGS_PATH` 分别用于覆盖模型目录和隔离测试设置。
ASR 与 TTS 各有工作线程，通过同一个 Burn GPU 注册共享设备；运行一个任务时禁用页面切换。

CLI 可用于可复现的合成与测量，输出浮点 WAV 和同名 JSON 统计：

```sh
python3 scripts/build_breeze.py
./target/release/breeze-tts synthesize \
  --text '你好，欢迎使用本地语音合成。' --output breeze.wav

./target/release/breeze-tts synthesize \
  --text 'Hello, welcome to local speech synthesis.' \
  --instruction 'A young woman with a clear and gentle voice.' \
  --guidance 4 --output english.wav

./target/release/breeze-tts synthesize \
  --reference-audio reference.wav --reference-text '参考录音里实际说出的文字。' \
  --text '这是用参考声音朗读的新内容。' --output clone.wav
```

`Ctrl-C` 停止 CLI 生成，并保存已输出音频。`--temperature 0` 使用贪心选择；默认温度 0.9、
top-k 50、top-p 1.0、seed 42。随机数发生器与 PyTorch 不同，因此相同种子不代表逐样本相同。
depth decoder 默认在 GPU 上采样并查询下一路 embedding；码字直接供下一帧和 codec 使用，
报告所需的完整码本在请求结束时一次回读。
`--cpu-depth-sampling` 可切回逐路 CPU 采样，用于对照；两种方式的贪心结果一致，
随机采样分别使用 Gumbel-max 和累积概率法，因此随机序列不同。
CLI 的 `--fp32` 是全模型 FP32 数值诊断选项；原生界面使用 FP16 生成网络和 FP32 音频解码器。
文本编码器的残差与后置归一化保留 FP32：英文声音描述对照样例在官方实现中的残差超过 7.1 万，
会溢出 FP16 的 65504 上限；矩阵乘仍使用 FP16，最终归一化后再转回 FP16。

## 模型与缓存

| 阶段 | 实现 |
|---|---|
| 文本编码 | T5Gemma2，26 层，1152 维，双向局部/全局注意力，Gemma RMSNorm |
| 参考声音编码 | 捆绑 Mimi 编码器，因果卷积、8 层 Transformer、1 路语义和 15 路声学 RVQ，FP32 |
| 首路语音 token | Qwen3，28 层，2048 维，GQA 与跨帧 KV cache |
| 其余 15 路 token | 12 层、1024 维 depth decoder，每帧独立 KV cache，特定 Llama 3 RoPE 缩放 |
| 波形解码 | 捆绑的 Qwen3-TTS 12 Hz tokenizer decoder，16 路量化码本、8 层注意力、因果卷积与转置卷积 |
| 音频输出 | 每帧 1920 个采样，即 80 ms；默认每两帧交付一块音频 |

文本模板沿用官方 `[S0]`、`<ins_bos>`、`<ins_eos>` 及 tokenizer 的特殊 token 处理。
声音描述默认在主生成器和 depth decoder 同时应用 CFG 4；没有描述时只计算一个分支。

官方仓库没有放在合成前面的提示词增强器。朗读文字原样接在 `[S0]` 后面，会被直接读出来。声音描述包在 `<ins_bos>` 与 `<ins_eos>` 之间。行内声音事件写在朗读文字里：英文用 `(laugh)`、`(cough)`、`(clears throat)`、`(sigh)`，中文用 `[笑]`、`[咳嗽]`、`[清嗓子]`、`[叹气]`。参考文字与参考录音成对出现，并覆盖录音里实际说过的词。桌面增强是本仓库在此之上加的可选步骤，采样仍用 MiniCPM-V-4.7-35B-A3B-int4。35B int4 走与图像提示词改写相同的裸 Cube 后端 `CubeBackend<WgpuRuntime, f32, i32, u32>`，原因见 [图像改写文档](qwen-image-rewrite.md) 的「独立非 fusion 运行时」。

盲选里的一段要能被选中，须是正常结束并且交付了音频。停止、到达帧数上限、或缓冲区为空，整组盲选作废。这个判断在 `enhancement::candidate_is_choosable`，CPU 测试覆盖它。
2048～2050 是保留 token，2051 才是主生成器 EOS；合法码本 ID 为 0～2047。

波形解码保留每层卷积历史和 72 帧注意力窗口，只处理新产生的帧。
最后不足整块的帧也会输出，卷积末尾裁剪与官方因果解码一致。
旋转位置编码表常驻 GPU，并在同一前向过程的各层之间复用。
大词表分片上传，每片限制 8192 行，避免单个巨型 GPU buffer。
备用文本 embedding 和未使用的旧 codec 不上传到 GPU；`audio_tokenizer` 中的 Mimi 编码器仅在克隆时加载。

克隆使用官方 `ref_clone_tata` 模板：参考文字 → 参考音频码字 → 音频 EOS → 目标文字。
参考文字与目标文字分别执行文本编码，重置位置编码并隔离双向注意力；生成主干按拼接后的顺序处理。
参考音频 EOS 使用 16 路全为 `codebook_eos_token_id=0` 的 embedding。
加入声音描述时使用 `ref_edit_tata`：CFG 的无描述分支仍携带同一参考音频和参考文字。
不填写描述时只运行一个生成分支。

参考 WAV 支持整数/浮点 PCM 和多声道平均，使用已有原生 sinc 重采样到 24 kHz。
官方包装器对非 24 kHz 输入使用 librosa/soxr，因此此类输入不能保证码字逐个一致；24 kHz 输入可用于精确对照。
编码器以 1.6 秒为块复用卷积历史，每个非末块对齐全部卷积步幅，仅最后不足整帧的输入补齐。
末级 stride-2 卷积使用边界复制，其余卷积使用零填充，输出帧数为 `ceil(samples / 1920)`。
Mimi 注意力遵循固定参考环境 Transformers 4.57.3 的 eager/SDPA 实现，使用完整因果遮罩；
该版本调用 `create_causal_mask`，尽管配置含 `sliding_window=250`。长于 10 秒的样例覆盖了这一差异。

ASR 与 TTS 的 Qwen3 主干均为 28 层、2048 维，但各自 checkpoint 的权重不同；
第 0、14、27 层 Q projection 的 BF16 权重哈希对照见
`artifacts/breeze-tts/asr-tts-weight-comparison.json`。可共享设备与算子实现，模型参数分别加载。
TTS 另有独立的 T5Gemma2 文本编码器，不能与 ASR 的文本解码器视为同一个模块。

Burn 0.21 的卷积 autotune 会把部分短输入归入同一尺寸桶；本机最后单帧解码曾命中
不适用的 CMMA 配置。TTS 中不超过 32 个时间点的普通卷积和 1×1 卷积采用等价的
unfold/matmul，避开这个选择问题。该处理位于 TTS crate，未修改 Burn 或 ASR/ANE 算子。

输出使用预分配的单生产者/单消费者无锁环形缓冲，容量为设备采样率下的 60 秒单声道 PCM，
48 kHz 时约 11.5 MB。这与本地单次生成上限对应，允许较慢生成积累足够的播放余量。
工作线程以已有的预计算 sinc 滤波器重采样到设备采样率，CPAL 回调只读取样本、转换格式、
复制到各声道，不分配内存、不争抢生产者的锁，也不执行重采样或推理。
队列满时生产者等待通知。正常结束会排空重采样尾部，并按 CPAL 的播放时间戳等待最后一块
实际播放完再释放输出流；手动停止取消缓冲和排空等待，新任务启用前关闭旧输出流。

播放默认采用自适应缓冲：首包之后的音频到达间隔用于估计生成速度，排除首包编译/预填充时间，
结合文本预计时长计算启动余量，给速度波动留出 25% 余量；最小目标为 640 ms。
生成完成后立即放行短音频，不等待凑够缓冲。播放中若再次缺数据，则重新积累缓冲，减少碎片式续播。
再次试听已有 PCM 使用 200 ms 启动目标。界面显示缓冲状态、已准备的音频时长和设备首次播放时间，
“生成”耗时不包含最后的播放排空时间。音频设备时间戳不能测量蓝牙链路等设备外部的听觉延迟。

## 范围与限制

当前入口支持文本朗读、音色设计、带描述的参考声音克隆，尚未移植双 CFG 和多人对话模板。
仅 Makepad 本地版加入 TTS，已有浏览器 ASR 页面保持原有能力。
每次输入最多 2048 个文本 token，默认最多生成 750 帧，即 60 秒；正常结束由模型 EOS 决定。
达到上限会明确提示，CLI 可以通过 `--max-frames` 调整，最终仍受模型上下文长度限制。
合成期间随时可以手动停止，ASR 麦克风的持续录音规则保持原样。
参考音频上限为编码器 8000 个位置对应的 320 秒，参考与目标文字合计最多 2048 个 token。
该入口通过选择 WAV 提供参考声音，尚无直接录制参考声音的按钮。

流式接口不等于实时性能保证。首段耗时包括文本编码和 GPU 内核编译/调优，模型加载单独统计。
语音 token 自回归生成和音频解码共用 GPU，填写声音描述会增加 CFG 分支计算量。
RTF 小于 1 才表示该次测试的平均生成速度快于播放速度。自适应缓冲用更晚的首声换取连续播放，
并没有让模型推理本身变快；实际语速比预计慢、GPU 突然被其他任务占用等情况仍可能触发重新缓冲。
性能和验证结果应以本机实测为准，不能直接套用官方 H100 的 CUDA Graph 数字。

## 验证方法

源码参考固定为 [Breeze commit 58ec70c](https://github.com/breezeblue-ai/breeze-tts/tree/58ec70ce5fa4cc361bdebf77ec40d1365da00ab2)，
权重固定为 [Hugging Face revision 3e28c51](https://huggingface.co/BreezeBlue/Breeze-TTS-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4)。
下载脚本核对 LFS SHA256 和文件长度，写入 `download-manifest.json`。

官方参考环境使用 Python 3.12、PyTorch/torchaudio 2.9.1、Transformers 4.57.3、qwen-tts 0.1.1。
`scripts/reference_breeze.py` 使用官方模型实现生成 FP32 teacher-forced 张量以及音频参考；
这些开发依赖不参与 Rust 应用运行。

```sh
git clone https://github.com/breezeblue-ai/breeze-tts artifacts/breeze-tts-upstream
git -C artifacts/breeze-tts-upstream checkout 58ec70ce5fa4cc361bdebf77ec40d1365da00ab2
uv venv --python 3.12 .venv-breeze
uv pip install --python .venv-breeze/bin/python \
  torch==2.9.1 torchaudio==2.9.1 transformers==4.57.3 qwen-tts==0.1.1 \
  accelerate safetensors numpy soundfile
.venv-breeze/bin/python scripts/reference_breeze.py trace
.venv-breeze/bin/python scripts/reference_breeze.py codec
./target/release/breeze-tts trace --text '你好，欢迎使用本地语音合成。' \
  --output artifacts/breeze-tts/rust-trace.json
./target/release/breeze-tts codec --codes artifacts/breeze-tts/reference/codes.json \
  --chunk-frames 2 --output artifacts/breeze-tts/rust-codec.wav
.venv-breeze/bin/python scripts/compare_breeze.py \
  artifacts/breeze-tts/reference/trace.json artifacts/breeze-tts/rust-trace.json \
  --output artifacts/breeze-tts/trace-comparison.json
.venv-breeze/bin/python scripts/compare_breeze.py \
  artifacts/breeze-tts/reference/codec.wav artifacts/breeze-tts/rust-codec.wav \
  --output artifacts/breeze-tts/codec-comparison.json
```

原生交互回归由 `prototypes/makepad/validate-tts.py --output <新目录>` 执行。
测试使用自己启动的隐藏窗口和完整 Metal 推理，保存界面截图、事件和 WAV。
系统文件面板用仅注入测试进程的 shim 回答，输出使用静音 sink；这不等同于真实扬声器听音测试。
增加 `--device-output-muted` 会改用真实 CPAL 设备时钟、重采样和播放队列，只将最终输出幅度设为零，
同时检查连续播放、所有样本排空，以及播放中/尾音排空中/再次试听时停止。它验证时序和断流，不能代替人工音质试听。
另有 `playback::tests::default_output_drains_and_cancels`，向真实默认输出设备发送数字静音，
验证 CPAL 回调、播放结束等待和取消；需要本机音频设备，因此默认忽略。

## 本机验证结果

设备为 Apple M5 Max（40 核 GPU、128 GB 统一内存），使用 Metal。
详细输出保存在 `artifacts/breeze-tts/`；下列为单次样本，不代表所有文本的速度或音质。

| 数值检查 | 结果 |
|---|---|
| Tokenizer 与官方模板 | token ID 完全一致 |
| 七组 teacher-forced 张量，包括全部 15 路 depth logits、下一帧 backbone logits | 中文/英文两组共 14 个张量；相对 L2 误差 0.021%～0.184%，余弦相似度不低于 0.9999983 |
| 80 帧流式 codec 对照官方整段解码，覆盖 72 帧缓存窗口 | 153600 个采样一致长，相对 L2 误差 0.00214%，SNR 93.4 dB |
| 3 帧、按 2+1 分块的尾帧回归 | 5760 个采样，相对 L2 误差 0.000166% |
| GPU / CPU 采样，温度 0，同一文本 | 56 帧、全部 896 个码本 ID 完全一致 |

数值报告：`chinese-fixed-comparison.json`、`english-fixed-comparison.json`、`codec-fixed-comparison.json`、
`codec-tail-fixed-comparison.json`、`sampling-comparison-final.json`。

中文“你好，今天天气很好。”使用默认随机采样，生成 1.76 秒音频；
独立 CLI 进程的首块约 1.57 秒，总生成约 4.02 秒，RTF 2.28。
保留本机已有的内核缓存；加载模型的时间不计入这些数字，未命中的内核编译/调优计入。
生成结果回送本项目 ASR，得到“你好，今天天气很好”。

同句贪心生成 4.48 秒音频，GPU 采样耗时 6.95 秒，CPU 采样 8.56 秒，
这次对照减少约 19%。FP32 codec 单独解码 6.4 秒音频耗时约 2.90 秒，
当前整体仍慢于实时，主要耗时在自回归生成网络。

Makepad 同一进程连续操作的实测（模型与相应内核已预热）：

| 输入 | 音频长度 | 首块 | 总生成时间 | RTF |
|---|---|---|---|---|
| 中文，无声音描述 | 1.76 s | 0.46 s | 3.29 s | 1.87 |
| 英文，带声音描述、CFG 4 | 2.96 s | 0.64 s | 6.40 s | 2.16 |

中文及英文生成音频回送现有 ASR，分别得到原句“你好，今天天气很好”和
“Hello, welcome to local speech synthesis.”。这验证了两个样例的可识别性，
不替代人工音质、音色匹配或多语料评估。

`ui-verified/report.json` 的 7 组交互检查通过：缺失模型选择与取消、加载进度和路径记忆、
空文本校验、流式音频、停止与重启、部分/完整 WAV 保存、中英文合成及返回 ASR。
此次生成中停止和首段前停止分别约 58 ms、59 ms；音频保存于同目录的
`chinese.wav`、`english-description.wav` 和 `stopped.wav`。
`asr-regression/native.json` 另记录了 10 项检查、12 轮 ASR 运行，包含 20.22 秒连续录音。
29 个受保护的 ASR/ANE/Web 源文件以及原有 `target/release/r2t2` 二进制哈希保持一致。
最终二进制、源码与验证报告的索引见 `artifacts/breeze-tts/delivery.json`。

### 参考声音克隆验证

本次新增验证保存在 `artifacts/breeze-clone/`，此前结果保留于 `artifacts/breeze-tts/`。

| 检查 | 结果 |
|---|---|
| 2.96 秒参考 WAV | 37 帧、592 个码字与官方完全一致 |
| 11.8457 秒 WAV，跨越 250 帧注意力边界且末尾不满一帧 | 149 帧、2384 个码字与官方完全一致 |
| 卷积、Transformer、量化前 latent | 两组共 6 个张量，相对 L2 误差不超过 0.00105% |
| 普通克隆 / 带声音描述克隆的提示词 | token ID 与官方模板一致；各 7 组 FP16/FP32 对照张量的相对 L2 误差不超过 0.152% |

参考音频来自本项目之前生成的英文样例，目标为中文，覆盖跨语言条件输入。
这些结果验证实现与官方数值接近，不代表已完成人工音色相似度评估。

`ui-initial/report.json` 的 11 组本地交互检查通过，覆盖普通 TTS 回归、参考文件选择/取消、
音频与文字配对检查、参考准备阶段停止、克隆与带描述克隆完整生成、编码缓存复用、
损坏 WAV 恢复、清除参考、保存音频，以及完成 TTS 后返回 ASR 转写。
此次参考准备阶段停止约 212 ms，生成阶段和首段前停止均约 53 ms。
参考编码、生成网络和波形解码均为真实 Metal 推理；音频输出为测试静音 sink。
两段克隆输出分别为 1.68 秒和 2.16 秒，回送 ASR 均识别为“你好，欢迎回来”。
首次完成克隆的首段约 4.74 秒（含参考编码），带描述并命中参考缓存时约 2.35 秒；
这是功能测试中的单次测量，仍慢于实时，不作为稳定性能基准。

6 项 TTS 单元测试、9 项原生应用单元测试、Clippy 和 WGPU 编译检查通过。
冻结的 ASR/ANE 源码与 `target/release/r2t2` 校验值保持一致；本次交付索引为
`artifacts/breeze-clone/delivery.json`。

复现编码器对照：

```sh
.venv-breeze/bin/python scripts/reference_breeze.py encode \
  --reference-audio reference-24k.wav --output artifacts/reference-encoder
./target/release/breeze-tts encode --audio reference-24k.wav \
  --output artifacts/reference-encoder/rust.json
.venv-breeze/bin/python scripts/compare_breeze.py \
  artifacts/reference-encoder/encode.json artifacts/reference-encoder/rust.json \
  --output artifacts/reference-encoder/comparison.json
```

`trace` 命令也接受 `--reference-audio` / `--reference-text`；开发对照可用 `--reference-codes` 载入预编码 JSON。

### 音色设计界面验证

`artifacts/breeze-design/ui-initial/report.json` 记录了 18 组真实 Metal 推理的界面回归，
包含原有普通合成/克隆流程和新增音色设计入口、描述必填、描述示例、候选切换、
已有 PCM 重播、设计音色复用、生成中修改输入的快照隔离、停止后禁止选用残缺结果。
测试在朗读模式留下损坏的参考 WAV，再进入设计模式成功生成，确认设计独立于参考输入。
重播前后保存的 WAV 哈希一致；相同描述和文字下，候选 1/2 的种子为 42/43，输出 WAV 不同。
选用候选 1 后继续生成新的文字，使用已缓存的参考码字；尝试和中断新候选后，原先选用的音色仍保留。

两次设计分别输出 6.0 秒和 5.2 秒音频。候选 1 的首段 11.74 秒、总生成 27.79 秒，
包含此次新输入形状的首次算子编译/调优；候选 2 首段约 0.44 秒、总生成 14.34 秒。
使用所选音色朗读新文字得到 1.84 秒音频，首段约 0.73 秒、总生成 4.53 秒。
这组功能测试未隔离冷启动和热缓存，不能直接作为候选间性能比较；整体生成仍慢于实时。
设计在首段前停止约 57 ms。输出使用静音 sink，音色遵从度与相似度未作人工听音评估。

回送 ASR 的原始结果保存在 `artifacts/breeze-design/asr-roundtrip.json`，本次不记为逐字一致通过：
候选 1 的“你好”被识别为“您好”，候选 2 与原句一致；复用音色的 1.84 秒短句直接识别为“你好，欢迎”，
仅为诊断追加一秒尾部静音后，识别为“你好，欢迎回来”。音色设计交付时的 Makepad 转写也复现了未补静音短句的句尾差异。
这些结果说明自动回送识别有边界和用词差异，不能代替试听判断。原始合成 WAV 保留不变，此次未修改 ASR 流程。

9 项原生单元测试及 Clippy、格式检查通过。此改动仅增加界面和本地任务流程，TTS 模型代码
与此前交付的 13 个 crate 文件哈希一致，29 个受保护 ASR/ANE/Web 源文件保持一致。
二进制与源码交付索引见 `artifacts/breeze-design/delivery.json`。

后续 ASR 收尾修复已在四个本地档位中，使同一原始 1.84 秒 WAV 完整输出“你好，欢迎回来。”；
原始合成音频未改，虚拟补尾不计入录音时长。“你好/您好”的识别差异仍存在。
修复及更广的输入场景、语料和界面验证见[ASR 收尾记录](native-pipeline-optimization.md#后续修复短音频结束时漏掉句尾)。

### 后续修复：流式播放断续

原实现每到一块约 160 ms 音频就开始播放，而本机生成经常慢于实时；队列耗尽后只能输出静音。
此外，旧设备回调的 `try_lock` 失败也会直接产生整块静音。现在采用上述无锁输出、工作线程重采样
和自适应缓冲，保持原来的两帧分块、采样参数和模型实现。

先用原有 CLI 对普通朗读和声音描述分别比较 2/4/8 帧分块。后一轮单次 RTF 分别为
1.574/1.738/1.529 和 1.839/1.788/1.768；它们来自独立进程，缓存状态不同，不能作为严格性能结论。
更大分块仍未达到实时，而且首次新形状可能触发较长的编译。相同模式下码字一致，
两帧与其余分块的 WAV 差异 RMSE 小于 4.1e-8；此次保留两帧，优先修复播放调度。

将 12 次实测到达时间，在 16/24/44.1/48/96 kHz 的 10 ms 模拟回调中回放，
共 120 个新旧调度对照，全部样本保持完整。默认两帧的 4 条时间线、5 种采样率共 20 个场景，
旧即时播放策略每次断流 41–58 次，新策略均为 0 次。以下是其中后一轮 48 kHz 的结果，
这里的“首声”是调度模拟时间，不是真实设备听音测量：

| 模式 | 音频 / 生成时长 | 即时播放首声 / 断流次数 | 自适应首声 / 断流次数 |
|---|---|---|---|
| 普通朗读 | 9.44 / 14.86 秒 | 2.13 秒 / 58 | 7.15 秒 / 0 |
| 声音描述 | 6.72 / 12.36 秒 | 1.77 秒 / 41 | 7.73 秒 / 0 |

所有 12 条时间线中有一条四帧对照仍出现一次约 0.49 秒重新缓冲；该分块未用于本地默认实现。
测试说明缓冲能改善这批样本的连续性，不能保证任意文本或任意 GPU 争用下均无断流。
原始日志、到达时间、波形比较和模拟结果保存在 `artifacts/tts-playback/`。

真实 Makepad 窗口、Metal 推理及 48 kHz CPAL 回调的 23 组交互检查通过，报告在
`artifacts/tts-playback/ui-final/report.json`。其中 7 段完整合成覆盖中英文、带描述、参考克隆、
设计候选和复用音色，均为 0 次断流，设备收到的样本时长与保存的音频一致；测试将设备输出静音，
没有做人工听音评价。6.0 秒设计音频首段生成 0.61 秒、实际首声 6.57 秒、生成完成 9.44 秒，
说明该例在生成完成前开始连续播放，同时也展示了启动缓冲的等待代价。
普通中文短句的 1.76 秒音频在 3.21 秒生成完成后开始播放，短句允许在 EOS 时立即释放缓冲。

播放中、再次试听中、生成结束后的尾音播放中停止，界面恢复耗时分别约 56/54/53 ms；
连续停止后的重播完整排空，ASR 官方例句仍完整转写。13 项普通单元测试、1 项真实默认设备测试、
Clippy 和格式检查通过；另验证 24 kHz 输入在五种输出采样率下按不同边界分包的重采样结果一致。
TTS 模型的 13 个 crate 文件、29 个受保护 ASR/ANE/Web 文件、上一轮 ASR 收尾改动以及冻结的 CLI
二进制均保持原哈希。新原生二进制与报告索引见 `artifacts/tts-playback/delivery.json`。

进一步缩短首声与提升生成吞吐的方案，见[流式优化调研](breeze-tts-optimization-research.md)，
涵盖固定 depth 工作区、Burn 0.22 图重放、M5 TensorOps、量化及验证目标。

后续按照“不做权重量化”的约束进行了实现与对照，见[流式优化实施记录](breeze-tts-streaming-optimization.md)。
采用 GPU 码字和 embedding 常驻、噪声合批与播放预测改进；没有稳定收益的算子合并和固定 KV
候选保留为实验记录，未设为默认。

随后在 depth decoder 接入了 Metal GEMV 与 RMSNorm 专用内核，继续保持原有精度。
配对热测的普通合成 / 声音描述耗时分别减少 6.5% / 8.1%，CPU 累计时间分别减少 12.2% / 9.4%；
仍未达到实时吞吐。实现边界、原始数据和复现方式见[Metal depth 优化](breeze-tts-metal-depth-optimization.md)。

```sh
CARGO_TARGET_DIR=target/makepad-native cargo run --offline --locked --release \
  --manifest-path prototypes/makepad/Cargo.toml --example playback-trace -- \
  artifacts/tts-playback/traces.json
python3 prototypes/makepad/validate-tts.py --device-output-muted --output <新目录>
```

## 许可

Rust 源码采用 Apache 2.0，参考代码归属见 [`NOTICE`](../crates/breeze-tts/NOTICE)。
Breeze-TTS-2 权重、派生模型及本地生成内容单独受研究和非商业许可约束；代码的 Apache 许可
不授予这些模型材料的商业使用权。详情见[官方模型卡](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
和[模型许可](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/3e28c5151381a722f1d8661b4118c298caa77aa4/LICENSE)。
