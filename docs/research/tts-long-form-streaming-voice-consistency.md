# TTS 流式生成能否做到十分钟以上并保持音色一致

2026-11-24。问题是：「tts 流式生成能生成很长的音频吗？10 分钟以上，并且能保证音色一致吗？」材料只有第一手来源：本仓库工作树（`git rev-parse HEAD` 为 `1373a67ae37351e6b1f052ccc64f0909648fb44b`；树是脏的，被引用文件中 `docs/breeze-tts.md`、`docs/breeze-tts-metal-depth-optimization.md`、`crates/breeze-tts/src/bin/breeze-tts.rs`、`prototypes/makepad/src/tts_ui.rs`、`crates/omni-api/src/worker.rs` 有未提交修改，本地文件按工作树状态读取）；上游 `breezeblue-ai/breeze-tts` 的固定提交 `58ec70ce…` 与 `main` 头 `a43d19e7…`；[模型卡](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/3e28c5151381a722f1d8661b4118c298caa77aa4/README.md)修订 `3e28c515…`（本地 `models/Breeze-TTS-2/` 被 gitignore，哈希与 `download-manifest.json` 一致，也是 Hugging Face 当前头）；第 8 节各系统的官方 README 与文档。远端内容均于 2026-11-24 读取，没有采用博客、评测或聚合页。下文 `lib.rs` 等在 `crates/breeze-tts/src/`，`tts_engine.rs` 等在 `prototypes/makepad/src/`。

**结论：三层都是「现在不能，也没有谁保证音色」。** 本仓库一次生成默认 750 帧即 60 秒，Makepad 界面没有调高的入口；没有分句、分段排队或跨段续接，报错和上限提示都让用户「分段合成」；播放环形缓冲只有 60 秒，满了就卡住生成。上游的流式同样是一次生成边算边出 PCM，CLI 与 API 上限 1500 帧即 120 秒、总序列 2048，`main` 头新增的批量接口是离线并行合成，段与段之间不传状态。音色方面没有来源使用「保证」：本仓库文档明写同一描述或种子不保证换文本后音色相同，也没量过相似度；上游只有一句克隆能力描述；其他系统报告的是指标或经验。本仓库要出十分钟音频，缺的是文本分段与排队、可跨段复用的音色条件、超过 60 秒的生成入口及其验证、不依赖「整段装进缓冲」的播放调度、音色相似度测量。

## 1. 「流式」指什么，以及本仓库的上限

「流式」有三种含义：(1) 一次生成的 PCM 边算边送；(2) 长文本逐段送入、逐段合成后接成一条长音频；(3) 全双工对话，如 Moshi（`e6a55d27…`）。本仓库与上游的流式都是 (1)，下表上限都只作用于一次生成；(2) 两边都没有。

| 上限 | 数值 | 设定处 | 性质 | 到达时 |
|---|---|---|---|---|
| 生成帧数 | 750 帧，即 60 秒（每帧 1920 采样） | `SynthesisOptions::default`，`lib.rs:95` | 默认值。只有 CLI 的 `--max-frames` 可改（`bin/breeze-tts.rs:38`）；Makepad 界面沿用默认（`tts_ui.rs:1127-1135`）；无环境变量 | 上限帧照常交付后停止，`reached_limit: true`（`synthesis.rs:261-272`），音频保留可保存；界面提示「已达到本段 60 秒生成上限 · 请将剩余文本分段合成」（`tts_engine.rs:574-575`） |
| HTTP API 帧数 | `1..=1500` 帧，即 120 秒；`text` 不超过 32768 字节 | `Speech::validate`，`crates/omni-api/src/worker.rs:71-73` | 硬上限，默认 750 | 超出返回 400；到达时同上一行 |
| 文本 token | 2048 | `prompt_ids`（`lib.rs:330`）、`prepare_prompt`（`lib.rs:376`） | 硬上限 | 合成前报错「文本和声音描述合计不得超过 2048 个 token，请分段合成」 |
| 参考音频 | 320 秒 | `MAX_REFERENCE_SAMPLES = 8000 * 960`，`reference.rs:14`、`48-55` | 硬上限 | 读文件时报错「参考 WAV 必须为 8–384 kHz、非空且不超过 320 秒；建议使用 3–10 秒清晰人声」 |
| 主干位置数 | 40960 | 模型 `config.json`；`limit = max_frames.min(40960 − prompt.ids.len())`，`synthesis.rs:148-150` | 硬上限 | 提示超过它时报错「参考音频和文本超过模型上下文长度」（`lib.rs:389-391`） |
| 播放环形缓冲 | 60 秒单声道 PCM | `CAPACITY_SECONDS`，`playback.rs:18-20` | 编译期常量 | 生产者等待，不覆盖、不丢弃、不报错（第 4 节） |

文档原话：「每次输入最多 2048 个文本 token，默认最多生成 750 帧，即 60 秒；正常结束由模型 EOS 决定。」（[docs/breeze-tts.md](../breeze-tts.md) 第 150 行）。EOS 是主干采样到 ID 2051（`synthesis.rs:181-183`），2048–2050 是保留 token（`sampling.rs:98-102`）。仓库记录过的最长生成是 55.12 秒。

## 2. 哪个上限先到

帧上限先到，差一个数量级以上。仓库和上游都没有给出每字 token 数或语速，以下是本文实测与推断。用仓库已有的 `.venv-breeze`（`tokenizers 0.22.2`）加载固定修订的 `tokenizer.json`，按 `prompt_ids` 同样的往返计数：`prototypes/makepad/validate-tts.py:257-262` 的长段文本 154 个汉字得 115 个 token，文档记录的长段输出为 49.2 秒和 55.12 秒，即每秒 2.8–3.1 个汉字；一段中文说明文每字 0.69 个 token。据此 60 秒约合 170–190 个汉字、120–140 个 token，2048 个 token 约合 2960 个汉字。英文说明文每词 1.23 个 token，没有实测语速，播放估计用每秒 2 个词（`playback_buffer.rs:145`）。

## 3. 没有跨段机制

`crates/breeze-tts`、`tts_engine.rs`、`tts_ui.rs`、`playback.rs` 里没有分句、分段队列或续接逻辑。可借用的零碎：

- 参考码字缓存：单条，按参考 PCM 逐样本相等匹配（`tts_engine.rs:185`、`404-405`），命中后各请求喂给主干的是同一组码字。
- 设计音色只在当前会话：试听 PCM 经参考编码器得到 `designed_reference`（`tts_engine.rs:634-681`）。CLI 的 `--reference-codes` 可把码字存成 JSON 跨进程复用（`bin/breeze-tts.rs:90-97`）。
- `SynthesisBatch`：最多 16 个互相独立的请求按帧合批，成员之间没有顺序或状态传递（`batch.rs:22-36`）。
- 环形缓冲不限制总长度：再次试听就是分块推入任意长的 PCM（`tts_engine.rs:606-621`）。

## 4. 播放路径

`CaptureBuffer::push` 只在一次写入超过剩余空间时返回 `Err("overflow")`（`capture_buffer.rs:34-36`），而 `write_pcm` 每轮最多写剩余空间那么多，写不进就等 50 ms（`playback.rs:157-179`）。文档原话：「队列满时生产者等待通知。」（`docs/breeze-tts.md:137`）。今天的 10 分钟输出到不了这里，生成在 60 秒就停了。本文推断：若把上限抬到 10 分钟，生成每攒满 60 秒被卡住；起播门槛初值是整个缓冲（`playback_buffer.rs:32`），RTF 大于 1 时余量按 `1 − 1/RTF` 被吃掉，RTF 1.6–2.22 时约 110–160 秒后断流。`docs/breeze-tts-optimization-research.md:34` 有同样的判断。

## 5. 音色条件与一致性

音色设计只有文字描述，与不带描述的分支做单路 CFG（`lib.rs:311-315`、`synthesis.rs:118-136`）。克隆走 `ref_clone_tata`：参考文字的编码结果、参考码字 16 路 embedding 之和加音频 EOS、目标文字的编码结果，拼成一条序列，一次预填充进主干 KV cache（`lib.rs:398-470`）；此后每一步只把新帧的 embedding 喂给主干（`synthesis.rs:273-287`）。本文推断：参考条件是留在 KV cache 里的前缀 token，每一步都被注意到；代码里没有单独的说话人向量或 ID。

采样状态不跨生成：每次生成新建采样器，状态就是种子（`sampling.rs:86-87`、`synthesis.rs:112`）；界面朗读固定用种子 42，音色设计用候选序号加 41（`tts_ui.rs:1081-1085`）。

- 一次生成之内：机制如上，仓库没有量过。长段验收只断言播放无断流（`prototypes/makepad/validate-tts.py:270-273`），文档写「时序通过不等于已经人工评价音色质量。」（`docs/breeze-tts-streaming-optimization.md:160`）。
- 同一长输出的各段之间：没有分段，也就没有这项声明。
- 跨请求、跨会话：「同一描述或随机种子不能保证换一段文字仍有完全相同的音色；需要持续使用某个设计结果时，使用参考音色复用流程。」（`docs/breeze-tts.md:52`）。复用流程是把试听 PCM 当参考音频，跨会话要重新载入保存的 WAV。相似度没有数字：「不代表已完成人工音色相似度评估」（`docs/breeze-tts.md:257`），研究文档把「音色一致性」列为待测项（`docs/breeze-tts-optimization-research.md:203`）。

## 6. 耗时与入口

M5 Max 长段实测：55.12 秒音频生成 88.07 秒，RTF 约 1.60，首声 43.45 秒（`docs/breeze-tts-metal-depth-optimization.md:95-96`）；49.2 秒音频生成 109.31 秒，RTF 2.22，首声 79.05 秒，文档写「持续 RTF ≤ 0.8 和热首声约一秒仍未达到。」（`docs/breeze-tts-streaming-optimization.md:131-133`）。本文推断：按 RTF 1.6–2.2 线性外推，10 分钟音频要生成 16–22 分钟；文档提醒短测不能外推到长段。

Makepad 每次点「合成」提交一条文本，选项取默认值（`tts_ui.rs:1126-1139`）；到达上限的结果不能选作音色（`tts_engine.rs:586-590`）。HTTP API 的 `POST /v1/audio/speech` 是异步任务，不接受参考音频（`crates/omni-api/src/worker.rs:50-80`），PCM 不向客户端流式推送（`worker.rs:261-264`）。

## 7. 上游与模型卡

固定提交 `58ec70ce…`（2026-09-30），路径相对仓库根：

- 上限是常量 `MAX_NEW_TOKENS = 1500`、`MAX_SEQ_LEN = 2048`（[infer.py](https://github.com/breezeblue-ai/breeze-tts/blob/58ec70ce5fa4cc361bdebf77ec40d1365da00ab2/infer.py) 第 25–27 行，`breeze_infer/api.py` 第 40–42 行），即 120 秒、提示加生成帧数共 2048 个位置；`models/fast_streaming.py` 第 33–34 行的默认值另为 750 和 1024。循环在主干 EOS 或 `prefill_len + step_idx >= self.config.max_seq_len - 1` 时 `break`（第 849–854 行），已生成的 PCM 照常送完，不报错。README 没有分句、长文本或单次时长的说明。
- 模板四个：`tts_plain`、`tts_instruction`、`ref_clone_tata`、`ref_edit_tata`（[breeze_infer/templates.py](https://github.com/breezeblue-ai/breeze-tts/blob/58ec70ce5fa4cc361bdebf77ec40d1365da00ab2/breeze_infer/templates.py) 第 108–132 行）。多说话人靠请求的 `speaker` 字段渲染成 `[S0]` 这类前缀，API 写死 `"speaker": "S0"`；检索 `dialog`、`[S1]`–`[S9]` 无结果。
- 双 CFG 只在 `models/generation_breeze.py` 里实现，CLI 与 API 都传 `guidance_scale_ref=None`，流式运行时直接拒绝（`tests/test_fast_streaming.py:106-108`）。
- 一致性只有一句能力描述：「Voice Clone — Uses reference audio with its exact transcript to preserve timbre, rhythm, emotion, and style.」

`main` 头 `a43d19e7…` 多 4 个提交，相关变化是新增 `POST /v1/audio/speech/batch`（[README](https://github.com/breezeblue-ai/breeze-tts/blob/a43d19e718b322e1cb3b2aa66f6d37d7945f4b21/README.md)）：离线批量，「`texts` is a JSON array of up to 128 non-empty strings.」「`max_new_tokens` is clamped to the server limit of 1500.」每段各自成一条提示，共用同一份 `instruction`、`ref_audio`、`seed`，结果「concatenated in request order」，不是逐段续接。

模型卡修订 `3e28c515…`：语言 `en`、`zh`；没有时长上限、长文本、一致性或评测数字。

架构事实：帧率 12.5。`sliding_window=250` 属于参考编码器（Mimi），不属于主干；主干 `sliding_window` 为 `None`，上游用 `create_causal_mask`（`models/breeze.py` 第 570、874 行）。移植对这个编码器也用完整因果掩码（`crates/breeze-tts/src/transformer.rs:108-111`），文档称「长于 10 秒的样例覆盖了这一差异」（`docs/breeze-tts.md:122`）。主干 `max_position_embeddings = 40960`，顶层另有 `2048`。移植把 2048 用作纯文本上限，上游用作总长；本文推断，长文本加满帧数时移植会超出上游流式路径允许的序列长度，仓库里没有这种组合的验证。上游 CLI 与 API 默认 `REPETITION_PENALTY = 1.1`，移植没有重复惩罚（在 `crates/breeze-tts/src`、Makepad 的 TTS 文件和 `crates/omni-api/src` 检索 `repetition_penalty` 无结果）。

## 8. 其他系统

| 系统 | 单次上限（原文） | 长音频机制 | 音色一致机制 | 保证还是指标 |
|---|---|---|---|---|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/README.md) | 未写；`max_new_tokens` 交给 `model.generate` | 未写 | `create_voice_clone_prompt` 建一次可复用的参考提示 | 用法建议 |
| [Fun-CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/cli/frontend.py) | 文本先切成 `token_max_n=80, token_min_n=60, merge_len=20` 的段 | 逐段合成，每段用同一份 `prompt_text`、`prompt_wav` | 同一份参考提示 | 指标：说话人 embedding 余弦相似度 |
| [VibeVoice](https://github.com/microsoft/vibevoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/README.md) | Realtime：「8k context window( ~10 minutes audio generation)」 | TTS 版「up to 90 minutes long with up to 4 distinct speakers」，但「we have removed the VibeVoice-TTS code from this repository」；Realtime 流式文本输入 | 「maintaining speaker consistency」 | 声明加指标 |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM/blob/300e4260b509066151d623dc9c0aa922cf82232b/src/voxcpm/core.py) | `max_len: int = 4096` 个 LM token，6.25 Hz，本文推算约 655 秒 | `generate_streaming` 逐块输出 | `prompt_wav_path` 加提示缓存 | 指标表 |
| [OpenAI](https://github.com/openai/openai-openapi/blob/373f26a9aec3608e3158f2a0d4b8ae49d13162a4/openapi.yaml) | 「The maximum length is 4096 characters.」 | 未写；分块传输边生成边播放 | 预设音色 ID，或由音频样本建的定制音色 | 无 |
| [ElevenLabs](https://elevenlabs.io/docs/overview/capabilities/text-to-speech) | 按模型：v4「100,000 character limit」、Multilingual v2「10,000 character limit」、v3「5,000 character limit」 | 「Split long text into segments and use streaming」，段间带 `previous_text`、`next_text` | `voice_id`；「The models are nondeterministic.」 | 明确不保证确定性 |
| [Azure Speech](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-synthesis) | REST「the audio length can't exceed 10 minutes」 | 批量合成「can create synthesized audio longer than 10 minutes」 | 预设命名音色 | 无 |
| [Google Cloud TTS](https://docs.cloud.google.com/text-to-speech/docs/create-audio-text-long-audio-synthesis) | 「Total bytes per request 5,000」 | 「Long Audio Synthesis asynchronously synthesizes up to 1 million bytes on input.」 | 预设命名音色 | 无 |
| [Amazon Polly](https://docs.aws.amazon.com/polly/latest/dg/limits.html) | 「up to 3000 billed characters (6000 total characters)」，输出「limited to 10 minutes」 | 异步 `StartSpeechSynthesisTask`「up to 100,000 billed characters」 | 预设命名音色 | 无 |

开源系统里单次生成就能超过十分钟的是 VibeVoice 和 VoxCPM2，其余要么由调用方切段（CosyVoice、ElevenLabs），要么走离线批量接口（上游新端点、Azure、Google、Polly）。切段后保持音色的做法都一样：每段带同一份参考，可选再带前后文。没有一家使用「保证」。

## 9. 空检索与未核实项

- 空检索：`crates/breeze-tts` 与 Makepad TTS 文件里的分句、分段逻辑（第 3 节）；`repetition_penalty`（范围同第 7 节）；上游固定提交里的 `dialog`、`[S1]`–`[S9]`；`sesame/csm`（GitHub API 返回 404）。
- 未入表的线索：Chatterbox（「retired research and is no longer maintained」）；Higgs Audio v2（README 头部转向 v3）；IndexTTS-2、Kyutai TTS（`4c4f65e1…`）、Fish Speech（`0f8a12e8…`），后三者 README 没有时长上限的数字。MiniMax-Speech、Step-Audio 未查。Breeze TTS 2 未找到作者论文或技术报告（检索 `"Breeze TTS 2" BreezeBlue technical report OR paper arXiv`；Hugging Face 仓库无 arXiv 标签）。
- 本仓库：`artifacts/` 下只有 `api-output`，文档引用的原始报告（如 `artifacts/tts-streaming-opt/ui-verified2/report.json`）不在工作树，文档数字无法对照原始记录。49.2 秒与 55.12 秒两次长段出自同一检查脚本是本文推断。
- 未核实：超过 55 秒的生成在音质、音色和 EOS 上的表现；移植与上游在长序列上的数值差异；M5 Max 以外硬件的速度。没有做任何听音或相似度测量。
