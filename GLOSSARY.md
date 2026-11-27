# Omni inference

Speech recognition, translation, speech synthesis and image generation run locally from a model checkpoint.

## Language

**Checkpoint**:
The model configuration, tokenizer and named learned tensors that together describe a trained model.

**Checkpoint layout**:
The expected names, dimensions and loading stages of the tensors in a checkpoint.

**KV cache**:
Attention keys and values retained for already processed positions in one decoding history. Reusing a prefix is valid only while the corresponding history remains unchanged.

**Session**:
One stream of speech recognition with its own audio history and decoding state. Sessions share a model but never share their evolving transcript.

**Omni conversation**:
An ordered exchange in which a user turn can combine text, images, video and audio, and an assistant turn contains text, speech, or interleaved text and speech. Video frames and their soundtrack retain their relative timing.
_Avoid_: transcription session

**Speculative cycle**:
One DSpark round: propose (draft block from the markov head), verify (one target forward over the block), accept (longest prefix matching the target's greedy argmax, plus one correction token), retain (KV rollback to the accepted prefix). The draft block length adapts over the ladder 3/8/16 by measured acceptance.

**τ (acceptance length)**:
Accepted draft tokens per verify cycle (`draft_tokens_accepted / cycles`). Content-dependent — ~2 on hard-to-predict prose, ~7.7 on repetitive code for the 35B int4 model — and the dominant lever for tokens per second once verify sits at its bandwidth floor.

**Calibrated verify size**:
The chain length or best-first live-row count chosen by the ledger's rule `E(n) − rate · cost[n]` from measured per-depth acceptance and verify-phase host wall clock. Bounded to `[3, 8, 16]` for chains and `[2, 6, 14]` for best-first trees so each verify shape stays autotuned; active only when `MINICPM_DSPARK_LEDGER=1`.

**Conversation checkpoint**:
A self-contained snapshot of one conversation slot's KV and linear state together with its KV-aligned token history, stored at a turn boundary and at prefill end. A later request restores the deepest checkpoint that is a strict prefix of its prompt and prefills only the suffix.

**State cache namespace**:
The hash that scopes persisted conversation checkpoints to the weights that produced them: model file names, sizes and modification times plus the config hash, the optional drafter directory, the state-layout version and the block row count. Entries from another namespace are ignored.

**Confidence prefix**:
The ancestor-closed set of candidate-tree nodes selected by best-first expansion, each scored by the summed log-softmax transition score along its root path. Choosing its size is choosing the verify prefix; there is no separate confidence threshold.

**Transcription**:
A complete speech-recognition run from incoming audio to committed text, including how input ends and the run's timing statistics.

**Committed text**:
Transcript text that will not be revised by later audio. Concatenating the emitted deltas reproduces this text.
_Avoid_: partial hypothesis, tentative text

**Normal end**:
The end of an audio source, after which remaining audio and the final decoding budget are processed.

**Interruption**:
A request to stop decoding and retain committed text, discarding audio that has not yet been processed.
_Avoid_: normal end, pause

**Learning utterance**:
A completed user-provided speech segment considered for local speech training. It is not verified training data until its audio and provenance have passed admission.
_Avoid_: transcription, ground truth

**Real-time factor (RTF)**:
Time spent in decoding divided by the duration of audio represented in the transcript events. Model loading and waiting for audio are excluded.

**Translation**:
One text-to-text generation from an immutable source-text snapshot and a target language. The translated text is separate from the source; using it as speech-synthesis input is an explicit action.

**Radix prefix cache**:
The index of completed translation token prefixes for one loaded model. Each indexed span records the KV pages that hold it, not a separate copy of the dense attention tensors. A later translation reuses the longest matching prefix and locks it until that translation finishes. Eviction drops only an unlocked leaf, and it never recycles a KV page that a translation still holds. Audio and image embeddings cannot be identified by text tokens alone, so they are never indexed here.
_Avoid_: dense KV trie

**KV page**:
A fixed-size group of attention-history positions whose physical location is independent of its logical position in a translation. Shared prefix pages remain unchanged until every translation using them has finished or acquired its own writable copy.

**Continuous translation batch**:
A changing set of translations that compute their next prompt or output tokens together while retaining independent histories, sampling and cancellation. New translations can join between computation steps, and completed translations release their place immediately.

**Decode history**:
The private attention state for an autoregressive sequence. A translation's decode history is the KV pages that translation currently holds. Sharing those pages with later translations is the radix prefix cache, not the decode history. Speech recognition can rewind its uncommitted suffix; speech synthesis extends its backbone history frame by frame. Metal page storage is optional and does not imply cross-request prefix reuse.

**Continuous transcription batch**:
Offline audio sources sharing text-decoder steps while retaining independent resampling, audio encoding, committed transcripts and termination. A live source that blocks waiting for microphone input cannot join this executor.

**Continuous synthesis batch**:
Speech requests sharing semantic-head and depth-decoder work at a codec-frame boundary. Each request owns its backbone history, sampler, audio codec and output. Finishing one voice does not reset another voice's state.

**Speech script**:
The complete text a speaker submits for synthesis in one action, spoken in order as one or more speech texts.
_Avoid_: speech text, long text, passage, prompt

**Speech text**:
The words one speech synthesis is asked to speak, including official inline vocal events.
_Avoid_: voice instruction, enhancer prompt, prompt, speech script

**Voice instruction**:
Optional natural-language direction for how a speech synthesis should sound, in the same language as the speech text. Empty means that synthesis has no voice instruction.
_Avoid_: speech text, reference transcript, enhancer prompt, prompt

**Reference transcript**:
The words actually spoken in a reference recording. It stays paired with that recording.
_Avoid_: speech text, voice instruction

**Voice-conditioning**:
The optional voice instruction and optional reference recording, with its reference transcript. It is fixed for one speech synthesis, and every speech synthesis of one speech script shares the same one.
_Avoid_: speech text, enhancer prompt

**Voice anchor**:
The one reference recording and reference transcript that every speech synthesis of a speech script shares. It is the recording the speaker supplied, or the opening of the script's own audio when none was.
_Avoid_: per-speech-text reference, anchor segment, voice drift

**Enhancer prompt**:
The standing instructions for rewriting a speech text and a voice instruction. It is neither the speech text nor the voice instruction.
_Avoid_: speech text, voice instruction, prompt

**Speech choice**:
A person's selection of one completed speech synthesis from the alternatives produced together for the same speech text and voice-conditioning.
_Avoid_: rating, score

**Speech synthesis**:
One generation of speech from a speech text and voice-conditioning snapshot, including its independent sampling, generated frames and delivered audio. A single request and a continuous synthesis batch stop at the same frame ceiling: pending audio is delivered, and the backbone is not continued past that frame. An interruption retains audio already delivered and discards audio that has not been delivered.

**First audio latency**:
Time from the start of speech synthesis to the beginning of its first successful audio handoff. It excludes the recipient's processing time and any delay before physical playback.

**Diffusion trajectory**:
One image request's evolving latent, timestep, conditioning and optional negative-guidance branch. Several trajectories can share DiT projections and MLPs even when their dimensions or timesteps differ; their attention histories remain separate.

**Observation row**:
One diffusion trajectory inside a denoising observation. The rows share one prompt, seed, canvas and initial noise, and stay in order: official text conditioning, then the fitted MiniCPM-V-4.7-1B conditioning, then the fitted MiniCPM-V-4.7-35B conditioning.
_Avoid_: a separate image request

**Inference job**:
One admitted HTTP request with its own progress, result and cancellation handle. Execution returns the result value and any files still to be written; it does not mark the job terminal. The terminal state is published once, after execution releases request resources and any result file is published. A cancellation can retain completed partial output; an execution or publication failure remains an error.

**Published result**:
A completed audio or image file whose download link remains valid while the file is retained, independently of inference-job history and application restarts. For a cancelled speech inference job, a published result requires at least one handed-off audio sample and contains every sample handed off before the interruption.

## 执行位置

**Compute backend**:
The portable Burn execution target (`wgpu` or `cpu`) that runs model graphs without platform-specific hardware calls.
_Avoid_: platform backend, Metal-only backend

**Apple accelerator layer**:
Optional Apple Silicon acceleration (Metal MSL passthrough, TensorOps, `ane`, `mtl4`, CoreML) layered alongside the compute backend behind explicit features and fallback guards.
_Avoid_: default backend, mandatory Metal

**Neural Accelerator**:
GPU 每个着色器核心里的矩阵单元，经 Metal 张量运算使用。
_Avoid_: Apple Neural Engine, ANE, NAX

**Apple Neural Engine**:
独立的神经引擎，实验性 ANE 后端走它。
_Avoid_: Neural Accelerator, NAX

## 模型获取与外部提供方

**Download route**:
The configured checkpoint source preset (ModelScope, Hugging Face, or a custom mirror endpoint) plus optional HTTP(S) proxy used to fetch checkpoint files and resolve per-model source availability.
_Avoid_: mirror URL, source flag

**Capability slot**:
An independently switchable task boundary in the desktop workflow—prompt enhancement, speech recognition, speech synthesis, or translation—whose provider selection never silently alters another slot or crosses the local/external boundary.
_Avoid_: global provider, model mode

**Capability provider**:
Either the default local Burn model or an explicitly selected OpenAI-compatible external endpoint (such as a local Magpie gateway) bound to one capability slot.
_Avoid_: cloud fallback, remote engine
