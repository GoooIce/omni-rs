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

**Speech text**:
The words one speech synthesis is asked to speak, including official inline vocal events.
_Avoid_: voice instruction, enhancer prompt, prompt

**Voice instruction**:
Optional natural-language direction for how a speech synthesis should sound, in the same language as the speech text. Empty means that synthesis has no voice instruction.
_Avoid_: speech text, reference transcript, enhancer prompt, prompt

**Reference transcript**:
The words actually spoken in a reference recording. It stays paired with that recording.
_Avoid_: speech text, voice instruction

**Voice-conditioning**:
The optional voice instruction and optional reference recording, with its reference transcript, fixed for one speech synthesis.
_Avoid_: speech text, enhancer prompt

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
A completed audio or image file made available for download. Its link remains valid while the file is retained, independently of inference-job history and application restarts.

## 执行位置

**Neural Accelerator**:
GPU 每个着色器核心里的矩阵单元，经 Metal 张量运算使用。
_Avoid_: Apple Neural Engine, ANE, NAX

**Apple Neural Engine**:
独立的神经引擎，实验性 ANE 后端走它。
_Avoid_: Neural Accelerator, NAX
