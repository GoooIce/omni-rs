//! One speech request owns sampling, complete frames, delivery and termination.
//! Serial and packed tensor execution are private adapters of this lifecycle.
use crate::*;
use std::collections::BTreeMap;
mod packed;
mod serial;

pub(crate) enum Started<B: Backend> {
    Running(Box<Synthesis<B>>),
    Stopped(SynthesisReport),
}
pub(crate) struct Member<'a, B: Backend> {
    pub request: &'a mut Synthesis<B>,
    pub cancel: &'a AtomicBool,
}
pub(crate) struct BatchStep {
    pub computed: usize,
    pub completed: Vec<(usize, Result<SynthesisReport>)>,
}
#[derive(Clone, Copy)]
enum End {
    Eos,
    Limit,
    Interrupted,
}

pub(crate) struct Synthesis<B: Backend> {
    options: SynthesisOptions,
    sampler: sampling::Sampler,
    cond: Tensor<B, 3>,
    uncond: Option<Tensor<B, 3>>,
    cond_cache: TransformerCache<B>,
    uncond_cache: TransformerCache<B>,
    codec: codec::CodecState<B>,
    codes: Vec<Tensor<B, 2, Int>>,
    pending: Vec<Tensor<B, 2, Int>>,
    delivery: Delivery,
    limit: usize,
    profiler: Option<profile::Profiler>,
    completed: bool,
}

/// Captures handoff time before calling the recipient; commits only on success.
#[derive(Default)]
struct Delivery {
    started: Option<Instant>,
    first_audio: Option<f64>,
    samples: usize,
}
impl Delivery {
    fn new(started: Instant) -> Self {
        Self {
            started: Some(started),
            ..Self::default()
        }
    }
    fn elapsed(&self) -> f64 {
        self.started
            .expect("synthesis start")
            .elapsed()
            .as_secs_f64()
    }
    fn emit(&mut self, audio: &[f32], emit: &mut impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
        let handed_off = self.elapsed();
        emit(audio)?;
        self.first_audio.get_or_insert(handed_off);
        self.samples += audio.len();
        Ok(())
    }
    fn report(
        &self,
        codes: Vec<[u32; CODEBOOKS]>,
        end: End,
        profile: Option<profile::Profile>,
    ) -> SynthesisReport {
        SynthesisReport {
            frames: codes.len(),
            audio_seconds: self.samples as f64 / SAMPLE_RATE as f64,
            elapsed_seconds: self.elapsed(),
            first_audio_seconds: self.first_audio,
            stopped: matches!(end, End::Interrupted),
            reached_limit: matches!(end, End::Limit),
            codes,
            profile,
        }
    }
}

impl<B: Backend> Synthesis<B> {
    pub(crate) fn prepare(
        model: &Model<B>,
        options: &SynthesisOptions,
        cancel: &AtomicBool,
        profile: bool,
    ) -> Result<Started<B>> {
        options.validate()?;
        let delivery = Delivery::new(Instant::now());
        let mut profiler = if profile {
            Some(profile::Profiler::new::<B>(&model.device)?)
        } else {
            None
        };
        let stopped = || Started::Stopped(delivery.report(Vec::new(), End::Interrupted, None));
        if cancel.load(Ordering::Acquire) {
            return Ok(stopped());
        }
        let prompt = model.prepare_prompt(
            &options.text,
            &options.instruction,
            options.reference.as_ref(),
        )?;
        let sampler = sampling::Sampler::new(options.sampling.clone())?;
        let mut cond_cache = model.backbone.cache_with_paging(options.paged_attention);
        let mut uncond_cache = model.backbone.cache_with_paging(options.paged_attention);
        if let Some(profiler) = &mut profiler {
            profiler.mark::<B>("prepare", &model.device)?;
        }
        let Some(cond) =
            model.prefill(&prompt, options.reference.as_ref(), &mut cond_cache, cancel)?
        else {
            return Ok(stopped());
        };
        let uncond = if !options.instruction.trim().is_empty() && options.guidance_scale > 1.0 {
            let Some(hidden) = model.prefill(
                &model.prepare_prompt(&options.text, "", options.reference.as_ref())?,
                options.reference.as_ref(),
                &mut uncond_cache,
                cancel,
            )?
            else {
                return Ok(stopped());
            };
            Some(hidden)
        } else {
            None
        };
        Ok(Started::Running(Box::new(Self {
            options: options.clone(),
            sampler,
            cond,
            uncond,
            cond_cache,
            uncond_cache,
            codec: model.codec.state(),
            codes: Vec::new(),
            pending: Vec::new(),
            delivery,
            limit: options
                .max_frames
                .min(model.config.backbone_config.max_position_embeddings - prompt.ids.len()),
            profiler,
            completed: false,
        })))
    }
    fn mark(&mut self, model: &Model<B>, name: &'static str) -> Result<()> {
        if let Some(profiler) = &mut self.profiler {
            profiler.mark::<B>(name, &model.device)?;
        }
        Ok(())
    }
    fn hidden(&self) -> Tensor<B, 3> {
        match &self.uncond {
            Some(u) => Tensor::cat(vec![self.cond.clone(), u.clone()], 0),
            None => self.cond.clone(),
        }
    }
    fn guided(&self, scores: Tensor<B, 3>) -> Tensor<B, 3> {
        Model::<B>::guided_logits(scores, self.options.guidance_scale)
    }
    fn device_sampling(&self) -> bool {
        !self.options.cpu_depth_sampling && self.options.sampling.top_p == 1.0
    }
    fn begin_frame(
        &mut self,
        model: &Model<B>,
        scores: &[f32],
        hidden: Tensor<B, 3>,
    ) -> Result<Option<Frame<B>>> {
        let first = self.sampler.sample(scores, true)?;
        self.mark(model, "semantic_sample")?;
        if first == model.config.vocab_size as u32 {
            return Ok(None);
        }
        let token = model.token(first);
        let embedding = model.codebook_embedding(0, token.clone());
        let lanes = hidden.dims()[0];
        let noise = if self.device_sampling() {
            self.sampler
                .prepare_noise::<B>(CODEBOOKS - 1, &model.device)
        } else {
            None
        };
        Ok(Some(Frame {
            lanes,
            input: Tensor::cat(vec![hidden, embedding.clone().repeat_dim(0, lanes)], 1),
            tokens: vec![token],
            embeddings: vec![embedding],
            noise,
        }))
    }
    fn sample_depth(
        &mut self,
        model: &Model<B>,
        frame: &mut Frame<B>,
        codebook: usize,
        scores: Tensor<B, 3>,
    ) -> Result<()> {
        let token = if self.device_sampling() {
            self.sampler.sample_device_with_noise(
                self.guided(scores),
                frame
                    .noise
                    .as_ref()
                    .map(|n| n.clone().slice([codebook - 1..codebook, 0..1, 0..2048])),
            )
        } else {
            model.token(self.sampler.sample(
                &Model::<B>::logits(scores, self.options.guidance_scale)?,
                false,
            )?)
        };
        let embedding = model.codebook_embedding(codebook, token.clone());
        frame.input = embedding.clone().repeat_dim(0, frame.lanes);
        frame.embeddings.push(embedding);
        frame.tokens.push(token);
        Ok(())
    }
    fn emit_pending(
        &mut self,
        model: &Model<B>,
        cancel: &AtomicBool,
        emit: &mut impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<bool> {
        if self.pending.is_empty() {
            return Ok(!cancel.load(Ordering::Acquire));
        }
        let audio = model.codec.decode_device(
            Tensor::cat(std::mem::take(&mut self.pending), 0),
            &mut self.codec,
        )?;
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        self.delivery.emit(&audio, emit)?;
        Ok(true)
    }
    fn accept_frame(
        &mut self,
        model: &Model<B>,
        frame: Frame<B>,
        cancel: &AtomicBool,
        serial: bool,
        emit: &mut impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<Option<End>> {
        self.mark(model, "depth")?;
        // Incomplete frames are never made observable as codes or PCM.
        assert_eq!(frame.tokens.len(), CODEBOOKS);
        let codes = Tensor::cat(frame.tokens, 0).reshape([1, CODEBOOKS]);
        self.codes.push(codes.clone());
        self.pending.push(codes);
        self.mark(model, "codes_device")?;
        let limit = self.codes.len() == self.limit;
        if (self.pending.len() >= self.options.chunk_frames || (!serial && limit))
            && !self.emit_pending(model, cancel, emit)?
        {
            return Ok(Some(End::Interrupted));
        }
        self.mark(model, "codec_and_callback")?;
        // Serial profiling includes its historical final backbone continuation.
        // Packed execution skips that unused work at the frame ceiling.
        if !serial && limit {
            return Ok(Some(End::Limit));
        }
        let embedding = Tensor::cat(frame.embeddings, 1)
            .cast(FloatDType::F32)
            .sum_dim(1)
            .cast(model.precision.dtype());
        self.mark(model, "audio_embedding")?;
        self.cond = Model::<B>::last(
            model
                .backbone
                .forward(embedding.clone(), &mut self.cond_cache),
        );
        if self.uncond.is_some() {
            self.uncond = Some(Model::<B>::last(
                model.backbone.forward(embedding, &mut self.uncond_cache),
            ));
        }
        self.mark(model, "backbone")?;
        Ok(limit.then_some(End::Limit))
    }
    fn finish(
        &mut self,
        model: &Model<B>,
        mut end: End,
        cancel: &AtomicBool,
        emit: &mut impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<SynthesisReport> {
        if self.completed {
            return Err(Error::Finished);
        }
        // Mark before delivery so a failing callback cannot be retried as success.
        self.completed = true;
        if cancel.load(Ordering::Acquire) {
            end = End::Interrupted;
        }
        if !matches!(end, End::Interrupted) && !self.emit_pending(model, cancel, emit)? {
            end = End::Interrupted;
        }
        // Cancellation raised by the final recipient still belongs to this
        // request, while successfully handed-off audio remains in its report.
        if cancel.load(Ordering::Acquire) {
            end = End::Interrupted;
        }
        self.pending.clear();
        let codes = if self.codes.is_empty() {
            Vec::new()
        } else {
            pollster::block_on(Tensor::cat(std::mem::take(&mut self.codes), 0).into_data_async())
                .map_err(|e| Error::Inference(e.to_string()))?
                .convert::<i32>()
                .to_vec::<i32>()
                .map_err(|e| Error::Inference(e.to_string()))?
                .as_chunks::<CODEBOOKS>()
                .0
                .iter()
                .map(|row| std::array::from_fn(|i| row[i] as u32))
                .collect()
        };
        self.mark(model, "tail_and_codes_readback")?;
        Ok(self.delivery.report(
            codes,
            end,
            self.profiler.take().and_then(profile::Profiler::finish),
        ))
    }
    pub(crate) fn run_serial(
        mut self,
        model: &Model<B>,
        cancel: &AtomicBool,
        emit: &mut impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<SynthesisReport> {
        let mut cache = model.depth.cache();
        self.mark(model, "prefill")?;
        let end = loop {
            if cancel.load(Ordering::Acquire) {
                break End::Interrupted;
            }
            if let Some(end) = serial::advance(model, &mut self, &mut cache, cancel, emit)? {
                break end;
            }
        };
        self.finish(model, end, cancel, emit)
    }
    pub(crate) fn advance_batch(
        model: &Model<B>,
        members: &mut [Member<'_, B>],
        emit: &mut impl FnMut(usize, &[f32]) -> Result<()>,
    ) -> BatchStep {
        let mut completed = BTreeMap::new();
        for (i, member) in members.iter_mut().enumerate() {
            if member.cancel.load(Ordering::Acquire) {
                completed.insert(
                    i,
                    member
                        .request
                        .finish(model, End::Interrupted, member.cancel, &mut |pcm| {
                            emit(i, pcm)
                        }),
                );
            }
        }
        let ready: Vec<_> = (0..members.len())
            .filter(|i| !completed.contains_key(i))
            .collect();
        let computed = ready.len();
        if !ready.is_empty() {
            #[allow(clippy::let_unit_value)]
            let _routing = crate::routing::metal_phase::<B>(&model.device);
            if let Err(error) = packed::advance(model, members, &ready, emit, &mut completed) {
                let _ = B::sync(&model.device);
                let message = error.to_string();
                for i in ready {
                    completed
                        .entry(i)
                        .or_insert_with(|| Err(Error::Inference(message.clone())));
                }
            }
        }
        BatchStep {
            computed,
            completed: completed.into_iter().collect(),
        }
    }
}
struct Frame<B: Backend> {
    lanes: usize,
    input: Tensor<B, 3>,
    embeddings: Vec<Tensor<B, 3>>,
    tokens: Vec<Tensor<B, 1, Int>>,
    noise: Option<Tensor<B, 3>>,
}
fn depth_logits<B: Backend>(
    model: &Model<B>,
    input: Tensor<B, 3>,
    cache: &mut TransformerCache<B>,
    book: usize,
) -> Tensor<B, 3> {
    let output = Model::<B>::last(
        model
            .depth
            .forward(model.depth_projection.forward(input), cache),
    );
    let width = model.config.depth_decoder_config.hidden_size;
    let vocab = model.config.vocab_size;
    let head = model
        .depth_head
        .clone()
        .slice([book - 1..book, 0..width, 0..vocab])
        .reshape([1, width, vocab]);
    output.matmul(head)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handoff_clock_excludes_recipient_work_and_counts_only_success() {
        let mut delivery = Delivery::new(Instant::now());
        let mut entered = 0.0;
        let clock = delivery.started.unwrap();
        delivery
            .emit(&[0.1; 8], &mut |_| {
                entered = clock.elapsed().as_secs_f64();
                std::thread::sleep(std::time::Duration::from_millis(20));
                Ok(())
            })
            .unwrap();
        let first = delivery.first_audio.unwrap();
        assert!(first <= entered);
        assert!(delivery.elapsed() - first >= 0.020);
        assert!(
            delivery
                .emit(&[0.2; 16], &mut |_| Err(Error::Audio("sink".into())))
                .is_err()
        );
        delivery.emit(&[0.3; 4], &mut |_| Ok(())).unwrap();
        let report = delivery.report(vec![[0; CODEBOOKS]], End::Interrupted, None);
        assert_eq!(report.first_audio_seconds, Some(first));
        assert_eq!(report.audio_seconds, 12.0 / SAMPLE_RATE as f64);
        assert!(report.stopped);
        assert!(!report.reached_limit);
    }
}
