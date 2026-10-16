use crate::{
    Error, Model, Request, Result, Translation, TranslationEvent, check_cancel,
    paged::{Batch, Segment},
    pages::{History, PAGE_SIZE, Pages},
    sampling::Sampler,
};
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::tensor::RadixStats;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;

/// Physical KV capacity includes active requests and cached prefixes. Admission
/// reserves each request's worst-case context plus one copy-on-write scratch
/// page, so active requests can always finish without memory-pressure deadlock.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_step_tokens: usize,
    pub kv_pool_tokens: usize,
}
impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            max_step_tokens: 256,
            kv_pool_tokens: 16384,
        }
    }
}
impl BatchConfig {
    pub fn validate(&self) -> Result<()> {
        if !(1..=64).contains(&self.max_batch_size)
            || !(self.max_batch_size..=4096).contains(&self.max_step_tokens)
            || self.kv_pool_tokens < 2 * PAGE_SIZE
            || self.kv_pool_tokens > 1_048_576
            || !self.kv_pool_tokens.is_multiple_of(PAGE_SIZE)
        {
            return Err(Error::Config("batch size must be 1..64, step tokens batch_size..4096, and KV pool 32..1048576 tokens in multiples of 16".into()));
        }
        Ok(())
    }
}
pub type RequestId = u64;
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct BatchStats {
    pub steps: u64,
    pub batched_steps: u64,
    pub max_batch_size: usize,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub copied_pages: u64,
    pub active_requests: usize,
    pub used_pages: usize,
    pub peak_pages: usize,
    pub total_pages: usize,
}
struct Active {
    id: RequestId,
    request: Request,
    prompt: Vec<u32>,
    history: History,
    cached_tokens: usize,
    reserved: usize,
    sampler: Sampler,
    sampled: Vec<u32>,
    generated: Vec<u32>,
    text: String,
    start: Instant,
    prefill_ms: f64,
    cancel: Arc<AtomicBool>,
}
pub struct Translator<B: Backend> {
    model: Model<B>,
    pages: Pages,
    atlas: Vec<Option<Tensor<B, 5>>>,
    config: BatchConfig,
    active: Vec<Active>,
    reserved: usize,
    next: RequestId,
    stats: BatchStats,
}
impl<B: Backend> Translator<B> {
    pub fn new(model: Model<B>, cache_tokens: usize) -> Self {
        Self::with_config(model, cache_tokens, BatchConfig::default())
            .expect("valid default batch configuration")
    }
    pub fn with_config(model: Model<B>, cache_tokens: usize, config: BatchConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            atlas: (0..model.config().num_hidden_layers)
                .map(|_| None)
                .collect(),
            model,
            pages: Pages::new(
                config.kv_pool_tokens,
                cache_tokens.min(config.kv_pool_tokens),
            ),
            config,
            active: Vec::new(),
            reserved: 0,
            next: 1,
            stats: BatchStats::default(),
        })
    }
    pub fn model(&self) -> &Model<B> {
        &self.model
    }
    pub fn cache_stats(&self) -> RadixStats {
        self.pages.stats()
    }
    pub fn clear_cache(&mut self) {
        self.pages.clear();
    }
    pub fn config(&self) -> BatchConfig {
        self.config
    }
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }
    pub fn batch_stats(&self) -> BatchStats {
        BatchStats {
            active_requests: self.active.len(),
            used_pages: self.pages.used(),
            total_pages: self.pages.capacity(),
            ..self.stats
        }
    }
    pub fn prompt_ids(&self, request: &Request) -> Result<Vec<u32>> {
        request.validate()?;
        let prompt = format!(
            "<|startoftext|>Translate the following text into {}. Note that you should only output the translated result without any additional explanation:\n\n{}<|extra_0|>",
            request.target_language.trim(),
            request.text
        );
        let ids = self
            .model
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        let limit = request
            .options
            .max_context_tokens
            .min(self.model.config().max_position_embeddings);
        if ids.is_empty()
            || ids
                .len()
                .checked_add(request.options.max_new_tokens)
                .is_none_or(|n| n > limit)
        {
            return Err(Error::Config(format!(
                "translation requires {} prompt tokens plus {} output tokens; context limit is {limit}",
                ids.len(),
                request.options.max_new_tokens
            )));
        }
        Ok(ids)
    }
    /// `None` means temporary backpressure; the caller retains the request and
    /// retries after `step`. Invalid requests return errors even when full.
    pub fn try_submit(
        &mut self,
        request: &Request,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<RequestId>> {
        let prompt = self.prompt_ids(request)?;
        check_cancel(&cancel)?;
        let reserved = (prompt.len() + request.options.max_new_tokens).div_ceil(PAGE_SIZE) + 1;
        if reserved > self.pages.capacity() {
            return Err(Error::Config(format!(
                "request needs {reserved} KV pages, pool has {}; increase kv_pool_tokens",
                self.pages.capacity()
            )));
        }
        if self.active.len() == self.config.max_batch_size
            || self.reserved + reserved > self.pages.capacity()
        {
            return Ok(None);
        }
        let history = if request.options.use_cache {
            self.pages.lookup(&prompt[..prompt.len() - 1])
        } else {
            History::default()
        };
        let id = self.next;
        self.next += 1;
        self.reserved += reserved;
        self.active.push(Active {
            id,
            request: request.clone(),
            cached_tokens: history.len,
            history,
            reserved,
            sampler: Sampler::new(request.options.clone()),
            sampled: prompt.clone(),
            prompt,
            generated: Vec::new(),
            text: String::new(),
            start: Instant::now(),
            prefill_ms: 0.0,
            cancel,
        });
        Ok(Some(id))
    }
    fn finish(&mut self, id: RequestId, reason: &str) -> Translation {
        let index = self.active.iter().position(|a| a.id == id).unwrap();
        let mut a = self.active.remove(index);
        self.pages.release(&mut a.history);
        self.reserved -= a.reserved;
        Translation {
            text: a.text,
            target_language: a.request.target_language,
            prompt_tokens: a.prompt.len(),
            generated_tokens: a.generated.len(),
            cached_tokens: a.cached_tokens,
            prefill_ms: a.prefill_ms,
            compute_ms: a.start.elapsed().as_secs_f64() * 1000.0,
            finish_reason: reason.into(),
            cache: self.pages.stats(),
        }
    }
    fn prepare(&mut self) -> Result<(Batch, Vec<RequestId>)> {
        let mut budget = self.config.max_step_tokens;
        let mut scheduled = Vec::new();
        // Every decoder advances once. Prefill shares the remaining budget and
        // cannot monopolize a step while other requests are decoding.
        for a in &self.active {
            if a.history.len >= a.prompt.len() {
                scheduled.push((a.id, 1));
                budget -= 1;
            }
        }
        let prefill: Vec<_> = self
            .active
            .iter()
            .filter(|a| a.history.len < a.prompt.len())
            .collect();
        for (i, a) in prefill.iter().enumerate() {
            let fair = budget / (prefill.len() - i);
            let count = fair
                .min(a.request.options.prefill_chunk)
                .min(a.prompt.len() - a.history.len);
            if count > 0 {
                scheduled.push((a.id, count));
                budget -= count;
            }
        }
        let mut batch = Batch::default();
        for &(id, count) in &scheduled {
            let a = self.active.iter_mut().find(|a| a.id == id).unwrap();
            batch
                .copies
                .extend(self.pages.extend(&mut a.history, count)?);
            let offset = a.history.len;
            let start = batch.ids.len();
            let table = batch.table.len();
            batch
                .table
                .extend(a.history.pages.iter().map(|p| **p as u32));
            if offset < a.prompt.len() {
                batch
                    .ids
                    .extend_from_slice(&a.prompt[offset..offset + count]);
                self.stats.prefill_tokens += count as u64;
            } else {
                batch
                    .ids
                    .push(*a.generated.last().expect("decoder has a sampled token"));
                self.stats.decode_tokens += 1;
            }
            for pos in offset..offset + count {
                batch.positions.push(pos);
                batch.rows.extend([
                    pos as u32,
                    table as u32,
                    (*a.history.pages[pos / PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE) as u32,
                ]);
            }
            batch.segments.push(Segment {
                start,
                count,
                offset,
                table,
            });
        }
        self.stats.steps += 1;
        self.stats.batched_steps += u64::from(scheduled.len() > 1);
        self.stats.max_batch_size = self.stats.max_batch_size.max(scheduled.len());
        self.stats.peak_pages = self.stats.peak_pages.max(self.pages.used());
        self.stats.copied_pages += batch.copies.len() as u64;
        Ok((batch, scheduled.into_iter().map(|(id, _)| id).collect()))
    }
    /// One bounded GPU forward, with per-request streaming and terminal results.
    /// Admission/cancellation can change between calls. A cancelled member never
    /// aborts another member's forward or publishes partially computed KV.
    pub async fn step(
        &mut self,
        mut emit: impl FnMut(RequestId, TranslationEvent),
    ) -> Vec<(RequestId, Result<Translation>)> {
        let mut completed = Vec::new();
        let cancelled: Vec<_> = self
            .active
            .iter()
            .filter(|a| a.cancel.load(Ordering::Acquire))
            .map(|a| a.id)
            .collect();
        for id in cancelled {
            completed.push((id, Ok(self.finish(id, "cancelled"))));
        }
        if self.active.is_empty() {
            return completed;
        }
        let computed = async {
            let (batch, ids) = self.prepare()?;
            let logits =
                self.model
                    .forward_batch(&batch, &mut self.atlas, self.pages.capacity())?;
            let data = logits
                .cast(FloatDType::F32)
                .into_data_async()
                .await
                .map_err(|e| Error::Inference(format!("translation readback: {e:?}")))?;
            let scores = data
                .to_vec::<f32>()
                .map_err(|e| Error::Inference(format!("translation logits: {e:?}")))?;
            Ok::<_, Error>((batch, ids, scores))
        }
        .await;
        let (batch, ids, scores) = match computed {
            Ok(value) => value,
            Err(error) => {
                // Flush deferred operations before physical page IDs can be reused.
                let _ = B::sync(self.model.device());
                for a in self.active.drain(..) {
                    completed.push((a.id, Err(Error::Inference(error.to_string()))));
                }
                self.reserved = 0;
                self.pages.clear();
                self.atlas.iter_mut().for_each(|a| *a = None);
                return completed;
            }
        };
        let vocab = self.model.config().vocab_size;
        let mut endings = Vec::new();
        for (row, id) in ids.into_iter().enumerate() {
            let a = self.active.iter_mut().find(|a| a.id == id).unwrap();
            if a.cancel.load(Ordering::Acquire) {
                endings.push((id, Ok("cancelled")));
                continue;
            }
            a.history.len += batch.segments[row].count;
            if a.history.len < a.prompt.len() {
                continue;
            }
            if a.history.len == a.prompt.len() {
                a.prefill_ms = a.start.elapsed().as_secs_f64() * 1000.0;
                if a.request.options.use_cache {
                    self.pages.publish(&a.prompt, &a.history);
                }
            }
            let sampled = (|| -> Result<&str> {
                let token = a
                    .sampler
                    .sample(scores[row * vocab..(row + 1) * vocab].to_vec(), &a.sampled)?;
                if self.model.eos.contains(&token) {
                    return Ok("stop");
                }
                a.generated.push(token);
                a.sampled.push(token);
                let decoded = self
                    .model
                    .tokenizer
                    .decode(&a.generated, true)
                    .map_err(|e| Error::Tokenizer(e.to_string()))?;
                let stable = decoded.trim_end_matches('\u{fffd}');
                if !stable.starts_with(&a.text) {
                    return Err(Error::Tokenizer(
                        "non-monotonic translation decoding".into(),
                    ));
                }
                let delta = stable[a.text.len()..].to_owned();
                a.text = stable.to_owned();
                emit(
                    id,
                    TranslationEvent {
                        delta,
                        text: a.text.clone(),
                        generated_tokens: a.generated.len(),
                    },
                );
                if a.cancel.load(Ordering::Acquire) {
                    return Ok("cancelled");
                }
                if a.generated.len() == a.request.options.max_new_tokens {
                    return Ok("length");
                }
                Ok("")
            })();
            if !matches!(sampled, Ok("")) {
                endings.push((id, sampled));
            }
        }
        // Drop copy pins only after all layer writes/reads have completed.
        drop(batch);
        for (id, end) in endings {
            match end {
                Ok(reason) => completed.push((id, Ok(self.finish(id, reason)))),
                Err(error) => {
                    self.finish(id, "error");
                    completed.push((id, Err(error)));
                }
            }
        }
        completed
    }
    /// Single-request convenience used by desktop and CLI; same paged execution
    /// and scheduler as HTTP. An existing batch must be driven with `step`.
    pub async fn translate(
        &mut self,
        request: &Request,
        cancel: &AtomicBool,
        mut emit: impl FnMut(TranslationEvent),
    ) -> Result<Translation> {
        check_cancel(cancel)?;
        if !self.is_idle() {
            return Err(Error::Config("drive active translations with step".into()));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let id = self
            .try_submit(request, stop.clone())?
            .expect("empty valid batch admits one request");
        loop {
            if cancel.load(Ordering::Acquire) {
                stop.store(true, Ordering::Release);
            }
            let results = self
                .step(|_, event| {
                    emit(event);
                    if cancel.load(Ordering::Acquire) {
                        stop.store(true, Ordering::Release);
                    }
                })
                .await;
            if let Some((_, result)) = results.into_iter().find(|(done, _)| *done == id) {
                return result;
            }
        }
    }
}
