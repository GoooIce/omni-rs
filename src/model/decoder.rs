use super::ops::{Linear, Norm, residual, rope, rope_tables};
use crate::{
    Error, Precision, Result,
    config::TextConfig,
    tensor::{AttentionOptions, DecodeCache, Embedding},
    weights::{
        WeightSource,
        layout::{DecoderLayerLayout, DecoderLayout},
    },
};
use burn::tensor::{Int, Tensor, TensorData, activation::silu, backend::Backend};

fn record_activation<B: Backend, const D: usize>(
    trace: &mut Option<&mut Vec<(String, Tensor<B, 2>)>>,
    name: &str,
    value: &Tensor<B, D>,
) {
    if let Some(trace) = trace {
        let dims = value.dims();
        let width = dims[D - 1];
        let rows = dims.iter().product::<usize>() / width;
        trace.push((name.into(), value.clone().reshape([rows, width])));
    }
}

struct Layer<B: Backend> {
    norm1: Norm<B>,
    norm2: Norm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}
impl<B: Backend> Layer<B> {
    fn load(
        w: &impl WeightSource<B>,
        names: &DecoderLayerLayout,
        c: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let width = c.hidden_size;
        Ok(Self {
            norm1: Norm::load(w, &names.norm1, width, false, c.rms_norm_eps, p, d)?,
            norm2: Norm::load(w, &names.norm2, width, false, c.rms_norm_eps, p, d)?,
            q: Linear::load(
                w,
                &names.q,
                width,
                c.num_attention_heads * c.head_dim,
                c.attention_bias,
                p,
                d,
            )?,
            k: Linear::load(
                w,
                &names.k,
                width,
                c.num_key_value_heads * c.head_dim,
                c.attention_bias,
                p,
                d,
            )?,
            v: Linear::load(
                w,
                &names.v,
                width,
                c.num_key_value_heads * c.head_dim,
                c.attention_bias,
                p,
                d,
            )?,
            out: Linear::load(
                w,
                &names.out,
                c.num_attention_heads * c.head_dim,
                width,
                c.attention_bias,
                p,
                d,
            )?,
            q_norm: Norm::load(w, &names.q_norm, c.head_dim, false, c.rms_norm_eps, p, d)?,
            k_norm: Norm::load(w, &names.k_norm, c.head_dim, false, c.rms_norm_eps, p, d)?,
            gate: Linear::load(w, &names.gate, width, c.intermediate_size, false, p, d)?,
            up: Linear::load(w, &names.up, width, c.intermediate_size, false, p, d)?,
            down: Linear::load(w, &names.down, c.intermediate_size, width, false, p, d)?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 2>,
        caches: &mut [&mut DecodeCache<B>],
        segments: &[(usize, usize, usize)],
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        mut trace: Option<&mut Vec<(String, Tensor<B, 2>)>>,
    ) -> Tensor<B, 2> {
        let n = x.dims()[0];
        record_activation(&mut trace, "input", &x);
        let norm = self.norm1.forward(x.clone());
        record_activation(&mut trace, "norm1", &norm);
        let q = self
            .q
            .forward(norm.clone())
            .reshape([1, n, self.heads, self.head_dim]);
        record_activation(&mut trace, "q-proj", &q);
        let q = self.q_norm.forward(q).swap_dims(1, 2);
        record_activation(&mut trace, "q-norm", &q);
        let k = self
            .k
            .forward(norm.clone())
            .reshape([1, n, self.kv_heads, self.head_dim]);
        record_activation(&mut trace, "k-proj", &k);
        let k = self.k_norm.forward(k).swap_dims(1, 2);
        record_activation(&mut trace, "k-norm", &k);
        let v = self
            .v
            .forward(norm)
            .reshape([1, n, self.kv_heads, self.head_dim])
            .swap_dims(1, 2);
        record_activation(&mut trace, "v-proj", &v);
        let q = rope(q, cos, sin);
        let k = rope(k, cos, sin);
        record_activation(&mut trace, "q-rope", &q);
        record_activation(&mut trace, "k-rope", &k);
        let y = Tensor::cat(
            caches
                .iter_mut()
                .zip(segments)
                .map(|(cache, &(start, count, offset))| {
                    cache.forward(
                        q.clone().slice([
                            0..1,
                            0..self.heads,
                            start..start + count,
                            0..self.head_dim,
                        ]),
                        k.clone().slice([
                            0..1,
                            0..self.kv_heads,
                            start..start + count,
                            0..self.head_dim,
                        ]),
                        v.clone().slice([
                            0..1,
                            0..self.kv_heads,
                            start..start + count,
                            0..self.head_dim,
                        ]),
                        AttentionOptions {
                            causal: true,
                            offset,
                            ..Default::default()
                        },
                    )
                })
                .collect(),
            2,
        )
        .swap_dims(1, 2)
        .reshape([n, self.heads * self.head_dim]);
        record_activation(&mut trace, "attention", &y);
        let y = self.out.forward(y);
        record_activation(&mut trace, "out-proj", &y);
        let x = residual(x, y);
        record_activation(&mut trace, "residual1", &x);
        let norm = self.norm2.forward(x.clone());
        record_activation(&mut trace, "norm2", &norm);
        let gate = self.gate.forward(norm.clone());
        record_activation(&mut trace, "gate-proj", &gate);
        let gate = silu(gate);
        record_activation(&mut trace, "silu", &gate);
        let up = self.up.forward(norm);
        record_activation(&mut trace, "up-proj", &up);
        let y = self.down.forward(gate * up);
        record_activation(&mut trace, "down-proj", &y);
        let out = residual(x, y);
        record_activation(&mut trace, "output", &out);
        out
    }
}

pub struct DecoderCache<B: Backend> {
    layers: Vec<DecodeCache<B>>,
    pub len: usize,
}
impl<B: Backend> DecoderCache<B> {
    pub fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| DecodeCache::new(false)).collect(),
            len: 0,
        }
    }
    pub fn with_paging(layers: usize, paged: bool) -> Self {
        Self {
            layers: (0..layers).map(|_| DecodeCache::new(paged)).collect(),
            len: 0,
        }
    }
    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.len);
        self.len = len;
    }
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.len = 0;
    }
}

pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    head: Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: Norm<B>,
    config: TextConfig,
    precision: Precision,
    device: B::Device,
}
impl<B: Backend> Decoder<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        c: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let names = DecoderLayout::new(c);
        // HF may omit either side of a tied embedding. Never substitute an untied head.
        let embedding_name = if w.contains(&names.embedding) {
            &names.embedding
        } else if c.tie_word_embeddings {
            &names.head
        } else {
            return Err(Error::Checkpoint("missing input embeddings".into()));
        };
        let embedding = Embedding::load(w, embedding_name, c.vocab_size, c.hidden_size, p, d)?;
        let head = if w.contains(&names.head) && embedding_name != &names.head {
            Embedding::load(w, &names.head, c.vocab_size, c.hidden_size, p, d)?
        } else if c.tie_word_embeddings {
            embedding.clone()
        } else {
            return Err(Error::Checkpoint(
                "missing untied language-model head".into(),
            ));
        };
        Ok(Self {
            embedding,
            head,
            layers: names
                .layers
                .iter()
                .map(|layer| Layer::load(w, layer, c, p, d))
                .collect::<Result<_>>()?,
            norm: Norm::load(w, &names.norm, c.hidden_size, false, c.rms_norm_eps, p, d)?,
            config: c.clone(),
            precision: p,
            device: d.clone(),
        })
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        B::sync(&self.device).map_err(|e| Error::Inference(e.to_string()))
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Tensor<B, 2>> {
        self.embedding.embed(ids)
    }

    pub fn forward(&self, x: Tensor<B, 2>, cache: &mut DecoderCache<B>) -> Result<Tensor<B, 2>> {
        self.forward_impl(vec![x], &mut [cache], None, None)
    }

    /// Diagnostic execution retaining every layer's hidden state. Extra live
    /// outputs can change backend fusion; do not use this path for timing.
    pub fn forward_with_layer_outputs(
        &self,
        x: Tensor<B, 2>,
        cache: &mut DecoderCache<B>,
        outputs: &mut Vec<Tensor<B, 2>>,
    ) -> Result<Tensor<B, 2>> {
        outputs.clear();
        self.forward_impl(vec![x], &mut [cache], Some(outputs), None)
    }

    /// Diagnostic snapshots inside the first decoder layer, including norms,
    /// Q/K projections, RoPE, attention and FFN. Additional live outputs can
    /// affect backend fusion. Compare final logits with an untraced control.
    pub fn forward_with_activation_trace(
        &self,
        x: Tensor<B, 2>,
        cache: &mut DecoderCache<B>,
        trace: &mut Vec<(String, Tensor<B, 2>)>,
    ) -> Result<Tensor<B, 2>> {
        trace.clear();
        self.forward_impl(vec![x], &mut [cache], None, Some(trace))
    }

    /// Ragged inputs share all projections/MLPs; each history keeps its own
    /// rotary positions, attention mask and rewindable cache.
    pub fn forward_batch(
        &self,
        inputs: Vec<Tensor<B, 2>>,
        caches: &mut [&mut DecoderCache<B>],
    ) -> Result<Tensor<B, 2>> {
        self.forward_impl(inputs, caches, None, None)
    }
    fn forward_impl(
        &self,
        inputs: Vec<Tensor<B, 2>>,
        caches: &mut [&mut DecoderCache<B>],
        mut outputs: Option<&mut Vec<Tensor<B, 2>>>,
        mut trace: Option<&mut Vec<(String, Tensor<B, 2>)>>,
    ) -> Result<Tensor<B, 2>> {
        // Keep prefill and recurrent KV updates on the same Metal path in the
        // native latency policy. Per-small-graph switching changes the numeric
        // boundaries inside cached decoding and adds repeated device fences.
        // The acoustic encoder can still offload its larger graphs to ANE.
        #[cfg(all(
            feature = "ane",
            feature = "metal",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let _routing = {
            use std::any::Any;
            let device = (&self.device as &dyn Any)
                .downcast_ref::<burn::backend::AneDevice<burn::backend::wgpu::Metal>>();
            match device {
                Some(device) if device.prefers_fallback_phases() => Some(
                    device
                        .fallback_scope()
                        .map_err(|e| Error::Inference(e.to_string()))?,
                ),
                _ => None,
            }
        };
        if inputs.is_empty() || inputs.len() != caches.len() {
            return Err(Error::Inference(
                "decoder batch must contain one cache per input".into(),
            ));
        }
        let mut cursor = 0;
        let mut segments = Vec::new();
        let mut cosines = Vec::new();
        let mut sines = Vec::new();
        let mut last = Vec::new();
        for (input, cache) in inputs.iter().zip(caches.iter()) {
            let n = input.dims()[0];
            if n == 0
                || input.dims()[1] != self.config.hidden_size
                || n + cache.len > self.config.max_position_embeddings
            {
                return Err(Error::Inference(
                    "decoder context limit or input width exceeded".into(),
                ));
            }
            segments.push((cursor, n, cache.len));
            cursor += n;
            last.push((cursor - 1) as i32);
            let (cos, sin) = rope_tables(
                cache.len,
                n,
                self.config.head_dim,
                self.config.rope_theta,
                self.precision,
                &self.device,
            );
            cosines.push(cos);
            sines.push(sin);
        }
        let cos = Tensor::cat(cosines, 2);
        let sin = Tensor::cat(sines, 2);
        let mut x = Tensor::cat(inputs, 0);
        for (i, layer) in self.layers.iter().enumerate() {
            let mut histories: Vec<_> = caches
                .iter_mut()
                .map(|cache| &mut cache.layers[i])
                .collect();
            x = layer.forward(x, &mut histories, &segments, &cos, &sin, trace.take());
            if let Some(outputs) = &mut outputs {
                outputs.push(x.clone());
            }
        }
        for (cache, &(_, n, _)) in caches.iter_mut().zip(&segments) {
            cache.len += n;
        }
        let last =
            Tensor::<B, 1, Int>::from_data(TensorData::new(last, [caches.len()]), &self.device);
        Ok(self.head.logits(self.norm.forward(x.select(0, last))))
    }
}
