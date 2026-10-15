//! SheetSage2's adapted MERT encoder and grammar-constrained BART decoder.
use crate::{
    Precision, Progress, Result, check_cancel, invalid, mert,
    ops::{self, Load},
    score::{self, Grammar, Score},
};
use burn::tensor::{FloatDType, Tensor, activation::softmax, backend::Backend};
use omni_r2t2::{
    tensor::{self, AttentionOptions, DecodeCache, Embedding, Linear, Norm},
    weights::Weights,
};
use serde::Deserialize;
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    decoder_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    max_output_seq_len: usize,
    input_audio_length: f64,
    sampling_rate: usize,
    lora_rank: usize,
    lora_alpha: f64,
    tokenizer_fingerprint: String,
    weights_format: String,
}
struct Attention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    heads: usize,
    width: usize,
}
impl<B: Backend> Attention<B> {
    fn load(l: &Load<B>, base: &str, c: &Config) -> Result<Self> {
        let w = c.hidden_size;
        Ok(Self {
            q: l.linear(&format!("{base}.q_proj"), w, w, true)?,
            k: l.linear(&format!("{base}.k_proj"), w, w, true)?,
            v: l.linear(&format!("{base}.v_proj"), w, w, true)?,
            out: l.linear(&format!("{base}.out_proj"), w, w, true)?,
            heads: c.num_attention_heads,
            width: w,
        })
    }
    fn heads(&self, x: Tensor<B, 2>) -> Tensor<B, 4> {
        let n = x.dims()[0];
        x.reshape([1, n, self.heads, self.width / self.heads])
            .swap_dims(1, 2)
    }
    fn output(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let n = x.dims()[2];
        self.out.forward(x.swap_dims(1, 2).reshape([n, self.width]))
    }
}
struct Layer<B: Backend> {
    self_attn: Attention<B>,
    cross: Attention<B>,
    self_norm: Norm<B>,
    cross_norm: Norm<B>,
    final_norm: Norm<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: Backend> Layer<B> {
    fn load(l: &Load<B>, i: usize, c: &Config) -> Result<Self> {
        let base = format!("decoder.layers.{i}");
        let w = c.hidden_size;
        Ok(Self {
            self_attn: Attention::load(l, &format!("{base}.self_attn"), c)?,
            cross: Attention::load(l, &format!("{base}.encoder_attn"), c)?,
            self_norm: l.norm(&format!("{base}.self_attn_layer_norm"), w, 1e-5)?,
            cross_norm: l.norm(&format!("{base}.encoder_attn_layer_norm"), w, 1e-5)?,
            final_norm: l.norm(&format!("{base}.final_layer_norm"), w, 1e-5)?,
            up: l.linear(&format!("{base}.fc1"), w, c.intermediate_size, true)?,
            down: l.linear(&format!("{base}.fc2"), c.intermediate_size, w, true)?,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 2>,
        cache: &mut DecodeCache<B>,
        memory: &(Tensor<B, 4>, Tensor<B, 4>),
    ) -> Tensor<B, 2> {
        let dt = tensor::dtype(&x);
        let a = &self.self_attn;
        let offset = cache.len();
        let q = a.heads(a.q.forward(x.clone()));
        let k = a.heads(a.k.forward(x.clone()));
        let v = a.heads(a.v.forward(x.clone()));
        let y = cache.forward(
            q,
            k,
            v,
            AttentionOptions {
                causal: true,
                offset,
                ..Default::default()
            },
        );
        let x = self
            .self_norm
            .forward(tensor::residual(x, a.output(y), dt), dt);
        let a = &self.cross;
        let q = a.heads(a.q.forward(x.clone()));
        let y = ops::attend(q, memory.0.clone(), memory.1.clone(), false, 0);
        let x = self
            .cross_norm
            .forward(tensor::residual(x, a.output(y), dt), dt);
        let y = self.down.forward(ops::gelu(self.up.forward(x.clone())));
        self.final_norm.forward(tensor::residual(x, y, dt), dt)
    }
}
pub struct Model<B: Backend> {
    encoder: mert::Model<B>,
    embedding: Embedding<B>,
    position: Embedding<B>,
    embedding_norm: Norm<B>,
    layers: Vec<Layer<B>>,
    projection: Linear<B>,
    mixture: Vec<f32>,
    config: Config,
}
impl<B: Backend> Model<B> {
    pub fn load(
        directory: &Path,
        mert_directory: &Path,
        p: Precision,
        d: &B::Device,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Self> {
        check_cancel(stop)?;
        let c: Config = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if c.vocab_size != 31678
            || c.sampling_rate != 24000
            || c.input_audio_length != 300.0
            || c.tokenizer_fingerprint != "5ba3325af0344c7f"
            || c.weights_format != "adapter"
            || c.lora_rank == 0
            || c.num_attention_heads == 0
            || !c.hidden_size.is_multiple_of(c.num_attention_heads)
            || c.max_output_seq_len != 5120
        {
            return Err(invalid("unsupported SheetSage2 configuration or tokenizer"));
        }
        let w = Weights::open(directory)?;
        let l = Load::<B> { w: &w, p, d };
        let encoder = mert::Model::load_with_adapter(
            mert_directory,
            p,
            d,
            stop,
            &progress,
            Some((&w, c.lora_rank, c.lora_alpha / c.lora_rank as f64)),
        )?;
        let mixture = ops::values(softmax(l.fp32("layer_weight", [25])?, 0))?;
        let projection = l.linear("encoder_projection", 1024, c.hidden_size, true)?;
        let embedding = Embedding::load(
            &w,
            "token_embedding.weight",
            c.vocab_size,
            c.hidden_size,
            p,
            d,
        )?;
        let position = Embedding::load(
            &w,
            "decoder.embed_positions.weight",
            c.max_output_seq_len + 2,
            c.hidden_size,
            p,
            d,
        )?;
        let embedding_norm = l.norm("decoder.layernorm_embedding", c.hidden_size, 1e-5)?;
        let mut layers = Vec::new();
        for i in 0..c.decoder_layers {
            check_cancel(stop)?;
            layers.push(Layer::load(&l, i, &c)?);
            progress(Progress {
                stage: "load_sheetsage",
                current: i + 1,
                total: c.decoder_layers,
            });
        }
        Ok(Self {
            encoder,
            embedding,
            position,
            embedding_norm,
            layers,
            projection,
            mixture,
            config: c,
        })
    }
    /// One complete official 300-second context; silence remains visible to attention.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        melody_only: bool,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Score> {
        if pcm.len() < 1025 || pcm.len() > 7200000 || pcm.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "score transcription accepts finite mono 24 kHz audio up to 300 seconds",
            ));
        }
        let seconds = pcm.len() as f64 / 24000.0;
        let mut padded = pcm.to_vec();
        padded.resize(7200000, 0.0);
        let hidden = self
            .encoder
            .encode(&padded, stop, &progress, Some(&self.mixture))?;
        let memory = self
            .projection
            .forward(hidden)
            .reshape([7500, self.config.hidden_size]);
        let cross: Vec<_> = self
            .layers
            .iter()
            .map(|l| {
                (
                    l.cross.heads(l.cross.k.forward(memory.clone())),
                    l.cross.heads(l.cross.v.forward(memory.clone())),
                )
            })
            .collect();
        let mut cache: Vec<_> = self
            .layers
            .iter()
            .map(|_| DecodeCache::new(false))
            .collect();
        let mut tokens = score::PREFIX.to_vec();
        let mut input = tokens.clone();
        let mut grammar = Grammar::default();
        let mut truncated = true;
        while tokens.len() < self.config.max_output_seq_len {
            check_cancel(stop)?;
            let pos = cache[0].len();
            let positions: Vec<_> = (pos + 2..pos + 2 + input.len()).map(|i| i as u32).collect();
            let x = self.embedding.embed(&input)? + self.position.embed(&positions)?;
            let dt = tensor::dtype(&x);
            let mut x = self.embedding_norm.forward(x, dt);
            for ((l, c), m) in self.layers.iter().zip(&mut cache).zip(&cross) {
                check_cancel(stop)?;
                x = l.forward(x, c, m);
            }
            let n = x.dims()[0];
            let logits = ops::values(
                self.embedding
                    .logits(x.slice([n - 1..n, 0..self.config.hidden_size]))
                    .cast(FloatDType::F32),
            )?;
            let id = grammar.select(&logits)?;
            tokens.push(id);
            progress(Progress {
                stage: "score_tokens",
                current: tokens.len(),
                total: self.config.max_output_seq_len,
            });
            if grammar.update(id) {
                truncated = false;
                break;
            }
            if (517..30517).contains(&id) && f64::from(id - 517) / 100.0 >= seconds {
                tokens.push(2);
                truncated = false;
                break;
            }
            input = vec![id];
        }
        if tokens.last() != Some(&2) {
            tokens.push(2);
        }
        score::decode(tokens, seconds, melody_only, truncated)
    }
}
