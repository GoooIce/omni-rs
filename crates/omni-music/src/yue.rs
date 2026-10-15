//! YuE2's separate autoregressive and acoustic branches share attention context.
use crate::{
    Precision, Progress, Result, check_cancel, invalid,
    ops::{self, Load},
    song::*,
    tokenizer::TextTokenizer,
};
use burn::tensor::{FloatDType, Tensor, activation::silu, backend::Backend};
use omni_r2t2::{
    tensor::{self, AttentionOptions, DecodeCache, Embedding, Linear},
    weights::Weights,
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    latent_dim: usize,
    max_position_embeddings: usize,
    timestep_shift: f64,
}
struct Rms<B: Backend> {
    weight: Tensor<B, 1>,
    eps: f64,
}
impl<B: Backend> Rms<B> {
    fn load(l: &Load<B>, name: &str, width: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: l.tensor(&format!("{name}.weight"), [width])?,
            eps,
        })
    }
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dt = tensor::dtype(&x);
        let scale = (x.clone().cast(FloatDType::F32).square().mean_dim(D - 1) + self.eps)
            .sqrt()
            .recip()
            .cast(dt);
        x * scale * self.weight.clone().unsqueeze::<D>()
    }
}
struct Attention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    qn: Rms<B>,
    kn: Rms<B>,
    heads: usize,
    kv: usize,
    width: usize,
}
impl<B: Backend> Attention<B> {
    fn load(l: &Load<B>, base: &str, c: &Config) -> Result<Self> {
        Ok(Self {
            q: l.linear(
                &format!("{base}.q_proj"),
                c.hidden_size,
                c.num_attention_heads * c.head_dim,
                false,
            )?,
            k: l.linear(
                &format!("{base}.k_proj"),
                c.hidden_size,
                c.num_key_value_heads * c.head_dim,
                false,
            )?,
            v: l.linear(
                &format!("{base}.v_proj"),
                c.hidden_size,
                c.num_key_value_heads * c.head_dim,
                false,
            )?,
            out: l.linear(
                &format!("{base}.o_proj"),
                c.num_attention_heads * c.head_dim,
                c.hidden_size,
                false,
            )?,
            qn: Rms::load(l, &format!("{base}.q_norm"), c.head_dim, c.rms_norm_eps)?,
            kn: Rms::load(l, &format!("{base}.k_norm"), c.head_dim, c.rms_norm_eps)?,
            heads: c.num_attention_heads,
            kv: c.num_key_value_heads,
            width: c.head_dim,
        })
    }
    fn project(
        &self,
        x: Tensor<B, 2>,
        r: &(Tensor<B, 4>, Tensor<B, 4>),
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let n = x.dims()[0];
        let q = self.qn.forward(
            self.q
                .forward(x.clone())
                .reshape([1, n, self.heads, self.width])
                .swap_dims(1, 2),
        );
        let k = self.kn.forward(
            self.k
                .forward(x.clone())
                .reshape([1, n, self.kv, self.width])
                .swap_dims(1, 2),
        );
        let v = self
            .v
            .forward(x)
            .reshape([1, n, self.kv, self.width])
            .swap_dims(1, 2);
        (ops::rope(q, r), ops::rope(k, r), v)
    }
    fn output(&self, y: Tensor<B, 4>) -> Tensor<B, 2> {
        let n = y.dims()[2];
        self.out
            .forward(y.swap_dims(1, 2).reshape([n, self.heads * self.width]))
    }
}
struct Branch<B: Backend> {
    norm: Rms<B>,
    attention: Attention<B>,
    post: Rms<B>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: Backend> Branch<B> {
    fn load(l: &Load<B>, base: &str, c: &Config, nar: bool) -> Result<Self> {
        let (norm, attn, post, mlp) = if nar {
            (
                "nar_input_layernorm",
                "nar_self_attn",
                "nar_pre_mlp_layernorm",
                "nar_mlp",
            )
        } else {
            (
                "input_layernorm",
                "self_attn",
                "post_attention_layernorm",
                "mlp",
            )
        };
        Ok(Self {
            norm: Rms::load(l, &format!("{base}.{norm}"), c.hidden_size, c.rms_norm_eps)?,
            attention: Attention::load(l, &format!("{base}.{attn}"), c)?,
            post: Rms::load(l, &format!("{base}.{post}"), c.hidden_size, c.rms_norm_eps)?,
            gate: l.linear(
                &format!("{base}.{mlp}.gate_proj"),
                c.hidden_size,
                c.intermediate_size,
                false,
            )?,
            up: l.linear(
                &format!("{base}.{mlp}.up_proj"),
                c.hidden_size,
                c.intermediate_size,
                false,
            )?,
            down: l.linear(
                &format!("{base}.{mlp}.down_proj"),
                c.intermediate_size,
                c.hidden_size,
                false,
            )?,
        })
    }
    fn finish(&self, x: Tensor<B, 2>, attended: Tensor<B, 4>) -> Tensor<B, 2> {
        let dt = tensor::dtype(&x);
        let x = tensor::residual(x, self.attention.output(attended), dt);
        let y = self.post.forward(x.clone());
        let y = silu(self.gate.forward(y.clone())) * self.up.forward(y);
        tensor::residual(x, self.down.forward(y), dt)
    }
}
struct Layer<B: Backend> {
    ar: Branch<B>,
    nar: Branch<B>,
}
pub struct Model<B: Backend> {
    config: Config,
    embedding: Embedding<B>,
    head: Embedding<B>,
    norm: Rms<B>,
    layers: Vec<Layer<B>>,
    vae_in: Linear<B>,
    vae_out: Linear<B>,
    time_in: Linear<B>,
    time_out: Linear<B>,
    tokenizer: TextTokenizer,
    precision: Precision,
    device: B::Device,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Plan {
    pub request: SongRequest,
    pub abc: Option<String>,
    pub abc_tokens: Vec<u32>,
    pub prefix: Vec<u32>,
    pub truncated: bool,
}
#[derive(Serialize, Deserialize)]
pub struct Semantic {
    pub plan: Plan,
    pub tokens: Vec<u32>,
    pub truncated: bool,
}
impl<B: Backend> Model<B> {
    pub fn load(
        path: &Path,
        p: Precision,
        d: &B::Device,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Self> {
        check_cancel(stop)?;
        let c: Config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        if c.vocab_size != 184704
            || c.latent_dim != 64
            || c.max_position_embeddings != CONTEXT
            || !c.head_dim.is_multiple_of(2)
            || c.num_hidden_layers == 0
            || c.num_attention_heads * c.head_dim != c.hidden_size
            || c.num_key_value_heads == 0
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
        {
            return Err(invalid("unsupported YuE2 configuration"));
        }
        let tokenizer = TextTokenizer::load(&path.join("qwen.tiktoken"))?;
        let w = Weights::open(path)?;
        let l = Load::<B> { w: &w, p, d };
        let embedding = Embedding::load(
            &w,
            "model.embed_tokens.weight",
            c.vocab_size,
            c.hidden_size,
            p,
            d,
        )?;
        let head = Embedding::load(&w, "lm_head.weight", c.vocab_size, c.hidden_size, p, d)?;
        let norm = Rms::load(&l, "model.norm", c.hidden_size, c.rms_norm_eps)?;
        let vae_in = l.linear("vae2llm", c.latent_dim, c.hidden_size, true)?;
        let vae_out = l.linear("llm2vae", c.hidden_size, c.latent_dim, true)?;
        let time_in = l.linear("time_embedder.mlp.0", 256, c.hidden_size, true)?;
        let time_out = l.linear("time_embedder.mlp.2", c.hidden_size, c.hidden_size, true)?;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            check_cancel(stop)?;
            let base = format!("model.layers.{i}");
            layers.push(Layer {
                ar: Branch::load(&l, &base, &c, false)?,
                nar: Branch::load(&l, &base, &c, true)?,
            });
            progress(Progress {
                stage: "load_yue",
                current: i + 1,
                total: c.num_hidden_layers,
            });
        }
        Ok(Self {
            config: c,
            embedding,
            head,
            norm,
            layers,
            vae_in,
            vae_out,
            time_in,
            time_out,
            tokenizer,
            precision: p,
            device: d.clone(),
        })
    }
    fn prefix(&self, r: &SongRequest, abc: Option<&[u32]>, negative: bool) -> Result<Vec<u32>> {
        let text = if negative {
            r.cot.instruction().to_string()
        } else {
            r.text()
        };
        let mut ids = vec![EOD];
        ids.extend(self.tokenizer.encode(&text)?);
        if r.cot == Planning::Off {
            if !negative {
                ids.extend([ABC_START, ABC_END]);
            }
            ids.push(MUSIC_START);
        } else {
            ids.push(ABC_START);
            if let Some(abc) = abc {
                if abc.iter().any(|&id| id >= EOD) {
                    return Err(invalid("ABC contains non-text tokens"));
                }
                ids.extend(abc);
                ids.extend([ABC_END, MUSIC_START]);
            }
        }
        Ok(ids)
    }
    pub fn plan(
        &self,
        request: &SongRequest,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Plan> {
        request.validate()?;
        check_cancel(stop)?;
        let (abc, abc_tokens, truncated) = if request.cot == Planning::Off {
            (None, Vec::new(), false)
        } else if let Some(abc) = &request.abc {
            (Some(abc.clone()), self.tokenizer.encode(abc)?, false)
        } else {
            let prefix = self.prefix(request, None, false)?;
            let (ids, truncated) = self.generate(
                &prefix,
                None,
                1.0,
                &request.abc_sampling,
                request.seed,
                true,
                false,
                request.paged_attention,
                stop,
                &progress,
            )?;
            (Some(self.tokenizer.decode(&ids)?), ids, truncated)
        };
        let prefix = self.prefix(request, Some(&abc_tokens), false)?;
        if prefix.len() + request.semantic_sampling.max_tokens > CONTEXT {
            return Err(invalid(
                "score and semantic token budget exceed YuE2 context",
            ));
        }
        Ok(Plan {
            request: request.clone(),
            abc,
            abc_tokens,
            prefix,
            truncated,
        })
    }
    pub fn semantic(
        &self,
        plan: Plan,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Semantic> {
        self.validate_plan(&plan)?;
        let r = &plan.request;
        let negative = if r.guidance() != 1.0 {
            Some(self.prefix(r, Some(&plan.abc_tokens), true)?)
        } else {
            None
        };
        let (ids, truncated) = self.generate(
            &plan.prefix,
            negative.as_deref(),
            r.guidance(),
            &r.semantic_sampling,
            r.seed,
            false,
            r.cot == Planning::Off,
            r.paged_attention,
            stop,
            &progress,
        )?;
        Ok(Semantic {
            plan,
            tokens: ids.into_iter().map(|id| id - CODEC_OFFSET).collect(),
            truncated,
        })
    }
    fn forward(
        &self,
        ids: &[u32],
        cache: &mut [DecodeCache<B>],
        stop: &AtomicBool,
    ) -> Result<Tensor<B, 2>> {
        if ids.is_empty() {
            return Err(invalid("empty YuE2 input"));
        }
        let offset = cache[0].len();
        let mut x = self.embedding.embed(ids)?;
        let r = ops::rotary(
            offset,
            ids.len(),
            self.config.head_dim,
            self.config.rope_theta,
            self.precision.dtype(),
            &self.device,
        );
        for (layer, kv) in self.layers.iter().zip(cache) {
            check_cancel(stop)?;
            let branch = &layer.ar;
            let (q, k, v) = branch.attention.project(branch.norm.forward(x.clone()), &r);
            let y = kv.forward(
                q,
                k,
                v,
                AttentionOptions {
                    causal: true,
                    offset,
                    kernel: tensor::AttentionKernel::ImageAligned,
                    ..Default::default()
                },
            );
            x = branch.finish(x, y);
        }
        let n = ids.len();
        Ok(self.head.logits(
            self.norm
                .forward(x.slice([n - 1..n, 0..self.config.hidden_size])),
        ))
    }
    fn prefill(
        &self,
        ids: &[u32],
        cache: &mut [DecodeCache<B>],
        stop: &AtomicBool,
    ) -> Result<Tensor<B, 2>> {
        let mut result = None;
        for block in ids.chunks(256) {
            result = Some(self.forward(block, cache, stop)?);
            B::sync(&self.device).map_err(|e| invalid(e.to_string()))?;
        }
        result.ok_or_else(|| invalid("empty YuE2 prompt"))
    }
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        prefix: &[u32],
        negative: Option<&[u32]>,
        guidance: f32,
        s: &Sampling,
        seed: u64,
        abc: bool,
        legacy: bool,
        paged: bool,
        stop: &AtomicBool,
        progress: &impl Fn(Progress),
    ) -> Result<(Vec<u32>, bool)> {
        if prefix.len() + s.max_tokens > CONTEXT
            || negative.is_some_and(|n| n.len() + s.max_tokens > CONTEXT)
        {
            return Err(invalid(
                "YuE2 token budget exceeds context; no implicit truncation",
            ));
        }
        let mut cache: Vec<_> = self
            .layers
            .iter()
            .map(|_| DecodeCache::new(paged))
            .collect();
        let mut nc: Vec<_> = self
            .layers
            .iter()
            .map(|_| DecodeCache::new(paged))
            .collect();
        let mut positive = self.prefill(prefix, &mut cache, stop)?;
        let mut negative = negative
            .map(|ids| self.prefill(ids, &mut nc, stop))
            .transpose()?;
        let mut history = Vec::new();
        let mut sampler = Sampler::new(seed);
        let end = if abc { ABC_END } else { MUSIC_END };
        for step in 0..s.max_tokens {
            check_cancel(stop)?;
            let logits = if let Some(n) = &negative {
                n.clone() + (positive - n.clone()) * guidance
            } else {
                positive
            };
            let id = sampler.choose(ops::values(logits)?, s, &history, abc, legacy)?;
            progress(Progress {
                stage: if abc { "score_plan" } else { "music_tokens" },
                current: step + 1,
                total: s.max_tokens,
            });
            if id == end {
                return Ok((history, false));
            }
            history.push(id);
            if step + 1 == s.max_tokens {
                break;
            }
            positive = self.forward(&[id], &mut cache, stop)?;
            if negative.is_some() {
                negative = Some(self.forward(&[id], &mut nc, stop)?);
            }
        }
        Ok((history, true))
    }
    pub fn synthesize(
        &self,
        semantic: &Semantic,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Vec<f32>> {
        self.validate_semantic(semantic)?;
        check_cancel(stop)?;
        let mut rng = StdRng::seed_from_u64(semantic.plan.request.seed);
        let noise: Vec<f32> = (0..semantic.tokens.len() * 64)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        self.synthesize_with_noise(semantic, &noise, stop, progress)
    }

    /// Accept the same FP32 Gaussian noise as another implementation for numerical
    /// comparisons, or retain it alongside semantic tokens for reproducible synthesis.
    pub fn synthesize_with_noise(
        &self,
        semantic: &Semantic,
        noise: &[f32],
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Vec<f32>> {
        self.validate_semantic(semantic)?;
        if noise.len() != semantic.tokens.len() * 64 || noise.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "music tokens or initial noise do not match the request",
            ));
        }
        let prefix = &semantic.plan.prefix;
        let chunk = CONTEXT
            .checked_sub(prefix.len() + 3)
            .map(|v| v / 2)
            .filter(|v| *v > 0)
            .ok_or_else(|| invalid("score leaves no acoustic context"))?;
        let mut latents = Vec::with_capacity(noise.len());
        let chunks = semantic.tokens.len().div_ceil(chunk);
        let steps = semantic.plan.request.ode_steps;
        for (index, tokens) in semantic.tokens.chunks(chunk).enumerate() {
            check_cancel(stop)?;
            let mut ids = prefix.clone();
            ids.extend(tokens.iter().map(|v| v + CODEC_OFFSET));
            ids.push(MUSIC_END);
            let ar_length = ids.len();
            let n = tokens.len();
            let cache = self.acoustic_prefill(&ids, stop)?;
            let r = ops::rotary(
                ar_length,
                n + 2,
                self.config.head_dim,
                self.config.rope_theta,
                self.precision.dtype(),
                &self.device,
            );
            let position = self.latent_positions(n + 2);
            let mut state = ops::data(
                noise[index * chunk * 64..index * chunk * 64 + n * 64].to_vec(),
                [n, 64],
                self.precision,
                &self.device,
            );
            let dt = 1.0 / steps as f64;
            for step in 0..steps {
                check_cancel(stop)?;
                let time = 1.0 - step as f64 * dt;
                let first = self.velocity(state.clone(), time, &cache, &r, &position, stop)?;
                let mid = state.clone() - first * (dt / 2.0);
                state =
                    state - self.velocity(mid, time - dt / 2.0, &cache, &r, &position, stop)? * dt;
                B::sync(&self.device).map_err(|e| invalid(e.to_string()))?;
                progress(Progress {
                    stage: "music_acoustics",
                    current: index * steps + step + 1,
                    total: chunks * steps,
                });
            }
            let result = ops::values(state)?;
            if result.iter().any(|v| !v.is_finite()) {
                return Err(invalid("non-finite music latents"));
            }
            latents.extend(result);
        }
        Ok(latents)
    }
    fn validate_plan(&self, plan: &Plan) -> Result<()> {
        plan.request.validate()?;
        if plan.abc_tokens.len() > CONTEXT || plan.prefix.len() > CONTEXT {
            return Err(invalid("saved score exceeds the model context"));
        }
        let valid_score = match plan.request.cot {
            Planning::Off => plan.abc.is_none() && plan.abc_tokens.is_empty(),
            _ => {
                if let Some(external) = &plan.request.abc {
                    plan.abc.as_ref() == Some(external)
                        && self.tokenizer.encode(external)? == plan.abc_tokens
                } else {
                    plan.abc.as_deref() == Some(self.tokenizer.decode(&plan.abc_tokens)?.as_str())
                }
            }
        };
        if !valid_score || self.prefix(&plan.request, Some(&plan.abc_tokens), false)? != plan.prefix
        {
            return Err(invalid(
                "saved plan score or prefix does not match its request",
            ));
        }
        Ok(())
    }
    fn validate_semantic(&self, semantic: &Semantic) -> Result<()> {
        self.validate_plan(&semantic.plan)?;
        if semantic.tokens.is_empty()
            || semantic.tokens.len() > semantic.plan.request.semantic_sampling.max_tokens
            || semantic.tokens.iter().any(|&v| v >= CODEC_SIZE)
        {
            return Err(invalid("empty, excessive or invalid music codec tokens"));
        }
        Ok(())
    }
    fn acoustic_prefill(
        &self,
        ids: &[u32],
        stop: &AtomicBool,
    ) -> Result<Vec<(Tensor<B, 4>, Tensor<B, 4>)>> {
        let mut x = self.embedding.embed(ids)?;
        let r = ops::rotary(
            0,
            ids.len(),
            self.config.head_dim,
            self.config.rope_theta,
            self.precision.dtype(),
            &self.device,
        );
        let mut cache = Vec::new();
        for layer in &self.layers {
            check_cancel(stop)?;
            let b = &layer.ar;
            let (q, k, v) = b.attention.project(b.norm.forward(x.clone()), &r);
            let y = ops::attend(q, k.clone(), v.clone(), true, 0);
            cache.push((k, v));
            x = b.finish(x, y);
            B::sync(&self.device).map_err(|e| invalid(e.to_string()))?;
        }
        Ok(cache)
    }
    fn latent_positions(&self, n: usize) -> Tensor<B, 2> {
        let w = self.config.hidden_size;
        let mut values = Vec::with_capacity(n * w);
        for pos in 0..n {
            for pair in 0..w / 2 {
                let angle = pos as f32 * (-10000f32.ln() * (2 * pair) as f32 / w as f32).exp();
                values.extend([angle.sin(), angle.cos()]);
            }
        }
        ops::data(values, [n, w], self.precision, &self.device)
    }
    fn velocity(
        &self,
        state: Tensor<B, 2>,
        time: f64,
        cache: &[(Tensor<B, 4>, Tensor<B, 4>)],
        r: &(Tensor<B, 4>, Tensor<B, 4>),
        position: &Tensor<B, 2>,
        stop: &AtomicBool,
    ) -> Result<Tensor<B, 2>> {
        let n = state.dims()[0];
        let pad = Tensor::<B, 2>::zeros([1, 64], &self.device).cast(self.precision.dtype());
        let z = Tensor::cat(vec![pad.clone(), state, pad], 0);
        let time = time.clamp(1.0 / (1.0 + 20f64.exp()), 1.0 / (1.0 + (-20f64).exp()));
        let t =
            self.config.timestep_shift * time / (1.0 + (self.config.timestep_shift - 1.0) * time);
        let angles: Vec<f32> = (0..128)
            .map(|i| t as f32 * (-10000f32.ln() * i as f32 / 128.0).exp())
            .collect();
        let values = angles
            .iter()
            .map(|v| v.cos())
            .chain(angles.iter().map(|v| v.sin()))
            .collect();
        let time = self.time_out.forward(silu(self.time_in.forward(ops::data(
            values,
            [1, 256],
            self.precision,
            &self.device,
        ))));
        let mut x = self.vae_in.forward(z) + time + position.clone();
        for (layer, (ar_k, ar_v)) in self.layers.iter().zip(cache) {
            check_cancel(stop)?;
            let b = &layer.nar;
            let (q, k, v) = b.attention.project(b.norm.forward(x.clone()), r);
            let y = ops::attend(
                q,
                Tensor::cat(vec![ar_k.clone(), k], 2),
                Tensor::cat(vec![ar_v.clone(), v], 2),
                false,
                0,
            );
            x = b.finish(x, y);
        }
        Ok(self
            .vae_out
            .forward(self.norm.forward(x))
            .slice([1..n + 1, 0..64]))
    }
}
