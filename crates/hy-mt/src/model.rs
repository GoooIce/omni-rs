use crate::{Config, Error, Precision, Result, check_cancel};
use burn::tensor::{Tensor, TensorData, activation::silu, backend::Backend};
use omni_r2t2::{
    tensor::{
        self, AttentionOptions, Embedding, KvCache, Linear, Norm, NormKind, NormReduction,
        RotaryLayout,
    },
    weights::WeightSource,
};
use std::sync::atomic::AtomicBool;

/// Burn 0.21 Metal can fuse the attention projection, FP16 residual and
/// following RMSNorm incorrectly for short unaligned rows (the 3-token
/// official reference reproduces it). An identity CustomOp separates those
/// graphs without a device fence, CPU readback or changing FP16 rounding.
fn attention_residual<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if x.dtype() == burn::tensor::DType::F16 {
        use burn::tensor::{Shape, TensorPrimitive};
        if let Some(value) = omni_kernel::dispatch::<B, 1>(
            "hy_mt_attention_residual",
            [x.clone().into_primitive().tensor()],
            Shape::new(x.dims()),
            |[x]| x,
        ) {
            return Tensor::from_primitive(TensorPrimitive::Float(value));
        }
    }
    x
}

struct Layer<B: Backend> {
    input_norm: Norm<B>,
    post_norm: Norm<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: Backend> Layer<B> {
    fn load(
        w: &impl WeightSource<B>,
        index: usize,
        c: &Config,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let base = format!("model.layers.{index}");
        let norm = |name: &str, width| {
            Norm::load(
                w,
                &format!("{base}.{name}"),
                width,
                c.rms_norm_eps,
                NormKind::Rms,
                NormReduction::MultiplyReciprocal,
                d,
            )
        };
        let linear = |name: &str, input, output, bias| {
            Linear::load(w, &format!("{base}.{name}"), input, output, bias, p, d)
        };
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        Ok(Self {
            input_norm: norm("input_layernorm", h)?,
            post_norm: norm("post_attention_layernorm", h)?,
            q_norm: norm("self_attn.query_layernorm", c.head_dim)?,
            k_norm: norm("self_attn.key_layernorm", c.head_dim)?,
            q: linear("self_attn.q_proj", h, h, c.attention_bias)?,
            k: linear("self_attn.k_proj", h, kv, c.attention_bias)?,
            v: linear("self_attn.v_proj", h, kv, c.attention_bias)?,
            out: linear("self_attn.o_proj", h, h, c.attention_bias)?,
            gate: linear("mlp.gate_proj", h, c.intermediate_size, false)?,
            up: linear("mlp.up_proj", h, c.intermediate_size, false)?,
            down: linear("mlp.down_proj", c.intermediate_size, h, false)?,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 2>,
        kv: &mut KvCache<B>,
        c: &Config,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let n = x.dims()[0];
        let dtype = tensor::dtype(&x);
        let y = self.input_norm.forward(x.clone(), dtype);
        let q = self
            .q
            .forward(y.clone())
            .reshape([1, n, c.num_attention_heads, c.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k
            .forward(y.clone())
            .reshape([1, n, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v
            .forward(y)
            .reshape([1, n, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        // Hunyuan normalizes AFTER rotation, unlike the Qwen ASR decoder.
        let q = self
            .q_norm
            .forward(tensor::rope(q, cos, sin, RotaryLayout::SplitHalf), dtype);
        let k = self
            .k_norm
            .forward(tensor::rope(k, cos, sin, RotaryLayout::SplitHalf), dtype);
        let offset = kv.len();
        let (k, v) = kv.append(k, v, offset);
        let y = tensor::attention(
            q,
            k,
            v,
            AttentionOptions {
                causal: true,
                offset,
                ..Default::default()
            },
        )
        .swap_dims(1, 2)
        .reshape([n, c.hidden_size]);
        let x = attention_residual(tensor::residual(x, self.out.forward(y), dtype));
        let y = self.post_norm.forward(x.clone(), dtype);
        let y = silu(self.gate.forward(y.clone())) * self.up.forward(y);
        tensor::residual(x, self.down.forward(y), dtype)
    }

    fn forward_paged(
        &self,
        x: Tensor<B, 2>,
        atlas: &mut Option<Tensor<B, 5>>,
        c: &Config,
        rotary: (&Tensor<B, 4>, &Tensor<B, 4>),
        batch: &crate::paged::Batch,
        capacity: usize,
    ) -> Tensor<B, 2> {
        let (cos, sin) = rotary;
        let n = x.dims()[0];
        let dtype = tensor::dtype(&x);
        let y = self.input_norm.forward(x.clone(), dtype);
        let q = self
            .q
            .forward(y.clone())
            .reshape([1, n, c.num_attention_heads, c.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k
            .forward(y.clone())
            .reshape([1, n, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v
            .forward(y)
            .reshape([n, c.num_key_value_heads, c.head_dim]);
        let q = self
            .q_norm
            .forward(tensor::rope(q, cos, sin, RotaryLayout::SplitHalf), dtype)
            .swap_dims(1, 2)
            .reshape([n, c.num_attention_heads, c.head_dim]);
        let k = self
            .k_norm
            .forward(tensor::rope(k, cos, sin, RotaryLayout::SplitHalf), dtype)
            .swap_dims(1, 2)
            .reshape([n, c.num_key_value_heads, c.head_dim]);
        let storage = atlas.take().unwrap_or_else(|| {
            Tensor::empty(
                [
                    capacity,
                    2,
                    crate::pages::PAGE_SIZE,
                    c.num_key_value_heads,
                    c.head_dim,
                ],
                (&x.device(), x.dtype()),
            )
        });
        let storage = crate::paged::write(storage, k, v, batch);
        let y = crate::paged::attend(q, &storage, batch).reshape([n, c.hidden_size]);
        *atlas = Some(storage);
        let x = attention_residual(tensor::residual(x, self.out.forward(y), dtype));
        let y = self.post_norm.forward(x.clone(), dtype);
        let y = silu(self.gate.forward(y.clone())) * self.up.forward(y);
        tensor::residual(x, self.down.forward(y), dtype)
    }
}

pub struct Model<B: Backend> {
    config: Config,
    embedding: Embedding<B>,
    head: Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: Norm<B>,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) eos: Vec<u32>,
    frequencies: Vec<f64>,
    precision: Precision,
    device: B::Device,
}
impl<B: Backend> Model<B> {
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn precision(&self) -> Precision {
        self.precision
    }
    pub(crate) fn device(&self) -> &B::Device {
        &self.device
    }

    /// All scheduled query rows share projection/MLP matmuls. The block table
    /// supplies independent causal histories and positions for ragged requests.
    pub(crate) fn forward_batch(
        &self,
        batch: &crate::paged::Batch,
        atlas: &mut [Option<Tensor<B, 5>>],
        capacity: usize,
    ) -> Result<Tensor<B, 2>> {
        let mut x = self.embedding.embed(&batch.ids)?;
        let dtype = tensor::dtype(&x);
        let width = self.config.head_dim;
        let mut cosine = Vec::with_capacity(batch.ids.len() * width);
        let mut sine = Vec::with_capacity(batch.ids.len() * width);
        for &pos in &batch.positions {
            for &freq in self.frequencies.iter().cycle().take(width) {
                let angle = pos as f64 * freq;
                cosine.push(angle.cos() as f32);
                sine.push(angle.sin() as f32);
            }
        }
        let shape = [1, 1, batch.ids.len(), width];
        let cos =
            Tensor::<B, 4>::from_data(TensorData::new(cosine, shape), &self.device).cast(dtype);
        let sin = Tensor::<B, 4>::from_data(TensorData::new(sine, shape), &self.device).cast(dtype);
        for (layer, storage) in self.layers.iter().zip(atlas) {
            x = layer.forward_paged(x, storage, &self.config, (&cos, &sin), batch, capacity);
        }
        let last = x.select(0, batch.last_rows::<B>(&self.device));
        Ok(self.head.logits(self.norm.forward(last, dtype)))
    }
    pub fn load(
        c: Config,
        tokenizer: &[u8],
        w: &impl WeightSource<B>,
        p: Precision,
        d: &B::Device,
        cancel: &AtomicBool,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let controlled = omni_r2t2::loading::CancellableWeights::new(w, cancel, |_: &str| {});
        let w = &controlled;
        c.validate()?;
        p.validate_device::<B>(d)?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        for (token, expected) in [
            ("<|startoftext|>", c.bos_token_id),
            ("<|eos|>", c.eos_token_id),
        ] {
            if tokenizer.token_to_id(token) != Some(expected) {
                return Err(Error::Tokenizer(format!(
                    "Hy-MT2 special token mismatch: {token}"
                )));
            }
        }
        if tokenizer.token_to_id("<|extra_0|>").is_none() {
            return Err(Error::Tokenizer("missing Hy-MT2 user separator".into()));
        }
        let embedding = Embedding::load_with(
            w,
            "model.embed_tokens.weight",
            c.vocab_size,
            c.hidden_size,
            p,
            d,
            || check_cancel(cancel),
        )?;
        let head = if c.tie_word_embeddings {
            embedding.clone()
        } else {
            Embedding::load_with(
                w,
                "lm_head.weight",
                c.vocab_size,
                c.hidden_size,
                p,
                d,
                || check_cancel(cancel),
            )?
        };
        progress(1, c.num_hidden_layers + 2);
        let mut layers = Vec::new();
        for index in 0..c.num_hidden_layers {
            check_cancel(cancel)?;
            layers.push(Layer::load(w, index, &c, p, d)?);
            B::sync(d).map_err(|e| Error::Inference(format!("weight upload: {e:?}")))?;
            progress(index + 2, c.num_hidden_layers + 2);
        }
        let norm = Norm::load(
            w,
            "model.norm",
            c.hidden_size,
            c.rms_norm_eps,
            NormKind::Rms,
            NormReduction::MultiplyReciprocal,
            d,
        )?;
        let frequencies = (0..c.head_dim / 2)
            .map(|i| {
                // Transformers constructs inverse frequencies and positions in FP32.
                (1.0_f32 / (c.rotary_base() as f32).powf((2 * i) as f32 / c.head_dim as f32)) as f64
            })
            .collect();
        let mut eos = vec![c.eos_token_id];
        // generation_config.json terminates both normal text and fast thinking.
        if let Some(id) = tokenizer.token_to_id("<|extra_4|>") {
            eos.push(id);
        }
        progress(c.num_hidden_layers + 2, c.num_hidden_layers + 2);
        check_cancel(cancel)?;
        Ok(Self {
            config: c,
            embedding,
            head,
            layers,
            norm,
            tokenizer,
            eos,
            frequencies,
            precision: p,
            device: d.clone(),
        })
    }
    #[cfg(feature = "native")]
    pub fn from_directory(
        path: impl AsRef<std::path::Path>,
        p: Precision,
        d: &B::Device,
        cancel: &AtomicBool,
        progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let path = path.as_ref();
        let c = Config::from_json(&std::fs::read(path.join("config.json"))?)?;
        check_cancel(cancel)?;
        let weights = omni_r2t2::weights::Weights::open(path)?;
        check_cancel(cancel)?;
        let mut model = Self::load(
            c,
            &std::fs::read(path.join("tokenizer.json"))?,
            &weights,
            p,
            d,
            cancel,
            progress,
        )?;
        if path.join("generation_config.json").is_file() {
            let generation: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join("generation_config.json"))?)?;
            if let Some(ids) = generation.get("eos_token_id") {
                let ids = if let Some(id) = ids.as_u64() {
                    vec![id]
                } else {
                    ids.as_array()
                        .ok_or_else(|| Error::Config("invalid EOS token list".into()))?
                        .iter()
                        .map(|id| {
                            id.as_u64()
                                .ok_or_else(|| Error::Config("invalid EOS token".into()))
                        })
                        .collect::<Result<Vec<_>>>()?
                };
                if ids.is_empty() || ids.iter().any(|&id| id >= model.config.vocab_size as u64) {
                    return Err(Error::Config("invalid EOS token IDs".into()));
                }
                model.eos = ids.into_iter().map(|id| id as u32).collect();
            }
        }
        check_cancel(cancel)?;
        Ok(model)
    }
    pub fn empty_history(&self) -> Vec<KvCache<B>> {
        (0..self.layers.len()).map(|_| KvCache::default()).collect()
    }
    /// Diagnostic and prefill interface. The caller publishes KV only after all
    /// layers succeed; an interrupted partially updated history must be dropped.
    pub fn forward_tokens(
        &self,
        ids: &[u32],
        history: &mut [KvCache<B>],
        cancel: &AtomicBool,
    ) -> Result<Tensor<B, 2>> {
        check_cancel(cancel)?;
        if history.len() != self.layers.len() {
            return Err(Error::Inference("incorrect layer count".into()));
        }
        let offset = history[0].len();
        if ids.is_empty()
            || offset
                .checked_add(ids.len())
                .is_none_or(|n| n > self.config.max_position_embeddings)
            || history.iter().any(|l| l.len() != offset)
        {
            return Err(Error::Inference("invalid translation context".into()));
        }
        let mut x = self.embedding.embed(ids)?;
        let dtype = tensor::dtype(&x);
        let (cos, sin) =
            tensor::rope_tables(&self.frequencies, offset, ids.len(), dtype, &self.device);
        for (layer, kv) in self.layers.iter().zip(history) {
            check_cancel(cancel)?;
            x = layer.forward(x, kv, &self.config, &cos, &sin);
        }
        let last = x.slice([ids.len() - 1..ids.len(), 0..self.config.hidden_size]);
        Ok(self.head.logits(self.norm.forward(last, dtype)))
    }
}
