use crate::{Error, Precision, Progress, Result, check_cancel, config::TextConfig, ops::*};
use burn::tensor::{Tensor, activation::silu, backend::Backend};
use omni_r2t2::{loading::CancellableWeights, weights::WeightSource};
use omni_r2t2::{tensor::Embedding, weights::Weights};
use std::{path::Path, sync::atomic::AtomicBool};

pub const SYSTEM: &str = "Comprehend and analyze the provided prompt.";

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ImageSpan {
    pub start: usize,
    pub end: usize,
    /// Spatial grid after the vision tower's 2×2 merger.
    pub height: usize,
    pub width: usize,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct PromptLayout {
    pub tokens: Vec<u32>,
    pub positions: Vec<[f64; 3]>,
    pub images: Vec<ImageSpan>,
    pub drop_tokens: usize,
}
impl PromptLayout {
    /// Diffusion sees the sequence after the leading system message is dropped.
    pub fn condition_spans(&self) -> Vec<ImageSpan> {
        self.images
            .iter()
            .map(|s| ImageSpan {
                start: s.start - self.drop_tokens,
                end: s.end - self.drop_tokens,
                height: s.height,
                width: s.width,
            })
            .collect()
    }
}

struct Layer<B: Backend> {
    norm1: Norm<B>,
    norm2: Norm<B>,
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
        name: &str,
        c: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let linear = |s: &str, i, o| Linear::load(w, &format!("{name}.{s}"), i, o, false, p, d);
        let norm = |s: &str, n| {
            Norm::load(
                w,
                &format!("{name}.{s}"),
                n,
                c.rms_norm_eps,
                false,
                false,
                d,
            )
        };
        let h = c.hidden_size;
        Ok(Self {
            norm1: norm("input_layernorm", h)?,
            norm2: norm("post_attention_layernorm", h)?,
            q_norm: norm("self_attn.q_norm", c.head_dim)?,
            k_norm: norm("self_attn.k_norm", c.head_dim)?,
            q: linear("self_attn.q_proj", h, c.num_attention_heads * c.head_dim)?,
            k: linear("self_attn.k_proj", h, c.num_key_value_heads * c.head_dim)?,
            v: linear("self_attn.v_proj", h, c.num_key_value_heads * c.head_dim)?,
            out: linear("self_attn.o_proj", c.num_attention_heads * c.head_dim, h)?,
            gate: linear("mlp.gate_proj", h, c.intermediate_size)?,
            up: linear("mlp.up_proj", h, c.intermediate_size)?,
            down: linear("mlp.down_proj", c.intermediate_size, h)?,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &Rotary<B>,
        c: &TextConfig,
        p: Precision,
        trace: &mut impl FnMut(&str, Tensor<B, 3>) -> Result<()>,
    ) -> Result<Tensor<B, 3>> {
        let [b, n, h] = x.dims();
        let y = self.norm1.forward(x.clone(), p);
        trace("norm1", y.clone())?;
        let q = self.q.forward(y.clone());
        let k = self.k.forward(y.clone());
        let v = self.v.forward(y);
        trace("q_proj", q.clone())?;
        trace("k_proj", k.clone())?;
        trace("v_proj", v.clone())?;
        let q = q.reshape([b, n, c.num_attention_heads, c.head_dim]);
        let k = k.reshape([b, n, c.num_key_value_heads, c.head_dim]);
        let v = v
            .reshape([b, n, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        let q = rope.apply(self.q_norm.forward(q, p).swap_dims(1, 2), 0);
        let k = rope.apply(self.k_norm.forward(k, p).swap_dims(1, 2), 0);
        let y = attention(q, k, v, true, 0)
            .swap_dims(1, 2)
            .reshape([b, n, h]);
        trace("attention", y.clone())?;
        let x = residual(x, self.out.forward(y), p);
        trace("post_attention", x.clone())?;
        let y = self.norm2.forward(x.clone(), p);
        trace("norm2", y.clone())?;
        let y = silu(self.gate.forward(y.clone())) * self.up.forward(y);
        Ok(residual(x, self.down.forward(y), p))
    }
}

pub struct TextEncoder<B: Backend> {
    config: TextConfig,
    embeddings: Embedding<B>,
    layers: Vec<Layer<B>>,
    tokenizer: tokenizers::Tokenizer,
    drop_tokens: usize,
    p: Precision,
    device: B::Device,
}
impl<B: Backend> TextEncoder<B> {
    pub fn load(
        root: &Path,
        c: TextConfig,
        p: Precision,
        d: &B::Device,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let weights = Weights::open(&root.join("text_encoder"))?;
        let w = CancellableWeights::new(&weights, cancel, |_: &str| {});
        let base = "model.language_model";
        let embeddings = Embedding::load_with(
            &w,
            &format!("{base}.embed_tokens.weight"),
            c.vocab_size,
            c.hidden_size,
            p,
            d,
            || check_cancel(cancel),
        )?;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            check_cancel(cancel)?;
            layers.push(Layer::load(&w, &format!("{base}.layers.{i}"), &c, p, d)?);
            B::sync(d).map_err(|e| Error::Config(e.to_string()))?;
            progress(Progress {
                stage: "load_text",
                current: i + 1,
                total: c.num_hidden_layers,
            });
        }
        let tokenizer = tokenizers::Tokenizer::from_file(root.join("processor/tokenizer.json"))
            .map_err(|e| Error::Config(e.to_string()))?;
        let system = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n");
        let drop_tokens = tokenizer
            .encode(system, false)
            .map_err(|e| Error::Config(e.to_string()))?
            .len();
        Ok(Self {
            config: c,
            embeddings,
            layers,
            tokenizer,
            drop_tokens,
            p,
            device: d.clone(),
        })
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        // An empty negative prompt is the official unconditional condition.
        // Positive-prompt validation belongs to Options, before model execution.
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let text = format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Config(e.to_string()))?
            .get_ids()
            .to_vec();
        if ids.len() > 4096 {
            return Err(Error::Config(
                "描述过长：当前本地入口最多支持 4096 个文本 token".into(),
            ));
        }
        Ok(ids)
    }

    pub fn layout(&self, prompt: &str, grids: &[(usize, usize)]) -> Result<PromptLayout> {
        let prompt = if prompt.is_empty() { " " } else { prompt };
        if grids.is_empty() {
            let tokens = self.tokenize(prompt)?;
            let positions = (0..tokens.len()).map(|i| [i as f64; 3]).collect();
            return Ok(PromptLayout {
                tokens,
                positions,
                images: vec![],
                drop_tokens: self.drop_tokens,
            });
        }
        if grids.len() > crate::reference::MAX_REFERENCES
            || grids
                .iter()
                .any(|(h, w)| *h == 0 || *w == 0 || h * w > 4096)
        {
            return Err(Error::Config("参考图数量或视觉 token 数超出限制".into()));
        }
        // This raw template is checkpoint-specific; apply_chat_template differs.
        let mut prefix = String::new();
        for (i, (h, w)) in grids.iter().enumerate() {
            if i > 0 {
                prefix.push(' ');
            }
            prefix.push_str(&format!(
                "<image{}><|vision_start|>{}<|vision_end|>",
                i + 1,
                "<|image_pad|>".repeat(h * w)
            ));
        }
        let text = format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let tokens = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Config(e.to_string()))?
            .get_ids()
            .to_vec();
        if tokens.len() > 8192 {
            return Err(Error::Config(
                "图片与指令合计超过 8192 个 token，请缩小参考图或缩短指令".into(),
            ));
        }
        let mut images = Vec::new();
        let mut positions = Vec::with_capacity(tokens.len());
        let mut cursor = 0;
        let mut position = 0;
        while cursor < tokens.len() {
            if tokens[cursor] == 151655 {
                let &(height, width) = grids
                    .get(images.len())
                    .ok_or_else(|| Error::Config("图片占位符不匹配".into()))?;
                let end = cursor + height * width;
                if end > tokens.len() || tokens[cursor..end].iter().any(|id| *id != 151655) {
                    return Err(Error::Config("图片 token 数不匹配".into()));
                }
                for y in 0..height {
                    for x in 0..width {
                        positions.push([
                            position as f64,
                            (position + y) as f64,
                            (position + x) as f64,
                        ]);
                    }
                }
                images.push(ImageSpan {
                    start: cursor,
                    end,
                    height,
                    width,
                });
                position += height.max(width);
                cursor = end;
            } else {
                positions.push([position as f64; 3]);
                position += 1;
                cursor += 1;
            }
        }
        if images.len() != grids.len() {
            return Err(Error::Config("图片占位符数量不匹配".into()));
        }
        Ok(PromptLayout {
            tokens,
            positions,
            images,
            drop_tokens: self.drop_tokens,
        })
    }

    /// The diffusion model consumes the last decoder output BEFORE final RMSNorm.
    pub fn encode(
        &self,
        prompt: &str,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Tensor<B, 3>> {
        self.encode_with_trace(prompt, cancel, progress, &mut |_, _| Ok(()))
    }

    pub fn encode_with_trace(
        &self,
        prompt: &str,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
        trace: &mut impl FnMut(&str, Tensor<B, 3>) -> Result<()>,
    ) -> Result<Tensor<B, 3>> {
        let layout = self.layout(prompt, &[])?;
        self.encode_multimodal(&layout, &[], cancel, progress, trace)
    }

    pub fn encode_multimodal(
        &self,
        layout: &PromptLayout,
        images: &[crate::vision::Features<B>],
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
        trace: &mut impl FnMut(&str, Tensor<B, 3>) -> Result<()>,
    ) -> Result<Tensor<B, 3>> {
        let n = layout.tokens.len();
        let h = self.config.hidden_size;
        if images.len() != layout.images.len() {
            return Err(Error::Config("视觉特征与文本占位符不匹配".into()));
        }
        let mut x = self.embeddings.forward(&layout.tokens)?;
        // Burn 0.21's fusion codegen can misindex the outputs when this variable
        // length shard gather/cat is fused with the first RMS reduction. Flush
        // once here; all transformer blocks still use GPU fusion normally.
        B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
        for (span, image) in layout.images.iter().zip(images) {
            if image.embedding.dims() != [1, span.end - span.start, h] || image.deepstack.len() != 3
            {
                return Err(Error::Config("视觉特征形状不匹配".into()));
            }
            x = x.slice_assign([0..1, span.start..span.end, 0..h], image.embedding.clone());
        }
        trace("embedding", x.clone())?;
        let rope = Rotary::multimodal(
            &layout.positions,
            self.config.rope_theta,
            self.p,
            &self.device,
        );
        for (i, layer) in self.layers.iter().enumerate() {
            check_cancel(cancel)?;
            x = layer.forward(x, &rope, &self.config, self.p, &mut |name, x| {
                if i == 0 { trace(name, x) } else { Ok(()) }
            })?;
            if [0, 1, 3, 11, 35].contains(&i) {
                trace(&format!("layer_{i}"), x.clone())?;
            }
            if i < 3 {
                for (span, image) in layout.images.iter().zip(images) {
                    let range = [0..1, span.start..span.end, 0..h];
                    let y = residual(
                        x.clone().slice(range.clone()),
                        image.deepstack[i].clone(),
                        self.p,
                    );
                    x = x.slice_assign(range, y);
                }
            }
            if (i + 1) % 4 == 0 || i + 1 == self.layers.len() {
                B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
                progress(Progress {
                    stage: "text",
                    current: i + 1,
                    total: self.layers.len(),
                });
            }
        }
        check_cancel(cancel)?;
        Ok(x.slice([0..1, self.drop_tokens..n, 0..self.config.hidden_size]))
    }
}
