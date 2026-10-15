use crate::{Precision, Result, config::TransformerConfig, ops::*};
use burn::tensor::{FloatDType, Tensor, activation::silu, backend::Backend};
use omni_r2t2::{
    tensor::{AttentionOptions, DecodeCache, KvCache},
    weights::WeightSource,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, PartialEq)]
pub enum Architecture {
    Gemma,
    Qwen,
    Llama,
    Codec,
    Mimi,
}

pub struct TransformerCache<B: Backend> {
    layers: Vec<Option<KvCache<B>>>,
    pub position: usize,
    paged_layers: Option<Vec<DecodeCache<B>>>,
}
impl<B: Backend> TransformerCache<B> {
    /// Reuse the small layer-slot allocation, with no history crossing frames.
    pub fn reset(&mut self) {
        self.position = 0;
        if let Some(layers) = &mut self.paged_layers {
            for layer in layers {
                layer.clear();
            }
        }
        for layer in &mut self.layers {
            *layer = None;
        }
    }
    pub fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| None).collect(),
            position: 0,
            paged_layers: None,
        }
    }
}

struct Layer<B: Backend> {
    norm1: Norm<B>,
    norm2: Norm<B>,
    post1: Option<Norm<B>>,
    post2: Option<Norm<B>>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    q_norm: Option<Norm<B>>,
    k_norm: Option<Norm<B>>,
    gate: Linear<B>,
    up: Option<Linear<B>>,
    down: Linear<B>,
    scale1: Option<Tensor<B, 1>>,
    scale2: Option<Tensor<B, 1>>,
    config: TransformerConfig,
    architecture: Architecture,
    rope_kind: usize,
    window: Option<usize>,
}
impl<B: Backend> Layer<B> {
    fn load(
        w: &impl WeightSource<B>,
        base: &str,
        c: &TransformerConfig,
        kind: Architecture,
        index: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let gemma = kind == Architecture::Gemma;
        let width = c.hidden_size;
        let norm = |name: &str, n| {
            Norm::load(
                w,
                &format!("{base}.{name}"),
                n,
                c.rms_norm_eps,
                if gemma {
                    NormKind::Gemma
                } else if kind == Architecture::Mimi {
                    NormKind::Layer
                } else {
                    NormKind::Rms
                },
                d,
            )
        };
        let linear = |name: &str, input, output, bias| {
            Linear::load(w, &format!("{base}.{name}"), input, output, bias, p, d)
        };
        let layer_type = if gemma {
            Some(c.layer_types[index].as_str())
        } else {
            None
        };
        let window = if gemma {
            if layer_type == Some("sliding_attention") {
                c.sliding_window
            } else {
                None
            }
        } else if kind == Architecture::Mimi {
            // Match the bundled Transformers 4.57.3 Mimi encoder: its eager /
            // SDPA forward uses create_causal_mask, not a sliding-window mask.
            None
        } else {
            c.sliding_window
        };
        Ok(Self {
            norm1: norm(
                if gemma {
                    "pre_self_attn_layernorm"
                } else {
                    "input_layernorm"
                },
                width,
            )?,
            norm2: norm(
                if gemma {
                    "pre_feedforward_layernorm"
                } else {
                    "post_attention_layernorm"
                },
                width,
            )?,
            post1: if gemma {
                Some(norm("post_self_attn_layernorm", width)?)
            } else {
                None
            },
            post2: if gemma {
                Some(norm("post_feedforward_layernorm", width)?)
            } else {
                None
            },
            q: linear(
                "self_attn.q_proj",
                width,
                c.num_attention_heads * c.head_dim,
                c.attention_bias,
            )?,
            k: linear(
                "self_attn.k_proj",
                width,
                c.num_key_value_heads * c.head_dim,
                c.attention_bias,
            )?,
            v: linear(
                "self_attn.v_proj",
                width,
                c.num_key_value_heads * c.head_dim,
                c.attention_bias,
            )?,
            out: linear(
                "self_attn.o_proj",
                c.num_attention_heads * c.head_dim,
                width,
                c.attention_bias,
            )?,
            q_norm: if gemma || kind == Architecture::Qwen {
                Some(norm("self_attn.q_norm", c.head_dim)?)
            } else {
                None
            },
            k_norm: if gemma || kind == Architecture::Qwen {
                Some(norm("self_attn.k_norm", c.head_dim)?)
            } else {
                None
            },
            gate: linear(
                if kind == Architecture::Mimi {
                    "mlp.fc1"
                } else {
                    "mlp.gate_proj"
                },
                width,
                c.intermediate_size,
                false,
            )?,
            up: if kind == Architecture::Mimi {
                None
            } else {
                Some(linear("mlp.up_proj", width, c.intermediate_size, false)?)
            },
            down: linear(
                if kind == Architecture::Mimi {
                    "mlp.fc2"
                } else {
                    "mlp.down_proj"
                },
                c.intermediate_size,
                width,
                false,
            )?,
            scale1: if matches!(kind, Architecture::Codec | Architecture::Mimi) {
                Some(w.tensor(
                    &format!("{base}.self_attn_layer_scale.scale"),
                    [width],
                    p,
                    d,
                )?)
            } else {
                None
            },
            scale2: if matches!(kind, Architecture::Codec | Architecture::Mimi) {
                Some(w.tensor(&format!("{base}.mlp_layer_scale.scale"), [width], p, d)?)
            } else {
                None
            },
            rope_kind: usize::from(layer_type == Some("full_attention")),
            window,
            config: c.clone(),
            architecture: kind,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 3>,
        cache: &mut Option<KvCache<B>>,
        paged: Option<&mut DecodeCache<B>>,
        tables: &(Tensor<B, 4>, Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        let c = &self.config;
        let [b, n, _] = x.dims();
        let precision = dtype(self.q.weight());
        // The transformer stores Gemma residuals at 1/16 scale. Compensate
        // normalization epsilon and scale post-norm weights before multiplication
        // so Direct ANE's FP16 intermediates never need the unscaled large value.
        // FP32 public arithmetic retains the existing Metal rounding boundary.
        let residual_scale = if self.architecture == Architecture::Gemma {
            1.0 / 16.0
        } else {
            1.0
        };
        let x = if self.architecture == Architecture::Gemma {
            x.cast(FloatDType::F32)
        } else {
            x
        };
        let y = self
            .norm1
            .forward_scaled(x.clone(), residual_scale, 1.0)
            .cast(precision);
        let mut q = self
            .q
            .forward(y.clone())
            .reshape([b, n, c.num_attention_heads, c.head_dim]);
        let mut k = self
            .k
            .forward(y.clone())
            .reshape([b, n, c.num_key_value_heads, c.head_dim]);
        let v = self
            .v
            .forward(y)
            .reshape([b, n, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        if let Some(norm) = &self.q_norm {
            q = norm.forward(q);
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }
        let q = rope(q.swap_dims(1, 2), tables);
        let k = rope(k.swap_dims(1, 2), tables);
        let y = if let Some(history) = paged {
            assert!(self.window.is_none() && self.architecture != Architecture::Gemma);
            history.forward(
                q,
                k,
                v,
                AttentionOptions {
                    causal: true,
                    offset: history.len(),
                    ..Default::default()
                },
            )
        } else {
            let previous = cache.as_ref().map_or(0, KvCache::len);
            let mut history = cache.take().unwrap_or_default();
            let (k, v) = history.append(k, v, previous);
            if self.architecture != Architecture::Gemma {
                let len = history.len();
                let keep = self
                    .window
                    .map_or(len, |w| len.min(w.saturating_sub(1).max(1)));
                history.retain_last(keep);
                *cache = Some(history);
            }
            attention(
                q,
                k,
                v,
                previous,
                self.architecture != Architecture::Gemma,
                self.window,
            )
        }
        .swap_dims(1, 2)
        .reshape([b, n, c.num_attention_heads * c.head_dim]);
        let mut y = self.out.forward(y);
        if let Some(norm) = &self.post1 {
            y = norm.forward_scaled(y.cast(FloatDType::F32), 1.0, residual_scale);
        }
        if let Some(scale) = &self.scale1 {
            y = y * scale.clone().unsqueeze::<3>();
        }
        let x = residual(x, y);
        let y = self
            .norm2
            .forward_scaled(x.clone(), residual_scale, 1.0)
            .cast(precision);
        let gate = self.gate.forward(y.clone());
        let gate = if self.architecture == Architecture::Gemma {
            gelu(gate, true)
        } else if self.architecture == Architecture::Mimi {
            gelu(gate, false)
        } else {
            silu(gate)
        };
        let mut y = self.down.forward(if let Some(up) = &self.up {
            gate * up.forward(y)
        } else {
            gate
        });
        if let Some(norm) = &self.post2 {
            y = norm.forward_scaled(y.cast(FloatDType::F32), 1.0, residual_scale);
        }
        if let Some(scale) = &self.scale2 {
            y = y * scale.clone().unsqueeze::<3>();
        }
        residual(x, y)
    }
}

pub struct Transformer<B: Backend> {
    layers: Vec<Layer<B>>,
    norm: Option<Norm<B>>,
    ropes: Vec<Rope>,
    tables: Vec<(Tensor<B, 4>, Tensor<B, 4>)>,
    residual_scale: f64,
}
impl<B: Backend> Transformer<B> {
    pub fn set_kernel_mode(&mut self, mode: crate::DepthKernelMode) {
        let fused = matches!(
            mode,
            crate::DepthKernelMode::Norm | crate::DepthKernelMode::GemvNorm
        );
        if let Some(norm) = &mut self.norm {
            norm.fused = fused;
        }
        for layer in &mut self.layers {
            layer.norm1.fused = fused;
            layer.norm2.fused = fused;
            for linear in [
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
                &mut layer.out,
                &mut layer.gate,
                &mut layer.down,
            ] {
                linear.mode = mode;
            }
            if let Some(up) = &mut layer.up {
                up.mode = mode;
            }
        }
    }
    pub fn load(
        w: &impl WeightSource<B>,
        base: &str,
        c: &TransformerConfig,
        kind: Architecture,
        p: Precision,
        d: &B::Device,
        progress: &mut impl FnMut(usize),
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            layers.push(Layer::load(
                w,
                &format!("{base}.layers.{i}"),
                c,
                kind,
                i,
                p,
                d,
            )?);
            progress(i + 1);
        }
        let ropes = if kind == Architecture::Gemma {
            vec![
                Rope::new(c, Some("sliding_attention")),
                Rope::new(c, Some("full_attention")),
            ]
        } else {
            vec![Rope::new(c, None)]
        };
        let capacity = if kind == Architecture::Gemma {
            2048
        } else {
            c.max_position_embeddings.max(33)
        };
        let tables = ropes
            .iter()
            .map(|rope| rope.tables(0, capacity, p.dtype(), d))
            .collect();
        Ok(Self {
            layers,
            ropes,
            tables,
            residual_scale: if kind == Architecture::Gemma {
                1.0 / 16.0
            } else {
                1.0
            },
            norm: if kind == Architecture::Mimi {
                None
            } else {
                Some(Norm::load(
                    w,
                    &format!("{base}.norm"),
                    c.hidden_size,
                    c.rms_norm_eps,
                    if kind == Architecture::Gemma {
                        NormKind::Gemma
                    } else {
                        NormKind::Rms
                    },
                    d,
                )?)
            },
        })
    }
    pub fn cache(&self) -> TransformerCache<B> {
        TransformerCache::new(self.layers.len())
    }
    pub fn cache_with_paging(&self, enabled: bool) -> TransformerCache<B> {
        let mut cache = self.cache();
        if enabled {
            cache.paged_layers = Some(
                (0..self.layers.len())
                    .map(|_| DecodeCache::new(true))
                    .collect(),
            );
        }
        cache
    }
    pub fn forward(&self, x: Tensor<B, 3>, cache: &mut TransformerCache<B>) -> Tensor<B, 3> {
        self.forward_inner(x, cache, None, None).unwrap()
    }
    pub(crate) fn trace_layers(&self, x: Tensor<B, 3>) -> Vec<Tensor<B, 3>> {
        let mut snapshots = vec![x.clone()];
        let y = self
            .forward_inner(x, &mut self.cache(), None, Some(&mut snapshots))
            .unwrap();
        snapshots.push(y);
        snapshots
    }
    pub fn forward_cancellable(
        &self,
        x: Tensor<B, 3>,
        cache: &mut TransformerCache<B>,
        cancel: &AtomicBool,
    ) -> Option<Tensor<B, 3>> {
        self.forward_inner(x, cache, Some(cancel), None)
    }
    fn forward_inner(
        &self,
        mut x: Tensor<B, 3>,
        cache: &mut TransformerCache<B>,
        cancel: Option<&AtomicBool>,
        mut snapshots: Option<&mut Vec<Tensor<B, 3>>>,
    ) -> Option<Tensor<B, 3>> {
        if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return None;
        }
        let _routing = crate::routing::metal_phase::<B>(&x.device());
        let precision = dtype(&x);
        if self.residual_scale != 1.0 {
            x = x.cast(FloatDType::F32) * self.residual_scale;
        }
        let count = x.dims()[1];
        // Share rotary tables across all layers and reuse the resident positions.
        let tables: Vec<_> = self
            .tables
            .iter()
            .zip(&self.ropes)
            .map(|((cos, sin), rope)| {
                let [_, _, capacity, width] = cos.dims();
                let start = cache.position;
                let end = start + count;
                if end <= capacity {
                    (
                        cos.clone().slice([0..1, 0..1, start..end, 0..width]),
                        sin.clone().slice([0..1, 0..1, start..end, 0..width]),
                    )
                } else {
                    rope.tables(start, count, dtype(&x), &x.device())
                }
            })
            .collect();
        for (i, (layer, kv)) in self.layers.iter().zip(&mut cache.layers).enumerate() {
            if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return None;
            }
            x = layer.forward(
                x,
                kv,
                cache.paged_layers.as_mut().map(|layers| &mut layers[i]),
                &tables[layer.rope_kind],
            );
            if let Some(snapshots) = &mut snapshots {
                snapshots.push(x.clone());
            }
        }
        cache.position += count;
        Some(if let Some(norm) = &self.norm {
            norm.forward_scaled(x, self.residual_scale, 1.0)
                .cast(precision)
        } else {
            x
        })
    }
}
