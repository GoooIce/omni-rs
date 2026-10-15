//! Qwen3.5-style text decoder: zero-centered RMSNorm, full attention with a
//! sigmoid output gate and partial interleaved M-RoPE, and SwiGLU MLP.
use crate::config::TextConfig;
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::{
    Precision, tensor as shared,
    tensor::{AttentionOptions, KvCache, NormReduction},
    weights::WeightSource,
};

type Linear<B> = shared::Linear<B>;
type Norm<B> = shared::Norm<B>;

/// Zero-centered RMSNorm (`(1 + w)` scaling), the Qwen3.5 text convention.
fn rms_norm<B: Backend>(
    w: &impl WeightSource<B>,
    name: &str,
    width: usize,
    eps: f64,
    d: &B::Device,
) -> omni_r2t2::Result<Norm<B>> {
    shared::Norm::load(
        w,
        name,
        width,
        eps,
        shared::NormKind::Gemma,
        NormReduction::Divide,
        d,
    )
}

/// Inverse frequencies for the partial rotary block: `theta^(-2j/rotary_dim)`.
fn inverse_frequencies(config: &TextConfig) -> Vec<f64> {
    let rope = &config.rope_parameters;
    let rotary_dim = (config.head_dim as f64 * rope.partial_rotary_factor) as usize & !1;
    (0..rotary_dim / 2)
        .map(|j| rope.rope_theta.powf(-(2.0 * j as f64 / rotary_dim as f64)))
        .collect()
}

/// Which M-RoPE channel a frequency pair reads: pairs interleave T/H/W with
/// period 3 until a section is exhausted, then fall back to T.
fn channel_of(j: usize, sections: &[usize]) -> usize {
    match j % 3 {
        1 => {
            if j < sections[1] * 3 {
                1
            } else {
                0
            }
        }
        2 if j < sections[2] * 3 => 2,
        _ => 0,
    }
}

/// Cosine/sine tables `(1, seq, rotary_dim)` in split-half pair layout for the
/// given `(T, H, W)` position triples.
pub fn mrope_tables<B: Backend>(
    config: &TextConfig,
    positions: &[[i64; 3]],
    output: FloatDType,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let frequencies = inverse_frequencies(config);
    let width = frequencies.len() * 2;
    let mut cosine = Vec::with_capacity(positions.len() * width);
    let mut sine = Vec::with_capacity(positions.len() * width);
    let sections = &config.rope_parameters.mrope_section;
    for position in positions {
        for (j, &frequency) in frequencies.iter().enumerate() {
            let angle = position[channel_of(j, sections)] as f64 * frequency;
            cosine.push(angle.cos() as f32);
            sine.push(angle.sin() as f32);
        }
    }
    // Split-half layout: each row's second half repeats its first.
    let mut cos_rows = Vec::with_capacity(positions.len() * width);
    let mut sin_rows = Vec::with_capacity(positions.len() * width);
    for chunk in cosine.chunks(width / 2) {
        cos_rows.extend_from_slice(chunk);
        cos_rows.extend_from_slice(chunk);
    }
    for chunk in sine.chunks(width / 2) {
        sin_rows.extend_from_slice(chunk);
        sin_rows.extend_from_slice(chunk);
    }
    let cosine = cos_rows;
    let sine = sin_rows;
    let shape = [1, positions.len(), width];
    let data = burn::tensor::TensorData::new(cosine, shape);
    let cos = Tensor::<B, 3>::from_data(data, device).cast(output);
    let data = burn::tensor::TensorData::new(sine, shape);
    let sin = Tensor::<B, 3>::from_data(data, device).cast(output);
    (cos, sin)
}

pub struct FullAttention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
}

pub struct Mlp<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}

pub struct FullLayer<B: Backend> {
    input_norm: Norm<B>,
    attn: FullAttention<B>,
    post_norm: Norm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> FullAttention<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        // The projection emits query and gate interleaved: 2 × heads × head_dim.
        let q_width = if config.attn_output_gate {
            heads * head_dim * 2
        } else {
            heads * head_dim
        };
        let rotary_dim =
            (head_dim as f64 * config.rope_parameters.partial_rotary_factor) as usize & !1;
        Ok(Self {
            q: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.q_proj.weight"),
                    [q_width, config.hidden_size],
                    p,
                    d,
                )?,
                None,
            ),
            k: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.k_proj.weight"),
                    [kv_heads * head_dim, config.hidden_size],
                    p,
                    d,
                )?,
                None,
            ),
            v: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.v_proj.weight"),
                    [kv_heads * head_dim, config.hidden_size],
                    p,
                    d,
                )?,
                None,
            ),
            o: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.o_proj.weight"),
                    [config.hidden_size, heads * head_dim],
                    p,
                    d,
                )?,
                None,
            ),
            q_norm: rms_norm(
                w,
                &format!("{name}.q_norm"),
                head_dim,
                config.rms_norm_eps,
                d,
            )?,
            k_norm: rms_norm(
                w,
                &format!("{name}.k_norm"),
                head_dim,
                config.rms_norm_eps,
                d,
            )?,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
        })
    }

    /// Causal self-attention with M-RoPE applied to the first `rotary_dim`
    /// dims of each head. `cos`/`sin` are `(1, seq, rotary_dim)`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 3>,
        sin: &Tensor<B, 3>,
        cache: Option<&mut KvCache<B>>,
    ) -> Tensor<B, 3> {
        let [_, seq, _width] = x.dims();
        let dtype = shared::dtype(&x);
        let projected = self.q.forward(x.clone());
        // Query and gate arrive head-interleaved: (seq, heads, 2, head_dim).
        debug_assert_eq!(2 * self.heads * self.head_dim, projected.dims()[2]);
        let view = projected.reshape([seq, self.heads, 2, self.head_dim]);
        let query = view
            .clone()
            .slice([0..seq, 0..self.heads, 0..1, 0..self.head_dim])
            .reshape([1, seq, self.heads * self.head_dim]);
        let gate = view
            .slice([0..seq, 0..self.heads, 1..2, 0..self.head_dim])
            .reshape([1, seq, self.heads * self.head_dim]);
        let shape = |t: Tensor<B, 3>, heads: usize| {
            t.reshape([1, seq, heads, self.head_dim])
                .permute([0, 2, 1, 3])
        };
        let mut q = shape(query, self.heads);
        let mut k = shape(self.k.forward(x.clone()), self.kv_heads);
        let v = shape(self.v.forward(x), self.kv_heads);
        // Norm before rotation, per head.
        q = self.q_norm.forward(q, dtype);
        k = self.k_norm.forward(k, dtype);
        // Partial rotation: rotate the leading rotary_dim dims, pass the rest.
        let rotate = |t: Tensor<B, 4>| -> Tensor<B, 4> {
            if self.rotary_dim == self.head_dim {
                let cos4 = cos.clone().unsqueeze::<4>();
                let sin4 = sin.clone().unsqueeze::<4>();
                shared::rope(t, &cos4, &sin4, shared::RotaryLayout::SplitHalf)
            } else {
                let pass =
                    t.clone()
                        .slice([0..1, 0..self.heads, 0..seq, self.rotary_dim..self.head_dim]);
                let rot = t.slice([0..1, 0..self.heads, 0..seq, 0..self.rotary_dim]);
                let cos4 = cos.clone().unsqueeze::<4>();
                let sin4 = sin.clone().unsqueeze::<4>();
                let rotated = shared::rope(rot, &cos4, &sin4, shared::RotaryLayout::SplitHalf);
                Tensor::cat(vec![rotated, pass], 3)
            }
        };
        let q = rotate(q);
        let k = rotate(k);
        let offset = cache.as_ref().map_or(0, |c| c.len());
        let (k, v) = match cache {
            Some(cache) => cache.append(k, v, offset),
            None => (k, v),
        };
        let attended = shared::attention(
            q,
            k,
            v,
            AttentionOptions {
                causal: true,
                offset,
                ..Default::default()
            },
        );
        let attended = attended
            .permute([0, 2, 1, 3])
            .reshape([1, seq, self.heads * self.head_dim]);
        let gate = burn::tensor::activation::sigmoid(gate);
        let attended = attended * gate;
        self.o.forward(attended)
    }
}

impl<B: Backend> Mlp<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        Ok(Self {
            gate: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.gate_proj.weight"),
                    [config.intermediate_size, config.hidden_size],
                    p,
                    d,
                )?,
                None,
            ),
            up: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.up_proj.weight"),
                    [config.intermediate_size, config.hidden_size],
                    p,
                    d,
                )?,
                None,
            ),
            down: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.down_proj.weight"),
                    [config.hidden_size, config.intermediate_size],
                    p,
                    d,
                )?,
                None,
            ),
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.gate.forward(x.clone()));
        self.down.forward(gate * self.up.forward(x))
    }
}

impl<B: Backend> FullLayer<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        Ok(Self {
            input_norm: rms_norm(
                w,
                &format!("{name}.input_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
                d,
            )?,
            attn: FullAttention::load(w, &format!("{name}.self_attn"), config, p, d)?,
            post_norm: rms_norm(
                w,
                &format!("{name}.post_attention_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
                d,
            )?,
            mlp: Mlp::load(w, &format!("{name}.mlp"), config, p, d)?,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 3>,
        sin: &Tensor<B, 3>,
        cache: Option<&mut KvCache<B>>,
    ) -> Tensor<B, 3> {
        let dtype = shared::dtype(&x);
        let normed = self.input_norm.forward(x.clone(), dtype);
        let attended = self.attn.forward(normed, cos, sin, cache);
        let x = shared::residual(x, attended, dtype);
        let mlp = self.mlp.forward(self.post_norm.forward(x.clone(), dtype));
        shared::residual(x, mlp, dtype)
    }
}

/// Recurrent + causal-conv state carried across decode steps.
#[derive(Clone)]
pub struct LinearState<B: Backend> {
    /// `(1, conv_dim, kernel - 1)` mixed QKV tail before the conv.
    pub conv: Tensor<B, 3>,
    /// `(1, heads, head_k, head_v)` FP32 SSM state.
    pub recurrent: Tensor<B, 4>,
}

pub struct LinearAttention<B: Backend> {
    qkv: Linear<B>,
    z: Linear<B>,
    b: Linear<B>,
    a: Linear<B>,
    out: Linear<B>,
    conv_weight: Tensor<B, 3>, // (conv_dim, 1, kernel) FP32
    a_log: Tensor<B, 1>,       // (heads,) FP32
    dt_bias: Tensor<B, 1>,     // (heads,) FP32
    norm_weight: Tensor<B, 1>, // (head_v,) FP32, plain scaling (no +1)
    heads: usize,
    head_k: usize,
    head_v: usize,
    kernel: usize,
    key_dim: usize,
    value_dim: usize,
    eps: f64,
}

impl<B: Backend> LinearAttention<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        let heads = config.linear_num_key_heads;
        let head_k = config.linear_key_head_dim;
        let head_v = config.linear_value_head_dim;
        let key_dim = heads * head_k;
        let value_dim = config.linear_num_value_heads * head_v;
        let kernel = config.linear_conv_kernel_dim.max(2);
        let conv_dim = 2 * key_dim + value_dim;
        let load = |suffix: &str, out: usize| -> omni_r2t2::Result<Linear<B>> {
            Ok(shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.{suffix}.weight"),
                    [out, config.hidden_size],
                    p,
                    d,
                )?,
                None,
            ))
        };
        Ok(Self {
            qkv: load("in_proj_qkv", conv_dim)?,
            z: load("in_proj_z", value_dim)?,
            b: load("in_proj_b", heads)?,
            a: load("in_proj_a", heads)?,
            out: shared::Linear::from_parts(
                w.tensor(
                    &format!("{name}.out_proj.weight"),
                    [config.hidden_size, value_dim],
                    p,
                    d,
                )?,
                None,
            ),
            conv_weight: w.tensor(
                &format!("{name}.conv1d.weight"),
                [conv_dim, 1, kernel],
                Precision::F32,
                d,
            )?,
            a_log: w.tensor(&format!("{name}.A_log"), [heads], Precision::F32, d)?,
            dt_bias: w.tensor(&format!("{name}.dt_bias"), [heads], Precision::F32, d)?,
            norm_weight: w.tensor(&format!("{name}.norm.weight"), [head_v], Precision::F32, d)?,
            heads,
            head_k,
            head_v,
            kernel,
            key_dim,
            value_dim,
            eps: config.rms_norm_eps,
        })
    }

    pub fn new_state(&self, device: &B::Device) -> LinearState<B> {
        LinearState {
            conv: Tensor::zeros(
                [1, 2 * self.key_dim + self.value_dim, self.kernel - 1],
                device,
            ),
            recurrent: Tensor::zeros(
                [1, self.heads, self.head_k, self.head_v],
                (device, burn::tensor::DType::F32),
            ),
        }
    }

    /// Causal depthwise conv with SiLU over `(1, T, conv)` input. Returns the
    /// activated stream and the new conv tail state.
    fn causal_conv(
        &self,
        mixed: Tensor<B, 3>,
        tail: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [_, seq, conv_dim] = mixed.dims();
        let dtype = shared::dtype(&mixed);
        let device = mixed.device();
        let stream = mixed.permute([0, 2, 1]); // (1, conv, T)
        let tail = tail.unwrap_or_else(|| Tensor::zeros([1, conv_dim, self.kernel - 1], &device));
        let padded = Tensor::cat(vec![tail, stream], 2);
        let [_, _, width] = padded.dims();
        let new_tail = padded
            .clone()
            .slice([0..1, 0..conv_dim, width - (self.kernel - 1)..width]);
        let mut output = Tensor::<B, 3>::zeros([1, conv_dim, seq], &device);
        for tap in 0..self.kernel {
            let weight = self
                .conv_weight
                .clone()
                .slice([0..conv_dim, 0..1, tap..tap + 1])
                .reshape([conv_dim, 1]);
            let window = padded.clone().slice([0..1, 0..conv_dim, tap..tap + seq]);
            output = output + window * weight.unsqueeze();
        }
        let activated = burn::tensor::activation::silu(output.cast(FloatDType::F32));
        (activated.permute([0, 2, 1]).cast(dtype), new_tail)
    }

    /// The FP32 gated delta rule recurrence; `q`,`k` are already L2-normalized
    /// and scaled, `g`/`beta` per head. State `(1, H, K, V)` evolves in place.
    fn recurrence(
        &self,
        q: Tensor<B, 4>,    // (1, T, H, K) FP32
        k: Tensor<B, 4>,    // (1, T, H, K) FP32
        v: Tensor<B, 4>,    // (1, T, H, V) FP32
        g: Tensor<B, 3>,    // (1, T, H) FP32
        beta: Tensor<B, 3>, // (1, T, H) FP32
        mut state: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let seq = q.dims()[1];
        let mut outputs = Vec::with_capacity(seq);
        let mut state = state.squeeze_dim::<3>(0); // (H, K, V) FP32
        for t in 0..seq {
            let pick = |tensor: &Tensor<B, 4>| -> Tensor<B, 2> {
                tensor
                    .clone()
                    .slice([0..1, t..t + 1])
                    .reshape([self.heads, tensor.dims()[3]])
            };
            let q_t = pick(&q); // (H, K)
            let k_t = pick(&k);
            let v_t = pick(&v); // (H, V)
            let decay = g
                .clone()
                .slice([0..1, t..t + 1])
                .reshape([self.heads, 1, 1])
                .exp();
            let beta_t = beta
                .clone()
                .slice([0..1, t..t + 1])
                .reshape([self.heads, 1]);
            state = state * decay;
            let kv = (state.clone() * k_t.clone().unsqueeze_dim(2))
                .sum_dim(1)
                .reshape([self.heads, self.head_v]);
            let delta = (v_t.clone() - kv) * beta_t;
            state = state + k_t.unsqueeze_dim(2) * delta.unsqueeze_dim(1);
            let out = (state.clone() * q_t.unsqueeze_dim(2))
                .sum_dim(1)
                .reshape([self.heads, self.head_v]);
            outputs.push(out.reshape([1, 1, self.heads, self.head_v]));
        }
        (Tensor::cat(outputs, 1), state.unsqueeze_dim(0)) // (1, T, H, V)
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: Option<&mut LinearState<B>>) -> Tensor<B, 3> {
        let [_, seq, _] = x.dims();
        let dtype = shared::dtype(&x);
        let mixed = self.qkv.forward(x.clone());
        let (conv, new_tail) = self.causal_conv(mixed, state.as_ref().map(|s| s.conv.clone()));

        let fp32 = |t: Tensor<B, 3>| t.cast(FloatDType::F32);
        let q = fp32(conv.clone().slice([0..1, 0..seq, 0..self.key_dim])).reshape([
            1,
            seq,
            self.heads,
            self.head_k,
        ]);
        let k = fp32(
            conv.clone()
                .slice([0..1, 0..seq, self.key_dim..2 * self.key_dim]),
        )
        .reshape([1, seq, self.heads, self.head_k]);
        let value_end = 2 * self.key_dim + self.value_dim;
        let v = fp32(
            conv.clone()
                .slice([0..1, 0..seq, 2 * self.key_dim..value_end]),
        )
        .reshape([1, seq, self.heads, self.head_v]);
        let l2 = |t: Tensor<B, 4>| -> Tensor<B, 4> {
            // Official FLA l2norm: x * rsqrt(sum(x^2) + eps).
            let scale = (t.clone().square().sum_dim(3) + 1e-6).sqrt().recip();
            t * scale
        };
        let q = l2(q) * (self.head_k as f32).powf(-0.5);
        let k = l2(k);

        let beta = burn::tensor::activation::sigmoid(fp32(self.b.forward(x.clone())));
        let z = self.z.forward(x.clone()).cast(FloatDType::F32);
        let a = fp32(self.a.forward(x.clone()));
        // softplus(a + dt_bias) with torch's threshold at 20, then the
        // negative decay.
        let a = a + self.dt_bias.clone().reshape([1, 1, self.heads]);
        let softplus = a.clone().exp().log1p();
        let softplus = softplus.mask_where(a.clone().greater_elem(20.0), a);
        let decay = -self.a_log.clone().reshape([1, 1, self.heads]).exp() * softplus;

        let initial = state
            .as_ref()
            .map(|s| s.recurrent.clone())
            .unwrap_or_else(|| {
                Tensor::zeros(
                    [1, self.heads, self.head_k, self.head_v],
                    (&x.device(), burn::tensor::DType::F32),
                )
            });
        let (core, recurrent) = self.recurrence(q, k, v, decay, beta, initial);
        if let Some(state) = state {
            state.conv = new_tail;
            state.recurrent = recurrent;
        }

        // RMSNormGated: FP32 norm, plain weight, SiLU gate.
        let flat = core.reshape([seq * self.heads, self.head_v]);
        let variance = (flat.clone().square().mean_dim(1) + self.eps).sqrt();
        let normed = flat / variance;
        let z = z.reshape([seq * self.heads, self.head_v]);
        let gated_norm =
            normed * self.norm_weight.clone().unsqueeze_dim(0) * burn::tensor::activation::silu(z);
        let out = gated_norm.reshape([1, seq, self.value_dim]).cast(dtype);
        self.out.forward(out)
    }
}

pub struct LinearLayer<B: Backend> {
    input_norm: Norm<B>,
    attn: LinearAttention<B>,
    post_norm: Norm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> LinearLayer<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        Ok(Self {
            input_norm: rms_norm(
                w,
                &format!("{name}.input_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
                d,
            )?,
            attn: LinearAttention::load(w, &format!("{name}.linear_attn"), config, p, d)?,
            post_norm: rms_norm(
                w,
                &format!("{name}.post_attention_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
                d,
            )?,
            mlp: Mlp::load(w, &format!("{name}.mlp"), config, p, d)?,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: Option<&mut LinearState<B>>) -> Tensor<B, 3> {
        let dtype = shared::dtype(&x);
        let normed = self.input_norm.forward(x.clone(), dtype);
        let attended = self.attn.forward(normed, state);
        let x = shared::residual(x, attended, dtype);
        let mlp = self.mlp.forward(self.post_norm.forward(x.clone(), dtype));
        shared::residual(x, mlp, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeParameters;
    use burn::backend::NdArray;
    use safetensors::SafeTensors;

    fn fixture() -> Vec<u8> {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/full-attention-layer.safetensors"),
        )
        .unwrap()
    }

    fn synthetic_config() -> TextConfig {
        TextConfig {
            hidden_size: 96,
            intermediate_size: 224,
            num_hidden_layers: 8,
            layer_types: vec!["full_attention".into()],
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 64,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            attn_output_gate: true,
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            max_position_embeddings: 4096,
            rope_parameters: RopeParameters {
                mrope_section: vec![3, 3, 2],
                rope_theta: 10000.0,
                partial_rotary_factor: 0.25,
            },
            mamba_ssm_dtype: None,
        }
    }

    // as_chunks would need a newer MSRV than the workspace
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn stage(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
        tensors
            .tensor(name)
            .unwrap()
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn compare(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut squared = 0f64;
        let mut energy = 0f64;
        let mut dot = 0f64;
        let mut mine = 0f64;
        for (&a, &b) in actual.iter().zip(expected) {
            squared += (a as f64 - b as f64).powi(2);
            energy += (b as f64).powi(2);
            dot += a as f64 * b as f64;
            mine += (a as f64).powi(2);
        }
        let rmse = squared.sqrt() / energy.sqrt().max(1e-12);
        let cosine = dot / (mine.sqrt() * energy.sqrt()).max(1e-12);
        assert!(
            rmse < 1e-4 && cosine > 0.99999,
            "{label}: RMSE {rmse:.8}, cosine {cosine:.9}"
        );
    }

    #[test]
    fn full_attention_layer_matches_official_synthetic_dump() {
        type B = NdArray;
        let device: burn::backend::ndarray::NdArrayDevice = Default::default();
        let bytes = fixture();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes.clone()]).unwrap();
        let config = synthetic_config();
        let layer: FullLayer<B> =
            FullLayer::load(&weights, "layers.0", &config, Precision::F32, &device).unwrap();

        // Canvas positions from the committed fixture (same as the dump run).
        let cases: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/canvas-positions.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let table = &cases["text-image-text"]["positions"];
        let positions: Vec<[i64; 3]> = (0..table[0].as_array().unwrap().len())
            .map(|i| {
                [
                    table[0][i].as_i64().unwrap(),
                    table[1][i].as_i64().unwrap(),
                    table[2][i].as_i64().unwrap(),
                ]
            })
            .collect();

        let seq = positions.len();
        let hidden: Vec<f32> = stage(&tensors, "stage_hidden_in");
        let x = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(hidden, [1, seq, config.hidden_size]),
            &device,
        );
        let (cos, sin) = mrope_tables::<B>(&config, &positions, FloatDType::F32, &device);
        let cos_values = cos
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let sin_values = sin
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        compare("cos", &cos_values, &stage(&tensors, "stage_cos"));
        compare("sin", &sin_values, &stage(&tensors, "stage_sin"));

        // Attention-only path via the same projections the layer uses.
        let normed = layer.input_norm.forward(x.clone(), FloatDType::F32);
        let q = layer.attn.q.forward(normed.clone());
        let q_values = q
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        compare("q_proj", &q_values, &stage(&tensors, "stage_q_proj"));

        let attended = layer.attn.forward(normed, &cos, &sin, None);
        let attn_values = attended
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        compare(
            "attention_out",
            &attn_values,
            &stage(&tensors, "stage_attention_out"),
        );

        let mlp_input = layer.post_norm.forward(
            shared::residual(x.clone(), attended, FloatDType::F32),
            FloatDType::F32,
        );
        let mlp = layer.mlp.forward(mlp_input);
        let mlp_values = mlp
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        compare("mlp_out", &mlp_values, &stage(&tensors, "stage_mlp_out"));

        let output = layer.forward(x, &cos, &sin, None);
        let out_values = output
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        compare(
            "layer_out",
            &out_values,
            &stage(&tensors, "stage_layer_out"),
        );
    }
}

#[cfg(test)]
mod rope_debug {
    use super::*;
    use crate::config::RopeParameters;
    use burn::backend::NdArray;

    #[test]
    fn print_rope_probe() {
        let config = TextConfig {
            hidden_size: 96,
            intermediate_size: 224,
            num_hidden_layers: 8,
            layer_types: vec!["full_attention".into()],
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 64,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            attn_output_gate: true,
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            max_position_embeddings: 4096,
            rope_parameters: RopeParameters {
                mrope_section: vec![3, 3, 2],
                rope_theta: 10000.0,
                partial_rotary_factor: 0.25,
            },
            mamba_ssm_dtype: None,
        };
        let device: burn::backend::ndarray::NdArrayDevice = Default::default();
        let positions = vec![[0i64, 0, 0], [1, 1, 1], [2, 1, 3]];
        let (cos, _) = mrope_tables::<NdArray>(&config, &positions, FloatDType::F32, &device);
        let values = cos
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        println!("mine cos rows:");
        for t in 0..3 {
            println!("  {:?}", &values[t * 16..t * 16 + 16]);
        }
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/full-attention-layer.safetensors"),
        )
        .unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let view = tensors.tensor("stage_cos").unwrap();
        let official: Vec<f32> = view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        println!("official cos rows (positions t=1: (1,1,1) etc. are at fixture indices):");
        for t in [0usize, 1, 4] {
            println!("  {:?}", &official[t * 16..t * 16 + 16]);
        }
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;
    use crate::config::RopeParameters;
    use burn::backend::NdArray;
    use safetensors::SafeTensors;

    fn config() -> TextConfig {
        TextConfig {
            hidden_size: 96,
            intermediate_size: 224,
            num_hidden_layers: 8,
            layer_types: vec!["linear_attention".into()],
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 64,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            attn_output_gate: true,
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            max_position_embeddings: 4096,
            rope_parameters: RopeParameters {
                mrope_section: vec![3, 3, 2],
                rope_theta: 10000.0,
                partial_rotary_factor: 0.25,
            },
            mamba_ssm_dtype: None,
        }
    }

    // as_chunks would need a newer MSRV than the workspace
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn stage(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
        tensors
            .tensor(name)
            .unwrap()
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn compare(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut squared = 0f64;
        let mut energy = 0f64;
        for (&a, &b) in actual.iter().zip(expected) {
            squared += (a as f64 - b as f64).powi(2);
            energy += (b as f64).powi(2);
        }
        let rmse = squared.sqrt() / energy.sqrt().max(1e-12);
        assert!(rmse < 2e-4, "{label}: RMSE {rmse:.8}");
    }

    fn values3(tensor: Tensor<NdArray, 3>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn gated_delta_layer_matches_official_prefill_and_decode() {
        type B = NdArray;
        let device: burn::backend::ndarray::NdArrayDevice = Default::default();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/linear-attention-layer.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes.clone()]).unwrap();
        let config = config();
        let layer: LinearLayer<B> =
            LinearLayer::load(&weights, "layers.0", &config, Precision::F32, &device).unwrap();

        let seq = 40;
        let hidden = stage(&tensors, "stage_hidden_in");
        let x = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(hidden, [1, seq, config.hidden_size]),
            &device,
        );
        let mut state = layer.attn.new_state(&device);
        let output = layer.forward(x, Some(&mut state));
        compare(
            "prefill layer_out",
            &values3(output),
            &stage(&tensors, "stage_layer_out"),
        );
        compare(
            "final recurrent state",
            &values4(state.recurrent.clone()),
            &stage(&tensors, "stage_final_state"),
        );
        compare(
            "final conv state",
            &values3(state.conv.clone()),
            &stage(&tensors, "stage_conv_state"),
        );

        // Cached decode; the fixture captures the attention-block output at
        // this seam because the reference kernels expose the cached path there.
        let decode_inputs = stage(&tensors, "stage_decode_inputs");
        let decode_expected = stage(&tensors, "stage_decode_steps");
        let expected_steps: Vec<&[f32]> = decode_expected.chunks(config.hidden_size).collect();
        for step in 0..3 {
            let token = Tensor::<B, 3>::from_data(
                burn::tensor::TensorData::new(
                    decode_inputs[step * config.hidden_size..(step + 1) * config.hidden_size]
                        .to_vec(),
                    [1, 1, config.hidden_size],
                ),
                &device,
            );
            let normed = layer.input_norm.forward(token, FloatDType::F32);
            let output = layer.attn.forward(normed, Some(&mut state));
            compare(
                &format!("decode step {step}"),
                &values3(output),
                expected_steps[step],
            );
        }

        // Reset semantics: a fresh state must not repeat carried-state outputs.
        let token = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(
                decode_inputs[..config.hidden_size].to_vec(),
                [1, 1, config.hidden_size],
            ),
            &device,
        );
        let mut fresh = layer.attn.new_state(&device);
        let normed = layer.input_norm.forward(token, FloatDType::F32);
        let fresh_output = layer.attn.forward(normed, Some(&mut fresh));
        assert!(
            values3(fresh_output)
                .iter()
                .zip(expected_steps[0].iter())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "fresh state must differ from carried state"
        );
    }

    fn values4(tensor: Tensor<NdArray, 4>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }
}
