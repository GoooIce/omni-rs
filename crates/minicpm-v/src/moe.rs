//! MoE sparse block: softmax top-k router, grouped experts, and a
//! sigmoid-gated shared expert (Qwen3.5-Moe). Experts load either fused
//! (bf16 3-D tensors) or pack-quantized int4 per projection (35B checkpoint).
use crate::{config::TextConfig, quant};
#[cfg(test)]
use burn::tensor::TensorData;
use burn::tensor::{Tensor, activation, backend::Backend};
use omni_r2t2::{Precision, Result, tensor as shared, weights::WeightSource};
use std::cell::Cell;

thread_local! {
    static MATCH_DECODE: Cell<bool> = const { Cell::new(false) };
}

/// While this guard is alive, a multi-token MoE block keeps the one-token
/// decode experts and slot-order sum, and projects each expert once.
pub(crate) struct MatchDecodeMoe {
    previous: bool,
}

impl MatchDecodeMoe {
    pub(crate) fn enter() -> Self {
        let previous = MATCH_DECODE.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for MatchDecodeMoe {
    fn drop(&mut self) {
        MATCH_DECODE.with(|flag| flag.set(self.previous));
    }
}

type Linear<B> = shared::Linear<B>;

/// MoE shape parameters (35B: 256 experts, top-8, 512 intermediate).
#[derive(Debug, Clone)]
pub struct MoeGeometry {
    pub num_experts: usize,
    pub top_k: usize,
    pub expert_intermediate: usize,
    pub shared_intermediate: usize,
}

enum ExpertLinear<B: Backend> {
    Fused(Tensor<B, 2>),
    Packed(quant::PackedWeight<B>),
}

impl<B: Backend> ExpertLinear<B> {
    fn matmul_transpose(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let rows = x.dims()[0];
        if rows > 1 && shared::decode_row_gemv() {
            let width = x.dims()[1];
            let mut pieces = Vec::with_capacity(rows);
            for index in 0..rows {
                pieces
                    .push(self.matmul_transpose_row(x.clone().slice([index..index + 1, 0..width])));
            }
            return Tensor::cat(pieces, 0);
        }
        self.matmul_transpose_row(x)
    }

    fn matmul_transpose_row(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            ExpertLinear::Fused(weight) => x.matmul(weight.clone().transpose()),
            ExpertLinear::Packed(weight) => weight.matmul_transpose(x),
        }
    }
}

struct ExpertWeights<B: Backend> {
    gate: ExpertLinear<B>,
    up: ExpertLinear<B>,
    down: ExpertLinear<B>,
}

/// One uploaded `PackedParts` pair.
struct DenseExpertsUpload<B: Backend> {
    words: Tensor<B, 1, burn::tensor::Int>,
    scales: Tensor<B, 1>,
}

/// All experts of one layer concatenated into three raw packed payloads for
/// launch-light multi-token forwards (speculative verify). `gate`/`up` are
/// logical `[E * inter, hidden]`; `down` is logical `[hidden, E * inter]` with
/// each row interleaving the experts. Words and scales live in flat 1-D
/// tensors read linearly by the grouped kernels. The router row-major
/// `[experts, hidden]` weight, the shared expert's projections and its sigmoid
/// gate vector are f32 (lossless casts of the bf16 checkpoint values) so the
/// fused kernels read them without a per-forward cast.
pub(crate) struct DenseExperts<B: Backend> {
    pub(crate) gate_words: Tensor<B, 1, burn::tensor::Int>,
    pub(crate) gate_scales: Tensor<B, 1>,
    pub(crate) up_words: Tensor<B, 1, burn::tensor::Int>,
    pub(crate) up_scales: Tensor<B, 1>,
    pub(crate) down_words: Tensor<B, 1, burn::tensor::Int>,
    pub(crate) down_scales: Tensor<B, 1>,
    pub(crate) router: Tensor<B, 2>,
    pub(crate) shared_gate_w: Tensor<B, 2>,
    pub(crate) shared_up_w: Tensor<B, 2>,
    pub(crate) shared_down_w: Tensor<B, 2>,
    pub(crate) shared_gate_vec: Tensor<B, 2>,
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) expert_intermediate: usize,
    pub(crate) shared_intermediate: usize,
}

/// Blocks at or below this many tokens take the all-experts dense path: every
/// expert projection is one matmul, at most `2^5 x` the routed compute, and
/// the router never syncs back to the host.
const DENSE_MOE_MAX_TOKENS: usize = 32;

pub struct SparseMoeBlock<B: Backend> {
    router: Tensor<B, 2>, // (experts, hidden)
    experts: Vec<ExpertWeights<B>>,
    dense: Option<DenseExperts<B>>,
    shared_gate: Linear<B>,
    shared_up: Linear<B>,
    shared_down: Linear<B>,
    shared_expert_gate: Linear<B>,
    geometry: MoeGeometry,
    hidden: usize,
}

fn geometry(config: &TextConfig) -> MoeGeometry {
    MoeGeometry {
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
        expert_intermediate: config.moe_intermediate_size,
        shared_intermediate: config.shared_expert_intermediate_size,
    }
}

impl<B: Backend> SparseMoeBlock<B> {
    /// Fused bf16 layout: 3-D `experts.gate_up_proj` and `experts.down_proj`.
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let geo = geometry(config);
        let hidden = config.hidden_size;
        let gate_up: Tensor<B, 3> = w.tensor(
            &format!("{name}.experts.gate_up_proj"),
            [geo.num_experts, 2 * geo.expert_intermediate, hidden],
            p,
            d,
        )?;
        let down: Tensor<B, 3> = w.tensor(
            &format!("{name}.experts.down_proj"),
            [geo.num_experts, hidden, geo.expert_intermediate],
            p,
            d,
        )?;
        let mut experts = Vec::with_capacity(geo.num_experts);
        for expert in 0..geo.num_experts {
            let fused = gate_up.clone().narrow(0, expert, 1).squeeze_dim::<2>(0);
            let [width, _] = fused.dims();
            experts.push(ExpertWeights {
                gate: ExpertLinear::Fused(fused.clone().slice([0..width / 2, 0..hidden]).clone()),
                up: ExpertLinear::Fused(fused.slice([width / 2..width, 0..hidden]).clone()),
                down: ExpertLinear::Fused(down.clone().narrow(0, expert, 1).squeeze_dim::<2>(0)),
            });
        }
        Self::assemble_dense(w, name, geo, hidden, experts, p, d)
    }

    /// 35B int4 layout: per-expert `{name}.experts.{e}.{gate,up,down}_proj`
    /// with `.weight_packed` payloads and `.weight_scale` group scales.
    /// Residence is chosen by `ExpertResidence` at checkpoint load.
    pub(crate) fn load_quantized(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        mode: quant::LoadMode,
        d: &B::Device,
    ) -> Result<Self> {
        let geo = geometry(config);
        let hidden = config.hidden_size;
        let intermediate = geo.expert_intermediate;
        let load = |expert: usize, projection: &str, shape: [usize; 2]| {
            quant::load_packed(
                w,
                &format!("{name}.experts.{expert}.{projection}.weight_packed"),
                &format!("{name}.experts.{expert}.{projection}.weight_scale"),
                shape,
                mode,
                p,
                d,
            )
            .map(ExpertLinear::Packed)
        };
        let mut experts = Vec::with_capacity(geo.num_experts);
        for expert in 0..geo.num_experts {
            experts.push(ExpertWeights {
                gate: load(expert, "gate_proj", [intermediate, hidden])?,
                up: load(expert, "up_proj", [intermediate, hidden])?,
                down: load(expert, "down_proj", [hidden, intermediate])?,
            });
        }
        let dense = if mode == quant::LoadMode::Packed {
            let names = |projection: &str| -> Vec<String> {
                (0..geo.num_experts)
                    .map(|expert| format!("{name}.experts.{expert}.{projection}.weight_packed"))
                    .collect()
            };
            let scale_names = |projection: &str| -> Vec<String> {
                (0..geo.num_experts)
                    .map(|expert| format!("{name}.experts.{expert}.{projection}.weight_scale"))
                    .collect()
            };
            let upload = |parts: quant::PackedParts| -> Result<DenseExpertsUpload<B>> {
                Ok(DenseExpertsUpload {
                    words: Tensor::<B, 1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::new(
                            parts.words.iter().map(|word| *word as i32).collect(),
                            [parts.words.len()],
                        ),
                        // The kernels read i32 words linearly; the device's
                        // default int dtype is i64 and would interleave the
                        // payload with sign-extension halves.
                        burn::tensor::TensorCreationOptions::<B> {
                            device: d.clone(),
                            dtype: Some(burn::tensor::DType::I32),
                        },
                    ),
                    scales: Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::new(parts.scales.clone(), [parts.scales.len()]),
                        burn::tensor::TensorCreationOptions::<B> {
                            device: d.clone(),
                            dtype: Some(burn::tensor::DType::F32),
                        },
                    ),
                })
            };
            let gate = upload(quant::packed_vconcat_kmajor_parts(
                w,
                &names("gate_proj"),
                &scale_names("gate_proj"),
                [intermediate, hidden],
            )?)?;
            let up = upload(quant::packed_vconcat_kmajor_parts(
                w,
                &names("up_proj"),
                &scale_names("up_proj"),
                [intermediate, hidden],
            )?)?;
            let down = upload(quant::packed_hconcat_kmajor_parts(
                w,
                &names("down_proj"),
                &scale_names("down_proj"),
                [hidden, intermediate],
            )?)?;
            Some((gate, up, down))
        } else {
            None
        };
        let mut block = Self::assemble_dense(w, name, geo, hidden, experts, p, d)?;
        if let Some((gate, up, down)) = dense {
            // The f32 copies for the fused kernels come from the tensors this
            // block already uploaded, not a second checkpoint read. Linear
            // keeps the executable [input, output] view; the kernels index
            // checkpoint-style [output, input] rows, so transpose back (the
            // cast materializes the transposed view contiguously).
            let f32_out_in = |linear: &Linear<B>| {
                linear
                    .weight()
                    .clone()
                    .transpose()
                    .cast(burn::tensor::FloatDType::F32)
            };
            block.dense = Some(DenseExperts {
                gate_words: gate.words,
                gate_scales: gate.scales,
                up_words: up.words,
                up_scales: up.scales,
                down_words: down.words,
                down_scales: down.scales,
                router: block.router.clone().cast(burn::tensor::FloatDType::F32),
                shared_gate_w: f32_out_in(&block.shared_gate),
                shared_up_w: f32_out_in(&block.shared_up),
                shared_down_w: f32_out_in(&block.shared_down),
                shared_gate_vec: f32_out_in(&block.shared_expert_gate),
                num_experts: block.geometry.num_experts,
                top_k: block.geometry.top_k,
                expert_intermediate: block.geometry.expert_intermediate,
                shared_intermediate: block.geometry.shared_intermediate,
            });
        }
        Ok(block)
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_dense(
        w: &impl WeightSource<B>,
        name: &str,
        geo: MoeGeometry,
        hidden: usize,
        experts: Vec<ExpertWeights<B>>,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let load = |suffix: &str, out: usize, input: usize| -> Result<Linear<B>> {
            Ok(shared::Linear::from_parts(
                w.tensor(&format!("{name}.{suffix}"), [out, input], p, d)?,
                None,
            ))
        };
        Ok(Self {
            router: w.tensor(
                &format!("{name}.gate.weight"),
                [geo.num_experts, hidden],
                p,
                d,
            )?,
            experts,
            dense: None,
            shared_gate: load(
                "shared_expert.gate_proj.weight",
                geo.shared_intermediate,
                hidden,
            )?,
            shared_up: load(
                "shared_expert.up_proj.weight",
                geo.shared_intermediate,
                hidden,
            )?,
            shared_down: load(
                "shared_expert.down_proj.weight",
                hidden,
                geo.shared_intermediate,
            )?,
            shared_expert_gate: load("shared_expert_gate.weight", 1, hidden)?,
            geometry: geo,
            hidden,
        })
    }

    /// Router probabilities renormalized over the top-k, matching the official
    /// SoftmaxTopKRouter.
    pub fn route(&self, hidden: &Tensor<B, 2>) -> (Vec<usize>, Vec<f32>) {
        let logits = hidden.clone().matmul(self.router.clone().transpose());
        let values = logits
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("router readback")
            .to_vec();
        let tokens = values.len() / self.geometry.num_experts;
        let mut experts = Vec::with_capacity(tokens * self.geometry.top_k);
        let mut weights = Vec::with_capacity(tokens * self.geometry.top_k);
        for token in 0..tokens {
            let row =
                &values[token * self.geometry.num_experts..(token + 1) * self.geometry.num_experts];
            let mut order: Vec<usize> = (0..self.geometry.num_experts).collect();
            // Stable top-k: higher probability first, index breaks ties.
            order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
            let top = &order[..self.geometry.top_k];
            // Softmax over all experts (fp32), then renormalize the top-k.
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            let top_sum: f32 = top.iter().map(|&e| exps[e]).sum();
            for &expert in top {
                experts.push(expert);
                weights.push(exps[expert] / top_sum);
            }
        }
        (experts, weights)
    }

    /// One expert's SwiGLU over the selected token rows.
    fn expert_forward(&self, expert: usize, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let weights = &self.experts[expert];
        let gate = weights.gate.matmul_transpose(x.clone());
        let up = weights.up.matmul_transpose(x.clone());
        let activated = activation::silu(gate) * up;
        weights.down.matmul_transpose(activated)
    }

    /// Decode routing and slot-order sum. Each expert is projected once for the
    /// block; repeating that projection per row is what made a verify step cost
    /// about fifteen decodes. The shared expert is the same batched projection
    /// used for a multi-token forward.
    fn forward_rows_matching_decode(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_, tokens, _] = x.dims();
        let flat = x.reshape([tokens, self.hidden]);
        let (experts, weights) = self.route(&flat);
        let mut computed: Vec<Option<Tensor<B, 2>>> = Vec::with_capacity(self.geometry.num_experts);
        computed.resize_with(self.geometry.num_experts, || None);
        for &expert in &experts {
            if computed[expert].is_none() {
                computed[expert] = Some(self.expert_forward(expert, &flat));
            }
        }
        let mut acc: Option<Tensor<B, 2>> = None;
        for slot in 0..self.geometry.top_k {
            let mut pieces = Vec::with_capacity(tokens);
            for token in 0..tokens {
                let expert = experts[token * self.geometry.top_k + slot];
                let weight = weights[token * self.geometry.top_k + slot];
                let rows = computed[expert]
                    .as_ref()
                    .expect("routed expert was projected")
                    .clone();
                pieces.push(rows.slice([token..token + 1, 0..self.hidden]) * weight);
            }
            let slot_out = Tensor::cat(pieces, 0);
            acc = Some(match acc {
                None => slot_out,
                Some(sum) => sum + slot_out,
            });
        }
        let routed = match acc {
            Some(sum) => sum,
            None => {
                let zero: Tensor<B, 2> = Tensor::zeros([tokens, self.hidden], &flat.device());
                zero.cast(shared::dtype(&flat))
            }
        };
        let shared = self.shared_down.forward_rows(
            activation::silu(self.shared_gate.forward_rows(flat.clone()))
                * self.shared_up.forward_rows(flat.clone()),
        );
        let gate = activation::sigmoid(self.shared_expert_gate.forward_rows(flat));
        routed + shared * gate
    }

    /// The grouped kernels of [`crate::moe_dense`]: four launches per layer —
    /// router (top-k + renormalized softmax + shared sigmoid gate), packed
    /// gate/up over the touched experts, shared gate/up, and a down that
    /// folds the weighted combine, the shared expert and the final add.
    /// Returns `None` when the kernels cannot run; the caller falls back to
    /// the dispatched paths.
    #[cfg(feature = "metal")]
    fn forward_dense(&self, flat: Tensor<B, 2>) -> Option<Tensor<B, 2>> {
        let dense = self.dense.as_ref()?;
        let dtype = shared::dtype(&flat);
        crate::moe_dense::try_dense_moe(flat, dense).map(|y| y.cast(dtype))
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_, tokens, _] = x.dims();
        if tokens > 1 && MATCH_DECODE.with(|flag| flag.get()) {
            return self.forward_rows_matching_decode(x);
        }
        #[cfg(feature = "metal")]
        if tokens > 1 && tokens <= DENSE_MOE_MAX_TOKENS && self.dense.is_some() {
            let width = x.dims()[2];
            let flat = x
                .clone()
                .slice([0..1, 0..tokens, 0..width])
                .reshape([tokens, self.hidden]);
            if let Some(output) = self.forward_dense(flat) {
                return output;
            }
        }
        let flat = x.reshape([tokens, self.hidden]);
        let (experts, weights) = self.route(&flat);

        // Expert accumulation stays on the device: a mid-stream host round
        // trip (into_data + from_data) corrupts subsequent kernels in this
        // burn-cubecl build.
        let output = if tokens == 1 {
            // Decode: one row, so each routed expert is already a single
            // matmul per projection. Slot-order accumulation keeps the
            // result bit-identical to the historical per-pair loop while
            // dropping the per-pair slice/add/slice_assign traffic.
            let mut acc: Option<Tensor<B, 2>> = None;
            for slot in 0..self.geometry.top_k {
                let weighted = self.expert_forward(experts[slot], &flat) * weights[slot];
                acc = Some(match acc {
                    None => weighted,
                    Some(sum) => sum + weighted,
                });
            }
            acc.unwrap_or_else(|| {
                let zero: Tensor<B, 2> = Tensor::zeros([tokens, self.hidden], &flat.device());
                zero.cast(shared::dtype(&flat))
            })
        } else {
            // Prefill: batch each touched expert over all token rows (one
            // matmul per projection per expert, not per (token, expert)
            // pair) and combine with the device-side renormalized top-k
            // weight matrix. The masked softmax equals softmax-then-
            // renormalize over the top-k because the global logit maximum
            // always survives the mask; masked-out columns are exact zeros.
            let logits = flat.clone().matmul(self.router.clone().transpose());
            let kth = logits.clone().topk(self.geometry.top_k, 1).min_dim(1);
            let outside = logits.clone().lower(kth);
            let masked = logits.mask_fill(outside, f32::NEG_INFINITY);
            let weights2d = activation::softmax(masked, 1);

            let mut touched = experts.clone();
            touched.sort_unstable();
            touched.dedup();

            let mut output: Tensor<B, 2> = Tensor::zeros([tokens, self.hidden], &flat.device());
            output = output.cast(shared::dtype(&flat));
            for expert in &touched {
                let rows = self.expert_forward(*expert, &flat);
                let column = weights2d.clone().slice([0..tokens, *expert..*expert + 1]);
                output = output + rows * column;
            }
            output
        };

        // Shared expert with sigmoid gate.
        let shared = self.shared_down.forward_rows(
            activation::silu(self.shared_gate.forward_rows(flat.clone()))
                * self.shared_up.forward_rows(flat.clone()),
        );
        let gate = activation::sigmoid(self.shared_expert_gate.forward_rows(flat));
        output + shared * gate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeParameters;
    use burn::backend::NdArray;
    use safetensors::SafeTensors;

    fn config() -> TextConfig {
        TextConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 4,
            layer_types: vec!["full_attention".into()],
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
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
                partial_rotary_factor: 1.0,
            },
            mamba_ssm_dtype: None,
            num_experts: 6,
            num_experts_per_tok: 2,
            moe_intermediate_size: 32,
            shared_expert_intermediate_size: 32,
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
        assert!(rmse < 5e-4, "{label}: RMSE {rmse:.8}");
    }

    #[test]
    fn moe_block_matches_official_synthetic_dump() {
        type B = NdArray;
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/moe-block.safetensors"),
        )
        .unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes.clone()]).unwrap();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = config();
        let block =
            SparseMoeBlock::<B>::load(&weights, "moe.block", &config, Precision::F32, &device)
                .unwrap();

        let hidden: Vec<f32> = stage(&tensors, "stage_hidden_in");
        let seq = 12;
        let x = Tensor::<B, 3>::from_data(
            TensorData::new(hidden, [1, seq, config.hidden_size]),
            &device,
        );

        // Router: experts and renormalized weights.
        let flat = x.clone().reshape([seq, config.hidden_size]);
        let (experts, weights_out) = block.route(&flat);
        // as_chunks would need a newer MSRV than the workspace
        #[allow(clippy::chunks_exact_to_as_chunks)]
        let expected_indices: Vec<i64> = tensors
            .tensor("stage_router_indices")
            .unwrap()
            .data()
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        let got: Vec<i64> = experts.iter().map(|&e| e as i64).collect();
        assert_eq!(got, expected_indices, "router expert selection");
        let expected_scores = stage(&tensors, "stage_router_scores");
        for (position, (a, b)) in weights_out.iter().zip(&expected_scores).enumerate() {
            assert!((a - b).abs() < 1e-6, "router weight {position}: {a} vs {b}");
        }

        let output = block.forward(x);
        // The block returns (tokens, hidden); compare against the fixture's
        // (1, tokens, hidden) dump.
        let actual = output
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        compare("block output", &actual, &stage(&tensors, "stage_block_out"));
    }

    #[test]
    fn grouped_expert_layout_feeds_int4_loader() {
        type B = NdArray;
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/moe-block.safetensors"),
        )
        .unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes.clone()]).unwrap();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // in_dim padded to the group multiple in the fixture (64 -> 128).
        let packed = quant::load_packed::<B>(
            &weights,
            "moe.expert_packed.weight",
            "moe.expert_packed.weight_scale",
            [64, 128],
            quant::LoadMode::Dequantize,
            Precision::F32,
            &device,
        )
        .unwrap();
        let actual = match &packed {
            quant::PackedWeight::Dequantized(weight) => weight.clone().into_data(),
            quant::PackedWeight::Quantized(_) => unreachable!("reference backend"),
        }
        .convert::<f32>()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
        let original = stage(&tensors, "moe.expert_padded_original");
        let worst = original
            .iter()
            .zip(&actual)
            .map(|(b, a)| (b - a).abs())
            .fold(0f32, f32::max);
        // The bf16-rounded scale adds up to a full int4 step at the extremes.
        let scale = original.iter().map(|v| v.abs()).fold(0f32, f32::max) / 7.0;
        assert!(
            worst <= scale + 1e-4,
            "worst dequant delta {worst} vs step {scale}"
        );
    }
}
