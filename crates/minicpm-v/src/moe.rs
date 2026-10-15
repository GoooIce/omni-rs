//! MoE sparse block: softmax top-k router, grouped experts, and a
//! sigmoid-gated shared expert (Qwen3.5-Moe). Experts load either fused
//! (bf16 3-D tensors) or pack-quantized int4 per projection (35B checkpoint).
use crate::{config::TextConfig, quant};
#[cfg(test)]
use burn::tensor::TensorData;
use burn::tensor::{Tensor, activation, backend::Backend};
use omni_r2t2::{Precision, Result, tensor as shared, weights::WeightSource};

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

pub struct SparseMoeBlock<B: Backend> {
    router: Tensor<B, 2>, // (experts, hidden)
    experts: Vec<ExpertWeights<B>>,
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
        Self::assemble(w, name, geo, hidden, experts, p, d)
    }

    /// 35B int4 layout: per-expert `{name}.experts.{e}.{gate,up,down}_proj`
    /// with `.weight_packed` payloads and `.weight_scale` group scales.
    pub fn load_quantized(
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
        Self::assemble(w, name, geo, hidden, experts, p, d)
    }

    fn assemble(
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

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_, tokens, _] = x.dims();
        let flat = x.reshape([tokens, self.hidden]);
        let (experts, weights) = self.route(&flat);

        // Per-token expert accumulation stays on the device: a mid-stream
        // host round trip (into_data + from_data) corrupts subsequent kernels
        // in this burn-cubecl build, so contributions are gathered with
        // slice/slice_assign instead.
        let mut output: Tensor<B, 2> = Tensor::zeros([tokens, self.hidden], &flat.device());
        output = output.cast(shared::dtype(&flat));
        for (slot, (&expert, &weight)) in experts.iter().zip(&weights).enumerate() {
            let token = slot / self.geometry.top_k;
            let row = flat
                .clone()
                .slice([token..token + 1, 0..self.hidden])
                .clone();
            let weighted = self.expert_forward(expert, &row) * weight;
            let updated = output.clone().slice([token..token + 1, 0..self.hidden]) + weighted;
            output = output.slice_assign([token..token + 1, 0..self.hidden], updated);
        }

        // Shared expert with sigmoid gate.
        let shared = self.shared_down.forward(
            activation::silu(self.shared_gate.forward(flat.clone()))
                * self.shared_up.forward(flat.clone()),
        );
        let gate = activation::sigmoid(self.shared_expert_gate.forward(flat));
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
