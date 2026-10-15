//! MoE sparse block: softmax top-k router, grouped experts with fused
//! gate|up projections, and a sigmoid-gated shared expert (Qwen3.5-Moe).
use crate::{config::TextConfig, quant};
use burn::tensor::{Tensor, TensorData, activation, backend::Backend};
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

pub struct SparseMoeBlock<B: Backend> {
    router: Tensor<B, 2>, // (experts, hidden)
    /// (experts, 2*intermediate, hidden)
    gate_up: Tensor<B, 3>,
    /// (experts, hidden, intermediate)
    down: Tensor<B, 3>,
    shared_gate: Linear<B>,
    shared_up: Linear<B>,
    shared_down: Linear<B>,
    shared_expert_gate: Linear<B>,
    geometry: MoeGeometry,
    hidden: usize,
}

impl<B: Backend> SparseMoeBlock<B> {
    /// 1B-style dense loading (bf16 safetensors, 3-D expert tensors).
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let geometry = MoeGeometry {
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            expert_intermediate: config.moe_intermediate_size,
            shared_intermediate: config.shared_expert_intermediate_size,
        };
        let hidden = config.hidden_size;
        let load3 = |suffix: &str, shape: [usize; 3]| -> Result<Tensor<B, 3>> {
            w.tensor(&format!("{name}.{suffix}"), shape, p, d)
        };
        Ok(Self {
            router: w.tensor(
                &format!("{name}.gate.weight"),
                [geometry.num_experts, hidden],
                p,
                d,
            )?,
            gate_up: load3(
                "experts.gate_up_proj",
                [
                    geometry.num_experts,
                    2 * geometry.expert_intermediate,
                    hidden,
                ],
            )?,
            down: load3(
                "experts.down_proj",
                [geometry.num_experts, hidden, geometry.expert_intermediate],
            )?,
            shared_gate: Linear::from_parts(
                w.tensor(
                    &format!("{name}.shared_expert.gate_proj.weight"),
                    [geometry.shared_intermediate, hidden],
                    p,
                    d,
                )?,
                None,
            ),
            shared_up: Linear::from_parts(
                w.tensor(
                    &format!("{name}.shared_expert.up_proj.weight"),
                    [geometry.shared_intermediate, hidden],
                    p,
                    d,
                )?,
                None,
            ),
            shared_down: Linear::from_parts(
                w.tensor(
                    &format!("{name}.shared_expert.down_proj.weight"),
                    [hidden, geometry.shared_intermediate],
                    p,
                    d,
                )?,
                None,
            ),
            shared_expert_gate: Linear::from_parts(
                w.tensor(
                    &format!("{name}.shared_expert_gate.weight"),
                    [1, hidden],
                    p,
                    d,
                )?,
                None,
            ),
            geometry,
            hidden,
        })
    }

    /// Router probabilities renormalized over the top-k, matching the official
    /// SoftmaxTopKRouter.
    pub fn route(&self, hidden: &Tensor<B, 2>) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
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
        let mut raw = Vec::with_capacity(tokens * self.geometry.top_k);
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
            let total: f32 = exps.iter().sum();
            let top_sum: f32 = top.iter().map(|&e| exps[e]).sum();
            for &expert in top {
                experts.push(expert);
                let weight = exps[expert] / top_sum;
                weights.push(weight);
                raw.push(exps[expert] / total);
            }
        }
        (experts, weights, raw)
    }

    /// One expert's SwiGLU over the selected token rows.
    fn expert_forward(&self, expert: usize, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let gate_up = self
            .gate_up
            .clone()
            .slice(vec![expert..expert + 1])
            .squeeze_dim::<2>(0);
        let (gate, up) = {
            let projected = x.clone().matmul(gate_up.transpose());
            let [tokens, width] = projected.dims();
            (
                projected.clone().slice([0..tokens, 0..width / 2]),
                projected.slice([0..tokens, width / 2..width]),
            )
        };
        let activated = activation::silu(gate) * up;
        let down = self
            .down
            .clone()
            .slice(vec![expert..expert + 1])
            .squeeze_dim::<2>(0);
        activated.matmul(down.transpose())
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, tokens, _] = x.dims();
        let flat = x.reshape([tokens, self.hidden]);
        let (experts, weights, _) = self.route(&flat);

        // Dense per-token accumulation in fp32; each token visits top-k experts.
        let mut output = vec![0f32; tokens * self.hidden];
        let device = flat.device();
        for (slot, (&expert, &weight)) in experts.iter().zip(&weights).enumerate() {
            let token = slot / self.geometry.top_k;
            let row = flat
                .clone()
                .slice([token..token + 1])
                .reshape([1, self.hidden]);
            let contributed = self.expert_forward(expert, &row);
            let values = contributed
                .into_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .expect("expert readback")
                .to_vec();
            for (position, value) in values.iter().enumerate() {
                output[token * self.hidden + position] += value * weight;
            }
        }
        let mut expert_out =
            Tensor::<B, 2>::from_data(TensorData::new(output, [tokens, self.hidden]), &device);

        // Shared expert with sigmoid gate.
        let shared = self.shared_down.forward(
            activation::silu(self.shared_gate.forward(flat.clone()))
                * self.shared_up.forward(flat.clone()),
        );
        let gate = activation::sigmoid(self.shared_expert_gate.forward(flat));
        expert_out = expert_out + shared * gate;
        expert_out.reshape([1, tokens, self.hidden])
    }
}

/// The MoE text-config fields (absent on the dense 1B config).
fn moe_fields_missing(config: &TextConfig) -> bool {
    config.num_experts == 0
}

pub fn layer_is_moe(config: &TextConfig) -> bool {
    !moe_fields_missing(config)
}

#[allow(dead_code)]
fn quant_hook<B: Backend>(_w: &impl WeightSource<B>) {
    let _ = quant::GROUP_SIZE;
}

#[cfg(test)]
mod tests {
    use super::*;

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
            rope_parameters: crate::config::RopeParameters {
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
        let (experts, weights_out, _) = block.route(&flat);
        let expected_indices: Vec<i64> = tensors
            .tensor("stage_router_indices")
            .unwrap()
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        let got: Vec<i64> = experts.iter().map(|&e| e as i64).collect();
        assert_eq!(got, expected_indices, "router expert selection");
        let expected_scores = stage(&tensors, "stage_router_scores");
        for (position, (a, b)) in weights_out.iter().zip(&expected_scores).enumerate() {
            assert!((a - b).abs() < 1e-6, "router weight {position}: {a} vs {b}");
        }

        let shared = stage(&tensors, "stage_shared_out");
        let _ = shared; // covered through block output parity

        let output = block.forward(x);
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
            [64, 128],
            quant::GROUP_SIZE,
            Precision::F32,
            &device,
        )
        .unwrap();
        let actual = packed
            .weight()
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let original = stage(&tensors, "moe.expert_padded_original");
        let step = original
            .iter()
            .zip(&actual)
            .map(|(b, a)| (b - a).abs())
            .fold(0f32, f32::max);
        // The bf16-rounded scale adds up to a full int4 step at the extremes.
        let scale = original.iter().map(|v| v.abs()).fold(0f32, f32::max) / 7.0;
        assert!(
            step <= scale + 1e-4,
            "worst dequant delta {step} vs step {scale}"
        );
    }
}
