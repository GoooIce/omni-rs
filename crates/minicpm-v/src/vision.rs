//! SigLIP2 vision tower: NaViT patch embedding with nearest-neighbor position
//! interpolation, 27 encoder layers with a 2×2 window-attention merger inserted
//! after layer `insert_layer_id`, and the final 2×2 → language-dimension MLP
//! merger (16× compression overall).
use crate::{chat::Grid, config::VisionConfig};
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::{Precision, tensor as shared, weights::WeightSource};

struct Attention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        width: usize,
        heads: usize,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        let head_dim = width / heads;
        Ok(Self {
            q: Linear::load(w, &format!("{name}.q_proj"), width, width, true, p, d)?,
            k: Linear::load(w, &format!("{name}.k_proj"), width, width, true, p, d)?,
            v: Linear::load(w, &format!("{name}.v_proj"), width, width, true, p, d)?,
            out: Linear::load(w, &format!("{name}.out_proj"), width, width, true, p, d)?,
            heads,
            head_dim,
        })
    }

    /// Bidirectional attention over `(batch, tokens, width)`; the batch is
    /// either one crop or one attention window per row.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, _] = x.dims();
        let split = |y: Tensor<B, 3>| {
            y.reshape([batch, tokens, self.heads, self.head_dim])
                .permute([0, 2, 1, 3])
        };
        let q = split(self.q.forward(x.clone()));
        let k = split(self.k.forward(x.clone()));
        let v = split(self.v.forward(x.clone()));
        let attended = shared::attention(
            q,
            k,
            v,
            shared::AttentionOptions {
                causal: false,
                ..Default::default()
            },
        );
        let concatenated =
            attended
                .permute([0, 2, 1, 3])
                .reshape([batch, tokens, self.heads * self.head_dim]);
        self.out.forward(concatenated)
    }
}

type Linear<B> = shared::Linear<B>;

struct EncoderLayer<B: Backend> {
    ln1: shared::Norm<B>,
    attn: Attention<B>,
    ln2: shared::Norm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &VisionConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        let width = config.hidden_size;
        Ok(Self {
            ln1: shared::Norm::load(
                w,
                &format!("{name}.layer_norm1"),
                width,
                config.layer_norm_eps,
                shared::NormKind::Layer,
                shared::NormReduction::Divide,
                d,
            )?,
            attn: Attention::load(
                w,
                &format!("{name}.self_attn"),
                width,
                config.num_attention_heads,
                p,
                d,
            )?,
            ln2: shared::Norm::load(
                w,
                &format!("{name}.layer_norm2"),
                width,
                config.layer_norm_eps,
                shared::NormKind::Layer,
                shared::NormReduction::Divide,
                d,
            )?,
            fc1: Linear::load(
                w,
                &format!("{name}.mlp.fc1"),
                width,
                config.intermediate_size,
                true,
                p,
                d,
            )?,
            fc2: Linear::load(
                w,
                &format!("{name}.mlp.fc2"),
                config.intermediate_size,
                width,
                true,
                p,
                d,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>, output: FloatDType) -> Tensor<B, 3> {
        let attended = self.attn.forward(self.ln1.forward(x.clone(), output));
        let x = shared::residual(x, attended, output);
        let mlp = self.fc2.forward(shared::gelu(
            self.fc1.forward(self.ln2.forward(x.clone(), output)),
            shared::GeluApproximation::Tanh,
            output,
        ));
        shared::residual(x, mlp, output)
    }
}

/// The 2×2 window-attention merger inserted mid-encoder; quarters the token
/// count and leaves the sequence in window-row-major order.
struct WindowMerger<B: Backend> {
    ln1: shared::Norm<B>,
    attn: Attention<B>,
    pre_norm: shared::Norm<B>,
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    width: usize,
    window: usize,
}

impl<B: Backend> WindowMerger<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &VisionConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        Ok(Self {
            ln1: shared::Norm::load(
                w,
                &format!("{name}.layer_norm1"),
                config.hidden_size,
                config.layer_norm_eps,
                shared::NormKind::Layer,
                shared::NormReduction::Divide,
                d,
            )?,
            attn: Attention::load(
                w,
                &format!("{name}.self_attn"),
                config.hidden_size,
                config.num_attention_heads,
                p,
                d,
            )?,
            pre_norm: shared::Norm::load(
                w,
                &format!("{name}.pre_norm"),
                config.window_hidden_size(),
                config.layer_norm_eps,
                shared::NormKind::Layer,
                shared::NormReduction::Divide,
                d,
            )?,
            linear_1: Linear::load(
                w,
                &format!("{name}.linear_1"),
                config.window_hidden_size(),
                config.window_intermediate_size(),
                true,
                p,
                d,
            )?,
            linear_2: Linear::load(
                w,
                &format!("{name}.linear_2"),
                config.window_intermediate_size(),
                config.hidden_size,
                true,
                p,
                d,
            )?,
            width: config.hidden_size,
            window: config.window_kernel_size.0,
        })
    }

    /// `grid` is the crop's full-resolution patch grid.
    fn forward(&self, x: Tensor<B, 3>, grid: Grid, output: FloatDType) -> Tensor<B, 3> {
        let [_, tokens, width] = x.dims();
        assert_eq!(width, self.width);
        assert_eq!(tokens, grid.height * grid.width);
        let windows_h = grid.height / self.window;
        let windows_w = grid.width / self.window;
        // Row-major token order regrouped into contiguous 2×2 windows.
        let mut window_index = Vec::with_capacity(tokens);
        for wy in 0..windows_h {
            for wx in 0..windows_w {
                for dy in 0..self.window {
                    for dx in 0..self.window {
                        window_index.push(
                            ((wy * self.window + dy) * grid.width + wx * self.window + dx) as i32,
                        );
                    }
                }
            }
        }
        let device = x.device();
        let index = Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(window_index, [tokens]),
            &device,
        );
        let permuted = x.clone().select(1, index);
        let normed = self.ln1.forward(permuted.clone(), output);
        // Batch windows as the sequence dimension: attention over 4 tokens each.
        let window_tokens = self.window * self.window;
        let windows = windows_h * windows_w;
        let batched = normed.reshape([windows, window_tokens, width]);
        let attended = self.attn.forward(batched);
        let residual = permuted.reshape([windows, window_tokens, width]);
        let merged_input = shared::residual(residual, attended, output);
        // (windows, 4, width) → per-window flat 4·width.
        let flat = merged_input
            .clone()
            .reshape([windows, window_tokens * width]);
        let patch_residual = merged_input
            .clone()
            .mean_dim(1)
            .reshape([windows, 1, width]);
        let projected = self.linear_2.forward(shared::gelu(
            self.linear_1.forward(self.pre_norm.forward(flat, output)),
            shared::GeluApproximation::Tanh,
            output,
        ));
        let out = shared::residual(
            projected.reshape([windows, 1, width]),
            patch_residual,
            output,
        );
        out.reshape([1, windows, width])
    }
}

/// Final 2×2 merge into the language dimension; exact-erf GELU like the official
/// `nn.GELU()`.
struct DownsampleMlp<B: Backend> {
    pre_norm: shared::Norm<B>,
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    kernel: usize,
    in_width: usize,
}

impl<B: Backend> DownsampleMlp<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        config: &VisionConfig,
        llm_width: usize,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        let in_width = config.hidden_size * 4;
        Ok(Self {
            pre_norm: shared::Norm::load(
                w,
                &format!("{name}.pre_norm"),
                in_width,
                1e-6,
                shared::NormKind::Layer,
                shared::NormReduction::Divide,
                d,
            )?,
            linear_1: Linear::load(
                w,
                &format!("{name}.linear_1"),
                in_width,
                in_width,
                true,
                p,
                d,
            )?,
            linear_2: Linear::load(
                w,
                &format!("{name}.linear_2"),
                in_width,
                llm_width,
                true,
                p,
                d,
            )?,
            kernel: 2,
            in_width,
        })
    }

    /// Tokens arrive in window-grid row-major order `(H/2, W/2)`.
    fn forward(&self, x: Tensor<B, 3>, grid: Grid, output: FloatDType) -> Tensor<B, 2> {
        let [_, tokens, width] = x.dims();
        assert_eq!(width, self.in_width / 4);
        let merged_h = grid.height / self.kernel;
        let merged_w = grid.width / self.kernel;
        assert_eq!(
            tokens,
            grid.height * grid.width,
            "tokens must tile the post-window grid"
        );
        // (merged_h, 2, merged_w, 2, width) → (merged_h·merged_w, 4·width)
        let grouped = x
            .reshape([merged_h, self.kernel, merged_w, self.kernel, width])
            .permute([0, 2, 1, 3, 4])
            .reshape([merged_h * merged_w, self.kernel * self.kernel * width]);
        self.linear_2.forward(shared::gelu(
            self.linear_1
                .forward(self.pre_norm.forward(grouped, output)),
            shared::GeluApproximation::Erf,
            output,
        ))
    }
}

/// Per-stage tensors captured during a forward pass, for layer parity tests.
/// Sequence stages concatenate across crops exactly like the official packed
/// dump; `encoder` holds one entry per probe layer, in probe order.
#[derive(Default)]
pub struct Stages<B: Backend> {
    pub embeddings: Option<Tensor<B, 3>>,
    pub after_window_merger: Option<Tensor<B, 3>>,
    pub after_post_layernorm: Option<Tensor<B, 3>>,
    pub merger_output: Option<Tensor<B, 2>>,
    pub encoder: Vec<Option<Tensor<B, 3>>>,
}

impl<B: Backend> Stages<B> {
    fn concat3(slot: &mut Option<Tensor<B, 3>>, next: Option<Tensor<B, 3>>) {
        *slot = match (slot.take(), next) {
            (Some(previous), Some(next)) => Some(Tensor::cat(vec![previous, next], 1)),
            (_, next) => next,
        };
    }
}

pub struct VisionTower<B: Backend> {
    patch: Linear<B>,
    positions: shared::Embedding<B>,
    layers: Vec<EncoderLayer<B>>,
    window_merger: WindowMerger<B>,
    post_layernorm: shared::Norm<B>,
    merger: DownsampleMlp<B>,
    config: crate::config::MiniCpmConfig,
}

impl<B: Backend> VisionTower<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        config: &crate::config::MiniCpmConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        let vision = &config.vision_config;
        let patch_flat = vision.patch_size * vision.patch_size * vision.num_channels;
        let mut layers = Vec::with_capacity(vision.num_hidden_layers);
        for index in 0..vision.num_hidden_layers {
            layers.push(EncoderLayer::load(
                w,
                &format!("model.vision_tower.encoder.layers.{index}"),
                vision,
                p,
                d,
            )?);
        }
        // The checkpoint stores the patch conv as (out, in, ky, kx); flatten to
        // a linear view over (out, in*ky*kx).
        let patch_weight: Tensor<B, 2> = w
            .tensor(
                "model.vision_tower.embeddings.patch_embedding.weight",
                [
                    vision.hidden_size,
                    vision.num_channels,
                    vision.patch_size,
                    vision.patch_size,
                ],
                p,
                d,
            )?
            .reshape([vision.hidden_size, patch_flat]);
        let patch_bias: Tensor<B, 1> = w.tensor(
            "model.vision_tower.embeddings.patch_embedding.bias",
            [vision.hidden_size],
            p,
            d,
        )?;
        Ok(Self {
            patch: Linear::from_parts(patch_weight, Some(patch_bias)),
            positions: shared::Embedding::load(
                w,
                "model.vision_tower.embeddings.position_embedding.weight",
                vision.num_patches_per_side().pow(2),
                vision.hidden_size,
                Precision::F32,
                d,
            )?,
            layers,
            window_merger: WindowMerger::load(w, "model.vision_tower.vit_merger", vision, p, d)?,
            post_layernorm: shared::Norm::load(
                w,
                "model.vision_tower.post_layernorm",
                vision.hidden_size,
                vision.layer_norm_eps,
                shared::NormKind::Layer,
                shared::NormReduction::Divide,
                d,
            )?,
            merger: DownsampleMlp::load(
                w,
                &format!("model.merger.mlp.{}", config.merger_times - 1),
                vision,
                config.text_config.hidden_size,
                p,
                d,
            )?,
            config: config.clone(),
        })
    }

    /// Nearest-neighbor position bucket ids for one crop's patch grid.
    fn position_ids(&self, grid: Grid) -> Vec<u32> {
        let side = self.config.vision_config.num_patches_per_side();
        let mut boundaries = Vec::with_capacity(side - 1);
        for k in 1..side {
            boundaries.push(k as f32 / side as f32);
        }
        let bucket = |values: usize| -> Vec<u32> {
            (0..values)
                .map(|i| {
                    let v = i as f32 / values as f32;
                    // torch.bucketize right=True counts boundaries <= v (ties right).
                    let mut slot = 0;
                    while slot < boundaries.len() && boundaries[slot] <= v {
                        slot += 1;
                    }
                    slot as u32
                })
                .collect()
        };
        let rows = bucket(grid.height);
        let cols = bucket(grid.width);
        let mut ids = Vec::with_capacity(grid.height * grid.width);
        for row in &rows {
            for col in &cols {
                ids.push(row * side as u32 + col);
            }
        }
        ids
    }

    /// All crops of one image; embeddings concatenate in crop order.
    pub fn forward(
        &self,
        crops: &[Tensor<B, 2>],
        target_sizes: &[Grid],
        output: FloatDType,
    ) -> omni_r2t2::Result<Tensor<B, 2>> {
        assert_eq!(crops.len(), target_sizes.len());
        let mut embeddings = Vec::with_capacity(crops.len());
        for (patches, grid) in crops.iter().zip(target_sizes) {
            embeddings.push(self.forward_crop(patches.clone(), *grid, output, &mut None)?);
        }
        Ok(Tensor::cat(embeddings, 0))
    }

    /// Forward with per-stage capture for parity tests and diagnostics.
    pub fn forward_staged(
        &self,
        crops: &[Tensor<B, 2>],
        target_sizes: &[Grid],
        output: FloatDType,
    ) -> omni_r2t2::Result<Stages<B>> {
        assert_eq!(crops.len(), target_sizes.len());
        // The reference dump probes these encoder layers, in this order.
        let probes = [
            0,
            self.config.insert_layer_id - 1,
            self.config.insert_layer_id + 1,
            self.config.vision_config.num_hidden_layers - 1,
        ];
        let mut stages = Stages::<B> {
            encoder: vec![None; probes.len()],
            ..Default::default()
        };
        let mut embeddings = Vec::with_capacity(crops.len());
        for (patches, grid) in crops.iter().zip(target_sizes) {
            let mut crop = Stages::<B> {
                encoder: vec![None; probes.len()],
                ..Default::default()
            };
            embeddings.push(self.forward_crop(
                patches.clone(),
                *grid,
                output,
                &mut Some(&mut crop),
            )?);
            Stages::concat3(&mut stages.embeddings, crop.embeddings);
            Stages::concat3(&mut stages.after_window_merger, crop.after_window_merger);
            Stages::concat3(&mut stages.after_post_layernorm, crop.after_post_layernorm);
            for (slot, next) in stages.encoder.iter_mut().zip(crop.encoder) {
                Stages::concat3(slot, next);
            }
        }
        stages.merger_output = Some(Tensor::cat(embeddings, 0));
        Ok(stages)
    }

    /// One crop's patches `(n, 588)` → language-space embeddings `(n/16, 1024)`.
    fn forward_crop(
        &self,
        patches: Tensor<B, 2>,
        grid: Grid,
        output: FloatDType,
        stages: &mut Option<&mut Stages<B>>,
    ) -> omni_r2t2::Result<Tensor<B, 2>> {
        let probes = [
            0,
            self.config.insert_layer_id - 1,
            self.config.insert_layer_id + 1,
            self.config.vision_config.num_hidden_layers - 1,
        ];
        let tokens = patches.dims()[0];
        let width = self.config.vision_config.hidden_size;
        let mut hidden = self.patch.forward(patches).reshape([1, tokens, width]);
        let positions = self
            .positions
            .embed(&self.position_ids(grid))?
            .reshape([1, tokens, width]);
        hidden = shared::residual(hidden, positions, output);
        if let Some(stages) = stages.as_deref_mut() {
            stages.embeddings = Some(hidden.clone());
        }
        let insert = self.config.insert_layer_id;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(hidden, output);
            if index == insert {
                hidden = self.window_merger.forward(hidden, grid, output);
                if let Some(stages) = stages.as_deref_mut() {
                    stages.after_window_merger = Some(hidden.clone());
                }
            }
            if let Some(slot) = probes.iter().position(|p| *p == index)
                && let Some(stages) = stages.as_deref_mut()
            {
                stages.encoder[slot] = Some(hidden.clone());
            }
        }
        hidden = self.post_layernorm.forward(hidden, output);
        if let Some(stages) = stages.as_deref_mut() {
            stages.after_post_layernorm = Some(hidden.clone());
        }
        let merged = self.merger.forward(
            hidden,
            Grid {
                height: grid.height / self.config.vision_config.window_kernel_size.0,
                width: grid.width / self.config.vision_config.window_kernel_size.1,
            },
            output,
        );
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{self, PreprocessConfig};
    use burn::backend::NdArray;
    use safetensors::SafeTensors;

    fn reference_available(name: &str) -> Option<std::path::PathBuf> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root
            .join("artifacts/minicpm-reference/vision")
            .join(format!("{name}.safetensors"));
        let model = root.join("models/MiniCPM-V-4.7-1B/model.safetensors");
        if path.is_file() && model.is_file() {
            Some(path)
        } else {
            eprintln!("skipping: vision reference or checkpoint not present");
            None
        }
    }

    fn stage_bytes(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
        let view = tensors.tensor(name).unwrap();
        view.data()
            #[allow(clippy::chunks_exact_to_as_chunks)] // as_chunks needs newer MSRV than the workspace
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn tensor_values(tensor: Tensor<NdArray, 3>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn tensor_values2(tensor: Tensor<NdArray, 2>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// Relative RMSE and cosine similarity against the official stage dump.
    fn compare(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut squared = 0f64;
        let mut energy = 0f64;
        let mut dot = 0f64;
        let mut a_norm = 0f64;
        for (&a, &b) in actual.iter().zip(expected) {
            squared += (a as f64 - b as f64).powi(2);
            energy += (b as f64).powi(2);
            dot += a as f64 * b as f64;
            a_norm += (a as f64).powi(2);
        }
        let rmse = squared.sqrt() / energy.sqrt().max(1e-12);
        let cosine = dot / (a_norm.sqrt() * energy.sqrt()).max(1e-12);
        // JPEG inputs decode differently between libjpeg and this crate
        // (<=3/255 on <1% of pixels); layer depth amplifies that to ~1%.
        let (rmse_limit, cosine_limit) = if label.starts_with("large.jpg") {
            (2e-2, 0.999)
        } else {
            (2e-3, 0.9995)
        };
        assert!(
            rmse < rmse_limit && cosine > cosine_limit,
            "{label}: relative RMSE {rmse:.6}, cosine {cosine:.8}"
        );
    }

    #[test]
    fn vision_stages_match_official_dumps() {
        type B = NdArray;
        let Some(_) = reference_available("small.png") else {
            return;
        };
        let device: burn::backend::ndarray::NdArrayDevice = Default::default();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let weights =
            omni_r2t2::weights::Weights::open(&root.join("models/MiniCPM-V-4.7-1B")).unwrap();
        let config = crate::config::ModelDirectory::new(root.join("models/MiniCPM-V-4.7-1B"))
            .config()
            .unwrap();
        let tower: VisionTower<B> =
            VisionTower::load(&weights, &config, Precision::F32, &device).unwrap();

        for name in ["small.png", "large.jpg"] {
            let bytes = std::fs::read(reference_available(name).unwrap()).unwrap();
            let tensors = SafeTensors::deserialize(&bytes).unwrap();
            let image = image::open(
                &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/images")
                    .join(name),
            )
            .unwrap();
            let preprocessed = image::preprocess(&image, &PreprocessConfig::default()).unwrap();
            let crops: Vec<Tensor<B, 2>> = preprocessed
                .crops
                .iter()
                .map(|values| {
                    let n = config.vision_config.patch_size
                        * config.vision_config.patch_size
                        * config.vision_config.num_channels;
                    Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(values.clone(), [values.len() / n, n]),
                        &device,
                    )
                })
                .collect();
            let stages = tower
                .forward_staged(&crops, &preprocessed.target_sizes, FloatDType::F32)
                .unwrap();

            compare(
                &tensor_values(stages.embeddings.clone().unwrap()),
                &stage_bytes(&tensors, "embeddings"),
                &format!("{name} embeddings"),
            );
            compare(
                &tensor_values(stages.encoder[0].clone().unwrap()),
                &stage_bytes(&tensors, "encoder.layers.0"),
                &format!("{name} layer 0"),
            );
            compare(
                &tensor_values(stages.encoder[1].clone().unwrap()),
                &stage_bytes(&tensors, "encoder.layers.5"),
                &format!("{name} layer 5"),
            );
            compare(
                &tensor_values(stages.after_window_merger.clone().unwrap()),
                &stage_bytes(&tensors, "vit_merger"),
                &format!("{name} window merger"),
            );
            compare(
                &tensor_values(stages.encoder[2].clone().unwrap()),
                &stage_bytes(&tensors, "encoder.layers.7"),
                &format!("{name} layer 7"),
            );
            compare(
                &tensor_values(stages.encoder[3].clone().unwrap()),
                &stage_bytes(&tensors, "encoder.layers.26"),
                &format!("{name} layer 26"),
            );
            compare(
                &tensor_values(stages.after_post_layernorm.clone().unwrap()),
                &stage_bytes(&tensors, "post_layernorm"),
                &format!("{name} post layernorm"),
            );
            compare(
                &tensor_values2(stages.merger_output.clone().unwrap()),
                &stage_bytes(&tensors, "merger_output"),
                &format!("{name} merger output"),
            );
        }
    }

    #[test]
    fn shapes_follow_sixteen_fold_compression() {
        type B = NdArray;
        let Some(_) = reference_available("small.png") else {
            return;
        };
        let device: burn::backend::ndarray::NdArrayDevice = Default::default();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let weights =
            omni_r2t2::weights::Weights::open(&root.join("models/MiniCPM-V-4.7-1B")).unwrap();
        let config = crate::config::ModelDirectory::new(root.join("models/MiniCPM-V-4.7-1B"))
            .config()
            .unwrap();
        let tower: VisionTower<B> =
            VisionTower::load(&weights, &config, Precision::F32, &device).unwrap();
        // Runtime grids are always multiples of 4 patches (the processor's
        // 56-pixel constraint); 980/14=70 is only the position-table side.
        // A 64×64 patch grid compresses 4096 patches to 256 language tokens.
        let side = 64;
        let patches = side * side;
        let crop = Tensor::<B, 2>::zeros([patches, 3 * 14 * 14], &device);
        let output = tower
            .forward(
                &[crop],
                &[Grid {
                    height: side,
                    width: side,
                }],
                FloatDType::F32,
            )
            .unwrap();
        assert_eq!(
            output.dims(),
            [patches / 16, config.text_config.hidden_size]
        );
    }
}
