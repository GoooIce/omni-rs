//! Qwen3-VL image tower, including the three DeepStack mergers.
use crate::{Error, Precision, Progress, Result, check_cancel, config::VisionConfig, ops::*};
use burn::tensor::{FloatDType, Int, Tensor, TensorData, backend::Backend};
use omni_r2t2::weights::Weights;
use omni_r2t2::{loading::CancellableWeights, weights::WeightSource};
use std::{path::Path, sync::atomic::AtomicBool};

pub struct Features<B: Backend> {
    pub embedding: Tensor<B, 3>,
    pub deepstack: Vec<Tensor<B, 3>>,
    pub height: usize,
    pub width: usize,
}

struct Block<B: Backend> {
    norm1: Norm<B>,
    norm2: Norm<B>,
    qkv: Linear<B>,
    out: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: Backend> Block<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        c: &VisionConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let linear = |s: &str, i, o| Linear::load(w, &format!("{name}.{s}"), i, o, true, p, d);
        let norm = |s: &str| Norm::load(w, &format!("{name}.{s}"), h, 1e-6, true, false, d);
        Ok(Self {
            norm1: norm("norm1")?,
            norm2: norm("norm2")?,
            qkv: linear("attn.qkv", h, 3 * h)?,
            out: linear("attn.proj", h, h)?,
            up: linear("mlp.linear_fc1", h, c.intermediate_size)?,
            down: linear("mlp.linear_fc2", c.intermediate_size, h)?,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &Rotary<B>,
        heads: usize,
        p: Precision,
    ) -> Tensor<B, 3> {
        let [b, n, h] = x.dims();
        let qkv = self.qkv.forward(self.norm1.forward(x.clone(), p));
        let part = |i| {
            qkv.clone()
                .slice([0..b, 0..n, i * h..(i + 1) * h])
                .reshape([b, n, heads, h / heads])
                .swap_dims(1, 2)
        };
        // Vision RoPE multiplies/accumulates in f32, then casts back to model dtype.
        let q = rope.apply(part(0).cast(FloatDType::F32), 0).cast(p.dtype());
        let k = rope.apply(part(1).cast(FloatDType::F32), 0).cast(p.dtype());
        let y = attention(q, k, part(2), false, 0)
            .swap_dims(1, 2)
            .reshape([b, n, h]);
        let x = residual(x, self.out.forward(y), p);
        let y = self
            .down
            .forward(gelu(self.up.forward(self.norm2.forward(x.clone(), p)), p));
        residual(x, y, p)
    }
}

struct Merger<B: Backend> {
    norm: Norm<B>,
    up: Linear<B>,
    down: Linear<B>,
    postshuffle: bool,
}
impl<B: Backend> Merger<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        c: &VisionConfig,
        postshuffle: bool,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let h = c.hidden_size * 4;
        Ok(Self {
            norm: Norm::load(
                w,
                &format!("{name}.norm"),
                if postshuffle { h } else { c.hidden_size },
                1e-6,
                true,
                false,
                d,
            )?,
            up: Linear::load(w, &format!("{name}.linear_fc1"), h, h, true, p, d)?,
            down: Linear::load(
                w,
                &format!("{name}.linear_fc2"),
                h,
                c.out_hidden_size,
                true,
                p,
                d,
            )?,
            postshuffle,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, p: Precision) -> Tensor<B, 3> {
        let [b, n, h] = x.dims();
        let x = if self.postshuffle {
            self.norm.forward(x.reshape([b, n / 4, h * 4]), p)
        } else {
            self.norm.forward(x, p).reshape([b, n / 4, h * 4])
        };
        let x = self.up.forward(x).cast(FloatDType::F32);
        // Merger uses exact GELU, unlike the vision blocks' tanh GELU.
        let y = x.clone() * 0.5 * ((x / std::f64::consts::SQRT_2).erf() + 1.0);
        self.down.forward(y.cast(p.dtype()))
    }
}

pub struct Vision<B: Backend> {
    patch: Linear<B>,
    positions: Tensor<B, 2>,
    blocks: Vec<Block<B>>,
    merger: Merger<B>,
    deepstack: Vec<Merger<B>>,
    config: VisionConfig,
    p: Precision,
    device: B::Device,
}
impl<B: Backend> Vision<B> {
    pub fn load(
        root: &Path,
        c: VisionConfig,
        p: Precision,
        d: &B::Device,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let weights = Weights::open(&root.join("text_encoder"))?;
        let w = CancellableWeights::new(&weights, cancel, |_: &str| {});
        let base = "model.visual";
        let patch_width = 3 * c.temporal_patch_size * c.patch_size * c.patch_size;
        let patch = Linear::from_parts(
            w.tensor(
                &format!("{base}.patch_embed.proj.weight"),
                [
                    c.hidden_size,
                    3,
                    c.temporal_patch_size,
                    c.patch_size,
                    c.patch_size,
                ],
                p,
                d,
            )?
            .reshape([c.hidden_size, patch_width]),
            w.tensor(
                &format!("{base}.patch_embed.proj.bias"),
                [c.hidden_size],
                p,
                d,
            )?,
        );
        let positions = w
            .tensor(
                &format!("{base}.pos_embed.weight"),
                [c.num_position_embeddings, c.hidden_size],
                p,
                d,
            )?
            .cast(FloatDType::F32);
        let mut blocks = Vec::new();
        for i in 0..c.depth {
            check_cancel(cancel)?;
            blocks.push(Block::load(&w, &format!("{base}.blocks.{i}"), &c, p, d)?);
            B::sync(d).map_err(|e| Error::Config(e.to_string()))?;
            progress(Progress {
                stage: "load_vision",
                current: i + 1,
                total: c.depth + 1,
            });
        }
        let merger = Merger::load(&w, &format!("{base}.merger"), &c, false, p, d)?;
        let deepstack = (0..c.deepstack_visual_indexes.len())
            .map(|i| {
                Merger::load(
                    &w,
                    &format!("{base}.deepstack_merger_list.{i}"),
                    &c,
                    true,
                    p,
                    d,
                )
            })
            .collect::<Result<_>>()?;
        B::sync(d).map_err(|e| Error::Config(e.to_string()))?;
        progress(Progress {
            stage: "load_vision",
            current: c.depth + 1,
            total: c.depth + 1,
        });
        Ok(Self {
            patch,
            positions,
            blocks,
            merger,
            deepstack,
            config: c,
            p,
            device: d.clone(),
        })
    }
    pub fn encode(
        &self,
        patches: Tensor<B, 3>,
        height: usize,
        width: usize,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
        trace: &mut impl FnMut(&str, Tensor<B, 3>) -> Result<()>,
    ) -> Result<Features<B>> {
        let c = &self.config;
        if height == 0
            || width == 0
            || height % 2 != 0
            || width % 2 != 0
            || patches.dims() != [1, height * width, 1536]
        {
            return Err(Error::Config("视觉图片 patch 尺寸不匹配".into()));
        }
        check_cancel(cancel)?;
        let n = height * width;
        let side = (c.num_position_embeddings as f64).sqrt() as usize;
        let mut coords = Vec::with_capacity(n);
        let mut indices = Vec::<i32>::with_capacity(n * 4);
        let mut weights = Vec::<f32>::with_capacity(n * 4);
        for gy in 0..height / 2 {
            for gx in 0..width / 2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let y = gy * 2 + dy;
                        let x = gx * 2 + dx;
                        coords.push([y as f64, x as f64, 0.0]);
                        let sy = (y * (side - 1)) as f32 / (height - 1) as f32;
                        let sx = (x * (side - 1)) as f32 / (width - 1) as f32;
                        let iy = sy.floor() as usize;
                        let ix = sx.floor() as usize;
                        let fy = sy - iy as f32;
                        let fx = sx - ix as f32;
                        for (yy, wy) in [(iy, 1.0 - fy), ((iy + 1).min(side - 1), fy)] {
                            for (xx, wx) in [(ix, 1.0 - fx), ((ix + 1).min(side - 1), fx)] {
                                indices.push((yy * side + xx) as i32);
                                weights.push(wy * wx);
                            }
                        }
                    }
                }
            }
        }
        let ids = Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [n * 4]), &self.device);
        let weights = Tensor::<B, 3>::from_data(TensorData::new(weights, [n, 4, 1]), &self.device);
        let pos = (self
            .positions
            .clone()
            .select(0, ids)
            .reshape([n, 4, c.hidden_size])
            * weights)
            .sum_dim(1)
            .reshape([1, n, c.hidden_size])
            .cast(self.p.dtype());
        let patch = self.patch.forward(patches);
        trace("patch", patch.clone())?;
        let mut x = residual(patch, pos, self.p);
        trace("position", x.clone())?;
        // Prevent a gather/reduction from fusing across the transformer boundary.
        B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
        let head = c.hidden_size / c.num_heads;
        let rope = Rotary::new(
            &coords,
            &[head / 2, head / 2],
            10000.0,
            false,
            Precision::F32,
            &self.device,
        );
        let mut deepstack = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            check_cancel(cancel)?;
            x = block.forward(x, &rope, c.num_heads, self.p);
            if let Some(j) = c.deepstack_visual_indexes.iter().position(|v| *v == i) {
                let feature = self.deepstack[j].forward(x.clone(), self.p);
                trace(&format!("deepstack_{j}"), feature.clone())?;
                deepstack.push(feature);
            }
            if [0, 8, 16, 24, 26].contains(&i) {
                trace(&format!("vision_layer_{i}"), x.clone())?;
            }
            if (i + 1) % 3 == 0 || i + 1 == self.blocks.len() {
                B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
                progress(Progress {
                    stage: "vision",
                    current: i + 1,
                    total: self.blocks.len(),
                });
            }
        }
        let embedding = self.merger.forward(x, self.p);
        trace("vision", embedding.clone())?;
        Ok(Features {
            embedding,
            deepstack,
            height: height / 2,
            width: width / 2,
        })
    }
}
