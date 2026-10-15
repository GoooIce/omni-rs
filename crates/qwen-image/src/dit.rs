//! Single-stream DiT with a causal, step-independent conditioning prefix.
use crate::{Error, Precision, Progress, Result, check_cancel, config::DitConfig, ops::*};
use burn::tensor::{Tensor, TensorData, activation::silu, backend::Backend};
use omni_r2t2::weights::Weights;
use omni_r2t2::{loading::CancellableWeights, weights::WeightSource};
use std::{path::Path, sync::atomic::AtomicBool};

pub struct ReferenceLatent<B: Backend> {
    pub latents: Tensor<B, 3>,
    pub height: usize,
    pub width: usize,
}

/// Runs in the interleaved conditioning sequence; each image is bidirectional
/// internally, text is causal, and every run can attend to all preceding runs.
#[derive(Debug, PartialEq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    pub text: bool,
}

fn positions_and_segments(
    text_len: usize,
    spans: &[crate::text::ImageSpan],
    height: usize,
    width: usize,
) -> Result<(Vec<[f64; 3]>, Vec<Segment>, usize)> {
    let mut positions = Vec::new();
    let mut segments = Vec::new();
    let mut cursor = 0;
    let mut position = 0;
    for span in spans {
        if span.start < cursor
            || span.end <= span.start
            || span.end > text_len
            || span.end - span.start != span.height * span.width
        {
            return Err(Error::Config("DiT 图片条件布局不匹配".into()));
        }
        let count = span.start - cursor;
        if count > 0 {
            let start = positions.len();
            positions.extend((position..position + count).map(|p| [p as f64; 3]));
            segments.push(Segment {
                start,
                end: positions.len(),
                text: true,
            });
            position += count;
        }
        let start = positions.len();
        let h = span.height * 2;
        let w = span.width * 2;
        for y in 0..h {
            for x in 0..w {
                positions.push([
                    position as f64,
                    y as f64 - h.div_ceil(2) as f64,
                    x as f64 - w.div_ceil(2) as f64,
                ]);
            }
        }
        segments.push(Segment {
            start,
            end: positions.len(),
            text: false,
        });
        position += h.max(w);
        cursor = span.end;
    }
    if cursor < text_len {
        let start = positions.len();
        positions.extend((position..position + text_len - cursor).map(|p| [p as f64; 3]));
        segments.push(Segment {
            start,
            end: positions.len(),
            text: true,
        });
        position += text_len - cursor;
    }
    let prefix = positions.len();
    for y in 0..height {
        for x in 0..width {
            positions.push([
                position as f64,
                y as f64 - height.div_ceil(2) as f64,
                x as f64 - width.div_ceil(2) as f64,
            ]);
        }
    }
    Ok((positions, segments, prefix))
}

type LayerCache<B> = omni_r2t2::tensor::KvCache<B>;
pub struct Cache<B: Backend> {
    layers: Vec<Option<LayerCache<B>>>,
}
impl<B: Backend> Cache<B> {
    pub fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| None).collect(),
        }
    }
}

struct Block<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: Backend> Block<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        c: &DitConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let h = c.width();
        let linear = |s: &str, i, o| {
            Linear::load(w, &format!("{name}.{s}"), i, o, false, p, d).map(Linear::with_tensorops)
        };
        Ok(Self {
            q: linear("attn.to_q", h, h)?,
            k: linear("attn.to_k", h, h)?,
            v: linear("attn.to_v", h, h)?,
            out: linear("attn.to_out.0", h, h)?,
            q_norm: Norm::load(
                w,
                &format!("{name}.attn.norm_q"),
                c.attention_head_dim,
                c.eps,
                false,
                false,
                d,
            )?,
            k_norm: Norm::load(
                w,
                &format!("{name}.attn.norm_k"),
                c.attention_head_dim,
                c.eps,
                false,
                false,
                d,
            )?,
            gate: linear("img_mlp.gate_layer", h, h * c.mlp_ratio)?,
            up: linear("img_mlp.proj", h, h * c.mlp_ratio)?,
            down: linear("img_mlp.out", h * c.mlp_ratio, h)?,
        })
    }
    fn cached_attention(
        &self,
        qkv: [Tensor<B, 4>; 3],
        prefix: usize,
        segments: &[Segment],
        cache: &mut Option<LayerCache<B>>,
        c: &DitConfig,
        decoding: bool,
    ) -> Tensor<B, 3> {
        let [q, k, v] = qkv;
        let [b, _, n, _] = q.dims();
        let h = c.width();
        let (k, v) = if let Some(previous) = cache {
            previous.concat(k, v, previous.len())
        } else {
            *cache = Some(LayerCache::copy_prefix(&k, &v, prefix));
            (k, v)
        };
        if decoding {
            dit_attention(q, k, v, false, 0)
        } else {
            let mut outputs = Vec::with_capacity(segments.len() + 1);
            for segment in segments {
                let range = [
                    0..b,
                    0..c.num_attention_heads,
                    0..segment.end,
                    0..c.attention_head_dim,
                ];
                outputs.push(dit_attention(
                    q.clone().slice([
                        0..b,
                        0..c.num_attention_heads,
                        segment.start..segment.end,
                        0..c.attention_head_dim,
                    ]),
                    k.clone().slice(range.clone()),
                    v.clone().slice(range),
                    segment.text,
                    segment.start,
                ));
            }
            let target = q.slice([
                0..b,
                0..c.num_attention_heads,
                prefix..n,
                0..c.attention_head_dim,
            ]);
            outputs.push(dit_attention(target, k, v, false, 0));
            Tensor::cat(outputs, 2)
        }
        .swap_dims(1, 2)
        .reshape([b, n, h])
    }
    fn forward(
        &self,
        x: Tensor<B, 3>,
        modulation: &[Tensor<B, 3>; 4],
        rope: &Rotary<B>,
        prefix: usize,
        segments: &[Segment],
        cache: &mut Option<LayerCache<B>>,
        c: &DitConfig,
        p: Precision,
    ) -> Tensor<B, 3> {
        let [b, n, _] = x.dims();
        let decoding = cache.is_some();
        let y = layer_norm(x.clone(), c.eps, p) * (modulation[0].clone() + 1.0);
        let reshape =
            |y: Tensor<B, 3>| y.reshape([b, n, c.num_attention_heads, c.attention_head_dim]);
        let q = self
            .q_norm
            .forward(reshape(self.q.forward(y.clone())), p)
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(reshape(self.k.forward(y.clone())), p)
            .swap_dims(1, 2);
        let v = reshape(self.v.forward(y)).swap_dims(1, 2);
        let q = rope.apply(q, if decoding { prefix } else { 0 });
        let k = rope.apply(k, if decoding { prefix } else { 0 });
        let y = self.cached_attention([q, k, v], prefix, segments, cache, c, decoding);
        let x = residual(x, modulation[1].clone().tanh() * self.out.forward(y), p);
        let y = layer_norm(x.clone(), c.eps, p) * (modulation[2].clone() + 1.0);
        let y = silu(self.gate.forward(y.clone())) * self.up.forward(y);
        let x = residual(x, modulation[3].clone().tanh() * self.down.forward(y), p);
        if p == Precision::F16 {
            omni_r2t2::tensor::clamp_f16(x, omni_r2t2::tensor::Fp16Range::Finite)
        } else {
            x
        }
    }
    fn forward_many(
        &self,
        frames: &mut [Prepared<B>],
        histories: &mut [&mut Cache<B>],
        layer: usize,
        c: &DitConfig,
        p: Precision,
    ) {
        let h = c.width();
        let n: usize = frames.iter().map(|f| f.x.dims()[1]).sum();
        let x = Tensor::cat(frames.iter().map(|f| f.x.clone()).collect(), 1);
        let mods: [Tensor<B, 3>; 4] = std::array::from_fn(|i| {
            Tensor::cat(
                frames
                    .iter()
                    .map(|f| {
                        let value = f.mods[i].clone();
                        if value.dims()[1] == 1 {
                            value.repeat_dim(1, f.x.dims()[1])
                        } else {
                            value
                        }
                    })
                    .collect(),
                1,
            )
        });
        let y = layer_norm(x.clone(), c.eps, p) * (mods[0].clone() + 1.0);
        let reshape =
            |y: Tensor<B, 3>| y.reshape([1, n, c.num_attention_heads, c.attention_head_dim]);
        let q = self
            .q_norm
            .forward(reshape(self.q.forward(y.clone())), p)
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(reshape(self.k.forward(y.clone())), p)
            .swap_dims(1, 2);
        let v = reshape(self.v.forward(y)).swap_dims(1, 2);
        let mut start = 0;
        let mut attended = Vec::new();
        for (frame, history) in frames.iter().zip(histories.iter_mut()) {
            let end = start + frame.x.dims()[1];
            let range = [
                0..1,
                0..c.num_attention_heads,
                start..end,
                0..c.attention_head_dim,
            ];
            let offset = if frame.decoding { frame.prefix } else { 0 };
            attended.push(self.cached_attention(
                [
                    frame.rope.apply(q.clone().slice(range.clone()), offset),
                    frame.rope.apply(k.clone().slice(range.clone()), offset),
                    v.clone().slice(range),
                ],
                frame.prefix,
                &frame.segments,
                &mut history.layers[layer],
                c,
                frame.decoding,
            ));
            start = end;
        }
        let y = Tensor::cat(attended, 1);
        let x = residual(x, mods[1].clone().tanh() * self.out.forward(y), p);
        let y = layer_norm(x.clone(), c.eps, p) * (mods[2].clone() + 1.0);
        let y = silu(self.gate.forward(y.clone())) * self.up.forward(y);
        let mut x = residual(x, mods[3].clone().tanh() * self.down.forward(y), p);
        if p == Precision::F16 {
            x = omni_r2t2::tensor::clamp_f16(x, omni_r2t2::tensor::Fp16Range::Finite);
        }
        let mut start = 0;
        for frame in frames {
            let end = start + frame.x.dims()[1];
            frame.x = x.clone().slice([0..1, start..end, 0..h]);
            start = end;
        }
    }
}

pub struct Dit<B: Backend> {
    config: DitConfig,
    text_norm: Norm<B>,
    text_in: Linear<B>,
    text_out: Linear<B>,
    image_in: Linear<B>,
    time_in: Linear<B>,
    time_out: Linear<B>,
    modulation: Linear<B>,
    final_modulation: Linear<B>,
    out: Linear<B>,
    layers: Vec<Block<B>>,
    p: Precision,
    device: B::Device,
}
impl<B: Backend> Dit<B> {
    pub fn load(
        root: &Path,
        c: DitConfig,
        p: Precision,
        d: &B::Device,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let weights = Weights::open_named(&root.join("transformer"), "diffusion_pytorch_model")?;
        let w = CancellableWeights::new(&weights, cancel, |_: &str| {});
        let h = c.width();
        let linear = |s: &str, i, o| Linear::load(&w, s, i, o, false, p, d);
        let text_norm = Norm::load(
            &w,
            "txt_in.text_norm",
            c.context_in_dim,
            c.eps,
            false,
            true,
            d,
        )?;
        let text_in = linear("txt_in.in_layer", c.context_in_dim, h)?;
        let text_out = linear("txt_in.out_layer", h, h)?;
        let image_in = linear("img_in", c.in_channels, h)?;
        let time_in = linear("time_text_embed.timestep_embedder.linear_1", 256, h)?;
        let time_out = linear("time_text_embed.timestep_embedder.linear_2", h, h)?;
        let modulation = linear("modulation.1", h, 4 * h)?;
        let final_modulation = linear("norm_out.linear", h, h)?;
        let out = linear("proj_out", h, c.out_channels)?;
        let mut layers = Vec::new();
        for i in 0..c.num_layers {
            check_cancel(cancel)?;
            layers.push(Block::load(
                &w,
                &format!("transformer_blocks.{i}"),
                &c,
                p,
                d,
            )?);
            B::sync(d).map_err(|e| Error::Config(e.to_string()))?;
            progress(Progress {
                stage: "load_dit",
                current: i + 1,
                total: c.num_layers,
            });
        }
        Ok(Self {
            config: c,
            text_norm,
            text_in,
            text_out,
            image_in,
            time_in,
            time_out,
            modulation,
            final_modulation,
            out,
            layers,
            p,
            device: d.clone(),
        })
    }

    fn time_embedding(&self, sigma: f32) -> Tensor<B, 2> {
        // Mirror the pipeline's timestep -> model dtype -> /1000 boundary.
        let t = if self.p == Precision::F16 {
            burn::tensor::f16::from_f32(
                burn::tensor::f16::from_f32(sigma * 1000.0).to_f32() / 1000.0,
            )
            .to_f32()
        } else {
            sigma * 1000.0 / 1000.0
        };
        let mut values = Vec::with_capacity(512);
        for t in [t, 0.0] {
            for sin in [false, true] {
                for i in 0..128 {
                    let angle = (t * 1000.0) as f64 * 10000_f64.powf(-(i as f64) / 128.0);
                    values.push(if sin {
                        angle.sin() as f32
                    } else {
                        angle.cos() as f32
                    });
                }
            }
        }
        let x = Tensor::<B, 2>::from_data(TensorData::new(values, [2, 256]), &self.device)
            .cast(self.p.dtype());
        self.time_out.forward(silu(self.time_in.forward(x)))
    }

    fn select_modulation(
        &self,
        x: Tensor<B, 2>,
        prefix: usize,
        target: usize,
        decoding: bool,
    ) -> Tensor<B, 3> {
        let width = x.dims()[1];
        let real = x.clone().slice([0..1, 0..width]).reshape([1, 1, width]);
        if decoding {
            return real;
        }
        let zero = x.slice([1..2, 0..width]).reshape([1, 1, width]);
        Tensor::cat(
            vec![zero.repeat_dim(1, prefix), real.repeat_dim(1, target)],
            1,
        )
    }

    pub fn predict(
        &self,
        latents: Tensor<B, 3>,
        text: Tensor<B, 3>,
        sigma: f32,
        height: usize,
        width: usize,
        cache: &mut Cache<B>,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Tensor<B, 3>> {
        self.predict_with_images(
            latents,
            text,
            &[],
            &[],
            sigma,
            height,
            width,
            cache,
            cancel,
            progress,
        )
    }

    pub fn predict_with_images(
        &self,
        latents: Tensor<B, 3>,
        text: Tensor<B, 3>,
        references: &[ReferenceLatent<B>],
        spans: &[crate::text::ImageSpan],
        sigma: f32,
        height: usize,
        width: usize,
        cache: &mut Cache<B>,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Tensor<B, 3>> {
        let mut frame = self.prepare(&Prediction {
            latents,
            text,
            references,
            spans,
            sigma,
            height,
            width,
            cache,
            cancel,
        })?;
        for (i, layer) in self.layers.iter().enumerate() {
            check_cancel(cancel)?;
            frame.x = layer.forward(
                frame.x,
                &frame.mods,
                &frame.rope,
                frame.prefix,
                &frame.segments,
                &mut cache.layers[i],
                &self.config,
                self.p,
            );
            if (i + 1) % 4 == 0 || i + 1 == self.layers.len() {
                B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
                progress(Progress {
                    stage: "denoise_layer",
                    current: i + 1,
                    total: self.layers.len(),
                });
            }
        }
        Ok(self.finish(frame))
    }
    fn prepare(&self, item: &Prediction<'_, B>) -> Result<Prepared<B>> {
        let latents = item.latents.clone();
        let text = item.text.clone();
        let (references, spans) = (item.references, item.spans);
        let (sigma, height, width) = (item.sigma, item.height, item.width);
        let cache = &*item.cache;
        let h = self.config.width();
        let text_len = text.dims()[1];
        let (positions, segments, prefix) = positions_and_segments(text_len, spans, height, width)?;
        if references.len() != spans.len() {
            return Err(Error::Config("DiT 参考图数量不匹配".into()));
        }
        let target = latents.dims()[1];
        if target != height * width || cache.layers.len() != self.layers.len() {
            return Err(Error::Config("无效的图像潜变量或 KV 缓存".into()));
        }
        let decoding = cache.layers.first().is_some_and(|v| v.is_some());
        let image = self.image_in.forward(latents);
        let x = if decoding {
            image
        } else {
            let text = self.text_norm.forward(text, self.p);
            let text = self
                .text_out
                .forward(gelu(self.text_in.forward(text), self.p));
            let mut pieces = Vec::with_capacity(references.len() * 2 + 2);
            let mut cursor = 0;
            for (reference, span) in references.iter().zip(spans) {
                if reference.height != span.height * 2
                    || reference.width != span.width * 2
                    || reference.latents.dims()
                        != [
                            1,
                            reference.height * reference.width,
                            self.config.in_channels,
                        ]
                {
                    return Err(Error::Config("参考图的 VAE 与视觉编码器网格不匹配".into()));
                }
                if span.start > cursor {
                    pieces.push(text.clone().slice([0..1, cursor..span.start, 0..h]));
                }
                pieces.push(self.image_in.forward(reference.latents.clone()));
                cursor = span.end;
            }
            if cursor < text_len {
                pieces.push(text.slice([0..1, cursor..text_len, 0..h]));
            }
            pieces.push(image);
            Tensor::cat(pieces, 1)
        };
        let rope = Rotary::new(
            &positions,
            &self.config.axes_dims_rope,
            10000.0,
            true,
            self.p,
            &self.device,
        );
        let temb = self.time_embedding(sigma);
        let modulation = self.modulation.forward(silu(temb.clone()));
        let mods: [Tensor<B, 3>; 4] = std::array::from_fn(|i| {
            self.select_modulation(
                modulation.clone().slice([0..2, i * h..(i + 1) * h]),
                prefix,
                target,
                decoding,
            )
        });
        let scale = self
            .final_modulation
            .forward(silu(temb))
            .slice([0..1, 0..h])
            .reshape([1, 1, h]);
        Ok(Prepared {
            x,
            mods,
            rope,
            prefix,
            segments,
            decoding,
            target,
            scale,
        })
    }
    fn finish(&self, mut frame: Prepared<B>) -> Tensor<B, 3> {
        if !frame.decoding {
            frame.x = frame.x.slice([
                0..1,
                frame.prefix..frame.prefix + frame.target,
                0..self.config.width(),
            ]);
        }
        self.out
            .forward(layer_norm(frame.x, self.config.eps, self.p) * (frame.scale + 1.0))
    }
    pub(crate) fn predict_many(
        &self,
        items: &mut [Prediction<'_, B>],
        progress: &mut impl FnMut(Progress),
    ) -> Result<Vec<Tensor<B, 3>>> {
        if items.len() == 1 {
            let item = &mut items[0];
            return self
                .predict_with_images(
                    item.latents.clone(),
                    item.text.clone(),
                    item.references,
                    item.spans,
                    item.sigma,
                    item.height,
                    item.width,
                    item.cache,
                    item.cancel,
                    progress,
                )
                .map(|value| vec![value]);
        }
        let mut frames = Vec::new();
        for item in items.iter() {
            frames.push(self.prepare(item)?);
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if items
                .iter()
                .all(|item| item.cancel.load(std::sync::atomic::Ordering::Acquire))
            {
                return Err(Error::Cancelled);
            }
            let mut caches: Vec<_> = items.iter_mut().map(|item| &mut *item.cache).collect();
            layer.forward_many(&mut frames, &mut caches, i, &self.config, self.p);
            if (i + 1) % 4 == 0 || i + 1 == self.layers.len() {
                B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
                progress(Progress {
                    stage: "denoise_layer",
                    current: i + 1,
                    total: self.layers.len(),
                });
            }
        }
        Ok(frames.into_iter().map(|frame| self.finish(frame)).collect())
    }
}

struct Prepared<B: Backend> {
    x: Tensor<B, 3>,
    mods: [Tensor<B, 3>; 4],
    rope: Rotary<B>,
    prefix: usize,
    segments: Vec<Segment>,
    decoding: bool,
    target: usize,
    scale: Tensor<B, 3>,
}
pub(crate) struct Prediction<'a, B: Backend> {
    pub latents: Tensor<B, 3>,
    pub text: Tensor<B, 3>,
    pub references: &'a [ReferenceLatent<B>],
    pub spans: &'a [crate::text::ImageSpan],
    pub sigma: f32,
    pub height: usize,
    pub width: usize,
    pub cache: &'a mut Cache<B>,
    pub cancel: &'a AtomicBool,
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    #[test]
    fn references_expand_slots_and_advance_frame_positions_by_spatial_extent() {
        let spans = [
            crate::text::ImageSpan {
                start: 4,
                end: 10,
                height: 2,
                width: 3,
            },
            crate::text::ImageSpan {
                start: 14,
                end: 20,
                height: 3,
                width: 2,
            },
        ];
        let (p, segments, prefix) = positions_and_segments(27, &spans, 8, 8).unwrap();
        assert_eq!(prefix, 63);
        assert_eq!(
            segments
                .iter()
                .map(|s| (s.start, s.end, s.text))
                .collect::<Vec<_>>(),
            vec![
                (0, 4, true),
                (4, 28, false),
                (28, 32, true),
                (32, 56, false),
                (56, 63, true)
            ]
        );
        assert_eq!(p[4], [4.0, -2.0, -3.0]);
        assert_eq!(p[28], [10.0; 3]);
        assert_eq!(p[32], [14.0, -3.0, -2.0]);
        assert_eq!(p[56], [20.0; 3]);
        assert_eq!(p[63], [27.0, -4.0, -4.0]);
        let (p, _, prefix) = positions_and_segments(19, &[], 8, 8).unwrap();
        assert_eq!(prefix, 19);
        assert_eq!(p[19], [19.0, -4.0, -4.0]);
    }
}
