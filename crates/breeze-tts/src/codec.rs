//! Stateful Qwen3-TTS 12 Hz decoder. Each call consumes only new codec frames.
//! Causal convolution history and the 72-frame attention window are bounded.
use crate::{Error, Precision, Result, config::*, ops::*, transformer::*};
use burn::tensor::{
    FloatDType, Int, Tensor, TensorData,
    backend::Backend,
    module,
    ops::{ConvOptions, ConvTransposeOptions},
};
use omni_r2t2::weights::{WeightSource, Weights};
use std::path::Path;

pub struct CodecState<B: Backend> {
    conv: Vec<Option<Tensor<B, 3>>>,
    transformer: TransformerCache<B>,
}

struct Conv<B: Backend> {
    weight: Tensor<B, 3>,
    linear_weight: Option<Tensor<B, 2>>,
    bias: Option<Tensor<B, 1>>,
    kernel: usize,
    dilation: usize,
    groups: usize,
    stride: usize,
    transpose: bool,
    index: usize,
}
impl<B: Backend> Conv<B> {
    #[allow(clippy::too_many_arguments)]
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        stride: usize,
        transpose: bool,
        bias: bool,
        next: &mut usize,
        d: &B::Device,
    ) -> Result<Self> {
        let shape = if transpose {
            [input, output / groups, kernel]
        } else {
            [output, input / groups, kernel]
        };
        let index = *next;
        *next += 1;
        let weight: Tensor<B, 3> = w.tensor(&format!("{name}.weight"), shape, Precision::F32, d)?;
        let linear_weight = if !transpose && groups == 1 {
            Some(
                weight
                    .clone()
                    .swap_dims(1, 2)
                    .reshape([output, kernel * input])
                    .transpose(),
            )
        } else {
            None
        };
        Ok(Self {
            weight,
            linear_weight,
            bias: if bias {
                Some(w.tensor(&format!("{name}.bias"), [output], Precision::F32, d)?)
            } else {
                None
            },
            kernel,
            dilation,
            groups,
            stride,
            transpose,
            index,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, state: &mut CodecState<B>) -> Tensor<B, 3> {
        let [b, c, n] = x.dims();
        let context = if self.transpose {
            (self.kernel - 1) / self.stride
        } else {
            (self.kernel - 1) * self.dilation
        };
        let x = if context > 0 {
            let previous = state.conv[self.index]
                .take()
                .unwrap_or_else(|| Tensor::zeros([b, c, context], &x.device()));
            let x = Tensor::cat(vec![previous, x], 2);
            let total = context + n;
            state.conv[self.index] = Some(x.clone().slice([0..b, 0..c, total - context..total]));
            x
        } else {
            x
        };
        if self.transpose {
            let y = module::conv_transpose1d(
                x,
                self.weight.clone(),
                self.bias.clone(),
                ConvTransposeOptions::new([self.stride], [0], [0], [1], self.groups),
            );
            let out = y.dims()[1];
            let start = context * self.stride;
            y.slice([0..b, 0..out, start..start + n * self.stride])
        } else if self.groups == 1 && (n <= 32 || self.kernel == 1) {
            // Burn 0.21 convolution tuning buckets together lengths on opposite
            // sides of Metal's CMMA availability boundary. A cached CMMA choice
            // can panic for the final one-frame chunk. Small convolutions and
            // pointwise projections are exactly an unfold followed by matmul.
            // This keeps the workaround local to the codec, and reuses weights.
            let columns = (0..self.kernel)
                .map(|k| {
                    let start = k * self.dilation;
                    x.clone().slice([0..b, 0..c, start..start + n])
                })
                .collect();
            let columns = Tensor::cat(columns, 1).swap_dims(1, 2);
            module::linear(
                columns,
                self.linear_weight.as_ref().unwrap().clone(),
                self.bias.clone(),
            )
            .swap_dims(1, 2)
        } else {
            module::conv1d(
                x,
                self.weight.clone(),
                self.bias.clone(),
                ConvOptions::new([1], [0], [self.dilation], self.groups),
            )
        }
    }
}

struct Snake<B: Backend> {
    alpha: Tensor<B, 3>,
    beta: Tensor<B, 3>,
}
impl<B: Backend> Snake<B> {
    fn load(w: &impl WeightSource<B>, name: &str, c: usize, d: &B::Device) -> Result<Self> {
        Ok(Self {
            alpha: w
                .tensor(&format!("{name}.alpha"), [c], Precision::F32, d)?
                .exp()
                .reshape([1, c, 1]),
            beta: (w
                .tensor(&format!("{name}.beta"), [c], Precision::F32, d)?
                .exp()
                + 1e-9)
                .recip()
                .reshape([1, c, 1]),
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x.clone() + (x * self.alpha.clone()).sin().square() * self.beta.clone()
    }
}

struct ConvNext<B: Backend> {
    dw: Conv<B>,
    norm: Norm<B>,
    up: Linear<B>,
    down: Linear<B>,
    gamma: Tensor<B, 1>,
}
impl<B: Backend> ConvNext<B> {
    fn load(
        w: &impl WeightSource<B>,
        base: &str,
        c: usize,
        next: &mut usize,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            dw: Conv::load(
                w,
                &format!("{base}.dwconv.conv"),
                c,
                c,
                7,
                1,
                c,
                1,
                false,
                true,
                next,
                d,
            )?,
            norm: Norm::load(w, &format!("{base}.norm"), c, 1e-6, NormKind::Layer, d)?,
            up: Linear::load(
                w,
                &format!("{base}.pwconv1"),
                c,
                4 * c,
                true,
                Precision::F32,
                d,
            )?,
            down: Linear::load(
                w,
                &format!("{base}.pwconv2"),
                4 * c,
                c,
                true,
                Precision::F32,
                d,
            )?,
            gamma: w.tensor(&format!("{base}.gamma"), [c], Precision::F32, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, state: &mut CodecState<B>) -> Tensor<B, 3> {
        let y = self.dw.forward(x.clone(), state).swap_dims(1, 2);
        let y = self.norm.forward(y);
        let y = self.down.forward(gelu(self.up.forward(y), false))
            * self.gamma.clone().unsqueeze::<3>();
        residual(x, y.swap_dims(1, 2))
    }
}

struct ResidualUnit<B: Backend> {
    act1: Snake<B>,
    conv1: Conv<B>,
    act2: Snake<B>,
    conv2: Conv<B>,
}
impl<B: Backend> ResidualUnit<B> {
    fn load(
        w: &impl WeightSource<B>,
        base: &str,
        c: usize,
        dilation: usize,
        next: &mut usize,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            act1: Snake::load(w, &format!("{base}.act1"), c, d)?,
            conv1: Conv::load(
                w,
                &format!("{base}.conv1.conv"),
                c,
                c,
                7,
                dilation,
                1,
                1,
                false,
                true,
                next,
                d,
            )?,
            act2: Snake::load(w, &format!("{base}.act2"), c, d)?,
            conv2: Conv::load(
                w,
                &format!("{base}.conv2.conv"),
                c,
                c,
                1,
                1,
                1,
                1,
                false,
                true,
                next,
                d,
            )?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, s: &mut CodecState<B>) -> Tensor<B, 3> {
        let y = self.conv1.forward(self.act1.forward(x.clone()), s);
        residual(x, self.conv2.forward(self.act2.forward(y), s))
    }
}

struct DecoderBlock<B: Backend> {
    act: Snake<B>,
    up: Conv<B>,
    residuals: Vec<ResidualUnit<B>>,
}
impl<B: Backend> DecoderBlock<B> {
    fn load(
        w: &impl WeightSource<B>,
        base: &str,
        input: usize,
        stride: usize,
        next: &mut usize,
        d: &B::Device,
    ) -> Result<Self> {
        let mut residuals = Vec::new();
        for (i, dilation) in [1, 3, 9].into_iter().enumerate() {
            residuals.push(ResidualUnit::load(
                w,
                &format!("{base}.block.{}", i + 2),
                input / 2,
                dilation,
                next,
                d,
            )?);
        }
        Ok(Self {
            act: Snake::load(w, &format!("{base}.block.0"), input, d)?,
            up: Conv::load(
                w,
                &format!("{base}.block.1.conv"),
                input,
                input / 2,
                2 * stride,
                1,
                1,
                stride,
                true,
                true,
                next,
                d,
            )?,
            residuals,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, s: &mut CodecState<B>) -> Tensor<B, 3> {
        let mut x = self.up.forward(self.act.forward(x), s);
        for r in &self.residuals {
            x = r.forward(x, s);
        }
        x
    }
}

pub struct Codec<B: Backend> {
    codebooks: Vec<Tensor<B, 2>>,
    semantic_projection: Tensor<B, 2>,
    acoustic_projection: Tensor<B, 2>,
    pre_conv: Conv<B>,
    input: Linear<B>,
    transformer: Transformer<B>,
    output: Linear<B>,
    upsample: Vec<(Conv<B>, ConvNext<B>)>,
    initial: Conv<B>,
    decoder: Vec<DecoderBlock<B>>,
    act: Snake<B>,
    last: Conv<B>,
    conv_count: usize,
    device: B::Device,
}
impl<B: Backend> Codec<B> {
    pub fn load(directory: &Path, d: &B::Device) -> Result<Self> {
        Self::load_cancellable(directory, d, &std::sync::atomic::AtomicBool::new(false))
    }
    pub fn load_cancellable(
        directory: &Path,
        d: &B::Device,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Self> {
        use omni_r2t2::loading::{CancellableWeights, check};
        check(cancel)?;
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if value["model_type"] != "qwen3_tts_tokenizer_12hz"
            || value["output_sample_rate"] != 24000
            || value["decode_upsample_rate"] != 1920
        {
            return Err(Error::Config(
                "expected bundled Qwen3-TTS 12 Hz audio tokenizer".into(),
            ));
        }
        let config: TransformerConfig = serde_json::from_value(value["decoder_config"].clone())?;
        check(cancel)?;
        let weights = Weights::open(directory)?;
        let w = CancellableWeights::new(&weights, cancel, |_: &str| {});
        let mut next = 0;
        let mut codebooks = Vec::new();
        for i in 0..CODEBOOKS {
            let part = if i == 0 { "first" } else { "rest" };
            let layer = if i == 0 { 0 } else { i - 1 };
            let base = format!("decoder.quantizer.rvq_{part}.vq.layers.{layer}._codebook");
            let sum = w.tensor(
                &format!("{base}.embedding_sum"),
                [CODEBOOK_SIZE, 256],
                Precision::F32,
                d,
            )?;
            let usage = w
                .tensor(
                    &format!("{base}.cluster_usage"),
                    [CODEBOOK_SIZE],
                    Precision::F32,
                    d,
                )?
                .clamp_min(1e-5)
                .reshape([CODEBOOK_SIZE, 1]);
            codebooks.push(sum / usage);
        }
        let projection = |part: &str| -> Result<Tensor<B, 2>> {
            Ok(w.tensor(
                &format!("decoder.quantizer.rvq_{part}.output_proj.weight"),
                [512, 256, 1],
                Precision::F32,
                d,
            )?
            .reshape([512, 256])
            .transpose())
        };
        let pre_conv = Conv::load(
            &w,
            "decoder.pre_conv.conv",
            512,
            1024,
            3,
            1,
            1,
            1,
            false,
            true,
            &mut next,
            d,
        )?;
        let input = Linear::load(
            &w,
            "decoder.pre_transformer.input_proj",
            1024,
            512,
            true,
            Precision::F32,
            d,
        )?;
        let transformer = Transformer::load(
            &w,
            "decoder.pre_transformer",
            &config,
            Architecture::Codec,
            Precision::F32,
            d,
            &mut |_| {},
        )?;
        let output = Linear::load(
            &w,
            "decoder.pre_transformer.output_proj",
            512,
            1024,
            true,
            Precision::F32,
            d,
        )?;
        let mut upsample = Vec::new();
        for i in 0..2 {
            upsample.push((
                Conv::load(
                    &w,
                    &format!("decoder.upsample.{i}.0.conv"),
                    1024,
                    1024,
                    2,
                    1,
                    1,
                    2,
                    true,
                    true,
                    &mut next,
                    d,
                )?,
                ConvNext::load(&w, &format!("decoder.upsample.{i}.1"), 1024, &mut next, d)?,
            ));
        }
        let initial = Conv::load(
            &w,
            "decoder.decoder.0.conv",
            1024,
            1536,
            7,
            1,
            1,
            1,
            false,
            true,
            &mut next,
            d,
        )?;
        let mut decoder = Vec::new();
        for (i, stride) in [8, 5, 4, 3].into_iter().enumerate() {
            decoder.push(DecoderBlock::load(
                &w,
                &format!("decoder.decoder.{}", i + 1),
                1536 >> i,
                stride,
                &mut next,
                d,
            )?);
        }
        let act = Snake::load(&w, "decoder.decoder.5", 96, d)?;
        let last = Conv::load(
            &w,
            "decoder.decoder.6.conv",
            96,
            1,
            7,
            1,
            1,
            1,
            false,
            true,
            &mut next,
            d,
        )?;
        check(cancel)?;
        Ok(Self {
            codebooks,
            semantic_projection: projection("first")?,
            acoustic_projection: projection("rest")?,
            pre_conv,
            input,
            transformer,
            output,
            upsample,
            initial,
            decoder,
            act,
            last,
            conv_count: next,
            device: d.clone(),
        })
    }
    pub fn state(&self) -> CodecState<B> {
        CodecState {
            conv: (0..self.conv_count).map(|_| None).collect(),
            transformer: self.transformer.cache(),
        }
    }
    pub fn quantize(&self, codes: &[[u32; CODEBOOKS]]) -> Result<Tensor<B, 3>> {
        if codes.is_empty() || codes.iter().flatten().any(|&x| x as usize >= CODEBOOK_SIZE) {
            return Err(Error::Inference(
                "codec requires nonempty frames with IDs in 0..2048".into(),
            ));
        }
        let ids: Vec<i32> = codes.iter().flatten().map(|&x| x as i32).collect();
        Ok(self.lookup_tokens(Tensor::from_data(
            TensorData::new(ids, [codes.len(), CODEBOOKS]),
            &self.device,
        )))
    }
    // IDs come from the validated sampler: reserved tokens/EOS never enter RVQ.
    fn lookup_tokens(&self, codes: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [frames, books] = codes.dims();
        assert!(frames > 0 && books == CODEBOOKS);
        let embed = |i: usize| {
            self.codebooks[i].clone().select(
                0,
                codes.clone().slice([0..frames, i..i + 1]).reshape([frames]),
            )
        };
        let semantic = embed(0).matmul(self.semantic_projection.clone());
        let mut acoustic = embed(1);
        for i in 2..CODEBOOKS {
            acoustic = acoustic + embed(i);
        }
        (semantic + acoustic.matmul(self.acoustic_projection.clone()))
            .reshape([1, frames, 512])
            .swap_dims(1, 2)
    }
    pub fn decode_tensor(
        &self,
        codes: &[[u32; CODEBOOKS]],
        state: &mut CodecState<B>,
    ) -> Result<Tensor<B, 3>> {
        let x = self.quantize(codes)?;
        Ok(self.decode_embeddings(x, state))
    }
    fn decode_embeddings(&self, x: Tensor<B, 3>, state: &mut CodecState<B>) -> Tensor<B, 3> {
        // Streaming decoder chunks repeatedly alternate convolution, Snake and
        // transposed convolution. Measured boundary costs exceed ANE compute
        // savings; keep this whole dependent phase on the existing Metal path.
        let _routing = crate::routing::metal_phase::<B>(&self.device);
        let x = self
            .input
            .forward(self.pre_conv.forward(x, state).swap_dims(1, 2));
        let mut x = self
            .output
            .forward(self.transformer.forward(x, &mut state.transformer))
            .swap_dims(1, 2);
        for (up, block) in &self.upsample {
            x = up.forward(x, state);
            x = block.forward(x, state);
        }
        x = self.initial.forward(x, state);
        for block in &self.decoder {
            x = block.forward(x, state);
        }
        self.last
            .forward(self.act.forward(x), state)
            .clamp(-1.0, 1.0)
    }
    pub(crate) fn decode_device(
        &self,
        codes: Tensor<B, 2, Int>,
        state: &mut CodecState<B>,
    ) -> Result<Vec<f32>> {
        let output = self.decode_embeddings(self.lookup_tokens(codes), state);
        Self::read_audio(output)
    }
    pub fn decode(
        &self,
        codes: &[[u32; CODEBOOKS]],
        state: &mut CodecState<B>,
    ) -> Result<Vec<f32>> {
        Self::read_audio(self.decode_tensor(codes, state)?)
    }
    fn read_audio(output: Tensor<B, 3>) -> Result<Vec<f32>> {
        let output = output.cast(FloatDType::F32);
        let data = pollster::block_on(output.into_data_async())
            .map_err(|e| Error::Inference(e.to_string()))?;
        data.to_vec::<f32>()
            .map_err(|e| Error::Inference(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::NdArray;

    #[test]
    fn causal_convolutions_match_whole_sequence_at_uneven_boundaries() {
        let device = Default::default();
        for (transpose, kernel, dilation, stride) in
            [(false, 7, 3, 1), (true, 8, 1, 4), (true, 2, 1, 2)]
        {
            let weight: Tensor<B, 3> = Tensor::from_data(
                TensorData::new(
                    (0..2 * 3 * kernel)
                        .map(|i| (i as f32 * 0.31).sin() * 0.1)
                        .collect::<Vec<_>>(),
                    if transpose {
                        [2, 3, kernel]
                    } else {
                        [3, 2, kernel]
                    },
                ),
                &device,
            );
            let bias = Tensor::<B, 1>::from_data([0.1, -0.2, 0.3], &device);
            let x = Tensor::<B, 3>::from_data(
                TensorData::new(
                    (0..2 * 17)
                        .map(|i| (i as f32 * 0.14).cos())
                        .collect::<Vec<_>>(),
                    [1, 2, 17],
                ),
                &device,
            );
            let full = if transpose {
                module::conv_transpose1d(
                    x.clone(),
                    weight.clone(),
                    Some(bias.clone()),
                    ConvTransposeOptions::new([stride], [0], [0], [1], 1),
                )
                .slice([0..1, 0..3, 0..17 * stride])
            } else {
                let padded = Tensor::cat(
                    vec![
                        Tensor::zeros([1, 2, (kernel - 1) * dilation], &device),
                        x.clone(),
                    ],
                    2,
                );
                module::conv1d(
                    padded,
                    weight.clone(),
                    Some(bias.clone()),
                    ConvOptions::new([1], [0], [dilation], 1),
                )
            };
            let conv = Conv {
                linear_weight: if !transpose {
                    Some(
                        weight
                            .clone()
                            .swap_dims(1, 2)
                            .reshape([3, kernel * 2])
                            .transpose(),
                    )
                } else {
                    None
                },
                weight,
                bias: Some(bias),
                kernel,
                dilation,
                stride,
                groups: 1,
                transpose,
                index: 0,
            };
            let mut state = CodecState {
                conv: vec![None],
                transformer: TransformerCache::new(0),
            };
            let mut chunks = Vec::new();
            let mut offset = 0;
            for len in [1, 3, 2, 11] {
                chunks.push(conv.forward(
                    x.clone().slice([0..1, 0..2, offset..offset + len]),
                    &mut state,
                ));
                offset += len;
            }
            let stream = Tensor::cat(chunks, 2);
            let a = full.into_data().to_vec::<f32>().unwrap();
            let b = stream.into_data().to_vec::<f32>().unwrap();
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5),
                "transpose={transpose}, kernel={kernel}"
            );
        }
    }
}
