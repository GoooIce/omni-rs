//! Hand-fused per-layer glue kernels: the Gemma RMS-norm chain, the plain
//! RMS norm, the split-half RoPE rotation and the residual add.
//!
//! Two consumers, both launch-bound where these kernels apply. The verify
//! forward is GPU-launch-bound (90% of its launches were burn glue chains,
//! `verify_profile`); the fused verify seams — Gemma norms, residual adds,
//! split-half RoPE — removed ~20% of the verify wall in same-window
//! interleaved A/B with greedy output byte-identical (#52, 2026-10-26), so
//! they run by default (`MINICPM_GLUE=0` is the same-binary kill switch).
//! The DSpark draft forward is CPU-dispatch-bound (~27 µs per burn op), so
//! its seams also run by default (`MINICPM_DRAFT_GLUE=0` disables them for
//! A/B). The kernels reproduce `Norm::forward` / `shared::rope` /
//! `shared::residual` arithmetic with f16 in/out and f32 math — except the
//! per-head (256-wide) Gemma norm: burn's autotuned reduce orders the
//! row differently there than the 2048-wide rows the kernel matches, so
//! that site stays on the burn chain (byte gate caught it on prose).

use crate::metal_bridge::take_as;
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use half::f16;

const LANES: u32 = 64;

/// One cube per token row, `LANES` units striding the hidden dimension.
/// The row's square sum crosses units through cube address space; every
/// unit then redundantly totals it (64 shared-memory reads) before the
/// normalized write. `plus_one` selects the Gemma learned-shift form; plain
/// RMS passes false.
#[cube(launch_unchecked)]
fn rms_norm(
    x: &Tensor<f16>,       // (rows, hidden)
    weight: &Tensor<f32>,  // (hidden,) stored f32
    out: &mut Tensor<f16>, // (rows, hidden)
    eps: f32,
    hidden: u32,
    #[comptime] lanes: u32,
    #[comptime] plus_one: bool,
) {
    let token = CUBE_POS_X;
    let unit = UNIT_POS_X;
    let per = hidden / lanes;
    let base = token * hidden;
    let start = base + unit * per;
    let mut sumsq = 0.0f32;
    for i in 0..per {
        let v = f32::cast_from(x[(start + i) as usize]);
        sumsq += v * v;
    }
    let mut sums = SharedMemory::<f32>::new(lanes as usize);
    sums[unit as usize] = sumsq;
    sync_cube();
    let mut total = 0.0f32;
    for i in 0..lanes {
        total += sums[i as usize];
    }
    let denom = (total / f32::cast_from(hidden) + eps).sqrt();
    for i in 0..per {
        let index = (start + i) as usize;
        let mut w = weight[index - base as usize];
        if plus_one {
            w += 1.0;
        }
        let value = f32::cast_from(x[index]) / denom * w;
        out[index] = f16::cast_from(value);
    }
}

/// Split-half RoPE over a contiguous `[1, heads, tokens, width]` block: one
/// cube per (head, token), one unit per frequency. cos/sin are
/// `[tokens, width/2]` and broadcast across heads.
#[cube(launch_unchecked)]
fn rope_split_half(
    x: &Tensor<f16>,       // (heads * tokens, width)
    cos: &Tensor<f16>,     // (tokens, width)
    sin: &Tensor<f16>,     // (tokens, width)
    out: &mut Tensor<f16>, // (heads * tokens, width)
    tokens: u32,
    width: u32,
) {
    let head = CUBE_POS_X;
    let token = CUBE_POS_Y;
    let d = UNIT_POS_X;
    let half = width / 2;
    if d < half {
        let base = ((head * tokens) + token) * width;
        // The rope tables are full-width; their first half carries the
        // distinct frequencies (the second half repeats it).
        let table = (token * width + d) as usize;
        let c = f32::cast_from(cos[table]);
        let s = f32::cast_from(sin[table]);
        let left = f32::cast_from(x[(base + d) as usize]);
        let right = f32::cast_from(x[(base + half + d) as usize]);
        out[(base + d) as usize] = f16::cast_from(left * c - right * s);
        out[(base + half + d) as usize] = f16::cast_from(right * c + left * s);
    }
}

/// out = f16(f32(x) + f32(y)) over a flat activation buffer.
#[cube(launch_unchecked)]
fn residual_add(x: &Tensor<f16>, y: &Tensor<f16>, out: &mut Tensor<f16>) {
    let index = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    out[index] = f16::cast_from(f32::cast_from(x[index]) + f32::cast_from(y[index]));
}

/// Fused `Norm::forward` for the verify seams. `None` off Metal or on shapes
/// the kernel does not cover (f16 activations, hidden a multiple of the
/// lane count); the caller falls back to the burn path.
/// Default ON (`MINICPM_GLUE=0` kills) — see the module doc for the #52 A/B.
fn verify_enabled() -> bool {
    crate::knobs::verify_glue()
}

/// The draft seams default ON: the draft forward is CPU-dispatch-bound, so
/// each replaced burn op is dispatch saved. `MINICPM_DRAFT_GLUE=0` is the
/// same-binary kill switch for A/B.
fn draft_enabled() -> bool {
    crate::knobs::draft_glue()
}

/// The shared RMS-norm launch: one cube per row, one kernel per call
/// (`plus_one` selects the Gemma learned shift).
fn norm_launch<B: Backend>(
    x: BurnTensor<B, 2>,
    weight: &BurnTensor<B, 1>,
    eps: f64,
    plus_one: bool,
) -> Option<BurnTensor<B, 2>> {
    use burn::backend::wgpu::Metal;
    if !crate::metal_bridge::borrowable::<B, 2>() {
        return None;
    }
    let [m, hidden] = x.dims();
    if m == 0 || hidden == 0 || hidden % LANES as usize != 0 || x.dtype() != DType::F16 {
        return None;
    }
    let x: BurnTensor<Metal, 2> = unsafe { take_as(x) };
    let weight: BurnTensor<Metal, 2> = unsafe { take_as(weight.clone().unsqueeze_dim::<2>(0)) };
    let x = crate::metal_bridge::cube_float(x)?;
    let weight = crate::metal_bridge::cube_float(weight)?;
    let client = x.client.clone();
    let device = x.device.clone();
    let out = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([m, hidden]),
        client.empty(m * hidden * size_of::<f16>()),
        DType::F16,
    );
    unsafe {
        rms_norm::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(m as u32, 1, 1),
            CubeDim::new_1d(LANES),
            x.into_tensor_arg(),
            weight.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            eps as f32,
            hidden as u32,
            LANES,
            plus_one,
        );
    }
    let out = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(out));
    Some(unsafe { take_as(out) })
}

pub(crate) fn gemma_norm_metal<B: Backend>(
    x: BurnTensor<B, 2>,
    weight: &BurnTensor<B, 1>,
    eps: f64,
) -> Option<BurnTensor<B, 2>> {
    if !verify_enabled() {
        return None;
    }
    norm_launch(x, weight, eps, true)
}

/// Plain `NormKind::Rms` norm for the DSpark draft seams.
pub(crate) fn plain_rms_norm_metal<B: Backend>(
    x: BurnTensor<B, 2>,
    weight: &BurnTensor<B, 1>,
    eps: f64,
) -> Option<BurnTensor<B, 2>> {
    if !draft_enabled() {
        return None;
    }
    norm_launch(x, weight, eps, false)
}

/// The shared residual launch.
fn residual_launch<B: Backend>(
    x: BurnTensor<B, 2>,
    y: BurnTensor<B, 2>,
) -> Option<BurnTensor<B, 2>> {
    use burn::backend::wgpu::Metal;
    if !crate::metal_bridge::borrowable::<B, 2>() {
        return None;
    }
    let [m, hidden] = x.dims();
    let values = m * hidden;
    if values == 0 || values % LANES as usize != 0 || x.dtype() != DType::F16 {
        return None;
    }
    let x: BurnTensor<Metal, 2> = unsafe { take_as(x) };
    let y: BurnTensor<Metal, 2> = unsafe { take_as(y) };
    let x = crate::metal_bridge::cube_float(x)?;
    let y = crate::metal_bridge::cube_float(y)?;
    let client = x.client.clone();
    let device = x.device.clone();
    let out = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([values]),
        client.empty(values * size_of::<f16>()),
        DType::F16,
    );
    unsafe {
        residual_add::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static((values / LANES as usize) as u32, 1, 1),
            CubeDim::new_1d(LANES),
            x.into_tensor_arg(),
            y.into_tensor_arg(),
            out.clone().into_tensor_arg(),
        );
    }
    let out = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(out));
    Some(unsafe { take_as(out) })
}

/// Fused `shared::residual` for the verify seams. Same fallback contract as
/// [`gemma_norm_metal`].
pub(crate) fn residual_metal<B: Backend>(
    x: BurnTensor<B, 2>,
    y: BurnTensor<B, 2>,
) -> Option<BurnTensor<B, 2>> {
    if !verify_enabled() {
        return None;
    }
    residual_launch(x, y)
}

/// Fused `shared::residual` for the draft seams (default on).
pub(crate) fn residual_draft_metal<B: Backend>(
    x: BurnTensor<B, 2>,
    y: BurnTensor<B, 2>,
) -> Option<BurnTensor<B, 2>> {
    if !draft_enabled() {
        return None;
    }
    residual_launch(x, y)
}

/// Fused `shared::rope` (split-half) for the draft seams: one cube per
/// (head, token), one unit per frequency. `x` is `[1, heads, tokens, width]`
/// f16 contiguous; `cos`/`sin` are the `[1, 1, tokens, width]` tables (their
/// first half carries the distinct frequencies).
pub(crate) fn rope_metal<B: Backend>(
    x: BurnTensor<B, 4>,
    cos: &BurnTensor<B, 4>,
    sin: &BurnTensor<B, 4>,
) -> Option<BurnTensor<B, 4>> {
    if !draft_enabled() {
        return None;
    }
    rope_launch(x, cos, sin)
}

/// The same launch behind the verify seams (`MINICPM_GLUE`).
pub(crate) fn rope_verify_metal<B: Backend>(
    x: BurnTensor<B, 4>,
    cos: &BurnTensor<B, 4>,
    sin: &BurnTensor<B, 4>,
) -> Option<BurnTensor<B, 4>> {
    if !verify_enabled() {
        return None;
    }
    rope_launch(x, cos, sin)
}

/// The shared split-half RoPE launch; the seam wrappers gate it.
fn rope_launch<B: Backend>(
    x: BurnTensor<B, 4>,
    cos: &BurnTensor<B, 4>,
    sin: &BurnTensor<B, 4>,
) -> Option<BurnTensor<B, 4>> {
    use burn::backend::wgpu::Metal;
    if !crate::metal_bridge::borrowable::<B, 4>() {
        return None;
    }
    let [batch, heads, tokens, width] = x.dims();
    let half = width / 2;
    if batch != 1
        || tokens == 0
        || width % 2 != 0
        || half == 0
        || x.dtype() != DType::F16
        || cos.dims()[2] < tokens
    {
        return None;
    }
    let x: BurnTensor<Metal, 4> = unsafe { take_as(x) };
    let cos: BurnTensor<Metal, 4> = unsafe { take_as(cos.clone()) };
    let sin: BurnTensor<Metal, 4> = unsafe { take_as(sin.clone()) };
    let x = crate::metal_bridge::cube_float(x)?;
    let cos = crate::metal_bridge::cube_float(cos)?;
    let sin = crate::metal_bridge::cube_float(sin)?;
    let client = x.client.clone();
    let device = x.device.clone();
    let values = heads * tokens * width;
    let out = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([batch, heads, tokens, width]),
        client.empty(values * size_of::<f16>()),
        DType::F16,
    );
    unsafe {
        rope_split_half::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(heads as u32, tokens as u32, 1),
            CubeDim::new_1d(half as u32),
            x.into_tensor_arg(),
            cos.into_tensor_arg(),
            sin.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            tokens as u32,
            width as u32,
        );
    }
    let out = BurnTensor::<Metal, 4>::from_primitive(TensorPrimitive::Float(out));
    Some(unsafe { take_as(out) })
}

#[cfg(test)]
mod tests {
    use super::{gemma_norm_metal, plain_rms_norm_metal, residual_metal, rope_metal};
    use burn::backend::wgpu::{Metal, WgpuDevice};
    use burn::tensor::{FloatDType, Tensor, TensorData};

    /// The fused kernels against the burn chains they replace, on the 35B
    /// seam shapes: f16 in, f32 math, f16 out.
    #[test]
    fn glue_kernels_match_burn_chains() {
        type B = Metal;
        let device = WgpuDevice::default();
        let (m, hidden) = (16usize, 2048usize);
        let x_values: Vec<f32> = (0..m * hidden)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.125)
            .collect();
        let w_values: Vec<f32> = (0..hidden)
            .map(|k| ((k % 13) as f32 - 6.0) * 0.05)
            .collect();
        let options = burn::tensor::TensorCreationOptions::<B> {
            device: device.clone(),
            dtype: Some(burn::tensor::DType::F16),
        };
        let x16 = Tensor::<B, 2>::from_data(
            TensorData::new(x_values.clone(), [m, hidden]),
            options.clone(),
        );
        let y_values: Vec<f32> = (0..m * hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.0625)
            .collect();
        let y16 = Tensor::<B, 2>::from_data(
            TensorData::new(y_values.clone(), [m, hidden]),
            options.clone(),
        );
        let weight = Tensor::<B, 1>::from_data(
            TensorData::new(w_values.clone(), [hidden]),
            burn::tensor::TensorCreationOptions::<B> {
                device: device.clone(),
                dtype: Some(burn::tensor::DType::F32),
            },
        );

        // Fused norm vs Norm::forward (Gemma, Divide, weight+1), f64 mirror.
        let fused = gemma_norm_metal(x16.clone(), &weight, 1e-6).expect("metal norm");
        let burn_chain = {
            let wide = x16.clone().cast(FloatDType::F32);
            let denom = (wide.clone().square().mean_dim(1) + 1e-6).sqrt();
            ((wide / denom) * (weight.clone() + 1.0).unsqueeze_dim(0)).cast(FloatDType::F16)
        };
        let fused_values = fused
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let burn_values = burn_chain
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut worst = 0f32;
        for (a, b) in fused_values.iter().zip(&burn_values) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 2e-2, "norm worst {worst}");

        // Fused residual vs shared::residual.
        let fused = residual_metal(x16.clone(), y16.clone()).expect("metal residual");
        let burn_chain =
            (x16.clone().cast(FloatDType::F32) + y16.cast(FloatDType::F32)).cast(FloatDType::F16);
        let fused_values = fused
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let burn_values = burn_chain
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut worst = 0f32;
        for (a, b) in fused_values.iter().zip(&burn_values) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 1e-3, "residual worst {worst}");

        // Plain fused RMS (the draft's NormKind::Rms) vs its burn chain.
        let fused = plain_rms_norm_metal(x16.clone(), &weight, 1e-6).expect("metal rms norm");
        let burn_chain = {
            let wide = x16.clone().cast(FloatDType::F32);
            let denom = (wide.clone().square().mean_dim(1) + 1e-6).sqrt();
            (wide / denom * weight.clone().unsqueeze::<2>()).cast(FloatDType::F16)
        };
        let fused_values = fused
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let burn_values = burn_chain
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut worst = 0f32;
        for (a, b) in fused_values.iter().zip(&burn_values) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 2e-2, "rms worst {worst}");

        // Fused split-half rope vs `shared::rope`, on the draft attention
        // shapes: 8 heads, 16 tokens, head dim 256, rotated at absolute
        // positions 100..116.
        let (heads, tokens, width) = (8usize, 16usize, 256usize);
        let x_values: Vec<f32> = (0..heads * tokens * width)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.25)
            .collect();
        let q16 = Tensor::<B, 4>::from_data(
            TensorData::new(x_values, [1, heads, tokens, width]),
            options.clone(),
        );
        let frequencies: Vec<f64> = (0..width / 2)
            .map(|d| 10000f64.powf(-(d as f64) / (width as f64 / 2.0)))
            .collect();
        let mut cosine = Vec::with_capacity(tokens * width);
        let mut sine = Vec::with_capacity(tokens * width);
        for pos in 100..100 + tokens {
            for &frequency in frequencies.iter().cycle().take(width) {
                let angle = pos as f64 * frequency;
                cosine.push(angle.cos() as f32);
                sine.push(angle.sin() as f32);
            }
        }
        let table = |values: Vec<f32>| {
            Tensor::<B, 4>::from_data(
                TensorData::new(values, [1, 1, tokens, width]),
                options.clone(),
            )
        };
        let cos = table(cosine);
        let sin = table(sine);
        let fused = rope_metal(q16.clone(), &cos, &sin).expect("metal rope");
        let burn_chain =
            omni_r2t2::tensor::rope(q16, &cos, &sin, omni_r2t2::tensor::RotaryLayout::SplitHalf);
        let fused_values = fused
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let burn_values = burn_chain
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut worst = 0f32;
        for (a, b) in fused_values.iter().zip(&burn_values) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 3e-2, "rope worst {worst}");
    }
}
