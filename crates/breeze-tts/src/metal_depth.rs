//! Small-row Metal kernels for the TTS depth decoder, with stock fallbacks.
use burn::tensor::f16;
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{prelude::*, wgpu::WgpuRuntime};
#[cfg(test)]
use omni_kernel::Metal as B;
use omni_kernel::Raw;

// One SIMD group reduces one output channel. The original Linear's transposed
// view still stores each output row contiguously. No weight rearrange or copy.
#[cube(launch_unchecked)]
fn gemv_kernel(
    x: &Tensor<f16>,
    w: &Tensor<f16>,
    out: &mut Tensor<f16>,
    #[comptime] k: usize,
    #[comptime] n: usize,
) {
    let lane = UNIT_POS % PLANE_DIM;
    let row = (CUBE_POS_X * 4 + UNIT_POS / PLANE_DIM) as usize;
    let batch = CUBE_POS_Y as usize;
    let mut sum = 0.0f32;
    if row < n {
        for i in 0..k / 32 {
            let col = i * 32 + lane as usize;
            sum += f32::cast_from(x[batch * k + col]) * f32::cast_from(w[row * k + col]);
        }
    }
    sum = plane_sum(sum);
    if lane == 0 && row < n {
        out[batch * n + row] = f16::cast_from(sum);
    }
}

fn raw_gemv(x: CubeTensor<WgpuRuntime>, w: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    let x = burn_cubecl::kernel::into_contiguous(x);
    let rank = x.meta.shape().num_dims();
    let k = x.meta.shape()[rank - 1];
    let props = &x.client.properties().hardware;
    if props.plane_size_min != 32
        || props.plane_size_max != 32
        || w.meta.strides()[0] != 1
        || w.meta.strides()[1] != k
    {
        use burn::tensor::ops::ModuleOps;
        return Raw::linear(x, w, None);
    }
    let m = x.meta.shape().iter().take(rank - 1).product::<usize>();
    let n = w.meta.shape()[1];
    assert_eq!(x.dtype, DType::F16);
    assert_eq!(w.dtype, DType::F16);
    assert_eq!(x.meta.strides()[rank - 1], 1);
    assert_eq!(w.meta.strides()[0], 1);
    assert_eq!(w.meta.strides()[1], k);
    assert_eq!(k % 32, 0);
    let mut shape = x.meta.shape().to_vec();
    shape[rank - 1] = n;
    let out = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::from(shape),
        x.client.empty(m * n * 2),
        DType::F16,
    );
    // SAFETY: the adapter bounds shape and K, and the runtime has 32-lane SIMD.
    // Input is contiguous, weights are output-major, and excess rows are masked.
    unsafe {
        gemv_kernel::launch_unchecked::<WgpuRuntime>(
            &x.client,
            CubeCount::Static(n.div_ceil(4) as u32, m as u32, 1),
            CubeDim::new_1d(128),
            x.clone().into_tensor_arg(),
            w.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            k,
            n,
        );
    }
    out
}

// A checked type bridge preserves the public Backend-generic model API.
pub(crate) fn try_linear<T: burn::tensor::backend::Backend, const D: usize>(
    x: &BurnTensor<T, D>,
    w: &BurnTensor<T, 2>,
) -> Option<BurnTensor<T, D>> {
    if x.device() != w.device() || omni_kernel::device::<T>(&x.device()).is_none() {
        return None;
    }
    let last = D.checked_sub(1)?;
    let shape = x.dims();
    let [k, n] = w.dims();
    let rows = shape.iter().take(last).product::<usize>();
    if !(1..=64).contains(&rows)
        || k == 0
        || !k.is_multiple_of(32)
        || n == 0
        || n > 4096
        || shape[last] != k
        || x.dtype() != DType::F16
        || w.dtype() != DType::F16
    {
        return None;
    }
    let xp = x.clone().into_primitive().tensor();
    let wp = w.clone().into_primitive().tensor();
    let mut output_shape = shape.to_vec();
    output_shape[last] = n;
    let output = omni_kernel::dispatch::<T, 2>(
        "breeze_f16_gemv",
        [xp, wp],
        Shape::from(output_shape),
        |[x, w]| raw_gemv(x, w),
    )?;
    Some(BurnTensor::<T, D>::from_primitive(TensorPrimitive::Float(
        output,
    )))
}

#[cfg(test)]
#[path = "metal_depth_tests.rs"]
mod tests;

#[cube(launch_unchecked)]
fn rms_kernel(
    x: &Tensor<f16>,
    w: &Tensor<f32>,
    out: &mut Tensor<f16>,
    eps: f32,
    #[comptime] k: usize,
) {
    let lane = UNIT_POS as usize;
    let row = CUBE_POS_X as usize;
    let mut sum = 0.0f32;
    for i in 0..k / 32 {
        let value = f32::cast_from(x[row * k + i * 32 + lane]);
        sum += value * value;
    }
    sum = plane_sum(sum);
    let inv = 1.0f32 / (sum / f32::cast_from(k) + eps).sqrt();
    for i in 0..k / 32 {
        let col = i * 32 + lane;
        let value = f32::cast_from(x[row * k + col]);
        // Match Rms: round normalized values and weights to half separately,
        // then multiply in half. Residual additions remain outside this kernel.
        out[row * k + col] = f16::cast_from(value * inv) * f16::cast_from(w[col]);
    }
}

fn raw_rms_fallback(
    x: CubeTensor<WgpuRuntime>,
    w: CubeTensor<WgpuRuntime>,
    eps: f32,
) -> CubeTensor<WgpuRuntime> {
    use burn::tensor::{FloatDType, ops::FloatTensorOps};
    let rank = x.meta.shape().num_dims();
    let mut weight_shape = vec![1; rank];
    weight_shape[rank - 1] = w.meta.shape()[0];
    let x = Raw::float_cast(x, FloatDType::F32);
    let mean = Raw::float_mean_dim(Raw::float_mul(x.clone(), x.clone()), rank - 1);
    let inv = Raw::float_recip(Raw::float_sqrt(Raw::float_add_scalar(mean, eps.into())));
    let normalized = Raw::float_cast(Raw::float_mul(x, inv), FloatDType::F16);
    let weight = Raw::float_reshape(Raw::float_cast(w, FloatDType::F16), weight_shape.into());
    Raw::float_mul(normalized, weight)
}

fn raw_rms(
    x: CubeTensor<WgpuRuntime>,
    w: CubeTensor<WgpuRuntime>,
    eps: f32,
) -> CubeTensor<WgpuRuntime> {
    let x = burn_cubecl::kernel::into_contiguous(x);
    let w = burn_cubecl::kernel::into_contiguous(w);
    let hardware = &x.client.properties().hardware;
    if hardware.plane_size_min != 32 || hardware.plane_size_max != 32 {
        return raw_rms_fallback(x, w, eps);
    }
    let k = w.meta.shape()[0];
    let rows = x.meta.shape().num_elements() / k;
    let output = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        x.meta.shape().clone(),
        x.client.empty(rows * k * 2),
        DType::F16,
    );
    // SAFETY: try_rms validates shape/dtypes, and the device has 32-lane SIMD.
    unsafe {
        rms_kernel::launch_unchecked::<WgpuRuntime>(
            &x.client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(32),
            x.clone().into_tensor_arg(),
            w.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            eps,
            k,
        );
    }
    output
}

pub(crate) fn try_rms<T: burn::tensor::backend::Backend, const D: usize>(
    x: &BurnTensor<T, D>,
    w: &BurnTensor<T, 1>,
    eps: f32,
) -> Option<BurnTensor<T, D>> {
    if x.device() != w.device() || omni_kernel::device::<T>(&x.device()).is_none() {
        return None;
    }
    let last = D.checked_sub(1)?;
    let shape = x.dims();
    let k = shape[last];
    let rows = shape.iter().take(last).product::<usize>();
    if !(1..=64).contains(&rows)
        || k == 0
        || !k.is_multiple_of(32)
        || w.dims() != [k]
        || x.dtype() != DType::F16
        || w.dtype() != DType::F32
        || !eps.is_finite()
        || eps <= 0.0
    {
        return None;
    }
    let xp = x.clone().into_primitive().tensor();
    let wp = w.clone().into_primitive().tensor();
    let output = omni_kernel::dispatch::<T, 2>(
        "breeze_depth_rms_f16",
        [xp, wp],
        Shape::from(shape),
        move |[x, w]| raw_rms(x, w, eps),
    )?;
    Some(BurnTensor::<T, D>::from_primitive(TensorPrimitive::Float(
        output,
    )))
}
