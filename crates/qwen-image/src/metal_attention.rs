//! Burn Fusion adapter for the Rust/CubeCL FlashAttention kernel.
use crate::flash_attention as kernel;
use burn::tensor::{DType, Shape, Tensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    prelude::{CubeCount, CubeDim},
    wgpu::WgpuRuntime,
};
use std::sync::OnceLock;

fn raw_attention(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    v: CubeTensor<WgpuRuntime>,
    causal: bool,
    offset: usize,
) -> CubeTensor<WgpuRuntime> {
    let aligned = |x: CubeTensor<WgpuRuntime>| {
        if x.meta.strides()[3] == 1
            && x.meta.strides()[..3].iter().all(|s| s % 4 == 0)
            && x.handle.offset_start.unwrap_or(0) % 8 == 0
        {
            return x;
        }
        // Force a GPU copy even for dense slices with an unaligned byte offset.
        let storage =
            cubecl::std::tensor::into_contiguous(&x.client, x.clone().binding(), x.dtype.into());
        CubeTensor::new(
            x.client,
            storage.handle,
            *storage.metadata,
            x.device,
            x.dtype,
        )
    };
    let (q, k, v) = (aligned(q), aligned(k), aligned(v));
    let kl = k.meta.shape()[2];
    let s = q.meta.shape();
    let (b, h, ql, d) = (s[0], s[1], s[2], s[3]);
    let out = CubeTensor::new_contiguous(
        q.client.clone(),
        q.device.clone(),
        Shape::new([b, h, ql, d]),
        q.client.empty(b * h * ql * d * 2),
        DType::F16,
    );
    // SAFETY: try_attention checks dtype, D=128, device family, dimensions and
    // GQA. The Rust kernel bounds all Q/K tails and uses each tensor's strides.
    unsafe {
        kernel::flash_attention::launch_unchecked::<WgpuRuntime>(
            &q.client,
            CubeCount::Static(ql.div_ceil(64) as u32, h as u32, b as u32),
            CubeDim::new_1d(128),
            q.clone().into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            offset as u32,
            causal,
            ql % 64 == 0,
            kl % 32 == 0,
        );
    }
    out
}

pub fn try_attention<T: Backend>(
    q: &Tensor<T, 4>,
    k: &Tensor<T, 4>,
    v: &Tensor<T, 4>,
    causal: bool,
    offset: usize,
) -> Option<Tensor<T, 4>> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !ENABLED.get_or_init(|| std::env::var("OMNI_QWEN_RUST_ATTN").as_deref() != Ok("0")) {
        return None;
    }
    let [b, h, ql, d] = q.dims();
    let [kb, kh, kl, kd] = k.dims();
    if d != 128
        || kd != d
        || b != kb
        || kh == 0
        || h % kh != 0
        || ql == 0
        || kl == 0
        || h == 0
        || b == 0
        || v.dims() != k.dims()
        || [b, h, ql, kl, offset]
            .into_iter()
            .any(|x| x > i32::MAX as usize / 2)
        || [q.dtype(), k.dtype(), v.dtype()]
            .into_iter()
            .any(|x| x != DType::F16)
        || q.device() != k.device()
        || q.device() != v.device()
    {
        return None;
    }
    if !super::metal_tensorops::supported(omni_kernel::device::<T>(&q.device())?) {
        return None;
    }
    let inputs = [q, k, v].map(|input| input.clone().into_primitive().tensor());
    let output = omni_kernel::dispatch::<T, 3>(
        "qwen_rust_flash_attention",
        inputs,
        Shape::new([b, h, ql, d]),
        move |[q, k, v]| raw_attention(q, k, v, causal, offset),
    )?;
    Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
}
