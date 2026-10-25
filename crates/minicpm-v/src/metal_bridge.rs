//! The one place raw burn tensors cross into cube kernels and back.
//!
//! Every hand-written Metal kernel in this crate (moe_dense, gdn_scan, glue,
//! markov) borrows backend-generic burn tensors, checks the backend really
//! is Metal, reinterprets the handle, converts it to a cube tensor, and
//! allocates its outputs. That adapter lived as four hand-copied sets of
//! near-identical helpers; it is concentrated here so the crate's only
//! `unsafe` soundness argument is made once:
//!
//! * the backend check ([`borrowable`]) and the transmute ([`take_as`]) are
//!   paired: converting a tensor is only sound after the check passed for
//!   that rank, which is why [`cube_float`]/[`cube_int`] take the generic
//!   tensor and do both in one place;
//! * the converter keeps burn's "float or int kind" contract (`None` when a
//!   float slot holds something unexpected);
//! * output buffers come from [`alloc_f32`] so every kernel writes the same
//!   contiguous-F32 layout it reads.

use burn::tensor::{Shape, Tensor as BurnTensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::wgpu::WgpuRuntime;
use std::any::TypeId;
use std::mem::{forget, size_of};

/// Whether rank-`D` tensors on backend `B` may be borrowed into a cube
/// kernel: the backend must be Metal and the burn handle layout must match
/// (the generic-over-`B` call sites stay monomorphic through this check).
pub(crate) fn borrowable<B: Backend, const D: usize>() -> bool {
    use burn::backend::wgpu::Metal;
    TypeId::of::<B>() == TypeId::of::<Metal>()
        && size_of::<BurnTensor<B, D>>() == size_of::<BurnTensor<Metal, D>>()
}

/// # Safety
/// `T` and `U` are the same type: the caller checked [`borrowable`] for the
/// involved rank (or holds another same-type proof).
pub(crate) unsafe fn take_as<T, U>(value: T) -> U {
    let out = unsafe { std::ptr::read((&value) as *const T as *const U) };
    forget(value);
    out
}

/// Borrow a float-kind burn tensor into a contiguous cube tensor. `None`
/// when the float slot does not actually hold a float primitive.
///
/// # Panics / soundness
/// Sound only after [`borrowable::<B, D>`] passed; callers gate on it.
pub(crate) fn cube_float<B: Backend, const D: usize>(
    tensor: BurnTensor<B, D>,
) -> Option<CubeTensor<WgpuRuntime>> {
    use burn::backend::wgpu::Metal;
    let tensor: BurnTensor<Metal, D> = unsafe { take_as(tensor) };
    let primitive = tensor.into_primitive();
    let TensorPrimitive::Float(tensor) = primitive else {
        return None;
    };
    Some(burn_cubecl::kernel::into_contiguous(tensor))
}

/// Borrow an int-kind burn tensor into a contiguous cube tensor. Sound only
/// after [`borrowable::<B, D>`] passed.
pub(crate) fn cube_int<B: Backend, const D: usize>(
    tensor: BurnTensor<B, D, burn::tensor::Int>,
) -> CubeTensor<WgpuRuntime> {
    use burn::backend::wgpu::Metal;
    let tensor: BurnTensor<Metal, D, burn::tensor::Int> = unsafe { take_as(tensor) };
    burn_cubecl::kernel::into_contiguous(tensor.into_primitive())
}

/// Allocate a contiguous F32 output buffer of the given shape.
pub(crate) fn alloc_f32<const N: usize>(
    client: &cubecl::client::ComputeClient<WgpuRuntime>,
    device: &cubecl::wgpu::WgpuDevice,
    shape: [usize; N],
) -> CubeTensor<WgpuRuntime> {
    let values: usize = shape.iter().product();
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from(shape),
        client.empty(values * std::mem::size_of::<f32>()),
        burn::tensor::DType::F32,
    )
}
