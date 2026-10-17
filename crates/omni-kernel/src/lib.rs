//! One adapter for lazy custom GPU kernels, used by the TTS and image models.
//!
//! Kernel owners validate their shapes and implement a raw kernel. This module
//! owns backend checks, device access, Fusion registration and the ANE handoff.
//! No tensor is copied to CPU or synchronously read to cross this seam.
use burn::tensor::{Shape, backend::Backend};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime, FusionTensor,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use std::{
    any::{Any, TypeId},
    fmt,
};

mod int8_host;
pub mod int8_matmul;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod int8_tensorops;
pub mod paged;

pub type Raw = CubeBackend<WgpuRuntime, f32, i32, u8>;
pub type Metal = Fusion<Raw>;
pub(crate) type Runtime = <Raw as FusionBackend>::FusionRuntime;
pub type RawTensor = CubeTensor<WgpuRuntime>;

/// Return the actual GPU device only for supported backend and fallback policy.
/// Image kernels can apply their hardware guard equally to Metal and Ane<Metal>.
pub fn device<B: Backend>(device: &B::Device) -> Option<&WgpuDevice> {
    if TypeId::of::<B>() == TypeId::of::<Metal>() {
        return (device as &dyn Any).downcast_ref::<WgpuDevice>();
    }
    #[cfg(all(feature = "ane", target_os = "macos", target_arch = "aarch64"))]
    if TypeId::of::<B>() == TypeId::of::<burn_ane::Ane<Metal>>() {
        let device = (device as &dyn Any).downcast_ref::<burn_ane::AneDevice<Metal>>()?;
        if device.options().ane_first
            || device.options().fallback != burn_ane::FallbackPolicy::Allow
        {
            return None;
        }
        return Some(device.fallback_device());
    }
    None
}

struct Kernel<const N: usize, F> {
    desc: CustomOpIr,
    run: F,
}
impl<const N: usize, F> fmt::Debug for Kernel<N, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.desc.fmt(f)
    }
}
impl<const N: usize, F> Operation<Runtime> for Kernel<N, F>
where
    F: Fn([RawTensor; N]) -> RawTensor + Send + Sync,
{
    fn execute(&self, handles: &mut HandleContainer<<Runtime as FusionRuntime>::FusionHandle>) {
        let (inputs, [output]) = self.desc.as_fixed::<N, 1>();
        let inputs = std::array::from_fn(|i| handles.get_float_tensor::<Raw>(&inputs[i]));
        handles.register_float_tensor::<Raw>(&output.id, (self.run)(inputs));
    }
}

/// Register a single-output operation after the caller's shape/hardware guards.
/// Unsupported backends decline before touching their primitive representation.
pub fn dispatch<B: Backend, const N: usize>(
    name: &'static str,
    inputs: [B::FloatTensorPrimitive; N],
    shape: Shape,
    run: impl Fn([RawTensor; N]) -> RawTensor + Send + Sync + 'static,
) -> Option<B::FloatTensorPrimitive> {
    if N == 0 {
        return None;
    }
    let register = move |inputs: [FusionTensor<Runtime>; N]| {
        let first = &inputs[0];
        let client = first.client.clone();
        assert!(
            inputs
                .iter()
                .all(|input| input.client.device() == client.device()),
            "custom kernel inputs must share a GPU device"
        );
        let streams = OperationStreams::with_inputs(inputs.iter());
        let output = TensorIr::uninit(client.create_empty_handle(), shape, first.dtype);
        let desc = CustomOpIr::new(name, &inputs.map(FusionTensor::into_ir), &[output]);
        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                Kernel::<N, _> { desc, run },
            )
            .output()
    };
    let output: Box<dyn Any> = if TypeId::of::<B>() == TypeId::of::<Metal>() {
        let inputs = inputs.map(|input| {
            let input: Box<dyn Any> = Box::new(input);
            *input
                .downcast::<FusionTensor<Runtime>>()
                .expect("checked Metal primitive")
        });
        Box::new(register(inputs))
    } else {
        #[cfg(all(feature = "ane", target_os = "macos", target_arch = "aarch64"))]
        {
            type Primitive =
                <burn_ane::Ane<Metal> as burn::tensor::backend::BackendTypes>::FloatTensorPrimitive;
            if TypeId::of::<B>() != TypeId::of::<burn_ane::Ane<Metal>>() {
                return None;
            }
            let inputs = inputs.map(|input| {
                let input: Box<dyn Any> = Box::new(input);
                *input
                    .downcast::<Primitive>()
                    .expect("checked ANE primitive")
            });
            let client = inputs[0].client.clone();
            if !client.prefers_fallback_kernels() {
                return None;
            }
            Box::new(
                client
                    .fallback_float(name, inputs, register)
                    .expect("custom GPU kernel handoff failed"),
            )
        }
        #[cfg(not(all(feature = "ane", target_os = "macos", target_arch = "aarch64")))]
        return None;
    };
    output
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|output| *output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, TensorPrimitive};

    #[test]
    fn unsupported_backend_declines_without_running_a_kernel() {
        type Cpu = burn::backend::NdArray;
        let cpu = Default::default();
        assert!(device::<Cpu>(&cpu).is_none());
        let input = Tensor::<Cpu, 1>::ones([3], &cpu).into_primitive().tensor();
        let result = dispatch::<Cpu, 1>("unreachable", [input], Shape::new([3]), |_| {
            panic!("unsupported kernel was run")
        });
        assert!(result.is_none());
    }

    #[cfg(all(feature = "ane", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn ane_device_guard_respects_routing_and_strict_fallback_policy() {
        use burn_ane::{Ane, AneDevice, AneOptions, FallbackPolicy};
        for (ane_first, fallback, allowed) in [
            (false, FallbackPolicy::Allow, true),
            (true, FallbackPolicy::Allow, false),
            (false, FallbackPolicy::Error, false),
        ] {
            let gpu = WgpuDevice::default();
            let ane = AneDevice::<Metal>::new(
                gpu.clone(),
                AneOptions {
                    ane_first,
                    fallback,
                    ..Default::default()
                },
            );
            assert_eq!(device::<Ane<Metal>>(&ane), allowed.then_some(&gpu));
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn fusion_registration_executes_the_same_lazy_inputs() {
        use burn::tensor::ops::FloatTensorOps;
        let gpu = WgpuDevice::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &gpu,
            Default::default(),
        );
        let x = Tensor::<Metal, 1>::from_data(TensorData::new(vec![1., 2., 3.], [3]), &gpu) * 2.0;
        let result = dispatch::<Metal, 1>(
            "adapter_double",
            [x.into_primitive().tensor()],
            Shape::new([3]),
            |[x]| Raw::float_add(x.clone(), x),
        )
        .unwrap();
        let result = Tensor::<Metal, 1>::from_primitive(TensorPrimitive::Float(result));
        assert_eq!(result.into_data().to_vec::<f32>().unwrap(), [4., 8., 12.]);
    }
}
