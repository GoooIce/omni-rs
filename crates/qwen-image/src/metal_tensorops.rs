//! M5 FP16 GEMM on the existing CubeCL queue and buffers. No CPU copies or waits.
use burn::tensor::{DType, Shape, Tensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    CubeTask, Runtime as _,
    device::Device as _,
    device_handle::DeviceHandle,
    ir::{StorageType, UIntKind},
    prelude::*,
    server::KernelArguments,
    wgpu::{AutoCompiler, AutoRepresentation, WgpuDevice, WgpuRuntime, WgpuServer},
};
use cubecl_cpp::shared::{Body, Elem, Flags, Item, KernelArg, MslComputeKernel};
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use objc2_metal::{MTLDevice, MTLGPUFamily};
use omni_kernel::Raw;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

pub(crate) fn supported(device: &WgpuDevice) -> bool {
    static DEVICES: OnceLock<Mutex<HashMap<WgpuDevice, bool>>> = OnceLock::new();
    let mut devices = DEVICES.get_or_init(Default::default).lock().unwrap();
    *devices.entry(device.clone()).or_insert_with(|| {
        let client = WgpuRuntime::client(device);
        if WgpuRuntime::name(&client) != "wgpu<msl>" {
            return false;
        }
        let server = DeviceHandle::<WgpuServer>::new(device.to_id());
        let device = server
            .submit_blocking(|server| server.wgpu_device())
            .unwrap();
        // SAFETY: the HAL guard retains the device; no native state is modified.
        let Some(hal) = (unsafe { device.as_hal::<wgpu::hal::api::Metal>() }) else {
            return false;
        };
        hal.raw_device().supportsFamily(MTLGPUFamily::Apple10)
    })
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct Gemm {
    m: usize,
    n: usize,
    k: usize,
}
impl KernelMetadata for Gemm {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(128))
    }
    fn address_type(&self) -> StorageType {
        UIntKind::U32.into()
    }
}
impl CubeTask<AutoCompiler> for Gemm {
    fn compile(
        &self,
        compiler: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> Result<CompiledKernel<AutoCompiler>, cubecl::CompilationError> {
        assert!(matches!(compiler, AutoCompiler::Msl(_)));
        let cube_dim = CubeDim::new_1d(128);
        // CubeCL uses this representation for resource bindings and limits only;
        // the actual source is the hand-written MSL below, with no metadata buffer.
        let repr = MslComputeKernel {
            tensor_maps: vec![],
            buffers: (0..3)
                .map(|id| KernelArg {
                    id,
                    item: Item::scalar(Elem::F16, true),
                    size: None,
                    vis: if id < 2 {
                        Visibility::Read
                    } else {
                        Visibility::ReadWrite
                    },
                })
                .collect(),
            scalars: vec![],
            info: Default::default(),
            meta_static_len: 0,
            body: Body {
                instructions: vec![],
                shared_memories: vec![],
                pipelines: vec![],
                barriers: vec![],
                const_arrays: vec![],
                local_arrays: vec![],
                info_by_ptr: false,
                has_dynamic_meta: false,
                address_type: Item::scalar(Elem::U32, true),
            },
            cube_dim,
            cluster_dim: None,
            extensions: vec![],
            flags: Flags {
                cube_dim,
                ..Default::default()
            },
            items: Default::default(),
            kernel_name: "qwen_tensorops".into(),
        };
        Ok(CompiledKernel {
            entrypoint_name: "qwen_tensorops".into(),
            debug_name: Some("qwen_tensorops"),
            source: format!(
                "#define ROWS {}\n#define COLS {}\n#define INNER {}\n{}",
                self.m,
                self.n,
                self.k,
                include_str!("tensorops.metal")
            ),
            repr: Some(AutoRepresentation::Msl(repr)),
            cube_dim,
            debug_info: None,
        })
    }
}

fn raw_linear(x: CubeTensor<WgpuRuntime>, w: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    let rank = x.meta.shape().num_dims();
    let k = x.meta.shape()[rank - 1];
    if w.meta.strides()[0] != 1 || w.meta.strides()[1] != k {
        use burn::tensor::ops::ModuleOps;
        return Raw::linear(x, w, None);
    }
    let x = burn_cubecl::kernel::into_contiguous(x);
    let m = x.meta.shape().iter().take(rank - 1).product::<usize>();
    let n = w.meta.shape()[1];
    let mut shape = x.meta.shape().to_vec();
    shape[rank - 1] = n;
    let out = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::from(shape),
        x.client.empty(m * n * 2),
        DType::F16,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        x.handle.clone().binding(),
        w.handle.binding(),
        out.handle.clone().binding(),
    ]);
    // SAFETY: validated FP16 dimensions/strides; MPP slices mask partial tiles;
    // all 128 threads participate. Handles retain allocations through execution.
    unsafe {
        x.client.launch_unchecked(
            Box::new(Gemm { m, n, k }),
            CubeCount::Static(n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1),
            bindings,
        );
    }
    out
}

pub fn try_linear<T: Backend, const D: usize>(
    x: &Tensor<T, D>,
    w: &Tensor<T, 2>,
) -> Option<Tensor<T, D>> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !ENABLED.get_or_init(|| std::env::var("OMNI_QWEN_TENSOROPS").as_deref() != Ok("0")) {
        return None;
    }
    let last = D.checked_sub(1)?;
    let shape = x.dims();
    let [k, n] = w.dims();
    let rows = shape.iter().take(last).product::<usize>();
    if rows < 256
        || rows > 65535
        || !k.is_multiple_of(64)
        || !n.is_multiple_of(64)
        || k < 4096
        || n < 4096
        || k > 12288
        || n > 12288
        || shape[last] != k
        || x.dtype() != DType::F16
        || w.dtype() != DType::F16
    {
        return None;
    }
    if x.device() != w.device() || !supported(omni_kernel::device::<T>(&x.device())?) {
        return None;
    }
    let inputs = [
        x.clone().into_primitive().tensor(),
        w.clone().into_primitive().tensor(),
    ];
    let mut shape = shape.to_vec();
    shape[last] = n;
    let output = omni_kernel::dispatch::<T, 2>(
        "qwen_tensorops_linear",
        inputs,
        Shape::from(shape),
        |[x, w]| raw_linear(x, w),
    )?;
    Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
}
