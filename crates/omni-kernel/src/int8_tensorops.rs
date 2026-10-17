//! int8 `matmul2d` on M5 TensorOps. The CubeCL kernel remains the fallback.
use crate::RawTensor;
use burn::tensor::TensorMetadata;
use burn::tensor::{DType, Shape};
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use cubecl::{
    CubeCount, CubeDim, CubeTask, ExecutionMode,
    device::Device as _,
    device_handle::DeviceHandle,
    ir::{StorageType, UIntKind},
    prelude::{KernelId, KernelMetadata},
    server::KernelArguments,
    wgpu::{AutoCompiler, AutoRepresentation, WgpuDevice, WgpuServer},
};
use cubecl_cpp::shared::{Body, Elem, Flags, Item, KernelArg, MslComputeKernel};
use cubecl_runtime::kernel::CompiledKernel;
use objc2_metal::{MTLDevice, MTLGPUFamily};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

const SOURCE: &str = include_str!("int8_tensorops.metal");

pub(crate) fn try_launch(
    x: RawTensor,
    w_packed: RawTensor,
    w_scale: RawTensor,
    w_zero: RawTensor,
) -> Option<RawTensor> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !ENABLED.get_or_init(|| std::env::var("OMNI_CONVROT_TENSOROPS").as_deref() != Ok("0")) {
        return None;
    }
    if !apple10(&x.device) {
        return None;
    }
    let x = burn_cubecl::kernel::into_contiguous(x);
    let w_packed = burn_cubecl::kernel::into_contiguous(w_packed);
    let w_scale = burn_cubecl::kernel::into_contiguous(w_scale);
    let w_zero = burn_cubecl::kernel::into_contiguous(w_zero);
    let rows = x.shape()[0];
    let width = x.shape()[1];
    let columns = w_scale.shape()[0];
    if rows == 0 || width % 4 != 0 || columns == 0 {
        return None;
    }
    let codes = empty(&x, rows * width, DType::U8);
    let x_scale = empty(&x, rows, DType::F32);
    let x_sum = empty(&x, rows, DType::I32);
    let weight = empty(&x, columns * width, DType::U8);
    let acc = empty(&x, rows * columns, DType::I32);
    let out = empty(&x, rows * columns, DType::F32);
    let out = RawTensor::new_contiguous(
        out.client.clone(),
        out.device.clone(),
        Shape::new([rows, columns]),
        out.handle,
        DType::F32,
    );
    launch(0, rows, columns, width, &x, |args| {
        args.with_buffers(vec![
            x.handle.clone().binding(),
            codes.handle.clone().binding(),
            x_scale.handle.clone().binding(),
            x_sum.handle.clone().binding(),
        ])
    });
    launch(1, rows, columns, width, &x, |args| {
        args.with_buffers(vec![
            w_packed.handle.clone().binding(),
            weight.handle.clone().binding(),
        ])
    });
    launch(2, rows, columns, width, &x, |args| {
        args.with_buffers(vec![
            codes.handle.clone().binding(),
            weight.handle.clone().binding(),
            acc.handle.clone().binding(),
        ])
    });
    launch(3, rows, columns, width, &x, |args| {
        args.with_buffers(vec![
            acc.handle.clone().binding(),
            x_scale.handle.clone().binding(),
            x_sum.handle.clone().binding(),
            w_scale.handle.clone().binding(),
            w_zero.handle.clone().binding(),
            out.handle.clone().binding(),
        ])
    });
    Some(out)
}

fn launch(
    kind: u32,
    rows: usize,
    columns: usize,
    width: usize,
    like: &RawTensor,
    bind: impl FnOnce(KernelArguments) -> KernelArguments,
) {
    let stage = Stage {
        rows,
        columns,
        width,
        kind,
    };
    let count = match kind {
        0 => CubeCount::Static(rows as u32, 1, 1),
        1 => CubeCount::Static((columns * width / 4).div_ceil(256) as u32, 1, 1),
        2 => CubeCount::Static(columns.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1),
        _ => CubeCount::Static(columns.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1),
    };
    unsafe {
        like.client
            .launch_unchecked(Box::new(stage), count, bind(KernelArguments::new()));
    }
}

fn empty(like: &RawTensor, len: usize, dtype: DType) -> RawTensor {
    let bytes = len * dtype.size();
    RawTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        Shape::new([len]),
        like.client.empty(bytes),
        dtype,
    )
}

fn apple10(device: &WgpuDevice) -> bool {
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
        let Some(hal) = (unsafe { device.as_hal::<wgpu::hal::api::Metal>() }) else {
            return false;
        };
        hal.raw_device().supportsFamily(MTLGPUFamily::Apple10)
    })
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Stage {
    rows: usize,
    columns: usize,
    width: usize,
    kind: u32,
}

impl Stage {
    fn cube_dim(self) -> CubeDim {
        match self.kind {
            0 | 1 => CubeDim::new_1d(256),
            2 => CubeDim::new_1d(128),
            _ => CubeDim::new_2d(16, 16),
        }
    }

    fn entry(self) -> &'static str {
        match self.kind {
            0 => "convrot_quantize",
            1 => "convrot_unpack",
            2 => "qwen_tensorops_int8",
            _ => "convrot_epilogue",
        }
    }
}

impl KernelMetadata for Stage {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(self.cube_dim())
    }
    fn address_type(&self) -> StorageType {
        UIntKind::U32.into()
    }
}

impl CubeTask<AutoCompiler> for Stage {
    fn compile(
        &self,
        compiler: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> Result<CompiledKernel<AutoCompiler>, cubecl::CompilationError> {
        assert!(matches!(compiler, AutoCompiler::Msl(_)));
        let cube_dim = self.cube_dim();
        let elems: &[_] = match self.kind {
            0 => &[Elem::F32, Elem::U8, Elem::F32, Elem::I32],
            1 => &[Elem::I32, Elem::U8],
            2 => &[Elem::U8, Elem::U8, Elem::I32],
            _ => &[
                Elem::I32,
                Elem::F32,
                Elem::I32,
                Elem::F32,
                Elem::I32,
                Elem::F32,
            ],
        };
        let repr = MslComputeKernel {
            tensor_maps: vec![],
            buffers: elems
                .iter()
                .enumerate()
                .map(|(id, elem)| KernelArg {
                    id: id as u32,
                    item: Item::scalar(*elem, true),
                    size: None,
                    vis: if id + 1 == elems.len() {
                        cubecl::prelude::Visibility::ReadWrite
                    } else {
                        cubecl::prelude::Visibility::Read
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
            kernel_name: self.entry().into(),
        };
        Ok(CompiledKernel {
            entrypoint_name: self.entry().into(),
            debug_name: Some(self.entry()),
            source: format!(
                "#define ROWS {}\n#define COLS {}\n#define INNER {}\n{SOURCE}",
                self.rows, self.columns, self.width
            ),
            repr: Some(AutoRepresentation::Msl(repr)),
            cube_dim,
            debug_info: None,
        })
    }
}
