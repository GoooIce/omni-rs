//! Metal/ANE tensor handoff. IOSurface is the allocation; Metal only wraps it.
//! Imported outputs are immutable, with an Arc lease retained by the MTLBuffer
//! itself until the last queued GPU reference is released. Layout conversion
//! happens on the GPU and never materializes activation arrays on the CPU.
use crate::{
    runtime::{Graph, Surface},
    storage::{Layout, NativeTensor},
};
use burn_backend::{DType, Device, Shape, TensorMetadata, ops::FloatTensorOps};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use burn_fusion::{FusionBackend, client::GlobalFusionClient};
use burn_ir::{BackendIr, TensorIr, TensorStatus};
use burn_router::{Runner, RunnerClient};
use cubecl::{
    prelude::*,
    wgpu::{WgpuDevice, WgpuRuntime, WgpuServer},
};
use cubecl_common::{device_handle::DeviceHandle, stream_id::StreamId};
use half::f16;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLBuffer;
use std::{
    any::{Any, TypeId},
    ffi::{CStr, c_char, c_void},
    sync::Arc,
};

type Metal = burn_wgpu::Metal;
type Raw = CubeBackend<WgpuRuntime, f32, i32, u8>;
type FusionRuntime = <Raw as FusionBackend>::FusionRuntime;

unsafe extern "C" {
    fn burn_direct_surface_info(
        surface: *mut c_void,
        values: *mut usize,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn burn_direct_surface_metal_buffer(
        surface: *mut c_void,
        device: *mut c_void,
        owner: *mut c_void,
        release: unsafe extern "C" fn(*mut c_void),
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
}

pub(crate) fn enabled<B: BackendIr>() -> bool {
    TypeId::of::<B>() == TypeId::of::<Metal>()
}

fn surface_info(surface: &Surface) -> Result<[usize; 13], String> {
    let mut info = [0; 13];
    let mut error = [0; 4096];
    let result = unsafe {
        burn_direct_surface_info(
            surface.raw_handle(),
            info.as_mut_ptr(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if result != 0 {
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned());
    }
    if info[4..8].iter().any(|n| n % 2 != 0) {
        return Err("ANE surface has an odd FP16 byte stride".into());
    }
    Ok(info)
}

unsafe extern "C" fn release_surface(owner: *mut c_void) {
    // Exactly one Arc was transferred to the successful native buffer wrapper.
    drop(unsafe { Arc::from_raw(owner.cast::<Surface>()) });
}

fn import(
    surface: &Arc<Surface>,
    device: &WgpuDevice,
    info: &[usize; 13],
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let client = WgpuRuntime::client(device);
    let server = DeviceHandle::<WgpuServer>::new(device.to_id());
    let wgpu_device = server
        .submit_blocking(|server| server.wgpu_device())
        .unwrap();
    let mut error = [0; 4096];
    let hal = unsafe { wgpu_device.as_hal::<wgpu::hal::api::Metal>() }
        .ok_or("ANE sharing needs Metal")?;
    let owner = Arc::into_raw(surface.clone()).cast_mut().cast();
    let ptr = unsafe {
        burn_direct_surface_metal_buffer(
            surface.raw_handle(),
            Retained::as_ptr(hal.raw_device()).cast_mut().cast(),
            owner,
            release_surface,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    drop(hal);
    if ptr.is_null() {
        unsafe {
            release_surface(owner);
        }
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned());
    }
    // The +1 native result owns the Arc and the physical IOSurface allocation.
    let raw = unsafe { Retained::<ProtocolObject<dyn MTLBuffer>>::from_raw(ptr.cast()).unwrap() };
    let bytes = info[12] as u64;
    let descriptor = wgpu::BufferDescriptor {
        label: Some("Burn shared ANE tensor"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };
    let hal_buffer = unsafe { wgpu::hal::metal::Device::buffer_from_raw(raw, bytes) };
    let buffer = unsafe {
        wgpu_device.create_buffer_from_hal::<wgpu::hal::api::Metal>(hal_buffer, &descriptor)
    };
    // New commands only see completed ANE outputs, or a private graph input
    // explicitly filled below. No outside producer may mutate a leased output.
    let stream = StreamId::current();
    let handle = server
        .submit_blocking(move |server| unsafe { server.import_buffer(buffer, stream) })
        .unwrap();
    Ok(CubeTensor::new_contiguous(
        client,
        device.clone(),
        Shape::from([info[12] / 2]),
        handle,
        DType::F16,
    ))
}

// Metadata stores canonical [B,C,H,W] extents, reported native strides, and
// valid/capacity segments. It supports compiler reshapes and KV padding holes.
fn metadata(layout: &Layout, info: &[usize; 13]) -> Result<Vec<u8>, String> {
    let logical = layout.physical_shape()?;
    let axis = layout
        .axis
        .map_or(4, |a| crate::mil::axis(layout.shape.len(), a));
    let mut physical = logical;
    if axis < 4 {
        let mut valid = 0usize;
        let mut capacity = 0usize;
        for s in &layout.segments {
            if s.valid > s.capacity {
                return Err("Invalid shared tensor padding".into());
            }
            valid = valid
                .checked_add(s.valid)
                .ok_or("Shared tensor size overflow")?;
            capacity = capacity
                .checked_add(s.capacity)
                .ok_or("Shared tensor size overflow")?;
        }
        if valid != logical[axis] {
            return Err("Shared tensor logical extent mismatch".into());
        }
        physical[axis] = capacity;
    } else if !layout.segments.is_empty() {
        return Err("Shared tensor segments need an axis".into());
    }
    if physical != info[8..12] || logical.contains(&0) {
        return Err("Shared tensor surface extent mismatch".into());
    }
    let mut values = vec![
        layout.shape.len(),
        layout.shape.iter().product(),
        axis,
        layout.segments.len(),
    ];
    values.extend(logical);
    values.extend_from_slice(&info[8..12]);
    values.extend_from_slice(&info[..4]);
    values.extend(info[4..8].iter().map(|n| n / 2));
    for segment in &layout.segments {
        values.extend([segment.valid, segment.capacity]);
    }
    values
        .into_iter()
        .map(|v| {
            u32::try_from(v)
                .map(u32::to_le_bytes)
                .map_err(|_| "Shared layout exceeds u32 addressing".to_string())
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|v| v.into_iter().flatten().collect())
}

#[cube]
fn offset(index: usize, m: &Array<u32>) -> usize {
    let mut position = Array::<usize>::new(4usize);
    let mut remaining = index;
    if m[0] < 4 {
        position[1] = remaining % m[5] as usize;
        remaining /= m[5] as usize;
        position[3] = remaining % m[7] as usize;
        remaining /= m[7] as usize;
        position[2] = 0usize;
        position[0] = remaining;
    } else {
        for d in 0..4usize {
            let dim = 3usize - d;
            position[dim] = remaining % m[4 + dim] as usize;
            remaining /= m[4 + dim] as usize;
        }
    }
    let axis = m[2] as usize;
    if axis < 4 {
        let mut valid = position[axis];
        let mut physical = 0usize;
        let mut found = false;
        for s in 0..m[3] as usize {
            if !found {
                if valid < m[20 + 2 * s] as usize {
                    position[axis] = physical + valid;
                    found = true;
                } else {
                    valid -= m[20 + 2 * s] as usize;
                    physical += m[21 + 2 * s] as usize;
                }
            }
        }
    }
    let mut flat = ((position[0] * m[9] as usize + position[1]) * m[10] as usize + position[2])
        * m[11] as usize
        + position[3];
    let w = flat % m[15] as usize;
    flat /= m[15] as usize;
    let h = flat % m[14] as usize;
    flat /= m[14] as usize;
    let c = flat % m[13] as usize;
    let b = flat / m[13] as usize;
    b * m[16] as usize + c * m[17] as usize + h * m[18] as usize + w
}

#[cube(launch_unchecked)]
fn clear_surface(out: &mut Tensor<f16>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f16::new(0.0f32);
    }
}

#[cube(launch_unchecked)]
fn transfer(input: &Tensor<f16>, out: &mut Tensor<f16>, m: &Array<u32>, #[comptime] pack: bool) {
    if ABSOLUTE_POS < m[1] as usize {
        let position = offset(ABSOLUTE_POS, m);
        if pack {
            out[position] = input[ABSOLUTE_POS];
        } else {
            out[ABSOLUTE_POS] = input[position];
        }
    }
}

fn copy_layout(
    input: CubeTensor<WgpuRuntime>,
    out: CubeTensor<WgpuRuntime>,
    layout: &Layout,
    info: &[usize; 13],
    pack: bool,
) -> Result<(), String> {
    let bytes = metadata(layout, info)?;
    let handle = input.client.create_from_slice(&bytes);
    let count = layout.shape.iter().product::<usize>();
    // Shapes and byte strides are validated by the native surface constructor;
    // the mapping touches only logical coordinates, including segmented padding.
    unsafe {
        if pack {
            clear_surface::launch_unchecked::<WgpuRuntime>(
                &input.client,
                CubeCount::Static((info[12] / 2).div_ceil(256) as u32, 1, 1),
                CubeDim::new_1d(256),
                out.clone().into_tensor_arg(),
            );
        }
        transfer::launch_unchecked::<WgpuRuntime>(
            &input.client,
            CubeCount::Static(count.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            input.clone().into_tensor_arg(),
            out.into_tensor_arg(),
            ArrayArg::from_raw_parts(handle, bytes.len() / 4),
            pack,
        );
    }
    Ok(())
}

fn export_raw(
    tensor: &NativeTensor,
    device: &WgpuDevice,
) -> Result<(CubeTensor<WgpuRuntime>, bool), String> {
    let info = surface_info(&tensor.surface)?;
    let mut raw = import(&tensor.surface, device, &info)?;
    let shape = Shape::from(tensor.layout.shape.clone());
    let affine = info[..4] == info[8..12] && tensor.layout.segments.len() <= 1;
    if affine {
        let strides = match tensor.layout.shape.len() {
            1 => vec![info[5] / 2],
            2 => vec![1, info[5] / 2],
            3 => vec![info[4] / 2, 1, info[5] / 2],
            4 => info[4..8].iter().map(|n| n / 2).collect(),
            _ => return Err("Unsupported shared tensor rank".into()),
        };
        raw.meta = Box::new(burn_std::Metadata::new(shape, strides));
    } else {
        let out = CubeTensor::new_contiguous(
            raw.client.clone(),
            device.clone(),
            shape.clone(),
            raw.client.empty(shape.num_elements() * 2),
            DType::F16,
        );
        copy_layout(raw, out.clone(), &tensor.layout, &info, false)?;
        raw = out;
    }
    let zero_copy = affine && tensor.dtype == DType::F16;
    if tensor.dtype != DType::F16 {
        raw = Raw::float_cast(raw, tensor.dtype.into());
    }
    Ok((raw, zero_copy))
}

pub(crate) fn materialize<B: BackendIr>(
    runner: &Runner<B>,
    id: burn_ir::TensorId,
    tensor: &NativeTensor,
) -> Result<bool, String> {
    assert!(enabled::<B>());
    let device = runner.device();
    let device = (&device as &dyn Any).downcast_ref::<WgpuDevice>().unwrap();
    let (raw, zero_copy) = export_raw(tensor, device)?;
    let shape = raw.shape();
    let dtype = raw.dtype;
    let client = GlobalFusionClient::<FusionRuntime>::load(device);
    let handle = client.register_external_tensor(Raw::float_tensor_handle(raw), shape, dtype);
    let handle: Box<dyn Any> = Box::new(handle);
    runner.register_tensor_handle_id(id, *handle.downcast::<B::Handle>().unwrap());
    Ok(zero_copy)
}

pub(crate) fn prepare_input<B: BackendIr>(
    runner: &Runner<B>,
    tensor: &TensorIr,
    native: Option<&NativeTensor>,
    layout: Layout,
    graph: &Graph,
    index: usize,
) -> Result<(NativeTensor, bool), String> {
    assert!(enabled::<B>());
    let device = runner.device();
    let device = (&device as &dyn Any).downcast_ref::<WgpuDevice>().unwrap();
    let surface = graph.input_surface(index)?;
    let info = surface_info(&surface)?;
    let raw = if let Some(native) = native {
        let same_layout = native.layout.shape == layout.shape
            && native.layout.axis == layout.axis
            && native.layout.segments.len() == layout.segments.len()
            && native
                .layout
                .segments
                .iter()
                .zip(&layout.segments)
                .all(|(a, b)| a.valid == b.valid && a.capacity == b.capacity);
        if same_layout && surface_info(&native.surface)?[..12] == info[..12] {
            return Ok((native.clone(), false));
        }
        export_raw(native, device)?.0
    } else {
        let mut tensor = tensor.clone();
        tensor.status = TensorStatus::ReadOnly;
        let handle = runner.get_tensor_handle(&tensor);
        let handle = (&handle as &dyn Any)
            .downcast_ref::<<Metal as BackendIr>::Handle>()
            .unwrap()
            .clone();
        handle.client.clone().resolve_tensor_float::<Raw>(handle)
    };
    let raw = if raw.dtype != DType::F16 {
        Raw::float_cast(raw, burn_backend::FloatDType::F16)
    } else {
        raw
    };
    let raw = burn_cubecl::kernel::into_contiguous(raw);
    let target = import(&surface, device, &info)?;
    copy_layout(raw, target, &layout, &info, true)?;
    Ok((
        NativeTensor {
            surface,
            layout,
            dtype: tensor.dtype,
        },
        true,
    ))
}

pub(crate) fn synchronize<B: BackendIr>(runner: &Runner<B>) -> Result<(), String> {
    runner.sync().map_err(|e| e.to_string())
}
