//! Apple10 gate for the CubeCL 64×64 int8 `matmul2d` lowering.
use crate::RawTensor;
use cubecl::Runtime;
use cubecl::{
    device::Device as _,
    device_handle::DeviceHandle,
    wgpu::{WgpuDevice, WgpuRuntime, WgpuServer},
};
use objc2_metal::{MTLDevice, MTLGPUFamily};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

pub(crate) fn matmul2d_lowered(device: &WgpuDevice) -> bool {
    if std::env::var("OMNI_CONVROT_TENSOROPS").as_deref() == Ok("0") {
        return false;
    }
    apple10(device)
}

pub(crate) fn enabled(x: &RawTensor) -> bool {
    matmul2d_lowered(&x.device)
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
