//! Native presentation of the model-owned checkpoint loading stages.
use burn::tensor::backend::Backend;
use omni_r2t2::{Model, Precision, Result};
use std::{path::Path, sync::atomic::AtomicBool};
pub fn load<B: Backend>(
    path: &Path,
    precision: Precision,
    device: &B::Device,
    cancel: &AtomicBool,
    report: impl Fn(f64, String),
) -> Result<Model<B>> {
    Model::from_directory_cancellable(path, precision, device, cancel, |progress, message| {
        report(progress, message.into())
    })
}
