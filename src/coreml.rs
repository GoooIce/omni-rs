//! macOS Core ML bridge for the audio transformer. Convolutions and decoding use Burn.
use crate::{Error, Result, config::AudioConfig};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    io::Read,
    path::Path,
    ptr::NonNull,
    sync::Mutex,
};

unsafe extern "C" {
    fn r2t2_coreml_open(
        path: *const c_char,
        capacity: isize,
        width: isize,
        output_width: isize,
        error: *mut c_char,
        error_capacity: isize,
    ) -> *mut c_void;
    fn r2t2_coreml_predict(
        handle: *mut c_void,
        input: *const f32,
        tokens: isize,
        output: *mut f32,
        error: *mut c_char,
        error_capacity: isize,
    ) -> i32;
    fn r2t2_coreml_ane_operations(handle: *mut c_void) -> isize;
    fn r2t2_coreml_close(handle: *mut c_void);
}

#[derive(Deserialize)]
struct Manifest {
    format_version: usize,
    tokens: usize,
    width: usize,
    output_dim: usize,
    layers: usize,
    n_window_infer: usize,
    weights_sha256: String,
}

struct Handle(NonNull<c_void>);
// Core ML supports use from any thread. The enclosing mutex serializes access to
// both MLModel and its reusable MLMultiArray buffers; no Swift reference escapes.
unsafe impl Send for Handle {}
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { r2t2_coreml_close(self.0.as_ptr()) }
    }
}

pub struct CoreMlEncoder {
    handle: Mutex<Handle>,
    capacity: usize,
    width: usize,
    output_dim: usize,
    ane_operations: usize,
}

impl CoreMlEncoder {
    pub fn open(directory: &Path, checkpoint: &Path, config: &AudioConfig) -> Result<Self> {
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
        if manifest.format_version != 1
            || manifest.width != config.d_model
            || manifest.output_dim != config.output_dim
            || manifest.layers != config.encoder_layers
            || manifest.n_window_infer != config.n_window_infer
            || manifest.tokens < (config.n_window_infer / 100) * 13
            || manifest.tokens > 4096
        {
            return Err(Error::Config(
                "Core ML encoder manifest does not match this model".into(),
            ));
        }
        let mut file = std::fs::File::open(checkpoint)?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        if format!("{:x}", digest.finalize()) != manifest.weights_sha256 {
            return Err(Error::Checkpoint(
                "Core ML encoder was exported from different weights".into(),
            ));
        }
        let compiled = directory.join("encoder.mlmodelc");
        if !compiled.is_dir() {
            return Err(Error::Config(format!(
                "missing {}; compile the exported Core ML model first",
                compiled.display()
            )));
        }
        let path = CString::new(compiled.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::Config("Core ML path contains a NUL byte".into()))?;
        let mut error = [0 as c_char; 2048];
        // Swift validates model shapes, output type and actual ANE assignment.
        let pointer = unsafe {
            r2t2_coreml_open(
                path.as_ptr(),
                manifest.tokens as isize,
                manifest.width as isize,
                manifest.output_dim as isize,
                error.as_mut_ptr(),
                error.len() as isize,
            )
        };
        let pointer = NonNull::new(pointer).ok_or_else(|| bridge_error(&error))?;
        let handle = Handle(pointer);
        let ane_operations = unsafe { r2t2_coreml_ane_operations(pointer.as_ptr()) } as usize;
        Ok(Self {
            handle: Mutex::new(handle),
            capacity: manifest.tokens,
            width: manifest.width,
            output_dim: manifest.output_dim,
            ane_operations,
        })
    }

    pub fn ane_operations(&self) -> usize {
        self.ane_operations
    }

    pub fn predict(&self, input: &[f32], tokens: usize) -> Result<Vec<f32>> {
        if tokens == 0
            || tokens > self.capacity
            || input.len() != tokens * self.width
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Inference("invalid Core ML encoder input".into()));
        }
        let handle = self
            .handle
            .lock()
            .map_err(|_| Error::Inference("Core ML encoder lock poisoned".into()))?;
        let mut output = vec![0.0; tokens * self.output_dim];
        let mut error = [0 as c_char; 2048];
        let status = unsafe {
            r2t2_coreml_predict(
                handle.0.as_ptr(),
                input.as_ptr(),
                tokens as isize,
                output.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len() as isize,
            )
        };
        if status != 0 {
            return Err(bridge_error(&error));
        }
        Ok(output)
    }
}

fn bridge_error(buffer: &[c_char]) -> Error {
    // The zero-initialized fixed buffer stays NUL terminated in every FFI path.
    let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
    Error::Inference(format!("Core ML: {message}"))
}
