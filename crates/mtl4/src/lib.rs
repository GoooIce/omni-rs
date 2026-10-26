//! #54 M0: the Metal 4 tensor path — device, MSL compile, shared-buffer
//! f16 tensors, argument tables, one-shot compute dispatch.
//!
//! Every bound call here is a header-verified MTL4 API (`MacOSX27.sdk`).
//! Two calls the generated objc2-metal 0.3.2 layer predates —
//! `MTLTensorBufferAttachments` and `newTensorWithDescriptor:attachments:`
//! — go through raw `msg_send!` in `tensor_f16`, gathered in one place so
//! the eventual binding refresh swaps a single function.
//!
//! Extent order throughout is MTL's: **the first dimension is the
//! innermost** (column-major from the host side).
//!
//! The `MTL4MachineLearningCommandEncoder` endpoint is deliberately absent:
//! on this OS build its pipeline creation asks the library for a private
//! `_MTLMLLibrary` executable (`executableWithDeviceSelection:`), which no
//! public source-compilation path produces — it is the Core ML-model
//! dispatch channel. Custom MSL tensor functions dispatch through the
//! compute encoder used here, matching Apple's "inline ML" sample.

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{AnyThread, class, msg_send};
use objc2_foundation::{NSInteger, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor, MTL4CompilerTaskOptions,
    MTL4ComputeCommandEncoder, MTL4ComputePipelineDescriptor, MTL4LibraryDescriptor,
    MTL4LibraryFunctionDescriptor, MTLBuffer, MTLComputePipelineState, MTLDevice,
    MTLResourceOptions, MTLSize, MTLTensor, MTLTensorDataType, MTLTensorDescriptor,
    MTLTensorExtents, MTLTensorUsage,
};

unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Option<Retained<ProtocolObject<dyn MTLDevice>>>;
}

fn extents(values: &[NSInteger]) -> Option<Retained<MTLTensorExtents>> {
    unsafe {
        MTLTensorExtents::initWithRank_values(
            MTLTensorExtents::alloc(),
            values.len(),
            values.as_ptr(),
        )
    }
}

fn mtl_size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

/// The default device plus its MTL4 compiler and queue.
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
}

/// A compiled cooperative-tensor compute function.
pub struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    threads: MTLSize,
}

/// An f16 rank-2 tensor backed by a shared (host-visible) `MTLBuffer`.
pub struct Tensor {
    tensor: Retained<ProtocolObject<dyn MTLTensor>>,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    elements: usize,
}

impl Device {
    pub fn new() -> Result<Device, String> {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.ok_or("no default Metal device")?;
        let queue = device
            .newMTL4CommandQueue()
            .ok_or("no MTL4 command queue")?;
        let allocator = device
            .newCommandAllocator()
            .ok_or("no MTL4 command allocator")?;
        let descriptor = MTL4CompilerDescriptor::new();
        let compiler = device
            .newCompilerWithDescriptor_error(&descriptor)
            .map_err(|error| format!("compiler: {error}"))?;
        Ok(Device {
            device,
            queue,
            compiler,
            allocator,
        })
    }

    /// Compile an MSL source string and build the compute pipeline for one
    /// of its functions. `threads_per_threadgroup` lands on
    /// `requiredThreadsPerThreadgroup` — mandatory for cooperative tensors.
    pub fn compile(
        &self,
        source: &str,
        function: &str,
        threads_per_threadgroup: usize,
    ) -> Result<Pipeline, String> {
        let library_descriptor = MTL4LibraryDescriptor::new();
        library_descriptor.setSource(Some(&NSString::from_str(source)));
        let library = self
            .compiler
            .newLibraryWithDescriptor_error(&library_descriptor)
            .map_err(|error| format!("MSL compile: {error}"))?;

        let function_descriptor = MTL4LibraryFunctionDescriptor::new();
        function_descriptor.setName(Some(&NSString::from_str(function)));
        function_descriptor.setLibrary(Some(&library));
        let pipeline_descriptor = MTL4ComputePipelineDescriptor::new();
        pipeline_descriptor.setComputeFunctionDescriptor(Some(&function_descriptor));
        pipeline_descriptor.setRequiredThreadsPerThreadgroup(mtl_size(threads_per_threadgroup));
        let state = self
            .compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(
                &pipeline_descriptor,
                None::<&MTL4CompilerTaskOptions>,
            )
            .map_err(|error| format!("pipeline: {error}"))?;
        Ok(Pipeline {
            state,
            threads: mtl_size(threads_per_threadgroup),
        })
    }

    /// An f16 `[rows, cols]` tensor over a shared buffer; `data` fills it
    /// (or zeros when `None`). Row bytes must be a multiple of 64 — the ML
    /// tensor stride rule.
    pub fn tensor_f16(
        &self,
        dims: [usize; 2],
        data: Option<&[half::f16]>,
    ) -> Result<Tensor, String> {
        let [rows, cols] = dims;
        let elements = rows * cols;
        if cols * 2 % 64 != 0 {
            return Err(format!(
                "row stride {cols}*2 B not a multiple of 64 (ML tensor rule)"
            ));
        }
        let buffer = self
            .device
            .newBufferWithLength_options(elements * 2, MTLResourceOptions::StorageModeShared)
            .ok_or("buffer alloc")?;
        unsafe {
            let base = buffer.contents().as_ptr().cast::<u8>();
            match data {
                Some(data) => {
                    debug_assert_eq!(data.len(), elements);
                    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, base, elements * 2);
                }
                None => std::ptr::write_bytes(base, 0, elements * 2),
            }
        }

        let descriptor = MTLTensorDescriptor::new();
        let dimensions = extents(&[cols as NSInteger, rows as NSInteger]).ok_or("extents")?;
        let strides = extents(&[1, cols as NSInteger]).ok_or("strides")?;
        descriptor.setDataType(MTLTensorDataType::Float16);
        descriptor.setDimensions(&dimensions);
        descriptor.setStrides(Some(&strides));
        descriptor.setUsage(MTLTensorUsage::Compute | MTLTensorUsage::MachineLearning);

        // Not yet in objc2-metal 0.3.2: buffer-backed tensors.
        let attachments: Retained<AnyObject> =
            unsafe { msg_send![class!(MTLTensorBufferAttachments), new] };
        unsafe {
            let _: () = msg_send![
                &attachments,
                setBuffer: &*buffer,
                offset: 0usize,
                forPlane: 0usize // MTLTensorPlaneTypeData
            ];
        }
        let mut error: Option<Retained<objc2_foundation::NSError>> = None;
        let tensor: Option<Retained<ProtocolObject<dyn MTLTensor>>> = unsafe {
            msg_send![
                &*self.device,
                newTensorWithDescriptor: &*descriptor,
                attachments: &*attachments,
                error: &mut error
            ]
        };
        let tensor =
            tensor.ok_or_else(|| format!("newTensorWithDescriptor:attachments: {error:?}"))?;
        Ok(Tensor {
            tensor,
            buffer,
            elements,
        })
    }

    /// Bind `tensors` at consecutive buffer indices and dispatch one
    /// threadgroup of `pipeline`'s required size.
    fn argument_table(
        &self,
        tensors: &[&Tensor],
    ) -> Result<Retained<ProtocolObject<dyn MTL4ArgumentTable>>, String> {
        let table_descriptor = MTL4ArgumentTableDescriptor::new();
        table_descriptor.setMaxBufferBindCount(tensors.len());
        let table = self
            .device
            .newArgumentTableWithDescriptor_error(&table_descriptor)
            .map_err(|error| format!("argument table: {error}"))?;
        for (index, tensor) in tensors.iter().enumerate() {
            unsafe { table.setResource_atBufferIndex(tensor.tensor.gpuResourceID(), index) };
        }
        Ok(table)
    }

    /// MTL4 replaced host-side `waitUntilCompleted` with commit feedback: a
    /// block registered on the commit options fires when the workload
    /// completes, and this side spins on the flag.
    fn commit_and_wait(
        &self,
        command: &ProtocolObject<dyn MTL4CommandBuffer>,
    ) -> Result<(), String> {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let handler = block2::RcBlock::new(move |_feedback: *mut AnyObject| {
            flag.store(true, Ordering::SeqCst);
        });
        let options: Retained<AnyObject> = unsafe { msg_send![class!(MTL4CommitOptions), new] };
        unsafe {
            let _: () = msg_send![&options, addFeedbackHandler: &*handler];
        }
        let buffers: [&ProtocolObject<dyn MTL4CommandBuffer>; 1] = [command];
        let _: () = unsafe {
            msg_send![
                &*self.queue,
                commit: buffers.as_ptr(),
                count: 1usize,
                options: &*options
            ]
        };
        let started = std::time::Instant::now();
        while !done.load(Ordering::SeqCst) {
            if started.elapsed().as_secs_f32() > 10.0 {
                return Err("dispatch did not complete within 10s".into());
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Bind `tensors` at consecutive buffer indices and dispatch a
    /// `groups_x x groups_y` grid of threadgroups of `pipeline`'s required
    /// size (the kernel's `threadgroup_position_in_grid` does the tiling).
    pub fn run(
        &self,
        pipeline: &Pipeline,
        tensors: &[&Tensor],
        groups: [usize; 2],
    ) -> Result<(), String> {
        let table = self.argument_table(tensors)?;
        let command = self.device.newCommandBuffer().ok_or("command buffer")?;
        command.beginCommandBufferWithAllocator(&self.allocator);
        let encoder = command.computeCommandEncoder().ok_or("compute encoder")?;
        unsafe {
            encoder.setComputePipelineState(&pipeline.state);
            encoder.setArgumentTable(Some(&table));
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups[0],
                    height: groups[1],
                    depth: 1,
                },
                MTLSize {
                    width: pipeline.threads.width,
                    height: 1,
                    depth: 1,
                },
            );
            let _: () = msg_send![&*encoder, endEncoding];
        }
        let _: () = unsafe { msg_send![&*command, endCommandBuffer] };
        self.commit_and_wait(&command)
    }
}

impl Tensor {
    pub fn read_f16(&self) -> Vec<half::f16> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().as_ptr().cast::<u8>(),
                self.elements * 2,
            )
        };
        let mut out = vec![half::f16::ZERO; self.elements];
        for (element, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *element = half::f16::from_le_bytes([pair[0], pair[1]]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_and_tensor_exist() {
        let device = Device::new().expect("Metal 4 device");
        let tensor = device.tensor_f16([2, 64], None).expect("tensor");
        assert_eq!(tensor.read_f16(), vec![half::f16::ZERO; 128]);
    }
}
