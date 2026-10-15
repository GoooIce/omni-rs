//! Synchronous native graphs with owned, bounded input/output buffers.

// Test-only, thread-local injection exercises the real output lease and retry
// path without exhausting the host's IOSurface resources. Production has no
// allocation failure switch.
#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
std::thread_local! {
    static FAIL_SURFACE_DUPLICATES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_SURFACE_OPENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Weak interning: only resident graphs keep native weight allocations alive.
/// A content hash includes blob headers/layout, so different immutable weights
/// and offsets cannot accidentally share a descriptor's weight data.
#[derive(Default)]
pub(crate) struct WeightPool {
    entries: std::collections::HashMap<[u8; 32], std::sync::Weak<Weights>>,
    pub hits: u64,
    pub misses: u64,
}
impl WeightPool {
    pub fn intern(&mut self, data: &[u8]) -> Result<Option<std::sync::Arc<Weights>>, String> {
        use sha2::{Digest, Sha256};
        use std::sync::Arc;
        if data.is_empty() {
            return Ok(None);
        }
        let key: [u8; 32] = Sha256::digest(data).into();
        if let Some(weights) = self.entries.get(&key).and_then(std::sync::Weak::upgrade) {
            self.hits += 1;
            return Ok(Some(weights));
        }
        // Keep the weak index bounded by the currently resident weight blobs.
        self.entries.retain(|_, weights| weights.strong_count() > 0);
        let weights = Arc::new(Weights::new(data)?);
        self.entries.insert(key, Arc::downgrade(&weights));
        self.misses += 1;
        Ok(Some(weights))
    }
    pub fn live(&self) -> (usize, usize) {
        self.entries
            .values()
            .filter_map(std::sync::Weak::upgrade)
            .fold((0, 0), |(count, bytes), w| (count + 1, bytes + w.len))
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

pub(crate) enum InputData {
    Host(Vec<half::f16>),
    Native(crate::storage::NativeTensor),
}
pub(crate) struct DirectInput {
    pub offset: usize,
    pub layout: crate::storage::Layout,
    pub data: InputData,
}

pub(crate) enum Source {
    CoreMl(Vec<u8>),
    Direct(crate::mil::Model),
}
impl Source {
    pub fn len(&self) -> usize {
        match self {
            Self::CoreMl(v) => v.len(),
            Self::Direct(m) => m.len(),
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use crate::AneOptions;
    use sha2::{Digest, Sha256};
    use std::{
        ffi::{CStr, CString, c_char, c_void},
        ptr::NonNull,
        sync::Arc,
    };
    unsafe extern "C" {
        fn burn_direct_open(
            mil: *const u8,
            mil_len: usize,
            weights: *mut c_void,
            io: *const u8,
            io_len: usize,
            error: *mut c_char,
            capacity: usize,
        ) -> *mut c_void;
        fn burn_direct_weights_create(
            bytes: *const u8,
            length: usize,
            error: *mut c_char,
            capacity: usize,
        ) -> *mut c_void;
        fn burn_direct_weights_release(handle: *mut c_void);
        fn burn_direct_predict_f16(
            handle: *mut c_void,
            input: *const u16,
            input_count: usize,
            output: *mut u16,
            output_count: usize,
            timings: *mut f64,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
        fn burn_direct_info(handle: *mut c_void, kind: i32) -> f64;
        fn burn_direct_output_retain(handle: *mut c_void, index: usize) -> *mut c_void;
        fn burn_direct_surface_clone(
            handle: *mut c_void,
            error: *mut c_char,
            capacity: usize,
        ) -> *mut c_void;
        fn burn_direct_surface_read(
            handle: *mut c_void,
            output: *mut u16,
            count: usize,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
        fn burn_direct_surface_release(handle: *mut c_void);
        fn burn_direct_predict_surfaces(
            handle: *mut c_void,
            inputs: *const InputBinding,
            input_count: usize,
            outputs: *const *mut c_void,
            output_count: usize,
            timings: *mut f64,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
        fn burn_direct_set_loaded(
            handle: *mut c_void,
            loaded: i32,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
        fn burn_direct_close(handle: *mut c_void);
        fn burn_ane_open(
            path: *const c_char,
            cache: *const c_char,
            cache_budget: u64,
            error: *mut c_char,
            capacity: usize,
        ) -> *mut c_void;
        fn burn_ane_predict(
            handle: *mut c_void,
            input: *const f32,
            input_count: usize,
            output: *mut f32,
            output_count: usize,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
        fn burn_ane_plan_count(handle: *mut c_void, kind: i32) -> usize;
        fn burn_ane_close(handle: *mut c_void);
        fn burn_ane_assignments(handle: *mut c_void, output: *mut u8, capacity: usize) -> usize;
        fn burn_ane_timing(handle: *mut c_void, kind: i32) -> f64;
    }
    pub(crate) struct Graph {
        handle: NonNull<c_void>,
        pub direct: bool,
        loaded: bool,
        direct_inputs: Vec<crate::mil::Io>,
        direct_outputs: Vec<crate::mil::Io>,
        output_pools: Vec<Vec<Arc<Surface>>>,
        // The descriptor independently retains the same immutable NSData.
        // This Arc provides ownership and interning visibility on the Rust side.
        weights: Option<Arc<Weights>>,
        pub output_allocations: u64,
        pub output_reuses: u64,
        pub ane_operations: usize,
        pub cpu_operations: usize,
        pub ane_compute_operations: usize,
        pub disk_cache_hit: bool,
        pub assignments: std::collections::BTreeMap<String, u64>,
        pub timings: [f64; 4],
        pub prediction_timings: [f64; 3],
    }
    // The enclosing Runner mutex serializes all access; Swift retains the model.
    unsafe impl Send for Graph {}

    pub(crate) struct Weights {
        handle: NonNull<c_void>,
        pub len: usize,
    }
    // The handle owns immutable NSData, independently of the input byte slice.
    unsafe impl Send for Weights {}
    unsafe impl Sync for Weights {}
    impl Weights {
        pub fn new(data: &[u8]) -> Result<Self, String> {
            let mut error = [0 as c_char; 4096];
            let handle = unsafe {
                burn_direct_weights_create(
                    data.as_ptr(),
                    data.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let handle = NonNull::new(handle).ok_or_else(|| {
                unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })?;
            Ok(Self {
                handle,
                len: data.len(),
            })
        }
    }
    impl Drop for Weights {
        fn drop(&mut self) {
            unsafe { burn_direct_weights_release(self.handle.as_ptr()) }
        }
    }

    pub(crate) struct Surface(NonNull<c_void>);
    // Leased surfaces are immutable. Only a Graph with the sole Rust Arc may
    // bind a surface as an output; its owning Runner serializes prediction.
    unsafe impl Send for Surface {}
    unsafe impl Sync for Surface {}
    impl Surface {
        #[cfg(feature = "metal-shared")]
        pub(crate) fn raw_handle(&self) -> *mut c_void {
            self.0.as_ptr()
        }
        fn duplicate(&self) -> Result<Self, String> {
            #[cfg(test)]
            if super::FAIL_SURFACE_DUPLICATES.with(|remaining| {
                let count = remaining.get();
                remaining.set(count.saturating_sub(1));
                count > 0
            }) {
                return Err("IOSurface allocation failed".into());
            }
            let mut error = [0 as c_char; 4096];
            let handle = unsafe {
                burn_direct_surface_clone(self.0.as_ptr(), error.as_mut_ptr(), error.len())
            };
            NonNull::new(handle).map(Self).ok_or_else(|| {
                unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
        }
        pub fn read(&self, output: &mut [half::f16]) -> Result<(), String> {
            let mut error = [0 as c_char; 4096];
            let status = unsafe {
                burn_direct_surface_read(
                    self.0.as_ptr(),
                    output.as_mut_ptr().cast(),
                    output.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned());
            }
            Ok(())
        }
    }
    impl Drop for Surface {
        fn drop(&mut self) {
            unsafe { burn_direct_surface_release(self.0.as_ptr()) }
        }
    }

    #[repr(C)]
    struct ViewBinding {
        shape: [usize; 4],
        axis: usize,
        segments: *const crate::storage::Segment,
        segment_count: usize,
    }
    impl ViewBinding {
        fn new(layout: &crate::storage::Layout) -> Result<Self, String> {
            Ok(Self {
                shape: layout.physical_shape()?,
                axis: layout
                    .axis
                    .map_or(4, |axis| crate::mil::axis(layout.shape.len(), axis)),
                segments: layout.segments.as_ptr(),
                segment_count: layout.segments.len(),
            })
        }
    }
    #[repr(C)]
    struct InputBinding {
        surface: *mut c_void,
        host: *const u16,
        host_count: usize,
        source: ViewBinding,
        target: ViewBinding,
    }
    impl Graph {
        #[cfg(feature = "metal-shared")]
        pub(crate) fn input_surface(&self, index: usize) -> Result<Arc<Surface>, String> {
            unsafe extern "C" {
                fn burn_direct_input_retain(handle: *mut c_void, index: usize) -> *mut c_void;
            }
            // The graph owns its input; the returned lease also covers pending
            // Metal packing work. A session serializes packing and prediction.
            let ptr = unsafe { burn_direct_input_retain(self.handle.as_ptr(), index) };
            NonNull::new(ptr)
                .map(|p| Arc::new(Surface(p)))
                .ok_or_else(|| "Missing shared ANE input".into())
        }

        pub fn compile(
            source: &super::Source,
            options: &AneOptions,
            weights: Option<&Arc<Weights>>,
        ) -> Result<Self, String> {
            match source {
                super::Source::CoreMl(bytes) => Self::coreml(bytes, options),
                super::Source::Direct(model) => Self::direct(model, weights),
            }
        }
        fn direct(
            model: &crate::mil::Model,
            weights: Option<&Arc<Weights>>,
        ) -> Result<Self, String> {
            #[cfg(test)]
            if super::FAIL_SURFACE_OPENS.with(|remaining| {
                let count = remaining.get();
                remaining.set(count.saturating_sub(1));
                count > 0
            }) {
                return Err("IOSurface allocation failed".into());
            }
            if weights.map_or(0, |w| w.len) != model.weights.len() {
                return Err("Direct ANE shared weight size mismatch".into());
            }
            let io = serde_json::to_vec(
                &serde_json::json!({"inputs":model.inputs,"outputs":model.outputs}),
            )
            .map_err(|e| e.to_string())?;
            let mut error = [0 as c_char; 4096];
            let handle = unsafe {
                burn_direct_open(
                    model.text.as_ptr(),
                    model.text.len(),
                    weights.map_or(std::ptr::null_mut(), |w| w.handle.as_ptr()),
                    io.as_ptr(),
                    io.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let handle = NonNull::new(handle).ok_or_else(|| {
                unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })?;
            Ok(Self {
                handle,
                direct: true,
                loaded: true,
                direct_inputs: model.inputs.clone(),
                direct_outputs: model.outputs.clone(),
                output_pools: vec![],
                weights: weights.cloned(),
                output_allocations: 0,
                output_reuses: 0,
                ane_operations: 0,
                cpu_operations: 0,
                ane_compute_operations: 0,
                disk_cache_hit: unsafe { burn_direct_info(handle.as_ptr(), 0) != 0. },
                assignments: Default::default(),
                timings: [
                    unsafe { burn_direct_info(handle.as_ptr(), 1) },
                    0.,
                    0.,
                    unsafe { burn_direct_info(handle.as_ptr(), 2) },
                ],
                prediction_timings: [0.; 3],
            })
        }
        fn coreml(bytes: &[u8], options: &AneOptions) -> Result<Self, String> {
            let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
            let path = directory.path().join("graph.mlmodel");
            std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
            let path =
                CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
            // Hash actual protobuf bytes, including weights, never transient Burn IDs.
            let cache = options
                .cache_dir
                .as_ref()
                .map(|root| {
                    let digest = format!("{:x}", Sha256::digest(bytes));
                    CString::new(root.join(digest).as_os_str().as_encoded_bytes())
                })
                .transpose()
                .map_err(|e| e.to_string())?;
            let mut error = [0 as c_char; 4096];
            let handle = unsafe {
                burn_ane_open(
                    path.as_ptr(),
                    cache.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                    options.max_disk_cache_bytes,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let handle = NonNull::new(handle).ok_or_else(|| {
                unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })?;
            let length = unsafe { burn_ane_assignments(handle.as_ptr(), std::ptr::null_mut(), 0) };
            let mut assignments = vec![0u8; length];
            unsafe { burn_ane_assignments(handle.as_ptr(), assignments.as_mut_ptr(), length) };
            // Internal Swift/Rust ABI: the serialized object contains only string keys and counters.
            let assignments = serde_json::from_slice(&assignments).unwrap_or_default();
            Ok(Self {
                handle,
                direct: false,
                loaded: true,
                direct_inputs: vec![],
                direct_outputs: vec![],
                output_pools: vec![],
                weights: None,
                output_allocations: 0,
                output_reuses: 0,
                ane_operations: unsafe { burn_ane_plan_count(handle.as_ptr(), 0) },
                cpu_operations: unsafe { burn_ane_plan_count(handle.as_ptr(), 1) },
                ane_compute_operations: unsafe { burn_ane_plan_count(handle.as_ptr(), 2) },
                disk_cache_hit: unsafe { burn_ane_plan_count(handle.as_ptr(), 3) != 0 },
                assignments,
                timings: std::array::from_fn(|i| unsafe {
                    burn_ane_timing(handle.as_ptr(), i as i32)
                }),
                prediction_timings: [0.; 3],
            })
        }
        pub fn loaded_direct(&self) -> bool {
            self.direct && self.loaded
        }
        pub fn weight_bytes(&self) -> usize {
            self.weights.as_ref().map_or(0, |w| w.len)
        }
        pub fn shared_weights(&self) -> Option<&Arc<Weights>> {
            self.weights.as_ref()
        }
        pub fn predict_surfaces(
            &mut self,
            inputs: &[super::DirectInput],
        ) -> Result<Vec<Arc<Surface>>, String> {
            self.prediction_timings = [0.; 3];
            self.output_allocations = 0;
            self.output_reuses = 0;
            if !self.direct {
                return Err("Native surfaces require Direct ANE".into());
            }
            let bindings = self
                .direct_inputs
                .iter()
                .map(|io| {
                    let index = inputs
                        .binary_search_by_key(&io.offset, |input| input.offset)
                        .map_err(|_| "Missing Direct ANE input")?;
                    let input = &inputs[index];
                    let target = ViewBinding::new(&input.layout)?;
                    match &input.data {
                        super::InputData::Host(values) => {
                            if values.len() != io.shape.iter().product::<usize>() {
                                return Err("Direct ANE host input size mismatch".into());
                            }
                            Ok(InputBinding {
                                surface: std::ptr::null_mut(),
                                host: values.as_ptr().cast(),
                                host_count: values.len(),
                                source: ViewBinding::new(&input.layout)?,
                                target,
                            })
                        }
                        super::InputData::Native(tensor) => Ok(InputBinding {
                            surface: tensor.surface.0.as_ptr(),
                            host: std::ptr::null(),
                            host_count: 0,
                            source: ViewBinding::new(&tensor.layout)?,
                            target,
                        }),
                    }
                })
                .collect::<Result<Vec<_>, String>>()?;
            let initial = self.output_pools.is_empty();
            if initial {
                for index in 0..self.direct_outputs.len() {
                    let handle = unsafe { burn_direct_output_retain(self.handle.as_ptr(), index) };
                    let handle = NonNull::new(handle).ok_or("Missing Direct ANE output")?;
                    self.output_pools.push(vec![Arc::new(Surface(handle))]);
                    self.output_allocations += 1;
                }
            }
            let mut outputs = Vec::with_capacity(self.output_pools.len());
            for pool in &mut self.output_pools {
                let surface = if let Some(surface) = pool.iter().find(|s| Arc::strong_count(s) == 1)
                {
                    self.output_reuses += u64::from(!initial);
                    Arc::clone(surface)
                } else {
                    let surface = Arc::new(pool[0].duplicate()?);
                    pool.push(Arc::clone(&surface));
                    self.output_allocations += 1;
                    surface
                };
                outputs.push(surface);
                // Keep live tensors and at most one extra free buffer per slot.
                // The selected buffer is leased above and cannot be removed.
                let mut free = 0;
                pool.retain(|s| {
                    if Arc::strong_count(s) > 1 {
                        true
                    } else {
                        free += 1;
                        free <= 1
                    }
                });
            }
            let handles: Vec<_> = outputs.iter().map(|s| s.0.as_ptr()).collect();
            let mut error = [0 as c_char; 4096];
            // bindings borrow inputs (including their segment vectors), while
            // outputs lease every writable surface until this call returns.
            let status = unsafe {
                burn_direct_predict_surfaces(
                    self.handle.as_ptr(),
                    bindings.as_ptr(),
                    bindings.len(),
                    handles.as_ptr(),
                    handles.len(),
                    self.prediction_timings.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned());
            }
            Ok(outputs)
        }
        pub fn set_loaded(&mut self, loaded: bool) -> Result<(), String> {
            if !self.direct {
                return Err("Program residency control requires Direct ANE".into());
            }
            let mut error = [0 as c_char; 4096];
            let status = unsafe {
                burn_direct_set_loaded(
                    self.handle.as_ptr(),
                    i32::from(loaded),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned());
            }
            self.loaded = loaded;
            Ok(())
        }
        pub fn predict(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            let mut error = [0 as c_char; 4096];
            if self.direct {
                return Err("Direct ANE requires FP16 graph buffers".into());
            }
            let status = unsafe {
                burn_ane_predict(
                    self.handle.as_ptr(),
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned())
            }
        }
        pub fn predict_f16(
            &mut self,
            input: &[half::f16],
            output: &mut [half::f16],
        ) -> Result<(), String> {
            self.prediction_timings = [0.; 3];
            if !self.direct {
                return Err("FP16 graph buffers require Direct ANE".into());
            }
            // The native bridge copies directly between logical Burn buffers
            // and strided IOSurfaces, including low-rank layout conversion.
            // Each I/O carries its original offset, preserving dead-input holes.
            for io in &self.direct_inputs {
                let count: usize = io.shape.iter().product();
                if io
                    .offset
                    .checked_add(count)
                    .is_none_or(|end| end > input.len())
                {
                    return Err("direct ANE input size mismatch".into());
                }
            }
            let count: usize = self
                .direct_outputs
                .iter()
                .map(|io| io.shape.iter().product::<usize>())
                .sum();
            if count != output.len() {
                return Err("direct ANE output size mismatch".into());
            }
            let mut error = [0 as c_char; 4096];
            // half::f16 is repr(transparent) over u16. The bridge copies bits,
            // uses element counts, and writes exactly three timing values.
            let status = unsafe {
                burn_direct_predict_f16(
                    self.handle.as_ptr(),
                    input.as_ptr().cast(),
                    input.len(),
                    output.as_mut_ptr().cast(),
                    output.len(),
                    self.prediction_timings.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned());
            }
            Ok(())
        }
    }
    impl Drop for Graph {
        fn drop(&mut self) {
            unsafe {
                if self.direct {
                    burn_direct_close(self.handle.as_ptr())
                } else {
                    burn_ane_close(self.handle.as_ptr())
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use apple::{Graph, Surface, Weights};

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) struct Weights {
    pub len: usize,
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl Weights {
    pub fn new(_: &[u8]) -> Result<Self, String> {
        Err("ANE unavailable".into())
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) struct Graph {
    pub direct: bool,
    pub ane_operations: usize,
    pub cpu_operations: usize,
    pub ane_compute_operations: usize,
    pub disk_cache_hit: bool,
    pub assignments: std::collections::BTreeMap<String, u64>,
    pub timings: [f64; 4],
    pub prediction_timings: [f64; 3],
    pub output_allocations: u64,
    pub output_reuses: u64,
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) struct Surface;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl Surface {
    pub fn read(&self, _: &mut [half::f16]) -> Result<(), String> {
        Err("ANE unavailable".into())
    }
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl Graph {
    pub fn predict_surfaces(
        &mut self,
        _: &[DirectInput],
    ) -> Result<Vec<std::sync::Arc<Surface>>, String> {
        Err("ANE unavailable".into())
    }
    pub fn loaded_direct(&self) -> bool {
        false
    }
    pub fn weight_bytes(&self) -> usize {
        0
    }
    pub fn shared_weights(&self) -> Option<&std::sync::Arc<Weights>> {
        None
    }
    pub fn set_loaded(&mut self, _: bool) -> Result<(), String> {
        Err("ANE unavailable".into())
    }
    pub fn compile(
        _: &Source,
        _: &crate::AneOptions,
        _: Option<&std::sync::Arc<Weights>>,
    ) -> Result<Self, String> {
        Err("ANE requires Apple Silicon and macOS 15+".into())
    }
    pub fn predict(&mut self, _: &[f32], _: &mut [f32]) -> Result<(), String> {
        Err("ANE unavailable".into())
    }
    pub fn predict_f16(&mut self, _: &[half::f16], _: &mut [half::f16]) -> Result<(), String> {
        Err("ANE unavailable".into())
    }
}
pub(crate) fn resource_exhausted(error: &str) -> bool {
    error.contains("ANE_RESOURCE_EXHAUSTED:") || surface_exhausted(error)
}

pub(crate) fn surface_exhausted(error: &str) -> bool {
    error == "IOSurface allocation failed" || error.starts_with("IOSurface allocation failed:")
}

#[cfg(test)]
mod resource_error_tests {
    #[test]
    fn allocation_pressure_is_transient_but_layout_errors_are_not() {
        assert!(super::resource_exhausted("ANE_RESOURCE_EXHAUSTED: slots"));
        assert!(super::resource_exhausted("IOSurface allocation failed"));
        assert!(super::surface_exhausted(
            "IOSurface allocation failed: bytes=4096"
        ));
        assert!(!super::surface_exhausted("ANE_RESOURCE_EXHAUSTED: slots"));
        assert!(!super::resource_exhausted(
            "ANE I/O allocation exceeds 4 GiB"
        ));
        assert!(!super::resource_exhausted("Unsupported ANE I/O layout"));
        assert!(!super::resource_exhausted(
            "ANE output surface layout mismatch"
        ));
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod surface_pressure_tests {
    use crate::{Ane, AneDevice, AneOptions, AneRuntime, FallbackPolicy};
    use burn_backend::DType;
    use burn_tensor::{Tensor, TensorData, module};

    fn input(device: &AneDevice, value: f32) -> Tensor<Ane, 2> {
        Tensor::from_data(
            TensorData::new(vec![half::f16::from_f32(value); 64], [2, 32]),
            (device, DType::F16),
        )
    }

    fn weight(device: &AneDevice, scale: f32) -> Tensor<Ane, 2> {
        Tensor::from_data(
            TensorData::new(
                (0..32 * 32)
                    .map(|i| half::f16::from_f32(if i / 32 == i % 32 { scale } else { 0. }))
                    .collect::<Vec<_>>(),
                [32, 32],
            ),
            (device, DType::F16),
        )
    }

    fn check(tensor: &Tensor<Ane, 2>, expected: f32) {
        let values = tensor
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert!(values.iter().all(|&v| v == expected));
    }

    #[test]
    fn surface_pressure_retries_without_losing_outputs_from_an_evicted_graph() {
        // Exercise both an unloaded victim and the all-loaded fallback.
        for slots in [1, 2] {
            surface_pressure_retains_outputs(slots);
        }
    }

    fn surface_pressure_retains_outputs(slots: usize) {
        let device = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket: 0,
                max_loaded_direct_graphs: slots,
                ..Default::default()
            },
        );
        let first_weight = weight(&device, 2.);
        let second_weight = weight(&device, 3.);
        let first = module::linear(input(&device, 1.), first_weight, None);
        <Ane as burn_backend::Backend>::sync(&device).unwrap();
        let second = module::linear(input(&device, 2.), second_weight.clone(), None);
        <Ane as burn_backend::Backend>::sync(&device).unwrap();
        let before = device.stats();
        assert_eq!(before.cached_graphs, 2);
        assert_eq!(before.loaded_direct_graphs, slots);
        assert_eq!(before.resident_native_tensors, 2);

        // Retaining the previous result forces a fresh output surface. The
        // other graph still owns buffers that can be released.
        super::FAIL_SURFACE_DUPLICATES.with(|n| n.set(1));
        let third = module::linear(input(&device, 4.), second_weight.clone(), None);
        check(&third, 12.);
        super::FAIL_SURFACE_DUPLICATES.with(|n| assert_eq!(n.get(), 0));
        let after = device.stats();
        assert_eq!(after.resource_retries - before.resource_retries, 1);
        assert_eq!(
            after.surface_resource_retries - before.surface_resource_retries,
            1
        );
        assert_eq!(after.cache_evictions - before.cache_evictions, 1);
        assert_eq!(after.cached_graphs, 1);
        assert_eq!(after.loaded_direct_graphs, 1);
        assert_eq!(after.cached_failures, 0);
        assert_eq!(after.failure_cache_hits, 0);
        assert_eq!(after.compilation_failures, 0);
        assert_eq!(after.fallback_operations, 0);
        assert_eq!(after.direct_ane_graph_executions, 3);

        let fourth = module::linear(input(&device, 5.), second_weight, None);
        check(&fourth, 15.);
        assert_eq!(device.stats().failure_cache_hits, 0);
        device.clear_cache();
        for (tensor, expected) in [(&first, 2.), (&second, 6.), (&third, 12.), (&fourth, 15.)] {
            check(tensor, expected);
        }
    }

    #[test]
    fn persistent_surface_pressure_is_bounded_and_not_memoized() {
        for (other_graphs, retries) in [(0, 0), (10, 8)] {
            let device = AneDevice::new(
                Default::default(),
                AneOptions {
                    runtime: AneRuntime::Direct,
                    fallback: FallbackPolicy::Error,
                    row_bucket: 0,
                    max_loaded_direct_graphs: 1,
                    ..Default::default()
                },
            );
            let mut retained = Vec::new();
            for i in 0..other_graphs {
                let scale = (10 + i) as f32;
                let output = module::linear(input(&device, 1.), weight(&device, scale), None);
                <Ane as burn_backend::Backend>::sync(&device).unwrap();
                retained.push((output, scale));
            }
            let weights = weight(&device, 3.);
            let old = module::linear(input(&device, 2.), weights.clone(), None);
            <Ane as burn_backend::Backend>::sync(&device).unwrap();
            retained.push((old, 6.));
            let before = device.stats();
            assert_eq!(before.cached_graphs, other_graphs + 1);
            assert_eq!(before.resident_native_tensors, other_graphs + 1);

            super::FAIL_SURFACE_DUPLICATES.with(|n| n.set(100));
            let _failed = module::linear(input(&device, 4.), weights, None);
            let error = <Ane as burn_backend::Backend>::sync(&device).unwrap_err();
            let remaining = super::FAIL_SURFACE_DUPLICATES.with(|n| n.replace(0));
            assert!(error.to_string().contains("IOSurface allocation failed"));
            assert_eq!(100 - remaining, 1 + retries);
            let after = device.stats();
            assert_eq!(
                after.resource_retries - before.resource_retries,
                retries as u64
            );
            assert_eq!(
                after.surface_resource_retries - before.surface_resource_retries,
                retries as u64
            );
            assert_eq!(
                after.cache_evictions - before.cache_evictions,
                retries as u64
            );
            assert_eq!(after.cached_graphs, other_graphs - retries);
            assert_eq!(after.cached_failures, 0);
            assert_eq!(after.failure_cache_hits, 0);
            assert_eq!(after.compilation_failures, 0);
            assert_eq!(
                after.direct_ane_graph_executions,
                before.direct_ane_graph_executions
            );
            assert_eq!(
                after.fallback_reasons.get("IOSurface allocation failed"),
                Some(&1)
            );

            // Error policy deliberately makes the session terminal; no fallback
            // matrix execution or failure-cache entry is allowed.
            assert_eq!(after.fallback_operations, 0);
            device.clear_cache();
            drop(retained);
        }
    }

    #[test]
    fn surface_pressure_during_compile_recovers_or_allows_a_later_ane_retry() {
        // NdArray's fallback supports F32. Prediction lease tests above use
        // strict FP16 so they don't require FP16 support from that backend.
        for (other_graphs, failures, retries, recovered) in
            [(0, 1, 0, false), (2, 1, 1, true), (10, 100, 8, false)]
        {
            let device = AneDevice::new(
                Default::default(),
                AneOptions {
                    runtime: AneRuntime::Direct,
                    fallback: FallbackPolicy::Allow,
                    row_bucket: 0,
                    max_loaded_direct_graphs: 1,
                    ..Default::default()
                },
            );
            let weights = |scale| {
                Tensor::<Ane, 2>::from_data(
                    TensorData::new(
                        (0..32 * 32)
                            .map(|i| if i / 32 == i % 32 { scale } else { 0. })
                            .collect::<Vec<f32>>(),
                        [32, 32],
                    ),
                    &device,
                )
            };
            let compute = |value, weights| {
                module::linear(
                    Tensor::from_data(TensorData::new(vec![value; 64], [2, 32]), &device),
                    weights,
                    None,
                )
            };
            for i in 0..other_graphs {
                let scale = (10 + i) as f32;
                check(&compute(1., weights(scale)), scale);
            }
            let weights = weights(3.);
            let before = device.stats();
            assert_eq!(before.cached_graphs, other_graphs);
            super::FAIL_SURFACE_OPENS.with(|n| n.set(failures));
            check(&compute(4., weights.clone()), 12.);
            let remaining = super::FAIL_SURFACE_OPENS.with(|n| n.replace(0));
            assert_eq!(failures - remaining, (1 + retries).min(failures));
            let after = device.stats();
            assert_eq!(
                after.surface_resource_retries - before.surface_resource_retries,
                retries as u64
            );
            assert_eq!(
                after.cache_evictions - before.cache_evictions,
                retries as u64
            );
            assert_eq!(after.cached_failures, 0);
            assert_eq!(after.failure_cache_hits, 0);
            assert_eq!(
                after.compilation_failures - before.compilation_failures,
                u64::from(!recovered)
            );
            assert_eq!(
                after.direct_ane_graph_executions - before.direct_ane_graph_executions,
                u64::from(recovered)
            );

            check(&compute(5., weights), 15.);
            let end = device.stats();
            assert_eq!(
                end.direct_ane_graph_executions,
                after.direct_ane_graph_executions + 1
            );
            assert_eq!(end.fallback_operations, after.fallback_operations);
            assert_eq!(end.failure_cache_hits, 0);
            device.clear_cache();
        }
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod weight_pool_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn immutable_weight_interning_is_content_based_and_weak() {
        let mut pool = WeightPool::default();
        assert!(pool.intern(&[]).unwrap().is_none());
        let first = pool.intern(&[1, 2, 3, 4]).unwrap().unwrap();
        let again = pool.intern(&[1, 2, 3, 4]).unwrap().unwrap();
        let changed = pool.intern(&[1, 2, 3, 5]).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert!(!Arc::ptr_eq(&first, &changed));
        assert_eq!(pool.live(), (2, 8));
        drop(first);
        assert_eq!(pool.live(), (2, 8));
        drop(again);
        assert_eq!(pool.live(), (1, 4));
        drop(changed);
        assert_eq!(pool.live(), (0, 0));
        let last = pool.intern(&[5, 6]).unwrap().unwrap();
        assert_eq!(
            pool.entries.len(),
            1,
            "dead weak entries must not accumulate"
        );
        assert_eq!(pool.live(), (1, 2));
        assert_eq!((pool.hits, pool.misses), (1, 3));
        drop(last);
        pool.clear();
        assert_eq!(pool.live(), (0, 0));
    }
}
