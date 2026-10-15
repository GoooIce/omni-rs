//! Experimental inference backend for Burn, compiling tensor subgraphs to Apple Neural Engine.
//!
//! `Ane<B>` implements Burn's Backend through a custom IR runner. `B` stores tensors and
//! executes unsupported operations (NdArray by default; Metal can be selected by callers).
//! Default execution uses public Core ML APIs. `AneRuntime::Direct` selects an
//! experimental private ANE runtime. Neither path needs Python or a model exporter.
//! ANE calculations use reduced precision; this backend is intended for inference.
mod bucket;
mod cache;
mod compiler;
mod host;
mod ir;
mod kv;
mod mil;
mod proto;
mod runner;
mod runtime;
mod scheduling;
#[cfg(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64"))]
mod shared;
mod storage;

use burn_backend::{Device, DeviceId, DeviceOps};
use burn_ir::BackendIr;
use burn_ndarray::NdArray;
use burn_router::{BackendRouter, get_client};
pub use runner::{AneChannel, AneClient};
pub use scheduling::FallbackScope;
use serde::Serialize;
use std::{
    any::{Any, TypeId},
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

/// ANE graph execution with a configurable Burn fallback backend.
pub type Ane<B = NdArray<f32>> = BackendRouter<AneChannel<B>>;

/// Native compiler and execution API. Direct mode is experimental and uses
/// private Apple interfaces; an incompatible graph follows `fallback` policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AneRuntime {
    #[default]
    CoreMl,
    Direct,
}

/// Whether unsupported operations / unavailable ANE graphs may use the fallback backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// Execute on `B` and record the reason in [`AneStats`].
    Allow,
    /// Surface an ExecutionError at the next read or sync; never silently compute on `B`.
    Error,
}

/// Which loaded Direct program to unload when the fixed slot budget is full.
/// This never evicts a compiled descriptor or changes graph numerics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProgramCachePolicy {
    /// Prefer small programs, then the least recently used within each class.
    #[default]
    PreferSmall,
    /// Least recently used, regardless of serialized size.
    Lru,
    /// Prefer a loaded graph sharing the incoming graph's immutable weight
    /// blob, commonly another shape. Otherwise use PreferSmall. This is only
    /// an unload preference, not graph equivalence. Opt in after profiling.
    ReplaceWeightShapes,
}

impl std::str::FromStr for ProgramCachePolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "prefer-small" => Ok(Self::PreferSmall),
            "lru" => Ok(Self::Lru),
            "replace-weight-shapes" => Ok(Self::ReplaceWeightShapes),
            _ => Err("expected prefer-small, lru, or replace-weight-shapes".into()),
        }
    }
}

/// Per-session graph compiler options. Immutable once the device is created.
#[derive(Clone, Debug)]
pub struct AneOptions {
    /// Try all supported subgraphs, including standalone elementwise graphs.
    /// No cost-based preference for Metal; rejection/failure still follows fallback.
    pub ane_first: bool,
    /// Opt into model-provided latency hints that keep complete dependent
    /// phases on the fallback backend. Off by default for generic ANE callers.
    /// Explicit ANE-first and strict policies take precedence in those hints.
    pub prefer_fallback_phases: bool,
    pub runtime: AneRuntime,
    pub fallback: FallbackPolicy,
    /// Maximum number of cached, successfully compiled graphs. Direct graphs
    /// can remain cached while their ANE program is unloaded.
    pub max_cached_graphs: usize,
    /// Direct runtime program-slot limit, separate from the byte budget. The
    /// local macOS runtime exhausted resources at 128 loaded programs; leave
    /// headroom by default and also recover from transient load failures.
    pub max_loaded_direct_graphs: usize,
    /// Program-slot replacement only; all count and byte budgets still apply.
    pub program_cache_policy: ProgramCachePolicy,
    /// Budget for serialized model bytes in the resident cache. Identical direct
    /// descriptor weight blobs share one immutable allocation, charged once.
    /// Excludes native runtime copies and I/O buffers; not a process RSS limit.
    pub max_cached_bytes: usize,
    /// Independent LRU budget for packed FP16 weights shared by direct graphs.
    /// Zero disables weight reuse. This budget is additional to model cache bytes.
    pub max_cached_weight_bytes: usize,
    /// Rejected graphs have a separate, bounded cache and cannot evict working models.
    pub max_cached_failures: usize,
    /// Optional persistent, content-addressed Core ML compilation cache.
    pub cache_dir: Option<PathBuf>,
    /// Maximum bytes of compiled bundles retained on disk. Open models remain valid.
    pub max_disk_cache_bytes: u64,
    /// Flush a graph before it can grow without bound.
    pub max_graph_operations: usize,
    /// Bound the total size of float32 constant data read for a single compilation.
    pub max_constant_bytes: usize,
    /// Keep graphs whose largest matrix multiplication has fewer rows on the fallback.
    /// Convolution graphs are exempt. One disables this inference scheduling heuristic.
    pub min_compute_rows: usize,
    /// Emit per-graph JSON profiling to stderr.
    pub trace: bool,
    /// Token-independent FFNs and proven self-attention graphs share shapes in
    /// multiples of this value (at least 64 for attention). Zero disables padding.
    /// No audio is buffered.
    pub row_bucket: usize,
    /// Allow single-row projections to share the multi-row bucket. This trades
    /// extra arithmetic for fewer resident programs; opt in after profiling.
    pub pad_single_row: bool,
    /// Direct runtime: round cached attention prefixes to this capacity. Current
    /// tokens occupy a separate segment; a runtime mask excludes unused slots.
    /// Zero disables KV specialization. This is not a context or recording limit.
    pub kv_bucket: usize,
    /// Direct runtime with row padding: keep cached attention in a weight-free
    /// graph so changing current-token counts do not reload large projections.
    pub separate_cached_attention: bool,
    /// Direct runtime: keep one-token projections, cached attention and FFNs
    /// fused. Multi-token prefill still uses separated, row-bucketed programs.
    pub fuse_single_token_attention: bool,
}
impl Default for AneOptions {
    fn default() -> Self {
        Self {
            ane_first: false,
            prefer_fallback_phases: false,
            runtime: AneRuntime::CoreMl,
            fallback: FallbackPolicy::Allow,
            max_cached_graphs: 256,
            max_loaded_direct_graphs: 112,
            program_cache_policy: ProgramCachePolicy::PreferSmall,
            max_cached_bytes: 2 * 1024 * 1024 * 1024,
            max_cached_weight_bytes: 2 * 1024 * 1024 * 1024,
            max_cached_failures: 512,
            cache_dir: None,
            max_disk_cache_bytes: 8 * 1024 * 1024 * 1024,
            max_graph_operations: 512,
            max_constant_bytes: 256 * 1024 * 1024,
            min_compute_rows: 1,
            trace: false,
            row_bucket: 32,
            pad_single_row: false,
            kv_bucket: 128,
            separate_cached_attention: true,
            fuse_single_token_attention: false,
        }
    }
}

/// Counters distinguish Core ML's ANE/CPU plan from Burn fallback execution.
/// A plan is a scheduling decision, not a hardware performance-counter measurement.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AneStats {
    /// Float inputs packed by Metal into shared ANE IOSurfaces, without a CPU read.
    pub shared_metal_inputs: u64,
    /// Matching native layouts bound directly as the next graph's input surface.
    pub shared_native_aliases: u64,
    /// Metal completion fences before ANE. Pure native aliases need no GPU wait.
    pub shared_metal_synchronizations: u64,
    /// Time waiting for Metal writes at real Metal-to-ANE boundaries only.
    pub shared_metal_synchronization_ms: f64,
    /// Native output tensors handed to Metal without host materialization.
    pub shared_metal_outputs: u64,
    /// Output handoffs needing neither a layout kernel nor a dtype conversion.
    pub shared_zero_copy_views: u64,
    pub compiled_graphs: u64,
    pub compilation_failures: u64,
    pub cache_hits: u64,
    pub failure_cache_hits: u64,
    pub cache_evictions: u64,
    pub disk_cache_hits: u64,
    pub ane_graph_executions: u64,
    /// Successful calls to the private ANE runtime (not Core ML plan counts).
    pub direct_ane_graph_executions: u64,
    /// Compiled programs found in the system-owned aned cache.
    pub direct_system_cache_hits: u64,
    pub ane_operations_planned: u64,
    /// Matmul/Linear/Conv assigned to ANE; excludes constant and layout layers.
    pub ane_compute_operations_planned: u64,
    pub coreml_cpu_operations_planned: u64,
    pub fallback_operations: u64,
    pub last_fallback_reason: Option<String>,
    pub cached_graphs: usize,
    /// Direct graphs that currently occupy ANE program slots (not all cached graphs).
    pub loaded_direct_graphs: usize,
    /// Loads reusing an in-memory compiled graph, without rebuilding its descriptor.
    pub direct_program_reloads: u64,
    /// Explicit unloads to free a program slot while retaining the compiled graph.
    pub direct_program_unloads: u64,
    /// Subset of unloads selecting the incoming graph's exact shared weight blob.
    pub direct_same_weight_unloads: u64,
    /// Failed reloads; non-resource failures retry by rebuilding the graph once.
    pub direct_reload_failures: u64,
    pub direct_reload_ms: f64,
    pub direct_unload_ms: f64,
    /// Retained serialized model bytes: metadata per graph plus unique direct
    /// weight allocations. Excludes runtime copies, I/O buffers and alignment.
    pub cached_model_bytes: usize,
    /// Sum of each graph's standalone serialized size before weight sharing.
    pub unshared_cached_model_bytes: usize,
    /// Live immutable NSData blobs shared between direct graph descriptors.
    /// Included in cached_model_bytes. Neither counter is a process RSS limit.
    pub shared_model_weight_blobs: usize,
    pub shared_model_weight_bytes: usize,
    /// Sum of model weight sizes before sharing, for cached direct graphs only.
    pub unshared_model_weight_bytes: usize,
    pub model_weight_share_hits: u64,
    pub model_weight_share_misses: u64,
    pub cached_weight_bytes: usize,
    pub weight_cache_hits: u64,
    pub weight_cache_misses: u64,
    pub weight_cache_evictions: u64,
    /// Retry attempts after releasing a Direct program slot or cached I/O owner.
    pub resource_retries: u64,
    /// Subset of resource_retries caused by transient IOSurface allocation failure.
    pub surface_resource_retries: u64,
    /// Resource release time; overlaps compile_load_ms when compiling or reloading.
    pub resource_recovery_ms: f64,
    pub cached_failures: usize,
    pub pending_operations: usize,
    pub max_executed_graph_operations: usize,
    pub cost_policy_skips: u64,
    /// Cost-policy decisions made before constructing/hashing an ANE graph.
    pub early_cost_policy_skips: u64,
    /// Operations dispatched directly inside a thread-local fallback phase.
    pub fallback_scope_operations: u64,
    /// Custom fallback kernels reached through the outer ANE backend.
    pub custom_fallback_operations: BTreeMap<String, u64>,
    pub bucketed_graph_executions: u64,
    pub kv_bucketed_graph_executions: u64,
    /// Graph I/O bytes: FP16 for Direct, FP32 for Core ML; excludes IOSurface alignment.
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub constant_bytes_read: u64,
    pub host_tensor_hits: u64,
    pub fallback_upload_bytes: u64,
    pub resident_host_tensors: usize,
    pub resident_native_tensors: usize,
    /// Padded FP16 payload of live native tensors, excluding alignment and graph pools.
    pub resident_native_bytes: usize,
    pub native_input_tensors: u64,
    pub native_input_bytes: u64,
    pub native_output_tensors: u64,
    pub native_output_allocations: u64,
    pub native_output_reuses: u64,
    pub native_materializations: u64,
    pub native_materialization_bytes: u64,
    pub native_materialization_ms: f64,
    /// Exact Select/Cat/Slice copies within the allowed fallback path, without a B upload.
    pub host_copy_operations: u64,
    pub host_copy_output_bytes: u64,
    /// Included in fallback_dispatch_ms; may include native materialization.
    pub host_copy_ms: f64,
    pub planning_ms: f64,
    pub constant_read_ms: f64,
    pub graph_build_ms: f64,
    /// Native compilation/loading, plus Core ML plan inspection when applicable; includes failures.
    pub compile_load_ms: f64,
    /// Successful graph loads only: Core ML or direct ANE compilation.
    pub model_compile_ms: f64,
    /// Successful graph loads only: cache I/O, validation and lock wait.
    pub cache_prepare_ms: f64,
    /// Successful graph loads only: MLComputePlan inspection / specialization.
    pub compute_plan_ms: f64,
    /// Successful graph loads only: native model loading.
    pub model_load_ms: f64,
    pub input_read_ms: f64,
    /// Native call including input/output conversion and copies.
    pub prediction_ms: f64,
    /// Direct runtime: input transfer/layout conversion and request rebinding.
    pub direct_input_copy_ms: f64,
    /// Direct runtime only: synchronous evaluate call, including driver/scheduling overhead.
    pub direct_evaluate_ms: f64,
    /// Direct runtime: IOSurface output copy and layout conversion into logical FP16 buffers.
    pub direct_output_copy_ms: f64,
    pub output_write_ms: f64,
    /// Dispatch time only: GPU fallback may execute asynchronously.
    pub fallback_dispatch_ms: f64,
    /// Bounded histogram of graph boundaries / fallback causes.
    pub fallback_reasons: BTreeMap<String, u64>,
    /// Core ML plan counts, e.g. `ane:convolution`, `cpu:load_constant`.
    pub planned_operations_by_device: BTreeMap<String, u64>,
}

#[derive(Clone)]
pub struct AneDevice<B: BackendIr = NdArray<f32>> {
    pub(crate) fallback: B::Device,
    pub(crate) options: Arc<AneOptions>,
    id: DeviceId,
}
type DeviceRegistry = HashMap<(TypeId, DeviceId), Box<dyn Any + Send + Sync>>;
static DEVICES: LazyLock<Mutex<DeviceRegistry>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_DEVICE: AtomicU32 = AtomicU32::new(1);

impl<B: BackendIr> AneDevice<B> {
    /// Create an independent graph/cache session on a fallback device.
    pub fn new(fallback: B::Device, options: AneOptions) -> Self {
        assert!(options.max_cached_graphs > 0 && options.max_graph_operations > 0);
        let index = NEXT_DEVICE.fetch_add(1, Ordering::Relaxed);
        assert!(index != u32::MAX, "ANE device identifier exhausted");
        let id = DeviceId {
            type_id: (index >> 16) as u16,
            index_id: index as u16,
        };
        let device = Self {
            fallback,
            options: Arc::new(options),
            id,
        };
        DEVICES
            .lock()
            .unwrap()
            .insert((TypeId::of::<B>(), id), Box::new(device.clone()));
        device
    }
    /// Flush with `Ane::sync` before reading counters for a completed inference.
    pub fn stats(&self) -> AneStats {
        get_client::<AneChannel<B>>(self).stats()
    }
    /// Release compiled models. Tensor values remain valid; subsequent inference recompiles.
    pub fn clear_cache(&self) {
        get_client::<AneChannel<B>>(self).clear_cache();
    }
    pub fn fallback_device(&self) -> &B::Device {
        &self.fallback
    }
    pub fn options(&self) -> &AneOptions {
        &self.options
    }
    pub fn prefers_fallback_phases(&self) -> bool {
        self.options.prefer_fallback_phases
            && !self.options.ane_first
            && self.options.fallback == FallbackPolicy::Allow
    }
    /// Keep a dependent phase on B without per-graph ANE planning. The scope is
    /// local to this thread and device, nests safely, and restores on unwind.
    /// It flushes preceding ANE work, adding no GPU fence beyond that work's
    /// existing dependencies.
    pub fn fallback_scope(&self) -> Result<FallbackScope, burn_backend::ExecutionError> {
        get_client::<AneChannel<B>>(self).begin_fallback_scope()
    }
}
impl<B: BackendIr> Default for AneDevice<B> {
    fn default() -> Self {
        Self::new(B::Device::default(), AneOptions::default())
    }
}
impl<B: BackendIr> std::fmt::Debug for AneDevice<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AneDevice")
            .field("id", &self.id)
            .field("fallback", &self.fallback)
            .field("options", &self.options)
            .finish()
    }
}
impl<B: BackendIr> PartialEq for AneDevice<B> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<B: BackendIr> Device for AneDevice<B> {
    fn from_id(id: DeviceId) -> Self {
        DEVICES
            .lock()
            .unwrap()
            .get(&(TypeId::of::<B>(), id))
            .and_then(|d| d.downcast_ref::<Self>())
            .expect("unknown ANE device ID")
            .clone()
    }
    fn to_id(&self) -> DeviceId {
        self.id
    }
}
impl<B: BackendIr> DeviceOps for AneDevice<B> {}
