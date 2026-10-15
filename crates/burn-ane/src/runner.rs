use crate::{
    AneDevice, AneOptions, AneStats, FallbackPolicy, ProgramCachePolicy,
    cache::Cache,
    compiler::{self, Plan},
    runtime::Graph,
    storage::{Layout, NativeTensor, StoredTensor},
};
use burn_backend::{
    DType, DTypeUsageSet, Element, ExecutionError, Shape, TensorData, TensorMetadata,
};
use burn_ir::{BackendIr, OperationIr, TensorHandle, TensorId, TensorIr, TensorStatus};
use burn_router::{MultiBackendBridge, RouterTensor, Runner, RunnerChannel, RunnerClient};
use burn_std::future::DynFut;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::{Arc, Mutex},
    time::Instant,
};

fn shared_backend<B: BackendIr>() -> bool {
    #[cfg(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64"))]
    {
        return crate::shared::enabled::<B>();
    }
    #[cfg(not(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64")))]
    {
        false
    }
}

struct State {
    pending: Vec<OperationIr>,
    pending_count: usize,
    pending_weights: HashSet<TensorId>,
    pending_weight_bytes: usize,
    pending_convolution: bool,
    cache: Cache<Graph>,
    weights: crate::mil::WeightCache,
    model_weights: crate::runtime::WeightPool,
    failures: Cache<String>,
    /// Direct FP16 outputs lease immutable IOSurfaces across graph boundaries.
    /// Other dtypes and Core ML outputs retain their original host storage.
    tensors: HashMap<TensorId, StoredTensor>,
    stats: AneStats,
    error: Option<String>,
}

/// Burn IR channel backed by the ANE graph compiler.
#[derive(Clone)]
pub struct AneChannel<B: BackendIr>(PhantomData<B>);

/// Tensor client used internally by `Ane`. All execution is serialized per session.
#[derive(Clone)]
pub struct AneClient<B: BackendIr> {
    runner: Runner<B>,
    state: Arc<Mutex<State>>,
    device: AneDevice<B>,
}

impl State {
    /// Serialized graph metadata plus each live native NSData blob once. During
    /// compilation/prediction, the pool also sees the prospective/in-flight graph
    /// so its weights remain charged while it is temporarily outside the cache.
    fn model_cache_bytes(&self) -> usize {
        self.cache.bytes - self.cache.sum(Graph::weight_bytes) + self.model_weights.live().1
    }

    fn unload_direct(
        &mut self,
        options: &AneOptions,
        incoming: Option<&Arc<crate::runtime::Weights>>,
    ) -> Result<bool, String> {
        let same_weights = |graph: &Graph| {
            graph
                .shared_weights()
                .zip(incoming)
                .is_some_and(|(cached, incoming)| Arc::ptr_eq(cached, incoming))
        };
        // Both Arcs remain live during selection. Pointer identity therefore
        // compares exactly the same interned blob, never a recycled address.
        // No slot is unloaded speculatively when there is still room.
        let replace_shape = options.program_cache_policy == ProgramCachePolicy::ReplaceWeightShapes
            && incoming.is_some()
            && self.cache.count(|g| g.loaded_direct() && same_weights(g)) > 0;
        let prefer_below =
            if replace_shape || options.program_cache_policy == ProgramCachePolicy::Lru {
                0
            } else {
                16 * 1024 * 1024
            };
        let Some(graph) = self.cache.victim_mut(prefer_below, |g| {
            g.loaded_direct() && (!replace_shape || same_weights(g))
        }) else {
            return Ok(false);
        };
        let start = Instant::now();
        let result = graph.set_loaded(false);
        self.stats.direct_unload_ms += start.elapsed().as_secs_f64() * 1000.;
        result?;
        self.stats.direct_program_unloads += 1;
        self.stats.direct_same_weight_unloads += u64::from(replace_shape);
        Ok(true)
    }

    fn make_direct_slot(
        &mut self,
        options: &AneOptions,
        incoming: Option<&Arc<crate::runtime::Weights>>,
    ) -> Result<(), String> {
        let count = options
            .max_cached_graphs
            .min(options.max_loaded_direct_graphs)
            .max(1);
        while self.cache.count(Graph::loaded_direct) >= count {
            if !self.unload_direct(options, incoming)? {
                return Err("No ANE program slot can be released".into());
            }
        }
        Ok(())
    }

    /// Program unload keeps its I/O buffers alive. Surface pressure instead
    /// needs a whole cached graph released, preferring an already unloaded one.
    /// The graph being compiled/predicted is outside the cache; live outputs
    /// independently own their surfaces and survive removal of their graph.
    fn recover_direct_resources(
        &mut self,
        error: &str,
        options: &AneOptions,
        incoming: Option<&Arc<crate::runtime::Weights>>,
    ) -> Result<bool, String> {
        if options.runtime != crate::AneRuntime::Direct
            || !crate::runtime::resource_exhausted(error)
        {
            return Ok(false);
        }
        let start = Instant::now();
        let surface = crate::runtime::surface_exhausted(error);
        let released = if surface {
            let evicted = self
                .cache
                .evict_where(0, |g| g.direct && !g.loaded_direct())
                || self.cache.evict_where(0, |g| g.direct);
            self.stats.cache_evictions += u64::from(evicted);
            Ok(evicted)
        } else {
            self.unload_direct(options, incoming)
        };
        self.stats.resource_recovery_ms += start.elapsed().as_secs_f64() * 1000.;
        if matches!(released, Ok(true)) {
            self.stats.resource_retries += 1;
            self.stats.surface_resource_retries += u64::from(surface);
        }
        released
    }

    fn reload_direct(&mut self, graph: &mut Graph, options: &AneOptions) -> Result<(), String> {
        let start = Instant::now();
        let result = (|| {
            self.make_direct_slot(options, graph.shared_weights())?;
            for attempt in 0..=8 {
                let load_start = Instant::now();
                let result = graph.set_loaded(true);
                let ms = load_start.elapsed().as_secs_f64() * 1000.;
                self.stats.direct_reload_ms += ms;
                match result {
                    Ok(()) => {
                        self.stats.model_load_ms += ms;
                        self.stats.direct_program_reloads += 1;
                        return Ok(());
                    }
                    Err(error) => {
                        if attempt == 8
                            || !self.recover_direct_resources(
                                &error,
                                options,
                                graph.shared_weights(),
                            )?
                        {
                            return Err(error);
                        }
                    }
                }
            }
            unreachable!()
        })();
        self.stats.compile_load_ms += start.elapsed().as_secs_f64() * 1000.;
        if result.is_err() {
            self.stats.direct_reload_failures += 1;
        }
        result
    }

    /// A budget boundary may be encountered just after constructing a weight
    /// transpose. Move that view to its consumer's graph, otherwise the previous
    /// graph exports a large dynamic matrix on every decoding step.
    fn detach_weight_views(&mut self, weights: &[&TensorIr]) -> Vec<OperationIr> {
        use burn_ir::BaseOperationIr as Base;
        let mut needed: HashSet<_> = weights.iter().map(|t| t.id).collect();
        let mut detached = Vec::new();
        while let Some(op) = self.pending.last() {
            let view = match op {
                OperationIr::BaseFloat(Base::Reshape(t)) => Some((&t.input, &t.out)),
                OperationIr::BaseFloat(Base::SwapDims(t)) => Some((&t.input, &t.out)),
                OperationIr::BaseFloat(Base::Permute(t)) => Some((&t.input, &t.out)),
                OperationIr::BaseFloat(Base::Cast(t)) => Some((&t.input, &t.out)),
                OperationIr::Drop(_) => None,
                _ => break,
            };
            if let Some((input, output)) = view {
                if !needed.remove(&output.id) {
                    break;
                }
                needed.insert(input.id);
                self.pending_count -= 1;
            }
            detached.push(self.pending.pop().unwrap());
        }
        detached.reverse();
        detached
    }

    fn flush<B: BackendIr>(
        &mut self,
        runner: &Runner<B>,
        options: &AneOptions,
    ) -> Result<(), String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.pending.is_empty() {
            return Ok(());
        }
        let ops = std::mem::take(&mut self.pending);
        self.pending_count = 0;
        self.pending_weights.clear();
        self.pending_weight_bytes = 0;
        self.pending_convolution = false;
        // Metal-bound graphs do not need dead-code analysis, shape bucketing,
        // constant discovery or a serialized ANE cache key. Keep them lazy in
        // B's fusion stream and avoid any GPU completion wait here.
        if !options.ane_first && options.fallback == FallbackPolicy::Allow {
            let rows = ops.iter().map(compute_rows).max().unwrap_or(0);
            if rows == 0 || rows < options.min_compute_rows {
                self.stats.early_cost_policy_skips += 1;
                if rows > 0 {
                    self.stats.cost_policy_skips += 1;
                }
                let reason = if rows == 0 {
                    "subgraph has no matrix multiplication or convolution"
                } else {
                    "below minimum compute rows"
                };
                return self.fallback(runner, ops, options, reason.into());
            }
        }
        let start = Instant::now();
        let plan = Plan::new(
            &ops,
            options.row_bucket,
            if options.runtime == crate::AneRuntime::Direct {
                options.kv_bucket
            } else {
                0
            },
            options.runtime != crate::AneRuntime::Direct || options.pad_single_row,
        );
        self.stats.planning_ms += start.elapsed().as_secs_f64() * 1000.;
        if plan.outputs.is_empty() {
            self.release_consumed(runner, &ops);
            return Ok(());
        }
        if !plan.has_compute() && !options.ane_first {
            return self.fallback(
                runner,
                ops,
                options,
                "subgraph has no matrix multiplication or convolution".into(),
            );
        }
        if !options.ane_first && !plan.worth_offloading(options.min_compute_rows) {
            self.stats.cost_policy_skips += 1;
            return self.fallback(runner, ops, options, "below minimum compute rows".into());
        }
        if let Some((error, bytes)) = self.failures.take(&plan.key) {
            self.stats.cache_hits += 1;
            self.stats.failure_cache_hits += 1;
            self.failures.insert(
                plan.key,
                error.clone(),
                bytes,
                options.max_cached_failures,
                64 * 1024 * 1024,
            );
            return self.fallback(runner, ops, options, error);
        }
        let cached = self.cache.take(&plan.key).and_then(|(mut graph, bytes)| {
            self.stats.cache_hits += 1;
            if graph.direct && !graph.loaded_direct() {
                match self.reload_direct(&mut graph, options) {
                    Ok(()) => Some(Ok((graph, bytes))),
                    Err(error) if crate::runtime::resource_exhausted(&error) => Some(Err(error)),
                    // The OS may have purged an unloaded compiled model. Drop
                    // its cached handle and use the normal compilation path;
                    // a failed reload is not evidence of unsupported Burn IR.
                    Err(_) => None,
                }
            } else {
                Some(Ok((graph, bytes)))
            }
        });
        let was_cached = cached.is_some();
        let compiled = if let Some(cached) = cached {
            cached
        } else {
            let mut compile = || -> Result<(Graph, usize), String> {
                // In particular, don't synchronously read back WebGPU tensors on
                // unsupported platforms before discovering that ANE is unavailable.
                if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                    return Err("ANE requires Apple Silicon and macOS 15+".into());
                }
                let bytes: usize = plan
                    .constants
                    .iter()
                    .map(|t| t.shape.num_elements().saturating_mul(4))
                    .sum();
                if bytes > options.max_constant_bytes {
                    return Err("graph exceeds constant memory limit".into());
                }
                let start = Instant::now();
                let read_before = self.stats.constant_read_ms;
                let model = match options.runtime {
                    crate::AneRuntime::CoreMl => {
                        let read_start = Instant::now();
                        let constants: Result<HashMap<_, _>, String> = plan
                            .constants
                            .iter()
                            .map(|t| self.read(runner, t).map(|data| (t.id, data)))
                            .collect();
                        self.stats.constant_read_ms += read_start.elapsed().as_secs_f64() * 1000.;
                        let constants = constants?;
                        self.stats.constant_bytes_read += bytes as u64;
                        plan.compile(constants).map(crate::runtime::Source::CoreMl)
                    }
                    crate::AneRuntime::Direct => {
                        // Read/fold/transpose/pack only on a packed-weight miss.
                        // Move the cache temporarily to allow the lazy reader to
                        // access host/fallback tensors without aliasing State.
                        let mut weights = std::mem::take(&mut self.weights);
                        let result = crate::mil::compile(
                            &plan,
                            &mut weights,
                            options.max_cached_weight_bytes,
                            &mut |t| {
                                let start = Instant::now();
                                let result = self.read(runner, t);
                                self.stats.constant_read_ms +=
                                    start.elapsed().as_secs_f64() * 1000.;
                                if result.is_ok() {
                                    self.stats.constant_bytes_read +=
                                        (t.shape.num_elements() * 4) as u64;
                                }
                                result
                            },
                        )
                        .map(crate::runtime::Source::Direct);
                        self.weights = weights;
                        self.stats.weight_cache_hits = self.weights.hits;
                        self.stats.weight_cache_misses = self.weights.misses;
                        self.stats.weight_cache_evictions = self.weights.evictions;
                        result
                    }
                };
                self.stats.graph_build_ms += start.elapsed().as_secs_f64() * 1000.
                    - (self.stats.constant_read_ms - read_before);
                let model = model?;
                let start = Instant::now();
                let shared_weights = match &model {
                    crate::runtime::Source::Direct(model) => {
                        self.model_weights.intern(&model.weights)?
                    }
                    crate::runtime::Source::CoreMl(_) => None,
                };
                if options.runtime == crate::AneRuntime::Direct {
                    let metadata = model.len() - shared_weights.as_ref().map_or(0, |w| w.len);
                    while self.model_cache_bytes()
                        > options.max_cached_bytes.saturating_sub(metadata)
                    {
                        if !self.cache.evict(0) {
                            break;
                        }
                        self.stats.cache_evictions += 1;
                    }
                    self.make_direct_slot(options, shared_weights.as_ref())?;
                }
                let mut graph = Graph::compile(&model, options, shared_weights.as_ref());
                for _ in 0..8 {
                    let Err(error) = &graph else { break };
                    if !self.recover_direct_resources(error, options, shared_weights.as_ref())? {
                        break;
                    }
                    graph = Graph::compile(&model, options, shared_weights.as_ref());
                }
                self.stats.compile_load_ms += start.elapsed().as_secs_f64() * 1000.;
                let graph = graph?;
                if graph.direct {
                    self.stats.direct_system_cache_hits += u64::from(graph.disk_cache_hit);
                } else {
                    self.stats.disk_cache_hits += u64::from(graph.disk_cache_hit);
                }
                self.stats.model_compile_ms += graph.timings[0];
                self.stats.cache_prepare_ms += graph.timings[1];
                self.stats.compute_plan_ms += graph.timings[2];
                self.stats.model_load_ms += graph.timings[3];
                Ok((graph, model.len()))
            };
            let result = compile();
            if result.is_ok() {
                self.stats.compiled_graphs += 1;
            } else {
                self.stats.compilation_failures += 1;
            }
            result
        };
        let mut compiled = compiled;
        let prediction = match &mut compiled {
            Ok((graph, _)) => {
                let result = if graph.direct
                    && plan
                        .logical_inputs
                        .iter()
                        .chain(&plan.logical_outputs)
                        .all(|t| t.shape.num_dims() <= 4)
                {
                    self.predict_native(runner, &plan, graph, options)
                } else if graph.direct {
                    self.predict(runner, &plan, graph, Graph::predict_f16)
                } else {
                    self.predict(runner, &plan, graph, Graph::predict)
                };
                if result.is_ok() {
                    self.stats.ane_graph_executions += 1;
                    self.stats.direct_ane_graph_executions += u64::from(graph.direct);
                    self.stats.bucketed_graph_executions += u64::from(plan.padding.is_some());
                    self.stats.kv_bucketed_graph_executions += u64::from(plan.kv_padding.is_some());
                    self.stats.ane_operations_planned += graph.ane_operations as u64;
                    self.stats.ane_compute_operations_planned +=
                        graph.ane_compute_operations as u64;
                    self.stats.coreml_cpu_operations_planned += graph.cpu_operations as u64;
                    for (kind, count) in &graph.assignments {
                        *self
                            .stats
                            .planned_operations_by_device
                            .entry(kind.clone())
                            .or_default() += count;
                    }
                    self.stats.max_executed_graph_operations =
                        self.stats.max_executed_graph_operations.max(plan.ops.len());
                }
                result
            }
            Err(error) => Err(error.clone()),
        };
        if let Err(error) = &prediction {
            compiled = Err(error.clone());
        }
        if options.trace {
            eprintln!(
                "burn_ane_graph {}",
                serde_json::json!({
                    "operations": plan.ops.len(), "inputs": plan.inputs.iter().map(|t| t.shape.as_slice()).collect::<Vec<_>>(),
                    "constants": plan.constants.len(), "error": prediction.as_ref().err(), "stats": self.stats,
                })
            );
        }
        match compiled {
            Ok((graph, bytes)) => {
                let budget = if graph.direct && bytes <= options.max_cached_bytes {
                    // Charge metadata per graph and immutable weight storage
                    // once. A cache hit was already admitted under the same
                    // immutable options; only misses need the accounting scan.
                    if !was_cached {
                        let metadata = bytes - graph.weight_bytes();
                        while self.cache.len() >= options.max_cached_graphs
                            || self.model_cache_bytes()
                                > options.max_cached_bytes.saturating_sub(metadata)
                        {
                            if !self.cache.evict(0) {
                                break;
                            }
                            self.stats.cache_evictions += 1;
                        }
                    }
                    // Keep standalone sizes for diagnostics and program-slot
                    // priorities; the actual byte budget was checked above.
                    usize::MAX
                } else {
                    options.max_cached_bytes
                };
                self.stats.cache_evictions +=
                    self.cache
                        .insert(plan.key, graph, bytes, options.max_cached_graphs, budget);
            }
            Err(error) => {
                // Resource exhaustion is not evidence of an unsupported graph.
                // Another inference/eviction can free slots or surfaces; never memoize it.
                if !crate::runtime::resource_exhausted(&error) {
                    let bytes = plan.key.len() + error.len();
                    self.failures.insert(
                        plan.key,
                        error,
                        bytes,
                        options.max_cached_failures,
                        64 * 1024 * 1024,
                    );
                }
            }
        }
        match prediction {
            Ok(tensors) => {
                // No fallback inputs were consumed during compilation/prediction. Commit
                // the result only after success so a failed graph can be replayed safely.
                let start = Instant::now();
                self.release_consumed(runner, &ops);
                self.tensors.extend(tensors);
                self.stats.output_write_ms += start.elapsed().as_secs_f64() * 1000.;
                Ok(())
            }
            Err(error) => self.fallback(runner, ops, options, error),
        }
    }

    fn predict<B: BackendIr, E: Element>(
        &mut self,
        runner: &Runner<B>,
        plan: &Plan,
        graph: &mut Graph,
        predict: impl FnOnce(&mut Graph, &[E], &mut [E]) -> Result<(), String>,
    ) -> Result<Vec<(TensorId, StoredTensor)>, String> {
        let start = Instant::now();
        let count = plan
            .inputs
            .iter()
            .map(|t| t.shape.num_elements())
            .sum::<usize>()
            + plan
                .auxiliary
                .iter()
                .map(|(_, _, v)| v.len())
                .sum::<usize>();
        let mut inputs = Vec::with_capacity(count);
        for t in &plan.logical_inputs {
            let values = self.read_view::<B, E>(runner, t)?;
            if let Some(padding) = &plan.padding {
                padding.pad_into(t, &values, &mut inputs);
            } else if let Some(padding) = &plan.kv_padding {
                padding.pad_into(t, &values, &mut inputs);
            } else {
                inputs.extend_from_slice(&values);
            }
        }
        for (_, _, values) in &plan.auxiliary {
            inputs.extend(values.iter().map(|v| E::from_elem(*v)));
        }
        self.stats.input_read_ms += start.elapsed().as_secs_f64() * 1000.;
        self.stats.input_bytes += (inputs.len() * size_of::<E>()) as u64;
        let mut outputs =
            vec![E::default(); plan.outputs.iter().map(|t| t.shape.num_elements()).sum()];
        let start = Instant::now();
        let result = predict(graph, &inputs, &mut outputs);
        self.stats.prediction_ms += start.elapsed().as_secs_f64() * 1000.;
        self.stats.direct_input_copy_ms += graph.prediction_timings[0];
        self.stats.direct_evaluate_ms += graph.prediction_timings[1];
        self.stats.direct_output_copy_ms += graph.prediction_timings[2];
        result?;
        self.stats.output_bytes += (outputs.len() * size_of::<E>()) as u64;

        let start = Instant::now();
        let mut tensors = Vec::with_capacity(plan.outputs.len());
        let mut offset = 0;
        for (tensor, logical) in plan.outputs.iter().zip(&plan.logical_outputs) {
            let len = tensor.shape.num_elements();
            let values = &outputs[offset..offset + len];
            let values = if let Some(padding) = &plan.padding {
                padding.trim(logical, values)
            } else if let Some(padding) = &plan.kv_padding {
                padding.trim(logical, values)
            } else {
                values.to_vec()
            };
            // FP16 model tensors remain FP16 across graph boundaries. Convert
            // only when the Burn tensor's public dtype actually differs.
            let data = TensorData::new(values, logical.shape.clone()).convert_dtype(tensor.dtype);
            tensors.push((tensor.id, StoredTensor::Host(data)));
            offset += len;
        }
        self.stats.output_write_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(tensors)
    }

    fn predict_native<B: BackendIr>(
        &mut self,
        runner: &Runner<B>,
        plan: &Plan,
        graph: &mut Graph,
        options: &AneOptions,
    ) -> Result<Vec<(TensorId, StoredTensor)>, String> {
        use crate::runtime::{DirectInput, InputData};
        let start = Instant::now();
        let mut inputs = Vec::with_capacity(plan.inputs.len() + plan.auxiliary.len());
        #[cfg(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64"))]
        let mut metal_writes = false;
        let mut offset = 0;
        for (_index, (logical, padded)) in plan.logical_inputs.iter().zip(&plan.inputs).enumerate()
        {
            let layout = Layout::tensor(plan, logical);
            #[cfg(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64"))]
            if shared_backend::<B>()
                && logical.dtype.is_float()
                && !matches!(self.tensors.get(&logical.id), Some(StoredTensor::Host(_)))
            {
                let native = match self.tensors.get(&logical.id) {
                    Some(StoredTensor::Native(native)) => Some(native),
                    _ => None,
                };
                let (input, packed) = crate::shared::prepare_input(
                    runner,
                    logical,
                    native,
                    layout.clone(),
                    graph,
                    _index,
                )?;
                self.stats.shared_metal_inputs += u64::from(packed);
                self.stats.shared_native_aliases += u64::from(!packed);
                metal_writes |= packed;
                inputs.push(DirectInput {
                    offset,
                    layout,
                    data: InputData::Native(input),
                });
                offset += padded.shape.num_elements();
                continue;
            }
            let data = if let Some(StoredTensor::Native(tensor)) = self.tensors.get(&logical.id) {
                self.stats.native_input_tensors += 1;
                self.stats.native_input_bytes += (padded.shape.num_elements() * 2) as u64;
                InputData::Native(tensor.clone())
            } else {
                let values = self.read_view::<B, half::f16>(runner, logical)?;
                let mut buffer = Vec::with_capacity(padded.shape.num_elements());
                if let Some(padding) = &plan.padding {
                    padding.pad_into(logical, &values, &mut buffer);
                } else if let Some(padding) = &plan.kv_padding {
                    padding.pad_into(logical, &values, &mut buffer);
                } else {
                    buffer.extend_from_slice(&values);
                }
                InputData::Host(buffer)
            };
            inputs.push(DirectInput {
                offset,
                layout,
                data,
            });
            offset += padded.shape.num_elements();
        }
        for (_, shape, values) in &plan.auxiliary {
            inputs.push(DirectInput {
                offset,
                layout: Layout::plain(shape),
                data: InputData::Host(values.iter().map(|v| half::f16::from_f32(*v)).collect()),
            });
            offset += values.len();
        }
        self.stats.input_read_ms += start.elapsed().as_secs_f64() * 1000.;
        self.stats.input_bytes += (offset * 2) as u64;
        #[cfg(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64"))]
        if metal_writes {
            // All Metal layout writes must complete before ANE consumes their
            // shared allocations. No activation readback is needed to wait.
            let start = Instant::now();
            crate::shared::synchronize(runner)?;
            self.stats.shared_metal_synchronization_ms += start.elapsed().as_secs_f64() * 1000.;
            self.stats.shared_metal_synchronizations += 1;
        }
        let mut retries = 0;
        let surfaces = loop {
            let start = Instant::now();
            let result = graph.predict_surfaces(&inputs);
            self.stats.prediction_ms += start.elapsed().as_secs_f64() * 1000.;
            self.stats.direct_input_copy_ms += graph.prediction_timings[0];
            self.stats.direct_evaluate_ms += graph.prediction_timings[1];
            self.stats.direct_output_copy_ms += graph.prediction_timings[2];
            self.stats.native_output_allocations += graph.output_allocations;
            self.stats.native_output_reuses += graph.output_reuses;
            match result {
                Ok(surfaces) => break surfaces,
                Err(error) => {
                    // Only allocation failures are replayed here. Inputs and
                    // existing output leases have not been consumed or committed.
                    if retries == 8
                        || !crate::runtime::surface_exhausted(&error)
                        || !self.recover_direct_resources(
                            &error,
                            options,
                            graph.shared_weights(),
                        )?
                    {
                        return Err(error);
                    }
                    retries += 1;
                }
            }
        };
        if surfaces.len() != plan.outputs.len() {
            return Err("Direct ANE native output count mismatch".into());
        }
        self.stats.output_bytes += (plan
            .outputs
            .iter()
            .map(|t| t.shape.num_elements())
            .sum::<usize>()
            * 2) as u64;
        let start = Instant::now();
        let mut tensors = Vec::with_capacity(surfaces.len());
        for ((surface, padded), logical) in surfaces
            .into_iter()
            .zip(&plan.outputs)
            .zip(&plan.logical_outputs)
        {
            let native = NativeTensor {
                surface,
                layout: Layout::tensor(plan, logical),
                dtype: logical.dtype,
            };
            let storage = if logical.dtype == DType::F16
                || (shared_backend::<B>() && logical.dtype.is_float())
            {
                self.stats.native_output_tensors += 1;
                StoredTensor::Native(native)
            } else {
                // Preserve the old FP16 -> public dtype conversion at each
                // boundary, including integer, bool and BF16 rounding behavior.
                StoredTensor::Host(self.read_native(&native)?.convert_dtype(padded.dtype))
            };
            tensors.push((logical.id, storage));
        }
        self.stats.output_write_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(tensors)
    }

    fn read_native(&mut self, tensor: &NativeTensor) -> Result<TensorData, String> {
        let start = Instant::now();
        let result = tensor.read();
        self.stats.native_materialization_ms += start.elapsed().as_secs_f64() * 1000.;
        self.stats.native_materializations += 1;
        self.stats.native_materialization_bytes += (tensor.layout.padded_len() * 2) as u64;
        result
    }

    fn ensure_host(&mut self, id: TensorId) -> Result<(), String> {
        if let Some(StoredTensor::Native(tensor)) = self.tensors.get(&id) {
            let tensor = tensor.clone();
            let data = self.read_native(&tensor)?;
            self.tensors.insert(id, StoredTensor::Host(data));
        }
        Ok(())
    }

    fn read<B: BackendIr, E: Element>(
        &mut self,
        runner: &Runner<B>,
        tensor: &TensorIr,
    ) -> Result<Vec<E>, String> {
        self.read_view::<B, E>(runner, tensor).map(Cow::into_owned)
    }

    fn read_view<'a, B: BackendIr, E: Element>(
        &'a mut self,
        runner: &Runner<B>,
        tensor: &TensorIr,
    ) -> Result<Cow<'a, [E]>, String> {
        self.ensure_host(tensor.id)?;
        if let Some(StoredTensor::Host(data)) = self.tensors.get(&tensor.id) {
            self.stats.host_tensor_hits += 1;
            if data.dtype == E::dtype() {
                return data
                    .as_slice::<E>()
                    .map(Cow::Borrowed)
                    .map_err(|e| format!("{e:?}"));
            }
            return data
                .clone()
                .convert::<E>()
                .into_vec()
                .map(Cow::Owned)
                .map_err(|e| format!("{e:?}"));
        }
        let mut tensor = tensor.clone();
        tensor.status = TensorStatus::ReadOnly;
        let data =
            pollster::block_on(runner.read_tensor_async(tensor)).map_err(|e| e.to_string())?;
        data.convert::<E>()
            .into_vec::<E>()
            .map(Cow::Owned)
            .map_err(|e| format!("{e:?}"))
    }

    fn materialize<B: BackendIr>(
        &mut self,
        runner: &Runner<B>,
        tensor: &TensorIr,
    ) -> Result<(), String> {
        #[cfg(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64"))]
        if shared_backend::<B>()
            && let Some(StoredTensor::Native(native)) = self.tensors.get(&tensor.id)
        {
            let zero_copy = crate::shared::materialize(runner, tensor.id, native)?;
            self.stats.shared_metal_outputs += 1;
            self.stats.shared_zero_copy_views += u64::from(zero_copy);
            self.tensors.remove(&tensor.id);
            return Ok(());
        }
        self.ensure_host(tensor.id)?;
        if let Some(StoredTensor::Host(data)) = self.tensors.remove(&tensor.id) {
            self.stats.fallback_upload_bytes += data.bytes.len() as u64;
            runner.register_tensor_data_id(tensor.id, data);
        }
        Ok(())
    }

    fn release_consumed<B: BackendIr>(&mut self, runner: &Runner<B>, ops: &[OperationIr]) {
        let mut removed = HashSet::new();
        for op in ops {
            for t in crate::ir::inputs(op) {
                if t.status == TensorStatus::ReadWrite && removed.insert(t.id) {
                    self.tensors.remove(&t.id);
                    runner.register_op(OperationIr::Drop(t.clone()));
                }
            }
            if let OperationIr::Drop(t) = op
                && removed.insert(t.id)
            {
                self.tensors.remove(&t.id);
                runner.register_op(OperationIr::Drop(t.clone()));
            }
        }
    }

    fn host_copy(&mut self, op: &OperationIr) -> Result<Option<(TensorId, TensorData)>, String> {
        use burn_ir::BaseOperationIr as Base;
        if !matches!(
            op,
            OperationIr::BaseFloat(Base::Select(_) | Base::Cat(_) | Base::Slice(_))
        ) || crate::ir::inputs(op).any(|t| !self.tensors.contains_key(&t.id))
        {
            return Ok(None);
        }
        // Do not read a fallback-device tensor just to speculate about a host
        // copy. Native tensors are already owned here and would need the same
        // readback before uploading to B on the normal fallback path.
        for input in crate::ir::inputs(op) {
            self.ensure_host(input.id)?;
        }
        let data = |id| match self.tensors.get(&id) {
            Some(StoredTensor::Host(data)) => Some(data),
            _ => None,
        };
        let result = match op {
            OperationIr::BaseFloat(Base::Select(t)) => data(t.tensor.id)
                .zip(data(t.indices.id))
                .and_then(|(x, indices)| crate::host::select(x, indices, t.dim))
                .map(|value| (&t.out, value)),
            OperationIr::BaseFloat(Base::Cat(t)) => t
                .tensors
                .iter()
                .map(|x| data(x.id))
                .collect::<Option<Vec<_>>>()
                .and_then(|values| crate::host::cat(&values, t.dim))
                .map(|value| (&t.out, value)),
            OperationIr::BaseFloat(Base::Slice(t)) => data(t.tensor.id)
                .and_then(|value| crate::host::slice(value, &t.ranges))
                .map(|value| (&t.out, value)),
            _ => None,
        };
        Ok(result
            .filter(|(out, value)| value.dtype == out.dtype && value.shape == out.shape)
            .map(|(out, value)| (out.id, value)))
    }

    fn fallback<B: BackendIr>(
        &mut self,
        runner: &Runner<B>,
        ops: Vec<OperationIr>,
        options: &AneOptions,
        reason: String,
    ) -> Result<(), String> {
        self.stats.last_fallback_reason = Some(reason.clone());
        if self.stats.fallback_reasons.len() < 64
            || self.stats.fallback_reasons.contains_key(&reason)
        {
            *self
                .stats
                .fallback_reasons
                .entry(reason.clone())
                .or_default() += 1;
        }
        if options.fallback == FallbackPolicy::Error {
            self.error = Some(reason.clone());
            self.release_consumed(runner, &ops);
            return Err(reason);
        }
        let start = Instant::now();
        for op in ops {
            if !matches!(op, OperationIr::Drop(_) | OperationIr::Init(_)) {
                self.stats.fallback_operations += 1;
            }
            if options.runtime == crate::AneRuntime::Direct && !shared_backend::<B>() {
                let copy_start = Instant::now();
                match self.host_copy(&op) {
                    Ok(Some((id, data))) => {
                        self.stats.host_copy_operations += 1;
                        self.stats.host_copy_output_bytes += data.bytes.len() as u64;
                        self.stats.host_copy_ms += copy_start.elapsed().as_secs_f64() * 1000.;
                        self.release_consumed(runner, std::slice::from_ref(&op));
                        self.tensors.insert(id, StoredTensor::Host(data));
                        continue;
                    }
                    Err(error) => {
                        self.error = Some(error.clone());
                        return Err(error);
                    }
                    Ok(None) => {}
                }
            }
            if let OperationIr::Drop(t) = &op {
                self.tensors.remove(&t.id);
            } else {
                for input in crate::ir::inputs(&op) {
                    if let Err(error) = self.materialize(runner, input) {
                        self.error = Some(error.clone());
                        return Err(error);
                    }
                }
            }
            runner.register_op(op);
        }
        self.stats.fallback_dispatch_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
}

impl<B: BackendIr> AneClient<B> {
    pub fn allows_fallback(&self) -> bool {
        self.device.options.fallback == FallbackPolicy::Allow
    }

    /// Whether the latency policy may select a custom kernel instead of ANE.
    pub fn prefers_fallback_kernels(&self) -> bool {
        self.allows_fallback() && !self.device.options.ane_first
    }

    pub(crate) fn begin_fallback_scope(&self) -> Result<crate::FallbackScope, ExecutionError> {
        if !self.allows_fallback() {
            return Err(ExecutionError::WithContext {
                reason: "fallback scopes require FallbackPolicy::Allow".into(),
            });
        }
        self.flush()?;
        Ok(crate::scheduling::enter(&self.device))
    }

    /// Run a custom float operation on the fallback backend, keeping its lazy
    /// tensor handles. Native inputs use the same shared-surface handoff as
    /// ordinary fallback operations. After flushing pending ANE dependencies,
    /// the handoff adds no CPU readback or GPU completion fence.
    /// Inputs must belong to this session. Strict ANE devices reject the call.
    pub fn fallback_float<const N: usize>(
        &self,
        name: &'static str,
        inputs: [RouterTensor<Self>; N],
        operation: impl FnOnce([B::FloatTensorPrimitive; N]) -> B::FloatTensorPrimitive,
    ) -> Result<RouterTensor<Self>, ExecutionError> {
        if self.device.options.fallback == FallbackPolicy::Error {
            return Err(ExecutionError::WithContext {
                reason: "custom fallback kernels require FallbackPolicy::Allow".into(),
            });
        }
        if inputs
            .iter()
            .any(|t| !Arc::ptr_eq(&self.state, &t.client.state))
        {
            return Err(ExecutionError::WithContext {
                reason: "custom fallback inputs belong to different ANE sessions".into(),
            });
        }
        if inputs.iter().any(|t| !t.dtype().is_float()) {
            return Err(ExecutionError::WithContext {
                reason: "custom float kernels require floating-point inputs".into(),
            });
        }
        self.flush()?;
        let inputs = inputs.map(RouterTensor::into_ir);
        let mut state = self.state.lock().unwrap();
        for input in &inputs {
            if let Err(reason) = state.materialize(&self.runner, input) {
                state.error = Some(reason.clone());
                for input in &inputs {
                    if input.status == TensorStatus::ReadWrite {
                        state.tensors.remove(&input.id);
                        self.runner.register_op(OperationIr::Drop(input.clone()));
                    }
                }
                return Err(ExecutionError::WithContext { reason });
            }
        }
        let start = Instant::now();
        let inputs = inputs.map(|t| {
            B::float_tensor(TensorHandle {
                handle: self.runner.get_tensor_handle(&t),
                shape: t.shape,
            })
        });
        let output = operation(inputs);
        let shape = output.shape();
        let dtype = output.dtype();
        let output =
            self.runner
                .register_tensor(B::float_tensor_handle(output), shape, dtype, self.clone());
        state.stats.fallback_operations += 1;
        if state.stats.custom_fallback_operations.len() < 64
            || state.stats.custom_fallback_operations.contains_key(name)
        {
            *state
                .stats
                .custom_fallback_operations
                .entry(name.into())
                .or_default() += 1;
        }
        state.stats.fallback_dispatch_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(output)
    }

    pub(crate) fn stats(&self) -> AneStats {
        let state = self.state.lock().unwrap();
        let mut stats = state.stats.clone();
        stats.cached_graphs = state.cache.len();
        stats.loaded_direct_graphs = state.cache.count(Graph::loaded_direct);
        stats.cached_model_bytes = state.model_cache_bytes();
        stats.unshared_cached_model_bytes = state.cache.bytes;
        (
            stats.shared_model_weight_blobs,
            stats.shared_model_weight_bytes,
        ) = state.model_weights.live();
        stats.unshared_model_weight_bytes = state.cache.sum(Graph::weight_bytes);
        stats.model_weight_share_hits = state.model_weights.hits;
        stats.model_weight_share_misses = state.model_weights.misses;
        stats.cached_weight_bytes = state.weights.cache.bytes;
        stats.cached_failures = state.failures.len();
        for tensor in state.tensors.values() {
            match tensor {
                StoredTensor::Host(_) => stats.resident_host_tensors += 1,
                StoredTensor::Native(t) => {
                    stats.resident_native_tensors += 1;
                    stats.resident_native_bytes += t.layout.padded_len() * 2;
                }
            }
        }
        stats.pending_operations = state.pending.len();
        stats
    }
    pub(crate) fn clear_cache(&self) {
        let mut state = self.state.lock().unwrap();
        state.cache.clear();
        state.model_weights.clear();
        state.weights.cache.clear();
        state.failures.clear();
    }
    fn flush(&self) -> Result<(), ExecutionError> {
        self.state
            .lock()
            .unwrap()
            .flush(&self.runner, &self.device.options)
            .map_err(|reason| ExecutionError::WithContext { reason })
    }
}

impl<B: BackendIr> RunnerClient for AneClient<B> {
    type Device = AneDevice<B>;
    fn register_op(&self, op: OperationIr) {
        let mut state = self.state.lock().unwrap();
        if matches!(op, OperationIr::Init(_)) {
            return;
        }
        if matches!(op, OperationIr::Drop(_)) && (state.pending.is_empty() || state.error.is_some())
        {
            if let OperationIr::Drop(t) = &op {
                state.tensors.remove(&t.id);
            }
            self.runner.register_op(op);
            return;
        }
        if state.error.is_some() {
            state.release_consumed(&self.runner, &[op]);
            return;
        }
        if crate::scheduling::active(&self.device) {
            // Another submitting thread may have left ANE work pending. Keep
            // its policy and dependency order before dispatching this phase.
            if let Err(error) = state.flush(&self.runner, &self.device.options) {
                state.error = Some(error);
                state.release_consumed(&self.runner, &[op]);
                return;
            }
            let start = Instant::now();
            // Drop is handled above when pending is empty. A concurrent
            // submitter can leave pending work, so handle that case too.
            if let OperationIr::Drop(t) = &op {
                state.tensors.remove(&t.id);
            } else {
                for input in crate::ir::inputs(&op) {
                    if let Err(error) = state.materialize(&self.runner, input) {
                        state.error = Some(error);
                        state.release_consumed(&self.runner, std::slice::from_ref(&op));
                        return;
                    }
                }
                state.stats.fallback_operations += 1;
                state.stats.fallback_scope_operations += 1;
            }
            self.runner.register_op(op);
            state.stats.fallback_dispatch_ms += start.elapsed().as_secs_f64() * 1000.;
            return;
        }
        let direct = self.device.options.runtime == crate::AneRuntime::Direct;
        let separate_attention = direct
            && self.device.options.separate_cached_attention
            && self.device.options.row_bucket > 0;
        if compiler::supported(&op, direct) || matches!(op, OperationIr::Drop(_)) {
            // Token concatenation changes the row axis, so padding cannot cross
            // it. End any preceding projection before preparing the new rows:
            // otherwise audio-block assembly also prevents the following large
            // transformer graph from sharing a row bucket. Only this layout
            // copy uses the fallback; strict mode keeps the original fusion.
            if separate_attention
                && self.device.options.fallback == FallbackPolicy::Allow
                && matches!(&op, OperationIr::BaseFloat(burn_ir::BaseOperationIr::Cat(t))
                    if t.dim == 0 && t.out.shape.num_dims() == 2 && t.tensors.len() > 1)
            {
                if state.flush(&self.runner, &self.device.options).is_ok() {
                    let _ = state.fallback(
                        &self.runner,
                        vec![op],
                        &self.device.options,
                        "input token concatenation".into(),
                    );
                } else {
                    state.release_consumed(&self.runner, &[op]);
                }
                return;
            }
            let masked_attention = matches!(&op, OperationIr::Module(burn_ir::ModuleOperationIr::Attention(t))
                if t.mask.is_some() || t.query.shape[2] != t.key.shape[2]);
            if separate_attention
                && masked_attention
                && state.pending.iter().any(has_matrix_weights)
                && !(self.device.options.fuse_single_token_attention
                    && single_row_projections(&state.pending))
                && state.flush(&self.runner, &self.device.options).is_err()
            {
                state.release_consumed(&self.runner, &[op]);
                return;
            }
            if separate_attention
                && cached_append_boundary(&op, &state.pending)
                && !(self.device.options.fuse_single_token_attention
                    && single_row_projections(&state.pending))
                && state.flush(&self.runner, &self.device.options).is_err()
            {
                state.release_consumed(&self.runner, &[op]);
                return;
            }
            // Convolution spatial dimensions cannot use token padding. Keep them
            // out of the subsequent linear/transformer graph so it can specialize.
            if state.pending_convolution
                && matches!(
                    &op,
                    OperationIr::Module(burn_ir::ModuleOperationIr::Linear(_))
                        | OperationIr::Float(_, burn_ir::FloatOperationIr::Matmul(_))
                )
                && state.flush(&self.runner, &self.device.options).is_err()
            {
                state.release_consumed(&self.runner, &[op]);
                return;
            }
            let weights = compiler::weight_candidates(&op);
            let extra: usize = weights
                .iter()
                .filter(|t| !state.pending_weights.contains(&t.id))
                .map(|t| t.shape.num_elements().saturating_mul(4))
                .sum();
            if state.pending_count > 0
                && state.pending_weight_bytes.saturating_add(extra)
                    > self.device.options.max_constant_bytes
            {
                let views = state.detach_weight_views(&weights);
                if state.flush(&self.runner, &self.device.options).is_err() {
                    state.release_consumed(&self.runner, &views);
                    state.release_consumed(&self.runner, &[op]);
                    return;
                }
                state.pending_count += views
                    .iter()
                    .filter(|op| !matches!(op, OperationIr::Drop(_)))
                    .count();
                state.pending.extend(views);
            }
            for t in weights {
                if state.pending_weights.insert(t.id) {
                    state.pending_weight_bytes = state
                        .pending_weight_bytes
                        .saturating_add(t.shape.num_elements().saturating_mul(4));
                }
            }
            state.pending_convolution |= matches!(
                &op,
                OperationIr::Module(burn_ir::ModuleOperationIr::Conv2d(_))
            );
            // A changing KV length or externally supplied mask should not make
            // the following FFN and next QKV projections shape-specific. End this
            // graph at attention; unmasked self-attention blocks stay fused.
            let attention_boundary = (!direct || separate_attention)
                && matches!(
                    &op,
                    OperationIr::Module(burn_ir::ModuleOperationIr::Attention(t))
                        if (t.mask.is_some() || t.query.shape[2] != t.key.shape[2])
                            && !(direct && self.device.options.fuse_single_token_attention && t.query.shape[2] == 1)
                );
            if !matches!(op, OperationIr::Drop(_)) {
                state.pending_count += 1;
            }
            state.pending.push(op);
            if attention_boundary || state.pending_count >= self.device.options.max_graph_operations
            {
                let _ = state.flush(&self.runner, &self.device.options);
            }
        } else {
            if state.flush(&self.runner, &self.device.options).is_ok() {
                let reason = format!("unsupported operation: {}", op_name(&op));
                let _ = state.fallback(&self.runner, vec![op], &self.device.options, reason);
            } else {
                state.release_consumed(&self.runner, &[op]);
            }
        }
    }
    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        match self.flush() {
            Ok(()) => {
                let mut state = self.state.lock().unwrap();
                if let Err(reason) = state.ensure_host(tensor.id) {
                    state.error = Some(reason.clone());
                    return Box::pin(async { Err(ExecutionError::WithContext { reason }) });
                }
                let data = if tensor.status == TensorStatus::ReadWrite {
                    state.tensors.remove(&tensor.id)
                } else {
                    state.tensors.get(&tensor.id).cloned()
                };
                match data {
                    Some(StoredTensor::Host(data)) => Box::pin(async { Ok(data) }),
                    Some(StoredTensor::Native(_)) => {
                        unreachable!("ensure_host materializes native tensors")
                    }
                    None => self.runner.read_tensor_async(tensor),
                }
            }
            Err(e) => {
                if tensor.status == TensorStatus::ReadWrite {
                    self.runner.register_op(OperationIr::Drop(tensor));
                }
                Box::pin(async { Err(e) })
            }
        }
    }
    fn sync(&self) -> Result<(), ExecutionError> {
        self.flush()?;
        self.runner.sync()
    }
    fn create_empty_handle(&self) -> TensorId {
        self.runner.create_empty_handle()
    }
    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        // Host-origin inputs/weights need no GPU upload if their next consumer is ANE.
        // Materialize a fallback handle only when an operation actually requires it.
        let id = self.runner.create_empty_handle();
        let shape = data.shape.clone();
        let dtype = data.dtype;
        self.state
            .lock()
            .unwrap()
            .tensors
            .insert(id, StoredTensor::Host(data));
        RouterTensor::new(id, shape, dtype, self.clone())
    }
    fn device(&self) -> Self::Device {
        self.device.clone()
    }
    fn seed(&self, seed: u64) {
        if self.flush().is_ok() {
            self.runner.seed(seed);
        }
    }
    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        self.runner.dtype_usage(dtype)
    }
}

/// Stop the projection/FFN graph before attaching an existing sequence prefix.
/// The following slice/cat/GQA/attention graph has no immutable model weights;
/// equivalent shapes can consequently share one program across decoder layers.
fn cached_append_boundary(op: &OperationIr, pending: &[OperationIr]) -> bool {
    use burn_ir::{
        BaseOperationIr as Base, FloatOperationIr as Float, ModuleOperationIr as Module,
    };
    // Prefill has no old-prefix slice. Split before grouped-head expansion so
    // its projection program has the same outputs as a cached decoder call.
    if let OperationIr::BaseFloat(Base::Reshape(t)) = op
        && t.input.shape.num_dims() == 4
        && t.out.shape.num_dims() == 5
        && t.out.shape.as_slice()
            == [
                t.input.shape[0],
                t.input.shape[1],
                1,
                t.input.shape[2],
                t.input.shape[3],
            ]
        && pending.iter().any(has_matrix_weights)
    {
        return true;
    }
    let prefix = match op {
        OperationIr::BaseFloat(Base::Slice(t))
            if t.tensor.shape.num_dims() == 4
                && t.ranges.iter().enumerate().all(|(d, r)| {
                    let range = r.to_range(t.tensor.shape[d]);
                    r.step == 1 && range.start == 0 && (d == 2 || range.end == t.tensor.shape[d])
                }) =>
        {
            &t.tensor
        }
        OperationIr::BaseFloat(Base::Cat(t))
            if t.tensors.len() == 2 && t.out.shape.num_dims() == 4 && t.dim == 2 =>
        {
            &t.tensors[0]
        }
        _ => return false,
    };
    !pending
        .iter()
        .any(|op| crate::ir::outputs(op).any(|t| t.id == prefix.id))
        && pending.iter().any(|op| {
            matches!(
                op,
                OperationIr::Float(_, Float::Matmul(_))
                    | OperationIr::Module(
                        Module::Linear(_) | Module::Conv2d(_) | Module::Attention(_)
                    )
            )
        })
}

fn single_row_projections(ops: &[OperationIr]) -> bool {
    let mut found = false;
    for op in ops {
        let rows = match op {
            OperationIr::Module(burn_ir::ModuleOperationIr::Linear(t)) => {
                t.x.shape.num_elements() / t.x.shape.as_slice().last().unwrap()
            }
            OperationIr::Float(_, burn_ir::FloatOperationIr::Matmul(t)) => {
                t.lhs.shape.num_elements() / t.lhs.shape.as_slice().last().unwrap()
            }
            OperationIr::Module(burn_ir::ModuleOperationIr::Conv2d(_)) => return false,
            OperationIr::Module(burn_ir::ModuleOperationIr::Attention(t))
                if t.query.shape[2] != 1 =>
            {
                return false;
            }
            _ => continue,
        };
        if rows != 1 {
            return false;
        }
        found = true;
    }
    found
}

fn compute_rows(op: &OperationIr) -> usize {
    match op {
        OperationIr::Module(burn_ir::ModuleOperationIr::Conv2d(_)) => usize::MAX,
        OperationIr::Module(burn_ir::ModuleOperationIr::Linear(t)) => {
            t.x.shape.num_elements() / t.x.shape.as_slice().last().unwrap()
        }
        OperationIr::Float(_, burn_ir::FloatOperationIr::Matmul(t)) => {
            t.lhs.shape[t.lhs.shape.num_dims() - 2]
        }
        OperationIr::Module(burn_ir::ModuleOperationIr::Attention(t)) => t.query.shape[2],
        _ => 0,
    }
}

fn has_matrix_weights(op: &OperationIr) -> bool {
    matches!(
        op,
        OperationIr::Float(_, burn_ir::FloatOperationIr::Matmul(_))
            | OperationIr::Module(
                burn_ir::ModuleOperationIr::Linear(_) | burn_ir::ModuleOperationIr::Conv2d(_)
            )
    )
}

fn op_name(op: &OperationIr) -> String {
    let value = serde_json::to_value(op).unwrap();
    let Some((kind, inner)) = value.as_object().and_then(|o| o.iter().next()) else {
        return "unknown".into();
    };
    let inner = inner.as_array().and_then(|v| v.last()).unwrap_or(inner);
    match inner.as_object().and_then(|o| o.keys().next()) {
        Some(name) => format!("{kind}::{name}"),
        None => kind.clone(),
    }
}

impl<B: BackendIr> RunnerChannel for AneChannel<B> {
    type Device = AneDevice<B>;
    type Bridge = Self;
    type Client = AneClient<B>;
    type FloatElem = B::FloatElem;
    type IntElem = B::IntElem;
    type BoolElem = B::BoolElem;
    fn name(device: &Self::Device) -> String {
        format!("ane<fallback={}>", B::name(&device.fallback))
    }
    fn init_client(device: &Self::Device) -> Self::Client {
        AneClient {
            runner: Runner::new(device.fallback.clone()),
            device: device.clone(),
            state: Arc::new(Mutex::new(State {
                pending: Vec::new(),
                pending_count: 0,
                pending_weights: HashSet::new(),
                pending_weight_bytes: 0,
                pending_convolution: false,
                cache: Cache::default(),
                weights: crate::mil::WeightCache::default(),
                model_weights: crate::runtime::WeightPool::default(),
                failures: Cache::default(),
                tensors: HashMap::new(),
                stats: AneStats::default(),
                error: None,
            })),
        }
    }
    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> B::Handle {
        client
            .flush()
            .expect("ANE graph failed before device transfer");
        client
            .state
            .lock()
            .unwrap()
            .materialize(&client.runner, tensor)
            .expect("ANE tensor read failed before device transfer");
        client.runner.get_tensor_handle(tensor)
    }
    fn register_tensor(
        client: &Self::Client,
        handle: B::Handle,
        shape: Shape,
        dtype: DType,
    ) -> RouterTensor<Self::Client> {
        client
            .runner
            .register_tensor(handle, shape, dtype, client.clone())
    }
}

impl<B: BackendIr> MultiBackendBridge for AneChannel<B> {
    type Device = AneDevice<B>;
    type TensorHandle = B::Handle;
    fn change_backend_float(handle: B::Handle, shape: Shape, target: &Self::Device) -> B::Handle {
        B::float_tensor_handle(B::float_to_device(
            B::float_tensor(TensorHandle { handle, shape }),
            &target.fallback,
        ))
    }
    fn change_backend_int(handle: B::Handle, shape: Shape, target: &Self::Device) -> B::Handle {
        B::int_tensor_handle(B::int_to_device(
            B::int_tensor(TensorHandle { handle, shape }),
            &target.fallback,
        ))
    }
    fn change_backend_bool(handle: B::Handle, shape: Shape, target: &Self::Device) -> B::Handle {
        B::bool_tensor_handle(B::bool_to_device(
            B::bool_tensor(TensorHandle { handle, shape }),
            &target.fallback,
        ))
    }
}
