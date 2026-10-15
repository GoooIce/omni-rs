use crate::proto::{self, Message};
use burn_backend::DType;
use burn_ir::{
    BaseOperationIr as Base, FloatOperationIr as Float, ModuleOperationIr as Module,
    NumericOperationIr as Numeric, OperationIr as Op, TensorId, TensorIr, TensorStatus,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

pub(crate) fn transpose_matrix(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.; values.len()];
    // Tile to avoid reading large matrices with a page-sized stride.
    for r0 in (0..rows).step_by(32) {
        for c0 in (0..cols).step_by(32) {
            for r in r0..(r0 + 32).min(rows) {
                for c in c0..(c0 + 32).min(cols) {
                    output[c * rows + r] = values[r * cols + c];
                }
            }
        }
    }
    output
}

pub(crate) fn permute_constant(values: &[f32], shape: &[usize], axes: &[usize]) -> Vec<f32> {
    if shape.len() == 2 && axes == [1, 0] {
        return transpose_matrix(values, shape[0], shape[1]);
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let output_shape: Vec<_> = axes.iter().map(|i| shape[*i]).collect();
    (0..values.len())
        .map(|mut flat| {
            let mut index = 0;
            for dim in (0..shape.len()).rev() {
                index += (flat % output_shape[dim]) * strides[axes[dim]];
                flat /= output_shape[dim];
            }
            values[index]
        })
        .collect()
}

pub(crate) fn weight_candidates(op: &Op) -> Vec<&TensorIr> {
    match op {
        Op::Module(Module::Linear(t)) => {
            std::iter::once(&t.weight).chain(t.bias.as_ref()).collect()
        }
        Op::Module(Module::Conv2d(t)) => {
            std::iter::once(&t.weight).chain(t.bias.as_ref()).collect()
        }
        Op::Float(_, Float::Matmul(t))
            if t.rhs.shape.as_slice()[..t.rhs.shape.num_dims().saturating_sub(2)]
                .iter()
                .all(|d| *d == 1) =>
        {
            vec![&t.rhs]
        }
        _ => vec![],
    }
}

pub(crate) fn supported(op: &Op, direct: bool) -> bool {
    // Keep non-finite scalars on the existing exact fallback path. A binary
    // cache key does not extend the numerical domain supported by graph lowering.
    let finite_scalar = match op {
        Op::NumericFloat(
            _,
            Numeric::AddScalar(t)
            | Numeric::SubScalar(t)
            | Numeric::MulScalar(t)
            | Numeric::LowerElem(t)
            | Numeric::LowerEqualElem(t),
        ) => t.rhs.elem::<f32>().is_finite(),
        Op::NumericFloat(_, Numeric::DivScalar(t)) => (1.0 / t.rhs.elem::<f32>()).is_finite(),
        Op::BaseFloat(Base::MaskFill(t)) => t.value.elem::<f32>().is_finite(),
        Op::NumericFloat(_, Numeric::Clamp(t)) => {
            t.min.elem::<f32>().is_finite() && t.max.elem::<f32>().is_finite()
        }
        Op::Float(_, Float::PowfScalar(t)) => t.rhs.elem::<f32>().is_finite(),
        Op::Module(Module::Attention(t)) => {
            t.query.shape.num_dims() == 4
                && t.key.shape.num_dims() == 4
                && t.value.shape.num_dims() == 4
                && t.options
                    .scale
                    .as_ref()
                    .is_none_or(|v| v.elem::<f32>().is_finite())
                && t.options
                    .softcap
                    .as_ref()
                    .is_none_or(|v| v.elem::<f32>().is_finite() && v.elem::<f32>() > 0.)
        }
        _ => true,
    };
    let types = crate::ir::inputs(op)
        .chain(crate::ir::outputs(op))
        .all(|t| {
            (matches!(t.dtype, DType::F32 | DType::F16) || t.dtype.is_bool())
                && t.shape.num_dims() > 0
                && t.shape.num_dims() <= 5
                && !t.shape.as_slice().contains(&0)
        });
    types
        && finite_scalar
        && (matches!(
            op,
            Op::BaseFloat(
                Base::Reshape(_)
                    | Base::Cast(_)
                    | Base::SwapDims(_)
                    | Base::Permute(_)
                    | Base::Expand(_)
                    | Base::Cat(_)
                    | Base::MaskFill(_)
                    | Base::MaskWhere(_),
            ) | Op::NumericFloat(
                _,
                Numeric::Add(_)
                    | Numeric::Sub(_)
                    | Numeric::Mul(_)
                    | Numeric::Div(_)
                    | Numeric::AddScalar(_)
                    | Numeric::SubScalar(_)
                    | Numeric::MulScalar(_)
                    | Numeric::DivScalar(_)
                    | Numeric::Abs(_)
                    | Numeric::MeanDim(_)
                    | Numeric::SumDim(_)
                    | Numeric::MaxDim(_)
                    | Numeric::MinDim(_)
                    | Numeric::LowerEqualElem(_),
            ) | Op::NumericFloat(_, Numeric::LowerElem(_) | Numeric::Clamp(_))
                | Op::Float(
                    _,
                    Float::Matmul(_)
                        | Float::Exp(_)
                        | Float::Log(_)
                        | Float::Sqrt(_)
                        | Float::Tanh(_)
                        | Float::Erf(_)
                        | Float::Recip(_),
                )
                | Op::Float(_, Float::PowfScalar(_))
                | Op::Module(Module::Linear(_) | Module::Conv2d(_) | Module::Attention(_))
        ) || (direct && matches!(op, Op::BaseFloat(Base::Slice(_) | Base::RepeatDim(_)))))
}

pub(crate) struct Plan {
    pub ops: Vec<Op>,
    pub inputs: Vec<TensorIr>,
    pub outputs: Vec<TensorIr>,
    pub constants: Vec<TensorIr>,
    pub key: Vec<u8>,
    /// Runtime-generated masks. Keeping them as inputs avoids Core ML constant
    /// folding causal attention into a CPU-only graph on some ANE shapes.
    pub auxiliary: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub logical_inputs: Vec<TensorIr>,
    pub logical_outputs: Vec<TensorIr>,
    pub padding: Option<crate::bucket::Padding>,
    pub kv_padding: Option<crate::kv::Padding>,
    compute_rows: usize,
    folded: HashSet<TensorId>,
    pub(crate) names: HashMap<TensorId, String>,
}

impl Plan {
    pub fn new(ops: &[Op], row_bucket: usize, kv_bucket: usize, pad_single_row: bool) -> Self {
        let mut live = HashMap::new();
        for op in ops {
            if let Op::Drop(t) = op {
                live.remove(&t.id);
                continue;
            }
            for input in crate::ir::inputs(op) {
                if input.status == TensorStatus::ReadWrite {
                    live.remove(&input.id);
                }
            }
            for output in crate::ir::outputs(op) {
                live.insert(output.id, output.clone());
            }
        }
        let mut needed: HashSet<_> = live.keys().copied().collect();
        let mut retained = Vec::new();
        for op in ops.iter().rev() {
            if matches!(op, Op::Drop(_)) {
                continue;
            }
            if crate::ir::outputs(op).any(|t| needed.contains(&t.id)) {
                needed.extend(crate::ir::inputs(op).map(|t| t.id));
                retained.push(op.clone());
            }
        }
        retained.reverse();
        let produced: HashSet<_> = retained
            .iter()
            .flat_map(|op| crate::ir::outputs(op).map(|t| t.id))
            .collect();
        let mut names = HashMap::new();
        let mut normalized_ids = HashMap::new();
        let mut leaves = HashMap::new();
        let mut constants = HashSet::new();
        for op in &retained {
            for t in crate::ir::inputs(op).chain(crate::ir::outputs(op)) {
                let index = names.len();
                names.entry(t.id).or_insert_with(|| {
                    normalized_ids.insert(t.id, TensorId::new(index as u64));
                    format!("t{index:06}")
                });
                if !produced.contains(&t.id) {
                    leaves.insert(t.id, t.clone());
                }
            }
            let weight = match op {
                Op::Float(_, Float::Matmul(t))
                    if t.rhs.shape.num_dims() >= 2
                        && t.rhs.shape.as_slice()[..t.rhs.shape.num_dims() - 2]
                            .iter()
                            .all(|d| *d == 1) =>
                {
                    Some(&t.rhs)
                }
                Op::Module(Module::Linear(t)) => Some(&t.weight),
                Op::Module(Module::Conv2d(t)) => Some(&t.weight),
                _ => None,
            };
            if let Some(w) = weight.filter(|w| !produced.contains(&w.id)) {
                constants.insert(w.id);
            }
            let bias = match op {
                Op::Module(Module::Linear(t)) => t.bias.as_ref(),
                Op::Module(Module::Conv2d(t)) => t.bias.as_ref(),
                _ => None,
            };
            if let Some(b) = bias.filter(|b| !produced.contains(&b.id)) {
                constants.insert(b.id);
            }
        }
        // Weight views (transpose/reshape/cast) are immutable too. Fold their
        // ancestry once, and key the graph by the original weight ID, rather than
        // treating a newly-created view as a dynamic input on every inference.
        let producers: HashMap<_, _> = retained
            .iter()
            .flat_map(|op| crate::ir::outputs(op).map(move |t| (t.id, op)))
            .collect();
        fn ancestry(
            id: TensorId,
            producers: &HashMap<TensorId, &Op>,
            roots: &mut HashSet<TensorId>,
            views: &mut HashSet<TensorId>,
        ) -> bool {
            let Some(op) = producers.get(&id) else {
                roots.insert(id);
                return true;
            };
            let input = match op {
                Op::BaseFloat(Base::Reshape(t)) => &t.input,
                Op::BaseFloat(Base::SwapDims(t)) => &t.input,
                Op::BaseFloat(Base::Permute(t)) => &t.input,
                Op::BaseFloat(Base::Cast(t)) => &t.input,
                _ => return false,
            };
            if !ancestry(input.id, producers, roots, views) {
                return false;
            }
            views.insert(id);
            true
        }
        let mut folded = HashSet::new();
        for op in &retained {
            for weight in weight_candidates(op) {
                let mut roots = HashSet::new();
                let mut views = HashSet::new();
                if ancestry(weight.id, &producers, &mut roots, &mut views) {
                    constants.extend(roots);
                    folded.extend(views);
                }
            }
        }
        let mut inputs: Vec<_> = leaves
            .values()
            .filter(|t| !constants.contains(&t.id))
            .cloned()
            .collect();
        let mut constants: Vec<_> = leaves
            .values()
            .filter(|t| constants.contains(&t.id))
            .cloned()
            .collect();
        let mut outputs: Vec<_> = live.into_values().collect();
        for tensors in [&mut inputs, &mut outputs, &mut constants] {
            tensors.sort_by(|a, b| names[&a.id].cmp(&names[&b.id]));
        }
        let compute_rows = retained
            .iter()
            .map(|op| match op {
                Op::Module(Module::Conv2d(_)) => usize::MAX,
                Op::Module(Module::Linear(t)) => {
                    t.x.shape.num_elements() / t.x.shape.as_slice().last().unwrap()
                }
                Op::Float(_, Float::Matmul(t)) => {
                    t.lhs.shape.as_slice()[t.lhs.shape.num_dims() - 2]
                }
                Op::Module(Module::Attention(t)) => t.query.shape.as_slice()[2],
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        let logical_inputs = inputs.clone();
        let logical_outputs = outputs.clone();
        let kv_padding = crate::kv::specialize(&mut retained, &mut inputs, &mut outputs, kv_bucket);
        let padding = if kv_padding.is_none() && (compute_rows != 1 || pad_single_row) {
            crate::bucket::specialize(&mut retained, &mut inputs, &mut outputs, row_bucket)
        } else {
            None
        };
        // Normalize directly on typed IR. A stable local ID and fixed ownership
        // status make the key independent of allocation order and reference count.
        let mut normalized = retained.clone();
        for op in &mut normalized {
            crate::ir::tensors_mut(op, |t| {
                t.id = normalized_ids[&t.id];
                t.status = TensorStatus::ReadOnly;
            });
        }
        // Tensor IDs are immutable values. Including constant IDs invalidates compiled
        // weights on replacement, without hashing/re-reading large weights on cache hits.
        let mut auxiliary = Vec::new();
        if let Some(padding) = padding.as_ref().filter(|p| p.attention) {
            auxiliary.push((
                "z_padding_mask".into(),
                vec![1, padding.padded, padding.padded],
                (0..padding.padded * padding.padded)
                    .map(|i| f32::from(i % padding.padded >= padding.valid))
                    .collect(),
            ));
        }
        if let Some(padding) = &padding {
            for (id, selection) in &padding.selections {
                let mut shape = vec![1; 4];
                shape[crate::mil::axis(selection.rank, selection.axis)] = padding.padded;
                let mut values = vec![0.; padding.padded];
                values[selection.position] = 1.;
                auxiliary.push((format!("z_select_{}", names[id]), shape, values));
            }
        }
        for op in &retained {
            if let Op::Module(Module::Attention(t)) = op
                && let Some(values) = kv_padding.as_ref().and_then(|p| p.mask(t.out.id))
            {
                auxiliary.push((
                    format!("z_kv_{}", names[&t.out.id]),
                    vec![1, 1, values.len()],
                    values,
                ));
            }
            if let Op::Module(Module::Attention(t)) = op
                && t.options.is_causal
            {
                let q = t.query.shape.as_slice()[2];
                let k = t.key.shape.as_slice()[2];
                let offset = k as i64 - q as i64;
                let values = kv_padding
                    .as_ref()
                    .and_then(|p| p.causal_mask(t.out.id, q))
                    .unwrap_or_else(|| {
                        (0..q)
                            .flat_map(|row| {
                                (0..k).map(move |col| f32::from(col as i64 > row as i64 + offset))
                            })
                            .collect()
                    });
                auxiliary.push((
                    format!("z_causal_{}", names[&t.out.id]),
                    vec![1, q, k],
                    values,
                ));
            }
        }
        auxiliary.sort_by(|a, b| a.0.cmp(&b.0));
        // In-memory keys need exact identity, not a textual representation.
        // This encodes the same normalized IR, constants, live outputs and mask
        // shapes without repeatedly formatting field names, enums and numbers.
        // Persistent model caches still use the generated model content hash.
        let key = bincode::serde::encode_to_vec(
            (
                normalized,
                constants.iter().map(|t| t.id.value()).collect::<Vec<_>>(),
                outputs.iter().map(|t| &names[&t.id]).collect::<Vec<_>>(),
                auxiliary
                    .iter()
                    .map(|(name, shape, _)| (name, shape))
                    .collect::<Vec<_>>(),
            ),
            bincode::config::standard(),
        )
        .unwrap();
        Self {
            ops: retained,
            inputs,
            outputs,
            constants,
            key,
            auxiliary,
            logical_inputs,
            logical_outputs,
            padding,
            kv_padding,
            compute_rows,
            folded,
            names,
        }
    }

    pub fn has_compute(&self) -> bool {
        self.ops.iter().any(|op| {
            matches!(
                op,
                Op::Float(_, Float::Matmul(_))
                    | Op::Module(Module::Linear(_) | Module::Conv2d(_) | Module::Attention(_))
            )
        })
    }

    pub fn worth_offloading(&self, minimum: usize) -> bool {
        self.compute_rows >= minimum
    }

    pub(crate) fn fold_constants(
        &self,
        constants: HashMap<TensorId, Vec<f32>>,
    ) -> HashMap<TensorId, Arc<Vec<f32>>> {
        let mut constants: HashMap<_, _> = constants
            .into_iter()
            .map(|(id, data)| (id, Arc::new(data)))
            .collect();
        for op in &self.ops {
            let output = crate::ir::outputs(op).next().unwrap();
            if !self.folded.contains(&output.id) {
                continue;
            }
            let input = crate::ir::inputs(op).next().unwrap();
            let data = constants[&input.id].clone();
            let data = match op {
                Op::BaseFloat(Base::Reshape(_)) => data,
                Op::BaseFloat(Base::Cast(t)) if t.out.dtype == DType::F16 => Arc::new(
                    data.iter()
                        .map(|v| half::f16::from_f32(*v).to_f32())
                        .collect(),
                ),
                Op::BaseFloat(Base::Cast(_)) => data,
                Op::BaseFloat(Base::SwapDims(t)) => {
                    let mut axes: Vec<_> = (0..input.shape.num_dims()).collect();
                    axes.swap(t.dim1, t.dim2);
                    Arc::new(permute_constant(&data, input.shape.as_slice(), &axes))
                }
                Op::BaseFloat(Base::Permute(t)) => {
                    Arc::new(permute_constant(&data, input.shape.as_slice(), &t.axes))
                }
                _ => unreachable!(),
            };
            constants.insert(output.id, data);
        }
        constants
    }

    /// Stable immutable weight identities, independent of activation shapes and
    /// freshly-created view IDs. Include the complete view recipe and root type.
    pub(crate) fn constant_keys(&self) -> HashMap<TensorId, Vec<u8>> {
        let mut recipes: HashMap<_, Vec<serde_json::Value>> = self
            .constants
            .iter()
            .map(|t| {
                (
                    t.id,
                    vec![serde_json::json!([t.id.value(), t.shape, t.dtype])],
                )
            })
            .collect();
        for op in &self.ops {
            let out = crate::ir::outputs(op).next().unwrap();
            if !self.folded.contains(&out.id) {
                continue;
            }
            let input = crate::ir::inputs(op).next().unwrap();
            let mut recipe = recipes[&input.id].clone();
            recipe.push(match op {
                Op::BaseFloat(Base::Reshape(t)) => serde_json::json!(["reshape", t.out.shape]),
                Op::BaseFloat(Base::Cast(t)) => serde_json::json!(["cast", t.out.dtype]),
                Op::BaseFloat(Base::SwapDims(t)) => serde_json::json!(["swap", t.dim1, t.dim2]),
                Op::BaseFloat(Base::Permute(t)) => serde_json::json!(["permute", t.axes]),
                _ => unreachable!(),
            });
            recipes.insert(out.id, recipe);
        }
        recipes
            .into_iter()
            .map(|(id, recipe)| (id, serde_json::to_vec(&recipe).unwrap()))
            .collect()
    }

    pub fn compile(&self, constants: HashMap<TensorId, Vec<f32>>) -> Result<Vec<u8>, String> {
        let constants = self.fold_constants(constants);
        let mut description = Message::default();
        for t in &self.inputs {
            description.message(1, proto::feature(&self.names[&t.id], t.shape.as_slice()));
        }
        for (name, shape, _) in &self.auxiliary {
            description.message(1, proto::feature(name, shape));
        }
        for t in &self.outputs {
            description.message(10, proto::feature(&self.names[&t.id], t.shape.as_slice()));
        }
        let mut network = Message::default();
        network.uint(5, 1); // EXACT_ARRAY_MAPPING
        let mut serial = 0;
        let mut layer = |inputs: Vec<String>, output: String, field: u32, parameters: Message| {
            let mut l = Message::default();
            l.string(1, &format!("op{serial}"));
            serial += 1;
            for input in inputs {
                l.string(2, &input);
            }
            l.string(3, &output);
            l.message(field, parameters);
            network.message(1, l);
        };
        // Embedded weights need no standalone constant layer. Emit a constant
        // only when it is also consumed as an ordinary tensor (e.g. a residual).
        let mut tensor_constants = HashSet::new();
        for op in &self.ops {
            if crate::ir::outputs(op).any(|t| self.folded.contains(&t.id)) {
                continue;
            }
            for input in crate::ir::inputs(op) {
                if !constants.contains_key(&input.id) {
                    continue;
                }
                let embedded = match op {
                    Op::Float(_, Float::Matmul(t)) => input.id == t.rhs.id,
                    Op::Module(Module::Linear(t)) => {
                        input.id == t.weight.id
                            || (constants.contains_key(&t.weight.id)
                                && t.bias.as_ref().is_some_and(|b| b.id == input.id))
                    }
                    Op::Module(Module::Conv2d(t)) => {
                        input.id == t.weight.id || t.bias.as_ref().is_some_and(|b| b.id == input.id)
                    }
                    _ => false,
                };
                if !embedded {
                    tensor_constants.insert(input.id);
                }
            }
        }
        for t in &self.outputs {
            if constants.contains_key(&t.id) {
                tensor_constants.insert(t.id);
            }
        }
        let constant_tensors = self.constants.iter().chain(
            self.ops
                .iter()
                .flat_map(crate::ir::outputs)
                .filter(|t| self.folded.contains(&t.id)),
        );
        for t in constant_tensors {
            if !tensor_constants.contains(&t.id) {
                continue;
            }
            let mut p = Message::default();
            p.shape(1, t.shape.as_slice());
            p.message(2, proto::weights(&constants[&t.id]));
            layer(vec![], self.names[&t.id].clone(), 1070, p);
        }
        for op in &self.ops {
            if crate::ir::outputs(op).any(|t| self.folded.contains(&t.id)) {
                continue;
            }
            let input_names: Vec<_> = crate::ir::inputs(op)
                .map(|t| self.names[&t.id].clone())
                .collect();
            let output = crate::ir::outputs(op).next().ok_or("missing output")?;
            let out = self.names[&output.id].clone();
            let mut p = Message::default();
            let field = match op {
                // Core ML's ANE graphs already compute at reduced precision. These casts
                // stay inside the graph; Burn dtype is restored when an output is read.
                Op::BaseFloat(Base::Cast(_)) => 600,
                Op::BaseFloat(Base::Reshape(t)) => {
                    p.shape(1, t.out.shape.as_slice());
                    1140
                }
                Op::BaseFloat(Base::Expand(t)) => {
                    p.shape(1, t.out.shape.as_slice());
                    1105
                }
                Op::BaseFloat(Base::SwapDims(t)) => {
                    let mut axes: Vec<_> = (0..t.input.shape.num_dims()).collect();
                    axes.swap(t.dim1, t.dim2);
                    p.shape(1, &axes);
                    985
                }
                Op::BaseFloat(Base::Permute(t)) => {
                    p.shape(1, &t.axes);
                    985
                }
                Op::BaseFloat(Base::Cat(t)) => {
                    if t.tensors.len() == 1 {
                        600
                    } else {
                        p.uint(1, t.dim as u64);
                        980
                    }
                }
                Op::BaseFloat(Base::MaskFill(t)) => {
                    let c = format!("{out}_fill");
                    let mut constant = Message::default();
                    constant.shape(1, &[1]);
                    constant.message(2, proto::weights(&[t.value.elem::<f32>()]));
                    layer(vec![], c.clone(), 1070, constant);
                    layer(
                        vec![
                            self.names[&t.mask.id].clone(),
                            c,
                            self.names[&t.tensor.id].clone(),
                        ],
                        out,
                        1330,
                        p,
                    );
                    continue;
                }
                Op::BaseFloat(Base::MaskWhere(t)) => {
                    layer(
                        vec![
                            self.names[&t.mask.id].clone(),
                            self.names[&t.value.id].clone(),
                            self.names[&t.tensor.id].clone(),
                        ],
                        out,
                        1330,
                        p,
                    );
                    continue;
                }
                Op::NumericFloat(_, Numeric::Add(_)) => 880,
                Op::NumericFloat(_, Numeric::Sub(_)) => 905,
                Op::NumericFloat(_, Numeric::Mul(t)) => {
                    if let Some((input, epsilon, centered)) = self.normalization_mul(t) {
                        self.normalization_layer(input, epsilon, centered, &out, &mut layer);
                        continue;
                    }
                    900
                }
                Op::NumericFloat(_, Numeric::Div(t)) => {
                    if let Some((input, epsilon, centered)) = self.normalization(t) {
                        self.normalization_layer(input, epsilon, centered, &out, &mut layer);
                        continue;
                    }
                    890
                }
                Op::NumericFloat(
                    _,
                    Numeric::AddScalar(t)
                    | Numeric::SubScalar(t)
                    | Numeric::MulScalar(t)
                    | Numeric::DivScalar(t),
                ) => {
                    let scalar = t.rhs.elem::<f32>();
                    let (alpha, beta) = match op {
                        Op::NumericFloat(_, Numeric::AddScalar(_)) => (1., scalar),
                        Op::NumericFloat(_, Numeric::SubScalar(_)) => (1., -scalar),
                        Op::NumericFloat(_, Numeric::MulScalar(_)) => (scalar, 0.),
                        _ => (1. / scalar, 0.),
                    };
                    let mut linear = Message::default();
                    linear.float(1, alpha);
                    linear.float(2, beta);
                    p.message(5, linear);
                    130
                }
                Op::NumericFloat(_, Numeric::LowerEqualElem(t) | Numeric::LowerElem(t)) => {
                    p.float(2, t.rhs.elem());
                    if matches!(op, Op::NumericFloat(_, Numeric::LowerElem(_))) {
                        825
                    } else {
                        827
                    }
                }
                Op::NumericFloat(_, Numeric::Clamp(t)) => {
                    p.float(1, t.min.elem());
                    p.float(2, t.max.elem());
                    660
                }
                Op::NumericFloat(
                    _,
                    Numeric::MeanDim(t)
                    | Numeric::SumDim(t)
                    | Numeric::MaxDim(t)
                    | Numeric::MinDim(t),
                ) => {
                    p.uint(1, t.axis as u64);
                    p.uint(2, 1);
                    match op {
                        Op::NumericFloat(_, Numeric::MeanDim(_)) => 1280,
                        Op::NumericFloat(_, Numeric::SumDim(_)) => 1270,
                        Op::NumericFloat(_, Numeric::MaxDim(_)) => 1260,
                        _ => 1265,
                    }
                }
                Op::Float(_, Float::Tanh(_)) => {
                    p.message(30, Message::default());
                    130
                }
                Op::Float(_, Float::Erf(_)) => 790,
                Op::Float(_, Float::PowfScalar(t)) => {
                    if t.rhs.elem::<f32>() == 2. {
                        layer(vec![self.names[&t.lhs.id].clone(); 2], out, 900, p);
                        continue;
                    }
                    p.uint(1, 3);
                    p.float(2, t.rhs.elem());
                    p.float(5, 1.);
                    220
                }
                Op::Float(_, Float::Exp(_) | Float::Log(_) | Float::Sqrt(_) | Float::Recip(_))
                | Op::NumericFloat(_, Numeric::Abs(_)) => {
                    if let Op::Float(_, Float::Exp(t)) = op
                        && let Some(input) = self.sigmoid_input(&t.input)
                    {
                        // Burn's default sigmoid is exp(-log(1 + exp(-x))).
                        // Keeping that expansion in FP16 loses the saturation
                        // behavior; Core ML's native activation is stable.
                        p.message(40, Message::default());
                        layer(vec![self.names[&input.id].clone()], out, 130, p);
                        continue;
                    }
                    let kind = match op {
                        Op::Float(_, Float::Exp(_)) => 4,
                        Op::Float(_, Float::Log(_)) => 5,
                        Op::Float(_, Float::Sqrt(_)) => 0,
                        Op::Float(_, Float::Recip(_)) => 2,
                        _ => 6,
                    };
                    p.uint(1, kind);
                    p.float(5, 1.);
                    220
                }
                Op::Float(_, Float::Matmul(t)) => {
                    if constants.contains_key(&t.rhs.id) {
                        self.linear_conv(&t.lhs, &t.rhs, None, &out, &constants, &mut layer);
                        continue;
                    }
                    let ins = self.matmul(&t.lhs, &t.rhs, None, &constants, &mut p)?;
                    layer(ins, out, 1045, p);
                    continue;
                }
                Op::Module(Module::Linear(t)) => {
                    let static_bias = t.bias.as_ref().filter(|b| {
                        constants.contains_key(&b.id) && constants.contains_key(&t.weight.id)
                    });
                    let embedded = constants.contains_key(&t.weight.id);
                    let linear_out = if t.bias.is_some() && static_bias.is_none() {
                        format!("{out}_linear")
                    } else {
                        out.clone()
                    };
                    if embedded {
                        self.linear_conv(
                            &t.x,
                            &t.weight,
                            static_bias,
                            &linear_out,
                            &constants,
                            &mut layer,
                        );
                    }
                    let ins = if embedded {
                        vec![]
                    } else {
                        self.matmul(&t.x, &t.weight, static_bias, &constants, &mut p)?
                    };
                    if let Some(bias) = t.bias.as_ref().filter(|_| static_bias.is_none()) {
                        let middle = format!("{out}_linear");
                        if !embedded {
                            layer(ins, middle.clone(), 1045, p);
                        }
                        layer(
                            vec![middle, self.names[&bias.id].clone()],
                            out,
                            880,
                            Message::default(),
                        );
                    } else if !embedded {
                        layer(ins, out, 1045, p);
                    }
                    continue;
                }
                Op::Module(Module::Conv2d(t)) => {
                    let w = constants
                        .get(&t.weight.id)
                        .ok_or("dynamic convolution weights are not supported yet")?;
                    let dims = &t.weight.shape.as_slice();
                    p.uint(1, dims[0] as u64);
                    p.uint(2, dims[1] as u64);
                    p.uint(10, t.options.groups as u64);
                    p.shape(20, &dims[2..]);
                    p.shape(30, &t.options.stride);
                    p.shape(40, &t.options.dilation);
                    let mut border = Message::default();
                    for pad in t.options.padding {
                        let mut edge = Message::default();
                        edge.uint(1, pad as u64);
                        edge.uint(2, pad as u64);
                        border.message(10, edge);
                    }
                    let mut valid = Message::default();
                    valid.message(1, border);
                    p.message(50, valid);
                    p.message(90, proto::weights(w));
                    if let Some(b) = &t.bias {
                        p.uint(70, 1);
                        p.message(
                            91,
                            proto::weights(constants.get(&b.id).ok_or("dynamic convolution bias")?),
                        );
                    }
                    layer(vec![self.names[&t.x.id].clone()], out, 100, p);
                    continue;
                }
                Op::Module(Module::Attention(t)) => {
                    self.attention(t, &out, &mut layer);
                    continue;
                }
                _ => return Err(format!("unsupported ANE lowering: {op:?}")),
            };
            layer(input_names, out, field, p);
        }
        let mut model = Message::default();
        model.uint(1, 5);
        model.message(2, description);
        model.message(500, network);
        Ok(model.0)
    }

    pub(crate) fn sigmoid_input(&self, exponent: &TensorIr) -> Option<&TensorIr> {
        let producer = |id| {
            self.ops
                .iter()
                .find(|op| crate::ir::outputs(op).any(|t| t.id == id))
        };
        let negative = |id| match producer(id) {
            Some(Op::NumericFloat(_, Numeric::MulScalar(t))) if t.rhs.elem::<f32>() == -1. => {
                Some(&t.lhs)
            }
            _ => None,
        };
        let log = negative(exponent.id)?;
        let Op::Float(_, Float::Log(log)) = producer(log.id)? else {
            return None;
        };
        let Op::NumericFloat(_, Numeric::AddScalar(add)) = producer(log.input.id)? else {
            return None;
        };
        if add.rhs.elem::<f32>() != 1. {
            return None;
        }
        let Op::Float(_, Float::Exp(exp)) = producer(add.lhs.id)? else {
            return None;
        };
        negative(exp.input.id)
    }

    /// Burn also emits x * reciprocal(sqrt(mean(x*x) + epsilon)). Recognize
    /// both operand orders so Direct's FP16 graph never materializes overflowing
    /// squares or an underflowing reciprocal for this complete normalization.
    pub(crate) fn normalization_mul(
        &self,
        product: &burn_ir::BinaryOpIr,
    ) -> Option<(&TensorIr, f32, bool)> {
        for (input, inverse) in [(&product.lhs, &product.rhs), (&product.rhs, &product.lhs)] {
            let reciprocal = self.ops.iter().find_map(|op| match op {
                Op::Float(_, Float::Recip(t)) if t.out.id == inverse.id => Some(t),
                _ => None,
            });
            if let Some(reciprocal) = reciprocal {
                let division = burn_ir::BinaryOpIr {
                    lhs: input.clone(),
                    rhs: reciprocal.input.clone(),
                    out: product.out.clone(),
                };
                if let Some(norm) = self.normalization(&division) {
                    return Some(norm);
                }
            }
        }
        None
    }

    pub(crate) fn normalization(
        &self,
        division: &burn_ir::BinaryOpIr,
    ) -> Option<(&TensorIr, f32, bool)> {
        let producer = |id| {
            self.ops
                .iter()
                .find(|op| crate::ir::outputs(op).any(|t| t.id == id))
        };
        let Op::Float(_, Float::Sqrt(sqrt)) = producer(division.rhs.id)? else {
            return None;
        };
        let Op::NumericFloat(_, Numeric::AddScalar(add)) = producer(sqrt.input.id)? else {
            return None;
        };
        let epsilon = add.rhs.elem::<f32>();
        if !epsilon.is_finite() || epsilon <= 0. {
            return None;
        }
        let Op::NumericFloat(_, Numeric::MeanDim(mean)) = producer(add.lhs.id)? else {
            return None;
        };
        if mean.axis + 1 != mean.input.shape.num_dims() {
            return None;
        }
        let square_input = match producer(mean.input.id)? {
            Op::Float(_, Float::PowfScalar(t)) if t.rhs.elem::<f32>() == 2. => &t.lhs,
            Op::NumericFloat(_, Numeric::Mul(t)) if t.lhs.id == t.rhs.id => &t.lhs,
            _ => return None,
        };
        if square_input.id != division.lhs.id {
            return None;
        }
        if let Some(Op::NumericFloat(_, Numeric::Sub(sub))) = producer(square_input.id)
            && let Some(Op::NumericFloat(_, Numeric::MeanDim(mean))) = producer(sub.rhs.id)
            && sub.lhs.id == mean.input.id
            && mean.axis + 1 == mean.input.shape.num_dims()
        {
            return Some((&sub.lhs, epsilon, true));
        }
        Some((square_input, epsilon, false))
    }

    fn normalization_layer(
        &self,
        input: &TensorIr,
        epsilon: f32,
        centered: bool,
        out: &str,
        layer: &mut impl FnMut(Vec<String>, String, u32, Message),
    ) {
        let width = *input.shape.as_slice().last().unwrap();
        if centered {
            let mut p = Message::default();
            p.shape(1, &[width]);
            p.float(2, epsilon);
            p.message(3, proto::weights_f32(&vec![1.; width]));
            p.message(4, proto::weights_f32(&vec![0.; width]));
            layer(vec![self.names[&input.id].clone()], out.into(), 1350, p);
        } else {
            // RMSNorm(x, eps) = sqrt(C) * L2Normalize(x, C * eps).
            // The native reduction avoids FP16 square/epsilon intermediates.
            let rows = input.shape.num_elements() / width;
            let mut p = Message::default();
            p.shape(1, &[rows, width, 1, 1]);
            layer(
                vec![self.names[&input.id].clone()],
                format!("{out}_rms_input"),
                1140,
                p,
            );
            let mut p = Message::default();
            p.float(1, epsilon * width as f32);
            layer(
                vec![format!("{out}_rms_input")],
                format!("{out}_rms_l2"),
                170,
                p,
            );
            let mut linear = Message::default();
            linear.float(1, (width as f32).sqrt());
            let mut p = Message::default();
            p.message(5, linear);
            layer(
                vec![format!("{out}_rms_l2")],
                format!("{out}_rms_scaled"),
                130,
                p,
            );
            let mut p = Message::default();
            p.shape(1, input.shape.as_slice());
            layer(vec![format!("{out}_rms_scaled")], out.into(), 1140, p);
        }
    }

    fn attention(
        &self,
        t: &burn_ir::AttentionOpIr,
        out: &str,
        layer: &mut impl FnMut(Vec<String>, String, u32, Message),
    ) {
        let batch = t.query.shape.as_slice()[0];
        let heads = t.query.shape.as_slice()[1];
        let prefix = format!("{out}_attention");
        // Keep all heads in two batched matrix operations. Splitting into separate
        // tiny heads can make Core ML schedule most of the heads on CPU.
        let mut flatten = |tensor: &TensorIr, role: &str| {
            let shape = tensor.shape.as_slice();
            let mut input = self.names[&tensor.id].clone();
            if shape[0] != batch || shape[1] != heads {
                let expanded = format!("{prefix}_{role}_expanded");
                let mut p = Message::default();
                p.shape(1, &[batch, heads, shape[2], shape[3]]);
                layer(vec![input], expanded.clone(), 1105, p);
                input = expanded;
            }
            let flat = format!("{prefix}_{role}");
            let mut p = Message::default();
            p.shape(1, &[batch * heads, shape[2], shape[3]]);
            layer(vec![input], flat.clone(), 1140, p);
            flat
        };
        let query = flatten(&t.query, "q");
        let key = flatten(&t.key, "k");
        let value = flatten(&t.value, "v");
        let mask = t.mask.as_ref().map(|t| flatten(t, "mask")).or_else(|| {
            self.padding
                .as_ref()
                .filter(|p| p.attention)
                .map(|_| "z_padding_mask".into())
        });
        let bias = t.attn_bias.as_ref().map(|t| flatten(t, "bias"));
        self.attention_scores(t, &prefix, query, key, value, mask, bias, layer);
        let mut p = Message::default();
        p.shape(1, t.out.shape.as_slice());
        layer(vec![prefix], out.into(), 1140, p);
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_scores(
        &self,
        t: &burn_ir::AttentionOpIr,
        out: &str,
        query: String,
        key: String,
        value: String,
        mask: Option<String>,
        bias: Option<String>,
        layer: &mut impl FnMut(Vec<String>, String, u32, Message),
    ) {
        let name = |suffix: &str| format!("{out}_{suffix}");
        let mut p = Message::default();
        p.uint(2, 1); // transpose key on the last two dimensions
        layer(vec![query, key], name("qk"), 1045, p);
        let scale = t
            .options
            .scale
            .as_ref()
            .map(|s| s.elem::<f32>())
            .unwrap_or_else(|| 1. / (t.query.shape.as_slice()[3] as f32).sqrt());
        let scale_params = |scale: f32| {
            let mut linear = Message::default();
            linear.float(1, scale);
            let mut p = Message::default();
            p.message(5, linear);
            p
        };
        layer(vec![name("qk")], name("scaled"), 130, scale_params(scale));
        let mut scores = name("scaled");
        if let Some(cap) = &t.options.softcap {
            let cap = cap.elem::<f32>();
            layer(
                vec![scores],
                name("capped_input"),
                130,
                scale_params(1. / cap),
            );
            let mut p = Message::default();
            p.message(30, Message::default());
            layer(vec![name("capped_input")], name("tanh"), 130, p);
            layer(vec![name("tanh")], name("capped"), 130, scale_params(cap));
            scores = name("capped");
        }
        let masked = mask.is_some() || t.options.is_causal;
        if masked {
            let mut p = Message::default();
            p.shape(1, &[1]);
            // ANE rejects NN graphs containing infinity constants. A finite floor
            // plus an explicit zero of masked numerators preserves masking semantics,
            // including an entirely masked row, without putting infinity in the graph.
            p.message(2, proto::weights(&[-65504.]));
            layer(vec![], name("negative_inf"), 1070, p);
            let mut p = Message::default();
            p.shape(1, &[1]);
            p.message(2, proto::weights(&[0.]));
            layer(vec![], name("zero"), 1070, p);
        }
        if let Some(mask) = &mask {
            layer(
                vec![mask.clone(), name("negative_inf"), scores],
                name("masked"),
                1330,
                Message::default(),
            );
            scores = name("masked");
        }
        if t.options.is_causal {
            let causal = format!("z_causal_{}", self.names[&t.out.id]);
            layer(
                vec![causal, name("negative_inf"), scores],
                name("causal_scores"),
                1330,
                Message::default(),
            );
            scores = name("causal_scores");
        }
        if let Some(bias) = bias {
            layer(vec![scores, bias], name("biased"), 880, Message::default());
            scores = name("biased");
        }
        if masked {
            // Match Burn's NaN-safe attention: a fully masked row produces zero,
            // including rectangular causal masks where q_len > k_len.
            let reduction = || {
                let mut p = Message::default();
                p.uint(1, 2);
                p.uint(2, 1);
                p
            };
            layer(vec![scores.clone()], name("max"), 1260, reduction());
            let mut p = Message::default();
            p.float(1, -65504.);
            p.float(2, 65504.);
            layer(vec![name("max")], name("finite_max"), 660, p);
            layer(
                vec![scores, name("finite_max")],
                name("shifted"),
                905,
                Message::default(),
            );
            let mut p = Message::default();
            p.uint(1, 4);
            p.float(5, 1.);
            layer(vec![name("shifted")], name("exp"), 220, p);
            let mut numerator = name("exp");
            if let Some(mask) = &mask {
                layer(
                    vec![mask.clone(), name("zero"), numerator],
                    name("unmasked_exp"),
                    1330,
                    Message::default(),
                );
                numerator = name("unmasked_exp");
            }
            if t.options.is_causal {
                let causal = format!("z_causal_{}", self.names[&t.out.id]);
                layer(
                    vec![causal, name("zero"), numerator],
                    name("causal_exp"),
                    1330,
                    Message::default(),
                );
                numerator = name("causal_exp");
            }
            layer(vec![numerator.clone()], name("sum"), 1270, reduction());
            let mut p = Message::default();
            p.float(1, half::f16::MIN_POSITIVE.to_f32());
            p.float(2, 65504.);
            layer(vec![name("sum")], name("denominator"), 660, p);
            layer(
                vec![numerator, name("denominator")],
                name("probabilities"),
                890,
                Message::default(),
            );
        } else {
            let mut p = Message::default();
            p.uint(1, 2);
            layer(vec![scores], name("probabilities"), 950, p);
        }
        layer(
            vec![name("probabilities"), value],
            out.into(),
            1045,
            Message::default(),
        );
    }

    // Static linear projections use the ANE's channels-first 4D convolution layout.
    // Flatten independent vectors into the spatial axis; preserve all Burn batch dims
    // at the graph boundary. This is a generic lowering, independent of any model.
    fn linear_conv(
        &self,
        x: &TensorIr,
        weight: &TensorIr,
        bias: Option<&TensorIr>,
        out: &str,
        constants: &HashMap<TensorId, Arc<Vec<f32>>>,
        layer: &mut impl FnMut(Vec<String>, String, u32, Message),
    ) {
        let shape = weight.shape.as_slice();
        let input = shape[shape.len() - 2];
        let output = shape[shape.len() - 1];
        let rows = x.shape.num_elements() / input;
        let name = |suffix: &str| format!("{out}_conv_{suffix}");
        let reshape = |shape: &[usize]| {
            let mut p = Message::default();
            p.shape(1, shape);
            p
        };
        let transpose = || {
            let mut p = Message::default();
            p.shape(1, &[1, 0]);
            p
        };
        layer(
            vec![self.names[&x.id].clone()],
            name("flat"),
            1140,
            reshape(&[rows, input]),
        );
        layer(vec![name("flat")], name("channels"), 985, transpose());
        layer(
            vec![name("channels")],
            name("input"),
            1140,
            reshape(&[1, input, 1, rows]),
        );
        let mut p = Message::default();
        p.uint(1, output as u64);
        p.uint(2, input as u64);
        p.uint(10, 1);
        p.shape(20, &[1, 1]);
        p.shape(30, &[1, 1]);
        p.shape(40, &[1, 1]);
        p.message(50, Message::default());
        let w = &constants[&weight.id];
        let transposed = transpose_matrix(w, input, output);
        p.message(90, proto::weights(&transposed));
        if let Some(bias) = bias {
            p.uint(70, 1);
            p.message(91, proto::weights(&constants[&bias.id]));
        }
        layer(vec![name("input")], name("output"), 100, p);
        layer(
            vec![name("output")],
            name("result"),
            1140,
            reshape(&[output, rows]),
        );
        layer(vec![name("result")], name("rows"), 985, transpose());
        let mut shape = x.shape.as_slice().to_vec();
        *shape.last_mut().unwrap() = output;
        layer(vec![name("rows")], out.into(), 1140, reshape(&shape));
    }

    fn matmul(
        &self,
        lhs: &TensorIr,
        rhs: &TensorIr,
        bias: Option<&TensorIr>,
        constants: &HashMap<TensorId, Arc<Vec<f32>>>,
        p: &mut Message,
    ) -> Result<Vec<String>, String> {
        if lhs.shape.num_dims() < 2 || rhs.shape.num_dims() < 2 {
            return Err("ANE matmul requires rank >= 2".into());
        }
        let mut inputs = vec![self.names[&lhs.id].clone()];
        if let Some(w) = constants.get(&rhs.id) {
            let rank = rhs.shape.num_dims();
            let rows = rhs.shape.as_slice()[rank - 2];
            let cols = rhs.shape.as_slice()[rank - 1];
            p.uint(5, rows as u64);
            p.uint(6, cols as u64);
            let transposed = transpose_matrix(w, rows, cols);
            p.message(8, proto::weights(&transposed));
            if let Some(b) = bias {
                p.uint(7, 1);
                p.message(9, proto::weights(&constants[&b.id]));
            }
        } else {
            if bias.is_some() {
                return Err("embedded bias requires embedded matmul weights".into());
            }
            inputs.push(self.names[&rhs.id].clone());
        }
        Ok(inputs)
    }
}
