//! Burn IR -> typed MIL, with channel-first projections and one packed weight blob.
//! This compiler is model independent. Unsupported layouts fail before evaluation.
use crate::compiler::{Plan, permute_constant, transpose_matrix};
use burn_ir::{
    BaseOperationIr as Base, FloatOperationIr as Float, ModuleOperationIr as Module,
    NumericOperationIr as Numeric, OperationIr as Op, TensorId, TensorIr,
};
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
    sync::Arc,
};

#[derive(Default)]
pub(crate) struct WeightCache {
    pub cache: crate::cache::Cache<Arc<[u8]>>,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

#[derive(Clone, Copy, Serialize)]
enum WeightLayout {
    Canonical,
    Linear,
    Bias,
}

#[derive(Clone, Serialize)]
pub(crate) struct Io {
    pub name: String,
    pub shape: Vec<usize>,
    pub logical: Vec<usize>,
    pub offset: usize,
}
pub(crate) struct Model {
    pub text: String,
    pub weights: Vec<u8>,
    pub inputs: Vec<Io>,
    pub outputs: Vec<Io>,
}
impl Model {
    pub fn len(&self) -> usize {
        self.text.len() + self.weights.len()
    }
}

// Low-rank activations stay [B,C,1,S] throughout FFNs; four-dimensional
// convolution/attention tensors retain their Burn layout. These are physical
// layouts only: reshape, broadcast and graph I/O preserve logical Burn order.
fn physical(shape: &[usize]) -> Vec<usize> {
    match shape {
        [c] => vec![1, *c, 1, 1],
        [s, c] => vec![1, *c, 1, *s],
        [b, s, c] => vec![*b, *c, 1, *s],
        _ => shape.to_vec(),
    }
}
pub(crate) fn axis(rank: usize, dim: usize) -> usize {
    match rank {
        1 => 1,
        2 => [3, 1][dim],
        3 => [0, 3, 1][dim],
        _ => dim,
    }
}
pub(crate) fn reorder<E: Copy + Default>(values: &[E], shape: &[usize], inverse: bool) -> Vec<E> {
    let mut out = vec![E::default(); values.len()];
    reorder_into(values, &mut out, shape, inverse);
    out
}

/// Reorder directly into the caller's packed buffer, without a temporary vector
/// for each graph input/output. This moves element bits without arithmetic.
pub(crate) fn reorder_into<E: Copy>(values: &[E], out: &mut [E], shape: &[usize], inverse: bool) {
    assert_eq!(values.len(), out.len());
    assert_eq!(values.len(), shape.iter().product::<usize>());
    let (b, s, c) = match shape {
        [s, c] => (1, *s, *c),
        [b, s, c] => (*b, *s, *c),
        _ => {
            out.copy_from_slice(values);
            return;
        }
    };
    if s == 1 || c == 1 {
        out.copy_from_slice(values);
        return;
    }
    for batch in 0..b {
        // Keep both sides of the transpose within a small working set. Copy
        // element bits directly, including FP16 NaNs and signed zero.
        for r0 in (0..s).step_by(32) {
            for c0 in (0..c).step_by(32) {
                for row in r0..(r0 + 32).min(s) {
                    for col in c0..(c0 + 32).min(c) {
                        let logical = (batch * s + row) * c + col;
                        let channel = (batch * c + col) * s + row;
                        if inverse {
                            out[logical] = values[channel];
                        } else {
                            out[channel] = values[logical];
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
struct Value {
    name: String,
    shape: Vec<usize>,
}
#[derive(Default)]
struct Builder {
    body: String,
    serial: usize,
    weights: Vec<u8>,
    blobs: u32,
}
impl Builder {
    fn name(&mut self) -> String {
        let n = format!("v{}", self.serial);
        self.serial += 1;
        n
    }
    fn raw(&mut self, ty: &str, expression: &str) -> String {
        let n = self.name();
        writeln!(
            self.body,
            "    {ty} {n} = {expression}[name=string(\"{n}\")];"
        )
        .unwrap();
        n
    }
    fn constant(&mut self, ty: &str, value: &str) -> String {
        let n = self.name();
        writeln!(
            self.body,
            "    {ty} {n} = const()[name=string(\"{n}\"),val={ty}({value})];"
        )
        .unwrap();
        n
    }
    fn ints(&mut self, v: &[usize]) -> String {
        self.constant(&format!("tensor<int32,[{}]>", v.len()), &format!("{v:?}"))
    }
    fn scalar(&mut self, x: f32) -> String {
        self.constant("fp16", &format!("{x:.9e}"))
    }
    fn op(&mut self, shape: &[usize], expression: &str) -> Value {
        Value {
            name: self.raw(&format!("tensor<fp16,{shape:?}>"), expression),
            shape: shape.to_vec(),
        }
    }
    fn unary(&mut self, op: &str, x: &Value) -> Value {
        let extra = if matches!(op, "inverse" | "rsqrt" | "log") {
            let epsilon = self.scalar(0.);
            format!(",epsilon={epsilon}")
        } else {
            String::new()
        };
        self.op(&x.shape, &format!("{op}(x={}{extra})", x.name))
    }
    fn binary(&mut self, op: &str, x: &Value, y: &Value, shape: &[usize]) -> Value {
        self.op(shape, &format!("{op}(x={},y={})", x.name, y.name))
    }
    fn scalar_op(&mut self, op: &str, x: &Value, n: f32) -> Value {
        let s = self.scalar(n);
        self.op(&x.shape, &format!("{op}(x={},y={s})", x.name))
    }
    fn reshape(&mut self, x: &Value, shape: &[usize]) -> Value {
        if x.shape == shape {
            return x.clone();
        }
        let s = self.ints(shape);
        self.op(shape, &format!("reshape(x={},shape={s})", x.name))
    }
    fn transpose(&mut self, x: &Value, axes: &[usize]) -> Value {
        if axes.iter().copied().eq(0..axes.len()) {
            return x.clone();
        }
        let perm = self.ints(axes);
        self.op(
            &axes.iter().map(|a| x.shape[*a]).collect::<Vec<_>>(),
            &format!("transpose(x={},perm={perm})", x.name),
        )
    }
    fn logical(&mut self, x: &Value, shape: &[usize]) -> Value {
        if shape.len() > 3 {
            return x.clone();
        }
        let transposed = self.transpose(x, &[0, 2, 3, 1]);
        self.reshape(&transposed, shape)
    }
    fn canonical(&mut self, x: &Value, shape: &[usize]) -> Value {
        if shape.len() > 3 {
            return x.clone();
        }
        let padded = match shape {
            [b, s, c] => vec![*b, 1, *s, *c],
            _ => {
                let mut p = vec![1; 4 - shape.len()];
                p.extend_from_slice(shape);
                p
            }
        };
        let padded = self.reshape(x, &padded);
        self.transpose(&padded, &[0, 3, 1, 2])
    }
    fn reduce(&mut self, op: &str, x: &Value, dim: usize) -> Value {
        let axes = self.ints(&[dim]);
        let keep = self.constant("bool", "true");
        let mut shape = x.shape.clone();
        shape[dim] = 1;
        self.op(
            &shape,
            &format!("{op}(x={},axes={axes},keep_dims={keep})", x.name),
        )
    }
    fn slice(&mut self, x: &Value, begin: &[usize], size: &[usize]) -> Value {
        let start = self.ints(begin);
        let count = self.ints(size);
        self.op(
            size,
            &format!("slice_by_size(x={},begin={start},size={count})", x.name),
        )
    }
    fn cat(&mut self, values: &[Value], dim: usize) -> Value {
        if values.len() == 1 {
            return values[0].clone();
        }
        let a = self.constant("int32", &dim.to_string());
        let interleave = self.constant("bool", "false");
        let mut shape = values[0].shape.clone();
        shape[dim] = values.iter().map(|v| v.shape[dim]).sum();
        let names = values
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        self.op(
            &shape,
            &format!("concat(values=({names}),axis={a},interleave={interleave})"),
        )
    }
    fn weight(&mut self, data: &[u8], shape: &[usize]) -> Value {
        if self.weights.is_empty() {
            self.weights.resize(64, 0);
            self.weights[4..8].copy_from_slice(&2u32.to_le_bytes());
        }
        self.weights
            .resize(self.weights.len().next_multiple_of(64), 0);
        let offset = self.weights.len();
        self.weights.resize(offset + 64, 0);
        self.weights[offset..offset + 4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        self.weights[offset + 4..offset + 8].copy_from_slice(&1u32.to_le_bytes()); // fp16
        self.weights[offset + 8..offset + 16].copy_from_slice(&(data.len() as u64).to_le_bytes());
        self.weights[offset + 16..offset + 24].copy_from_slice(&(offset as u64 + 64).to_le_bytes());
        self.weights.extend_from_slice(data);
        self.blobs += 1;
        self.weights[..4].copy_from_slice(&self.blobs.to_le_bytes());
        let name = self.constant(
            &format!("tensor<fp16,{shape:?}>"),
            &format!(
                "BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),offset=uint64({offset}))"
            ),
        );
        Value {
            name,
            shape: shape.to_vec(),
        }
    }
    fn matmul(&mut self, x: &Value, y: &Value, shape: &[usize], transpose_y: bool) -> Value {
        let no = self.constant("bool", "false");
        let yes = self.constant("bool", if transpose_y { "true" } else { "false" });
        self.op(
            shape,
            &format!(
                "matmul(x={},y={},transpose_x={no},transpose_y={yes})",
                x.name, y.name
            ),
        )
    }
    #[allow(clippy::too_many_arguments)] // The fixed MIL convolution signature.
    fn conv(
        &mut self,
        x: &Value,
        w: &Value,
        bias: Option<&Value>,
        shape: &[usize],
        stride: &[usize],
        pad: &[usize],
        dilation: &[usize],
        groups: usize,
    ) -> Value {
        let st = self.ints(stride);
        let pd = self.ints(pad);
        let dl = self.ints(dilation);
        let gr = self.constant("int32", &groups.to_string());
        let pt = self.constant("string", "\"custom\"");
        let bias = bias
            .map(|b| format!(",bias={}", b.name))
            .unwrap_or_default();
        self.op(shape, &format!("conv(x={},weight={},strides={st},pad={pd},pad_type={pt},dilations={dl},groups={gr}{bias})",x.name,w.name))
    }
    fn norm(&mut self, x: &Value, dim: usize, eps: f32, centered: bool) -> Value {
        if !centered {
            // concat(x,-x) has exactly zero mean. Native LayerNorm avoids
            // materializing fp16 squares and retains epsilon for tiny inputs.
            // Do not replace with MIL l2_norm: this device ignored its epsilon
            // in the 1e-5/1e-4 variance stress cases (see VALIDATION.md).
            let negative = self.scalar_op("mul", x, -1.);
            let source = self.cat(&[x.clone(), negative], dim);
            let axes = self.ints(&[dim]);
            let epsilon = self.constant("fp32", &format!("{eps:.9e}"));
            let normalized = self.op(
                &source.shape,
                &format!(
                    "layer_norm(x={},axes={axes},epsilon={epsilon})",
                    source.name
                ),
            );
            return self.slice(&normalized, &vec![0; x.shape.len()], &x.shape);
        }
        let source = x.clone();
        let axes = self.ints(&[dim]);
        let epsilon = self.constant("fp32", &format!("{eps:.9e}"));
        self.op(
            &source.shape,
            &format!(
                "layer_norm(x={},axes={axes},epsilon={epsilon})",
                source.name
            ),
        )
    }
    fn select(&mut self, mask: &Value, yes: &Value, no: &Value, shape: &[usize]) -> Value {
        let zero = self.scalar(0.);
        let cond = self.raw(
            &format!("tensor<bool,{:?}>", mask.shape),
            &format!("greater(x={},y={zero})", mask.name),
        );
        self.op(
            shape,
            &format!("select(cond={cond},a={},b={})", yes.name, no.name),
        )
    }

    fn sigmoid(&mut self, x: &Value) -> Value {
        // On the M5 runtime, native sigmoid (including fused SiLU) has visibly
        // inaccurate central values. exp(-abs(x)) is bounded by one, retaining
        // saturation without the overflow of a naive exp(-x) FP16 expansion.
        let magnitude = self.unary("abs", x);
        let negative = self.scalar_op("mul", &magnitude, -1.);
        let exponential = self.unary("exp", &negative);
        let denominator = self.scalar_op("add", &exponential, 1.);
        let positive = self.unary("inverse", &denominator);
        let negative = self.binary("mul", &exponential, &positive, &x.shape);
        self.select(x, &positive, &negative, &x.shape)
    }
}

struct Lower<'a> {
    plan: &'a Plan,
    b: Builder,
    values: HashMap<TensorId, Value>,
    constants: HashMap<TensorId, Arc<Vec<f32>>>,
    constant_keys: HashMap<TensorId, Vec<u8>>,
    weights: &'a mut WeightCache,
    weight_budget: usize,
    read: &'a mut dyn FnMut(&TensorIr) -> Result<Vec<f32>, String>,
    producers: HashMap<TensorId, &'a Op>,
    inputs: Vec<Io>,
    used: HashSet<String>,
}
impl Lower<'_> {
    fn constant(&mut self, t: &TensorIr) -> Result<Arc<Vec<f32>>, String> {
        if let Some(data) = self.constants.get(&t.id) {
            return Ok(data.clone());
        }
        let data = if let Some(op) = self.producers.get(&t.id).copied() {
            let input = crate::ir::inputs(op)
                .next()
                .ok_or("constant view without input")?;
            let data = self.constant(input)?;
            match op {
                Op::BaseFloat(Base::Reshape(_)) | Op::BaseFloat(Base::Cast(_)) => data,
                // Direct runtime stores all weights as FP16, so a cast view has
                // no additional rounding beyond the final packed representation.
                Op::BaseFloat(Base::SwapDims(v)) => {
                    let mut axes: Vec<_> = (0..input.shape.num_dims()).collect();
                    axes.swap(v.dim1, v.dim2);
                    Arc::new(permute_constant(&data, input.shape.as_slice(), &axes))
                }
                Op::BaseFloat(Base::Permute(v)) => {
                    Arc::new(permute_constant(&data, input.shape.as_slice(), &v.axes))
                }
                _ => return Err("unsupported constant view".into()),
            }
        } else {
            Arc::new((self.read)(t)?)
        };
        self.constants.insert(t.id, data.clone());
        Ok(data)
    }
    fn weight(&mut self, t: &TensorIr, layout: WeightLayout) -> Result<Value, String> {
        let dims = t.shape.as_slice();
        let shape = match layout {
            WeightLayout::Canonical => physical(dims),
            WeightLayout::Linear => vec![dims[dims.len() - 1], dims[dims.len() - 2], 1, 1],
            WeightLayout::Bias => vec![t.shape.num_elements()],
        };
        let identity = self
            .constant_keys
            .get(&t.id)
            .ok_or("weight is not immutable")?;
        let key = serde_json::to_vec(&(identity, layout, dims)).unwrap();
        let packed = if let Some(packed) = self.weights.cache.get(&key) {
            self.weights.hits += 1;
            packed.clone()
        } else {
            self.weights.misses += 1;
            let data = self.constant(t)?;
            let data = match layout {
                WeightLayout::Canonical => reorder(&data, dims, false),
                WeightLayout::Linear => {
                    transpose_matrix(&data, dims[dims.len() - 2], dims[dims.len() - 1])
                }
                WeightLayout::Bias => data.as_ref().clone(),
            };
            use half::slice::HalfFloatSliceExt;
            let mut converted = vec![half::f16::ZERO; data.len()];
            converted.convert_from_f32_slice(&data);
            let packed: Arc<[u8]> = converted.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.weights.evictions += self.weights.cache.insert(
                key,
                packed.clone(),
                packed.len(),
                16384,
                self.weight_budget,
            );
            packed
        };
        Ok(self.b.weight(&packed, &shape))
    }
    fn get(&mut self, t: &TensorIr) -> Result<Value, String> {
        if let Some(v) = self.values.get(&t.id) {
            return Ok(v.clone());
        }
        let v = if self.constant_keys.contains_key(&t.id) {
            self.weight(t, WeightLayout::Canonical)?
        } else if let Some(op) = self.producers.get(&t.id).copied() {
            self.lower(op)?
        } else {
            let name = format!("i{}", self.plan.names[&t.id]);
            self.used.insert(name.clone());
            Value {
                name,
                shape: physical(t.shape.as_slice()),
            }
        };
        self.values.insert(t.id, v.clone());
        Ok(v)
    }
    fn adapt(&mut self, t: &TensorIr, rank: usize) -> Result<Value, String> {
        let x = self.get(t)?;
        let mut shape = vec![1; rank.saturating_sub(t.shape.num_dims())];
        shape.extend_from_slice(t.shape.as_slice());
        if physical(&shape) == x.shape {
            return Ok(x);
        }
        let x = self.b.logical(&x, t.shape.as_slice());
        let x = self.b.reshape(&x, &shape);
        Ok(self.b.canonical(&x, &shape))
    }
    fn aux(&mut self, name: &str) -> Result<Value, String> {
        let (_, shape, _) = self
            .plan
            .auxiliary
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or("missing attention mask")?;
        self.used.insert(name.into());
        // Attention masks [1,Q,K] become [1,1,Q,K]. Token selectors already use
        // physical four-dimensional layout, so do not reorder them again.
        let mut s = vec![1; 4 - shape.len()];
        s.extend_from_slice(shape);
        Ok(Value {
            name: name.into(),
            shape: s,
        })
    }
    fn linear(
        &mut self,
        x: &TensorIr,
        w: &TensorIr,
        bias: Option<&TensorIr>,
        out: &TensorIr,
    ) -> Result<Value, String> {
        let value = self.get(x)?;
        let result = if self.constant_keys.contains_key(&w.id) {
            let dims = w.shape.as_slice();
            let cols = dims[dims.len() - 1];
            let inner = dims[dims.len() - 2];
            let weight = self.weight(w, WeightLayout::Linear)?;
            let rows = x.shape.num_elements() / inner;
            let projected = if x.shape.num_dims() <= 3
                && x.shape.as_slice()[..x.shape.num_dims().saturating_sub(2)]
                    .iter()
                    .all(|x| *x == 1)
            {
                self.b.reshape(&value, &[1, inner, 1, rows])
            } else {
                let logical = self.b.logical(&value, x.shape.as_slice());
                let logical = self.b.reshape(&logical, &[rows, inner]);
                self.b.canonical(&logical, &[rows, inner])
            };
            let y = self.b.conv(
                &projected,
                &weight,
                None,
                &[1, cols, 1, rows],
                &[1, 1],
                &[0, 0, 0, 0],
                &[1, 1],
                1,
            );
            if out.shape.num_dims() <= 3 && physical(out.shape.as_slice()) == y.shape {
                y
            } else {
                let y = self.b.logical(&y, &[rows, cols]);
                let y = self.b.reshape(&y, out.shape.as_slice());
                self.b.canonical(&y, out.shape.as_slice())
            }
        } else {
            let rhs = self.get(w)?;
            let lhs = self.b.logical(&value, x.shape.as_slice());
            let rhs = self.b.logical(&rhs, w.shape.as_slice());
            let y = self.b.matmul(&lhs, &rhs, out.shape.as_slice(), false);
            self.b.canonical(&y, out.shape.as_slice())
        };
        if let Some(bias) = bias {
            let bias = self.adapt(bias, out.shape.num_dims())?;
            Ok(self.b.binary("add", &result, &bias, &result.shape))
        } else {
            Ok(result)
        }
    }
    fn lower(&mut self, op: &Op) -> Result<Value, String> {
        let output = crate::ir::outputs(op)
            .next()
            .ok_or("MIL operation has no output")?;
        let shape = physical(output.shape.as_slice());
        let rank = output.shape.num_dims();
        Ok(match op {
            Op::BaseFloat(Base::Cast(t)) => self.get(&t.input)?,
            Op::BaseFloat(Base::Reshape(t)) => {
                let x = self.get(&t.input)?;
                let x = self.b.logical(&x, t.input.shape.as_slice());
                let x = self.b.reshape(&x, t.out.shape.as_slice());
                self.b.canonical(&x, t.out.shape.as_slice())
            }
            Op::BaseFloat(Base::SwapDims(t)) => {
                let x = self.get(&t.input)?;
                let x = self.b.logical(&x, t.input.shape.as_slice());
                let mut axes: Vec<_> = (0..rank).collect();
                axes.swap(t.dim1, t.dim2);
                let x = self.b.transpose(&x, &axes);
                self.b.canonical(&x, t.out.shape.as_slice())
            }
            Op::BaseFloat(Base::Permute(t)) => {
                let x = self.get(&t.input)?;
                let x = self.b.logical(&x, t.input.shape.as_slice());
                let x = self.b.transpose(&x, &t.axes);
                self.b.canonical(&x, t.out.shape.as_slice())
            }
            Op::BaseFloat(Base::Expand(t)) => {
                let x = self.adapt(&t.input, rank)?;
                let reps: Vec<_> = shape.iter().zip(&x.shape).map(|(a, b)| a / b).collect();
                let reps = self.b.ints(&reps);
                self.b
                    .op(&shape, &format!("tile(x={},reps={reps})", x.name))
            }
            Op::BaseFloat(Base::RepeatDim(t)) => {
                let x = self.get(&t.tensor)?;
                let mut reps = vec![1; shape.len()];
                reps[axis(rank, t.dim)] = t.times;
                let reps = self.b.ints(&reps);
                self.b
                    .op(&shape, &format!("tile(x={},reps={reps})", x.name))
            }
            Op::BaseFloat(Base::Slice(t)) => {
                let x = self.get(&t.tensor)?;
                if let Some(selection) = self
                    .plan
                    .padding
                    .as_ref()
                    .and_then(|p| p.selections.get(&t.out.id))
                {
                    let dim = axis(rank, selection.axis);
                    let mask = self.aux(&format!("z_select_{}", self.plan.names[&t.out.id]))?;
                    let zero = Value {
                        name: self.b.scalar(0.),
                        shape: vec![],
                    };
                    let selected = self.b.select(&mask, &x, &zero, &x.shape);
                    return Ok(self.b.reduce("reduce_sum", &selected, dim));
                }
                let mut start = vec![0; shape.len()];
                for (i, r) in t.ranges.iter().enumerate() {
                    if r.step != 1 {
                        return Err("direct ANE slicing requires step one".into());
                    }
                    start[axis(rank, i)] = r.to_range(t.tensor.shape[i]).start;
                }
                self.b.slice(&x, &start, &shape)
            }
            Op::BaseFloat(Base::Cat(t)) => {
                let vs: Result<Vec<_>, _> = t.tensors.iter().map(|x| self.get(x)).collect();
                self.b.cat(&vs?, axis(rank, t.dim))
            }
            Op::BaseFloat(Base::MaskFill(t)) => {
                let x = self.get(&t.tensor)?;
                let mask = self.adapt(&t.mask, rank)?;
                let scalar = self.b.scalar(t.value.elem());
                let y = Value {
                    name: scalar,
                    shape: vec![],
                };
                self.b.select(&mask, &y, &x, &shape)
            }
            Op::BaseFloat(Base::MaskWhere(t)) => {
                let x = self.adapt(&t.tensor, rank)?;
                let mask = self.adapt(&t.mask, rank)?;
                let y = self.adapt(&t.value, rank)?;
                self.b.select(&mask, &y, &x, &shape)
            }
            Op::NumericFloat(
                _,
                Numeric::Add(t) | Numeric::Sub(t) | Numeric::Mul(t) | Numeric::Div(t),
            ) => {
                let norm = match op {
                    Op::NumericFloat(_, Numeric::Div(_)) => self.plan.normalization(t),
                    Op::NumericFloat(_, Numeric::Mul(_)) => self.plan.normalization_mul(t),
                    _ => None,
                };
                if let Some((input, eps, centered)) = norm {
                    let x = self.get(input)?;
                    self.b.norm(
                        &x,
                        axis(input.shape.num_dims(), input.shape.num_dims() - 1),
                        eps,
                        centered,
                    )
                } else {
                    let x = self.adapt(&t.lhs, rank)?;
                    let y = self.adapt(&t.rhs, rank)?;
                    let kind = match op {
                        Op::NumericFloat(_, Numeric::Add(_)) => "add",
                        Op::NumericFloat(_, Numeric::Sub(_)) => "sub",
                        Op::NumericFloat(_, Numeric::Mul(_)) => "mul",
                        _ => "real_div",
                    };
                    self.b.binary(kind, &x, &y, &shape)
                }
            }
            Op::NumericFloat(
                _,
                Numeric::AddScalar(t)
                | Numeric::SubScalar(t)
                | Numeric::MulScalar(t)
                | Numeric::DivScalar(t),
            ) => {
                let x = self.get(&t.lhs)?;
                let mut n = t.rhs.elem::<f32>();
                let kind = match op {
                    Op::NumericFloat(_, Numeric::AddScalar(_)) => "add",
                    Op::NumericFloat(_, Numeric::SubScalar(_)) => "sub",
                    Op::NumericFloat(_, Numeric::MulScalar(_)) => "mul",
                    _ => {
                        n = 1. / n;
                        "mul"
                    }
                };
                self.b.scalar_op(kind, &x, n)
            }
            Op::NumericFloat(
                _,
                Numeric::MeanDim(t) | Numeric::SumDim(t) | Numeric::MaxDim(t) | Numeric::MinDim(t),
            ) => {
                let x = self.get(&t.input)?;
                let kind = match op {
                    Op::NumericFloat(_, Numeric::MeanDim(_)) => "reduce_mean",
                    Op::NumericFloat(_, Numeric::SumDim(_)) => "reduce_sum",
                    Op::NumericFloat(_, Numeric::MaxDim(_)) => "reduce_max",
                    _ => "reduce_min",
                };
                self.b.reduce(kind, &x, axis(rank, t.axis))
            }
            Op::NumericFloat(_, Numeric::LowerElem(t) | Numeric::LowerEqualElem(t)) => {
                let x = self.get(&t.lhs)?;
                let s = self.b.scalar(t.rhs.elem());
                let kind = if matches!(op, Op::NumericFloat(_, Numeric::LowerElem(_))) {
                    "less"
                } else {
                    "less_equal"
                };
                let boolv = self.b.raw(
                    &format!("tensor<bool,{shape:?}>"),
                    &format!("{kind}(x={},y={s})", x.name),
                );
                let dtype = self.b.constant("string", "\"fp16\"");
                self.b.op(&shape, &format!("cast(x={boolv},dtype={dtype})"))
            }
            Op::NumericFloat(_, Numeric::Clamp(t)) => {
                let x = self.get(&t.tensor)?;
                let x = self.b.scalar_op("maximum", &x, t.min.elem());
                self.b.scalar_op("minimum", &x, t.max.elem())
            }
            Op::Float(_, Float::PowfScalar(t)) => {
                let x = self.get(&t.lhs)?;
                if t.rhs.elem::<f32>() == 2. {
                    self.b.binary("mul", &x, &x, &shape)
                } else {
                    self.b.scalar_op("pow", &x, t.rhs.elem())
                }
            }
            Op::Float(
                _,
                Float::Exp(t)
                | Float::Log(t)
                | Float::Sqrt(t)
                | Float::Tanh(t)
                | Float::Erf(t)
                | Float::Recip(t),
            )
            | Op::NumericFloat(_, Numeric::Abs(t)) => {
                if matches!(op, Op::Float(_, Float::Exp(_)))
                    && let Some(input) = self.plan.sigmoid_input(&t.input)
                {
                    let x = self.get(input)?;
                    self.b.sigmoid(&x)
                } else {
                    let x = self.get(&t.input)?;
                    let kind = match op {
                        Op::Float(_, Float::Exp(_)) => "exp",
                        Op::Float(_, Float::Log(_)) => "log",
                        Op::Float(_, Float::Sqrt(_)) => "sqrt",
                        Op::Float(_, Float::Tanh(_)) => "tanh",
                        Op::Float(_, Float::Erf(_)) => "erf",
                        Op::Float(_, Float::Recip(_)) => "inverse",
                        _ => "abs",
                    };
                    self.b.unary(kind, &x)
                }
            }
            Op::Float(_, Float::Matmul(t)) => self.linear(&t.lhs, &t.rhs, None, &t.out)?,
            Op::Module(Module::Linear(t)) => {
                self.linear(&t.x, &t.weight, t.bias.as_ref(), &t.out)?
            }
            Op::Module(Module::Conv2d(t)) => {
                let x = self.get(&t.x)?;
                let w = self.get(&t.weight)?;
                let bias = t
                    .bias
                    .as_ref()
                    .map(|bias| self.weight(bias, WeightLayout::Bias))
                    .transpose()?;
                self.b.conv(
                    &x,
                    &w,
                    bias.as_ref(),
                    &shape,
                    &t.options.stride,
                    &[
                        t.options.padding[0],
                        t.options.padding[0],
                        t.options.padding[1],
                        t.options.padding[1],
                    ],
                    &t.options.dilation,
                    t.options.groups,
                )
            }
            Op::Module(Module::Attention(t)) => self.attention(t)?,
            _ => return Err(format!("unsupported direct ANE lowering: {op:?}")),
        })
    }
    fn attention(&mut self, t: &burn_ir::AttentionOpIr) -> Result<Value, String> {
        let q = self.get(&t.query)?;
        let k = self.get(&t.key)?;
        let v = self.get(&t.value)?;
        let s = t.query.shape.as_slice();
        let klen = t.key.shape[2];
        let score_shape = [s[0], s[1], s[2], klen];
        let mut scores = self.b.matmul(&q, &k, &score_shape, true);
        let scale = t
            .options
            .scale
            .as_ref()
            .map(|v| v.elem::<f32>())
            .unwrap_or(1. / (s[3] as f32).sqrt());
        scores = self.b.scalar_op("mul", &scores, scale);
        if let Some(cap) = &t.options.softcap {
            let cap = cap.elem::<f32>();
            scores = self.b.scalar_op("mul", &scores, 1. / cap);
            scores = self.b.unary("tanh", &scores);
            scores = self.b.scalar_op("mul", &scores, cap);
        }
        if let Some(b) = &t.attn_bias {
            let bias = self.adapt(b, 4)?;
            scores = self.b.binary("add", &scores, &bias, &score_shape);
        }
        let mut masks = Vec::new();
        if let Some(m) = &t.mask {
            masks.push(self.adapt(m, 4)?);
        }
        if t.options.is_causal {
            masks.push(self.aux(&format!("z_causal_{}", self.plan.names[&t.out.id]))?);
        }
        let kv_mask = format!("z_kv_{}", self.plan.names[&t.out.id]);
        if self.plan.auxiliary.iter().any(|(n, _, _)| n == &kv_mask) {
            masks.push(self.aux(&kv_mask)?);
        }
        if self.plan.padding.is_some()
            && self
                .plan
                .auxiliary
                .iter()
                .any(|(n, _, _)| n == "z_padding_mask")
        {
            masks.push(self.aux("z_padding_mask")?);
        }
        // Negative infinity excludes even the lowest finite scores without
        // an extra row-minimum reduction.
        let masked = Value {
            name: self.b.scalar(f32::NEG_INFINITY),
            shape: vec![],
        };
        for mask in &masks {
            scores = self.b.select(mask, &masked, &scores, &score_shape);
        }
        let peak = self.b.reduce("reduce_max", &scores, 3);
        let shifted = self.b.binary("sub", &scores, &peak, &score_shape);
        let mut exp = self.b.unary("exp", &shifted);
        let zero = Value {
            name: self.b.scalar(0.),
            shape: vec![],
        };
        // Explicit zeroing also removes NaNs from fully masked rows.
        for mask in &masks {
            exp = self.b.select(mask, &zero, &exp, &score_shape);
        }
        let sum = self.b.reduce("reduce_sum", &exp, 3);
        // A nonempty valid row contains exp(0)=1. All-masked rows divide zero by one.
        let denominator = self.b.scalar_op("maximum", &sum, 1.);
        let probs = self.b.binary("real_div", &exp, &denominator, &score_shape);
        Ok(self.b.matmul(&probs, &v, t.out.shape.as_slice(), false))
    }
}

pub(crate) fn compile(
    plan: &Plan,
    weights: &mut WeightCache,
    weight_budget: usize,
    read: &mut dyn FnMut(&TensorIr) -> Result<Vec<f32>, String>,
) -> Result<Model, String> {
    let mut offset = 0;
    let mut inputs = Vec::new();
    for t in &plan.inputs {
        inputs.push(Io {
            name: format!("i{}", plan.names[&t.id]),
            shape: physical(t.shape.as_slice()),
            logical: t.shape.as_slice().to_vec(),
            offset,
        });
        offset += t.shape.num_elements();
    }
    for (name, shape, values) in &plan.auxiliary {
        let mut padded = vec![1; 4 - shape.len()];
        padded.extend_from_slice(shape);
        inputs.push(Io {
            name: name.clone(),
            shape: padded.clone(),
            logical: padded,
            offset,
        });
        offset += values.len();
    }
    let producers = plan
        .ops
        .iter()
        .flat_map(|op| crate::ir::outputs(op).map(move |t| (t.id, op)))
        .collect();
    let mut l = Lower {
        plan,
        b: Builder::default(),
        values: HashMap::new(),
        constants: HashMap::new(),
        constant_keys: plan.constant_keys(),
        weights,
        weight_budget,
        read,
        producers,
        inputs,
        used: HashSet::new(),
    };
    let mut outputs = Vec::new();
    offset = 0;
    for t in &plan.outputs {
        let value = l.get(t)?;
        let name = format!("o{}", plan.names[&t.id]);
        let shape = physical(t.shape.as_slice());
        writeln!(
            l.b.body,
            "    tensor<fp16,{shape:?}> {name} = identity(x={})[name=string(\"{name}\")];",
            value.name
        )
        .unwrap();
        outputs.push(Io {
            name,
            shape,
            logical: t.shape.as_slice().to_vec(),
            offset,
        });
        offset += t.shape.num_elements();
    }
    l.inputs.retain(|i| l.used.contains(&i.name));
    let signature = l
        .inputs
        .iter()
        .map(|i| format!("tensor<fp16,{:?}> {}", i.shape, i.name))
        .collect::<Vec<_>>()
        .join(",");
    let result = outputs
        .iter()
        .map(|o| o.name.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let text = format!(
        "program(1.3)\n[buildInfo = dict<string,string>({{{{\"coremlc-component-MIL\",\"3510.2.1\"}},{{\"coremlc-version\",\"3505.4.1\"}},{{\"coremltools-version\",\"9.0\"}}}})]\n{{\n  func main<ios18>({signature}) {{\n{}  }} -> ({result});\n}}\n",
        l.b.body
    );
    Ok(Model {
        text,
        weights: l.b.weights,
        inputs: l.inputs,
        outputs,
    })
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn channel_layout_preserves_batch_and_element_bits() {
        let values = [0x8000_u16, 0x7e01, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut packed = [0; 12];
        reorder_into(&values, &mut packed, &[2, 2, 3], false);
        assert_eq!(packed, [0x8000, 3, 0x7e01, 4, 2, 5, 6, 9, 7, 10, 8, 11]);
        let mut restored = [0; 12];
        reorder_into(&packed, &mut restored, &[2, 2, 3], true);
        assert_eq!(restored, values);
    }

    #[test]
    fn layouts_round_trip_across_tile_boundaries_and_identity_axes() {
        for shape in [
            vec![37],
            vec![1, 37],
            vec![37, 1],
            vec![2, 33, 37],
            vec![2, 3, 4, 5],
        ] {
            let values: Vec<u16> = (0..shape.iter().product::<usize>())
                .map(|i| (i as u16).wrapping_mul(193))
                .collect();
            let mut packed = vec![0; values.len()];
            let mut restored = vec![0; values.len()];
            reorder_into(&values, &mut packed, &shape, false);
            reorder_into(&packed, &mut restored, &shape, true);
            assert_eq!(values, restored, "{shape:?}");
        }
    }
}
