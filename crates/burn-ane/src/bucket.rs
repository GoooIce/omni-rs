//! Conservative shape specialization. Track a token axis through the IR; never
//! infer it from coincidentally equal dimensions after an operation. Reject graphs
//! that mix/reduce that axis, or use cross-attention / externally supplied masks.
use burn_ir::{
    BaseOperationIr as Base, FloatOperationIr as Float, ModuleOperationIr as Module,
    NumericOperationIr as Numeric, OperationIr as Op, TensorId, TensorIr,
};
use std::collections::HashMap;

pub(crate) struct Padding {
    pub axes: HashMap<TensorId, usize>,
    pub valid: usize,
    pub padded: usize,
    pub attention: bool,
    pub selections: HashMap<TensorId, Selection>,
}

pub(crate) struct Selection {
    pub axis: usize,
    pub position: usize,
    pub rank: usize,
}

pub(crate) fn specialize(
    ops: &mut [Op],
    inputs: &mut [TensorIr],
    outputs: &mut [TensorIr],
    quantum: usize,
) -> Option<Padding> {
    if quantum == 0 {
        return None;
    }
    let valid = inputs.iter().find_map(|t| {
        let s = t.shape.as_slice();
        (s.len() >= 2 && s[s.len() - 2] > 0).then(|| s[s.len() - 2])
    })?;
    let mut axes: HashMap<_, _> = inputs
        .iter()
        .filter_map(|t| {
            let s = t.shape.as_slice();
            (s.len() >= 2 && s[s.len() - 2] == valid).then(|| (t.id, s.len() - 2))
        })
        .collect();
    let mut attention = false;
    let mut selections = HashMap::new();
    for op in ops.iter() {
        let marked: Vec<_> = crate::ir::inputs(op)
            .filter_map(|t| axes.get(&t.id).map(|a| (t, *a)))
            .collect();
        if marked.is_empty() {
            continue;
        }
        let output = crate::ir::outputs(op).next()?;
        let axis = match op {
            Op::BaseFloat(Base::Cat(t)) if t.tensors.len() == 1 => *axes.get(&t.tensors[0].id)?,
            Op::BaseFloat(Base::Cat(t)) => {
                let axis = *axes.get(&t.tensors[0].id)?;
                if t.dim == axis || t.tensors.iter().any(|v| axes.get(&v.id) != Some(&axis)) {
                    return None;
                }
                axis
            }
            Op::BaseFloat(Base::Slice(t)) => {
                let axis = *axes.get(&t.tensor.id)?;
                let range = t.ranges.get(axis)?.to_range(t.tensor.shape[axis]);
                if t.ranges.iter().any(|r| r.step != 1) {
                    return None;
                }
                let single_selection = range.len() == 1
                    && t.tensor.shape.num_dims() <= 4
                    && t.ranges.iter().enumerate().all(|(d, r)| {
                        d == axis || r.to_range(t.tensor.shape[d]) == (0..t.tensor.shape[d])
                    });
                if single_selection {
                    // A fixed-size token selection (often the last hidden state)
                    // terminates the padded axis. A runtime one-hot selector makes
                    // its index independent of the graph, keeping the vocabulary
                    // projection at one row rather than padding huge logits.
                    // This also applies when valid == 1: a full one-token slice
                    // is still a selection, not a request to pad the vocabulary.
                    selections.insert(
                        output.id,
                        Selection {
                            axis,
                            position: range.start,
                            rank: t.tensor.shape.num_dims(),
                        },
                    );
                    continue;
                }
                if range != (0..valid) {
                    return None;
                }
                axis
            }
            Op::BaseFloat(Base::RepeatDim(t)) => {
                let axis = *axes.get(&t.tensor.id)?;
                if t.dim == axis {
                    return None;
                }
                axis
            }
            Op::BaseFloat(Base::Reshape(t)) => {
                let input_axis = axes[&t.input.id];
                let before: usize = t.input.shape.as_slice()[..input_axis].iter().product();
                let after: usize = t.input.shape.as_slice()[input_axis + 1..].iter().product();
                let dims = t.out.shape.as_slice();
                let choices: Vec<_> = (0..dims.len())
                    .filter(|i| {
                        dims[*i] == valid
                            && dims[..*i].iter().product::<usize>() == before
                            && dims[*i + 1..].iter().product::<usize>() == after
                    })
                    .collect();
                if choices.len() > 1 && valid != 1 {
                    return None;
                }
                // Singleton axes with identical prefix/suffix products represent
                // the same flat vector. Choose the innermost one (e.g. token in
                // [1, 1, heads, width]); every later consumer is still checked.
                // Rejecting this ambiguity would prevent single-token QKV graphs
                // from sharing a useful ANE shape.
                *choices.last()?
            }
            Op::BaseFloat(Base::SwapDims(t)) => {
                let axis = axes[&t.input.id];
                if axis == t.dim1 {
                    t.dim2
                } else if axis == t.dim2 {
                    t.dim1
                } else {
                    axis
                }
            }
            Op::BaseFloat(Base::Permute(t)) => {
                t.axes.iter().position(|a| *a == axes[&t.input.id])?
            }
            Op::NumericFloat(
                _,
                Numeric::MeanDim(t) | Numeric::SumDim(t) | Numeric::MaxDim(t) | Numeric::MinDim(t),
            ) => {
                let axis = axes[&t.input.id];
                if axis == t.axis {
                    return None;
                }
                axis
            }
            Op::Float(_, Float::Matmul(t)) => {
                if axes.contains_key(&t.rhs.id) {
                    return None;
                }
                let axis = *axes.get(&t.lhs.id)?;
                if axis >= t.lhs.shape.num_dims() - 1 {
                    return None;
                }
                axis + output
                    .shape
                    .num_dims()
                    .checked_sub(t.lhs.shape.num_dims())?
            }
            Op::Module(Module::Linear(t)) => {
                if axes.contains_key(&t.weight.id)
                    || t.bias.as_ref().is_some_and(|b| axes.contains_key(&b.id))
                {
                    return None;
                }
                let axis = *axes.get(&t.x.id)?;
                if axis >= t.x.shape.num_dims() - 1 {
                    return None;
                }
                axis
            }
            Op::Module(Module::Attention(t)) => {
                if t.mask.is_some()
                    || t.attn_bias.is_some()
                    || [&t.query, &t.key, &t.value]
                        .iter()
                        .any(|t| axes.get(&t.id) != Some(&2))
                {
                    return None;
                }
                attention = true;
                2
            }
            Op::BaseFloat(
                Base::Cast(_) | Base::Expand(_) | Base::MaskFill(_) | Base::MaskWhere(_),
            )
            | Op::NumericFloat(
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
                | Numeric::LowerElem(_)
                | Numeric::LowerEqualElem(_)
                | Numeric::Clamp(_),
            )
            | Op::Float(
                _,
                Float::Exp(_)
                | Float::Log(_)
                | Float::Sqrt(_)
                | Float::Recip(_)
                | Float::Tanh(_)
                | Float::Erf(_)
                | Float::PowfScalar(_),
            ) => {
                let rank = output.shape.num_dims();
                let mut result = None;
                for (t, axis) in &marked {
                    let axis = *axis + rank.checked_sub(t.shape.num_dims())?;
                    if result.is_some_and(|previous| previous != axis) {
                        return None;
                    }
                    result = Some(axis);
                }
                let axis = result?;
                // An unmarked tensor may only broadcast on the token axis.
                for t in crate::ir::inputs(op).filter(|t| !axes.contains_key(&t.id)) {
                    let leading = rank.checked_sub(t.shape.num_dims())?;
                    if axis >= leading && t.shape.as_slice()[axis - leading] != 1 {
                        return None;
                    }
                }
                axis
            }
            _ => return None,
        };
        if output.shape.as_slice().get(axis) != Some(&valid) {
            return None;
        }
        axes.insert(output.id, axis);
    }
    // Small masked attention shapes can be scheduled entirely on CPU. A 64-token
    // minimum also avoids repeated specialization during short streaming prefixes.
    let quantum = if attention { quantum.max(64) } else { quantum };
    let padded = valid
        .checked_add(quantum - 1)?
        .checked_div(quantum)?
        .checked_mul(quantum)?;
    let resize = |t: &mut TensorIr| {
        if let Some(axis) = axes.get(&t.id) {
            t.shape[*axis] = padded;
        }
    };
    for op in ops.iter_mut() {
        crate::ir::tensors_mut(op, resize);
    }
    for op in ops.iter_mut() {
        if let Op::BaseFloat(Base::Slice(t)) = op {
            if let Some(s) = selections.get(&t.out.id) {
                t.ranges[s.axis] = (0..1).into();
            } else if let Some(axis) = axes.get(&t.out.id) {
                t.ranges[*axis] = (0..padded).into();
            }
        }
    }
    for tensor in inputs.iter_mut().chain(outputs.iter_mut()) {
        resize(tensor);
    }
    Some(Padding {
        axes,
        valid,
        padded,
        attention,
        selections,
    })
}

impl Padding {
    pub fn pad_into<E: Copy + Default>(
        &self,
        tensor: &TensorIr,
        values: &[E],
        result: &mut Vec<E>,
    ) {
        let Some(axis) = self.axes.get(&tensor.id) else {
            result.extend_from_slice(values);
            return;
        };
        if self.valid == self.padded {
            result.extend_from_slice(values);
            return;
        }
        let shape = tensor.shape.as_slice();
        let inner: usize = shape[axis + 1..].iter().product();
        for chunk in values.chunks(self.valid * inner) {
            result.extend_from_slice(chunk);
            result.resize(
                result.len() + (self.padded - self.valid) * inner,
                E::default(),
            );
        }
    }
    pub fn trim<E: Copy>(&self, tensor: &TensorIr, values: &[E]) -> Vec<E> {
        let Some(axis) = self.axes.get(&tensor.id) else {
            return values.to_vec();
        };
        let inner: usize = tensor.shape.as_slice()[axis + 1..].iter().product();
        values
            .chunks(self.padded * inner)
            .flat_map(|chunk| chunk[..self.valid * inner].iter().copied())
            .collect()
    }
}
