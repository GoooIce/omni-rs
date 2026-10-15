//! Capacity specialization for cached attention. Past tokens are padded BEFORE
//! concatenating current tokens, so appending never moves a token into a hole.
//! I/O gathers/scatters the valid segments; masks use the same logical positions.
//! Unproven consumers reject the entire rewrite, leaving the exact-shape graph.
use burn_ir::{
    BaseOperationIr as Base, ModuleOperationIr as Module, OperationIr as Op, TensorId, TensorIr,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Axis {
    dim: usize,
    // (logical length, physical capacity) in concatenation order.
    segments: Vec<(usize, usize)>,
}
impl Axis {
    fn prefix(dim: usize, valid: usize, quantum: usize) -> Option<Self> {
        Some(Self {
            dim,
            segments: vec![(valid, valid.checked_add(quantum - 1)? / quantum * quantum)],
        })
    }
    fn valid(&self) -> usize {
        self.segments.iter().map(|s| s.0).sum()
    }
    fn capacity(&self) -> usize {
        self.segments.iter().map(|s| s.1).sum()
    }
    fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        let mut offset = 0;
        self.segments.iter().flat_map(move |&(valid, capacity)| {
            let range = offset..offset + valid;
            offset += capacity;
            range
        })
    }
}

pub(crate) struct Padding {
    axes: HashMap<TensorId, Axis>,
    attention: HashMap<TensorId, Axis>,
}

fn reshape_axis(input: &TensorIr, output: &TensorIr, axis: usize) -> Option<usize> {
    let before: usize = input.shape.as_slice()[..axis].iter().product();
    let after: usize = input.shape.as_slice()[axis + 1..].iter().product();
    (0..output.shape.num_dims()).rev().find(|&d| {
        output.shape[d] == input.shape[axis]
            && output.shape.as_slice()[..d].iter().product::<usize>() == before
            && output.shape.as_slice()[d + 1..].iter().product::<usize>() == after
    })
}

// Only traverse views that preserve an independent sequence axis. This also
// recognizes grouped-query attention's reshape -> repeat heads -> reshape.
fn leaf<'a>(
    t: &'a TensorIr,
    dim: usize,
    producers: &HashMap<TensorId, &'a Op>,
) -> Option<(&'a TensorIr, usize)> {
    let Some(op) = producers.get(&t.id) else {
        return Some((t, dim));
    };
    match op {
        Op::BaseFloat(Base::Reshape(v)) => {
            leaf(&v.input, reshape_axis(&v.out, &v.input, dim)?, producers)
        }
        Op::BaseFloat(Base::RepeatDim(v)) if v.dim != dim => leaf(&v.tensor, dim, producers),
        Op::BaseFloat(Base::Cast(v)) => leaf(&v.input, dim, producers),
        Op::BaseFloat(Base::SwapDims(v)) => leaf(
            &v.input,
            if dim == v.dim1 {
                v.dim2
            } else if dim == v.dim2 {
                v.dim1
            } else {
                dim
            },
            producers,
        ),
        Op::BaseFloat(Base::Permute(v)) => leaf(&v.input, v.axes[dim], producers),
        Op::BaseFloat(Base::Slice(v))
            if v.ranges.iter().enumerate().all(|(d, r)| {
                r.step == 1
                    && r.to_range(v.tensor.shape[d]).start == 0
                    && (d == dim || v.out.shape[d] == v.tensor.shape[d])
            }) =>
        {
            leaf(&v.tensor, dim, producers)
        }
        _ => None,
    }
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
    let producers: HashMap<_, _> = ops
        .iter()
        .flat_map(|op| crate::ir::outputs(op).map(move |t| (t.id, op)))
        .collect();
    let leaves: HashSet<_> = inputs.iter().map(|t| t.id).collect();
    let mut axes = HashMap::new();
    // A graph can end between KV append and attention at the weight budget. The
    // saved outputs must still have stable physical shapes across those graphs.
    for op in ops.iter() {
        let candidates: Vec<_> = match op {
            Op::BaseFloat(Base::Cat(t))
                if t.tensors.len() == 2 && t.out.shape.num_dims() == 4 && t.dim == 2 =>
            {
                vec![(&t.tensors[0], 2)]
            }
            Op::Module(Module::Attention(t)) => vec![(&t.key, 2), (&t.value, 2)],
            _ => Vec::new(),
        };
        for (t, dim) in candidates {
            if let Some((root, dim)) = leaf(t, dim, &producers)
                && leaves.contains(&root.id)
            {
                let a = Axis::prefix(dim, root.shape[dim], quantum)?;
                if axes.get(&root.id).is_some_and(|old| old != &a) {
                    return None;
                }
                axes.insert(root.id, a);
            }
        }
    }
    if axes.is_empty() {
        return None;
    }
    let mut attention = HashMap::new();
    let mut masks = HashSet::new();
    for op in ops.iter() {
        let marked: Vec<_> = crate::ir::inputs(op)
            .filter_map(|t| axes.get(&t.id).map(|a| (t, a.clone())))
            .collect();
        if marked.is_empty() {
            continue;
        }
        let out = crate::ir::outputs(op).next()?;
        let mut a = marked[0].1.clone();
        match op {
            Op::BaseFloat(Base::Reshape(t)) => a.dim = reshape_axis(&t.input, out, a.dim)?,
            Op::BaseFloat(Base::SwapDims(t)) => {
                a.dim = if a.dim == t.dim1 {
                    t.dim2
                } else if a.dim == t.dim2 {
                    t.dim1
                } else {
                    a.dim
                };
            }
            Op::BaseFloat(Base::Permute(t)) => a.dim = t.axes.iter().position(|d| *d == a.dim)?,
            Op::BaseFloat(Base::Cast(_)) => {}
            Op::BaseFloat(Base::RepeatDim(t)) if t.dim != a.dim => {}
            Op::BaseFloat(Base::Slice(t)) => {
                let range = t.ranges.get(a.dim)?.to_range(t.tensor.shape[a.dim]);
                if a.segments.len() != 1 || range.start != 0 || t.ranges.iter().any(|r| r.step != 1)
                {
                    return None;
                }
                a = Axis::prefix(a.dim, out.shape[a.dim], quantum)?;
            }
            Op::BaseFloat(Base::Cat(t)) if t.dim == a.dim => {
                a.segments.clear();
                for v in &t.tensors {
                    if let Some(p) = axes.get(&v.id) {
                        if p.dim != a.dim {
                            return None;
                        }
                        a.segments.extend_from_slice(&p.segments);
                    } else {
                        a.segments.push((v.shape[a.dim], v.shape[a.dim]));
                    }
                }
            }
            Op::BaseFloat(Base::Cat(t))
                if t.tensors.iter().all(|v| axes.get(&v.id) == Some(&a)) => {}
            Op::Module(Module::Attention(t)) => {
                let k = axes.get(&t.key.id)?.clone();
                if k.dim != 2 || axes.get(&t.value.id) != Some(&k) || axes.contains_key(&t.query.id)
                {
                    return None;
                }
                for m in t.mask.iter().chain(t.attn_bias.iter()) {
                    let dim = m.shape.num_dims() - 1;
                    if m.shape[dim] == 1 {
                        continue;
                    }
                    if m.shape[dim] != k.valid() || !leaves.contains(&m.id) {
                        return None;
                    }
                    let mut ma = k.clone();
                    ma.dim = dim;
                    if axes.get(&m.id).is_some_and(|old| old != &ma) {
                        return None;
                    }
                    axes.insert(m.id, ma);
                    masks.insert(m.id);
                }
                attention.insert(t.out.id, k);
                continue;
            }
            _ => return None,
        }
        if out.shape.as_slice().get(a.dim) != Some(&a.valid()) {
            return None;
        }
        axes.insert(out.id, a);
    }
    // A mask/bias leaf may have been discovered after another consumer. Do not
    // change that earlier operation without proving its broadcasting semantics.
    for op in ops.iter() {
        if crate::ir::inputs(op).any(|t| masks.contains(&t.id)) {
            let Op::Module(Module::Attention(t)) = op else {
                return None;
            };
            if [&t.query, &t.key, &t.value]
                .iter()
                .any(|t| masks.contains(&t.id))
            {
                return None;
            }
        }
    }
    for op in ops.iter_mut() {
        crate::ir::tensors_mut(op, |t| {
            if let Some(a) = axes.get(&t.id) {
                t.shape[a.dim] = a.capacity();
            }
        });
    }
    for op in ops.iter_mut() {
        if let Op::BaseFloat(Base::Slice(t)) = op
            && let Some(a) = axes.get(&t.out.id)
        {
            t.ranges[a.dim] = (0..a.capacity()).into();
        }
    }
    for t in inputs.iter_mut().chain(outputs.iter_mut()) {
        if let Some(a) = axes.get(&t.id) {
            t.shape[a.dim] = a.capacity();
        }
    }
    Some(Padding { axes, attention })
}

impl Padding {
    pub fn segments(&self, id: TensorId) -> Option<(usize, &[(usize, usize)])> {
        self.axes.get(&id).map(|a| (a.dim, a.segments.as_slice()))
    }
    pub fn pad_into<E: Copy + Default>(&self, t: &TensorIr, values: &[E], result: &mut Vec<E>) {
        let Some(a) = self.axes.get(&t.id) else {
            result.extend_from_slice(values);
            return;
        };
        let inner: usize = t.shape.as_slice()[a.dim + 1..].iter().product();
        for src in values.chunks(a.valid() * inner) {
            let mut logical = 0;
            for &(valid, capacity) in &a.segments {
                let end = logical + valid * inner;
                result.extend_from_slice(&src[logical..end]);
                result.resize(result.len() + (capacity - valid) * inner, E::default());
                logical = end;
            }
        }
    }
    pub fn trim<E: Copy>(&self, t: &TensorIr, values: &[E]) -> Vec<E> {
        let Some(a) = self.axes.get(&t.id) else {
            return values.to_vec();
        };
        let inner: usize = t.shape.as_slice()[a.dim + 1..].iter().product();
        let mut result = Vec::with_capacity(t.shape.num_elements());
        for chunk in values.chunks(a.capacity() * inner) {
            let mut physical = 0;
            for &(valid, capacity) in &a.segments {
                result.extend_from_slice(&chunk[physical..physical + valid * inner]);
                physical += capacity * inner;
            }
        }
        result
    }
    pub fn mask(&self, out: TensorId) -> Option<Vec<f32>> {
        let a = self.attention.get(&out)?;
        let mut values = vec![1.; a.capacity()];
        for i in a.positions() {
            values[i] = 0.;
        }
        Some(values)
    }
    pub fn causal_mask(&self, out: TensorId, q: usize) -> Option<Vec<f32>> {
        let a = self.attention.get(&out)?;
        let mut values = vec![1.; q * a.capacity()];
        let offset = a.valid() as i64 - q as i64;
        for row in 0..q {
            for (logical, physical) in a.positions().enumerate() {
                values[row * a.capacity() + physical] =
                    f32::from(logical as i64 > row as i64 + offset);
            }
        }
        Some(values)
    }
}
