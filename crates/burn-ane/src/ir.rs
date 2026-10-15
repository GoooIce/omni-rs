//! Traversal of tensor records in the inference IR.
//!
//! Burn's read-only iterators allocate. Use borrowed slices or a small inline
//! array on the common path, delegating other operations to the upstream API.
//! Mutable traversal preserves scalar/options fields during specialization.
use burn_ir::{
    BaseOperationIr as Base, FloatOperationIr as Float, ModuleOperationIr as Module,
    NumericOperationIr as Numeric, OperationIr as Op, TensorIr,
};

enum Tensors<'a, I> {
    Inline {
        tensors: [Option<&'a TensorIr>; 5],
        offset: usize,
        len: usize,
    },
    Slice(std::slice::Iter<'a, TensorIr>),
    Upstream(I),
}

impl<'a, I> Tensors<'a, I> {
    #[inline]
    fn one(tensor: &'a TensorIr) -> Self {
        Self::Slice(std::slice::from_ref(tensor).iter())
    }

    #[inline]
    fn inline<const N: usize>(values: [Option<&'a TensorIr>; N]) -> Self {
        let mut tensors = [None; 5];
        let mut len = 0;
        for tensor in values.into_iter().flatten() {
            tensors[len] = Some(tensor);
            len += 1;
        }
        Self::Inline {
            tensors,
            offset: 0,
            len,
        }
    }
}

impl<'a, I: Iterator<Item = &'a TensorIr>> Iterator for Tensors<'a, I> {
    type Item = &'a TensorIr;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline {
                tensors,
                offset,
                len,
            } => {
                if *offset == *len {
                    return None;
                }
                let tensor = tensors[*offset];
                *offset += 1;
                tensor
            }
            Self::Slice(tensors) => tensors.next(),
            Self::Upstream(tensors) => tensors.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Inline { offset, len, .. } => {
                let remaining = len - offset;
                (remaining, Some(remaining))
            }
            Self::Slice(tensors) => tensors.size_hint(),
            Self::Upstream(tensors) => tensors.size_hint(),
        }
    }
}

#[inline]
pub(crate) fn inputs(op: &Op) -> impl Iterator<Item = &TensorIr> {
    match op {
        Op::BaseFloat(Base::Reshape(t)) => Tensors::one(&t.input),
        Op::BaseFloat(Base::Cast(t)) => Tensors::one(&t.input),
        Op::BaseFloat(Base::SwapDims(t)) => Tensors::one(&t.input),
        Op::BaseFloat(Base::Permute(t)) => Tensors::one(&t.input),
        Op::BaseFloat(Base::Expand(t)) => Tensors::one(&t.input),
        Op::BaseFloat(Base::Slice(t)) => Tensors::one(&t.tensor),
        Op::BaseFloat(Base::RepeatDim(t)) => Tensors::one(&t.tensor),
        Op::BaseFloat(Base::Cat(t)) => Tensors::Slice(t.tensors.iter()),
        Op::BaseFloat(Base::Select(t)) => Tensors::inline([Some(&t.tensor), Some(&t.indices)]),
        Op::BaseFloat(Base::MaskFill(t)) => Tensors::inline([Some(&t.tensor), Some(&t.mask)]),
        Op::BaseFloat(Base::MaskWhere(t)) => {
            Tensors::inline([Some(&t.tensor), Some(&t.mask), Some(&t.value)])
        }
        Op::NumericFloat(
            _,
            Numeric::Add(t) | Numeric::Sub(t) | Numeric::Mul(t) | Numeric::Div(t),
        ) => Tensors::inline([Some(&t.lhs), Some(&t.rhs)]),
        Op::Float(_, Float::Matmul(t)) => Tensors::inline([Some(&t.lhs), Some(&t.rhs)]),
        Op::NumericFloat(
            _,
            Numeric::AddScalar(t)
            | Numeric::SubScalar(t)
            | Numeric::MulScalar(t)
            | Numeric::DivScalar(t)
            | Numeric::LowerElem(t)
            | Numeric::LowerEqualElem(t),
        )
        | Op::Float(_, Float::PowfScalar(t)) => Tensors::one(&t.lhs),
        Op::NumericFloat(
            _,
            Numeric::MeanDim(t) | Numeric::SumDim(t) | Numeric::MaxDim(t) | Numeric::MinDim(t),
        ) => Tensors::one(&t.input),
        Op::NumericFloat(_, Numeric::Clamp(t)) => Tensors::one(&t.tensor),
        Op::NumericFloat(_, Numeric::Abs(t))
        | Op::Float(
            _,
            Float::Exp(t)
            | Float::Log(t)
            | Float::Sqrt(t)
            | Float::Tanh(t)
            | Float::Erf(t)
            | Float::Recip(t),
        ) => Tensors::one(&t.input),
        Op::Module(Module::Linear(t)) => {
            Tensors::inline([Some(&t.x), Some(&t.weight), t.bias.as_ref()])
        }
        Op::Module(Module::Conv2d(t)) => {
            Tensors::inline([Some(&t.x), Some(&t.weight), t.bias.as_ref()])
        }
        Op::Module(Module::Attention(t)) => Tensors::inline([
            Some(&t.query),
            Some(&t.key),
            Some(&t.value),
            t.mask.as_ref(),
            t.attn_bias.as_ref(),
        ]),
        Op::Custom(t) => Tensors::Slice(t.inputs.iter()),
        Op::Drop(t) => Tensors::one(t),
        // Keep the opaque upstream iterator as a generic variant: no second
        // box and no assumptions about unhandled operations or their arity.
        _ => Tensors::Upstream(op.inputs()),
    }
}

#[inline]
pub(crate) fn outputs(op: &Op) -> impl Iterator<Item = &TensorIr> {
    let out = match op {
        Op::BaseFloat(Base::Reshape(t)) => &t.out,
        Op::BaseFloat(Base::Cast(t)) => &t.out,
        Op::BaseFloat(Base::SwapDims(t)) => &t.out,
        Op::BaseFloat(Base::Permute(t)) => &t.out,
        Op::BaseFloat(Base::Expand(t)) => &t.out,
        Op::BaseFloat(Base::Slice(t)) => &t.out,
        Op::BaseFloat(Base::RepeatDim(t)) => &t.out,
        Op::BaseFloat(Base::Cat(t)) => &t.out,
        Op::BaseFloat(Base::Select(t)) => &t.out,
        Op::BaseFloat(Base::MaskFill(t)) => &t.out,
        Op::BaseFloat(Base::MaskWhere(t)) => &t.out,
        Op::NumericFloat(
            _,
            Numeric::Add(t) | Numeric::Sub(t) | Numeric::Mul(t) | Numeric::Div(t),
        ) => &t.out,
        Op::Float(_, Float::Matmul(t)) => &t.out,
        Op::NumericFloat(
            _,
            Numeric::AddScalar(t)
            | Numeric::SubScalar(t)
            | Numeric::MulScalar(t)
            | Numeric::DivScalar(t)
            | Numeric::LowerElem(t)
            | Numeric::LowerEqualElem(t),
        )
        | Op::Float(_, Float::PowfScalar(t)) => &t.out,
        Op::NumericFloat(
            _,
            Numeric::MeanDim(t) | Numeric::SumDim(t) | Numeric::MaxDim(t) | Numeric::MinDim(t),
        ) => &t.out,
        Op::NumericFloat(_, Numeric::Clamp(t)) => &t.out,
        Op::NumericFloat(_, Numeric::Abs(t))
        | Op::Float(
            _,
            Float::Exp(t)
            | Float::Log(t)
            | Float::Sqrt(t)
            | Float::Tanh(t)
            | Float::Erf(t)
            | Float::Recip(t),
        ) => &t.out,
        Op::Module(Module::Linear(t)) => &t.out,
        Op::Module(Module::Conv2d(t)) => &t.out,
        Op::Module(Module::Attention(t)) => &t.out,
        Op::Custom(t) => return Tensors::Slice(t.outputs.iter()),
        Op::Drop(_) => return Tensors::Slice([].iter()),
        _ => return Tensors::Upstream(op.outputs()),
    };
    Tensors::one(out)
}

pub(crate) fn tensors_mut(op: &mut Op, mut visit: impl FnMut(&mut TensorIr)) {
    macro_rules! nodes {
        ($($t:expr),* $(,)?) => {{ $(visit($t);)* }};
    }
    match op {
        Op::BaseFloat(Base::Reshape(t)) => nodes!(&mut t.input, &mut t.out),
        Op::BaseFloat(Base::Cast(t)) => nodes!(&mut t.input, &mut t.out),
        Op::BaseFloat(Base::SwapDims(t)) => nodes!(&mut t.input, &mut t.out),
        Op::BaseFloat(Base::Permute(t)) => nodes!(&mut t.input, &mut t.out),
        Op::BaseFloat(Base::Expand(t)) => nodes!(&mut t.input, &mut t.out),
        Op::BaseFloat(Base::Slice(t)) => nodes!(&mut t.tensor, &mut t.out),
        Op::BaseFloat(Base::RepeatDim(t)) => nodes!(&mut t.tensor, &mut t.out),
        Op::BaseFloat(Base::Cat(t)) => {
            for tensor in &mut t.tensors {
                visit(tensor);
            }
            visit(&mut t.out);
        }
        Op::BaseFloat(Base::MaskFill(t)) => nodes!(&mut t.tensor, &mut t.mask, &mut t.out),
        Op::BaseFloat(Base::MaskWhere(t)) => {
            nodes!(&mut t.tensor, &mut t.mask, &mut t.value, &mut t.out);
        }
        Op::NumericFloat(
            _,
            Numeric::Add(t) | Numeric::Sub(t) | Numeric::Mul(t) | Numeric::Div(t),
        ) => {
            nodes!(&mut t.lhs, &mut t.rhs, &mut t.out);
        }
        Op::Float(_, Float::Matmul(t)) => nodes!(&mut t.lhs, &mut t.rhs, &mut t.out),
        Op::NumericFloat(
            _,
            Numeric::AddScalar(t)
            | Numeric::SubScalar(t)
            | Numeric::MulScalar(t)
            | Numeric::DivScalar(t)
            | Numeric::LowerElem(t)
            | Numeric::LowerEqualElem(t),
        )
        | Op::Float(_, Float::PowfScalar(t)) => nodes!(&mut t.lhs, &mut t.out),
        Op::NumericFloat(
            _,
            Numeric::MeanDim(t) | Numeric::SumDim(t) | Numeric::MaxDim(t) | Numeric::MinDim(t),
        ) => nodes!(&mut t.input, &mut t.out),
        Op::NumericFloat(_, Numeric::Clamp(t)) => nodes!(&mut t.tensor, &mut t.out),
        Op::NumericFloat(_, Numeric::Abs(t))
        | Op::Float(
            _,
            Float::Exp(t)
            | Float::Log(t)
            | Float::Sqrt(t)
            | Float::Tanh(t)
            | Float::Erf(t)
            | Float::Recip(t),
        ) => nodes!(&mut t.input, &mut t.out),
        Op::Module(Module::Linear(t)) => {
            nodes!(&mut t.x, &mut t.weight);
            if let Some(bias) = &mut t.bias {
                visit(bias);
            }
            visit(&mut t.out);
        }
        Op::Module(Module::Conv2d(t)) => {
            nodes!(&mut t.x, &mut t.weight);
            if let Some(bias) = &mut t.bias {
                visit(bias);
            }
            visit(&mut t.out);
        }
        Op::Module(Module::Attention(t)) => {
            nodes!(&mut t.query, &mut t.key, &mut t.value);
            for tensor in t.mask.iter_mut().chain(t.attn_bias.iter_mut()) {
                visit(tensor);
            }
            visit(&mut t.out);
        }
        Op::Drop(t) => visit(t),
        _ => unreachable!("only supported inference operations enter ANE planning: {op:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::DType;
    use burn_ir::{
        AttentionOpIr, AttentionOptionsIr, CatOpIr, Conv2dOpIr, Conv2dOptionsIr, CustomOpIr,
        LinearOpIr, ReduceDimWithIndicesOpIr, SelectOpIr, TensorId, UnaryOpIr,
    };

    fn tensor(id: u64) -> TensorIr {
        TensorIr::uninit(TensorId::new(id), [1, 2].into(), DType::F16)
    }

    // The upstream API is the oracle, including tensor order and identity.
    // Check hints after every next() because planners collect these iterators.
    fn compare(op: &Op) {
        fn check<'a>(mut actual: impl Iterator<Item = &'a TensorIr>, expected: Vec<&TensorIr>) {
            for (index, tensor) in expected.iter().enumerate() {
                let remaining = expected.len() - index;
                let (lower, upper) = actual.size_hint();
                assert!(lower <= remaining && upper.is_none_or(|n| n >= remaining));
                assert!(std::ptr::eq(actual.next().unwrap(), *tensor));
            }
            assert!(actual.next().is_none());
            assert!(actual.next().is_none());
            assert_eq!(actual.size_hint(), (0, Some(0)));
        }
        check(inputs(op), op.inputs().collect());
        check(outputs(op), op.outputs().collect());
    }

    #[test]
    fn optional_inputs_preserve_upstream_order() {
        for bias in [None, Some(tensor(3))] {
            compare(&Op::Module(Module::Linear(LinearOpIr {
                x: tensor(1),
                weight: tensor(2),
                bias: bias.clone(),
                out: tensor(4),
            })));
            compare(&Op::Module(Module::Conv2d(Conv2dOpIr {
                x: tensor(1),
                weight: tensor(2),
                bias,
                options: Conv2dOptionsIr {
                    stride: [1, 1],
                    padding: [0, 0],
                    dilation: [1, 1],
                    groups: 1,
                },
                out: tensor(4),
            })));
        }
        for mask in [None, Some(tensor(4))] {
            for attn_bias in [None, Some(tensor(5))] {
                compare(&Op::Module(Module::Attention(AttentionOpIr {
                    query: tensor(1),
                    key: tensor(2),
                    value: tensor(3),
                    mask: mask.clone(),
                    attn_bias,
                    options: AttentionOptionsIr {
                        scale: None,
                        softcap: None,
                        is_causal: true,
                    },
                    out: tensor(6),
                })));
            }
        }
    }

    #[test]
    fn variadic_drop_and_fallback_preserve_upstream_arity() {
        for count in [0, 1, 8] {
            // Repeated IDs still represent separate input positions.
            let tensors: Vec<_> = (0..count).map(|i| tensor(i % 3)).collect();
            compare(&Op::BaseFloat(Base::Cat(CatOpIr {
                tensors: tensors.clone(),
                dim: 1,
                out: tensor(10),
            })));
            compare(&Op::Custom(CustomOpIr::new("test", &tensors, &tensors)));
        }
        compare(&Op::Drop(tensor(1)));
        let select = SelectOpIr {
            tensor: tensor(1),
            indices: tensor(2),
            dim: 1,
            out: tensor(3),
        };
        compare(&Op::BaseFloat(Base::Select(select.clone())));
        compare(&Op::BaseInt(Base::Select(select)));
        compare(&Op::Float(
            DType::F16,
            Float::Sin(UnaryOpIr {
                input: tensor(1),
                out: tensor(2),
            }),
        ));
        compare(&Op::NumericFloat(
            DType::F16,
            Numeric::MaxDimWithIndices(ReduceDimWithIndicesOpIr {
                tensor: tensor(1),
                dim: 1,
                out: tensor(2),
                out_indices: tensor(3),
            }),
        ));
    }
}
