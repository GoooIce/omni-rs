//! Owned views of immutable native outputs. Padding is part of the view, so
//! keeping a KV output never makes its capacity holes into real tokens.
use crate::{compiler::Plan, runtime::Surface};
use burn_backend::TensorData;
use burn_ir::TensorIr;
use std::sync::Arc;

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Segment {
    pub valid: usize,
    pub capacity: usize,
}

#[derive(Clone)]
pub(crate) struct Layout {
    pub shape: Vec<usize>,
    pub axis: Option<usize>,
    pub segments: Vec<Segment>,
}

impl Layout {
    pub fn plain(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            axis: None,
            segments: vec![],
        }
    }

    pub fn tensor(plan: &Plan, tensor: &TensorIr) -> Self {
        let mut layout = Self::plain(tensor.shape.as_slice());
        if let Some(padding) = &plan.padding {
            if let Some(&axis) = padding.axes.get(&tensor.id) {
                layout.axis = Some(axis);
                layout.segments.push(Segment {
                    valid: padding.valid,
                    capacity: padding.padded,
                });
            }
        } else if let Some(padding) = &plan.kv_padding
            && let Some((axis, segments)) = padding.segments(tensor.id)
        {
            layout.axis = Some(axis);
            layout.segments = segments
                .iter()
                .map(|&(valid, capacity)| Segment { valid, capacity })
                .collect();
        }
        layout
    }

    pub fn padded_len(&self) -> usize {
        self.shape
            .iter()
            .enumerate()
            .map(|(axis, &dim)| {
                if Some(axis) == self.axis {
                    self.segments.iter().map(|s| s.capacity).sum()
                } else {
                    dim
                }
            })
            .product()
    }

    pub fn physical_shape(&self) -> Result<[usize; 4], String> {
        match self.shape.as_slice() {
            [] => Ok([1, 1, 1, 1]),
            [c] => Ok([1, *c, 1, 1]),
            [s, c] => Ok([1, *c, 1, *s]),
            [b, s, c] => Ok([*b, *c, 1, *s]),
            [b, c, h, w] => Ok([*b, *c, *h, *w]),
            _ => Err("native views require rank <= 4".into()),
        }
    }

    pub fn trim(&self, values: Vec<half::f16>) -> Vec<half::f16> {
        let Some(axis) = self.axis else { return values };
        if self.segments.iter().all(|s| s.valid == s.capacity) {
            return values;
        }
        let inner: usize = self.shape[axis + 1..].iter().product();
        let capacity: usize = self.segments.iter().map(|s| s.capacity).sum();
        let mut result = Vec::with_capacity(self.shape.iter().product());
        for chunk in values.chunks_exact(capacity * inner) {
            let mut offset = 0;
            for segment in &self.segments {
                result.extend_from_slice(&chunk[offset..offset + segment.valid * inner]);
                offset += segment.capacity * inner;
            }
        }
        result
    }
}

#[derive(Clone)]
pub(crate) struct NativeTensor {
    pub surface: Arc<Surface>,
    pub layout: Layout,
    pub dtype: burn_backend::DType,
}

impl NativeTensor {
    pub fn read(&self) -> Result<TensorData, String> {
        let mut values = vec![half::f16::ZERO; self.layout.padded_len()];
        self.surface.read(&mut values)?;
        Ok(
            TensorData::new(self.layout.trim(values), self.layout.shape.clone())
                .convert_dtype(self.dtype),
        )
    }
}

#[derive(Clone)]
pub(crate) enum StoredTensor {
    Host(TensorData),
    Native(NativeTensor),
}
