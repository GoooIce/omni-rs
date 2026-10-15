//! Exact layout copies for tensors that already have host storage. No reduced
//! precision arithmetic, GPU readback, or matrix execution happens here.
use burn_backend::{DType, Element, Slice, TensorData};
use std::ops::Range;

fn select_typed<E: Element>(
    data: &TensorData,
    indices: &[usize],
    axis: usize,
) -> Option<TensorData> {
    let source = data.as_slice::<E>().ok()?;
    let mut shape = data.shape.clone();
    let extent = *shape.get(axis)?;
    if indices.iter().any(|&index| index >= extent) {
        return None;
    }
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    shape[axis] = indices.len();
    let mut values = Vec::with_capacity(shape.iter().product());
    for before in 0..outer {
        for &index in indices {
            let offset = (before * extent + index) * inner;
            values.extend_from_slice(&source[offset..offset + inner]);
        }
    }
    Some(TensorData::new(values, shape))
}

pub(crate) fn select(data: &TensorData, indices: &TensorData, axis: usize) -> Option<TensorData> {
    if indices.shape.len() != 1 {
        return None;
    }
    let indices = match indices.dtype {
        DType::I64 => indices
            .as_slice::<i64>()
            .ok()?
            .iter()
            .map(|&v| usize::try_from(v).ok())
            .collect::<Option<Vec<_>>>()?,
        DType::I32 => indices
            .as_slice::<i32>()
            .ok()?
            .iter()
            .map(|&v| usize::try_from(v).ok())
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    match data.dtype {
        DType::F16 => select_typed::<half::f16>(data, &indices, axis),
        DType::F32 => select_typed::<f32>(data, &indices, axis),
        _ => None,
    }
}

fn cat_typed<E: Element>(inputs: &[&TensorData], axis: usize) -> Option<TensorData> {
    let mut shape = inputs.first()?.shape.clone();
    *shape.get_mut(axis)? = 0;
    let mut sources = Vec::with_capacity(inputs.len());
    for data in inputs {
        if data.shape.len() != shape.len()
            || data
                .shape
                .iter()
                .zip(shape.iter())
                .enumerate()
                .any(|(d, (a, b))| d != axis && a != b)
        {
            return None;
        }
        shape[axis] = shape[axis].checked_add(data.shape[axis])?;
        sources.push(data.as_slice::<E>().ok()?);
    }
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let mut values = Vec::with_capacity(shape.iter().product());
    for before in 0..outer {
        for (data, source) in inputs.iter().zip(sources.iter()) {
            let count = data.shape[axis] * inner;
            values.extend_from_slice(&source[before * count..(before + 1) * count]);
        }
    }
    Some(TensorData::new(values, shape))
}

pub(crate) fn cat(inputs: &[&TensorData], axis: usize) -> Option<TensorData> {
    match inputs.first()?.dtype {
        DType::F16 => cat_typed::<half::f16>(inputs, axis),
        DType::F32 => cat_typed::<f32>(inputs, axis),
        _ => None,
    }
}

fn slice_typed<E: Element>(data: &TensorData, ranges: &[Range<usize>]) -> Option<TensorData> {
    let source = data.as_slice::<E>().ok()?;
    let shape: Vec<_> = ranges.iter().map(Range::len).collect();
    let Some(axis) = ranges
        .iter()
        .zip(data.shape.iter())
        .rposition(|(r, &dim)| r.start != 0 || r.end != dim)
    else {
        return Some(TensorData::new(source.to_vec(), shape));
    };
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let count = shape[axis] * inner;
    let mut values = Vec::with_capacity(shape.iter().product());
    for row in 0..outer {
        let mut position = row;
        let mut offset = ranges[axis].start * inner;
        let mut stride = data.shape[axis] * inner;
        for d in (0..axis).rev() {
            offset += (ranges[d].start + position % shape[d]) * stride;
            position /= shape[d];
            stride *= data.shape[d];
        }
        values.extend_from_slice(&source[offset..offset + count]);
    }
    Some(TensorData::new(values, shape))
}

pub(crate) fn slice(data: &TensorData, slices: &[Slice]) -> Option<TensorData> {
    if slices.len() > data.shape.len() || slices.iter().any(|s| s.step != 1) {
        return None;
    }
    let mut ranges: Vec<_> = data.shape.iter().map(|&d| 0..d).collect();
    for (axis, slice) in slices.iter().enumerate() {
        let range = slice.to_range(data.shape[axis]);
        if range.start > range.end || range.end > data.shape[axis] {
            return None;
        }
        ranges[axis] = range;
    }
    match data.dtype {
        DType::F16 => slice_typed::<half::f16>(data, &ranges),
        DType::F32 => slice_typed::<f32>(data, &ranges),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selecting_columns_and_repeated_rows_preserves_half_bits() {
        let bits = [
            0x8000, 0x7e31, 0xfeff, 0x0001, 0x7c00, 0xfc00, 0x03ff, 0x0000,
        ];
        let data = TensorData::new(bits.map(half::f16::from_bits).to_vec(), [2, 4]);
        for (axis, index, expected) in [
            (0, vec![1, 0, 1], vec![4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]),
            (1, vec![3, 1, 3], vec![3, 1, 3, 7, 5, 7]),
        ] {
            let len = index.len();
            let out = select(&data, &TensorData::new(index, [len]), axis).unwrap();
            let actual: Vec<_> = out
                .as_slice::<half::f16>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(
                actual,
                expected.iter().map(|&i| bits[i]).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn concat_matches_independent_ndarray_for_each_axis() {
        use burn_ndarray::NdArray;
        use burn_tensor::Tensor;
        for axis in 0..3 {
            let mut left_shape = [2, 3, 4];
            let mut right_shape = [2, 3, 4];
            left_shape[axis] = 1;
            right_shape[axis] = 2;
            let left = TensorData::new(
                (0..left_shape.iter().product())
                    .map(|v| v as f32 * -0.5)
                    .collect::<Vec<_>>(),
                left_shape,
            );
            let right = TensorData::new(
                (0..right_shape.iter().product())
                    .map(|v| v as f32 + 100.)
                    .collect::<Vec<_>>(),
                right_shape,
            );
            let actual = cat(&[&left, &right], axis).unwrap();
            let expected = Tensor::<NdArray, 3>::cat(
                vec![
                    Tensor::from_data(left, &Default::default()),
                    Tensor::from_data(right, &Default::default()),
                ],
                axis,
            )
            .into_data();
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(
                actual.as_slice::<f32>().unwrap(),
                expected.as_slice::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn unsupported_indices_and_mismatched_layouts_are_not_interpreted() {
        let data = TensorData::new(vec![1f32, 2., 3., 4.], [2, 2]);
        assert!(select(&data, &TensorData::new(vec![0i64], [1]), 0).is_some());
        assert!(select(&data, &TensorData::new(vec![-1i64], [1]), 0).is_none());
        assert!(select(&data, &TensorData::new(vec![2i64], [1]), 0).is_none());
        assert!(select(&data, &TensorData::new(vec![0i64], [1]), 2).is_none());
        assert!(select(&data, &TensorData::new(vec![0f32], [1]), 0).is_none());
        assert!(cat(&[], 0).is_none());
        assert!(cat(&[&data, &TensorData::new(vec![1f32, 2., 3.], [1, 3])], 0).is_none());
        assert!(
            cat(
                &[&data, &TensorData::new(vec![half::f16::ZERO; 4], [2, 2])],
                0
            )
            .is_none()
        );
    }

    #[test]
    fn multidimensional_slice_matches_ndarray_with_partial_and_negative_bounds() {
        use burn_ndarray::NdArray;
        use burn_tensor::Tensor;
        let data = TensorData::new(
            (0..40).map(|i| i as f32 - 5.).collect::<Vec<_>>(),
            [2, 4, 5],
        );
        for ranges in [
            vec![Slice::from(0..2), Slice::from(1..3), Slice::from(1..4)],
            vec![Slice::from(1..2)],
            vec![Slice::from(0..2), Slice::from(-3..-1)],
        ] {
            let actual = slice(&data, &ranges).unwrap();
            let expected = Tensor::<NdArray, 3>::from_data(data.clone(), &Default::default())
                .slice(&ranges)
                .into_data();
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(
                actual.as_slice::<f32>().unwrap(),
                expected.as_slice::<f32>().unwrap()
            );
        }
        assert!(
            slice(
                &data,
                &[Slice {
                    start: 0,
                    end: None,
                    step: 2
                }]
            )
            .is_none()
        );
    }
}
