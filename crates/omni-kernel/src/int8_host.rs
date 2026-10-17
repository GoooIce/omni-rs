//! Host int8 gemm used when the backend has no Metal fusion runtime.

pub fn round_half_away(value: f32) -> i32 {
    let rounded = if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        (value - 0.5).ceil()
    };
    rounded.clamp(-128.0, 127.0) as i32
}

pub fn unpack_i8(words: &[i32]) -> Vec<i8> {
    words
        .iter()
        .flat_map(|word| (0..4).map(move |lane| (word >> (8 * lane)) as i8))
        .collect()
}

/// Integer dot of row-quantized activations with packed int8 weights.
pub fn gemm(
    x: &[f32],
    rows: usize,
    width: usize,
    weights: &[i8],
    scale: &[f32],
    zero: &[i32],
) -> Vec<f32> {
    let columns = scale.len();
    let mut out = vec![0.0; rows * columns];
    for row in 0..rows {
        let sample = &x[row * width..(row + 1) * width];
        let peak = sample
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let row_scale = (peak / 127.0).max(1.0e-30);
        let codes: Vec<i32> = sample
            .iter()
            .map(|value| round_half_away(value / row_scale))
            .collect();
        for column in 0..columns {
            let mut acc = 0i32;
            let mut sum = 0i32;
            for k in 0..width {
                let weight = i32::from(weights[column * width + k]);
                acc += codes[k] * weight;
                sum += codes[k];
            }
            out[row * columns + column] =
                (acc - sum * zero[column]) as f32 * row_scale * scale[column];
        }
    }
    out
}
