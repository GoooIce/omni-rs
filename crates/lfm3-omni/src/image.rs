//! RGB image resizing, normalization and merge-block patch packing.
use omni_r2t2::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ImageConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub rescale_factor: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}
impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            min_pixels: 65_536,
            max_pixels: 2_097_152,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
}
#[derive(Deserialize)]
struct ProcessorConfig {
    image_processor: ProcessorImage,
}
#[derive(Deserialize)]
struct ProcessorImage {
    size: ImageSize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    rescale_factor: f32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
    do_convert_rgb: bool,
    resample: usize,
}
#[derive(Deserialize)]
struct ImageSize {
    shortest_edge: usize,
    longest_edge: usize,
}
impl ImageConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let parsed: ProcessorConfig =
            serde_json::from_slice(&std::fs::read(directory.join("processor_config.json"))?)?;
        let c = parsed.image_processor;
        if !(c.do_resize && c.do_rescale && c.do_normalize && c.do_convert_rgb) || c.resample != 3 {
            return Err(Error::Config(
                "unsupported image processor operations".into(),
            ));
        }
        let config = Self {
            min_pixels: c.size.shortest_edge,
            max_pixels: c.size.longest_edge,
            patch_size: c.patch_size,
            temporal_patch_size: c.temporal_patch_size,
            merge_size: c.merge_size,
            rescale_factor: c.rescale_factor,
            mean: c.image_mean,
            std: c.image_std,
        };
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        if self.min_pixels == 0
            || self.max_pixels < self.min_pixels
            || self.patch_size == 0
            || self.temporal_patch_size == 0
            || self.merge_size == 0
            || !self.rescale_factor.is_finite()
            || self.mean.iter().any(|v| !v.is_finite())
            || self.std.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.patch_size.checked_mul(self.merge_size).is_none()
        {
            return Err(Error::Config(
                "invalid image processor configuration".into(),
            ));
        }
        Ok(())
    }
}
/// Flattened normalized patches in merge-block order.
pub struct ImagePatches {
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Decode a PNG, JPEG or WebP upload and apply the configured image processor.
pub fn preprocess_encoded(bytes: &[u8], config: &ImageConfig) -> Result<ImagePatches> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| Error::Inference(format!("image format: {e}")))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16_384);
    limits.max_image_height = Some(16_384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|e| Error::Inference(format!("decode image: {e}")))?
        .to_rgb8();
    preprocess_rgb(
        image.as_raw(),
        image.width() as usize,
        image.height() as usize,
        config,
    )
}

pub fn preprocess_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: &ImageConfig,
) -> Result<ImagePatches> {
    config.validate()?;
    let count = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Inference("image size overflow".into()))?;
    if width == 0 || height == 0 || rgb.len() != count {
        return Err(Error::Inference("invalid RGB image shape".into()));
    }
    if width.max(height) as f64 / width.min(height) as f64 > 200.0 {
        return Err(Error::Inference("image aspect ratio exceeds 200".into()));
    }
    let factor = config.patch_size * config.merge_size;
    let mut h = ((height as f64 / factor as f64).round_ties_even() as usize) * factor;
    let mut w = ((width as f64 / factor as f64).round_ties_even() as usize) * factor;
    let area = h
        .checked_mul(w)
        .ok_or_else(|| Error::Inference("image size overflow".into()))?;
    let original_area = width
        .checked_mul(height)
        .ok_or_else(|| Error::Inference("image size overflow".into()))?;
    if area > config.max_pixels {
        let beta = (original_area as f64 / config.max_pixels as f64).sqrt();
        h = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if area < config.min_pixels {
        let beta = (config.min_pixels as f64 / original_area as f64).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    if h == 0 || w == 0 || h > u32::MAX as usize || w > u32::MAX as usize {
        return Err(Error::Inference("invalid resized image dimensions".into()));
    }
    let pixels = resize_bicubic(rgb, width, height, w, h);
    let (grid_h, grid_w) = (h / config.patch_size, w / config.patch_size);
    let patch_values = 3usize
        .checked_mul(config.temporal_patch_size)
        .and_then(|n| n.checked_mul(config.patch_size))
        .and_then(|n| n.checked_mul(config.patch_size))
        .ok_or_else(|| Error::Inference("image patch size overflow".into()))?;
    let capacity = grid_h
        .checked_mul(grid_w)
        .and_then(|n| n.checked_mul(patch_values))
        .ok_or_else(|| Error::Inference("image patch count overflow".into()))?;
    let mut patches = Vec::with_capacity(capacity);
    for by in 0..grid_h / config.merge_size {
        for bx in 0..grid_w / config.merge_size {
            for dy in 0..config.merge_size {
                for dx in 0..config.merge_size {
                    let patch_y = by * config.merge_size + dy;
                    let patch_x = bx * config.merge_size + dx;
                    for channel in 0..3 {
                        for _ in 0..config.temporal_patch_size {
                            for py in 0..config.patch_size {
                                for px in 0..config.patch_size {
                                    let y = patch_y * config.patch_size + py;
                                    let x = patch_x * config.patch_size + px;
                                    let value = pixels[(y * w + x) * 3 + channel] as f32
                                        * config.rescale_factor;
                                    patches
                                        .push((value - config.mean[channel]) / config.std[channel]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(ImagePatches {
        patches,
        grid_h,
        grid_w,
    })
}

// TorchVision resizes uint8 images in two fixed-point passes, rounding and
// clamping the intermediate image. Float-only cubic resizers differ notably
// near strong edges, even when they use the same cubic kernel.
fn resize_coefficients(input: usize, output: usize) -> (Vec<(usize, Vec<i64>)>, usize) {
    let scale = input as f64 / output as f64;
    let support = 2.0 * scale.max(1.0);
    let inv_scale = 1.0 / scale.max(1.0);
    let mut max_weight = 0.0_f64;
    let mut all = Vec::with_capacity(output);
    for index in 0..output {
        let center = scale * (index as f64 + 0.5);
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let mut weights: Vec<f64> = (start..end)
            .map(|i| {
                let x = ((i as f64 - center + 0.5) * inv_scale).abs();
                if x < 1.0 {
                    1.5 * x * x * x - 2.5 * x * x + 1.0
                } else if x < 2.0 {
                    -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
                max_weight = max_weight.max(*weight);
            }
        }
        all.push((start, weights));
    }
    let mut precision = 0;
    while precision < 22 && (0.5 + max_weight * (1_u64 << (precision + 1)) as f64) < 32768.0 {
        precision += 1;
    }
    let multiplier = (1_u64 << precision) as f64;
    let all = all
        .into_iter()
        .map(|(start, weights)| {
            (
                start,
                weights
                    .into_iter()
                    .map(|w| (w * multiplier).round() as i64)
                    .collect(),
            )
        })
        .collect();
    (all, precision)
}

pub(crate) fn resize_bicubic(
    rgb: &[u8],
    input_w: usize,
    input_h: usize,
    output_w: usize,
    output_h: usize,
) -> Vec<u8> {
    let horizontal = if input_w == output_w {
        rgb.to_vec()
    } else {
        let (coefficients, precision) = resize_coefficients(input_w, output_w);
        let mut out = vec![0u8; output_w * input_h * 3];
        for y in 0..input_h {
            for (x, (start, weights)) in coefficients.iter().enumerate() {
                for c in 0..3 {
                    let mut sum = 1_i64 << (precision - 1);
                    for (i, weight) in weights.iter().enumerate() {
                        sum += *weight * rgb[(y * input_w + start + i) * 3 + c] as i64;
                    }
                    out[(y * output_w + x) * 3 + c] = (sum >> precision).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    if input_h == output_h {
        return horizontal;
    }
    let (coefficients, precision) = resize_coefficients(input_h, output_h);
    let mut out = vec![0u8; output_w * output_h * 3];
    for (y, (start, weights)) in coefficients.iter().enumerate() {
        for x in 0..output_w {
            for c in 0..3 {
                let mut sum = 1_i64 << (precision - 1);
                for (i, weight) in weights.iter().enumerate() {
                    sum += *weight * horizontal[((start + i) * output_w + x) * 3 + c] as i64;
                }
                out[(y * output_w + x) * 3 + c] = (sum >> precision).clamp(0, 255) as u8;
            }
        }
    }
    out
}
