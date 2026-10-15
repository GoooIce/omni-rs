//! Image preprocessing: decode with EXIF orientation, white-matte alpha,
//! Pillow-compatible bicubic resizing, MiniCPM smart slicing and NaViT patch
//! packing. Pure CPU; no GPU weights involved.
use crate::chat::Grid;
use image::{ImageDecoder, RgbaImage};
use std::path::Path;

/// Official processor defaults; `preprocessor_config.json` overrides the mean/std.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub max_slice_nums: usize,
    pub scale_resolution: usize,
    pub patch_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            max_slice_nums: 9,
            scale_resolution: 448,
            patch_size: 14,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
}

impl PreprocessConfig {
    pub fn from_directory(directory: &Path) -> Self {
        let mut config = Self::default();
        let Ok(text) = std::fs::read_to_string(directory.join("preprocessor_config.json")) else {
            return config;
        };
        let value: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
        if let Some(nums) = value["max_slice_nums"].as_u64() {
            config.max_slice_nums = nums as usize;
        }
        if let Some(mean) = value["image_mean"].as_array() {
            for (slot, item) in config.mean.iter_mut().zip(mean) {
                *slot = item.as_f64().unwrap_or(0.5) as f32;
            }
        }
        if let Some(std) = value["image_std"].as_array() {
            for (slot, item) in config.std.iter_mut().zip(std) {
                *slot = item.as_f64().unwrap_or(0.5) as f32;
            }
        }
        config
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("decode {path}: {source}")]
    Decode {
        path: String,
        source: image::ImageError,
    },
    #[error("empty image")]
    Empty,
}

/// One resized crop: row-major RGB pixels plus its patch grid.
pub struct Crop {
    pub pixels: Vec<u8>, // (height * width * 3)
    pub width: usize,
    pub height: usize,
}

/// Fully preprocessed image: normalized patch columns per crop, the crop patch
/// grids in sequence order (thumbnail first), and the slice grid layout.
pub struct Preprocessed {
    /// Per crop, `(patches, channels * patch * patch)` normalized inputs to the
    /// patch embedding, patch rows in row-major grid order.
    pub crops: Vec<Vec<f32>>,
    pub target_sizes: Vec<Grid>,
    /// Slice grid rows/columns; `(0, 0)` when the image was not sliced.
    pub slice_grid: (usize, usize),
}

pub fn preprocess(
    image: &RgbaImage,
    config: &PreprocessConfig,
) -> Result<Preprocessed, ImageError> {
    if image.width() == 0 || image.height() == 0 {
        return Err(ImageError::Empty);
    }
    let rgb = white_matte(image);
    let (height, width) = (rgb.height() as usize, rgb.width() as usize);
    let slice_grid = sliced_grid(height, width, config);

    // Thumbnail: upscaled only when there are no slices.
    let (thumb_h, thumb_w) = find_best_resize(height, width, config, slice_grid.is_none());
    let mut crops = vec![Crop {
        pixels: resize_bicubic(&rgb, thumb_w, thumb_h),
        width: thumb_w,
        height: thumb_h,
    }];
    let mut target_sizes = vec![Grid {
        height: thumb_h / config.patch_size,
        width: thumb_w / config.patch_size,
    }];

    if let Some((rows, cols)) = slice_grid {
        let (refine_h, refine_w) = refine_size(height, width, rows, cols, config);
        let refined = resize_bicubic(&rgb, refine_w, refine_h);
        let cell_h = refine_h / rows;
        let cell_w = refine_w / cols;
        for row in 0..rows {
            for col in 0..cols {
                let mut pixels = Vec::with_capacity(cell_h * cell_w * 3);
                for y in row * cell_h..(row + 1) * cell_h {
                    pixels.extend_from_slice(
                        &refined[y * refine_w * 3 + col * cell_w * 3
                            ..y * refine_w * 3 + (col + 1) * cell_w * 3],
                    );
                }
                crops.push(Crop {
                    pixels,
                    width: cell_w,
                    height: cell_h,
                });
                target_sizes.push(Grid {
                    height: cell_h / config.patch_size,
                    width: cell_w / config.patch_size,
                });
            }
        }
    }

    let crops = crops
        .iter()
        .map(|crop| pack_patches(crop, config))
        .collect();
    Ok(Preprocessed {
        crops,
        target_sizes,
        slice_grid: slice_grid.unwrap_or((0, 0)),
    })
}

/// EXIF-oriented decode; the product applies orientation regardless of the
/// official backend's behavior.
pub fn open(path: &Path) -> Result<RgbaImage, ImageError> {
    let decode = |source: image::ImageError| ImageError::Decode {
        path: path.display().to_string(),
        source,
    };
    let mut reader = image::ImageReader::open(path)
        .map_err(image::ImageError::IoError)
        .map_err(decode)?
        .with_guessed_format()
        .map_err(image::ImageError::IoError)
        .map_err(decode)?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(decode)?;
    let orientation = decoder.orientation().map_err(decode)?;
    let mut decoded = image::DynamicImage::from_decoder(decoder).map_err(decode)?;
    decoded.apply_orientation(orientation);
    Ok(decoded.to_rgba8())
}

/// PIL paste(mask=alpha) over white: integer round-to-nearest compositing.
pub fn white_matte(image: &RgbaImage) -> image::RgbImage {
    let mut out = image::RgbImage::new(image.width(), image.height());
    for (pixel, target) in image.pixels().zip(out.pixels_mut()) {
        let a = pixel[3] as u32;
        for c in 0..3 {
            target[c] = ((pixel[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        }
    }
    out
}

fn ensure_divide(length: usize, divisor: usize) -> usize {
    // Python round() breaks ties to even; match it so grid math stays identical.
    ((length as f64 / divisor as f64).round_ties_even() as usize).max(1) * divisor
}

fn find_best_resize_raw(
    height: f64,
    width: f64,
    scale_resolution: usize,
    patch: usize,
) -> (usize, usize) {
    let aspect = width / height;
    let height = (scale_resolution as f64 / aspect.sqrt()) as usize;
    let width = (height as f64 * aspect) as usize;
    (
        ensure_divide(height, patch * 4),
        ensure_divide(width, patch * 4),
    )
}

/// Thumbnail target: bounded by `scale_resolution²` (upscale only when unsliced).
pub fn find_best_resize(
    height: usize,
    width: usize,
    config: &PreprocessConfig,
    allow_upscale: bool,
) -> (usize, usize) {
    if height * width > config.scale_resolution.pow(2) || allow_upscale {
        return find_best_resize_raw(
            height as f64,
            width as f64,
            config.scale_resolution,
            config.patch_size,
        );
    }
    let patch = config.patch_size;
    (
        ensure_divide(height, patch * 4),
        ensure_divide(width, patch * 4),
    )
}

/// Best slice grid `[rows, cols]` matching the aspect ratio, or `None`.
pub fn sliced_grid(
    height: usize,
    width: usize,
    config: &PreprocessConfig,
) -> Option<(usize, usize)> {
    let log_ratio = (width as f64 / height as f64).ln();
    let ratio = (width * height) as f64 / config.scale_resolution.pow(2) as f64;
    let multiple = (ratio.ceil() as usize).min(config.max_slice_nums);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1usize, 1usize);
    let mut min_error = f64::INFINITY;
    for num_slices in [multiple - 1, multiple, multiple + 1] {
        if num_slices <= 1 || num_slices > config.max_slice_nums {
            continue;
        }
        for rows in 1..=num_slices {
            if num_slices % rows == 0 {
                let cols = num_slices / rows;
                let error = (log_ratio - (cols as f64 / rows as f64).ln()).abs();
                if error < min_error || (error == min_error && rows > best.0) {
                    best = (rows, cols);
                    min_error = error;
                }
            }
        }
    }
    Some(best)
}

fn refine_size(
    height: usize,
    width: usize,
    rows: usize,
    cols: usize,
    config: &PreprocessConfig,
) -> (usize, usize) {
    let refine_h = ensure_divide(height, rows);
    let refine_w = ensure_divide(width, cols);
    let (best_h, best_w) = find_best_resize_raw(
        refine_h as f64 / rows as f64,
        refine_w as f64 / cols as f64,
        config.scale_resolution,
        config.patch_size,
    );
    (best_h * rows, best_w * cols)
}

/// Normalized patch rows `(patches, C*patch*patch)` in row-major patch order;
/// the flat channel order is `c * patch * patch + dy * patch + dx`, matching the
/// checkpoint's conv weight layout.
fn pack_patches(crop: &Crop, config: &PreprocessConfig) -> Vec<f32> {
    let patch = config.patch_size;
    let grid_h = crop.height / patch;
    let grid_w = crop.width / patch;
    let mut out = vec![0f32; grid_h * grid_w * 3 * patch * patch];
    for ih in 0..grid_h {
        for iw in 0..grid_w {
            let row = (ih * grid_w + iw) * 3 * patch * patch;
            for dy in 0..patch {
                for dx in 0..patch {
                    let y = ih * patch + dy;
                    let x = iw * patch + dx;
                    for c in 0..3 {
                        let value = crop.pixels[(y * crop.width + x) * 3 + c] as f32 / 255.0;
                        let normalized = (value - config.mean[c]) / config.std[c];
                        out[row + c * patch * patch + dy * patch + dx] = normalized;
                    }
                }
            }
        }
    }
    out
}

// ---- Pillow-compatible bicubic resize -------------------------------------

fn bicubic_weight(v: f64) -> f64 {
    // Pillow BICUBIC: cubic BC filter with B=0, C=0.5, support 2.
    let a = v.abs();
    if a < 1.0 {
        1.5 * a.powi(3) - 2.5 * a.powi(2) + 1.0
    } else if a < 2.0 {
        -0.5 * a.powi(3) + 2.5 * a.powi(2) - 4.0 * a + 2.0
    } else {
        0.0
    }
}

fn resize_coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter = scale.max(1.0) * 2.0;
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let left = ((center - filter + 0.5) as isize).max(0) as usize;
            let right = ((center + filter + 0.5) as usize).min(input);
            let weights: Vec<f64> = (left..right)
                .map(|x| bicubic_weight((x as f64 - center + 0.5) / filter * 2.0))
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum == 0.0 {
                return (left, vec![1 << 22; right - left]);
            }
            (
                left,
                weights
                    .into_iter()
                    .map(|w| (w / sum * (1 << 22) as f64).round() as i64)
                    .collect(),
            )
        })
        .collect()
}

/// Separable fixed-point bicubic over opaque RGB, rounding after each pass,
/// mirroring Pillow's uint8 resample.
pub fn resize_bicubic(image: &image::RgbImage, width: usize, height: usize) -> Vec<u8> {
    let (iw, ih) = (image.width() as usize, image.height() as usize);
    if (iw, ih) == (width, height) {
        return image.as_raw().clone();
    }
    let source = image.as_raw();
    let horizontal = if iw == width {
        source.clone()
    } else {
        let weights = resize_coefficients(iw, width);
        let mut out = vec![0u8; width * ih * 3];
        for y in 0..ih {
            for (x, (left, coeffs)) in weights.iter().enumerate() {
                for c in 0..3 {
                    let sum = coeffs.iter().enumerate().fold(1 << 21, |acc, (i, k)| {
                        acc + *k * source[(y * iw + left + i) * 3 + c] as i64
                    });
                    out[(y * width + x) * 3 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    if ih == height {
        return horizontal;
    }
    let weights = resize_coefficients(ih, height);
    let mut out = vec![0u8; width * height * 3];
    for (y, (top, coeffs)) in weights.iter().enumerate() {
        for x in 0..width {
            for c in 0..3 {
                let sum = coeffs.iter().enumerate().fold(1 << 21, |acc, (i, k)| {
                    acc + *k * horizontal[((top + i) * width + x) * 3 + c] as i64
                });
                out[(y * width + x) * 3 + c] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::SafeTensors;

    fn fixtures_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn load_fixture(name: &str) -> (Vec<Vec<u8>>, Vec<Grid>, (usize, usize)) {
        let bytes = std::fs::read(
            fixtures_dir()
                .join("preprocess")
                .join(format!("{name}.safetensors")),
        )
        .unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let mut sizes = Vec::new();
        let ts = tensors.tensor("target_sizes").unwrap();
        let ts_data: Vec<i64> = ts.data().chunks_exact::<8>().map(le_i64).collect();
        for pair in ts_data.chunks(2) {
            sizes.push(Grid {
                height: pair[0] as usize,
                width: pair[1] as usize,
            });
        }
        let mut crops = Vec::new();
        let mut index = 0;
        while let Ok(view) = tensors.tensor(&format!("crop_{index}")) {
            crops.push(view.data().to_vec());
            index += 1;
        }
        (crops, sizes, (0, 0))
    }

    fn le_i64(bytes: &[u8]) -> i64 {
        let mut buffer = [0u8; 8];
        buffer.copy_from_slice(bytes);
        i64::from_le_bytes(buffer)
    }

    fn grid_of(name: &str) -> (usize, usize) {
        match name {
            "large.jpg" => (3, 2),
            _ => (0, 0),
        }
    }

    #[test]
    fn resized_crops_match_official_preprocessor() {
        for name in ["small.png", "wide.png", "large.jpg"] {
            let image = open(&fixtures_dir().join("images").join(name)).unwrap();
            let config = PreprocessConfig::default();
            let preprocessed = preprocess(&image, &config).unwrap();
            let (crops, sizes, _) = load_fixture(name);
            assert_eq!(preprocessed.target_sizes, sizes, "{name} target sizes");
            assert_eq!(preprocessed.slice_grid, grid_of(name), "{name} grid");
            assert_eq!(preprocessed.crops.len(), crops.len(), "{name} crops");
            for (index, (actual, expected)) in
                preprocessed.crops.iter().zip(crops.iter()).enumerate()
            {
                let grid = preprocessed.target_sizes[index];
                // Compare against the official crop through the same packing.
                let expected_packed = pack_official(expected, grid, &config);
                let mut max_diff = 0f32;
                let mut worst = usize::MAX;
                for (i, (a, b)) in actual.iter().zip(&expected_packed).enumerate() {
                    let d = (a - b).abs();
                    if d > max_diff {
                        max_diff = d;
                        worst = i;
                    }
                }
                // PNG sources match Pillow bit-exactly; JPEG decoders (libjpeg
                // vs this crate's) differ by up to 3/255 on well under 1% of
                // pixels, which the resize can amplify to at most ~4/255.
                let tolerance = if name.ends_with(".jpg") {
                    4.0 / 127.5
                } else {
                    1e-4
                };
                assert!(
                    max_diff < tolerance,
                    "{name} crop {index}: pixels diverge by {max_diff} at flat {worst}"
                );
            }
        }
    }

    fn pack_official(crop: &[u8], grid: Grid, config: &PreprocessConfig) -> Vec<f32> {
        let patch = config.patch_size;
        let crop_image = image::RgbImage::from_raw(
            (grid.width * patch) as u32,
            (grid.height * patch) as u32,
            crop.to_vec(),
        )
        .unwrap();
        pack_patches(
            &Crop {
                pixels: crop_image.as_raw().clone(),
                width: grid.width * patch,
                height: grid.height * patch,
            },
            config,
        )
    }

    #[test]
    fn exif_orientation_and_white_matte_are_applied() {
        let rotated = open(&fixtures_dir().join("images").join("rotated.jpg")).unwrap();
        // Orientation 6 (rotate 90 CW): stored 320x240 displays as 240x320.
        assert_eq!((rotated.width(), rotated.height()), (240, 320));
        let alpha = open(&fixtures_dir().join("images").join("alpha.webp")).unwrap();
        let matted = white_matte(&alpha);
        // Fully transparent pixel composites to pure white.
        let pixel = matted.get_pixel(88, 88);
        assert!(pixel.0.iter().all(|&c| c > 250), "matte {pixel:?}");
    }

    #[test]
    fn slice_grids_follow_official_algorithm() {
        let config = PreprocessConfig::default();
        assert_eq!(sliced_grid(64, 96, &config), None);
        assert_eq!(sliced_grid(200, 600, &config), None);
        assert_eq!(sliced_grid(1400, 900, &config), Some((3, 2)));
        let (h, w) = find_best_resize(240, 320, &config, true);
        assert_eq!((h, w), (392, 504));
    }
}
