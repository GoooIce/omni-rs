//! Reference image preparation: Pillow-compatible RGBA Lanczos, white-matted VL
//! patches, and four-channel VAE input. The application never invokes Python.
use crate::{Error, Result, check_cancel};
use image::{ImageDecoder, RgbaImage};
use std::{path::Path, sync::atomic::AtomicBool};

pub const MAX_REFERENCES: usize = 4;

/// Bound decoding before allocating full-resolution pixels; honor EXIF orientation.
pub fn open(path: &Path) -> Result<RgbaImage> {
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.to_rgba8())
}

pub fn dimensions(width: u32, height: u32, resolution: usize) -> Result<(usize, usize)> {
    if width == 0 || height == 0 || !(256..=1024).contains(&resolution) || resolution % 32 != 0 {
        return Err(Error::Config(
            "参考图分辨率须为 256–1024 内 32 的倍数".into(),
        ));
    }
    let ratio = width as f64 / height as f64;
    let w = (resolution as f64 * ratio.sqrt() / 32.0).round_ties_even() as usize * 32;
    let h = (resolution as f64 / ratio.sqrt() / 32.0).round_ties_even() as usize * 32;
    // VL smart_resize must leave the VAE grid unchanged (4 latent tokens per VL slot).
    if w < 32 || h < 32 || w.max(h) > 4096 {
        return Err(Error::Config(
            "参考图过于狭长；请先裁剪图片或提高参考图分辨率".into(),
        ));
    }
    if w * h < 65536 {
        return Err(Error::Config(
            "缩放后的参考图不足 256² 像素；请提高参考图分辨率或输出尺寸".into(),
        ));
    }
    Ok((w, h))
}

pub struct Prepared {
    pub rgba: RgbaImage,
    pub patches: Vec<f32>,
    pub vae_pixels: Vec<f32>,
    pub grid_height: usize,
    pub grid_width: usize,
}
pub fn prepare(image: &RgbaImage, resolution: usize, cancel: &AtomicBool) -> Result<Prepared> {
    check_cancel(cancel)?;
    let (w, h) = dimensions(image.width(), image.height(), resolution)?;
    let rgba = resize(image, w, h, cancel)?;
    let mut rgb = Vec::with_capacity(w * h * 3);
    let mut vae_pixels = vec![0.0; w * h * 4];
    for (i, pixel) in rgba.pixels().enumerate() {
        for c in 0..4 {
            vae_pixels[c * w * h + i] = pixel[c] as f32 / 255.0 * 2.0 - 1.0;
        }
        let a = pixel[3] as u32;
        for c in 0..3 {
            // PIL paste(mask=alpha): integer round-to-nearest over a white background.
            let value = (pixel[c] as u32 * a + 255 * (255 - a) + 127) / 255;
            rgb.push((value as f32 * (1.0 / 255.0) - 0.5) / 0.5);
        }
    }
    let gh = h / 16;
    let gw = w / 16;
    let mut patches = Vec::with_capacity(gh * gw * 1536);
    for gy in 0..gh / 2 {
        for gx in 0..gw / 2 {
            check_cancel(cancel)?;
            for dy in 0..2 {
                for dx in 0..2 {
                    for c in 0..3 {
                        for _ in 0..2 {
                            for y in 0..16 {
                                for x in 0..16 {
                                    let index =
                                        (((gy * 2 + dy) * 16 + y) * w + (gx * 2 + dx) * 16 + x) * 3
                                            + c;
                                    patches.push(rgb[index]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Prepared {
        rgba,
        patches,
        vae_pixels,
        grid_height: gh,
        grid_width: gw,
    })
}

fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter = scale.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let left = ((center - filter * 3.0 + 0.5) as isize).max(0) as usize;
            let right = ((center + filter * 3.0 + 0.5) as usize).min(input);
            let weights: Vec<f64> = (left..right)
                .map(|x| {
                    let v = (x as f64 - center + 0.5) / filter;
                    if v == 0.0 {
                        1.0
                    } else if v.abs() < 3.0 {
                        let t = v * std::f64::consts::PI;
                        (t.sin() / t) * ((t / 3.0).sin() / (t / 3.0))
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
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

/// Match Pillow's separable 8-bit Lanczos, including premultiplied alpha and
/// rounding after each pass. Resizing RGBA as four straight channels causes halos.
pub fn resize(image: &RgbaImage, w: usize, h: usize, cancel: &AtomicBool) -> Result<RgbaImage> {
    let iw = image.width() as usize;
    let ih = image.height() as usize;
    if (iw, ih) == (w, h) {
        return Ok(image.clone());
    }
    let mut data = image.as_raw().clone();
    for pixel in data.chunks_exact_mut(4) {
        let a = pixel[3] as u32;
        for c in 0..3 {
            pixel[c] = ((pixel[c] as u32 * a + 127) / 255) as u8;
        }
    }
    let mut horizontal = if iw == w {
        data.clone()
    } else {
        vec![0u8; w * ih * 4]
    };
    if iw != w {
        let weights = coefficients(iw, w);
        for y in 0..ih {
            check_cancel(cancel)?;
            for (x, (left, coeffs)) in weights.iter().enumerate() {
                for c in 0..4 {
                    let sum = coeffs.iter().enumerate().fold(1 << 21, |acc, (i, k)| {
                        acc + *k * data[(y * iw + left + i) * 4 + c] as i64
                    });
                    horizontal[(y * w + x) * 4 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
    }
    let mut output = if ih == h {
        horizontal.clone()
    } else {
        vec![0u8; w * h * 4]
    };
    if ih != h {
        let weights = coefficients(ih, h);
        for (y, (top, coeffs)) in weights.iter().enumerate() {
            check_cancel(cancel)?;
            for x in 0..w {
                for c in 0..4 {
                    let sum = coeffs.iter().enumerate().fold(1 << 21, |acc, (i, k)| {
                        acc + *k * horizontal[((top + i) * w + x) * 4 + c] as i64
                    });
                    output[(y * w + x) * 4 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
    }
    for pixel in output.chunks_exact_mut(4) {
        let a = pixel[3] as u32;
        if a != 0 && a != 255 {
            for c in 0..3 {
                pixel[c] = (pixel[c] as u32 * 255 / a).min(255) as u8;
            }
        }
    }
    RgbaImage::from_raw(w as u32, h as u32, output)
        .ok_or_else(|| Error::Config("参考图缩放失败".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn jpeg_orientation_and_lossless_webp_match_pillow_decoding() {
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/decoded.json")).unwrap();
        for name in ["rotated.jpg", "alpha.webp"] {
            let image = open(
                &Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures")
                    .join(name),
            )
            .unwrap();
            let values = &expected[name];
            assert_eq!(image.width(), values["width"].as_u64().unwrap() as u32);
            assert_eq!(image.height(), values["height"].as_u64().unwrap() as u32);
            let pixels: Vec<u8> = serde_json::from_value(values["pixels"].clone()).unwrap();
            let tolerance = if name.ends_with("jpg") { 2 } else { 0 };
            assert!(
                image
                    .as_raw()
                    .iter()
                    .zip(pixels)
                    .all(|(a, b)| a.abs_diff(b) <= tolerance)
            );
        }
    }
    #[test]
    fn alpha_and_rounding_match_pillow_in_both_resize_passes() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/rgba-resize.json")).unwrap();
        let pixels: Vec<u8> = serde_json::from_value(fixture["source"].clone()).unwrap();
        let source = RgbaImage::from_raw(5, 3, pixels).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let expected: Vec<u8> = serde_json::from_value(case["pixels"].clone()).unwrap();
            let actual = resize(
                &source,
                case["width"].as_u64().unwrap() as usize,
                case["height"].as_u64().unwrap() as usize,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(actual.as_raw(), &expected);
        }
    }
    #[test]
    fn cancel_before_resize_and_reject_grids_that_diverge_between_encoders() {
        assert!(matches!(
            prepare(&RgbaImage::new(64, 64), 256, &AtomicBool::new(true)),
            Err(Error::Cancelled)
        ));
        assert_eq!(dimensions(1024, 512, 1024).unwrap(), (1440, 736));
        assert!(dimensions(16384, 32, 1024).is_err());
        assert!(dimensions(0, 100, 1024).is_err());
        assert!(dimensions(317, 193, 255).is_err());
    }
}
