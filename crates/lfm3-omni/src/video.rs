//! Sampling and temporal patch packing for already-decoded RGB video frames.
use omni_r2t2::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct VideoConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_frames: usize,
    pub max_frames: usize,
    pub fps: f64,
    pub rescale_factor: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}
impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            min_pixels: 4096,
            max_pixels: 25_165_824,
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_frames: 4,
            max_frames: 384,
            fps: 2.0,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
}
#[derive(Deserialize)]
struct FileConfig {
    size: Size,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    min_frames: usize,
    max_frames: usize,
    fps: f64,
    rescale_factor: f32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
    do_convert_rgb: bool,
    resample: usize,
    cap_pixels_per_frame: bool,
}
#[derive(Deserialize)]
struct Size {
    shortest_edge: usize,
    longest_edge: usize,
}
impl VideoConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let raw: FileConfig = serde_json::from_slice(&std::fs::read(
            directory.join("video_preprocessor_config.json"),
        )?)?;
        if !(raw.do_resize && raw.do_rescale && raw.do_normalize && raw.do_convert_rgb)
            || raw.resample != 3
            || raw.cap_pixels_per_frame
        {
            return Err(Error::Config(
                "unsupported video processor operations".into(),
            ));
        }
        let c = Self {
            min_pixels: raw.size.shortest_edge,
            max_pixels: raw.size.longest_edge,
            patch_size: raw.patch_size,
            merge_size: raw.merge_size,
            temporal_patch_size: raw.temporal_patch_size,
            min_frames: raw.min_frames,
            max_frames: raw.max_frames,
            fps: raw.fps,
            rescale_factor: raw.rescale_factor,
            mean: raw.image_mean,
            std: raw.image_std,
        };
        c.validate()?;
        Ok(c)
    }
    fn validate(&self) -> Result<()> {
        if self.min_pixels == 0
            || self.max_pixels < self.min_pixels
            || self.patch_size == 0
            || self.merge_size == 0
            || self.temporal_patch_size == 0
            || self.min_frames == 0
            || self.max_frames < self.min_frames
            || !self.fps.is_finite()
            || self.fps <= 0.0
            || !self.rescale_factor.is_finite()
            || self.mean.iter().any(|v| !v.is_finite())
            || self.std.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.patch_size.checked_mul(self.merge_size).is_none()
        {
            return Err(Error::Config(
                "invalid video processor configuration".into(),
            ));
        }
        Ok(())
    }
}
/// One decoded RGB frame, in row-major interleaved channel order.
pub struct RgbFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}
/// One patch matrix per temporal frame group, each in merge-block order.
pub struct VideoPatches {
    pub groups: Vec<Vec<f32>>,
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Match the processor's evenly spaced samples and retain source frame indices.
pub fn sample_indices(total: usize, source_fps: f64, config: &VideoConfig) -> Result<Vec<usize>> {
    config.validate()?;
    if total == 0 || !source_fps.is_finite() || source_fps <= 0.0 {
        return Err(Error::Inference("invalid video frame count or rate".into()));
    }
    let count = ((total as f64 / source_fps * config.fps) as usize)
        .max(config.min_frames)
        .min(config.max_frames)
        .min(total);
    if count == 1 {
        return Ok(vec![0]);
    }
    Ok((0..count)
        .map(|i| (i as f64 * (total - 1) as f64 / (count - 1) as f64).round_ties_even() as usize)
        .collect())
}

pub fn preprocess_video(frames: &[RgbFrame], config: &VideoConfig) -> Result<VideoPatches> {
    config.validate()?;
    if frames.len() < config.temporal_patch_size || frames.len() > config.max_frames {
        return Err(Error::Inference("invalid video frame count".into()));
    }
    let (source_w, source_h) = (frames[0].width, frames[0].height);
    let bytes = source_w
        .checked_mul(source_h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Inference("video frame size overflow".into()))?;
    if source_w == 0
        || source_h == 0
        || frames
            .iter()
            .any(|f| f.width != source_w || f.height != source_h || f.rgb.len() != bytes)
    {
        return Err(Error::Inference(
            "video frames must have the same nonempty RGB shape".into(),
        ));
    }
    let factor = config.patch_size * config.merge_size;
    let mut width = source_w;
    let mut height = source_h;
    if height < factor || width < factor {
        let scale = (factor as f64 / height as f64).max(factor as f64 / width as f64);
        height = (height as f64 * scale) as usize;
        width = (width as f64 * scale) as usize;
    }
    if width.max(height) as f64 / width.min(height) as f64 > 200.0 {
        return Err(Error::Inference("video aspect ratio exceeds 200".into()));
    }
    let mut h = (height as f64 / factor as f64).round_ties_even() as usize * factor;
    let mut w = (width as f64 / factor as f64).round_ties_even() as usize * factor;
    let t = (frames.len() as f64 / config.temporal_patch_size as f64).round_ties_even() as usize
        * config.temporal_patch_size;
    let area = t
        .checked_mul(h)
        .and_then(|n| n.checked_mul(w))
        .ok_or_else(|| Error::Inference("video size overflow".into()))?;
    let original = frames
        .len()
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| Error::Inference("video size overflow".into()))?;
    if area > config.max_pixels {
        let beta = (original as f64 / config.max_pixels as f64).sqrt();
        h = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if area < config.min_pixels {
        let beta = (config.min_pixels as f64 / original as f64).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    if h == 0 || w == 0 {
        return Err(Error::Inference("invalid video output dimensions".into()));
    }
    let resized: Vec<_> = frames
        .iter()
        .map(|f| crate::image::resize_bicubic(&f.rgb, source_w, source_h, w, h))
        .collect();
    let (grid_h, grid_w) = (h / config.patch_size, w / config.patch_size);
    let patch_size = config.patch_size;
    let mut groups = Vec::new();
    for group in 0..frames.len().div_ceil(config.temporal_patch_size) {
        let mut patches = Vec::new();
        for by in 0..grid_h / config.merge_size {
            for bx in 0..grid_w / config.merge_size {
                for dy in 0..config.merge_size {
                    for dx in 0..config.merge_size {
                        let patch_y = by * config.merge_size + dy;
                        let patch_x = bx * config.merge_size + dx;
                        for channel in 0..3 {
                            for t in 0..config.temporal_patch_size {
                                let frame =
                                    (group * config.temporal_patch_size + t).min(frames.len() - 1);
                                for py in 0..patch_size {
                                    for px in 0..patch_size {
                                        let y = patch_y * patch_size + py;
                                        let x = patch_x * patch_size + px;
                                        let value = resized[frame][(y * w + x) * 3 + channel]
                                            as f32
                                            * config.rescale_factor;
                                        patches.push(
                                            (value - config.mean[channel]) / config.std[channel],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        groups.push(patches);
    }
    Ok(VideoPatches {
        groups,
        grid_h,
        grid_w,
    })
}
