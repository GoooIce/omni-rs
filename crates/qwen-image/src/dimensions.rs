//! Qwen-Image-2.1-Turbo's published presets and Diffusers output alignment.
//! https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo
use crate::{Error, Result};

pub const DEFAULT_SIDE: usize = 2048;
pub const ALIGNMENT: usize = 32;
pub const MIN_SIDE: usize = 256;
// Local resource limit, chosen to include the longest official preset edge.
// The checkpoint itself does not specify a hard maximum image dimension.
pub const MAX_SIDE: usize = 2752;
pub const MAX_PIXELS: usize = 2400 * 1792;
pub const OFFICIAL_PRESETS: [(&str, usize, usize); 7] = [
    ("1:1", 2048, 2048),
    ("4:3", 2400, 1792),
    ("3:4", 1792, 2400),
    ("3:2", 2528, 1696),
    ("2:3", 1696, 2528),
    ("16:9", 2752, 1536),
    ("9:16", 1536, 2752),
];

pub fn preset(ratio: &str) -> Option<(usize, usize)> {
    OFFICIAL_PRESETS
        .iter()
        .find(|(r, _, _)| *r == ratio)
        .map(|(_, width, height)| (*width, *height))
}

/// Match QwenImage21Pipeline: round each dimension down to a multiple of 32.
/// Callers display the resolved dimensions before starting inference.
pub fn resolve(width: usize, height: usize) -> Result<(usize, usize)> {
    if [width, height]
        .iter()
        .any(|side| !(MIN_SIDE..=MAX_SIDE).contains(side))
    {
        return Err(Error::Config(format!(
            "宽和高须在 {MIN_SIDE}–{MAX_SIDE} 之间；将按官方规则向下对齐到 {ALIGNMENT} 的倍数"
        )));
    }
    let (width, height) = (
        width / ALIGNMENT * ALIGNMENT,
        height / ALIGNMENT * ALIGNMENT,
    );
    if width * height > MAX_PIXELS {
        return Err(Error::Config(format!(
            "输出总像素须不超过 {MAX_PIXELS}（官方最大预设）；请减小宽高或选择官方比例"
        )));
    }
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_presets_and_quick_start_follow_pipeline_alignment() {
        for (_, width, height) in OFFICIAL_PRESETS {
            assert_eq!(resolve(width, height).unwrap(), (width, height));
        }
        // The official quick-start example is rounded by Diffusers at runtime.
        assert_eq!(resolve(1680, 2512).unwrap(), (1664, 2496));
        assert_eq!(resolve(513, 543).unwrap(), (512, 512));
        assert_eq!(resolve(256, 2752).unwrap(), (256, 2752));
        assert!(resolve(2752, 2752).is_err());
        for side in [0, 255, 2753, usize::MAX] {
            assert!(resolve(side, 512).is_err());
            assert!(resolve(512, side).is_err());
        }
    }
}
