use omni_lfm3::video::{RgbFrame, VideoConfig, preprocess_video, sample_indices};
use serde::Deserialize;
#[derive(Deserialize)]
struct Reference {
    height: usize,
    width: usize,
    count: usize,
    min_pixels: usize,
    max_pixels: usize,
    grid: [usize; 3],
    patches: Vec<f32>,
}
#[test]
fn video_frame_pairing_and_patch_order_match_official_processor() {
    let r: Reference =
        serde_json::from_str(include_str!("fixtures/video_preprocessing.json")).unwrap();
    let mut frames = Vec::new();
    for t in 0..r.count {
        let mut rgb = vec![0; r.width * r.height * 3];
        for y in 0..r.height {
            for x in 0..r.width {
                for c in 0..3 {
                    rgb[(y * r.width + x) * 3 + c] =
                        ((x * 17 + y * 23 + c * 53 + (x * y) % 17 + t * 37) % 256) as u8;
                }
            }
        }
        frames.push(RgbFrame {
            width: r.width,
            height: r.height,
            rgb,
        });
    }
    let config = VideoConfig {
        min_pixels: r.min_pixels,
        max_pixels: r.max_pixels,
        ..Default::default()
    };
    let output = preprocess_video(&frames, &config).unwrap();
    assert_eq!([output.groups.len(), output.grid_h, output.grid_w], r.grid);
    let actual: Vec<_> = output.groups.into_iter().flatten().collect();
    assert_eq!(actual.len(), r.patches.len());
    let max = actual
        .iter()
        .zip(&r.patches)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max < 0.00001, "video preprocessing error {max}");
}
#[test]
fn video_sampling_carries_source_frame_indices() {
    assert_eq!(
        sample_indices(10, 10.0, &VideoConfig::default()).unwrap(),
        vec![0, 3, 6, 9]
    );
    assert!(sample_indices(10, 0.0, &VideoConfig::default()).is_err());
}
