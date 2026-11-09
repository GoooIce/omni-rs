use omni_lfm3::image::{ImageConfig, preprocess_rgb};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
    grid: [usize; 3],
    pixels: Vec<f32>,
}
#[test]
fn image_resize_normalization_and_patch_order_match_official_processor() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/image_preprocessing.json")).unwrap();
    for case in cases {
        let mut rgb = vec![0u8; case.height * case.width * 3];
        for y in 0..case.height {
            for x in 0..case.width {
                for c in 0..3 {
                    rgb[(y * case.width + x) * 3 + c] =
                        ((x * 17 + y * 23 + c * 53 + (x * y) % 17) % 256) as u8;
                }
            }
        }
        let config = ImageConfig {
            min_pixels: case.min_pixels,
            max_pixels: case.max_pixels,
            ..Default::default()
        };
        let result = preprocess_rgb(&rgb, case.width, case.height, &config).unwrap();
        assert_eq!(
            [1, result.grid_h, result.grid_w],
            case.grid,
            "{}",
            case.name
        );
        assert_eq!(result.patches.len(), case.pixels.len());
        let max = result
            .patches
            .iter()
            .zip(&case.pixels)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max < 0.00001,
            "{}: max image preprocessing error {max}",
            case.name
        );
    }
}
