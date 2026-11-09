use omni_lfm3::media::{MediaSpec, MediaTokens, expand};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    prompt: String,
    media: Vec<MediaSpec>,
    expected: String,
}
#[test]
fn image_audio_and_timed_video_placeholders_match_official_processor() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/media_expansions.json")).unwrap();
    for case in cases {
        let result = expand(&case.prompt, &case.media, &MediaTokens::default()).unwrap();
        assert_eq!(result.text, case.expected, "{}", case.name);
    }
}

#[test]
fn media_order_and_placeholder_counts_are_not_silently_changed() {
    let tokens = MediaTokens::default();
    let prompt = "<|vision_start|><|image|><|vision_end|><|vision_start|><|audio|><|vision_end|>";
    assert!(
        expand(
            prompt,
            &[
                MediaSpec::Audio { tokens: 1 },
                MediaSpec::Image { tokens: 1 }
            ],
            &tokens
        )
        .is_err()
    );
    assert!(expand(prompt, &[MediaSpec::Image { tokens: 1 }], &tokens).is_err());
    assert!(expand("plain text", &[MediaSpec::Image { tokens: 1 }], &tokens).is_err());
}
