use omni_rsi::speech_asr::{
    AsrEvent, assess, final_event, fingerprints, matching_ratio, simplified,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    conversion: Vec<Conversion>,
    matching: Vec<Matching>,
}
#[derive(Deserialize)]
struct Conversion {
    input: String,
    output: String,
}
#[derive(Deserialize)]
struct Matching {
    reference: String,
    observed: String,
    ratio: f64,
}

#[test]
fn native_simplification_and_matching_agree_with_previous_verifier() {
    let reference: Reference =
        serde_json::from_str(include_str!("fixtures/speech-asr-reference.json")).unwrap();
    for case in reference.conversion {
        assert_eq!(
            simplified(&case.input),
            case.output,
            "input: {}",
            case.input
        );
    }
    for case in reference.matching {
        let actual = matching_ratio(
            &case.reference.chars().collect::<Vec<_>>(),
            &case.observed.chars().collect::<Vec<_>>(),
        );
        assert!(
            (actual - case.ratio).abs() < 1e-12,
            "{} / {}",
            case.reference,
            case.observed
        );
    }
}

#[test]
fn final_event_and_chinese_quality_are_checked_without_trusting_language_labels() {
    let final_line = r#"{"text":"你好，我是朋友","language":"English","is_final":true,"cancelled":false,"token_limit_reached":false}"#;
    let event = final_event(final_line).unwrap();
    assert!(assess("你好我是朋友", &event).passed);
    let event = AsrEvent {
        text: "你好 hello hello".into(),
        language: "Chinese".into(),
    };
    assert!(!assess("你好我是朋友", &event).passed);
    assert!(final_event(&format!("{final_line}\n{final_line}")).is_err());
    assert!(final_event(&format!("42\n{final_line}")).is_err());
    assert!(
        final_event(
            r#"{"text":"你好","is_final":true,"cancelled":true,"token_limit_reached":false}"#
        )
        .is_err()
    );
}

#[test]
fn earlier_token_limit_makes_a_later_final_event_incomplete_evidence() {
    let partial = r#"{"text":"","language":"English","is_final":false,"cancelled":false,"token_limit_reached":true}"#;
    let complete_partial = partial.replace(
        "\"token_limit_reached\":true",
        "\"token_limit_reached\":false",
    );
    let final_line = r#"{"text":"I want chengniu yao","language":"English","is_final":true,"cancelled":false,"token_limit_reached":false}"#;
    assert!(final_event(&format!("{complete_partial}\n{final_line}")).is_ok());
    assert!(final_event(&format!("{partial}\n{final_line}")).is_err());
    let without_status = r#"{"text":"","is_final":false,"cancelled":false}"#;
    assert!(final_event(&format!("{without_status}\n{final_line}")).is_err());
    let captured = concat!(
        r#"{"delta":"","text":"","language":"English","audio_seconds":2.16,"compute_ms":435.56925,"generated_tokens":4,"reused_audio_tokens":0,"is_final":false,"token_limit_reached":true,"total_compute_ms":435.56925,"event_count":1,"rtf":0.20165243055555557,"cancelled":false}"#,
        "\n",
        r#"{"delta":"I want chengniu yao","text":"I want chengniu yao","language":"English","audio_seconds":3.44,"compute_ms":382.709083,"generated_tokens":8,"reused_audio_tokens":0,"is_final":true,"token_limit_reached":false,"total_compute_ms":818.278333,"event_count":2,"rtf":0.23787160843023256,"cancelled":false}"#
    );
    assert!(final_event(captured).is_err());
}

#[test]
fn fingerprints_bind_native_and_historical_evidence_to_the_model_and_binary() {
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("model");
    std::fs::create_dir(&model).unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ] {
        std::fs::write(model.join(name), name).unwrap();
    }
    let binary = directory.path().join("r2t2");
    std::fs::write(&binary, "v1").unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/asr-fingerprint-reference.json")).unwrap();
    let first = fingerprints(&model, &binary).unwrap();
    assert_eq!(first.legacy, expected["legacy"]);
    assert_ne!(first.legacy, first.native);
    std::fs::write(&binary, "v2").unwrap();
    let second = fingerprints(&model, &binary).unwrap();
    assert_ne!(first.legacy, second.legacy);
    assert_ne!(first.native, second.native);
}

#[cfg(unix)]
#[test]
fn native_r2t2_process_is_checked_for_a_valid_final_event_and_panics() {
    use std::{os::unix::fs::PermissionsExt, sync::atomic::AtomicBool};
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("model");
    std::fs::create_dir(&model).unwrap();
    let audio = directory.path().join("audio.wav");
    std::fs::write(&audio, b"fixture").unwrap();
    let binary = directory.path().join("r2t2");
    std::fs::write(&binary, "#!/bin/sh\nprintf '%s\\n' '{\"text\":\"你好\",\"is_final\":true,\"cancelled\":false,\"token_limit_reached\":false}'\n").unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let event =
        omni_rsi::speech_asr::transcribe(&audio, &model, &binary, &AtomicBool::new(false)).unwrap();
    assert_eq!(event.text, "你好");
    std::fs::write(
        &binary,
        "#!/bin/sh\necho 'panicked at example' >&2\nexit 0\n",
    )
    .unwrap();
    assert!(
        omni_rsi::speech_asr::transcribe(&audio, &model, &binary, &AtomicBool::new(false)).is_err()
    );
}
