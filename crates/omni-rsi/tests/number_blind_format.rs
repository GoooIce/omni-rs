use omni_rsi::number_blind_holdout::{
    BlindCase, BlindManifest, PublicMetadata, seal_for_custodian,
};

fn manifest() -> BlindManifest {
    BlindManifest {
        cases: vec![BlindCase {
            case_id: "holdout-opaque".into(),
            spoken_number: 37,
            answer: 38,
            audio_file: "private-number-audio.wav".into(),
            audio_sha256: "a".repeat(64),
            recording_sha256: "b".repeat(64),
            family_sha256: "c".repeat(64),
            speaker_sha256: "d".repeat(64),
            session_sha256: "e".repeat(64),
            private_source: serde_json::Map::new(),
        }],
    }
}

#[test]
fn public_metadata_exposes_opaque_speaker_and_session_groups() {
    let seal = seal_for_custodian(&manifest()).unwrap();
    let metadata = PublicMetadata::from_json(&seal.public().to_json().unwrap()).unwrap();
    assert_eq!(metadata.sources()[0].speaker_sha256, "d".repeat(64));
    assert_eq!(metadata.sources()[0].session_sha256, "e".repeat(64));
}

#[test]
fn public_commitment_exposes_no_answer_number_or_audio_path() {
    let sealed = seal_for_custodian(&manifest()).unwrap();
    let public = sealed.public().to_json().unwrap();
    let text = std::str::from_utf8(&public).unwrap();
    assert!(!text.contains("private-number-audio.wav"));
    assert!(!text.contains("spoken_number"));
    assert!(!text.contains("answer"));
    let value: serde_json::Value = serde_json::from_slice(&public).unwrap();
    assert_eq!(value.as_object().unwrap().len(), 3);
}

#[cfg(unix)]
#[test]
fn custodian_seals_use_distinct_secret_salts_and_private_files() {
    use std::os::unix::fs::PermissionsExt;
    let first = seal_for_custodian(&manifest()).unwrap();
    let second = seal_for_custodian(&manifest()).unwrap();
    assert_ne!(
        first.public().to_json().unwrap(),
        second.public().to_json().unwrap()
    );
    let root = tempfile::tempdir().unwrap();
    let first_dir = root.path().join("first");
    let second_dir = root.path().join("second");
    first.persist_private(&first_dir).unwrap();
    second.persist_private(&second_dir).unwrap();
    let first_salt = std::fs::read(first_dir.join("salt.bin")).unwrap();
    let second_salt = std::fs::read(second_dir.join("salt.bin")).unwrap();
    assert_eq!(first_salt.len(), 32);
    assert_ne!(first_salt, second_salt);
    assert_eq!(
        std::fs::metadata(first_dir.join("salt.bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(first.persist_private(&first_dir).is_err());
}

#[test]
fn public_reader_rejects_answer_and_media_fields() {
    let seal = seal_for_custodian(&manifest()).unwrap();
    let bytes = seal.public().to_json().unwrap();
    PublicMetadata::from_json(&bytes).unwrap();
    let mut public: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    public["answer"] = 38.into();
    assert!(PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).is_err());
    public.as_object_mut().unwrap().remove("answer");
    public["audio_file"] = "secret.wav".into();
    assert!(PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).is_err());
}

#[test]
fn source_evidence_is_committed_privately_and_never_published() {
    let mut input = manifest();
    input.cases[0].private_source = serde_json::json!({
        "authorization": {"kind":"user-provided", "consent":"test-only consent"},
        "human_reference": {"labeler":"private-reviewer", "spoken_number":37}
    })
    .as_object()
    .unwrap()
    .clone();
    let seal = seal_for_custodian(&input).unwrap();
    let public = String::from_utf8(seal.public().to_json().unwrap()).unwrap();
    assert!(!public.contains("test-only consent"));
    assert!(!public.contains("private-reviewer"));
    let root = tempfile::tempdir().unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(private.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        saved["cases"][0]["private_source"]["human_reference"]["labeler"],
        "private-reviewer"
    );
}

#[test]
fn custodian_rejects_audio_paths_outside_the_private_directory() {
    let mut invalid = manifest();
    invalid.cases[0].audio_file = "../outside.wav".into();
    assert!(seal_for_custodian(&invalid).is_err());
}

#[test]
fn custodian_rejects_conflicting_families_for_the_same_original() {
    let mut invalid = manifest();
    let mut other = invalid.cases[0].clone();
    other.case_id = "second-case".into();
    other.audio_file = "different.wav".into();
    other.family_sha256 = "d".repeat(64);
    invalid.cases.push(other);
    assert!(seal_for_custodian(&invalid).is_err());
}

#[test]
fn custodian_rejects_forged_truth_missing_cases_and_invalid_group_ids() {
    let mut invalid = manifest();
    invalid.cases[0].answer = 39;
    assert!(seal_for_custodian(&invalid).is_err());
    invalid = manifest();
    invalid.cases[0].speaker_sha256 = "speaker-name".into();
    assert!(seal_for_custodian(&invalid).is_err());
    invalid.cases.clear();
    assert!(seal_for_custodian(&invalid).is_err());
}

#[test]
fn custodian_rejects_a_manifest_larger_than_the_reader_can_accept() {
    let template = manifest().cases.remove(0);
    let cases = (0..10_000)
        .map(|index| {
            let mut case = template.clone();
            case.case_id = format!("{index:04}{}", "x".repeat(123));
            case.audio_file = format!("{index:04}{}.wav", "y".repeat(245));
            case.recording_sha256 = format!("{index:064x}");
            case
        })
        .collect();
    assert!(seal_for_custodian(&BlindManifest { cases }).is_err());
}

#[test]
fn changing_a_source_group_changes_the_public_protocol_identity() {
    let seal = seal_for_custodian(&manifest()).unwrap();
    let mut public: serde_json::Value =
        serde_json::from_slice(&seal.public().to_json().unwrap()).unwrap();
    public["sources"][0]["family_sha256"] = "e".repeat(64).into();
    let changed = PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).unwrap();
    assert_ne!(seal.public().fingerprint(), changed.fingerprint());
}
