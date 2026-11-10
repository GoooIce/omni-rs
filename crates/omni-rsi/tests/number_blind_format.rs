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
fn private_and_public_source_identities_share_validation_and_flat_json() {
    let input = manifest();
    let private_json = serde_json::to_value(&input).unwrap();
    let case = private_json["cases"][0].as_object().unwrap();
    let fields = [
        ("recording_sha256", 'b'),
        ("family_sha256", 'c'),
        ("speaker_sha256", 'd'),
        ("session_sha256", 'e'),
    ];
    for (field, digit) in fields {
        assert_eq!(case[field], digit.to_string().repeat(64));
    }
    assert!(!case.contains_key("source"));
    let roundtrip: BlindManifest = serde_json::from_value(private_json.clone()).unwrap();
    let seal = seal_for_custodian(&roundtrip).unwrap();
    let public_json: serde_json::Value =
        serde_json::from_slice(&seal.public().to_json().unwrap()).unwrap();
    for (field, _) in fields {
        assert_eq!(
            public_json["sources"][0][field],
            private_json["cases"][0][field]
        );
        for malformed in ["", "A", &"g".repeat(64), &"a".repeat(63), &"A".repeat(64)] {
            let mut bad_private = private_json.clone();
            bad_private["cases"][0][field] = malformed.into();
            let bad_manifest: BlindManifest = serde_json::from_value(bad_private).unwrap();
            assert!(
                seal_for_custodian(&bad_manifest).is_err(),
                "private {field}: {malformed}"
            );
            let mut bad_public = public_json.clone();
            bad_public["sources"][0][field] = malformed.into();
            assert!(
                PublicMetadata::from_json(&serde_json::to_vec(&bad_public).unwrap()).is_err(),
                "public {field}: {malformed}"
            );
        }
    }
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
fn public_reader_accepts_both_versions_and_rejects_extra_fields() {
    let seal = seal_for_custodian(&manifest()).unwrap();
    let legacy = seal.public().to_json().unwrap();
    let legacy_metadata = PublicMetadata::from_json(&legacy).unwrap();
    assert_eq!(legacy_metadata.to_json().unwrap(), legacy);
    let mut public: serde_json::Value = serde_json::from_slice(&legacy).unwrap();
    public["version"] = 2.into();
    let metadata = PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&metadata.to_json().unwrap()).unwrap(),
        public
    );
    for version in [1, 2] {
        public["version"] = version.into();
        public["audio_total_bytes"] = 1.into();
        assert!(PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).is_err());
        public.as_object_mut().unwrap().remove("audio_total_bytes");
    }
    public["version"] = 3.into();
    assert!(PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).is_err());
}

#[test]
fn public_capacity_commitment_does_not_disclose_media_sizes() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("private-number-audio.wav"), b"audio").unwrap();
    let seal = omni_rsi::number_blind_holdout::seal_for_custodian_with_media_metadata(
        &manifest(),
        root.path(),
    )
    .unwrap();
    let public: serde_json::Value =
        serde_json::from_slice(&seal.public().to_json().unwrap()).unwrap();
    assert_eq!(public["version"], 2);
    assert_eq!(public.as_object().unwrap().len(), 3);
    assert!(public.get("audio_total_bytes").is_none());
    assert!(public.get("media_sizes").is_none());
}

#[test]
fn custodian_declares_capacity_before_sealing_without_reading_audio() {
    use omni_rsi::number_blind_holdout::seal_for_custodian_with_media_metadata;
    let root = tempfile::tempdir().unwrap();
    let mut input = manifest();
    let template = input.cases[0].clone();
    input.cases = (0..17)
        .map(|index| {
            let mut case = template.clone();
            case.case_id = format!("case-{index}");
            case.audio_file = format!("missing-{index}.wav");
            case
        })
        .collect();
    let mib = 1024_u64 * 1024;
    let mut sizes = vec![32 * mib; 15];
    sizes.extend([32 * mib - 1, 1]);
    for (case, size) in input.cases.iter().zip(&sizes) {
        std::fs::File::create(root.path().join(&case.audio_file))
            .unwrap()
            .set_len(*size)
            .unwrap();
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            root.path().join("missing-0.wav"),
            std::fs::Permissions::from_mode(0o000),
        )
        .unwrap();
    }
    let seal = seal_for_custodian_with_media_metadata(&input, root.path()).unwrap();
    let public = PublicMetadata::from_json(&seal.public().to_json().unwrap()).unwrap();
    let public: serde_json::Value = serde_json::from_slice(&public.to_json().unwrap()).unwrap();
    assert_eq!(public["version"], 2);
    assert_eq!(public.as_object().unwrap().len(), 3);
    std::fs::OpenOptions::new()
        .write(true)
        .open(root.path().join("missing-16.wav"))
        .unwrap()
        .set_len(2)
        .unwrap();
    let error = seal_for_custodian_with_media_metadata(&input, root.path())
        .err()
        .expect("sealed an oversized holdout");
    assert!(error.contains("capacity"), "{error}");
    std::fs::OpenOptions::new()
        .write(true)
        .open(root.path().join("missing-16.wav"))
        .unwrap()
        .set_len(32 * mib + 1)
        .unwrap();
    let error = seal_for_custodian_with_media_metadata(&input, root.path())
        .err()
        .expect("sealed a file over the individual limit");
    assert!(error.contains("size"), "{error}");
    std::fs::OpenOptions::new()
        .write(true)
        .open(root.path().join("missing-16.wav"))
        .unwrap()
        .set_len(0)
        .unwrap();
    assert!(seal_for_custodian_with_media_metadata(&input, root.path()).is_err());
    std::fs::remove_file(root.path().join("missing-16.wav")).unwrap();
    assert!(seal_for_custodian_with_media_metadata(&input, root.path()).is_err());
}

fn manifest_with_source_evidence() -> BlindManifest {
    let mut input = manifest();
    input.cases[0].private_source = serde_json::json!({
        "authorization": {"kind":"user-provided", "consent":"test-only consent"},
        "human_reference": {"labeler":"private-reviewer", "spoken_number":37}
    })
    .as_object()
    .unwrap()
    .clone();
    input
}

#[test]
fn source_evidence_is_never_published() {
    let seal = seal_for_custodian(&manifest_with_source_evidence()).unwrap();
    let public = String::from_utf8(seal.public().to_json().unwrap()).unwrap();
    assert!(!public.contains("test-only consent"));
    assert!(!public.contains("private-reviewer"));
}

#[cfg(unix)]
#[test]
fn source_evidence_is_persisted_in_the_private_manifest() {
    let seal = seal_for_custodian(&manifest_with_source_evidence()).unwrap();
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

#[cfg(not(unix))]
#[test]
fn private_persistence_is_rejected_on_unsupported_platforms() {
    let seal = seal_for_custodian(&manifest_with_source_evidence()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let private = root.path().join("private");
    let error = seal.persist_private(&private).unwrap_err();
    assert!(error.contains("not supported"), "{error}");
    assert!(!private.exists());
    let media = root.path().join("media");
    std::fs::create_dir(&media).unwrap();
    std::fs::write(media.join("private-number-audio.wav"), b"audio").unwrap();
    let seal = omni_rsi::number_blind_holdout::seal_for_custodian_with_media_metadata(
        &manifest_with_source_evidence(),
        &media,
    )
    .unwrap();
    let error = seal.persist_private(&private).unwrap_err();
    assert!(error.contains("not supported"), "{error}");
    assert!(!private.exists());
}

#[cfg(unix)]
#[test]
fn custodian_rejects_symlink_or_non_file_media_metadata() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let audio = root.path().join("private-number-audio.wav");
    std::fs::write(root.path().join("actual.wav"), b"audio").unwrap();
    symlink("actual.wav", &audio).unwrap();
    assert!(
        omni_rsi::number_blind_holdout::seal_for_custodian_with_media_metadata(
            &manifest(),
            root.path()
        )
        .is_err()
    );
    std::fs::remove_file(&audio).unwrap();
    std::fs::create_dir(&audio).unwrap();
    assert!(
        omni_rsi::number_blind_holdout::seal_for_custodian_with_media_metadata(
            &manifest(),
            root.path()
        )
        .is_err()
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
