#![cfg(unix)]

use omni_rsi::{
    speech_tone::{
        AnnotationDecision, ToneAnnotation, ToneAudio, ToneCase, ToneDataset, ToneEvaluationPlan,
        ToneInput, ToneRecord, ToneSource, ToneSourceKind, ToneSplit,
    },
    speech_tone_input::{ToneInputLimits, prepare_development_inputs},
};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

fn synthetic_record(root: &Path, split: ToneSplit, id: &str, seed: i16) -> ToneRecord {
    let directory = match split {
        ToneSplit::Train => "train",
        ToneSplit::Dev => "dev",
        ToneSplit::Holdout => panic!("the shared contract fixture has no holdout audio"),
    };
    fs::create_dir_all(root.join(directory)).unwrap();
    let path = format!("{directory}/{id}.wav");
    let full_path = root.join(&path);
    let mut wav = hound::WavWriter::create(
        &full_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for index in 0..320 {
        wav.write_sample(seed * 100 + index as i16).unwrap();
    }
    wav.finalize().unwrap();
    let sha256 = format!("{:x}", Sha256::digest(fs::read(&full_path).unwrap()));
    ToneRecord {
        id: id.into(),
        split,
        audio: ToneAudio { path, sha256 },
        text: format!("synthetic text {id}"),
        text_family: format!("text-{id}"),
        speaker_id: format!("speaker-{id}"),
        session_id: format!("session-{id}"),
        derived_family: format!("family-{id}"),
        source: ToneSource {
            name: "synthetic fixture".into(),
            license: "test fixture".into(),
            kind: ToneSourceKind::Synthetic,
        },
        annotations: vec![ToneAnnotation {
            annotator_id: "synthetic-rater".into(),
            source: "test fixture".into(),
            decision: AnnotationDecision::Label("steady".into()),
        }],
        reference: None,
    }
}

fn fixture() -> (tempfile::TempDir, ToneDataset) {
    let root = tempfile::tempdir().unwrap();
    let records = vec![
        synthetic_record(root.path(), ToneSplit::Train, "train-a", 1),
        synthetic_record(root.path(), ToneSplit::Train, "train-b", 2),
        synthetic_record(root.path(), ToneSplit::Dev, "dev-a", 3),
        synthetic_record(root.path(), ToneSplit::Dev, "dev-b", 4),
    ];
    let data = ToneDataset {
        version: 1,
        labels: vec!["steady".into()],
        records,
    };
    (root, data)
}

fn plan(split: ToneSplit) -> ToneEvaluationPlan {
    let record_id = match split {
        ToneSplit::Train => "train-a",
        ToneSplit::Dev => "dev-a",
        ToneSplit::Holdout => panic!("no holdout plan is constructed"),
    };
    ToneEvaluationPlan {
        run_id: "synthetic-contract".into(),
        split,
        cases: vec![ToneCase {
            id: "selected-case".into(),
            record_id: record_id.into(),
            input: ToneInput::Full,
        }],
        pairs: vec![],
    }
}

fn assert_both_accept(data: &ToneDataset, root: &Path) {
    data.validate(root).expect("legacy rejected valid fixture");
    for split in [ToneSplit::Train, ToneSplit::Dev] {
        let plan = plan(split);
        let prepared = prepare_development_inputs(
            data,
            &plan,
            root,
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .expect("development preparer rejected valid fixture");
        assert_eq!(prepared.cases().len(), 1);
        assert_eq!(prepared.cases()[0].record_id(), plan.cases[0].record_id);
        assert!(!prepared.cases()[0].samples().is_empty());
        assert_eq!(prepared.holdout_pcm_independence(), "unverified");
        assert_eq!(prepared.execution_provenance(), "not_run");
    }
}

fn assert_both_reject(data: &ToneDataset, root: &Path, kind: &str) {
    let legacy = data.validate(root).expect_err("legacy accepted conflict");
    assert!(
        legacy.contains(kind) && legacy.contains("cross"),
        "legacy rejected {kind} for an unrelated reason: {legacy}"
    );
    for split in [ToneSplit::Train, ToneSplit::Dev] {
        let error = prepare_development_inputs(
            data,
            &plan(split),
            root,
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .expect_err("development preparer accepted conflict");
        assert!(
            error.contains(kind) && error.contains("cross"),
            "{split:?} rejected {kind} for an unrelated reason: {error}"
        );
    }
}

#[test]
fn normalized_text_conflicts_are_shared_and_same_split_reuse_is_allowed() {
    let (root, mut data) = fixture();
    assert_both_accept(&data, root.path());
    data.records[0].text = "Ｔｅｓｔ， Ａ！".into();
    data.records[1].text = "test a".into();
    data.records[2].text = "你好，世界！".into();
    data.records[3].text = "你 好 世 界".into();
    assert_both_accept(&data, root.path());

    for (training, development) in [
        ("Ｔｅｓｔ， Ａ！", "test a"),
        ("你好，世界！", "你 好 世 界"),
    ] {
        for reverse in [false, true] {
            let (root, mut data) = fixture();
            data.records[0].text = training.into();
            data.records[2].text = development.into();
            if reverse {
                data.records.reverse();
            }
            assert_both_reject(&data, root.path(), "text");
        }
    }
    let (root, mut data) = fixture();
    data.records[0].text = "！？".into();
    assert_both_reject(&data, root.path(), "text");
}

#[derive(Clone, Copy, Debug)]
enum Group {
    Speaker,
    Session,
    DerivedFamily,
    TextFamily,
    AudioHash,
}

impl Group {
    fn name(self) -> &'static str {
        match self {
            Self::Speaker => "speaker",
            Self::Session => "session",
            Self::DerivedFamily => "derived family",
            Self::TextFamily => "text family",
            Self::AudioHash => "audio",
        }
    }

    fn copy(self, data: &mut ToneDataset, root: &Path, source: usize, target: usize) {
        match self {
            Self::Speaker => {
                data.records[target].speaker_id = data.records[source].speaker_id.clone()
            }
            Self::Session => {
                data.records[target].session_id = data.records[source].session_id.clone()
            }
            Self::DerivedFamily => {
                data.records[target].derived_family = data.records[source].derived_family.clone()
            }
            Self::TextFamily => {
                data.records[target].text_family = data.records[source].text_family.clone()
            }
            Self::AudioHash => {
                let bytes = fs::read(root.join(&data.records[source].audio.path)).unwrap();
                fs::write(root.join(&data.records[target].audio.path), bytes).unwrap();
                data.records[target].audio.sha256 = data.records[source].audio.sha256.clone();
            }
        }
    }
}

#[test]
fn public_groups_reject_cross_split_and_allow_same_split_reuse() {
    for group in [
        Group::Speaker,
        Group::Session,
        Group::DerivedFamily,
        Group::TextFamily,
        Group::AudioHash,
    ] {
        let (root, mut same_split) = fixture();
        group.copy(&mut same_split, root.path(), 0, 1);
        group.copy(&mut same_split, root.path(), 2, 3);
        assert_both_accept(&same_split, root.path());

        for reverse in [false, true] {
            let (root, mut cross_split) = fixture();
            group.copy(&mut cross_split, root.path(), 0, 2);
            if reverse {
                cross_split.records.reverse();
            }
            assert_both_reject(&cross_split, root.path(), group.name());
        }
    }
}

#[test]
fn development_preparation_succeeds_with_unreadable_synthetic_holdout_decoy() {
    use std::os::unix::fs::symlink;

    let (root, mut data) = fixture();
    fs::create_dir(root.path().join("holdout")).unwrap();
    let decoy_path = root.path().join("holdout/decoy.wav");
    symlink("missing-synthetic-target.wav", &decoy_path).unwrap();
    assert!(
        fs::symlink_metadata(&decoy_path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        fs::metadata(&decoy_path).unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
    let mut decoy = data.records[0].clone();
    decoy.id = "holdout-decoy".into();
    decoy.split = ToneSplit::Holdout;
    decoy.audio.path = "holdout/decoy.wav".into();
    decoy.audio.sha256 = "f".repeat(64);
    decoy.text = "unreadable synthetic decoy".into();
    decoy.speaker_id = "decoy-speaker".into();
    decoy.session_id = "decoy-session".into();
    decoy.derived_family = "decoy-derived".into();
    decoy.text_family = "decoy-text-family".into();
    data.records.push(decoy);

    let error = data
        .validate(root.path())
        .expect_err("legacy must examine every record");
    assert!(
        error.contains("audio is missing"),
        "unexpected legacy error: {error}"
    );
    for split in [ToneSplit::Train, ToneSplit::Dev] {
        let prepared = prepare_development_inputs(
            &data,
            &plan(split),
            root.path(),
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .expect("preparing development data must not read the holdout decoy");
        assert_eq!(prepared.cases().len(), 1);
        assert_eq!(prepared.holdout_pcm_independence(), "unverified");
        assert_eq!(prepared.execution_provenance(), "not_run");
    }
    assert_eq!(
        fs::metadata(&decoy_path).unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
}
