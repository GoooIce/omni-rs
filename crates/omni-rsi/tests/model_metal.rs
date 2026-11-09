#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use burn::backend::wgpu::WgpuDevice;
use omni_rsi::{
    chart::{ChartSpec, Question, render, verify},
    model::FrozenReader,
};
use std::{path::PathBuf, sync::atomic::AtomicBool};

#[test]
#[ignore = "requires the official LFM3 Omni checkpoint and Apple Metal"]
fn real_model_writes_a_verified_chart_and_extracts_image_conditioned_features() {
    let directory = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR"));
    let reader = FrozenReader::<omni_lfm3::NativeGpu>::load(
        &directory,
        omni_lfm3::Precision::F16,
        &WgpuDevice::default(),
    )
    .unwrap();
    let stop = AtomicBool::new(false);
    let spec = ChartSpec::new(3, 7).unwrap();
    let program = reader.write_program(spec, &stop).unwrap().unwrap();
    let image = render(&program).unwrap();
    verify(&spec, &program, &image).unwrap();
    let left = reader
        .features(&image, Question::Taller, false, &stop)
        .unwrap();
    let opposite = ChartSpec::new(7, 3).unwrap();
    let image2 = render(&omni_rsi::chart::ChartProgram {
        left: opposite.left,
        right: opposite.right,
    })
    .unwrap();
    let right = reader
        .features(&image2, Question::Taller, true, &stop)
        .unwrap();
    assert_eq!(left.features.len(), 2048);
    assert!(left.features.iter().all(|x| x.is_finite()) && left.base_logit.is_finite());
    assert!(
        left.features
            .iter()
            .zip(&right.features)
            .any(|(a, b)| (a - b).abs() > 0.00001)
    );
}

#[test]
#[ignore = "requires the official LFM3 Omni checkpoint and Apple Metal"]
fn real_round_updates_a_reader_and_publishes_only_after_dev_validation() {
    use omni_rsi::{
        evolution::{RoundOptions, run_round},
        fingerprint::model_fingerprint,
        storage::Store,
    };
    let directory = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR"));
    let stop = AtomicBool::new(false);
    let fingerprint = model_fingerprint(&directory, &stop).unwrap();
    let reader = FrozenReader::<omni_lfm3::NativeGpu>::load(
        &directory,
        omni_lfm3::Precision::F16,
        &WgpuDevice::default(),
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path());
    let options = RoundOptions {
        families: 2,
        epochs: 120,
        learning_rate: 0.2,
        seed: 261003002,
    };
    let report = run_round(&reader, &store, &fingerprint, &options, &stop).unwrap();
    let candidate = store
        .load_candidate(&report.candidate, &fingerprint)
        .unwrap();
    assert_eq!(candidate.head.updates, options.epochs as u64);
    assert!(candidate.head.weights.iter().any(|weight| *weight != 0.0));
    assert!(
        report.train_examples >= 2
            && report
                .samples
                .iter()
                .all(|sample| sample.image_sha256.len() == 64)
    );
    assert_eq!(
        store.load_active(&fingerprint).unwrap().is_some(),
        report.promoted
    );
    eprintln!(
        "real RSI round: {}",
        serde_json::to_string(&report).unwrap()
    );
}
