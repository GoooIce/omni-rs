#![cfg(any(feature = "cpu", feature = "metal"))]

use burn::{backend::Autodiff, tensor::backend::AutodiffBackend};
use omni_rsi::{
    idle::IdlePolicy,
    lease::Lease,
    platform,
    speech_checkpoint::SpeechCheckpoint,
    speech_round::{SpeechRoundOptions, train_round},
    speech_training::SpeechOptimizer,
    watchdog::Watchdog,
};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

fn real_burn_training_updates_and_resumes<B: AutodiffBackend>(require_idle: bool) {
    let path = |name| PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")));
    let model = path("LFM3_MODEL");
    let data = path("LFM3_SPEECH_DATA");
    let asr = path("LFM3_ASR");
    let binary = path("LFM3_ASR_BINARY");
    let lease_root = path("LFM3_RSI_LEASE");
    let policy = IdlePolicy {
        min_idle_seconds: 300.0,
        ..Default::default()
    };
    let _device = if require_idle {
        assert!(
            policy.blocker(&platform::snapshot(&lease_root)).is_none(),
            "system is not idle"
        );
        Some(
            Lease::try_acquire(&lease_root.join("device.lock"))
                .unwrap()
                .expect("device in use"),
        )
    } else {
        None
    };
    let stop = Arc::new(AtomicBool::new(false));
    let check_root = lease_root.clone();
    let _watchdog = require_idle.then(|| {
        Watchdog::start(stop.clone(), Duration::from_millis(500), move || {
            policy.blocker(&platform::snapshot(&check_root)).is_none()
                && Lease::try_acquire(&check_root.join("foreground.lock"))
                    .is_ok_and(|lease| lease.is_some())
        })
    });
    let state = tempfile::tempdir().unwrap();
    let options = SpeechRoundOptions {
        model: &model,
        data: &data,
        online: None,
        state: state.path(),
        asr: &asr,
        asr_binary: &binary,
        epochs: 5,
        max_updates: 1,
    };
    let first = train_round::<B, _>(&options, &Default::default(), &stop, |_, _, _, _, _| {
        panic!("generation gate must wait until the epoch ends")
    })
    .unwrap();
    assert_eq!((first.progress.updates, first.progress.next_index), (1, 1));
    let checkpoint = SpeechCheckpoint::load(state.path()).unwrap().unwrap();
    let optimizer =
        SpeechOptimizer::<B>::restore(checkpoint.optimizer, &Default::default()).unwrap();
    assert!(
        optimizer
            .export()
            .unwrap()
            .1
            .iter()
            .any(|value| *value != 0.0)
    );
    let second = train_round::<B, _>(&options, &Default::default(), &stop, |_, _, _, _, _| {
        panic!("generation gate must wait until the epoch ends")
    })
    .unwrap();
    assert_eq!(
        (second.progress.updates, second.progress.next_index),
        (2, 2)
    );
    assert_ne!(first.checkpoint, second.checkpoint);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires the local LFM3 checkpoint and verified speech data"]
fn cpu_training_updates_and_resumes_without_python() {
    real_burn_training_updates_and_resumes::<Autodiff<burn::backend::NdArray<f32>>>(false);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires the local LFM3 checkpoint, verified data and an idle Metal device"]
fn metal_training_updates_and_resumes_without_python() {
    real_burn_training_updates_and_resumes::<Autodiff<omni_lfm3::NativeGpu>>(true);
}
