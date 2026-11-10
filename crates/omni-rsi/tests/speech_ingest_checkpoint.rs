#![cfg(feature = "cpu")]

use burn::backend::NdArray;
use omni_lfm3::{Precision, speech_inbox::SpeechInbox};
use omni_rsi::{
    idle::IdlePolicy,
    lease::Lease,
    platform,
    speech_ingest::{IngestResult, LocalVerifier, ingest_one},
    watchdog::Watchdog,
};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

#[test]
#[ignore = "requires local LFM3, R2T2, verified data and a completed speech fixture"]
fn real_recording_passes_native_admission() {
    let path = |name| PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")));
    let model = path("LFM3_MODEL");
    let data = path("LFM3_SPEECH_DATA");
    let asr = path("LFM3_ASR");
    let binary = path("LFM3_ASR_BINARY");
    let original = path("LFM3_RECORDING");
    let lease_root = path("LFM3_RSI_LEASE");
    let policy = IdlePolicy {
        min_idle_seconds: 300.0,
        ..Default::default()
    };
    let state = platform::snapshot(&lease_root);
    assert!(
        policy.blocker(&state).is_none(),
        "system is not idle: {:?}",
        policy.blocker(&state)
    );
    let _device = Lease::try_acquire(&lease_root.join("device.lock"))
        .unwrap()
        .expect("device in use");
    let stop = Arc::new(AtomicBool::new(false));
    let check_root = lease_root.clone();
    let _watchdog = Watchdog::start(stop.clone(), Duration::from_millis(500), move || {
        policy.blocker(&platform::snapshot(&check_root)).is_none()
            && Lease::try_acquire(&check_root.join("foreground.lock"))
                .is_ok_and(|lease| lease.is_some())
    });
    let temporary = tempfile::tempdir().unwrap();
    let _inbox = SpeechInbox::open(temporary.path()).unwrap();
    let id = original.file_name().unwrap();
    let destination = temporary.path().join("inbox").join(id);
    std::fs::create_dir(&destination).unwrap();
    for filename in ["audio.wav", "manifest.json"] {
        std::fs::copy(original.join(filename), destination.join(filename)).unwrap();
    }
    let outcome = ingest_one(
        temporary.path(),
        &data,
        || {
            LocalVerifier::<NdArray<f32>>::new(
                &model,
                &asr,
                &binary,
                Precision::F32,
                &Default::default(),
            )
        },
        &stop,
    )
    .unwrap();
    assert!(
        matches!(
            outcome,
            IngestResult::Accepted {
                recovered: false,
                ..
            }
        ),
        "{outcome:?}"
    );
}
