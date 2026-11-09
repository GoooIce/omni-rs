use omni_lfm3::speech_inbox::{Completion, Receipt, SpeechCapture, SpeechInbox};

#[test]
fn completed_speech_is_recoverable_and_repeated_audio_is_deduplicated() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let samples: Vec<f32> = (0..16_000)
        .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
        .collect();

    let Receipt::Queued { id } = inbox.submit(&samples, Completion::Normal).unwrap() else {
        panic!("completed speech should be queued");
    };
    let record = inbox
        .get(&id)
        .unwrap()
        .expect("queued speech must survive reopening");
    assert_eq!(record.sample_rate, 16_000);
    assert_eq!(record.samples, 16_000);
    assert_eq!(record.source, "user");
    assert!(record.audio.is_file());
    assert_eq!(
        inbox.submit(&samples, Completion::Normal).unwrap(),
        Receipt::Duplicate { id }
    );
}

#[test]
fn cancelled_or_invalid_speech_does_not_become_a_completed_sample() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let speech = vec![0.25; 16_000];
    assert_eq!(
        inbox.submit(&speech, Completion::Cancelled).unwrap(),
        Receipt::Cancelled
    );
    assert!(matches!(
        inbox.submit(&[0.0; 100], Completion::Normal).unwrap(),
        Receipt::Rejected { .. }
    ));
    let mut invalid = speech;
    invalid[0] = f32::NAN;
    assert!(matches!(
        inbox.submit(&invalid, Completion::Normal).unwrap(),
        Receipt::Rejected { .. }
    ));
    assert_eq!(inbox.pending().unwrap().len(), 0);
}

#[test]
fn live_capture_keeps_a_bounded_utterance_and_discards_cancelled_input() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let mut capture = SpeechCapture::new();
    for _ in 0..10 {
        capture.push(&vec![0.1; 1_600]);
    }
    assert!(matches!(
        capture.finish(&inbox, Completion::Normal).unwrap(),
        Receipt::Queued { .. }
    ));

    let mut cancelled = SpeechCapture::new();
    cancelled.push(&vec![0.2; 16_000]);
    assert_eq!(
        cancelled.finish(&inbox, Completion::Cancelled).unwrap(),
        Receipt::Cancelled
    );

    let mut long = SpeechCapture::new();
    for _ in 0..17 {
        long.push(&vec![0.3; 16_000]);
    }
    assert!(matches!(
        long.finish(&inbox, Completion::Normal).unwrap(),
        Receipt::Rejected { .. }
    ));
    assert_eq!(inbox.pending().unwrap().len(), 1);
}

#[test]
fn previously_claimed_or_archived_speech_is_reported_as_duplicate() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let samples = vec![0.15; 16_000];
    let Receipt::Queued { id } = inbox.submit(&samples, Completion::Normal).unwrap() else {
        panic!("missing queued item")
    };
    for destination in ["processing", "processed"] {
        std::fs::create_dir_all(directory.path().join(destination)).unwrap();
    }
    std::fs::rename(
        directory.path().join("inbox").join(&id),
        directory.path().join("processing").join(&id),
    )
    .unwrap();
    assert_eq!(
        inbox.submit(&samples, Completion::Normal).unwrap(),
        Receipt::Duplicate { id: id.clone() }
    );
    std::fs::rename(
        directory.path().join("processing").join(&id),
        directory.path().join("processed").join(&id),
    )
    .unwrap();
    assert_eq!(
        inbox.submit(&samples, Completion::Normal).unwrap(),
        Receipt::Duplicate { id }
    );
    assert!(inbox.pending().unwrap().is_empty());
}
