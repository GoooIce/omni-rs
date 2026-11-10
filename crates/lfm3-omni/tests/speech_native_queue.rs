#![cfg(unix)]

use omni_lfm3::speech_inbox::{
    Completion, Receipt, SpeechClaim, SpeechInbox, SpeechOutcome, SpeechWorker,
};

fn queued(inbox: &SpeechInbox) -> String {
    let Receipt::Queued { id } = inbox
        .submit(&vec![0.25; 16_000], Completion::Normal)
        .unwrap()
    else {
        panic!("recording was not queued");
    };
    id
}

#[test]
fn native_claim_recovers_and_archives_rust_recording() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let id = queued(&inbox);
    {
        let worker = SpeechWorker::open(directory.path()).unwrap();
        let Some(SpeechClaim::Valid(entry)) = worker.claim().unwrap() else {
            panic!("valid claim expected")
        };
        assert_eq!(entry.id, id);
        assert_eq!((entry.sample_rate, entry.samples), (16_000, 16_000));
        assert!(directory.path().join("processing").join(&id).is_dir());
        assert!(SpeechWorker::open(directory.path()).is_err());
    }
    let worker = SpeechWorker::open(directory.path()).unwrap();
    let Some(SpeechClaim::Valid(entry)) = worker.claim().unwrap() else {
        panic!("recoverable claim expected")
    };
    assert_eq!(entry.id, id);
    let other = tempfile::tempdir().unwrap();
    let wrong_worker = SpeechWorker::open(other.path()).unwrap();
    assert!(
        wrong_worker
            .finish(&entry, SpeechOutcome::Accepted, "")
            .is_err()
    );
    worker.finish(&entry, SpeechOutcome::Accepted, "").unwrap();
    assert!(worker.claim().unwrap().is_none());
    assert!(
        directory
            .path()
            .join("processed")
            .join(&id)
            .join("audio.wav")
            .is_file()
    );
    assert_eq!(
        inbox
            .submit(&vec![0.25; 16_000], Completion::Normal)
            .unwrap(),
        Receipt::Duplicate { id }
    );
}

#[test]
fn corrupt_record_is_quarantined_before_it_can_be_used() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let id = queued(&inbox);
    std::fs::write(
        directory.path().join("inbox").join(&id).join("audio.wav"),
        b"corrupt",
    )
    .unwrap();
    let worker = SpeechWorker::open(directory.path()).unwrap();
    let Some(SpeechClaim::Rejected {
        id: rejected,
        reason,
    }) = worker.claim().unwrap()
    else {
        panic!("rejection expected")
    };
    assert_eq!(rejected, id);
    assert!(!reason.is_empty());
    assert!(directory.path().join("rejected").join(id).is_dir());
    assert!(worker.claim().unwrap().is_none());
}

#[test]
fn prior_outcome_can_be_completed_after_interruption() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let id = queued(&inbox);
    {
        let worker = SpeechWorker::open(directory.path()).unwrap();
        let Some(SpeechClaim::Valid(entry)) = worker.claim().unwrap() else {
            panic!("valid claim expected")
        };
        std::fs::write(
            entry.audio.parent().unwrap().join("result.json"),
            br#"{"outcome":"accepted","reason":""}"#,
        )
        .unwrap();
    }
    let worker = SpeechWorker::open(directory.path()).unwrap();
    let Some(SpeechClaim::Valid(entry)) = worker.claim().unwrap() else {
        panic!("recovery claim expected")
    };
    assert!(
        worker
            .finish(&entry, SpeechOutcome::Rejected, "conflict")
            .is_err()
    );
    worker.finish(&entry, SpeechOutcome::Accepted, "").unwrap();
    assert!(directory.path().join("processed").join(id).is_dir());
}
