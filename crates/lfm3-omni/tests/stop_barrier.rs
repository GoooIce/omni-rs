use std::{path::PathBuf, time::Duration};

use burn::backend::NdArray;
use omni_lfm3::{
    Precision,
    audio_encoder::{AudioEncoder, AudioEncoderConfig},
    causal_audio::{
        AudioSession,
        software_output::{Error, StopBarrier},
    },
    interruptible::{SessionId, State, StopReason, TimelineEvent},
    prepared::PreparedMedia,
};
use omni_r2t2::weights::Weights;

type Cpu = NdArray<f32>;

fn at(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

#[test]
fn stop_waits_for_producer_and_callback_while_input_continues() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(2), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 8).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 8).unwrap();
    owner
        .enqueue(at(4), &producer, &[1., 2., 3., 4., 5., 6.])
        .unwrap();
    let loan = owner.lend(at(5), 4).unwrap().unwrap();

    owner
        .stop(
            at(6),
            &producer,
            StopReason::Interrupted { input_end: 3200 },
        )
        .unwrap();
    assert!(producer.is_cancelled());
    assert!(owner.enqueue(at(7), &producer, &[7.]).is_err());
    owner.push(at(7), &vec![0.2; 3200], 16_000, false).unwrap();
    let next = owner.snapshot().unwrap().unwrap();
    let (_, next) = owner.start(at(8), next, 24_000).unwrap_err().into_parts();
    assert!(owner.finish_stop(at(8), &producer).is_err());
    owner.producer_quiesced(at(9), &producer).unwrap();
    assert!(owner.finish_stop(at(9), &producer).is_err());
    owner.return_playback(at(10), loan, 3).unwrap();
    owner.finish_stop(at(11), &producer).unwrap();
    assert_eq!(owner.conversation().state(), State::Listening);
    assert!(owner.conversation().timeline().iter().any(|entry| matches!(
        entry.event,
        TimelineEvent::PlaybackStopped {
            played_end: 3,
            queued_end: 6,
            generated_end: 8,
            ..
        }
    )));
    owner.start(at(12), next, 24_000).unwrap();
    assert!(
        owner.lend(at(13), 8).unwrap().is_none(),
        "old unplayed tail must not reach the replacement turn"
    );
}

#[test]
fn foreign_same_length_loan_cannot_fail_or_complete_another_owner_or_later_turn() {
    let encoder = encoder();
    let a = AudioSession::new(SessionId(3), &encoder, &Default::default(), at(0)).unwrap();
    let b = AudioSession::new(SessionId(3), &encoder, &Default::default(), at(0)).unwrap();
    let mut a = StopBarrier::new(a, 4).unwrap();
    let mut b = StopBarrier::new(b, 4).unwrap();
    a.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    b.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let a_snapshot = a.snapshot().unwrap().unwrap();
    let b_snapshot = b.snapshot().unwrap().unwrap();
    let old = a.start(at(2), a_snapshot, 24_000).unwrap();
    let other = b.start(at(2), b_snapshot, 24_000).unwrap();
    assert_eq!(old.turn(), other.turn());
    a.generated(at(3), &old, 2).unwrap();
    b.generated(at(3), &other, 2).unwrap();
    a.enqueue(at(4), &old, &[1., 2.]).unwrap();
    b.enqueue(at(4), &other, &[3., 4.]).unwrap();
    let loan = a.lend(at(5), 2).unwrap().unwrap();
    let foreign = b.lend(at(5), 2).unwrap().unwrap();

    assert!(matches!(
        a.playback_failed(at(6), &foreign),
        Err(Error::WrongLoan)
    ));
    let (error, foreign) = a
        .return_playback(at(6), foreign, 2)
        .unwrap_err()
        .into_parts();
    assert!(matches!(error, Error::WrongLoan));
    a.playback_failed(at(7), &loan).unwrap();
    assert!(old.is_cancelled());
    a.producer_quiesced(at(8), &old).unwrap();
    assert!(matches!(
        a.finish_stop(at(8), &old),
        Err(Error::BarrierPending)
    ));
    a.return_playback(at(9), loan, 1).unwrap();
    a.finish_stop(at(10), &old).unwrap();

    let snapshot = a.snapshot().unwrap().unwrap();
    let next = a.start(at(11), snapshot, 24_000).unwrap();
    assert!(a.generated(at(12), &old, 2).is_err());
    a.generated(at(12), &next, 2).unwrap();
    a.enqueue(at(13), &next, &[5., 6.]).unwrap();
    let next_loan = a.lend(at(14), 2).unwrap().unwrap();
    let (error, foreign) = a
        .return_playback(at(15), foreign, 2)
        .unwrap_err()
        .into_parts();
    assert!(matches!(error, Error::WrongLoan));
    a.stop(at(16), &next, StopReason::Cancelled).unwrap();
    a.producer_quiesced(at(17), &next).unwrap();
    assert!(matches!(
        a.finish_stop(at(17), &next),
        Err(Error::BarrierPending)
    ));
    a.return_playback(at(18), next_loan, 1).unwrap();
    a.finish_stop(at(19), &next).unwrap();
    b.return_playback(at(20), foreign, 2).unwrap();
}

#[test]
fn normal_finish_waits_for_partial_callbacks_and_drains_the_remainder_in_order() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(4), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 4).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 4).unwrap();
    owner.enqueue(at(4), &producer, &[1., 2., 3., 4.]).unwrap();
    let first = owner.lend(at(5), 2).unwrap().unwrap();
    owner.generation_finished(at(6), &producer).unwrap();
    assert!(matches!(
        owner.conversation().state(),
        State::Responding { .. }
    ));
    owner.return_playback(at(7), first, 1).unwrap();
    assert!(matches!(
        owner.conversation().state(),
        State::Responding { .. }
    ));
    let rest = owner.lend(at(8), 3).unwrap().unwrap();
    assert_eq!(rest.samples(), &[2., 3., 4.]);
    owner.return_playback(at(9), rest, 3).unwrap();
    assert_eq!(owner.conversation().state(), State::Listening);
    assert_eq!(
        owner
            .conversation()
            .timeline()
            .iter()
            .filter(|entry| matches!(entry.event, TimelineEvent::TurnCompleted { .. }))
            .count(),
        1
    );
}

#[test]
fn producer_borrows_its_original_input_snapshot_until_stop() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(5), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 4).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    assert_eq!(snapshot.input_end(), 3200);
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    assert_eq!(producer.input_end(), 3200);
    let values = |media: &PreparedMedia<Cpu>| {
        let PreparedMedia::Audio(tensor) = media else {
            panic!("audio snapshot required")
        };
        tensor.clone().into_data().to_vec::<f32>().unwrap()
    };
    let before = values(owner.media(&producer).unwrap());
    assert!(!before.is_empty());
    owner.push(at(3), &vec![0.2; 3200], 16_000, false).unwrap();
    assert_eq!(values(owner.media(&producer).unwrap()), before);
    owner
        .stop(
            at(4),
            &producer,
            StopReason::Interrupted { input_end: 6400 },
        )
        .unwrap();
    assert!(owner.media(&producer).is_err());
}

#[test]
fn invalid_callback_and_repeated_stop_do_not_allow_an_early_session_change() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(6), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 4).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 2).unwrap();
    owner.enqueue(at(4), &producer, &[1., 2.]).unwrap();
    let loan = owner.lend(at(5), 2).unwrap().unwrap();
    let timeline = owner.conversation().timeline().to_vec();
    let (error, loan) = owner
        .return_playback(at(4), loan, 1)
        .unwrap_err()
        .into_parts();
    assert!(matches!(error, Error::TimeWentBackwards));
    let (error, loan) = owner
        .return_playback(at(6), loan, 3)
        .unwrap_err()
        .into_parts();
    assert!(matches!(error, Error::InvalidSamples));
    assert_eq!(owner.conversation().timeline(), timeline);

    owner.stop(at(7), &producer, StopReason::Cancelled).unwrap();
    let stopping_timeline = owner.conversation().timeline().to_vec();
    assert!(owner.stop(at(8), &producer, StopReason::Cancelled).is_err());
    assert!(owner.change_session(at(9), SessionId(7)).is_err());
    assert_eq!(owner.conversation().timeline(), stopping_timeline);
    owner.producer_quiesced(at(10), &producer).unwrap();
    assert!(matches!(
        owner.finish_stop(at(10), &producer),
        Err(Error::BarrierPending)
    ));
    owner.return_playback(at(11), loan, 1).unwrap();
    assert!(matches!(
        owner.finish_stop(at(9), &producer),
        Err(Error::TimeWentBackwards)
    ));
    owner.finish_stop(at(12), &producer).unwrap();
    owner.change_session(at(13), SessionId(7)).unwrap();
    assert!(owner.generated(at(14), &producer, 3).is_err());
    owner.push(at(14), &vec![0.2; 3200], 16_000, false).unwrap();
    assert_eq!(owner.snapshot().unwrap().unwrap().session(), SessionId(7));
}

#[test]
fn a_lost_loan_keeps_stop_and_session_change_fail_closed() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(8), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 4).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 2).unwrap();
    owner.enqueue(at(4), &producer, &[1., 2.]).unwrap();
    let loan = owner.lend(at(5), 2).unwrap().unwrap();
    owner.stop(at(6), &producer, StopReason::Cancelled).unwrap();
    owner.producer_quiesced(at(7), &producer).unwrap();
    drop(loan);
    assert!(matches!(
        owner.finish_stop(at(8), &producer),
        Err(Error::BarrierPending)
    ));
    assert!(owner.change_session(at(9), SessionId(9)).is_err());
    assert!(matches!(
        owner.conversation().state(),
        State::Stopping { .. }
    ));
    assert!(
        !owner.conversation().timeline().iter().any(|entry| matches!(
            entry.event,
            TimelineEvent::PlaybackStopped { .. } | TimelineEvent::TurnCompleted { .. }
        ))
    );
}

#[test]
fn observed_input_failure_and_producer_failure_preserve_stop_time_and_wait_for_ack() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(10), &encoder, &Default::default(), at(0)).unwrap();
    let input_cancel = session.cancellation();
    let mut owner = StopBarrier::new(session, 4).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 2).unwrap();
    owner.enqueue(at(4), &producer, &[1., 2.]).unwrap();
    let loan = owner.lend(at(5), 2).unwrap().unwrap();
    input_cancel.cancel();
    assert!(owner.push(at(7), &[0.1], 16_000, false).is_err());
    assert!(matches!(
        owner.conversation().timeline().last().unwrap().event,
        TimelineEvent::InputFailed
    ));
    assert!(matches!(
        owner.stop(at(6), &producer, StopReason::GenerationFailed),
        Err(Error::TimeWentBackwards)
    ));
    owner
        .stop(at(8), &producer, StopReason::GenerationFailed)
        .unwrap();
    owner.return_playback(at(9), loan, 1).unwrap();
    assert!(matches!(
        owner.finish_stop(at(10), &producer),
        Err(Error::BarrierPending)
    ));
    owner.producer_quiesced(at(11), &producer).unwrap();
    owner.finish_stop(at(12), &producer).unwrap();
    assert_eq!(owner.conversation().state(), State::Listening);
    assert!(
        !owner
            .conversation()
            .timeline()
            .iter()
            .any(|entry| matches!(entry.event, TimelineEvent::TurnCompleted { .. }))
    );
}

#[test]
fn input_control_remains_scoped_and_serialized_through_the_owner() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(11), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 4).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    owner.user_paused(at(2), 3200).unwrap();
    assert!(matches!(
        owner.conversation().timeline().last().unwrap().event,
        TimelineEvent::UserPaused { input_end: 3200 }
    ));
    let old_cancel = owner.input_cancellation();
    old_cancel.cancel();
    assert!(owner.push(at(3), &[0.1], 16_000, false).is_err());
    owner.change_session(at(4), SessionId(12)).unwrap();
    old_cancel.cancel();
    owner.push(at(5), &vec![0.2; 3200], 16_000, false).unwrap();
    assert_eq!(owner.snapshot().unwrap().unwrap().session(), SessionId(12));
}

#[test]
fn an_empty_playback_poll_establishes_a_monotonic_observation_time() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(13), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 4).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    assert!(owner.lend(at(8), 2).unwrap().is_none());
    assert!(matches!(
        owner.generated(at(7), &producer, 1),
        Err(Error::TimeWentBackwards)
    ));
    owner.generated(at(8), &producer, 1).unwrap();
}

#[test]
fn finished_producer_can_drain_its_previously_generated_unqueued_tail() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(14), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 2).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 4).unwrap();
    owner.enqueue(at(4), &producer, &[1., 2.]).unwrap();
    owner.generation_finished(at(5), &producer).unwrap();
    let first = owner.lend(at(6), 2).unwrap().unwrap();
    owner.return_playback(at(7), first, 2).unwrap();
    assert!(matches!(
        owner.conversation().state(),
        State::Responding { .. }
    ));
    owner.enqueue(at(8), &producer, &[3., 4.]).unwrap();
    let tail = owner.lend(at(9), 2).unwrap().unwrap();
    assert_eq!(tail.samples(), &[3., 4.]);
    owner.return_playback(at(10), tail, 2).unwrap();
    assert_eq!(owner.conversation().state(), State::Listening);
}

#[test]
fn callback_failure_after_producer_finish_does_not_report_normal_completion() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(15), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 2).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 2).unwrap();
    owner.enqueue(at(4), &producer, &[1., 2.]).unwrap();
    let loan = owner.lend(at(5), 2).unwrap().unwrap();
    owner.generation_finished(at(6), &producer).unwrap();
    owner.playback_failed(at(7), &loan).unwrap();
    assert!(matches!(
        owner.finish_stop(at(7), &producer),
        Err(Error::BarrierPending)
    ));
    owner.return_playback(at(8), loan, 2).unwrap();
    owner.finish_stop(at(9), &producer).unwrap();
    assert!(
        !owner
            .conversation()
            .timeline()
            .iter()
            .any(|entry| matches!(entry.event, TimelineEvent::TurnCompleted { .. }))
    );
    assert!(owner.conversation().timeline().iter().any(|entry| matches!(
        entry.event,
        TimelineEvent::StopRequested {
            reason: StopReason::PlaybackFailed,
            ..
        }
    )));
}

fn encoder() -> AudioEncoder<Cpu> {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio_encoder_128");
    let config = AudioEncoderConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    AudioEncoder::from_weights(&weights, config, Precision::F32, &Default::default()).unwrap()
}

#[test]
fn borrowed_samples_count_toward_capacity_and_failed_admission_does_not_advance_the_timeline() {
    let encoder = encoder();
    let session = AudioSession::new(SessionId(1), &encoder, &Default::default(), at(0)).unwrap();
    let mut owner = StopBarrier::new(session, 2).unwrap();
    owner.push(at(1), &vec![0.1; 3200], 16_000, false).unwrap();
    let snapshot = owner.snapshot().unwrap().unwrap();
    let producer = owner.start(at(2), snapshot, 24_000).unwrap();
    owner.generated(at(3), &producer, 3).unwrap();
    owner.enqueue(at(4), &producer, &[0.1, 0.2]).unwrap();

    let timeline = owner.conversation().timeline().to_vec();
    assert!(matches!(
        owner.enqueue(at(5), &producer, &[0.3]),
        Err(Error::QueueFull)
    ));
    assert_eq!(owner.conversation().timeline(), timeline);
    assert!(
        owner
            .stop(
                at(5),
                &producer,
                StopReason::Interrupted { input_end: 3201 }
            )
            .is_err()
    );
    assert!(!producer.is_cancelled());
    assert_eq!(owner.conversation().timeline(), timeline);

    let loan = owner.lend(at(6), 2).unwrap().unwrap();
    assert_eq!(loan.samples(), &[0.1, 0.2]);
    assert!(matches!(
        owner.enqueue(at(7), &producer, &[0.3]),
        Err(Error::QueueFull)
    ));
}
