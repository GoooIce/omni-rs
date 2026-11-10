use std::{path::PathBuf, time::Duration};

use burn::backend::NdArray;
use omni_lfm3::{
    Precision,
    audio_encoder::{AudioEncoder, AudioEncoderConfig},
    causal_audio::{AudioSession, AudioTurn, TurnEvent},
    interruptible::{SessionId, StopReason, TimelineEvent},
    prepared::PreparedMedia,
};
use omni_r2t2::weights::Weights;

type Cpu = NdArray<f32>;

fn synthetic_encoder() -> AudioEncoder<Cpu> {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio_encoder_128");
    let config = AudioEncoderConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    AudioEncoder::from_weights(&weights, config, Precision::F32, &Default::default()).unwrap()
}

fn synthetic_pcm(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| ((n as f32) * 0.013).sin() * 0.1)
        .collect()
}

fn encoded_values(turn: &AudioTurn<Cpu>) -> Vec<f32> {
    let PreparedMedia::Audio(embeddings) = turn.media().unwrap() else {
        panic!("expected encoded audio")
    };
    embeddings.clone().into_data().to_vec::<f32>().unwrap()
}

#[test]
fn generation_uses_the_snapshot_actual_ingested_prefix() {
    let encoder = synthetic_encoder();
    let session_id = SessionId(7);
    let mut session =
        AudioSession::new(session_id, &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();

    let snapshot = session.snapshot().unwrap().expect("completed audio tokens");
    assert_eq!(snapshot.session(), session_id);
    assert_eq!(snapshot.input_end(), 3200);
    let started = session
        .start_generation(Duration::from_millis(2), snapshot, 24_000)
        .unwrap();
    assert!(matches!(
        session.conversation().timeline().last().unwrap().event,
        TimelineEvent::GenerationStarted {
            turn,
            input_end: 3200,
            output_rate: 24_000,
        } if turn == started.turn()
    ));
}

#[test]
fn same_prefix_and_different_future_leave_started_snapshots_unchanged() {
    let encoder = synthetic_encoder();
    let device = Default::default();
    let mut first = AudioSession::new(SessionId(10), &encoder, &device, Duration::ZERO).unwrap();
    let mut second = AudioSession::new(SessionId(11), &encoder, &device, Duration::ZERO).unwrap();
    let pcm = synthetic_pcm(6503);
    let mut other = pcm.clone();
    for sample in &mut other[3200..] {
        *sample = -*sample;
    }
    assert_ne!(&pcm[3200..], &other[3200..]);
    first
        .push(Duration::from_millis(1), &pcm[..3200], 16_000, false)
        .unwrap();
    second
        .push(Duration::from_millis(1), &other[..3200], 16_000, false)
        .unwrap();
    let first_snapshot = first.snapshot().unwrap().unwrap();
    let second_snapshot = second.snapshot().unwrap().unwrap();
    let first_turn = first
        .start_generation(Duration::from_millis(2), first_snapshot, 24_000)
        .unwrap();
    let second_turn = second
        .start_generation(Duration::from_millis(2), second_snapshot, 24_000)
        .unwrap();
    let before = encoded_values(&first_turn);
    assert!(!before.is_empty());
    assert_eq!(before, encoded_values(&second_turn));

    first
        .push(Duration::from_millis(3), &pcm[3200..], 16_000, false)
        .unwrap();
    second
        .push(Duration::from_millis(3), &other[3200..], 16_000, false)
        .unwrap();
    assert_eq!(first_turn.input_end(), 3200);
    assert_eq!(second_turn.input_end(), 3200);
    assert_eq!(before, encoded_values(&first_turn));
    assert_eq!(before, encoded_values(&second_turn));
}

#[test]
fn snapshot_from_another_stream_with_the_same_session_id_is_rejected() {
    let encoder = synthetic_encoder();
    let device = Default::default();
    let id = SessionId(42);
    let mut original = AudioSession::new(id, &encoder, &device, Duration::ZERO).unwrap();
    let mut other = AudioSession::new(id, &encoder, &device, Duration::ZERO).unwrap();
    original
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    other
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let snapshot = original.snapshot().unwrap().unwrap();
    let timeline_len = other.conversation().timeline().len();

    assert!(
        other
            .start_generation(Duration::from_millis(2), snapshot, 24_000)
            .is_err()
    );
    assert_eq!(other.conversation().timeline().len(), timeline_len);
}

#[test]
fn session_change_discards_the_old_stream_and_rejects_its_snapshot() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(1), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let old_snapshot = session.snapshot().unwrap().unwrap();

    session
        .change_session(Duration::from_millis(2), SessionId(2))
        .unwrap();
    assert!(session.snapshot().unwrap().is_none());
    session
        .push(
            Duration::from_millis(3),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let timeline_len = session.conversation().timeline().len();
    assert!(
        session
            .start_generation(Duration::from_millis(4), old_snapshot, 24_000)
            .is_err()
    );
    assert_eq!(session.conversation().timeline().len(), timeline_len);
    let new_snapshot = session.snapshot().unwrap().unwrap();
    assert_eq!(new_snapshot.session(), SessionId(2));
    assert_eq!(new_snapshot.input_end(), 3200);
    assert!(
        session
            .change_session(Duration::from_millis(4), SessionId(1))
            .is_err()
    );
    assert!(
        session
            .start_generation(Duration::from_millis(5), new_snapshot, 24_000)
            .is_ok()
    );
}

#[test]
fn session_switch_waits_for_the_turn_and_invalidates_started_audio() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(1), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let snapshot = session.snapshot().unwrap().unwrap();
    let turn = session
        .start_generation(Duration::from_millis(2), snapshot, 24_000)
        .unwrap();
    assert!(turn.media().is_ok());
    assert!(
        session
            .change_session(Duration::from_millis(3), SessionId(2))
            .is_err()
    );
    assert!(turn.media().is_ok());

    session
        .apply_turn(
            Duration::from_millis(4),
            &turn,
            TurnEvent::GenerationFinished,
        )
        .unwrap();
    session
        .change_session(Duration::from_millis(5), SessionId(2))
        .unwrap();
    assert!(turn.media().is_err());
}

#[test]
fn input_failure_poisons_the_stream_and_previously_created_snapshots() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(1), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let snapshot = session.snapshot().unwrap().unwrap();
    let timeline_len = session.conversation().timeline().len();

    assert!(
        session
            .push(Duration::from_millis(2), &[f32::NAN], 16_000, false)
            .is_err()
    );
    assert_eq!(session.conversation().timeline().len(), timeline_len);
    assert!(session.snapshot().is_err());
    assert!(
        session
            .push(Duration::from_millis(3), &[0.1], 16_000, false)
            .is_err()
    );
    assert!(
        session
            .start_generation(Duration::from_millis(4), snapshot, 24_000)
            .is_err()
    );
}

#[test]
fn accepted_stop_revokes_only_that_turn_and_input_can_continue() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(1), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let old_snapshot = session.snapshot().unwrap().unwrap();
    let turn = session
        .start_generation(Duration::from_millis(2), old_snapshot, 24_000)
        .unwrap();
    assert!(turn.media().is_ok());

    assert!(
        session
            .apply_turn(
                Duration::from_millis(3),
                &turn,
                TurnEvent::Stop {
                    reason: StopReason::Interrupted { input_end: 3201 }
                }
            )
            .is_err()
    );
    assert!(turn.media().is_ok());
    session
        .apply_turn(
            Duration::from_millis(3),
            &turn,
            TurnEvent::Stop {
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    assert!(turn.media().is_err());
    session
        .push(
            Duration::from_millis(4),
            &synthetic_pcm(1600),
            16_000,
            false,
        )
        .unwrap();
    let next_snapshot = session.snapshot().unwrap().unwrap();
    assert_eq!(next_snapshot.input_end(), 4800);
    session
        .apply_turn(
            Duration::from_millis(5),
            &turn,
            TurnEvent::Stopped { played_end: 0 },
        )
        .unwrap();
    let next_turn = session
        .start_generation(Duration::from_millis(6), next_snapshot, 24_000)
        .unwrap();
    assert!(next_turn.media().is_ok());
    assert!(
        session
            .apply_turn(
                Duration::from_millis(7),
                &turn,
                TurnEvent::Generated { end: 1 }
            )
            .is_err()
    );
    assert!(turn.media().is_err());
}

#[test]
fn cancelling_input_revokes_unconsumed_and_started_snapshots() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(1), &encoder, &Default::default(), Duration::ZERO).unwrap();
    let cancel = session.cancellation();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let unused = session.snapshot().unwrap().unwrap();
    let started = session.snapshot().unwrap().unwrap();
    let turn = session
        .start_generation(Duration::from_millis(2), started, 24_000)
        .unwrap();
    assert!(turn.media().is_ok());

    cancel.cancel();
    assert!(turn.media().is_err());
    assert!(session.snapshot().is_err());
    assert!(
        session
            .push(Duration::from_millis(3), &[0.1], 16_000, false)
            .is_err()
    );
    assert!(
        session
            .apply_turn(
                Duration::from_millis(4),
                &turn,
                TurnEvent::Generated { end: 1 }
            )
            .is_err()
    );
    session
        .apply_turn(
            Duration::from_millis(5),
            &turn,
            TurnEvent::Stop {
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    session
        .apply_turn(
            Duration::from_millis(6),
            &turn,
            TurnEvent::Stopped { played_end: 0 },
        )
        .unwrap();
    assert!(
        session
            .start_generation(Duration::from_millis(7), unused, 24_000)
            .is_err()
    );

    session
        .change_session(Duration::from_millis(8), SessionId(2))
        .unwrap();
    cancel.cancel();
    session
        .push(
            Duration::from_millis(9),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    assert_eq!(session.snapshot().unwrap().unwrap().session(), SessionId(2));
}

#[test]
fn dropping_a_session_revokes_unconsumed_and_started_snapshots() {
    let encoder = synthetic_encoder();
    let device = Default::default();
    let (turn, unused) = {
        let mut session =
            AudioSession::new(SessionId(9), &encoder, &device, Duration::ZERO).unwrap();
        session
            .push(
                Duration::from_millis(1),
                &synthetic_pcm(3200),
                16_000,
                false,
            )
            .unwrap();
        let unused = session.snapshot().unwrap().unwrap();
        let snapshot = session.snapshot().unwrap().unwrap();
        let turn = session
            .start_generation(Duration::from_millis(2), snapshot, 24_000)
            .unwrap();
        assert!(turn.media().is_ok());
        (turn, unused)
    };
    assert!(turn.media().is_err());
    let mut replacement =
        AudioSession::new(SessionId(9), &encoder, &device, Duration::ZERO).unwrap();
    replacement
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    assert!(
        replacement
            .start_generation(Duration::from_millis(2), unused, 24_000)
            .is_err()
    );
}

#[test]
fn input_failure_revokes_active_turn_but_still_allows_stop_barrier() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(9), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let snapshot = session.snapshot().unwrap().unwrap();
    let turn = session
        .start_generation(Duration::from_millis(2), snapshot, 24_000)
        .unwrap();
    assert!(
        session
            .push(Duration::from_millis(3), &[f32::INFINITY], 16_000, false)
            .is_err()
    );
    assert!(turn.media().is_err());
    assert!(
        session
            .apply_turn(
                Duration::from_millis(4),
                &turn,
                TurnEvent::Generated { end: 1 }
            )
            .is_err()
    );
    session
        .apply_turn(
            Duration::from_millis(5),
            &turn,
            TurnEvent::Stop {
                reason: StopReason::GenerationFailed,
            },
        )
        .unwrap();
    session
        .apply_turn(
            Duration::from_millis(6),
            &turn,
            TurnEvent::Stopped { played_end: 0 },
        )
        .unwrap();
    session
        .change_session(Duration::from_millis(7), SessionId(10))
        .unwrap();
    session
        .push(
            Duration::from_millis(8),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    assert_eq!(
        session.snapshot().unwrap().unwrap().session(),
        SessionId(10)
    );
}

#[test]
fn matching_turn_ids_from_different_streams_cannot_cross_sessions() {
    let encoder = synthetic_encoder();
    let device = Default::default();
    let mut first = AudioSession::new(SessionId(9), &encoder, &device, Duration::ZERO).unwrap();
    let mut second = AudioSession::new(SessionId(9), &encoder, &device, Duration::ZERO).unwrap();
    first
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    second
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let first_snapshot = first.snapshot().unwrap().unwrap();
    let second_snapshot = second.snapshot().unwrap().unwrap();
    let first_turn = first
        .start_generation(Duration::from_millis(2), first_snapshot, 24_000)
        .unwrap();
    let second_turn = second
        .start_generation(Duration::from_millis(2), second_snapshot, 24_000)
        .unwrap();
    assert_eq!(first_turn.turn(), second_turn.turn());

    assert!(
        second
            .apply_turn(
                Duration::from_millis(3),
                &first_turn,
                TurnEvent::Stop {
                    reason: StopReason::Cancelled
                }
            )
            .is_err()
    );
    assert!(first_turn.media().is_ok());
    assert!(second_turn.media().is_ok());
    second
        .apply_turn(
            Duration::from_millis(3),
            &second_turn,
            TurnEvent::Stop {
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    assert!(second_turn.media().is_err());
    assert!(first_turn.media().is_ok());
}

#[test]
fn snapshot_requires_completed_tokens_and_final_padding_keeps_the_true_boundary() {
    let encoder = synthetic_encoder();
    let device = Default::default();
    let mut session = AudioSession::new(SessionId(1), &encoder, &device, Duration::ZERO).unwrap();
    assert!(session.snapshot().unwrap().is_none());
    assert_eq!(
        session
            .push(
                Duration::from_millis(1),
                &synthetic_pcm(1519),
                16_000,
                false
            )
            .unwrap(),
        0
    );
    assert!(session.snapshot().unwrap().is_none());
    assert_eq!(
        session
            .push(Duration::from_millis(2), &[0.1], 16_000, false)
            .unwrap(),
        1
    );
    assert_eq!(session.snapshot().unwrap().unwrap().input_end(), 1520);

    let mut short = AudioSession::new(SessionId(2), &encoder, &device, Duration::ZERO).unwrap();
    short
        .push(Duration::from_millis(1), &synthetic_pcm(40), 16_000, true)
        .unwrap();
    let finished = short.snapshot().unwrap().unwrap();
    assert_eq!(finished.input_end(), 40);
    assert!(
        short
            .push(Duration::from_millis(2), &[0.0], 16_000, false)
            .is_err()
    );
    assert!(
        short
            .start_generation(Duration::from_millis(3), finished, 24_000)
            .is_err()
    );
}

#[test]
fn snapshots_taken_before_and_after_more_pcm_keep_their_own_boundaries() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(3), &encoder, &Default::default(), Duration::ZERO).unwrap();
    let pcm = synthetic_pcm(6503);
    session
        .push(Duration::from_millis(1), &pcm[..3200], 16_000, false)
        .unwrap();
    let earlier = session.snapshot().unwrap().unwrap();
    session
        .push(Duration::from_millis(2), &pcm[3200..], 16_000, false)
        .unwrap();
    let later = session.snapshot().unwrap().unwrap();
    assert_eq!((earlier.input_end(), later.input_end()), (3200, 6503));

    let first = session
        .start_generation(Duration::from_millis(3), earlier, 24_000)
        .unwrap();
    let first_values = encoded_values(&first);
    assert!(matches!(
        session.conversation().timeline().last().unwrap().event,
        TimelineEvent::GenerationStarted {
            input_end: 3200,
            ..
        }
    ));
    session
        .apply_turn(
            Duration::from_millis(4),
            &first,
            TurnEvent::GenerationFinished,
        )
        .unwrap();
    assert!(first.media().is_err());
    let second = session
        .start_generation(Duration::from_millis(5), later, 24_000)
        .unwrap();
    assert!(matches!(
        session.conversation().timeline().last().unwrap().event,
        TimelineEvent::GenerationStarted {
            input_end: 6503,
            ..
        }
    ));
    assert!(encoded_values(&second).len() > first_values.len());
}

#[test]
fn backwards_input_time_is_rejected_before_pcm_is_ingested() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(1), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(2),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let timeline_len = session.conversation().timeline().len();
    assert!(
        session
            .push(
                Duration::from_millis(1),
                &synthetic_pcm(3200),
                16_000,
                false
            )
            .is_err()
    );
    assert_eq!(session.conversation().timeline().len(), timeline_len);
    let snapshot = session.snapshot().unwrap().unwrap();
    assert_eq!(snapshot.input_end(), 3200);
    let turn = session
        .start_generation(Duration::from_millis(3), snapshot, 24_000)
        .unwrap();
    assert_eq!(turn.input_end(), 3200);
}
