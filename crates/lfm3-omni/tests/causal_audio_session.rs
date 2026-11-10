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
fn session_pcm_matches_fixed_official_features_and_rejects_a_perturbed_prefix() {
    use burn::tensor::{Tensor, TensorData};
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Reference {
        samples: Vec<f32>,
        features: Vec<Vec<f32>>,
    }
    let reference: Reference =
        serde_json::from_slice(include_bytes!("fixtures/audio_features.json")).unwrap();
    let encoder = synthetic_encoder();
    let device = Default::default();
    let frames = reference.features[0].len();
    assert_eq!(reference.features.len(), 128);
    assert!(reference.features.iter().all(|row| row.len() == frames));
    let features = reference.features.iter().flatten().copied().collect();
    let expected = encoder
        .encode(Tensor::from_data(
            TensorData::new(features, [1, 128, frames]),
            &device,
        ))
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let run = |id, samples: &[f32]| {
        let mut session =
            AudioSession::new(SessionId(id), &encoder, &device, Duration::ZERO).unwrap();
        session
            .push(Duration::from_millis(1), samples, 16_000, true)
            .unwrap();
        let snapshot = session.snapshot().unwrap().unwrap();
        let turn = session
            .start_generation(Duration::from_millis(2), snapshot, 24_000)
            .unwrap();
        encoded_values(&turn)
    };
    let compare = |actual: &[f32]| {
        actual.len() == expected.len()
            && actual.iter().all(|x| x.is_finite())
            && actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 0.0001)
    };
    assert!(!expected.is_empty() && expected.iter().all(|x| x.is_finite()));
    assert!(
        compare(&run(51, &reference.samples)),
        "session must assemble the official reference PCM"
    );
    let mut changed = reference.samples.clone();
    changed[..800].fill(0.0);
    assert!(
        !compare(&run(52, &changed)),
        "perturbed prefix must fail the same comparison"
    );
}

#[test]
fn oversized_input_chunk_is_rejected_without_advancing_or_poisoning_the_session() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(53), &encoder, &Default::default(), Duration::ZERO).unwrap();
    assert!(
        session
            .push(
                Duration::from_millis(1),
                &vec![0.0; 16_000 * 5 + 1],
                16_000,
                false
            )
            .is_err()
    );
    assert_eq!(session.conversation().timeline().len(), 1);
    session
        .push(
            Duration::from_millis(2),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    assert_eq!(session.snapshot().unwrap().unwrap().input_end(), 3200);
}

#[test]
fn outstanding_snapshots_have_a_recoverable_per_session_limit() {
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(54), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let mut snapshots: Vec<_> = (0..8)
        .map(|_| session.snapshot().unwrap().unwrap())
        .collect();
    assert!(
        session.snapshot().is_err(),
        "a ninth live snapshot exceeds the limit"
    );
    snapshots.pop();
    assert!(session.snapshot().unwrap().is_some());
}

#[test]
fn tighter_limits_bound_total_input_and_snapshots_across_session_changes() {
    use omni_lfm3::causal_audio::AudioLimits;
    let encoder = synthetic_encoder();
    let limits = AudioLimits {
        max_push_samples: 3200,
        max_session_samples: 3200,
        max_live_snapshots: 2,
    };
    let mut session = AudioSession::with_limits(
        SessionId(55),
        &encoder,
        &Default::default(),
        Duration::ZERO,
        limits,
    )
    .unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let first = session.snapshot().unwrap().unwrap();
    let second = session.snapshot().unwrap().unwrap();
    assert!(
        session
            .push(Duration::from_millis(2), &[0.0], 16_000, false)
            .is_err()
    );
    assert_eq!(first.input_end(), 3200);
    session
        .change_session(Duration::from_millis(3), SessionId(56))
        .unwrap();
    session
        .push(
            Duration::from_millis(4),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    assert!(
        session.snapshot().is_err(),
        "old outstanding tensors still consume the session quota"
    );
    drop(first);
    assert!(session.snapshot().unwrap().is_some());
    drop(second);
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

    let error = other
        .start_generation(Duration::from_millis(2), snapshot, 24_000)
        .err()
        .expect("foreign stream must reject the snapshot");
    let (_, returned) = error.into_parts();
    assert_eq!(returned.input_end(), 3200);
    assert_eq!(other.conversation().timeline().len(), timeline_len);
    assert!(
        original
            .start_generation(Duration::from_millis(2), returned, 24_000)
            .is_ok()
    );
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
    assert_eq!(session.conversation().timeline().len(), timeline_len + 1);
    assert!(matches!(
        session.conversation().timeline().last().unwrap().event,
        TimelineEvent::InputFailed
    ));
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
fn losing_the_media_turn_keeps_explicit_stop_and_barrier_control() {
    use omni_lfm3::causal_audio::ControlEvent;
    use omni_lfm3::interruptible::{Outcome, State};
    let encoder = synthetic_encoder();
    let mut session =
        AudioSession::new(SessionId(31), &encoder, &Default::default(), Duration::ZERO).unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let first = session
        .start_generation(
            Duration::from_millis(2),
            session.snapshot().unwrap().unwrap(),
            24_000,
        )
        .unwrap();
    let saved = session.current_control().unwrap();
    let mut other =
        AudioSession::new(SessionId(31), &encoder, &Default::default(), Duration::ZERO).unwrap();
    other
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let other_turn = other
        .start_generation(
            Duration::from_millis(2),
            other.snapshot().unwrap().unwrap(),
            24_000,
        )
        .unwrap();
    assert!(
        other
            .apply_control(
                Duration::from_millis(3),
                &saved,
                ControlEvent::Stop {
                    reason: StopReason::Cancelled
                }
            )
            .is_err()
    );
    assert!(other_turn.media().is_ok());
    drop(first);
    let recovered = session
        .current_control()
        .expect("active turn remains controllable");
    assert!(matches!(
        session.conversation().state(),
        State::Responding { .. }
    ));
    assert!(matches!(
        session
            .apply_control(
                Duration::from_millis(3),
                &recovered,
                ControlEvent::Stop {
                    reason: StopReason::GenerationFailed
                }
            )
            .unwrap(),
        Outcome::Stop(_)
    ));
    assert!(matches!(
        session.conversation().state(),
        State::Stopping { .. }
    ));
    assert!(
        session
            .change_session(Duration::from_millis(4), SessionId(32))
            .is_err()
    );
    // A caller, not the reducer, confirms the producer and playback barrier.
    session
        .apply_control(
            Duration::from_millis(5),
            &recovered,
            ControlEvent::Stopped { played_end: 0 },
        )
        .unwrap();
    assert!(matches!(session.conversation().state(), State::Listening));
    let next = session
        .start_generation(
            Duration::from_millis(6),
            session.snapshot().unwrap().unwrap(),
            24_000,
        )
        .unwrap();
    assert!(next.media().is_ok());
    assert!(
        session
            .apply_control(
                Duration::from_millis(7),
                &saved,
                ControlEvent::Stopped { played_end: 0 }
            )
            .is_err()
    );
}

#[test]
fn failed_input_cannot_be_stopped_before_the_failure_time() {
    use omni_lfm3::causal_audio::ControlEvent;
    use omni_lfm3::interruptible::State;
    let encoder = synthetic_encoder();
    let mut session = AudioSession::new(
        SessionId(131),
        &encoder,
        &Default::default(),
        Duration::ZERO,
    )
    .unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let turn = session
        .start_generation(
            Duration::from_millis(2),
            session.snapshot().unwrap().unwrap(),
            24_000,
        )
        .unwrap();
    let control = session.current_control().unwrap();
    assert!(
        session
            .push(Duration::from_millis(10), &[f32::NAN], 16_000, false)
            .is_err()
    );
    assert!(turn.media().is_err());
    assert!(
        session
            .conversation()
            .timeline()
            .iter()
            .any(|entry| entry.at == Duration::from_millis(10)
                && matches!(entry.event, TimelineEvent::InputFailed))
    );
    assert!(
        session
            .apply_control(
                Duration::from_millis(3),
                &control,
                ControlEvent::Stop {
                    reason: StopReason::Cancelled
                }
            )
            .is_err()
    );
    assert!(matches!(
        session.conversation().state(),
        State::Responding { .. }
    ));
    session
        .apply_control(
            Duration::from_millis(10),
            &control,
            ControlEvent::Stop {
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    session
        .apply_control(
            Duration::from_millis(11),
            &control,
            ControlEvent::Stopped { played_end: 0 },
        )
        .unwrap();
    assert!(matches!(session.conversation().state(), State::Listening));
    assert!(
        session
            .conversation()
            .timeline()
            .iter()
            .all(|event| event.at != Duration::from_millis(3))
    );
}

#[test]
fn recoverable_input_limit_rejection_does_not_advance_the_time() {
    use omni_lfm3::causal_audio::ControlEvent;
    let encoder = synthetic_encoder();
    let mut session = AudioSession::new(
        SessionId(132),
        &encoder,
        &Default::default(),
        Duration::ZERO,
    )
    .unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let turn = session
        .start_generation(
            Duration::from_millis(2),
            session.snapshot().unwrap().unwrap(),
            24_000,
        )
        .unwrap();
    let control = session.current_control().unwrap();
    assert!(
        session
            .push(Duration::from_millis(10), &vec![0.0; 80_001], 16_000, false)
            .is_err()
    );
    assert!(turn.media().is_ok());
    session
        .apply_control(
            Duration::from_millis(3),
            &control,
            ControlEvent::Stop {
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    session
        .apply_control(
            Duration::from_millis(4),
            &control,
            ControlEvent::Stopped { played_end: 0 },
        )
        .unwrap();
}

#[test]
fn empty_new_session_returns_none_even_with_old_snapshot_quota_full() {
    use omni_lfm3::causal_audio::AudioLimits;
    let encoder = synthetic_encoder();
    let limits = AudioLimits {
        max_push_samples: 3200,
        max_session_samples: 6400,
        max_live_snapshots: 1,
    };
    let mut session = AudioSession::with_limits(
        SessionId(133),
        &encoder,
        &Default::default(),
        Duration::ZERO,
        limits,
    )
    .unwrap();
    session
        .push(
            Duration::from_millis(1),
            &synthetic_pcm(3200),
            16_000,
            false,
        )
        .unwrap();
    let old = session.snapshot().unwrap().unwrap();
    session
        .change_session(Duration::from_millis(2), SessionId(134))
        .unwrap();
    assert!(session.snapshot().unwrap().is_none());
    session
        .push(Duration::from_millis(3), &synthetic_pcm(40), 16_000, false)
        .unwrap();
    assert!(session.snapshot().unwrap().is_none());
    session
        .push(
            Duration::from_millis(4),
            &synthetic_pcm(3160),
            16_000,
            false,
        )
        .unwrap();
    assert!(
        session.snapshot().is_err(),
        "old live snapshot still consumes the quota"
    );
    drop(old);
    assert!(session.snapshot().unwrap().is_some());
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
fn empty_final_flush_advances_the_causal_time_before_generation() {
    let encoder = synthetic_encoder();
    let device = Default::default();
    let mut session = AudioSession::new(SessionId(91), &encoder, &device, Duration::ZERO).unwrap();
    session
        .push(Duration::from_millis(1), &synthetic_pcm(40), 16_000, false)
        .unwrap();
    assert!(session.snapshot().unwrap().is_none());
    session
        .push(Duration::from_millis(10), &[], 16_000, true)
        .unwrap();
    let snapshot = session
        .snapshot()
        .unwrap()
        .expect("final flush emits encoded tail");
    assert_eq!(snapshot.input_end(), 40);
    assert_eq!(
        session.conversation().timeline().last().unwrap().at,
        Duration::from_millis(10)
    );
    assert!(matches!(
        session.conversation().timeline().last().unwrap().event,
        TimelineEvent::InputFinalized { end: 40 }
    ));
    assert!(
        session
            .start_generation(Duration::from_millis(2), snapshot, 24_000)
            .is_err()
    );
    assert!(
        session
            .push(Duration::from_millis(9), &[], 16_000, true)
            .is_err()
    );
    let snapshot = session.snapshot().unwrap().unwrap();
    assert_eq!(
        session
            .start_generation(Duration::from_millis(10), snapshot, 24_000)
            .unwrap()
            .input_end(),
        40
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
fn rejected_start_returns_the_same_snapshot_for_retry_after_input_advances() {
    let encoder = synthetic_encoder();
    let device = Default::default();
    let mut session = AudioSession::new(SessionId(41), &encoder, &device, Duration::ZERO).unwrap();
    let pcm = synthetic_pcm(6503);
    session
        .push(Duration::from_millis(1), &pcm[..3200], 16_000, false)
        .unwrap();
    let snapshot = session.snapshot().unwrap().unwrap();
    session
        .push(Duration::from_millis(2), &pcm[3200..], 16_000, false)
        .unwrap();
    let failed = session
        .start_generation(Duration::from_millis(1), snapshot, 24_000)
        .err()
        .expect("backwards start must fail");
    let (_, snapshot) = failed.into_parts();
    let failed = session
        .start_generation(Duration::from_millis(3), snapshot, 0)
        .err()
        .expect("zero output rate must fail");
    let (_, snapshot) = failed.into_parts();
    let active = session
        .start_generation(
            Duration::from_millis(4),
            session.snapshot().unwrap().unwrap(),
            24_000,
        )
        .unwrap();
    let failed = session
        .start_generation(Duration::from_millis(5), snapshot, 24_000)
        .err()
        .expect("active turn must reject another start");
    let (_, same_snapshot) = failed.into_parts();
    assert_eq!(same_snapshot.input_end(), 3200);
    session
        .apply_turn(
            Duration::from_millis(6),
            &active,
            TurnEvent::Stop {
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    session
        .apply_turn(
            Duration::from_millis(7),
            &active,
            TurnEvent::Stopped { played_end: 0 },
        )
        .unwrap();
    let started = session
        .start_generation(Duration::from_millis(8), same_snapshot, 24_000)
        .unwrap();
    assert_eq!(started.input_end(), 3200);
    assert!(started.media().is_ok());
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
