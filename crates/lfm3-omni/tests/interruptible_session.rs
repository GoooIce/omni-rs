use std::{collections::VecDeque, time::Duration};

use omni_lfm3::interruptible::{
    Conversation, Error, Event, Outcome, SessionId, State, StopReason, TimelineEvent,
};

struct FakeClock;

impl FakeClock {
    fn at(milliseconds: u64) -> Duration {
        Duration::from_millis(milliseconds)
    }
}

#[derive(Default)]
struct FakePlayback {
    queued: VecDeque<i16>,
    heard: Vec<i16>,
    played: u64,
    fail_next_enqueue: bool,
}

impl FakePlayback {
    fn enqueue(&mut self, samples: &[i16]) {
        self.queued.extend(samples);
    }

    fn try_enqueue(&mut self, samples: &[i16]) -> Result<(), &'static str> {
        if std::mem::take(&mut self.fail_next_enqueue) {
            return Err("queue unavailable");
        }
        self.enqueue(samples);
        Ok(())
    }

    fn render(&mut self, count: usize) {
        for _ in 0..count {
            self.heard.push(self.queued.pop_front().unwrap());
            self.played += 1;
        }
    }

    fn stop_and_clear(&mut self) -> u64 {
        self.queued.clear();
        let played = self.played;
        self.played = 0;
        played
    }
}

#[test]
fn generation_can_only_use_an_available_input_prefix() {
    let session = SessionId(7);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    conversation
        .apply(
            FakeClock::at(10),
            Event::InputAvailable { session, end: 320 },
        )
        .unwrap();

    let future = conversation.apply(
        FakeClock::at(20),
        Event::StartGeneration {
            session,
            input_end: 321,
            output_rate: 24_000,
        },
    );
    assert_eq!(future, Err(Error::InputBeyondAvailable));

    let started = conversation
        .apply(
            FakeClock::at(21),
            Event::StartGeneration {
                session,
                input_end: 240,
                output_rate: 24_000,
            },
        )
        .unwrap();
    let Outcome::Started(turn) = started else {
        panic!("starting generation must return its turn token");
    };
    assert!(matches!(
        conversation.timeline().last().unwrap().event,
        TimelineEvent::GenerationStarted {
            turn: recorded,
            input_end: 240,
            output_rate: 24_000,
        } if recorded == turn
    ));
}

#[test]
fn interruption_preserves_rendered_audio_and_clears_the_tail_before_resuming() {
    let session = SessionId(7);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    let mut playback = FakePlayback::default();
    conversation
        .apply(FakeClock::at(1), Event::InputAvailable { session, end: 8 })
        .unwrap();
    let Outcome::Started(turn) = conversation
        .apply(
            FakeClock::at(2),
            Event::StartGeneration {
                session,
                input_end: 8,
                output_rate: 24_000,
            },
        )
        .unwrap()
    else {
        panic!("expected a turn");
    };
    conversation
        .apply(FakeClock::at(3), Event::Generated { turn, end: 6 })
        .unwrap();
    playback.enqueue(&[10, 11, 12, 13, 14, 15]);
    conversation
        .apply(FakeClock::at(4), Event::Queued { turn, end: 6 })
        .unwrap();
    playback.render(2);
    conversation
        .apply(FakeClock::at(5), Event::Played { turn, end: 2 })
        .unwrap();

    assert_eq!(
        conversation
            .apply(
                FakeClock::at(6),
                Event::Stop {
                    turn,
                    reason: StopReason::Interrupted { input_end: 8 },
                },
            )
            .unwrap(),
        Outcome::Stop(turn)
    );
    assert!(matches!(conversation.state(), State::Stopping { .. }));
    conversation
        .apply(FakeClock::at(7), Event::InputAvailable { session, end: 12 })
        .unwrap();
    // A callback already in flight can render more before the stop barrier.
    playback.render(1);
    let played_end = playback.stop_and_clear();
    conversation
        .apply(FakeClock::at(8), Event::Stopped { turn, played_end })
        .unwrap();
    assert_eq!(conversation.state(), State::Listening);
    assert!(conversation.timeline().iter().any(|entry| matches!(
        entry.event,
        TimelineEvent::PlaybackStopped {
            turn: recorded,
            played_end: 3,
            queued_end: 6,
            generated_end: 6,
        } if recorded == turn
    )));

    let Outcome::Started(next) = conversation
        .apply(
            FakeClock::at(9),
            Event::StartGeneration {
                session,
                input_end: 12,
                output_rate: 24_000,
            },
        )
        .unwrap()
    else {
        panic!("expected a resumed turn");
    };
    conversation
        .apply(FakeClock::at(10), Event::Generated { turn: next, end: 2 })
        .unwrap();
    playback.enqueue(&[20, 21]);
    conversation
        .apply(FakeClock::at(11), Event::Queued { turn: next, end: 2 })
        .unwrap();
    playback.render(2);
    conversation
        .apply(FakeClock::at(12), Event::Played { turn: next, end: 2 })
        .unwrap();
    conversation
        .apply(FakeClock::at(13), Event::GenerationFinished { turn: next })
        .unwrap();
    assert_eq!(conversation.state(), State::Listening);
    assert_eq!(playback.heard, [10, 11, 12, 20, 21]);
}

#[test]
fn session_switch_waits_for_stop_and_rejects_old_callbacks_and_prefixes() {
    let old = SessionId(7);
    let new = SessionId(8);
    let mut conversation = Conversation::new(old, 16_000, FakeClock::at(0)).unwrap();
    conversation
        .apply(
            FakeClock::at(1),
            Event::InputAvailable {
                session: old,
                end: 8,
            },
        )
        .unwrap();
    let Outcome::Started(turn) = conversation
        .apply(
            FakeClock::at(2),
            Event::StartGeneration {
                session: old,
                input_end: 8,
                output_rate: 24_000,
            },
        )
        .unwrap()
    else {
        panic!("expected a turn");
    };
    assert_eq!(
        conversation.apply(
            FakeClock::at(3),
            Event::ChangeSession {
                session: new,
                input_rate: 16_000
            },
        ),
        Err(Error::InvalidState)
    );
    conversation
        .apply(
            FakeClock::at(4),
            Event::Stop {
                turn,
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    conversation
        .apply(
            FakeClock::at(5),
            Event::Stopped {
                turn,
                played_end: 0,
            },
        )
        .unwrap();
    conversation
        .apply(
            FakeClock::at(6),
            Event::ChangeSession {
                session: new,
                input_rate: 16_000,
            },
        )
        .unwrap();

    assert_eq!(
        conversation.apply(
            FakeClock::at(7),
            Event::InputAvailable {
                session: old,
                end: 9
            }
        ),
        Err(Error::WrongSession)
    );
    assert_eq!(
        conversation.apply(FakeClock::at(7), Event::Played { turn, end: 0 }),
        Err(Error::StaleTurn)
    );
    assert_eq!(
        conversation.apply(
            FakeClock::at(7),
            Event::StartGeneration {
                session: new,
                input_end: 1,
                output_rate: 24_000
            },
        ),
        Err(Error::InputBeyondAvailable)
    );
    assert_eq!(
        conversation.apply(
            FakeClock::at(7),
            Event::ChangeSession {
                session: old,
                input_rate: 16_000
            },
        ),
        Err(Error::SessionReused)
    );
    assert!(matches!(
        conversation.timeline().last().unwrap(),
        entry if entry.session == new
            && entry.event == TimelineEvent::SessionChanged { previous: old, input_rate: 16_000 }
    ));
}

#[test]
fn a_backwards_clock_cannot_advance_the_input_or_the_timeline() {
    let session = SessionId(3);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(10)).unwrap();
    conversation
        .apply(FakeClock::at(20), Event::InputAvailable { session, end: 8 })
        .unwrap();
    assert_eq!(
        conversation.apply(
            FakeClock::at(19),
            Event::InputAvailable { session, end: 100 }
        ),
        Err(Error::TimeWentBackwards)
    );
    assert_eq!(conversation.timeline().len(), 2);
    assert_eq!(
        conversation.apply(
            FakeClock::at(20),
            Event::StartGeneration {
                session,
                input_end: 100,
                output_rate: 24_000
            },
        ),
        Err(Error::InputBeyondAvailable)
    );
}

#[test]
fn impossible_sample_boundaries_cannot_advance_or_rewind_playback() {
    let session = SessionId(3);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    conversation
        .apply(FakeClock::at(1), Event::InputAvailable { session, end: 8 })
        .unwrap();
    assert_eq!(
        conversation.apply(FakeClock::at(2), Event::InputAvailable { session, end: 7 }),
        Err(Error::InvalidSampleBoundary)
    );
    let Outcome::Started(turn) = conversation
        .apply(
            FakeClock::at(3),
            Event::StartGeneration {
                session,
                input_end: 8,
                output_rate: 24_000,
            },
        )
        .unwrap()
    else {
        panic!("expected a turn");
    };
    conversation
        .apply(FakeClock::at(4), Event::Generated { turn, end: 8 })
        .unwrap();
    assert_eq!(
        conversation.apply(FakeClock::at(5), Event::Queued { turn, end: 9 }),
        Err(Error::InvalidSampleBoundary)
    );
    conversation
        .apply(FakeClock::at(6), Event::Queued { turn, end: 5 })
        .unwrap();
    assert_eq!(
        conversation.apply(FakeClock::at(7), Event::Played { turn, end: 6 }),
        Err(Error::InvalidSampleBoundary)
    );
    conversation
        .apply(FakeClock::at(8), Event::Played { turn, end: 3 })
        .unwrap();
    assert_eq!(
        conversation.apply(FakeClock::at(9), Event::Played { turn, end: 2 }),
        Err(Error::InvalidSampleBoundary)
    );
    conversation
        .apply(
            FakeClock::at(10),
            Event::Stop {
                turn,
                reason: StopReason::Cancelled,
            },
        )
        .unwrap();
    assert_eq!(
        conversation.apply(
            FakeClock::at(11),
            Event::Stopped {
                turn,
                played_end: 2
            }
        ),
        Err(Error::InvalidSampleBoundary)
    );
    assert_eq!(
        conversation.apply(
            FakeClock::at(11),
            Event::Stopped {
                turn,
                played_end: 6
            }
        ),
        Err(Error::InvalidSampleBoundary)
    );
    conversation
        .apply(
            FakeClock::at(12),
            Event::Stopped {
                turn,
                played_end: 4,
            },
        )
        .unwrap();
    assert!(matches!(
        conversation.timeline().last().unwrap().event,
        TimelineEvent::PlaybackStopped {
            played_end: 4,
            queued_end: 5,
            generated_end: 8,
            ..
        }
    ));
}

#[test]
fn late_callbacks_cannot_cross_a_stop_barrier_or_corrupt_the_next_turn() {
    let session = SessionId(1);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    conversation
        .apply(FakeClock::at(1), Event::InputAvailable { session, end: 4 })
        .unwrap();
    let start = Event::StartGeneration {
        session,
        input_end: 4,
        output_rate: 24_000,
    };
    let Outcome::Started(old) = conversation.apply(FakeClock::at(2), start).unwrap() else {
        panic!("expected a turn");
    };
    conversation
        .apply(FakeClock::at(3), Event::Generated { turn: old, end: 4 })
        .unwrap();
    conversation
        .apply(FakeClock::at(4), Event::Queued { turn: old, end: 4 })
        .unwrap();
    conversation
        .apply(
            FakeClock::at(5),
            Event::Stop {
                turn: old,
                reason: StopReason::Interrupted { input_end: 4 },
            },
        )
        .unwrap();
    assert_eq!(
        conversation.apply(FakeClock::at(6), start),
        Err(Error::InvalidState)
    );
    assert_eq!(
        conversation.apply(FakeClock::at(6), Event::Generated { turn: old, end: 5 }),
        Err(Error::InvalidState)
    );
    conversation
        .apply(FakeClock::at(7), Event::Played { turn: old, end: 2 })
        .unwrap();
    conversation
        .apply(
            FakeClock::at(8),
            Event::Stopped {
                turn: old,
                played_end: 2,
            },
        )
        .unwrap();
    let Outcome::Started(next) = conversation.apply(FakeClock::at(9), start).unwrap() else {
        panic!("expected the next turn");
    };
    assert_ne!(old, next);
    let count = conversation.timeline().len();
    assert_eq!(
        conversation.apply(FakeClock::at(10), Event::Played { turn: old, end: 3 }),
        Err(Error::StaleTurn)
    );
    assert_eq!(
        conversation.apply(FakeClock::at(10), Event::Generated { turn: old, end: 5 }),
        Err(Error::StaleTurn)
    );
    assert_eq!(conversation.timeline().len(), count);
    assert_eq!(conversation.state(), State::Responding { turn: next });
}

#[test]
fn a_user_pause_is_recorded_at_an_available_input_boundary() {
    let session = SessionId(2);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    conversation
        .apply(FakeClock::at(1), Event::InputAvailable { session, end: 10 })
        .unwrap();
    assert_eq!(
        conversation.apply(
            FakeClock::at(2),
            Event::UserPaused {
                session,
                input_end: 11
            }
        ),
        Err(Error::InputBeyondAvailable)
    );
    conversation
        .apply(
            FakeClock::at(3),
            Event::UserPaused {
                session,
                input_end: 8,
            },
        )
        .unwrap();
    assert_eq!(
        conversation.timeline().last().unwrap().event,
        TimelineEvent::UserPaused { input_end: 8 }
    );
}

#[test]
fn zero_sample_rates_are_rejected_without_changing_the_session() {
    let session = SessionId(2);
    assert!(matches!(
        Conversation::new(session, 0, FakeClock::at(0)),
        Err(Error::InvalidSampleRate)
    ));
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    let next = SessionId(3);
    assert_eq!(
        conversation.apply(
            FakeClock::at(1),
            Event::ChangeSession {
                session: next,
                input_rate: 0
            }
        ),
        Err(Error::InvalidSampleRate)
    );
    assert_eq!(
        conversation.apply(
            FakeClock::at(1),
            Event::StartGeneration {
                session,
                input_end: 0,
                output_rate: 0
            },
        ),
        Err(Error::InvalidSampleRate)
    );
    conversation
        .apply(
            FakeClock::at(2),
            Event::ChangeSession {
                session: next,
                input_rate: 16_000,
            },
        )
        .unwrap();
}

#[test]
fn failed_playback_queue_can_be_stopped_and_the_session_reused() {
    let session = SessionId(4);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    let mut playback = FakePlayback {
        fail_next_enqueue: true,
        ..FakePlayback::default()
    };
    let start = Event::StartGeneration {
        session,
        input_end: 0,
        output_rate: 24_000,
    };
    let Outcome::Started(turn) = conversation.apply(FakeClock::at(1), start).unwrap() else {
        panic!("expected a turn");
    };
    conversation
        .apply(FakeClock::at(2), Event::Generated { turn, end: 3 })
        .unwrap();
    assert_eq!(playback.try_enqueue(&[1, 2, 3]), Err("queue unavailable"));
    assert_eq!(
        conversation.apply(
            FakeClock::at(3),
            Event::Stop {
                turn,
                reason: StopReason::PlaybackFailed
            },
        ),
        Ok(Outcome::Stop(turn))
    );
    let played_end = playback.stop_and_clear();
    conversation
        .apply(FakeClock::at(4), Event::Stopped { turn, played_end })
        .unwrap();
    assert!(matches!(
        conversation.timeline().last().unwrap().event,
        TimelineEvent::PlaybackStopped {
            generated_end: 3,
            queued_end: 0,
            played_end: 0,
            ..
        }
    ));
    assert!(matches!(
        conversation.apply(FakeClock::at(5), start),
        Ok(Outcome::Started(_))
    ));
}

#[test]
fn interruption_records_its_input_boundary_without_claiming_future_samples() {
    let session = SessionId(5);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    conversation
        .apply(FakeClock::at(1), Event::InputAvailable { session, end: 9 })
        .unwrap();
    let Outcome::Started(turn) = conversation
        .apply(
            FakeClock::at(2),
            Event::StartGeneration {
                session,
                input_end: 8,
                output_rate: 24_000,
            },
        )
        .unwrap()
    else {
        panic!("expected a turn");
    };
    assert_eq!(
        conversation.apply(
            FakeClock::at(3),
            Event::Stop {
                turn,
                reason: StopReason::Interrupted { input_end: 10 }
            },
        ),
        Err(Error::InputBeyondAvailable)
    );
    let reason = StopReason::Interrupted { input_end: 9 };
    conversation
        .apply(FakeClock::at(4), Event::Stop { turn, reason })
        .unwrap();
    assert_eq!(
        conversation.timeline().last().unwrap().event,
        TimelineEvent::StopRequested { turn, reason }
    );
}

#[test]
fn completed_generation_remains_active_until_its_whole_output_is_played() {
    let session = SessionId(6);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    let Outcome::Started(turn) = conversation
        .apply(
            FakeClock::at(1),
            Event::StartGeneration {
                session,
                input_end: 0,
                output_rate: 24_000,
            },
        )
        .unwrap()
    else {
        panic!("expected a turn");
    };
    conversation
        .apply(FakeClock::at(2), Event::Generated { turn, end: 6 })
        .unwrap();
    conversation
        .apply(FakeClock::at(3), Event::Queued { turn, end: 2 })
        .unwrap();
    conversation
        .apply(FakeClock::at(4), Event::GenerationFinished { turn })
        .unwrap();
    assert_eq!(conversation.state(), State::Responding { turn });
    conversation
        .apply(FakeClock::at(5), Event::Queued { turn, end: 6 })
        .unwrap();
    conversation
        .apply(FakeClock::at(6), Event::Played { turn, end: 2 })
        .unwrap();
    assert_eq!(conversation.state(), State::Responding { turn });
    conversation
        .apply(FakeClock::at(7), Event::Played { turn, end: 6 })
        .unwrap();
    assert_eq!(conversation.state(), State::Listening);
    assert_eq!(
        conversation.timeline().last().unwrap().event,
        TimelineEvent::TurnCompleted { turn }
    );
}

#[test]
fn generation_failure_stops_queued_output_and_is_distinguishable_in_the_timeline() {
    let session = SessionId(6);
    let mut conversation = Conversation::new(session, 16_000, FakeClock::at(0)).unwrap();
    let mut playback = FakePlayback::default();
    let Outcome::Started(turn) = conversation
        .apply(
            FakeClock::at(1),
            Event::StartGeneration {
                session,
                input_end: 0,
                output_rate: 24_000,
            },
        )
        .unwrap()
    else {
        panic!("expected a turn");
    };
    conversation
        .apply(FakeClock::at(2), Event::Generated { turn, end: 4 })
        .unwrap();
    playback.enqueue(&[1, 2, 3, 4]);
    conversation
        .apply(FakeClock::at(3), Event::Queued { turn, end: 4 })
        .unwrap();
    playback.render(1);
    conversation
        .apply(
            FakeClock::at(4),
            Event::Stop {
                turn,
                reason: StopReason::GenerationFailed,
            },
        )
        .unwrap();
    let played_end = playback.stop_and_clear();
    conversation
        .apply(FakeClock::at(5), Event::Stopped { turn, played_end })
        .unwrap();
    assert!(conversation.timeline().iter().any(|entry| matches!(
        entry.event,
        TimelineEvent::StopRequested { turn: recorded, reason: StopReason::GenerationFailed }
            if recorded == turn
    )));
    assert_eq!(playback.heard, [1]);
    assert_eq!(conversation.state(), State::Listening);
}
