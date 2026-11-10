//! Device-independent conversation events and their audit timeline.
//!
//! This reducer does not access a clock, audio device, queue, model, or cache.
//! The caller serializes events with the corresponding adapter operations;
//! see `docs/interruptible-speech.md` for the stop barrier and sample domains.

use std::{collections::BTreeSet, fmt, time::Duration};

/// An identifier that must not be reused while old callbacks can still arrive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SessionId(pub u64);

/// A generation token returned to all producer and playback adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurnId {
    pub session: SessionId,
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InputBeyondAvailable,
    InvalidState,
    InvalidSampleBoundary,
    StaleTurn,
    WrongSession,
    SessionReused,
    TimeWentBackwards,
    InvalidSampleRate,
    TurnIdExhausted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "interruptible conversation: {self:?}")
    }
}

impl std::error::Error for Error {}

/// Why an active generation and its pending output must stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Interrupted { input_end: u64 },
    Cancelled,
    PlaybackFailed,
    GenerationFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Listening,
    Responding { turn: TurnId },
    Stopping { turn: TurnId, reason: StopReason },
}

/// Input to the reducer. Every `end` is an exclusive sample-frame boundary.
/// Input positions are session-local; output positions are turn-local and in
/// the output rate selected by `StartGeneration`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    InputAvailable {
        session: SessionId,
        end: u64,
    },
    /// Input finalization has flushed the tail without adding padded samples.
    InputFinalized {
        session: SessionId,
        end: u64,
    },
    /// The input stream failed and was invalidated at this event's time.
    InputFailed {
        session: SessionId,
    },
    UserPaused {
        session: SessionId,
        input_end: u64,
    },
    ChangeSession {
        session: SessionId,
        input_rate: u32,
    },
    StartGeneration {
        session: SessionId,
        input_end: u64,
        output_rate: u32,
    },
    Generated {
        turn: TurnId,
        end: u64,
    },
    Queued {
        turn: TurnId,
        end: u64,
    },
    Played {
        turn: TurnId,
        end: u64,
    },
    GenerationFinished {
        turn: TurnId,
    },
    Stop {
        turn: TurnId,
        reason: StopReason,
    },
    Stopped {
        turn: TurnId,
        played_end: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Applied,
    Started(TurnId),
    /// Cancel this producer, stop and clear its queue, then report `Stopped`.
    Stop(TurnId),
}

/// An accepted event at a monotonic caller-supplied time. Entries with equal
/// times are ordered by their position in `Conversation::timeline()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimelineEntry {
    pub at: Duration,
    pub session: SessionId,
    pub event: TimelineEvent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimelineEvent {
    SessionStarted {
        input_rate: u32,
    },
    SessionChanged {
        previous: SessionId,
        input_rate: u32,
    },
    InputAvailable {
        end: u64,
    },
    InputFinalized {
        end: u64,
    },
    InputFailed,
    UserPaused {
        input_end: u64,
    },
    GenerationStarted {
        turn: TurnId,
        input_end: u64,
        output_rate: u32,
    },
    Generated {
        turn: TurnId,
        start: u64,
        end: u64,
    },
    Queued {
        turn: TurnId,
        start: u64,
        end: u64,
    },
    Played {
        turn: TurnId,
        start: u64,
        end: u64,
    },
    GenerationFinished {
        turn: TurnId,
    },
    StopRequested {
        turn: TurnId,
        reason: StopReason,
    },
    /// Final played boundary and the queued/generated boundaries at the stop
    /// barrier. `[played_end, queued_end)` and `[queued_end, generated_end)`
    /// are the discarded queued and unqueued tails respectively.
    PlaybackStopped {
        turn: TurnId,
        played_end: u64,
        queued_end: u64,
        generated_end: u64,
    },
    TurnCompleted {
        turn: TurnId,
    },
}

struct ActiveTurn {
    turn: TurnId,
    generated: u64,
    queued: u64,
    played: u64,
    finished: bool,
    stopping: Option<StopReason>,
}

/// A session-scoped event reducer. The full timeline is retained in memory.
pub struct Conversation {
    session: SessionId,
    available: u64,
    next_turn: u64,
    sessions: BTreeSet<SessionId>,
    active: Option<ActiveTurn>,
    timeline: Vec<TimelineEntry>,
}

impl Conversation {
    /// Begin at an empty input prefix, with a nonzero input rate.
    pub fn new(session: SessionId, input_rate: u32, at: Duration) -> Result<Self, Error> {
        if input_rate == 0 {
            return Err(Error::InvalidSampleRate);
        }
        Ok(Self {
            session,
            available: 0,
            next_turn: 0,
            sessions: BTreeSet::from([session]),
            active: None,
            timeline: vec![TimelineEntry {
                at,
                session,
                event: TimelineEvent::SessionStarted { input_rate },
            }],
        })
    }

    /// Accepted events, in order; rejected events never alter this slice.
    pub fn timeline(&self) -> &[TimelineEntry] {
        &self.timeline
    }

    pub fn state(&self) -> State {
        match &self.active {
            None => State::Listening,
            Some(active) => match active.stopping {
                None => State::Responding { turn: active.turn },
                Some(reason) => State::Stopping {
                    turn: active.turn,
                    reason,
                },
            },
        }
    }

    fn active(&mut self, turn: TurnId) -> Result<&mut ActiveTurn, Error> {
        self.active
            .as_mut()
            .filter(|active| active.turn == turn)
            .ok_or(Error::StaleTurn)
    }

    /// Validate time before a caller performs an irreversible stream operation.
    pub(crate) fn check_time(&self, at: Duration) -> Result<(), Error> {
        if at
            < self
                .timeline
                .last()
                .expect("a session always has an entry")
                .at
        {
            return Err(Error::TimeWentBackwards);
        }
        Ok(())
    }

    /// Apply one event. The time must share one monotonic origin with all
    /// previous events. Rejection leaves state and timeline unchanged.
    pub fn apply(&mut self, at: Duration, event: Event) -> Result<Outcome, Error> {
        self.check_time(at)?;
        let (event, outcome) = match event {
            Event::InputAvailable { session, end } => {
                if session != self.session {
                    return Err(Error::WrongSession);
                }
                if end <= self.available {
                    return Err(Error::InvalidSampleBoundary);
                }
                self.available = end;
                (TimelineEvent::InputAvailable { end }, Outcome::Applied)
            }
            Event::InputFinalized { session, end } => {
                if session != self.session {
                    return Err(Error::WrongSession);
                }
                if end != self.available {
                    return Err(Error::InvalidSampleBoundary);
                }
                (TimelineEvent::InputFinalized { end }, Outcome::Applied)
            }
            Event::InputFailed { session } => {
                if session != self.session {
                    return Err(Error::WrongSession);
                }
                (TimelineEvent::InputFailed, Outcome::Applied)
            }
            Event::UserPaused { session, input_end } => {
                if session != self.session {
                    return Err(Error::WrongSession);
                }
                if input_end > self.available {
                    return Err(Error::InputBeyondAvailable);
                }
                (TimelineEvent::UserPaused { input_end }, Outcome::Applied)
            }
            Event::ChangeSession {
                session,
                input_rate,
            } => {
                if self.active.is_some() {
                    return Err(Error::InvalidState);
                }
                if input_rate == 0 {
                    return Err(Error::InvalidSampleRate);
                }
                if !self.sessions.insert(session) {
                    return Err(Error::SessionReused);
                }
                let previous = self.session;
                self.session = session;
                self.available = 0;
                (
                    TimelineEvent::SessionChanged {
                        previous,
                        input_rate,
                    },
                    Outcome::Applied,
                )
            }
            Event::StartGeneration {
                session,
                input_end,
                output_rate,
            } => {
                if session != self.session {
                    return Err(Error::WrongSession);
                }
                if self.active.is_some() {
                    return Err(Error::InvalidState);
                }
                if output_rate == 0 {
                    return Err(Error::InvalidSampleRate);
                }
                if input_end > self.available {
                    return Err(Error::InputBeyondAvailable);
                }
                let turn = TurnId {
                    session: self.session,
                    sequence: self.next_turn,
                };
                self.next_turn = self
                    .next_turn
                    .checked_add(1)
                    .ok_or(Error::TurnIdExhausted)?;
                self.active = Some(ActiveTurn {
                    turn,
                    generated: 0,
                    queued: 0,
                    played: 0,
                    finished: false,
                    stopping: None,
                });
                (
                    TimelineEvent::GenerationStarted {
                        turn,
                        input_end,
                        output_rate,
                    },
                    Outcome::Started(turn),
                )
            }
            Event::Generated { turn, end } => {
                let active = self.active(turn)?;
                if active.stopping.is_some() || active.finished {
                    return Err(Error::InvalidState);
                }
                if end <= active.generated {
                    return Err(Error::InvalidSampleBoundary);
                }
                let start = active.generated;
                active.generated = end;
                (
                    TimelineEvent::Generated { turn, start, end },
                    Outcome::Applied,
                )
            }
            Event::Queued { turn, end } => {
                let active = self.active(turn)?;
                if active.stopping.is_some() {
                    return Err(Error::InvalidState);
                }
                if end <= active.queued || end > active.generated {
                    return Err(Error::InvalidSampleBoundary);
                }
                let start = active.queued;
                active.queued = end;
                (TimelineEvent::Queued { turn, start, end }, Outcome::Applied)
            }
            Event::Played { turn, end } => {
                let active = self.active(turn)?;
                if end < active.played || end > active.queued {
                    return Err(Error::InvalidSampleBoundary);
                }
                let start = active.played;
                active.played = end;
                (TimelineEvent::Played { turn, start, end }, Outcome::Applied)
            }
            Event::GenerationFinished { turn } => {
                let active = self.active(turn)?;
                if active.finished || active.stopping.is_some() {
                    return Err(Error::InvalidState);
                }
                active.finished = true;
                (TimelineEvent::GenerationFinished { turn }, Outcome::Applied)
            }
            Event::Stop { turn, reason } => {
                if let StopReason::Interrupted { input_end } = reason
                    && input_end > self.available
                {
                    return Err(Error::InputBeyondAvailable);
                }
                let active = self.active(turn)?;
                if active.stopping.is_some() {
                    return Err(Error::InvalidState);
                }
                active.stopping = Some(reason);
                (
                    TimelineEvent::StopRequested { turn, reason },
                    Outcome::Stop(turn),
                )
            }
            Event::Stopped { turn, played_end } => {
                let active = self.active(turn)?;
                if active.stopping.is_none() {
                    return Err(Error::InvalidState);
                }
                if played_end < active.played || played_end > active.queued {
                    return Err(Error::InvalidSampleBoundary);
                }
                let event = TimelineEvent::PlaybackStopped {
                    turn,
                    played_end,
                    queued_end: active.queued,
                    generated_end: active.generated,
                };
                self.active = None;
                (event, Outcome::Applied)
            }
        };
        self.timeline.push(TimelineEntry {
            at,
            session: self.session,
            event,
        });
        if let Some(active) = &self.active
            && active.stopping.is_none()
            && active.finished
            && active.played == active.queued
            && active.queued == active.generated
        {
            self.timeline.push(TimelineEntry {
                at,
                session: self.session,
                event: TimelineEvent::TurnCompleted { turn: active.turn },
            });
            self.active = None;
        }
        Ok(outcome)
    }
}
