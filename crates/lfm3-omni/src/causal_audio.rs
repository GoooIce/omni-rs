//! Session-owned PCM encoding and generation boundaries.
//!
//! The guarded interface owns both the PCM stream and reducer. It does not
//! load weights or manage a playback device or a producer stop barrier.

use std::{
    fmt,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use burn::tensor::backend::Backend;

use crate::{
    audio_encoder::AudioEncoder,
    input_stream::AudioInputStream,
    interruptible::{Conversation, Event, Outcome, SessionId, State, StopReason, TurnId},
    prepared::PreparedMedia,
};

#[derive(Debug)]
pub enum Error {
    Protocol(crate::interruptible::Error),
    Stream(omni_r2t2::Error),
    WrongStream,
    StreamUnavailable,
    ResourceLimit(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => error.fmt(f),
            Self::Stream(error) => error.fmt(f),
            Self::WrongStream => {
                f.write_str("snapshot or turn belongs to a different audio stream")
            }
            Self::StreamUnavailable => f.write_str("audio stream or turn is no longer available"),
            Self::ResourceLimit(reason) => write!(f, "audio session resource limit: {reason}"),
        }
    }
}

impl std::error::Error for Error {}

struct Scope {
    live: AtomicBool,
    cancelled: AtomicBool,
}

impl Scope {
    fn new() -> Self {
        Self {
            live: AtomicBool::new(true),
            cancelled: AtomicBool::new(false),
        }
    }

    fn available(&self) -> bool {
        self.live.load(Ordering::Acquire) && !self.cancelled.load(Ordering::Acquire)
    }
}

/// A monotonic cancellation handle for one input-stream instance. A handle
/// retained after a session change cannot cancel the new stream.
#[derive(Clone)]
pub struct Cancellation {
    scope: Arc<Scope>,
}

impl Cancellation {
    pub fn cancel(&self) {
        self.scope.cancelled.store(true, Ordering::Release);
    }
}

struct PoisonOnDrop {
    scope: Arc<Scope>,
    armed: bool,
}

impl PoisonOnDrop {
    fn new(scope: &Arc<Scope>) -> Self {
        Self {
            scope: Arc::clone(scope),
            armed: true,
        }
    }

    fn commit(&mut self) {
        self.armed = false;
    }
}

impl Drop for PoisonOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.scope.live.store(false, Ordering::Release);
        }
    }
}

/// An encoded snapshot and its automatically recorded input boundary.
///
/// `input_end` is the complete successfully ingested PCM prefix at creation;
/// it is a conservative bound, not the exact dependency end of every token.
/// The caller cannot change that boundary or construct a snapshot from media.
///
/// ```compile_fail
/// use burn::tensor::backend::Backend;
/// use omni_lfm3::causal_audio::AudioSnapshot;
/// fn relabel<B: Backend>(mut snapshot: AudioSnapshot<B>) {
///     snapshot.input_end = 1;
/// }
/// ```
pub struct AudioSnapshot<B: Backend> {
    session: SessionId,
    input_end: u64,
    media: PreparedMedia<B>,
    scope: Arc<Scope>,
    _lease: Arc<()>,
}

impl<B: Backend> AudioSnapshot<B> {
    pub fn session(&self) -> SessionId {
        self.session
    }

    pub fn input_end(&self) -> u64 {
        self.input_end
    }
}

/// A rejected start and the unchanged snapshot the caller may retry.
pub struct StartError<B: Backend> {
    error: Error,
    snapshot: Box<AudioSnapshot<B>>,
}

impl<B: Backend> StartError<B> {
    pub fn error(&self) -> &Error {
        &self.error
    }

    pub fn into_parts(self) -> (Error, AudioSnapshot<B>) {
        (self.error, *self.snapshot)
    }
}

impl<B: Backend> fmt::Debug for StartError<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StartError")
            .field("error", &self.error)
            .field("session", &self.snapshot.session)
            .field("input_end", &self.snapshot.input_end)
            .finish_non_exhaustive()
    }
}

impl<B: Backend> fmt::Display for StartError<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl<B: Backend> std::error::Error for StartError<B> {}

impl<B: Backend> From<StartError<B>> for Error {
    fn from(error: StartError<B>) -> Self {
        error.error
    }
}

/// A reducer turn paired with the exact encoded snapshot that started it.
pub struct AudioTurn<B: Backend> {
    turn: TurnId,
    snapshot: AudioSnapshot<B>,
    live: Arc<AtomicBool>,
}

impl<B: Backend> AudioTurn<B> {
    pub fn turn(&self) -> TurnId {
        self.turn
    }

    pub fn input_end(&self) -> u64 {
        self.snapshot.input_end
    }

    /// Borrow its original media while both stream and turn remain live.
    /// This cannot revoke tensors already borrowed or cloned by a producer;
    /// the adapter must cancel and quiesce that producer at its stop barrier.
    pub fn media(&self) -> Result<&PreparedMedia<B>, Error> {
        if !self.snapshot.scope.available() || !self.live.load(Ordering::Acquire) {
            return Err(Error::StreamUnavailable);
        }
        Ok(&self.snapshot.media)
    }
}

/// Instance-bound control without access to the producer's encoded media.
#[derive(Clone)]
pub struct AudioTurnControl {
    turn: TurnId,
    scope: Arc<Scope>,
    live: Arc<AtomicBool>,
}

/// Events needed to stop and drain a turn even if its media owner is lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlEvent {
    Played { end: u64 },
    Stop { reason: StopReason },
    Stopped { played_end: u64 },
}

/// An event for the particular, instance-bound token supplied to `apply_turn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnEvent {
    Generated { end: u64 },
    Queued { end: u64 },
    Played { end: u64 },
    GenerationFinished,
    Stop { reason: StopReason },
    Stopped { played_end: u64 },
}

pub const MAX_PUSH_SAMPLES: usize = 16_000 * 5;
pub const MAX_SESSION_SAMPLES: usize = 16_000 * 60 * 10;
pub const MAX_LIVE_SNAPSHOTS: usize = 8;

/// Per-session bounds; callers may choose tighter values but not raise them.
#[derive(Clone, Copy, Debug)]
pub struct AudioLimits {
    pub max_push_samples: usize,
    pub max_session_samples: usize,
    pub max_live_snapshots: usize,
}

impl Default for AudioLimits {
    fn default() -> Self {
        Self {
            max_push_samples: MAX_PUSH_SAMPLES,
            max_session_samples: MAX_SESSION_SAMPLES,
            max_live_snapshots: MAX_LIVE_SNAPSHOTS,
        }
    }
}

/// Owns a reducer and the PCM stream from which its snapshots are made.
///
/// Only this interface advances input, starts generation, or changes session;
/// `conversation()` is read-only. A later adapter must serialize its own
/// producer and playback operations with these transitions.
pub struct AudioSession<'a, B: Backend> {
    encoder: &'a AudioEncoder<B>,
    device: B::Device,
    stream: AudioInputStream<B>,
    conversation: Conversation,
    session: SessionId,
    scope: Arc<Scope>,
    turn_live: Option<Arc<AtomicBool>>,
    limits: AudioLimits,
    snapshots: Mutex<Vec<Weak<()>>>,
}

impl<'a, B: Backend> AudioSession<'a, B> {
    pub fn new(
        session: SessionId,
        encoder: &'a AudioEncoder<B>,
        device: &B::Device,
        at: Duration,
    ) -> Result<Self, Error> {
        Self::with_limits(session, encoder, device, at, AudioLimits::default())
    }

    pub fn with_limits(
        session: SessionId,
        encoder: &'a AudioEncoder<B>,
        device: &B::Device,
        at: Duration,
        limits: AudioLimits,
    ) -> Result<Self, Error> {
        if limits.max_push_samples == 0
            || limits.max_push_samples > MAX_PUSH_SAMPLES
            || limits.max_session_samples == 0
            || limits.max_session_samples > MAX_SESSION_SAMPLES
            || limits.max_push_samples > limits.max_session_samples
            || limits.max_live_snapshots == 0
            || limits.max_live_snapshots > MAX_LIVE_SNAPSHOTS
        {
            return Err(Error::ResourceLimit("invalid session limits"));
        }
        Ok(Self {
            encoder,
            device: device.clone(),
            stream: AudioInputStream::new(encoder, device),
            conversation: Conversation::new(session, 16_000, at).map_err(Error::Protocol)?,
            session,
            scope: Arc::new(Scope::new()),
            turn_live: None,
            limits,
            snapshots: Mutex::new(Vec::new()),
        })
    }

    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Recover control of the current turn without reconstructing media.
    /// This does not stop a producer or acknowledge the playback barrier.
    pub fn current_control(&self) -> Option<AudioTurnControl> {
        let turn = match self.conversation.state() {
            State::Listening => return None,
            State::Responding { turn } | State::Stopping { turn, .. } => turn,
        };
        Some(AudioTurnControl {
            turn,
            scope: Arc::clone(&self.scope),
            live: Arc::clone(self.turn_live.as_ref()?),
        })
    }

    /// Apply control only to the originating turn. `Stopped` is an explicit
    /// adapter assertion that its producer and playback barrier completed.
    pub fn apply_control(
        &mut self,
        at: Duration,
        control: &AudioTurnControl,
        event: ControlEvent,
    ) -> Result<Outcome, Error> {
        let event = match event {
            ControlEvent::Played { end } => TurnEvent::Played { end },
            ControlEvent::Stop { reason } => TurnEvent::Stop { reason },
            ControlEvent::Stopped { played_end } => TurnEvent::Stopped { played_end },
        };
        self.apply_bound(at, control, event)
    }

    /// Cancel this input stream permanently, including all its snapshots.
    pub fn cancellation(&self) -> Cancellation {
        Cancellation {
            scope: Arc::clone(&self.scope),
        }
    }

    /// Apply an event to this exact started turn. An accepted `Stop` or
    /// `GenerationFinished` revokes further media access for that turn only;
    /// input can continue during stopping. `Stopped` still requires the
    /// adapter's producer/playback barrier. Rejected events revoke nothing.
    pub fn apply_turn(
        &mut self,
        at: Duration,
        turn: &AudioTurn<B>,
        event: TurnEvent,
    ) -> Result<Outcome, Error> {
        let control = AudioTurnControl {
            turn: turn.turn,
            scope: Arc::clone(&turn.snapshot.scope),
            live: Arc::clone(&turn.live),
        };
        self.apply_bound(at, &control, event)
    }

    fn apply_bound(
        &mut self,
        at: Duration,
        control: &AudioTurnControl,
        event: TurnEvent,
    ) -> Result<Outcome, Error> {
        if !Arc::ptr_eq(&self.scope, &control.scope) {
            return Err(Error::WrongStream);
        }
        if !self
            .turn_live
            .as_ref()
            .is_some_and(|live| Arc::ptr_eq(live, &control.live))
        {
            return Err(Error::Protocol(crate::interruptible::Error::StaleTurn));
        }
        if !self.scope.available()
            && matches!(
                event,
                TurnEvent::Generated { .. }
                    | TurnEvent::Queued { .. }
                    | TurnEvent::GenerationFinished
            )
        {
            return Err(Error::StreamUnavailable);
        }
        let reducer_event = match event {
            TurnEvent::Generated { end } => Event::Generated {
                turn: control.turn,
                end,
            },
            TurnEvent::Queued { end } => Event::Queued {
                turn: control.turn,
                end,
            },
            TurnEvent::Played { end } => Event::Played {
                turn: control.turn,
                end,
            },
            TurnEvent::GenerationFinished => Event::GenerationFinished { turn: control.turn },
            TurnEvent::Stop { reason } => Event::Stop {
                turn: control.turn,
                reason,
            },
            TurnEvent::Stopped { played_end } => Event::Stopped {
                turn: control.turn,
                played_end,
            },
        };
        let outcome = self
            .conversation
            .apply(at, reducer_event)
            .map_err(Error::Protocol)?;
        if matches!(
            event,
            TurnEvent::Stop { .. } | TurnEvent::GenerationFinished | TurnEvent::Stopped { .. }
        ) {
            control.live.store(false, Ordering::Release);
        }
        if self.conversation.state() == State::Listening {
            self.turn_live = None;
        }
        Ok(outcome)
    }

    pub fn user_paused(&mut self, at: Duration, input_end: u64) -> Result<(), Error> {
        self.conversation
            .apply(
                at,
                Event::UserPaused {
                    session: self.session,
                    input_end,
                },
            )
            .map_err(Error::Protocol)?;
        Ok(())
    }

    /// Change only after the reducer is listening. A successful change drops
    /// the old stream and invalidates its snapshots, turns and cancel handle.
    pub fn change_session(&mut self, at: Duration, session: SessionId) -> Result<(), Error> {
        self.conversation
            .apply(
                at,
                Event::ChangeSession {
                    session,
                    input_rate: 16_000,
                },
            )
            .map_err(Error::Protocol)?;
        self.scope.live.store(false, Ordering::Release);
        self.stream = AudioInputStream::new(self.encoder, &self.device);
        self.session = session;
        self.scope = Arc::new(Scope::new());
        self.turn_live = None;
        Ok(())
    }

    /// Ingest mono 16 kHz PCM. Any stream error permanently invalidates this
    /// stream and its snapshots. A backwards event time is rejected before
    /// touching PCM and does not poison the stream.
    pub fn push(
        &mut self,
        at: Duration,
        samples: &[f32],
        sample_rate: u32,
        is_final: bool,
    ) -> Result<usize, Error> {
        if !self.scope.available() {
            // An asynchronous cancellation has no timestamp of its own. A
            // valid timed call does establish when it was observed, so later
            // events cannot be backdated before that observation.
            if self.conversation.check_time(at).is_ok() {
                self.conversation
                    .apply(
                        at,
                        Event::InputFailed {
                            session: self.session,
                        },
                    )
                    .expect("preflighted unavailable input must fit the timeline");
            }
            return Err(Error::StreamUnavailable);
        }
        self.conversation.check_time(at).map_err(Error::Protocol)?;
        if samples.len() > self.limits.max_push_samples {
            return Err(Error::ResourceLimit("input chunk is too large"));
        }
        if self
            .stream
            .samples()
            .checked_add(samples.len())
            .is_none_or(|total| total > self.limits.max_session_samples)
        {
            return Err(Error::ResourceLimit("session input is too long"));
        }
        let mut guard = PoisonOnDrop::new(&self.scope);
        let result = (|| {
            let before = self.stream.samples();
            let tokens = self
                .stream
                .push(
                    self.encoder,
                    samples,
                    sample_rate,
                    is_final,
                    &self.scope.cancelled,
                )
                .map_err(Error::Stream)?;
            if !self.scope.available() {
                return Err(Error::StreamUnavailable);
            }
            let after = self.stream.samples();
            if after > before {
                let end = u64::try_from(after).map_err(|_| {
                    Error::Stream(omni_r2t2::Error::Audio("audio length overflow".into()))
                })?;
                self.conversation
                    .apply(
                        at,
                        Event::InputAvailable {
                            session: self.session,
                            end,
                        },
                    )
                    .map_err(Error::Protocol)?;
            }
            if is_final {
                let end = u64::try_from(after).map_err(|_| {
                    Error::Stream(omni_r2t2::Error::Audio("audio length overflow".into()))
                })?;
                self.conversation
                    .apply(
                        at,
                        Event::InputFinalized {
                            session: self.session,
                            end,
                        },
                    )
                    .map_err(Error::Protocol)?;
            }
            Ok(tokens)
        })();
        match result {
            Ok(tokens) => {
                guard.commit();
                Ok(tokens)
            }
            Err(error) => {
                // The preflight established this session and time, and any
                // events appended during the attempted push use the same time.
                self.conversation
                    .apply(
                        at,
                        Event::InputFailed {
                            session: self.session,
                        },
                    )
                    .expect("preflighted input failure must fit the timeline");
                Err(error)
            }
        }
    }

    /// Capture the completed encoding at the current ingested prefix, without
    /// flushing the tail. `None` means that no complete token is available.
    pub fn snapshot(&self) -> Result<Option<AudioSnapshot<B>>, Error> {
        if !self.scope.available() {
            return Err(Error::StreamUnavailable);
        }
        if !self.stream.has_completed_tokens() {
            return if self.scope.available() {
                Ok(None)
            } else {
                Err(Error::StreamUnavailable)
            };
        }
        let mut snapshots = self
            .snapshots
            .lock()
            .map_err(|_| Error::StreamUnavailable)?;
        snapshots.retain(|lease| lease.strong_count() > 0);
        if snapshots.len() >= self.limits.max_live_snapshots {
            return Err(Error::ResourceLimit("too many live snapshots"));
        }
        let mut guard = PoisonOnDrop::new(&self.scope);
        let end = u64::try_from(self.stream.samples())
            .map_err(|_| Error::Stream(omni_r2t2::Error::Audio("audio length overflow".into())))?;
        let snapshot = self.stream.snapshot().map_err(Error::Stream)?.map(|media| {
            let lease = Arc::new(());
            snapshots.push(Arc::downgrade(&lease));
            AudioSnapshot {
                session: self.session,
                input_end: end,
                media,
                scope: Arc::clone(&self.scope),
                _lease: lease,
            }
        });
        if !self.scope.available() {
            return Err(Error::StreamUnavailable);
        }
        guard.commit();
        Ok(snapshot)
    }

    /// Start using exactly this instance's snapshot; no caller-supplied input
    /// boundary is accepted. An earlier snapshot remains usable after later
    /// input arrives as long as the stream is still live.
    ///
    /// ```compile_fail
    /// use burn::tensor::backend::Backend;
    /// use omni_lfm3::causal_audio::{AudioSession, AudioSnapshot};
    /// use std::time::Duration;
    /// fn relabel<B: Backend>(session: &mut AudioSession<'_, B>, snapshot: AudioSnapshot<B>) {
    ///     let _ = session.start_generation(Duration::ZERO, snapshot, 1, 24_000);
    /// }
    /// ```
    pub fn start_generation(
        &mut self,
        at: Duration,
        snapshot: AudioSnapshot<B>,
        output_rate: u32,
    ) -> Result<AudioTurn<B>, StartError<B>> {
        if !Arc::ptr_eq(&self.scope, &snapshot.scope) {
            return Err(StartError {
                error: Error::WrongStream,
                snapshot: Box::new(snapshot),
            });
        }
        if !self.scope.available() {
            return Err(StartError {
                error: Error::StreamUnavailable,
                snapshot: Box::new(snapshot),
            });
        }
        let live = Arc::new(AtomicBool::new(true));
        let outcome = match self.conversation.apply(
            at,
            Event::StartGeneration {
                session: snapshot.session,
                input_end: snapshot.input_end,
                output_rate,
            },
        ) {
            Ok(outcome) => outcome,
            Err(error) => {
                return Err(StartError {
                    error: Error::Protocol(error),
                    snapshot: Box::new(snapshot),
                });
            }
        };
        let Outcome::Started(turn) = outcome else {
            unreachable!("start generation returns its turn")
        };
        self.turn_live = Some(Arc::clone(&live));
        Ok(AudioTurn {
            turn,
            snapshot,
            live,
        })
    }
}

impl<B: Backend> Drop for AudioSession<'_, B> {
    fn drop(&mut self) {
        self.scope.live.store(false, Ordering::Release);
    }
}
