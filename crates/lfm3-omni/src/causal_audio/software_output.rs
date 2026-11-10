//! A single-owner software output queue for a causal audio session.

use std::{
    collections::VecDeque,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use burn::tensor::backend::Backend;

use super::{
    AudioSession, AudioSnapshot, AudioTurn, Cancellation, Error as SessionError, TurnEvent,
};
use crate::interruptible::{Conversation, SessionId, State, StopReason, TurnId};
use crate::prepared::PreparedMedia;

/// Maximum number of samples retained by the software queue and playback loan.
pub const MAX_OUTPUT_SAMPLES: usize = 24_000 * 30;

#[derive(Debug)]
pub enum Error {
    Session(SessionError),
    InvalidCapacity,
    InvalidState,
    InvalidSamples,
    TimeWentBackwards,
    WrongProducer,
    QueueFull,
    CallbackOutstanding,
    WrongLoan,
    BarrierPending,
    Allocation,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "software output: {self:?}")
    }
}

impl std::error::Error for Error {}

/// A failed start retains the original snapshot so the caller can retry it.
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
            .finish_non_exhaustive()
    }
}

/// An instance-bound token for the producer of one turn.
#[derive(Clone, Debug)]
pub struct Producer {
    identity: Arc<ProducerIdentity>,
}

#[derive(Debug)]
struct ProducerIdentity {
    owner: Arc<()>,
    turn: TurnId,
    input_end: u64,
    cancelled: AtomicBool,
}

impl Producer {
    pub fn is_cancelled(&self) -> bool {
        self.identity.cancelled.load(Ordering::Acquire)
    }

    pub fn turn(&self) -> TurnId {
        self.identity.turn
    }

    pub fn input_end(&self) -> u64 {
        self.identity.input_end
    }
}

/// Samples lent to one playback callback. Dropping it does not acknowledge
/// playback or release the owner's accounting.
///
/// ```compile_fail
/// use omni_lfm3::causal_audio::software_output::PlaybackLoan;
/// fn duplicate(loan: &PlaybackLoan) -> PlaybackLoan { loan.clone() }
/// ```
#[derive(Debug)]
pub struct PlaybackLoan {
    identity: Arc<ProducerIdentity>,
    serial: u64,
    samples: Vec<f32>,
}

impl PlaybackLoan {
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

/// Failed callback returns the same loan; dropping it leaves the barrier pending.
#[derive(Debug)]
pub struct ReturnError {
    error: Error,
    loan: PlaybackLoan,
}

impl ReturnError {
    pub fn into_parts(self) -> (Error, PlaybackLoan) {
        (self.error, self.loan)
    }
}

struct OutstandingLoan {
    serial: u64,
    len: usize,
}

struct Active<B: Backend> {
    turn: AudioTurn<B>,
    producer: Arc<ProducerIdentity>,
    generated: u64,
    queued: u64,
    played: u64,
    loan: Option<OutstandingLoan>,
    next_loan: u64,
    stopping: bool,
    quiesced: bool,
}

/// Owns the session and serializes its output events with the bounded queue.
pub struct StopBarrier<'a, B: Backend> {
    session: AudioSession<'a, B>,
    identity: Arc<()>,
    active: Option<Active<B>>,
    pending: VecDeque<f32>,
    capacity: usize,
    last_at: Duration,
}

impl<'a, B: Backend> StopBarrier<'a, B> {
    pub fn new(session: AudioSession<'a, B>, capacity: usize) -> Result<Self, Error> {
        if capacity == 0 || capacity > MAX_OUTPUT_SAMPLES {
            return Err(Error::InvalidCapacity);
        }
        if session.conversation().state() != State::Listening {
            return Err(Error::InvalidState);
        }
        let last_at = session
            .conversation()
            .timeline()
            .last()
            .expect("session start")
            .at;
        Ok(Self {
            session,
            identity: Arc::new(()),
            active: None,
            pending: VecDeque::new(),
            capacity,
            last_at,
        })
    }

    pub fn conversation(&self) -> &Conversation {
        self.session.conversation()
    }

    fn check_time(&self, at: Duration) -> Result<(), Error> {
        let session_at = self
            .conversation()
            .timeline()
            .last()
            .expect("session start")
            .at;
        if at < self.last_at || at < session_at {
            Err(Error::TimeWentBackwards)
        } else {
            Ok(())
        }
    }

    pub fn push(
        &mut self,
        at: Duration,
        samples: &[f32],
        rate: u32,
        final_chunk: bool,
    ) -> Result<usize, Error> {
        self.check_time(at)?;
        let tokens = self
            .session
            .push(at, samples, rate, final_chunk)
            .map_err(Error::Session)?;
        self.last_at = at;
        Ok(tokens)
    }

    pub fn snapshot(&self) -> Result<Option<AudioSnapshot<B>>, Error> {
        self.session.snapshot().map_err(Error::Session)
    }

    pub fn input_cancellation(&self) -> Cancellation {
        self.session.cancellation()
    }

    pub fn user_paused(&mut self, at: Duration, input_end: u64) -> Result<(), Error> {
        self.check_time(at)?;
        self.session
            .user_paused(at, input_end)
            .map_err(Error::Session)?;
        self.last_at = at;
        Ok(())
    }

    /// A session can change only after the active output barrier has finished.
    pub fn change_session(&mut self, at: Duration, session: SessionId) -> Result<(), Error> {
        self.check_time(at)?;
        if self.active.is_some() {
            return Err(Error::InvalidState);
        }
        self.session
            .change_session(at, session)
            .map_err(Error::Session)?;
        self.last_at = at;
        Ok(())
    }

    pub fn start(
        &mut self,
        at: Duration,
        snapshot: AudioSnapshot<B>,
        rate: u32,
    ) -> Result<Producer, StartError<B>> {
        if let Err(error) = self.check_time(at) {
            return Err(StartError {
                error,
                snapshot: Box::new(snapshot),
            });
        }
        let turn = match self.session.start_generation(at, snapshot, rate) {
            Ok(turn) => turn,
            Err(error) => {
                let (error, snapshot) = error.into_parts();
                return Err(StartError {
                    error: Error::Session(error),
                    snapshot: Box::new(snapshot),
                });
            }
        };
        let identity = Arc::new(ProducerIdentity {
            owner: Arc::clone(&self.identity),
            turn: turn.turn(),
            input_end: turn.input_end(),
            cancelled: AtomicBool::new(false),
        });
        self.active = Some(Active {
            turn,
            producer: Arc::clone(&identity),
            generated: 0,
            queued: 0,
            played: 0,
            loan: None,
            next_loan: 0,
            stopping: false,
            quiesced: false,
        });
        self.last_at = at;
        Ok(Producer { identity })
    }

    /// Borrow exactly the snapshot that started this producer's turn. Existing
    /// external clones cannot be revoked; producer quiescence remains required.
    pub fn media(&self, producer: &Producer) -> Result<&PreparedMedia<B>, Error> {
        let active = self.active.as_ref().ok_or(Error::InvalidState)?;
        if !Arc::ptr_eq(&active.producer, &producer.identity) {
            return Err(Error::WrongProducer);
        }
        active.turn.media().map_err(Error::Session)
    }

    pub fn generated(&mut self, at: Duration, producer: &Producer, end: u64) -> Result<(), Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::InvalidState)?;
        if !Arc::ptr_eq(&active.producer, &producer.identity) {
            return Err(Error::WrongProducer);
        }
        self.session
            .apply_turn(at, &active.turn, TurnEvent::Generated { end })
            .map_err(Error::Session)?;
        active.generated = end;
        self.last_at = at;
        Ok(())
    }

    /// The producer has finished normally; queued samples must still drain.
    pub fn generation_finished(&mut self, at: Duration, producer: &Producer) -> Result<(), Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::InvalidState)?;
        if !Arc::ptr_eq(&active.producer, &producer.identity) {
            return Err(Error::WrongProducer);
        }
        self.session
            .apply_turn(at, &active.turn, TurnEvent::GenerationFinished)
            .map_err(Error::Session)?;
        active.quiesced = true;
        self.last_at = at;
        if self.session.conversation().state() == State::Listening {
            self.active = None;
        }
        Ok(())
    }

    pub fn enqueue(
        &mut self,
        at: Duration,
        producer: &Producer,
        samples: &[f32],
    ) -> Result<(), Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::InvalidState)?;
        if !Arc::ptr_eq(&active.producer, &producer.identity) {
            return Err(Error::WrongProducer);
        }
        if samples.is_empty() {
            return Err(Error::InvalidSamples);
        }
        let count = u64::try_from(samples.len()).map_err(|_| Error::InvalidSamples)?;
        let end = active
            .queued
            .checked_add(count)
            .ok_or(Error::InvalidSamples)?;
        if end > active.generated {
            return Err(Error::InvalidSamples);
        }
        let loaned = active.loan.as_ref().map_or(0, |loan| loan.len);
        if self
            .pending
            .len()
            .checked_add(loaned)
            .and_then(|n| n.checked_add(samples.len()))
            .is_none_or(|n| n > self.capacity)
        {
            return Err(Error::QueueFull);
        }
        self.pending
            .try_reserve(samples.len())
            .map_err(|_| Error::Allocation)?;
        self.session
            .apply_turn(at, &active.turn, TurnEvent::Queued { end })
            .map_err(Error::Session)?;
        self.pending.extend(samples.iter().copied());
        active.queued = end;
        self.last_at = at;
        Ok(())
    }

    pub fn lend(&mut self, at: Duration, limit: usize) -> Result<Option<PlaybackLoan>, Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::InvalidState)?;
        if limit == 0 {
            return Err(Error::InvalidSamples);
        }
        if active.loan.is_some() {
            return Err(Error::CallbackOutstanding);
        }
        if active.stopping {
            return Err(Error::InvalidState);
        }
        let count = self.pending.len().min(limit);
        if count == 0 {
            self.last_at = at;
            return Ok(None);
        }
        let serial = active.next_loan;
        let next = serial.checked_add(1).ok_or(Error::InvalidState)?;
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        samples.extend(self.pending.drain(..count));
        active.loan = Some(OutstandingLoan { serial, len: count });
        active.next_loan = next;
        self.last_at = at;
        Ok(Some(PlaybackLoan {
            identity: Arc::clone(&active.producer),
            serial,
            samples,
        }))
    }

    fn enter_stopping(
        session: &mut AudioSession<'a, B>,
        active: &mut Active<B>,
        pending: &mut VecDeque<f32>,
        at: Duration,
        reason: StopReason,
    ) -> Result<(), Error> {
        session
            .apply_turn(at, &active.turn, TurnEvent::Stop { reason })
            .map_err(Error::Session)?;
        active.stopping = true;
        active.producer.cancelled.store(true, Ordering::Release);
        pending.clear();
        Ok(())
    }

    /// Stop admissions and request cooperative cancellation immediately.
    pub fn stop(
        &mut self,
        at: Duration,
        producer: &Producer,
        reason: StopReason,
    ) -> Result<(), Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::InvalidState)?;
        if !Arc::ptr_eq(&active.producer, &producer.identity) {
            return Err(Error::WrongProducer);
        }
        Self::enter_stopping(&mut self.session, active, &mut self.pending, at, reason)?;
        self.last_at = at;
        Ok(())
    }

    /// A failing callback starts the same stop barrier. The loan remains
    /// outstanding until its owner can return an authoritative boundary.
    pub fn playback_failed(&mut self, at: Duration, loan: &PlaybackLoan) -> Result<(), Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::WrongLoan)?;
        if !Arc::ptr_eq(&self.identity, &loan.identity.owner)
            || !Arc::ptr_eq(&active.producer, &loan.identity)
            || active.loan.as_ref().is_none_or(|outstanding| {
                outstanding.serial != loan.serial || outstanding.len != loan.samples.len()
            })
        {
            return Err(Error::WrongLoan);
        }
        if !active.stopping {
            Self::enter_stopping(
                &mut self.session,
                active,
                &mut self.pending,
                at,
                StopReason::PlaybackFailed,
            )?;
        }
        self.last_at = at;
        Ok(())
    }

    /// Assert that the external producer has stopped and cannot publish more.
    pub fn producer_quiesced(&mut self, at: Duration, producer: &Producer) -> Result<(), Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::InvalidState)?;
        if !Arc::ptr_eq(&active.producer, &producer.identity) {
            return Err(Error::WrongProducer);
        }
        if !active.stopping || active.quiesced {
            return Err(Error::InvalidState);
        }
        active.quiesced = true;
        self.last_at = at;
        Ok(())
    }

    /// Return the callback's authoritative played sample count for this loan.
    pub fn return_playback(
        &mut self,
        at: Duration,
        loan: PlaybackLoan,
        played: usize,
    ) -> Result<(), ReturnError> {
        if let Err(error) = self.check_time(at) {
            return Err(ReturnError { error, loan });
        }
        let Some(active) = self.active.as_mut() else {
            return Err(ReturnError {
                error: Error::WrongLoan,
                loan,
            });
        };
        if !Arc::ptr_eq(&self.identity, &loan.identity.owner)
            || !Arc::ptr_eq(&active.producer, &loan.identity)
            || active.loan.as_ref().is_none_or(|outstanding| {
                outstanding.serial != loan.serial || outstanding.len != loan.samples.len()
            })
        {
            return Err(ReturnError {
                error: Error::WrongLoan,
                loan,
            });
        }
        if played > loan.samples.len() {
            return Err(ReturnError {
                error: Error::InvalidSamples,
                loan,
            });
        }
        let Some(end) = u64::try_from(played)
            .ok()
            .and_then(|n| active.played.checked_add(n))
        else {
            return Err(ReturnError {
                error: Error::InvalidSamples,
                loan,
            });
        };
        if !active.stopping
            && self
                .pending
                .try_reserve(loan.samples.len() - played)
                .is_err()
        {
            return Err(ReturnError {
                error: Error::Allocation,
                loan,
            });
        }
        if let Err(error) = self
            .session
            .apply_turn(at, &active.turn, TurnEvent::Played { end })
        {
            return Err(ReturnError {
                error: Error::Session(error),
                loan,
            });
        }
        if !active.stopping {
            for &sample in loan.samples[played..].iter().rev() {
                self.pending.push_front(sample);
            }
        }
        active.played = end;
        active.loan = None;
        self.last_at = at;
        if self.session.conversation().state() == State::Listening {
            self.active = None;
        }
        Ok(())
    }

    /// Acknowledge stop only when producer and every loan are quiescent.
    pub fn finish_stop(&mut self, at: Duration, producer: &Producer) -> Result<(), Error> {
        self.check_time(at)?;
        let active = self.active.as_mut().ok_or(Error::InvalidState)?;
        if !Arc::ptr_eq(&active.producer, &producer.identity) {
            return Err(Error::WrongProducer);
        }
        if !active.stopping {
            return Err(Error::InvalidState);
        }
        if !active.quiesced || active.loan.is_some() {
            return Err(Error::BarrierPending);
        }
        self.session
            .apply_turn(
                at,
                &active.turn,
                TurnEvent::Stopped {
                    played_end: active.played,
                },
            )
            .map_err(Error::Session)?;
        self.active = None;
        self.last_at = at;
        Ok(())
    }
}
