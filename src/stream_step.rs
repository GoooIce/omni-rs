//! Session owns chunking and decode turns for both sequential and grouped work.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    NeedsAudio,
    Ready,
    Complete,
}
pub(crate) struct Advance<T> {
    pub computed: usize,
    pub results: Vec<Result<Option<T>>>,
}
impl<B: Backend> Session<B> {
    pub(crate) fn phase(&self) -> Phase {
        if self.finished {
            Phase::Complete
        } else if self.turn.is_some()
            || self.input_ended
            || self.pending.len() >= self.chunk_samples()
        {
            Phase::Ready
        } else {
            Phase::NeedsAudio
        }
    }
    fn chunk_samples(&self) -> usize {
        (self.config.chunk_ms
            + if self.first {
                self.config.lookahead_ms
            } else {
                0
            })
            * (SAMPLE_RATE / 1000)
    }
    pub(crate) fn receive(&mut self, pcm: &[f32], end: bool) -> Result<()> {
        if self.finished || self.input_ended {
            return Err(Error::Finished);
        }
        self.check_stop()?;
        if pcm.iter().any(|v| !v.is_finite()) {
            return Err(Error::Audio("PCM contains NaN or infinity".into()));
        }
        self.pending.extend(pcm);
        self.input_ended = end;
        Ok(())
    }
    pub(super) async fn drain(&mut self) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        while self.phase() == Phase::Ready {
            match Self::advance_many(&mut [self], 256)
                .await
                .results
                .pop()
                .unwrap()
            {
                Ok(Some(event)) => events.push(event),
                Ok(None) => {}
                Err(Error::Interrupted) if !events.is_empty() => return Ok(events),
                Err(error) => return Err(error),
            }
        }
        Ok(events)
    }
    fn prepare_turn(&mut self) -> Result<Option<StreamEvent>> {
        self.check_stop()?;
        if self.turn.is_some() || self.phase() != Phase::Ready {
            return Ok(None);
        }
        let needed = self.chunk_samples();
        let final_chunk = self.input_ended && self.pending.len() < needed;
        let count = needed.min(self.pending.len());
        let chunk: Vec<_> = self.pending.drain(..count).collect();
        self.append(&chunk)?;
        if final_chunk && self.consumed == 0 {
            self.close_decode();
            return Ok(Some(StreamEvent {
                delta: String::new(),
                text: String::new(),
                language: String::new(),
                audio_seconds: 0.0,
                compute_ms: 0.0,
                generated_tokens: 0,
                reused_audio_tokens: 0,
                is_final: true,
                token_limit_reached: false,
            }));
        }
        self.turn = Some(self.begin_decode(final_chunk)?);
        Ok(None)
    }
    fn close_decode(&mut self) {
        self.finished = true;
        self.turn = None;
        self.pending.clear();
        self.encoder_cache.clear();
        self.decoder_cache.clear();
    }
    fn commit_turn(&mut self, turn: DecodeTurn<B>) -> Result<StreamEvent> {
        let final_chunk = turn.final_chunk;
        let event = self.commit_decode(turn)?;
        self.first = false;
        if final_chunk {
            self.close_decode();
        }
        Ok(event)
    }
    pub(crate) async fn advance_many(
        sessions: &mut [&mut Self],
        budget: usize,
    ) -> Advance<StreamEvent> {
        let mut results: Vec<_> = (0..sessions.len()).map(|_| Ok(None)).collect();
        if sessions.is_empty() {
            return Advance {
                computed: 0,
                results,
            };
        }
        let model = sessions[0].model.clone();
        if budget < sessions.len() || sessions.iter().any(|s| !Arc::ptr_eq(&s.model, &model)) {
            return Advance {
                computed: 0,
                results: sessions
                    .iter()
                    .map(|_| {
                        Err(Error::Config(
                            "decode group requires one model and a token budget per member".into(),
                        ))
                    })
                    .collect(),
            };
        }
        let mut ready = Vec::new();
        for (i, session) in sessions.iter_mut().enumerate() {
            match session.prepare_turn() {
                Ok(Some(event)) => results[i] = Ok(Some(event)),
                Ok(None) if session.turn.is_some() => ready.push(i),
                Ok(None) => {}
                Err(error) => results[i] = Err(error),
            }
        }
        let chunk = budget / ready.len().max(1);
        let mut inputs = Vec::new();
        let mut rows = Vec::new();
        for i in ready {
            match sessions[i]
                .turn
                .as_mut()
                .unwrap()
                .next_input(model.decoder(), chunk)
            {
                Ok((input, sample)) => {
                    inputs.push(input);
                    rows.push((i, sample));
                }
                Err(error) => results[i] = Err(error),
            }
        }
        let computed = rows.len();
        if rows.is_empty() {
            return Advance { computed, results };
        }
        let forward = if rows.len() == 1 {
            model.decoder().forward(
                inputs.pop().unwrap(),
                &mut sessions[rows[0].0].decoder_cache,
            )
        } else {
            let mut caches: Vec<_> = sessions
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| rows.iter().any(|(row, _)| row == i))
                .map(|(_, s)| &mut s.decoder_cache)
                .collect();
            model.decoder().forward_batch(inputs, &mut caches)
        };
        let tokens = async {
            let logits = forward?;
            if rows.iter().all(|(_, sample)| !sample) {
                return Ok(Vec::new());
            }
            logits
                .argmax(1)
                .into_data_async()
                .await
                .map_err(|e| Error::Inference(format!("GPU readback: {e:?}")))?
                .convert::<i32>()
                .to_vec::<i32>()
                .map_err(|e| Error::Inference(format!("invalid token readback: {e:?}")))
        }
        .await;
        let tokens = match tokens {
            Ok(tokens) => tokens,
            Err(error) => {
                #[cfg(feature = "native")]
                let _ = model.decoder().synchronize();
                let message = error.to_string();
                for (i, _) in rows {
                    sessions[i].close_decode();
                    results[i] = Err(Error::Inference(message.clone()));
                }
                return Advance { computed, results };
            }
        };
        for (row, &(i, sample)) in rows.iter().enumerate() {
            let session = &mut sessions[i];
            if let Err(error) = session.check_stop() {
                results[i] = Err(error);
                continue;
            }
            if !sample {
                continue;
            }
            let mut turn = session.turn.take().unwrap();
            results[i] = match session.accept_token(&mut turn, tokens[row] as u32) {
                Ok(true) => session.commit_turn(turn).map(Some),
                Ok(false) => {
                    session.turn = Some(turn);
                    Ok(None)
                }
                Err(error) => Err(error),
            };
        }
        Advance { computed, results }
    }
}
