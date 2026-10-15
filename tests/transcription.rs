//! Exercise the shared lifecycle with a complete CPU model, through the same
//! interface native input adapters and WebAssembly use.
mod support;

use omni_r2t2::{
    Error, Session, StreamConfig, Transcription, TranscriptionEvent, audio::Resampler,
};
use std::sync::atomic::Ordering;

fn config() -> StreamConfig {
    StreamConfig {
        chunk_ms: 80,
        lookahead_ms: 0,
        max_new_tokens: 2,
        final_max_tokens: 4,
        max_context_tokens: 1024,
        ..Default::default()
    }
}

#[test]
#[cfg(feature = "native")]
fn batch_source_and_sink_failures_do_not_cancel_other_transcriptions() {
    use omni_r2t2::{input::AudioSource, stream::TranscriptionBatch};
    use std::{
        collections::BTreeMap,
        sync::{Arc, atomic::AtomicBool},
    };
    struct Source {
        fail: bool,
        ended: bool,
    }
    impl AudioSource for Source {
        fn sample_rate(&self) -> u32 {
            16000
        }
        fn next_packet(&mut self, stop: &AtomicBool) -> omni_r2t2::Result<Option<Vec<f32>>> {
            if self.fail {
                stop.store(true, Ordering::Release);
                return Err(Error::Audio("source failed while stopping".into()));
            }
            if self.ended {
                return Ok(None);
            }
            self.ended = true;
            Ok(Some(vec![0.1; 1280]))
        }
    }
    let mut engine = TranscriptionBatch::new(support::tiny_model(), 3).unwrap();
    let mut ids = Vec::new();
    for fail in [true, false, false] {
        ids.push(
            engine
                .try_submit(
                    config(),
                    Source { fail, ended: false },
                    Arc::new(AtomicBool::new(false)),
                )
                .unwrap()
                .unwrap(),
        );
    }
    let mut results = BTreeMap::new();
    while !engine.is_idle() {
        for (id, result) in engine.step(|id, _| {
            if id == ids[1] {
                Err(Error::Audio("sink failed".into()))
            } else {
                Ok(())
            }
        }) {
            assert!(results.insert(id, result).is_none());
        }
    }
    assert!(
        matches!(&results[&ids[0]],Err(Error::Audio(message)) if message=="source failed while stopping")
    );
    assert!(matches!(&results[&ids[1]],Err(Error::Audio(message)) if message=="sink failed"));
    assert!(!results[&ids[2]].as_ref().unwrap().cancelled);
    assert_eq!(
        results[&ids[2]].as_ref().unwrap().statistics.audio_seconds,
        0.08
    );
}

#[test]
#[cfg(feature = "native")]
fn continuous_transcription_preserves_eof_resampling_and_independent_cancellation() {
    use omni_r2t2::{
        input::{AudioSource, RunOptions, transcribe},
        stream::TranscriptionBatch,
    };
    use std::{
        collections::BTreeMap,
        sync::{Arc, atomic::AtomicBool},
    };
    struct Source {
        rate: u32,
        samples: Vec<f32>,
        cursor: usize,
    }
    impl AudioSource for Source {
        fn sample_rate(&self) -> u32 {
            self.rate
        }
        fn next_packet(&mut self, stop: &AtomicBool) -> omni_r2t2::Result<Option<Vec<f32>>> {
            if stop.load(Ordering::Acquire) || self.cursor == self.samples.len() {
                return Ok(None);
            }
            let end = (self.cursor + 137).min(self.samples.len());
            let out = self.samples[self.cursor..end].to_vec();
            self.cursor = end;
            Ok(Some(out))
        }
    }
    let source = |rate, n| Source {
        rate,
        samples: (0..n).map(|i| (i as f32 * 0.071).sin() * 0.1).collect(),
        cursor: 0,
    };
    let model = support::tiny_model();
    let mut serial = Vec::new();
    transcribe(
        model.clone(),
        config(),
        source(48000, 7680),
        RunOptions::default(),
        |e| {
            serial.push(e);
            Ok(())
        },
    )
    .unwrap();
    let mut batch = TranscriptionBatch::new(model, 2).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let cancelled = batch
        .try_submit(config(), source(16000, 6400), stop.clone())
        .unwrap()
        .unwrap();
    let mut events: BTreeMap<u64, Vec<TranscriptionEvent>> = BTreeMap::new();
    assert!(
        batch
            .step(|id, e| {
                events.entry(id).or_default().push(e);
                Ok(())
            })
            .is_empty()
    );
    let good = batch
        .try_submit(
            config(),
            source(48000, 7680),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
        .unwrap();
    assert!(
        batch
            .try_submit(config(), source(16000, 1), Arc::new(AtomicBool::new(false)))
            .unwrap()
            .is_none()
    );
    batch.step(|id, e| {
        events.entry(id).or_default().push(e);
        Ok(())
    });
    stop.store(true, Ordering::Release);
    let mut reports = BTreeMap::new();
    while !batch.is_idle() {
        for (id, result) in batch.step(|id, e| {
            events.entry(id).or_default().push(e);
            Ok(())
        }) {
            assert!(reports.insert(id, result.unwrap()).is_none());
        }
    }
    assert!(reports[&cancelled].cancelled);
    assert!(!reports[&good].cancelled);
    assert_eq!(batch.stats().max_batch_size, 2);
    for events in events.values() {
        assert_consistent(events);
    }
    let signature = |events: &[TranscriptionEvent]| {
        events
            .iter()
            .map(|e| {
                (
                    e.event.text.clone(),
                    e.event.delta.clone(),
                    e.event.audio_seconds,
                    e.event.generated_tokens,
                    e.event.is_final,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&events[&good]), signature(&serial));
    let retry = batch
        .try_submit(
            config(),
            source(48000, 7680),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
        .unwrap();
    let mut retry_events = Vec::new();
    while !batch.is_idle() {
        for (id, result) in batch.step(|id, e| {
            assert_eq!(id, retry);
            retry_events.push(e);
            Ok(())
        }) {
            assert_eq!(id, retry);
            result.unwrap();
        }
    }
    assert_eq!(signature(&retry_events), signature(&serial));
}

fn assert_consistent(events: &[TranscriptionEvent]) {
    let mut text = String::new();
    let mut compute = 0.0;
    for (index, event) in events.iter().enumerate() {
        text.push_str(&event.event.delta);
        compute += event.event.compute_ms;
        assert_eq!(text, event.event.text);
        assert_eq!(event.event_count, index + 1);
        assert_eq!(event.total_compute_ms, compute);
        let rtf = if event.event.audio_seconds == 0.0 {
            0.0
        } else {
            compute / (event.event.audio_seconds * 1000.0)
        };
        assert_eq!(event.rtf, rtf);
    }
    assert_eq!(
        events.iter().filter(|event| event.event.is_final).count(),
        1
    );
}

#[test]
fn rate_conversion_and_packets_match_the_session_interface() {
    let model = support::tiny_model();
    let pcm: Vec<_> = (0..10001).map(|i| (i as f32 * 0.071).sin() * 0.1).collect();
    let mut runner = Transcription::new(model.clone(), config(), 44100).unwrap();
    let mut events = Vec::new();
    for packet in pcm.chunks(137) {
        events.extend(pollster::block_on(runner.push(packet)).unwrap());
    }
    events.extend(pollster::block_on(runner.finish()).unwrap());
    assert_consistent(&events);
    let normalized = Resampler::new(44100, 16000)
        .unwrap()
        .push(&pcm, true)
        .unwrap();
    let mut session = Session::new(model, config()).unwrap();
    let mut reference = pollster::block_on(session.push(&normalized)).unwrap();
    reference.push(pollster::block_on(session.finish()).unwrap());
    assert_eq!(events.len(), reference.len());
    for (actual, expected) in events.iter().zip(reference) {
        assert_eq!(
            (
                &actual.event.delta,
                &actual.event.text,
                actual.event.audio_seconds,
                actual.event.is_final,
                actual.event.generated_tokens
            ),
            (
                &expected.delta,
                &expected.text,
                expected.audio_seconds,
                expected.is_final,
                expected.generated_tokens
            )
        );
    }
    let stats = runner.statistics();
    assert_eq!(stats.event_count, events.len());
    assert_eq!(
        stats.total_compute_ms,
        events.last().unwrap().total_compute_ms
    );
    assert_eq!(stats.audio_seconds, normalized.len() as f64 / 16000.0);
    assert!(stats.p95_step_ms.is_finite());
    assert!(matches!(
        pollster::block_on(runner.finish()),
        Err(Error::Finished)
    ));
}

#[test]
fn eof_delivers_resampler_tail_events_before_the_terminal_event() {
    let mut runner = Transcription::new(support::tiny_model(), config(), 48000).unwrap();
    // Filter lookahead withholds the end of this exact 80 ms chunk until EOF.
    assert!(
        pollster::block_on(runner.push(&vec![0.1; 3840]))
            .unwrap()
            .is_empty()
    );
    let events = pollster::block_on(runner.finish()).unwrap();
    assert_eq!(events.len(), 2);
    assert!(!events[0].event.is_final);
    assert!(events[1].event.is_final);
    assert_eq!(events[1].event.audio_seconds, 0.08);
    assert_consistent(&events);
}

#[test]
fn stop_preserves_committed_text_discards_pending_audio_and_is_idempotent() {
    let mut runner = Transcription::new(support::tiny_model(), config(), 16000).unwrap();
    let mut events = pollster::block_on(runner.push(&vec![0.1; 1400])).unwrap();
    let last_text = events.last().unwrap().event.text.clone();
    runner.stop_handle().store(true, Ordering::Release);
    assert!(matches!(
        pollster::block_on(runner.push(&[0.2; 1280])),
        Err(Error::Interrupted)
    ));
    events.extend(pollster::block_on(runner.finish()).unwrap());
    let terminal = events.last().unwrap();
    assert!(terminal.cancelled);
    assert_eq!(terminal.event.text, last_text);
    assert_eq!(terminal.event.audio_seconds, 0.08);
    assert_eq!(terminal.event.compute_ms, 0.0);
    assert!(terminal.event.delta.is_empty());
    assert_consistent(&events);
    assert_eq!(
        runner.interrupt().unwrap().event_count,
        terminal.event_count
    );
    assert_eq!(runner.statistics().event_count, events.len());
    assert!(matches!(
        pollster::block_on(runner.push(&[])),
        Err(Error::Finished)
    ));
}

#[test]
fn empty_input_and_invalid_pcm_do_not_poison_the_run() {
    let model = support::tiny_model();
    assert!(Transcription::new(model.clone(), config(), 0).is_err());
    let mut runner = Transcription::new(model, config(), 48000).unwrap();
    assert_eq!(runner.statistics().p95_step_ms, 0.0);
    assert!(matches!(
        pollster::block_on(runner.push(&[f32::NAN])),
        Err(Error::Audio(_))
    ));
    let events = pollster::block_on(runner.finish()).unwrap();
    assert_consistent(&events);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].rtf, 0.0);
    assert!(!events[0].cancelled);
    assert!(!runner.interrupt().unwrap().cancelled);
}

#[cfg(feature = "native")]
mod native {
    use super::*;
    use omni_r2t2::input::{self, AudioSource, PcmSource, RunOptions, WavSource};
    use std::{
        io::{Cursor, Read},
        sync::atomic::AtomicBool,
    };

    struct ShortReads {
        bytes: Cursor<Vec<u8>>,
        interrupted: bool,
    }
    impl Read for ShortReads {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            self.interrupted = !self.interrupted;
            if self.interrupted {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let limit = buffer.len().min(3);
            self.bytes.read(&mut buffer[..limit])
        }
    }

    #[test]
    fn pcm_adapter_carries_split_samples_and_rejects_incomplete_eof() {
        let samples: [i16; 5] = [i16::MIN, -17, 0, 42, i16::MAX];
        let bytes: Vec<_> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut source = PcmSource::new(ShortReads {
            bytes: Cursor::new(bytes),
            interrupted: false,
        });
        let mut actual = Vec::new();
        while let Some(pcm) = source.next_packet(&AtomicBool::new(false)).unwrap() {
            actual.extend(pcm);
        }
        assert_eq!(actual, samples.map(|v| v as f32 / 32768.0));
        let mut odd = PcmSource::new(Cursor::new([1]));
        assert!(
            odd.next_packet(&AtomicBool::new(false))
                .unwrap()
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            odd.next_packet(&AtomicBool::new(false)),
            Err(Error::Audio(_))
        ));
    }

    #[test]
    fn wav_and_pcm_sources_share_finalization_and_statistics() {
        let samples: Vec<i16> = (0..3200).map(|i| ((i * 73 % 997) - 498) as i16).collect();
        let mut wav = Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut wav, spec).unwrap();
        for &value in &samples {
            writer.write_sample(value).unwrap();
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        wav.set_position(0);
        let model = support::tiny_model();
        let mut wav_events = Vec::new();
        let wav_stats = input::transcribe(
            model.clone(),
            config(),
            WavSource::new(wav).unwrap(),
            RunOptions::default(),
            |event| {
                wav_events.push(event);
                Ok(())
            },
        )
        .unwrap();
        let bytes: Vec<_> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut pcm_events = Vec::new();
        let pcm_stats = input::transcribe(
            model,
            config(),
            PcmSource::new(Cursor::new(bytes)),
            RunOptions::default(),
            |event| {
                pcm_events.push(event);
                Ok(())
            },
        )
        .unwrap();
        assert_consistent(&wav_events);
        assert_consistent(&pcm_events);
        assert_eq!(wav_stats.audio_seconds, pcm_stats.audio_seconds);
        assert_eq!(wav_stats.event_count, pcm_stats.event_count);
        assert_eq!(
            wav_events.last().unwrap().event.text,
            pcm_events.last().unwrap().event.text
        );
    }

    struct StopsWhileReading {
        fail: bool,
    }
    impl AudioSource for StopsWhileReading {
        fn sample_rate(&self) -> u32 {
            16000
        }
        fn next_packet(&mut self, stop: &AtomicBool) -> omni_r2t2::Result<Option<Vec<f32>>> {
            stop.store(true, Ordering::Release);
            if self.fail {
                Err(Error::Audio("capture failed".into()))
            } else {
                Ok(Some(vec![0.1; 16000]))
            }
        }
    }

    #[test]
    fn source_stop_discards_backlog_but_does_not_hide_failures() {
        let model = support::tiny_model();
        let mut events = Vec::new();
        let stats = input::transcribe(
            model.clone(),
            config(),
            StopsWhileReading { fail: false },
            RunOptions::default(),
            |event| {
                events.push(event);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stats.audio_seconds, 0.0);
        assert_eq!(events.len(), 1);
        assert!(events[0].cancelled);
        let error = input::transcribe(
            model,
            config(),
            StopsWhileReading { fail: true },
            RunOptions::default(),
            |_| panic!("a capture failure must not report success"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Audio(message) if message == "capture failed"));
    }

    #[test]
    fn cancellation_checks_capture_failure_that_happened_during_inference() {
        struct Capture {
            failed: std::rc::Rc<std::cell::Cell<bool>>,
        }
        impl AudioSource for Capture {
            fn sample_rate(&self) -> u32 {
                16000
            }
            fn next_packet(&mut self, stop: &AtomicBool) -> omni_r2t2::Result<Option<Vec<f32>>> {
                if self.failed.get() {
                    return Err(Error::Audio("capture queue overflowed".into()));
                }
                if stop.load(Ordering::Acquire) {
                    return Ok(None);
                }
                Ok(Some(vec![0.1; 1280]))
            }
        }
        let failed = std::rc::Rc::new(std::cell::Cell::new(false));
        let options = RunOptions::default();
        let stop = options.stop.clone();
        let error = input::transcribe(
            support::tiny_model(),
            config(),
            Capture {
                failed: failed.clone(),
            },
            options,
            |event| {
                assert!(
                    !event.event.is_final,
                    "capture errors must not become successful terminal events"
                );
                // Simulate the producer failing and Ctrl-C arriving while the
                // previous input packet was being decoded.
                failed.set(true);
                stop.store(true, Ordering::Release);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::Audio(message) if message == "capture queue overflowed"));
    }
}

#[test]
#[cfg(feature = "native")]
fn serial_and_batch_share_lookahead_exact_eof_and_packet_independent_events() {
    use omni_r2t2::{input::AudioSource, stream::TranscriptionBatch};
    use std::{
        collections::BTreeMap,
        sync::{Arc, atomic::AtomicBool},
    };
    struct Packets {
        samples: Vec<f32>,
        width: usize,
        cursor: usize,
    }
    impl AudioSource for Packets {
        fn sample_rate(&self) -> u32 {
            16000
        }
        fn next_packet(&mut self, _: &AtomicBool) -> omni_r2t2::Result<Option<Vec<f32>>> {
            if self.cursor == self.samples.len() {
                return Ok(None);
            }
            let end = (self.cursor + self.width).min(self.samples.len());
            let packet = self.samples[self.cursor..end].to_vec();
            self.cursor = end;
            Ok(Some(packet))
        }
    }
    let model = support::tiny_model();
    let mut cfg = config();
    cfg.lookahead_ms = 40;
    // Empty, just before first lookahead, exact first window, exact later window,
    // and a final partial chunk. Compare every observable event, not just text.
    let counts = [0, 1919, 1920, 3200, 3201];
    let mut batch = TranscriptionBatch::new(model.clone(), counts.len()).unwrap();
    let mut expected = BTreeMap::new();
    let signature = |events: &[TranscriptionEvent]| {
        events
            .iter()
            .map(|e| {
                (
                    e.event.text.clone(),
                    e.event.delta.clone(),
                    e.event.audio_seconds,
                    e.event.is_final,
                    e.event.generated_tokens,
                    e.event.token_limit_reached,
                    e.event_count,
                    e.cancelled,
                )
            })
            .collect::<Vec<_>>()
    };
    for (i, n) in counts.into_iter().enumerate() {
        let samples: Vec<_> = (0..n).map(|i| (i as f32 * 0.071).sin() * 0.1).collect();
        let mut run = Transcription::new(model.clone(), cfg.clone(), 16000).unwrap();
        let mut events = Vec::new();
        for chunk in samples.chunks(137) {
            events.extend(pollster::block_on(run.push(chunk)).unwrap());
        }
        events.extend(pollster::block_on(run.finish()).unwrap());
        assert_consistent(&events);
        let terminal = events.last().unwrap();
        let statistics = serde_json::to_value(run.statistics()).unwrap();
        run.stop_handle().store(true, Ordering::Release);
        assert_eq!(
            serde_json::to_value(run.interrupt().unwrap()).unwrap(),
            serde_json::to_value(terminal).unwrap()
        );
        assert_eq!(serde_json::to_value(run.statistics()).unwrap(), statistics);
        assert!(matches!(
            pollster::block_on(run.finish()),
            Err(Error::Finished)
        ));
        let id = batch
            .try_submit(
                cfg.clone(),
                Packets {
                    samples,
                    width: if i % 2 == 0 { 1 } else { 4096 },
                    cursor: 0,
                },
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap()
            .unwrap();
        expected.insert(id, signature(&events));
    }
    let mut events: BTreeMap<u64, Vec<TranscriptionEvent>> = BTreeMap::new();
    let mut reports = BTreeMap::new();
    while !batch.is_idle() {
        for (id, result) in batch.step(|id, e| {
            events.entry(id).or_default().push(e);
            Ok(())
        }) {
            assert!(reports.insert(id, result.unwrap()).is_none());
        }
    }
    assert_eq!(reports.len(), counts.len());
    for (id, expected) in expected {
        assert_consistent(&events[&id]);
        assert_eq!(signature(&events[&id]), expected);
        assert!(!reports[&id].cancelled);
        assert_eq!(reports[&id].statistics.event_count, events[&id].len());
        assert_eq!(
            reports[&id].statistics.audio_seconds,
            events[&id].last().unwrap().event.audio_seconds
        );
    }
}
