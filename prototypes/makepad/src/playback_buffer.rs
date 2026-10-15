//! Playback scheduling and lock-free consumption; no device or model dependency.
use crate::capture_buffer::CaptureBuffer;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};

pub struct Buffer {
    pub samples: CaptureBuffer,
    pub capacity: usize,
    pub rate: usize,
    pub ready_frames: AtomicUsize,
    pub finished: AtomicBool,
    pub playing: AtomicBool,
    pub first_playback_ns: AtomicU64,
    pub audible_until_ns: AtomicU64,
    pub rendered: AtomicUsize,
    pub underruns: AtomicUsize,
    pub gap_frames: AtomicUsize,
    pub cancel: Arc<AtomicBool>,
}

impl Buffer {
    pub fn new(rate: usize, seconds: usize, cancel: Arc<AtomicBool>, adaptive: bool) -> Self {
        let capacity = rate * seconds;
        Self {
            samples: CaptureBuffer::new(capacity),
            capacity,
            rate,
            // Need two arrivals to distinguish prefill latency from generation
            // throughput. EOS bypasses this gate for a one-chunk utterance.
            ready_frames: AtomicUsize::new(if adaptive { capacity } else { rate / 5 }),
            finished: AtomicBool::new(false),
            playing: AtomicBool::new(false),
            first_playback_ns: AtomicU64::new(0),
            audible_until_ns: AtomicU64::new(0),
            rendered: AtomicUsize::new(0),
            underruns: AtomicUsize::new(0),
            gap_frames: AtomicUsize::new(0),
            cancel,
        }
    }
}

pub struct Consumer {
    buffer: Arc<Buffer>,
    playing: bool,
    started: bool,
}

impl Consumer {
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Self {
            buffer,
            playing: false,
            started: false,
        }
    }

    /// No allocation, mutex, resampling or model work on the audio callback.
    /// The pre-zeroed suffix is only used while buffering or after the last PCM.
    pub fn render<T: Copy>(
        &mut self,
        output: &mut [T],
        channels: usize,
        zero: T,
        convert: impl Fn(f32) -> T,
    ) -> usize {
        output.fill(zero);
        let b = &self.buffer;
        if b.cancel.load(Ordering::Acquire) {
            self.playing = false;
            b.playing.store(false, Ordering::Release);
            return 0;
        }
        let frames = output.len() / channels;
        let finished = b.finished.load(Ordering::Acquire);
        let available = b.samples.pending();
        if !self.playing {
            self.playing =
                available > 0 && (finished || available >= b.ready_frames.load(Ordering::Acquire));
            if !self.playing {
                if self.started && !finished {
                    b.gap_frames.fetch_add(frames, Ordering::Relaxed);
                }
                return 0;
            }
            b.playing.store(true, Ordering::Release);
        }
        let mut scratch = [0.0; 512];
        let mut written = 0;
        while written < frames {
            let count = (frames - written).min(scratch.len());
            let count = b.samples.pop(&mut scratch[..count]);
            if count == 0 {
                break;
            }
            for (frame, &sample) in output[written * channels..(written + count) * channels]
                .chunks_exact_mut(channels)
                .zip(&scratch[..count])
            {
                frame.fill(convert(sample));
            }
            written += count;
        }
        self.started |= written > 0;
        b.rendered.fetch_add(written, Ordering::Relaxed);
        if written < frames && !b.finished.load(Ordering::Acquire) {
            self.playing = false;
            b.playing.store(false, Ordering::Release);
            b.underruns.fetch_add(1, Ordering::Relaxed);
            b.gap_frames.fetch_add(frames - written, Ordering::Relaxed);
        }
        written
    }
}

/// Estimate a conservative duration from the text, not from the first packet's
/// prefill/compilation delay. This is a buffer estimate, never a generation limit.
pub fn estimated_seconds(text: &str) -> f64 {
    let mut syllables = 0;
    let mut words = 0;
    let mut in_word = false;
    let mut pauses = 0.0;
    for c in text.chars() {
        if matches!(c as u32, 0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af | 0x20000..=0x313ff)
            || c.is_numeric()
        {
            syllables += 1;
            in_word = false;
        } else if c.is_alphabetic() {
            if !in_word {
                words += 1;
            }
            in_word = true;
        } else {
            in_word = false;
            if matches!(c, '.' | '。' | '!' | '！' | '?' | '？' | ';' | '；') {
                pauses += 0.3;
            } else if matches!(c, ',' | '，' | ':' | '：') {
                pauses += 0.15;
            }
        }
    }
    0.5 + syllables as f64 / 2.6 + words as f64 / 2.0 + pauses
}

pub struct AdaptiveBuffer {
    estimate: f64,
    generated: f64,
    first: Option<(f64, f64)>,
    previous: f64,
    recent_rtf: f64,
    intervals: std::collections::VecDeque<f64>,
}

impl AdaptiveBuffer {
    pub fn new(estimate: f64) -> Self {
        Self {
            estimate,
            generated: 0.0,
            first: None,
            previous: 0.0,
            recent_rtf: 0.0,
            intervals: std::collections::VecDeque::new(),
        }
    }

    /// Return the buffer needed before starting/resuming, in audio seconds.
    /// Fast generation starts after a short jitter buffer. When slower than
    /// playback, retain enough lead for the estimated remaining synthesis.
    pub fn observe(&mut self, seconds: f64, now: f64) -> Option<f64> {
        self.generated += seconds;
        let Some((first_time, first_audio)) = self.first else {
            self.first = Some((now, self.generated));
            self.previous = now;
            return None;
        };
        let interval_rtf = (now - self.previous).max(0.0) / seconds.max(0.001);
        self.intervals.push_back(interval_rtf);
        if self.intervals.len() > 32 {
            self.intervals.pop_front();
        }
        self.previous = now;
        self.recent_rtf = if self.recent_rtf == 0.0 {
            interval_rtf
        } else {
            0.75 * self.recent_rtf + 0.25 * interval_rtf
        };
        let mean_rtf = (now - first_time) / (self.generated - first_audio).max(0.001);
        // Retain the conservative prior until enough real packets arrive. Then
        // use the observed slow tail as well as the mean, instead of charging a
        // fixed 25% penalty even for a steady stream. Never include prefill.
        let rtf = if self.intervals.len() < 8 {
            mean_rtf.max(self.recent_rtf) * 1.25
        } else {
            let mut intervals: Vec<_> = self.intervals.iter().copied().collect();
            intervals.sort_unstable_by(f64::total_cmp);
            let p90 = intervals[((intervals.len() - 1) as f64 * 0.9).ceil() as usize];
            (mean_rtf.max(self.recent_rtf) * 1.10).max(p90 * 1.05)
        };
        let remaining = (self.estimate - self.generated).max(1.0);
        Some((remaining * (rtf - 1.0).max(0.0) + seconds * 2.0).max(0.64))
    }
}

/// Session-local duration calibration for a voice/language. Learn only from
/// completed utterances; an interrupted or capped result is not a duration.
#[derive(Default)]
pub struct DurationHistory {
    ratios: std::collections::HashMap<u64, std::collections::VecDeque<(f64, f64)>>,
}
impl DurationHistory {
    pub fn estimate(&self, voice: u64, text: &str) -> f64 {
        let base = estimated_seconds(text);
        self.ratios.get(&voice).map_or(base, |ratios| {
            // Short phrases can be much faster per character than paragraphs.
            // Never extrapolate their calibration to a much longer utterance.
            let comparable: Vec<_> = ratios
                .iter()
                .filter(|(length, _)| *length >= base * 0.5 && *length <= base * 2.0)
                .map(|(_, ratio)| *ratio)
                .collect();
            let slowest = comparable.iter().copied().reduce(f64::max);
            slowest.map_or(base, |ratio| {
                let calibrated = base * (ratio * 1.20).max(0.5);
                // One short result is weak evidence for a faster voice. Keep
                // the text prior until several comparable completions agree;
                // a slower result increases the estimate immediately.
                if comparable.len() < 3 {
                    calibrated.max(base)
                } else {
                    calibrated
                }
            })
        })
    }
    pub fn observe(&mut self, voice: u64, text: &str, audio_seconds: f64) {
        if audio_seconds < 1.0 || !audio_seconds.is_finite() {
            return;
        }
        if self.ratios.len() >= 64 && !self.ratios.contains_key(&voice) {
            self.ratios.clear();
        }
        let values = self.ratios.entry(voice).or_default();
        let base = estimated_seconds(text);
        values.push_back((base, audio_seconds / base));
        if values.len() > 8 {
            values.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_history_is_voice_local_and_keeps_slow_sample_headroom() {
        let text = "你好，今天天气很好。让我们一起听听这个声音。";
        let base = estimated_seconds(text);
        let mut history = DurationHistory::default();
        assert_eq!(history.estimate(1, text), base);
        history.observe(1, text, 5.0);
        assert_eq!(history.estimate(1, text), base);
        history.observe(1, text, 4.0);
        assert_eq!(history.estimate(1, text), base);
        history.observe(1, text, 4.5);
        assert!((history.estimate(1, text) - 6.0).abs() < 1e-6);
        history.observe(1, text, 8.0);
        assert!((history.estimate(1, text) - 9.6).abs() < 1e-6);
        assert_eq!(history.estimate(2, text), base);
        history.observe(2, text, f64::NAN);
        assert_eq!(history.estimate(2, text), base);
        let paragraph = text.repeat(6);
        let long_base = estimated_seconds(&paragraph);
        assert_eq!(history.estimate(1, &paragraph), long_base);
        history.observe(1, &paragraph, 40.0);
        assert_eq!(history.estimate(1, &paragraph), long_base.max(48.0));
        history.observe(1, &paragraph, 40.0);
        history.observe(1, &paragraph, 40.0);
        assert!((history.estimate(1, &paragraph) - 48.0).abs() < 1e-6);
        assert!((history.estimate(1, text) - 9.6).abs() < 1e-6);
    }

    #[test]
    fn slow_voice_increases_duration_estimate_from_first_completion() {
        let text = "你好，欢迎回来。";
        let mut history = DurationHistory::default();
        history.observe(1, text, 10.0);
        assert!((history.estimate(1, text) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn eof_releases_short_buffer_and_preserves_every_frame_across_callbacks() {
        let b = Arc::new(Buffer::new(1000, 2, Arc::new(AtomicBool::new(false)), true));
        let mut reader = Consumer::new(b.clone());
        let pcm: Vec<f32> = (0..701).map(|i| i as f32 + 1.0).collect();
        b.samples.push(&pcm, 1, |x| x).unwrap();
        let mut output = [0.0; 274];
        assert_eq!(reader.render(&mut output, 2, 0.0, |x| x), 0);
        assert_eq!(b.samples.pending(), pcm.len());
        b.finished.store(true, Ordering::Release);
        let mut actual = Vec::new();
        loop {
            let n = reader.render(&mut output, 2, 0.0, |x| x);
            if n == 0 {
                break;
            }
            for f in output[..n * 2].chunks_exact(2) {
                assert_eq!(f[0], f[1]);
                actual.push(f[0]);
            }
            assert!(output[n * 2..].iter().all(|&x| x == 0.0));
        }
        assert_eq!(actual, pcm);
        assert_eq!(b.underruns.load(Ordering::Relaxed), 0);
        assert_eq!(b.rendered.load(Ordering::Relaxed), pcm.len());
    }

    #[test]
    fn producer_starvation_rebuffers_instead_of_playing_each_tiny_packet() {
        let stop = Arc::new(AtomicBool::new(false));
        let b = Arc::new(Buffer::new(1000, 2, stop.clone(), false));
        let mut reader = Consumer::new(b.clone());
        let mut output = [0.0; 250];
        b.samples.push(&[1.0; 220], 1, |x| x).unwrap();
        assert_eq!(reader.render(&mut output, 1, 0.0, |x| x), 220);
        assert_eq!(b.underruns.load(Ordering::Relaxed), 1);
        b.samples.push(&[2.0; 160], 1, |x| x).unwrap();
        assert_eq!(reader.render(&mut output, 1, 0.0, |x| x), 0);
        b.samples.push(&[3.0; 160], 1, |x| x).unwrap();
        assert_eq!(reader.render(&mut output, 1, 0.0, |x| x), 250);
        assert_eq!(&output[..160], &[2.0; 160]);
        assert_eq!(&output[160..], &[3.0; 90]);
        stop.store(true, Ordering::Release);
        assert_eq!(reader.render(&mut output, 1, 0.0, |x| x), 0);
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn adaptive_buffer_accounts_for_speed_without_counting_prefill_as_throughput() {
        let mut fast = AdaptiveBuffer::new(10.0);
        assert_eq!(fast.observe(0.16, 40.0), None);
        assert_eq!(fast.observe(0.16, 40.08), Some(0.64));
        let mut slow = AdaptiveBuffer::new(10.0);
        assert_eq!(slow.observe(0.16, 40.0), None);
        assert!(slow.observe(0.16, 40.32).unwrap() > 10.0);
        assert!(estimated_seconds("你好，欢迎回来。") > 2.0);
        assert!(estimated_seconds("Hello, welcome to local speech synthesis.") > 3.0);
    }
}
