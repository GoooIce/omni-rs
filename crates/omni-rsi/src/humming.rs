//! Monophonic note and playback-clock evaluation for music-conditioned humming.
//! This module consumes annotations; it does not extract pitch or record audio.

pub mod training;

/// A sample clock whose origin is expressed in a shared monotonic time domain.
///
/// The origin must be calibrated against the other clock. It does not by itself
/// account for a hardware buffer or prove when sound reached a listener.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Clock {
    pub sample_rate_hz: u32,
    pub origin_ns: i64,
}

/// A reference note on the input clock, with an exclusive end.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Note {
    pub start: u64,
    pub end: u64,
    pub pitch_midi: f64,
}

/// An output interval, with an exclusive end and the input prefix available at its start.
///
/// `None` denotes unvoiced audio. `note_onset` denotes a new articulation, not
/// every pitch change. Split spans whenever pitch or the consumed prefix changes.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct OutputSpan {
    pub start: u64,
    pub end: u64,
    pub pitch_midi: Option<f64>,
    pub note_onset: bool,
    pub input_prefix_end: u64,
}

/// Whether an evaluation uses only arrived input or permits future input offline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum InputMode {
    Causal,
    Offline,
}

/// One complete input interval and an output recording covering that interval.
///
/// Reference gaps are unvoiced; output spans must cover the whole recording.
/// See `docs/humming-evaluation.md` for timing, denominators and limitations.
pub struct Evaluation<'a> {
    pub input_clock: Clock,
    pub input_samples: u64,
    pub notes: &'a [Note],
    pub output_clock: Clock,
    pub output_samples: u64,
    pub output: &'a [OutputSpan],
    pub input_mode: InputMode,
    pub onset_tolerance_seconds: f64,
}

/// Metrics over the input interval; undefined denominators are `None`.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Report {
    pub input_clock: Clock,
    pub input_samples: u64,
    pub output_clock: Clock,
    pub output_samples: u64,
    pub input_mode: InputMode,
    pub onset_tolerance_seconds: f64,
    pub reference_voiced_seconds: f64,
    pub output_voiced_seconds: f64,
    pub voiced_overlap_seconds: f64,
    pub output_voiced_rate: f64,
    pub voicing_recall: Option<f64>,
    pub voicing_precision: Option<f64>,
    pub voicing_false_alarm_rate: Option<f64>,
    pub mean_absolute_pitch_error_cents: Option<f64>,
    pub pitch_accuracy: Option<f64>,
    pub chroma_accuracy: Option<f64>,
    pub octave_error_rate: Option<f64>,
    pub onset_matches: Vec<OnsetMatch>,
    pub missed_notes: usize,
    pub extra_onsets: usize,
    pub onset_recall: Option<f64>,
    pub onset_precision: Option<f64>,
    pub mean_signed_onset_error_seconds: Option<f64>,
    pub mean_absolute_onset_error_seconds: Option<f64>,
    pub duration_compared_notes: usize,
    pub mean_absolute_duration_error_seconds: Option<f64>,
    pub contour_reference_pairs: usize,
    pub contour_compared_pairs: usize,
    pub contour_correct_pairs: usize,
    pub contour_accuracy: Option<f64>,
}

/// One association, preserving both original clocks and the recorded prefix.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct OnsetMatch {
    pub reference_note: usize,
    pub output_sample: u64,
    pub input_prefix_end: u64,
    pub deviation_seconds: f64,
    pub duration_error_seconds: Option<f64>,
}

#[derive(Clone, Copy)]
struct OutputNote {
    start: u64,
    end: u64,
    input_prefix_end: u64,
    pitch_sum: f64,
}

#[derive(Clone, Copy, Default)]
struct MatchScore {
    count: usize,
    total_error: f64,
}

impl MatchScore {
    fn better_than(self, other: Self) -> bool {
        self.count > other.count
            || (self.count == other.count && self.total_error < other.total_error)
    }
}

const MAX_SAMPLES: u64 = 1 << 53;
const MAX_SAMPLE_RATE: u32 = 1_000_000;

fn valid_pitch(pitch: f64) -> bool {
    pitch.is_finite() && (0.0..=127.0).contains(&pitch)
}

/// Convert a positive fundamental frequency to fractional MIDI pitch.
pub fn pitch_midi_from_hz(hz: f64) -> Result<f64, String> {
    if !hz.is_finite() || hz <= 0.0 {
        return Err("frequency must be positive and finite".into());
    }
    let pitch = 69.0 + 12.0 * (hz / 440.0).log2();
    if !valid_pitch(pitch) {
        return Err("frequency is outside the MIDI pitch range".into());
    }
    Ok(pitch)
}

fn seconds(sample: u64, clock: Clock) -> f64 {
    sample as f64 / f64::from(clock.sample_rate_hz)
}

// Compare sample times in nanoseconds scaled by both sample rates.
fn input_minus_output_ticks(
    input: Clock,
    input_sample: u64,
    output: Clock,
    output_sample: u64,
) -> i128 {
    (i128::from(input.origin_ns) - i128::from(output.origin_ns))
        * i128::from(input.sample_rate_hz)
        * i128::from(output.sample_rate_hz)
        + i128::from(input_sample) * 1_000_000_000 * i128::from(output.sample_rate_hz)
        - i128::from(output_sample) * 1_000_000_000 * i128::from(input.sample_rate_hz)
}

fn onset_deviation_seconds(
    input: Clock,
    input_sample: u64,
    output: Clock,
    output_sample: u64,
) -> f64 {
    -(input_minus_output_ticks(input, input_sample, output, output_sample) as f64)
        / (1_000_000_000.0 * f64::from(input.sample_rate_hz) * f64::from(output.sample_rate_hz))
}

impl Evaluation<'_> {
    /// Evaluate on absolute time without resampling or time warping.
    pub fn evaluate(&self) -> Result<Report, String> {
        self.validate()?;

        // One tick is 1 / (1e9 * input_rate * output_rate) seconds. Intersect
        // intervals as integers before converting durations or ratios to f64.
        let input_sample_ticks = 1_000_000_000_i128 * i128::from(self.output_clock.sample_rate_hz);
        let input_ticks = |sample: u64| i128::from(sample) * input_sample_ticks;
        let ticks_per_second =
            (input_sample_ticks * i128::from(self.input_clock.sample_rate_hz)) as f64;
        let duration = input_ticks(self.input_samples);
        let reference_voiced: i128 = self
            .notes
            .iter()
            .map(|note| input_ticks(note.end - note.start))
            .sum();
        let mut output_voiced = 0_i128;
        let mut overlap = 0_i128;
        let mut absolute_pitch_error = 0.0;
        let mut pitch_correct = 0_i128;
        let mut chroma_correct = 0_i128;
        let mut octave_error = 0_i128;
        let mut first_note = 0;
        for span in self.output.iter().filter(|span| span.pitch_midi.is_some()) {
            let start =
                (-input_minus_output_ticks(self.input_clock, 0, self.output_clock, span.start))
                    .max(0);
            let end = (-input_minus_output_ticks(self.input_clock, 0, self.output_clock, span.end))
                .min(duration);
            if end <= start {
                continue;
            }
            output_voiced += end - start;
            while first_note < self.notes.len() && input_ticks(self.notes[first_note].end) <= start
            {
                first_note += 1;
            }
            for note in &self.notes[first_note..] {
                let note_start = input_ticks(note.start);
                if note_start >= end {
                    break;
                }
                let note_end = input_ticks(note.end);
                let compared = (end.min(note_end) - start.max(note_start)).max(0);
                overlap += compared;
                let cents =
                    (span.pitch_midi.expect("filtered voiced span") - note.pitch_midi) * 100.0;
                let absolute_cents = cents.abs();
                let chroma_cents = (cents + 600.0).rem_euclid(1200.0) - 600.0;
                absolute_pitch_error += compared as f64 * absolute_cents;
                if absolute_cents <= 50.0 {
                    pitch_correct += compared;
                }
                if chroma_cents.abs() <= 50.0 {
                    chroma_correct += compared;
                    if absolute_cents > 50.0 {
                        octave_error += compared;
                    }
                }
            }
        }
        let unvoiced = duration - reference_voiced;
        let output_notes = self.output_notes();
        let pairs = self.match_onsets(&output_notes)?;
        let mut onset_matches = Vec::with_capacity(pairs.len());
        let mut signed_onset_error = 0.0;
        let mut absolute_onset_error = 0.0;
        let mut absolute_duration_error = 0.0;
        let mut duration_compared_notes = 0;
        let mut matched_pitch = vec![None; self.notes.len()];
        for (reference_index, output_index) in pairs {
            let reference = self.notes[reference_index];
            let output = output_notes[output_index];
            let deviation = onset_deviation_seconds(
                self.input_clock,
                reference.start,
                self.output_clock,
                output.start,
            );
            let duration_error = (output.end < self.output_samples).then(|| {
                seconds(output.end - output.start, self.output_clock)
                    - seconds(reference.end - reference.start, self.input_clock)
            });
            signed_onset_error += deviation;
            absolute_onset_error += deviation.abs();
            if let Some(error) = duration_error {
                absolute_duration_error += error.abs();
                duration_compared_notes += 1;
            }
            matched_pitch[reference_index] =
                Some(output.pitch_sum / (output.end - output.start) as f64);
            onset_matches.push(OnsetMatch {
                reference_note: reference_index,
                output_sample: output.start,
                input_prefix_end: output.input_prefix_end,
                deviation_seconds: deviation,
                duration_error_seconds: duration_error,
            });
        }
        let matched = onset_matches.len();
        let mut contour_compared_pairs = 0;
        let mut contour_correct_pairs = 0;
        for index in 1..self.notes.len() {
            if let (Some(previous), Some(current)) =
                (matched_pitch[index - 1], matched_pitch[index])
            {
                contour_compared_pairs += 1;
                if direction(current - previous)
                    == direction(self.notes[index].pitch_midi - self.notes[index - 1].pitch_midi)
                {
                    contour_correct_pairs += 1;
                }
            }
        }
        Ok(Report {
            input_clock: self.input_clock,
            input_samples: self.input_samples,
            output_clock: self.output_clock,
            output_samples: self.output_samples,
            input_mode: self.input_mode,
            onset_tolerance_seconds: self.onset_tolerance_seconds,
            reference_voiced_seconds: reference_voiced as f64 / ticks_per_second,
            output_voiced_seconds: output_voiced as f64 / ticks_per_second,
            voiced_overlap_seconds: overlap as f64 / ticks_per_second,
            output_voiced_rate: output_voiced as f64 / duration as f64,
            voicing_recall: (reference_voiced > 0)
                .then(|| overlap as f64 / reference_voiced as f64),
            voicing_precision: (output_voiced > 0).then(|| overlap as f64 / output_voiced as f64),
            voicing_false_alarm_rate: (unvoiced > 0)
                .then(|| (output_voiced - overlap) as f64 / unvoiced as f64),
            mean_absolute_pitch_error_cents: (overlap > 0)
                .then(|| absolute_pitch_error / overlap as f64),
            pitch_accuracy: (overlap > 0).then(|| pitch_correct as f64 / overlap as f64),
            chroma_accuracy: (overlap > 0).then(|| chroma_correct as f64 / overlap as f64),
            octave_error_rate: (overlap > 0).then(|| octave_error as f64 / overlap as f64),
            missed_notes: self.notes.len() - matched,
            extra_onsets: output_notes.len() - matched,
            onset_recall: (!self.notes.is_empty())
                .then(|| matched as f64 / self.notes.len() as f64),
            onset_precision: (!output_notes.is_empty())
                .then(|| matched as f64 / output_notes.len() as f64),
            mean_signed_onset_error_seconds: (matched > 0)
                .then(|| signed_onset_error / matched as f64),
            mean_absolute_onset_error_seconds: (matched > 0)
                .then(|| absolute_onset_error / matched as f64),
            duration_compared_notes,
            mean_absolute_duration_error_seconds: (duration_compared_notes > 0)
                .then(|| absolute_duration_error / duration_compared_notes as f64),
            contour_reference_pairs: self.notes.len().saturating_sub(1),
            contour_compared_pairs,
            contour_correct_pairs,
            contour_accuracy: (contour_compared_pairs > 0)
                .then(|| contour_correct_pairs as f64 / contour_compared_pairs as f64),
            onset_matches,
        })
    }

    fn output_notes(&self) -> Vec<OutputNote> {
        let mut notes: Vec<OutputNote> = Vec::new();
        let mut active = None;
        for span in self.output {
            if span.pitch_midi.is_none() {
                active = None;
                continue;
            }
            if span.note_onset {
                notes.push(OutputNote {
                    start: span.start,
                    end: span.end,
                    input_prefix_end: span.input_prefix_end,
                    pitch_sum: span.pitch_midi.expect("voiced span")
                        * (span.end - span.start) as f64,
                });
                active = Some(notes.len() - 1);
            } else if let Some(index) = active {
                notes[index].end = span.end;
                notes[index].pitch_sum +=
                    span.pitch_midi.expect("voiced span") * (span.end - span.start) as f64;
            }
        }
        notes.retain(|note| {
            input_minus_output_ticks(self.input_clock, 0, self.output_clock, note.start) <= 0
                && input_minus_output_ticks(
                    self.input_clock,
                    self.input_samples,
                    self.output_clock,
                    note.start,
                ) > 0
        });
        notes
    }

    // Maximum cardinality, then minimum total absolute onset error; no samples
    // are shifted or stretched. Cap the dynamic-programming matrix explicitly.
    fn match_onsets(&self, output: &[OutputNote]) -> Result<Vec<(usize, usize)>, String> {
        let rows = self.notes.len().checked_add(1).ok_or("too many notes")?;
        let columns = output.len().checked_add(1).ok_or("too many notes")?;
        let cells = rows
            .checked_mul(columns)
            .filter(|cells| *cells <= 16_000_000)
            .ok_or("too many onset associations")?;
        let mut directions = vec![0_u8; cells];
        let mut previous = vec![MatchScore::default(); columns];
        let mut current = previous.clone();
        for i in 1..rows {
            current[0] = MatchScore::default();
            for j in 1..columns {
                let mut best = previous[j];
                let mut direction = 1;
                if current[j - 1].better_than(best) {
                    best = current[j - 1];
                    direction = 2;
                }
                let error = onset_deviation_seconds(
                    self.input_clock,
                    self.notes[i - 1].start,
                    self.output_clock,
                    output[j - 1].start,
                )
                .abs();
                if error <= self.onset_tolerance_seconds {
                    let candidate = MatchScore {
                        count: previous[j - 1].count + 1,
                        total_error: previous[j - 1].total_error + error,
                    };
                    if candidate.better_than(best)
                        || (candidate.count == best.count
                            && candidate.total_error == best.total_error)
                    {
                        best = candidate;
                        direction = 3;
                    }
                }
                current[j] = best;
                directions[i * columns + j] = direction;
            }
            std::mem::swap(&mut previous, &mut current);
        }
        let (mut i, mut j) = (self.notes.len(), output.len());
        let mut pairs = Vec::new();
        while i > 0 && j > 0 {
            match directions[i * columns + j] {
                1 => i -= 1,
                2 => j -= 1,
                3 => {
                    pairs.push((i - 1, j - 1));
                    i -= 1;
                    j -= 1;
                }
                _ => unreachable!("filled association matrix"),
            }
        }
        pairs.reverse();
        Ok(pairs)
    }

    fn validate(&self) -> Result<(), String> {
        for clock in [self.input_clock, self.output_clock] {
            if clock.sample_rate_hz == 0 || clock.sample_rate_hz > MAX_SAMPLE_RATE {
                return Err("sample rate must be in 1..=1000000 Hz".into());
            }
        }
        if self.input_samples == 0
            || self.output_samples == 0
            || self.input_samples > MAX_SAMPLES
            || self.output_samples > MAX_SAMPLES
        {
            return Err("invalid recording length".into());
        }
        if !self.onset_tolerance_seconds.is_finite() || self.onset_tolerance_seconds < 0.0 {
            return Err("invalid onset tolerance".into());
        }
        if input_minus_output_ticks(self.input_clock, 0, self.output_clock, 0) < 0
            || input_minus_output_ticks(
                self.input_clock,
                self.input_samples,
                self.output_clock,
                self.output_samples,
            ) > 0
        {
            return Err("output recording does not cover the input interval".into());
        }
        let mut previous_end = 0;
        for note in self.notes {
            if note.start < previous_end
                || note.start >= note.end
                || note.end > self.input_samples
                || !valid_pitch(note.pitch_midi)
            {
                return Err("invalid or overlapping reference notes".into());
            }
            previous_end = note.end;
        }
        let mut previous_end = 0;
        let mut previous_prefix = 0;
        let mut previous_voiced = false;
        for span in self.output {
            if span.start != previous_end
                || span.start >= span.end
                || span.end > self.output_samples
                || span.input_prefix_end < previous_prefix
                || span.input_prefix_end > self.input_samples
                || span.pitch_midi.is_some_and(|pitch| !valid_pitch(pitch))
                || (span.note_onset && span.pitch_midi.is_none())
                || (span.pitch_midi.is_some() && !previous_voiced && !span.note_onset)
            {
                return Err("invalid output spans or input prefix".into());
            }
            if self.input_mode == InputMode::Causal
                && span.input_prefix_end > 0
                && input_minus_output_ticks(
                    self.input_clock,
                    span.input_prefix_end,
                    self.output_clock,
                    span.start,
                ) > 0
            {
                return Err("input prefix has not arrived at output time".into());
            }
            previous_end = span.end;
            previous_prefix = span.input_prefix_end;
            previous_voiced = span.pitch_midi.is_some();
        }
        if previous_end != self.output_samples {
            return Err("output spans do not cover the output recording".into());
        }
        Ok(())
    }
}

fn direction(semitones: f64) -> i8 {
    if semitones > 0.5 {
        1
    } else if semitones < -0.5 {
        -1
    } else {
        0
    }
}
