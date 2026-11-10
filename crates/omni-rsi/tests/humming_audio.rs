use omni_rsi::{
    humming::{Clock, Note},
    humming_audio::{AudioEvaluation, PitchConfig, estimate_pitch, synthetic_melody},
};

#[test]
fn a4_sine_is_estimated_from_pcm() {
    let sample_rate = 24_000;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();
    let spans = estimate_pitch(&samples, &PitchConfig::default()).unwrap();
    let middle = spans
        .iter()
        .find(|span| span.start <= 12_000 && span.end > 12_000)
        .unwrap();
    let pitch = middle.pitch_midi.expect("A4 must be voiced");
    assert!((pitch - 69.0).abs() < 0.05, "estimated MIDI {pitch}");
    let hz = middle.f0_hz.expect("A4 must have an F0 estimate");
    assert!((hz - 440.0).abs() < 1.5, "estimated frequency {hz} Hz");
}

#[test]
fn fixed_score_generates_a_harmonic_melody_with_known_rests() {
    let melody = synthetic_melody();
    assert_eq!(melody.sample_rate_hz, 24_000);
    assert_eq!(melody.samples.len(), 75_600);
    assert_eq!(melody.notes.len(), 5);
    assert_eq!(
        melody.notes[3],
        Note {
            start: 45_600,
            end: 57_600,
            pitch_midi: 69.0
        }
    );
    assert_eq!(melody.samples[15_000], 0.0);
    // At 750 samples into A4, the fundamental is -1, the second harmonic
    // is 0 and the third is +1: 0.3 * (-0.6 + 0.1) = -0.15.
    assert!((melody.samples[46_350] + 0.15).abs() < 1e-5);
}

#[test]
fn harmonic_melody_is_measurable_against_its_score() {
    let melody = synthetic_melody();
    let clock = Clock {
        sample_rate_hz: melody.sample_rate_hz,
        origin_ns: 0,
    };
    let result = AudioEvaluation {
        input_clock: clock,
        input_samples: melody.samples.len() as u64,
        notes: &melody.notes,
        output_clock: clock,
        output: &melody.samples,
        pitch_config: &PitchConfig::default(),
        input_prefix_end: melody.samples.len() as u64,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert!(result.metrics.voicing_recall.unwrap() > 0.95);
    assert!(result.metrics.pitch_accuracy.unwrap() > 0.95);
    assert!(result.metrics.voicing_false_alarm_rate.unwrap() < 0.03);
    assert_eq!(result.metrics.missed_notes, 0);
    assert_eq!(result.metrics.extra_onsets, 0);
}

#[test]
fn silence_has_no_pitch_and_keeps_undefined_metrics_in_the_saved_report() {
    let samples = vec![0.0; 2_400];
    let clock = Clock {
        sample_rate_hz: 24_000,
        origin_ns: 0,
    };
    let result = AudioEvaluation {
        input_clock: clock,
        input_samples: 2_400,
        notes: &[],
        output_clock: clock,
        output: &samples,
        pitch_config: &PitchConfig::default(),
        input_prefix_end: 0,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert!(result.spans.iter().all(|span| span.pitch_midi.is_none()));
    let saved = serde_json::to_value(&result.metrics).unwrap();
    assert_eq!(saved["output_voiced_rate"], 0.0);
    assert!(saved["pitch_accuracy"].is_null());
    assert_eq!(saved["input_mode"], "Offline");
}

#[test]
fn octave_up_pcm_is_an_error_even_when_chroma_matches() {
    // The score is A4 from 0.2 to 0.7 s, but the PCM contains A5 there.
    let clock = Clock {
        sample_rate_hz: 24_000,
        origin_ns: 0,
    };
    let notes = [Note {
        start: 4_800,
        end: 16_800,
        pitch_midi: 69.0,
    }];
    let mut samples = vec![0.0; 21_600];
    for i in 4_800..16_800 {
        samples[i] = (2.0 * std::f32::consts::PI * 880.0 * (i - 4_800) as f32 / 24_000.0).sin();
    }
    let result = AudioEvaluation {
        input_clock: clock,
        input_samples: samples.len() as u64,
        notes: &notes,
        output_clock: clock,
        output: &samples,
        pitch_config: &PitchConfig::default(),
        input_prefix_end: samples.len() as u64,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert!(result.metrics.octave_error_rate.unwrap() > 0.98);
    assert!(result.metrics.pitch_accuracy.unwrap() < 0.02);
    assert!(result.metrics.chroma_accuracy.unwrap() > 0.98);
    assert!(result.metrics.voicing_recall.unwrap() > 0.95);
    assert!(result.metrics.voicing_false_alarm_rate.unwrap() < 0.15);
}

#[test]
fn delayed_pcm_keeps_its_measured_onset_offset() {
    let clock = Clock {
        sample_rate_hz: 24_000,
        origin_ns: 0,
    };
    let notes = [Note {
        start: 4_800,
        end: 14_400,
        pitch_midi: 69.0,
    }];
    let mut samples = vec![0.0; 21_600];
    for i in 7_200..16_800 {
        samples[i] = (2.0 * std::f32::consts::PI * 440.0 * (i - 7_200) as f32 / 24_000.0).sin();
    }
    let result = AudioEvaluation {
        input_clock: clock,
        input_samples: samples.len() as u64,
        notes: &notes,
        output_clock: clock,
        output: &samples,
        pitch_config: &PitchConfig::default(),
        input_prefix_end: samples.len() as u64,
        onset_tolerance_seconds: 0.2,
    }
    .evaluate()
    .unwrap();
    let offset = result.metrics.mean_signed_onset_error_seconds.unwrap();
    assert!((offset - 0.1).abs() <= 0.02, "onset offset: {offset}");
    assert!(result.metrics.mean_absolute_duration_error_seconds.unwrap() <= 0.02);
}

#[test]
fn a_legato_semitone_change_is_reported_as_a_new_onset() {
    // Equal-tempered A4 followed immediately by A-sharp4, each 0.5 s.
    let mut samples = vec![0.0; 24_000];
    for (i, value) in samples.iter_mut().enumerate() {
        let (frequency, index) = if i < 12_000 {
            (440.0, i)
        } else {
            (466.163_761_5, i - 12_000)
        };
        *value = (2.0 * std::f64::consts::PI * frequency * index as f64 / 24_000.0).sin() as f32;
    }
    let spans = estimate_pitch(&samples, &PitchConfig::default()).unwrap();
    let onsets: Vec<_> = spans
        .iter()
        .filter(|span| span.note_onset)
        .map(|span| span.start)
        .collect();
    assert_eq!(onsets.len(), 2, "onsets: {onsets:?}");
    assert!(onsets[1].abs_diff(12_000) <= 960, "onsets: {onsets:?}");
}
