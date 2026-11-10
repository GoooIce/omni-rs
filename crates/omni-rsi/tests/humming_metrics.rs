use omni_rsi::humming::{Clock, Evaluation, InputMode, Note, OutputSpan, pitch_midi_from_hz};

#[test]
fn differently_clocked_recordings_preserve_aligned_voicing() {
    // Input spans 0..2 s. The output clock starts 250 ms earlier at 24 kHz;
    // its 12,000..24,000 samples therefore cover input time 0.25..0.75 s.
    let notes = [Note {
        start: 4_000,
        end: 12_000,
        pitch_midi: 69.0,
    }];
    let output = [
        OutputSpan {
            start: 0,
            end: 12_000,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 12_000,
            end: 24_000,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 4_000,
        },
        OutputSpan {
            start: 24_000,
            end: 54_000,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 12_000,
        },
    ];
    let report = Evaluation {
        input_clock: Clock {
            sample_rate_hz: 16_000,
            origin_ns: 0,
        },
        input_samples: 32_000,
        notes: &notes,
        output_clock: Clock {
            sample_rate_hz: 24_000,
            origin_ns: -250_000_000,
        },
        output_samples: 54_000,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.output_voiced_rate, 0.25);
    assert_eq!(report.voicing_recall, Some(1.0));
    assert_eq!(report.voicing_false_alarm_rate, Some(0.0));
}

#[test]
fn missed_notes_and_singing_during_rest_are_both_visible() {
    // Four seconds: two one-second reference notes separated by rests. Output
    // sings the first note, misses the second and sings during the last rest.
    let notes = [
        Note {
            start: 0,
            end: 100,
            pitch_midi: 69.0,
        },
        Note {
            start: 200,
            end: 300,
            pitch_midi: 71.0,
        },
    ];
    let output = [
        OutputSpan {
            start: 0,
            end: 100,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 100,
            end: 300,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 100,
        },
        OutputSpan {
            start: 300,
            end: 400,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 300,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 400,
        notes: &notes,
        output_clock: clock,
        output_samples: 400,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.reference_voiced_seconds, 2.0);
    assert_eq!(report.output_voiced_seconds, 2.0);
    assert_eq!(report.voiced_overlap_seconds, 1.0);
    assert_eq!(report.voicing_recall, Some(0.5));
    assert_eq!(report.voicing_precision, Some(0.5));
    assert_eq!(report.voicing_false_alarm_rate, Some(0.5));
}

#[test]
fn octave_errors_remain_errors_even_when_chroma_is_correct() {
    // A4 (440 Hz) for two seconds; the second second is A5 (880 Hz).
    let notes = [Note {
        start: 0,
        end: 200,
        pitch_midi: 69.0,
    }];
    let output = [
        OutputSpan {
            start: 0,
            end: 100,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 100,
            end: 200,
            pitch_midi: Some(81.0),
            note_onset: false,
            input_prefix_end: 100,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 200,
        notes: &notes,
        output_clock: clock,
        output_samples: 200,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.mean_absolute_pitch_error_cents, Some(600.0));
    assert_eq!(report.pitch_accuracy, Some(0.5));
    assert_eq!(report.chroma_accuracy, Some(1.0));
    assert_eq!(report.octave_error_rate, Some(0.5));
}

#[test]
fn fixed_delay_is_reported_on_the_original_clock() {
    // Three one-second notes, all played 250 ms late, with a silent tail.
    let notes = [
        Note {
            start: 0,
            end: 100,
            pitch_midi: 69.0,
        },
        Note {
            start: 100,
            end: 200,
            pitch_midi: 71.0,
        },
        Note {
            start: 200,
            end: 300,
            pitch_midi: 67.0,
        },
    ];
    let output = [
        OutputSpan {
            start: 0,
            end: 25,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 25,
            end: 125,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 25,
        },
        OutputSpan {
            start: 125,
            end: 225,
            pitch_midi: Some(71.0),
            note_onset: true,
            input_prefix_end: 125,
        },
        OutputSpan {
            start: 225,
            end: 325,
            pitch_midi: Some(67.0),
            note_onset: true,
            input_prefix_end: 225,
        },
        OutputSpan {
            start: 325,
            end: 400,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 325,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 400,
        notes: &notes,
        output_clock: clock,
        output_samples: 400,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.3,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.onset_matches.len(), 3);
    assert_eq!(report.missed_notes, 0);
    assert_eq!(report.extra_onsets, 0);
    assert_eq!(report.mean_signed_onset_error_seconds, Some(0.25));
    assert_eq!(report.mean_absolute_onset_error_seconds, Some(0.25));
    assert_eq!(report.mean_absolute_duration_error_seconds, Some(0.0));
    assert_eq!(report.onset_matches[0].input_prefix_end, 25);
}

#[test]
fn melody_contour_detects_reversed_movements() {
    // Reference rises, falls, then stays level; output falls, rises, stays level.
    let notes = [
        Note {
            start: 0,
            end: 100,
            pitch_midi: 69.0,
        },
        Note {
            start: 100,
            end: 200,
            pitch_midi: 71.0,
        },
        Note {
            start: 200,
            end: 300,
            pitch_midi: 67.0,
        },
        Note {
            start: 300,
            end: 400,
            pitch_midi: 67.0,
        },
    ];
    let output = [
        OutputSpan {
            start: 0,
            end: 100,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 100,
            end: 200,
            pitch_midi: Some(65.0),
            note_onset: true,
            input_prefix_end: 100,
        },
        OutputSpan {
            start: 200,
            end: 300,
            pitch_midi: Some(67.0),
            note_onset: true,
            input_prefix_end: 200,
        },
        OutputSpan {
            start: 300,
            end: 400,
            pitch_midi: Some(67.1),
            note_onset: true,
            input_prefix_end: 300,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 400,
        notes: &notes,
        output_clock: clock,
        output_samples: 400,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.contour_reference_pairs, 3);
    assert_eq!(report.contour_compared_pairs, 3);
    assert_eq!(report.contour_correct_pairs, 1);
    assert_eq!(report.contour_accuracy, Some(1.0 / 3.0));
}

#[test]
fn known_fundamentals_convert_to_midi_without_treating_silence_as_pitch() {
    assert_eq!(pitch_midi_from_hz(440.0).unwrap(), 69.0);
    assert_eq!(pitch_midi_from_hz(880.0).unwrap(), 81.0);
    assert!(pitch_midi_from_hz(0.0).is_err());
    assert!(pitch_midi_from_hz(f64::NAN).is_err());
}

#[test]
fn causal_input_prefix_cannot_arrive_after_the_output_begins() {
    let notes = [Note {
        start: 0,
        end: 100,
        pitch_midi: 69.0,
    }];
    let output = [OutputSpan {
        start: 0,
        end: 100,
        pitch_midi: Some(69.0),
        note_onset: true,
        input_prefix_end: 1,
    }];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let mut evaluation = Evaluation {
        input_clock: clock,
        input_samples: 100,
        notes: &notes,
        output_clock: clock,
        output_samples: 100,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    };
    assert!(evaluation.evaluate().is_err()); // One input sample arrives 10 ms too late.
    evaluation.input_mode = InputMode::Offline;
    assert_eq!(
        evaluation.evaluate().unwrap().input_mode,
        InputMode::Offline
    );
}

#[test]
fn silent_output_reports_missed_notes_instead_of_perfect_pitch() {
    let notes = [Note {
        start: 0,
        end: 100,
        pitch_midi: 69.0,
    }];
    let output = [OutputSpan {
        start: 0,
        end: 200,
        pitch_midi: None,
        note_onset: false,
        input_prefix_end: 0,
    }];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 200,
        notes: &notes,
        output_clock: clock,
        output_samples: 200,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.output_voiced_rate, 0.0);
    assert_eq!(report.voicing_recall, Some(0.0));
    assert_eq!(report.voicing_precision, None);
    assert_eq!(report.pitch_accuracy, None);
    assert_eq!(report.onset_recall, Some(0.0));
    assert_eq!(report.onset_precision, None);
    assert_eq!(report.missed_notes, 1);
    assert_eq!(report.mean_signed_onset_error_seconds, None);
}

#[test]
fn singing_against_silent_reference_is_a_false_alarm() {
    let output = [OutputSpan {
        start: 0,
        end: 100,
        pitch_midi: Some(69.0),
        note_onset: true,
        input_prefix_end: 0,
    }];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 100,
        notes: &[],
        output_clock: clock,
        output_samples: 100,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.voicing_recall, None);
    assert_eq!(report.voicing_false_alarm_rate, Some(1.0));
    assert_eq!(report.voicing_precision, Some(0.0));
    assert_eq!(report.pitch_accuracy, None);
    assert_eq!(report.extra_onsets, 1);
    assert_eq!(report.onset_precision, Some(0.0));
    assert_eq!(report.onset_recall, None);
}

#[test]
fn causal_prefix_boundary_uses_both_rates_and_clock_origins() {
    // Output sample 11 at 1 kHz with an origin at -10 ms is input time 1 ms.
    // Exactly 16 samples have arrived on the 16 kHz input clock, but not 17.
    let mut output = [
        OutputSpan {
            start: 0,
            end: 11,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 11,
            end: 1_010,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 16,
        },
    ];
    let evaluate = |output: &[OutputSpan]| {
        Evaluation {
            input_clock: Clock {
                sample_rate_hz: 16_000,
                origin_ns: 0,
            },
            input_samples: 16_000,
            notes: &[],
            output_clock: Clock {
                sample_rate_hz: 1_000,
                origin_ns: -10_000_000,
            },
            output_samples: 1_010,
            output,
            input_mode: InputMode::Causal,
            onset_tolerance_seconds: 0.1,
        }
        .evaluate()
    };
    assert!(evaluate(&output).is_ok());
    output[1].input_prefix_end = 17;
    assert!(evaluate(&output).is_err());
}

#[test]
fn incomplete_or_invalid_annotations_cannot_be_silently_scored() {
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let valid = OutputSpan {
        start: 0,
        end: 100,
        pitch_midi: None,
        note_onset: false,
        input_prefix_end: 0,
    };
    let evaluate = |notes: &[Note], output: &[OutputSpan], output_samples| {
        Evaluation {
            input_clock: clock,
            input_samples: 100,
            notes,
            output_clock: clock,
            output_samples,
            output,
            input_mode: InputMode::Causal,
            onset_tolerance_seconds: 0.1,
        }
        .evaluate()
    };
    assert!(evaluate(&[], &[OutputSpan { end: 99, ..valid }], 99).is_err());
    assert!(evaluate(&[], &[OutputSpan { start: 1, ..valid }], 100).is_err());
    assert!(
        evaluate(
            &[],
            &[OutputSpan {
                pitch_midi: Some(f64::NAN),
                ..valid
            }],
            100
        )
        .is_err()
    );
    assert!(
        evaluate(
            &[],
            &[OutputSpan {
                input_prefix_end: 101,
                ..valid
            }],
            100
        )
        .is_err()
    );
    let overlapping = [
        Note {
            start: 0,
            end: 60,
            pitch_midi: 69.0,
        },
        Note {
            start: 59,
            end: 80,
            pitch_midi: 71.0,
        },
    ];
    assert!(evaluate(&overlapping, &[valid], 100).is_err());
}

#[test]
fn note_duration_is_unknown_when_recording_ends_while_still_voiced() {
    let notes = [Note {
        start: 0,
        end: 100,
        pitch_midi: 69.0,
    }];
    let output = [OutputSpan {
        start: 0,
        end: 100,
        pitch_midi: Some(69.0),
        note_onset: true,
        input_prefix_end: 0,
    }];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 100,
        notes: &notes,
        output_clock: clock,
        output_samples: 100,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.duration_compared_notes, 0);
    assert_eq!(report.mean_absolute_duration_error_seconds, None);
    assert_eq!(report.onset_matches[0].duration_error_seconds, None);
}

#[test]
fn onset_exactly_on_tolerance_is_not_lost_to_float_cancellation() {
    // 400 ms - 300 ms is exactly the allowed 100 ms on these clocks.
    let notes = [Note {
        start: 30,
        end: 50,
        pitch_midi: 69.0,
    }];
    let output = [
        OutputSpan {
            start: 0,
            end: 40,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 40,
            end: 60,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 40,
        },
        OutputSpan {
            start: 60,
            end: 100,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 60,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 100,
        notes: &notes,
        output_clock: clock,
        output_samples: 100,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.onset_matches.len(), 1);
    assert_eq!(report.mean_signed_onset_error_seconds, Some(0.1));
}

#[test]
fn onset_association_prefers_the_closest_note_and_does_not_bridge_missing_contour() {
    let notes = [
        Note {
            start: 0,
            end: 10,
            pitch_midi: 69.0,
        },
        Note {
            start: 20,
            end: 30,
            pitch_midi: 71.0,
        },
        Note {
            start: 40,
            end: 50,
            pitch_midi: 72.0,
        },
    ];
    let output = [
        OutputSpan {
            start: 0,
            end: 15,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 15,
            end: 25,
            pitch_midi: Some(71.0),
            note_onset: true,
            input_prefix_end: 15,
        },
        OutputSpan {
            start: 25,
            end: 100,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 25,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 100,
        notes: &notes,
        output_clock: clock,
        output_samples: 100,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.2,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.onset_matches.len(), 1);
    assert_eq!(report.onset_matches[0].reference_note, 1);
    assert_eq!(report.missed_notes, 2);
    assert_eq!(report.contour_reference_pairs, 2);
    assert_eq!(report.contour_compared_pairs, 0);
    assert_eq!(report.contour_accuracy, None);
}

#[test]
fn onset_association_maximizes_matches_before_minimizing_error() {
    let notes = [
        Note {
            start: 0,
            end: 10,
            pitch_midi: 69.0,
        },
        Note {
            start: 20,
            end: 30,
            pitch_midi: 71.0,
        },
    ];
    let output = [
        OutputSpan {
            start: 0,
            end: 15,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 15,
            end: 25,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 15,
        },
        OutputSpan {
            start: 25,
            end: 35,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 25,
        },
        OutputSpan {
            start: 35,
            end: 45,
            pitch_midi: Some(71.0),
            note_onset: true,
            input_prefix_end: 35,
        },
        OutputSpan {
            start: 45,
            end: 100,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 45,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 100,
        notes: &notes,
        output_clock: clock,
        output_samples: 100,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.2,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.onset_matches.len(), 2);
    assert_eq!(report.onset_matches[0].reference_note, 0);
    assert_eq!(report.onset_matches[1].reference_note, 1);
    assert_eq!(report.mean_signed_onset_error_seconds, Some(0.15));
}

#[test]
fn pitch_and_octave_thresholds_have_independent_denominators() {
    // Four equal durations: +50, +51, +1250, +1251 cents from A4.
    let notes = [Note {
        start: 0,
        end: 400,
        pitch_midi: 69.0,
    }];
    let output = [
        OutputSpan {
            start: 0,
            end: 100,
            pitch_midi: Some(69.5),
            note_onset: true,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 100,
            end: 200,
            pitch_midi: Some(69.51),
            note_onset: false,
            input_prefix_end: 100,
        },
        OutputSpan {
            start: 200,
            end: 300,
            pitch_midi: Some(81.5),
            note_onset: false,
            input_prefix_end: 200,
        },
        OutputSpan {
            start: 300,
            end: 400,
            pitch_midi: Some(81.51),
            note_onset: false,
            input_prefix_end: 300,
        },
    ];
    let clock = Clock {
        sample_rate_hz: 100,
        origin_ns: 0,
    };
    let report = Evaluation {
        input_clock: clock,
        input_samples: 400,
        notes: &notes,
        output_clock: clock,
        output_samples: 400,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert!((report.mean_absolute_pitch_error_cents.unwrap() - 650.5).abs() < 1e-9);
    assert_eq!(report.pitch_accuracy, Some(0.25));
    assert_eq!(report.chroma_accuracy, Some(0.5));
    assert_eq!(report.octave_error_rate, Some(0.25));
}

#[test]
fn identical_notes_have_exact_voicing_recall_at_fractional_seconds() {
    // A one-sample note from 2/3 to 3/3 seconds: both tracks are identical.
    let clock = Clock {
        sample_rate_hz: 3,
        origin_ns: 0,
    };
    let notes = [Note {
        start: 2,
        end: 3,
        pitch_midi: 69.0,
    }];
    let output = [
        OutputSpan {
            start: 0,
            end: 2,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 0,
        },
        OutputSpan {
            start: 2,
            end: 3,
            pitch_midi: Some(69.0),
            note_onset: true,
            input_prefix_end: 2,
        },
        OutputSpan {
            start: 3,
            end: 6,
            pitch_midi: None,
            note_onset: false,
            input_prefix_end: 3,
        },
    ];
    let report = Evaluation {
        input_clock: clock,
        input_samples: 6,
        notes: &notes,
        output_clock: clock,
        output_samples: 6,
        output: &output,
        input_mode: InputMode::Causal,
        onset_tolerance_seconds: 0.1,
    }
    .evaluate()
    .unwrap();
    assert_eq!(report.voicing_recall, Some(1.0));
    assert_eq!(
        report.voiced_overlap_seconds,
        report.reference_voiced_seconds
    );
}
