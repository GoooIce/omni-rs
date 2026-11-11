//! Recompute control scores from the actual opened cases and executor outputs.
//! This module cannot issue a qualification or prove who supplied the data.

use super::{
    Executor,
    journal::{Journal, end_value},
};
use crate::{
    number_blind_holdout::{BlindCase, OpenedHoldout},
    number_development::update_pcm_digest,
    number_gate::{PairedCase, PairedOutcome, paired_statistics},
    number_v3_evaluation::TextEnd,
    speech_number_evaluation::{answer_number, mentions_number},
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::Cursor,
    sync::atomic::{AtomicBool, Ordering},
};

const MAX_SAMPLES: usize = 16_000 * 20;
const MAX_TOTAL_SAMPLES: usize = 16_000 * 1_200;
const MAX_OBSERVATIONS: usize = 6_000;
const OBSERVATIONS_PER_CASE: usize = 6;

pub(super) fn check_observation_capacity(case_lower_bound: usize) -> Result<(), String> {
    if case_lower_bound
        .checked_mul(OBSERVATIONS_PER_CASE)
        .is_none_or(|n| n > MAX_OBSERVATIONS)
    {
        Err("holdout observation limit exceeded".into())
    } else {
        Ok(())
    }
}

fn interrupted(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn pcm_hash(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    update_pcm_digest(&mut hash, samples);
    format!("{:x}", hash.finalize())
}

fn load_audio(opened: &OpenedHoldout, stop: &AtomicBool) -> Result<Vec<Vec<f32>>, String> {
    let mut total = 0_usize;
    let mut labels = HashMap::new();
    let mut audio = Vec::with_capacity(opened.manifest().cases.len());
    for (index, case) in opened.manifest().cases.iter().enumerate() {
        interrupted(stop)?;
        let bytes = opened.audio(index).ok_or("missing holdout audio")?;
        let wav = hound::WavReader::new(Cursor::new(bytes)).map_err(|error| error.to_string())?;
        let rate = wav.spec().sample_rate;
        if rate == 0 || u64::from(wav.duration()) > u64::from(rate) * 20 {
            return Err("invalid or overlong holdout WAV".into());
        }
        let predicted =
            usize::try_from((u64::from(wav.duration()) * 16_000).div_ceil(u64::from(rate)))
                .map_err(|_| "holdout sample count overflow")?;
        total = total
            .checked_add(predicted)
            .filter(|n| *n <= MAX_TOTAL_SAMPLES)
            .ok_or("holdout decoded sample budget exceeded")?;
        let samples =
            omni_lfm3::media_io::decode_wav(bytes, stop).map_err(|error| error.to_string())?;
        if samples.is_empty()
            || samples.len() > MAX_SAMPLES
            || samples.len() > predicted
            || samples
                .iter()
                .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
        {
            return Err("invalid decoded holdout audio".into());
        }
        if !samples.iter().any(|sample| *sample != 0.0) {
            return Err("silent holdout original cannot be contrasted with silence".into());
        }
        if let Some(old) = labels.insert(pcm_hash(&samples), case.answer) {
            if old != case.answer {
                return Err("identical holdout PCM has contradictory answers".into());
            }
            return Err("duplicate holdout PCM cannot be counted as independent".into());
        }
        audio.push(samples);
    }
    Ok(audio)
}

/// Sort by answer and case ID, then rotate by the largest answer group. The
/// rotation is bijective and differs in answer exactly when feasible.
fn swap_mapping(cases: &[BlindCase]) -> Result<Vec<usize>, String> {
    let mut order: Vec<_> = (0..cases.len()).collect();
    order.sort_by(|a, b| {
        (cases[*a].answer, &cases[*a].case_id).cmp(&(cases[*b].answer, &cases[*b].case_id))
    });
    let mut largest = 0;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && cases[order[end]].answer == cases[order[start]].answer {
            end += 1;
        }
        largest = largest.max(end - start);
        start = end;
    }
    if cases.len() < 2 || largest > cases.len() / 2 {
        return Err("holdout has no one-to-one different-answer swap".into());
    }
    let mut mapping = vec![0; cases.len()];
    for (position, &source) in order.iter().enumerate() {
        let destination = order[(position + largest) % order.len()];
        if cases[source].answer == cases[destination].answer {
            return Err("invalid holdout swap mapping".into());
        }
        mapping[source] = destination;
    }
    Ok(mapping)
}

pub(super) fn compare(
    opened: &OpenedHoldout,
    executor: &mut Executor,
    stop_tokens: &[u32],
    stop: &AtomicBool,
    journal: &mut Journal,
) -> Result<serde_json::Value, String> {
    let cases = &opened.manifest().cases;
    check_observation_capacity(cases.len())?;
    if stop_tokens.is_empty() {
        return Err("holdout stop policy is invalid".into());
    }
    let swaps = swap_mapping(cases)?;
    let audio = load_audio(opened, stop)?;
    let mut observations = Vec::with_capacity(cases.len() * OBSERVATIONS_PER_CASE);
    let mut counts = [[0_usize; 6]; 2];
    let mut pairs = Vec::with_capacity(cases.len());
    let mut text_bytes = 0_usize;
    for (case_index, case) in cases.iter().enumerate() {
        let silence = vec![0.0; audio[case_index].len()];
        let mut originals = [
            PairedOutcome {
                correct: false,
                normal_end: false,
            },
            PairedOutcome {
                correct: false,
                normal_end: false,
            },
        ];
        for (arm_index, arm) in ["baseline", "candidate"].into_iter().enumerate() {
            for (condition_index, condition, input) in [
                (0, "original", Some(case_index)),
                (1, "swapped", Some(swaps[case_index])),
                (2, "silence", None),
            ] {
                interrupted(stop)?;
                let samples = input.map_or(silence.as_slice(), |index| audio[index].as_slice());
                let pcm = pcm_hash(samples);
                // This identity is shared by both audit records and the
                // eventual report so their input fields cannot drift apart.
                let observation_id = observations.len();
                let identity = serde_json::json!({"observation_id":observation_id,
                    "arm":arm, "case_id":case.case_id, "condition":condition,
                    "audio_case_id":input.map(|i| &cases[i].case_id),
                    "expected_answer":input.map(|i| cases[i].answer),
                    "audio_file_sha256":input.map(|i| &cases[i].audio_sha256),
                    "audio_recording_sha256":input.map(|i| &cases[i].recording_sha256),
                    "audio_pcm_sha256":pcm, "samples":samples.len()});
                let mut started = identity.clone();
                started["record"] = "inference_started".into();
                journal.append(&started)?;
                let result = executor.infer(arm, samples, stop, journal.stream(observation_id))?;
                #[cfg(all(test, unix))]
                if std::env::var_os("OMNI_RSI_TEST_CRASH_AFTER_RETURN").as_deref()
                    == Some(std::ffi::OsStr::new("yes"))
                {
                    unsafe { libc::_exit(75) }
                }
                let (text, text_end) = result.into_parts();
                let mut raw = identity.clone();
                raw["record"] = "observation".into();
                raw["text"] = serde_json::json!(text);
                raw["text_complete"] = true.into();
                raw["text_bytes"] = serde_json::json!(text.len());
                raw["end"] = end_value(&text_end);
                journal.append(&raw)?;
                interrupted(stop)?;
                text_bytes = text_bytes
                    .checked_add(text.len())
                    .ok_or("output text overflow")?;
                if text.len() > 4096 || text_bytes > 16 * 1024 * 1024 {
                    return Err("holdout output text exceeds limit".into());
                }
                let (normal_end, end) = match text_end {
                    TextEnd::Stop(token) if stop_tokens.contains(&token) => {
                        (true, serde_json::json!({"stop_token":token}))
                    }
                    TextEnd::Stop(_) => return Err("stop token is not in the frozen policy".into()),
                    TextEnd::Limit => (false, serde_json::json!({"kind":"limit"})),
                    TextEnd::Cancelled => return Err("interrupted".into()),
                };
                let correct = input
                    .map(|index| normal_end && answer_number(&text) == Some(cases[index].answer));
                if condition_index < 2 {
                    counts[arm_index][condition_index] += usize::from(correct == Some(true));
                } else {
                    counts[arm_index][2] += usize::from(mentions_number(&text));
                }
                counts[arm_index][3 + condition_index] += usize::from(normal_end);
                if condition_index == 0 {
                    originals[arm_index] = PairedOutcome {
                        correct: correct == Some(true),
                        normal_end,
                    };
                }
                let mut observation = identity;
                observation["text"] = serde_json::json!(text);
                observation["end"] = end;
                observation["normal_end"] = serde_json::json!(normal_end);
                observation["correct"] = serde_json::json!(correct);
                observations.push(observation);
            }
        }
        pairs.push(PairedCase {
            case_id: case.case_id.clone(),
            original_id: case.recording_sha256.clone(),
            family_id: case.family_sha256.clone(),
            speaker_id: case.speaker_sha256.clone(),
            session_id: case.session_sha256.clone(),
            baseline: originals[0].clone(),
            candidate: originals[1].clone(),
        });
    }
    let stats = paired_statistics(&pairs)?;
    let metrics = |values: [usize; 6]| {
        serde_json::json!({
            "original_correct":values[0],"swapped_correct":values[1],"silence_numeric_responses":values[2],
            "normal_end":{"original":values[3],"swapped":values[4],"silence":values[5]}
        })
    };
    Ok(serde_json::json!({
        "status":"insufficient_evidence", "candidate_eligible":false,
        "execution":executor.execution(), "baseline":metrics(counts[0]), "candidate":metrics(counts[1]),
        "observations":observations, "paired_statistics":{"n":stats.n,
            "independence":"conditional_on_unverified_source_declarations",
            "baseline_correct":stats.baseline_correct,"candidate_correct":stats.candidate_correct,
            "candidate_only":stats.candidate_only,"baseline_only":stats.baseline_only,
            "improvement":stats.improvement,"interval_95":stats.interval_95},
        "output_speech_r2t2_cer":"not_run", "human_listening":"not_verified",
        "source_authorization":"not_verified", "registered_regressions":"not_verified"
    }))
}
