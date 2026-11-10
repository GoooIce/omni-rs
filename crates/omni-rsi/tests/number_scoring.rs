//! Synthetic rows exercise the rule scorer; they are not model-quality evidence.
use omni_rsi::{
    speech_audio_data::AudioDataset,
    speech_number_evaluation::{EvaluationBinary, NumberObservation, summarize},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, sync::atomic::AtomicBool};

fn fixture() -> (tempfile::TempDir, AudioDataset) {
    fixture_with_dev((12, "十二", "十三"), (15, "十五", "十六"))
}

fn fixture_with_dev(
    first: (u32, &str, &str),
    second: (u32, &str, &str),
) -> (tempfile::TempDir, AudioDataset) {
    let root = tempfile::tempdir().unwrap();
    let mut records = Vec::new();
    for (index, split, number, spoken, answer) in [
        (1_i16, "train", 7_u32, "七", "八"),
        (2, "dev", first.0, first.1, first.2),
        (3, "dev", second.0, second.1, second.2),
        (4, "test", 18, "十八", "十九"),
    ] {
        let name = format!("case-{index}.wav");
        let path = root.path().join(&name);
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in [0_i16, index * 3000, -index * 3000, 0] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        records.push(json!({
            "id": format!("case-{index}"), "split": split, "input_audio": name,
            "input_sha256": format!("{:x}", Sha256::digest(fs::read(path).unwrap())),
            "authorization": {"kind":"licensed-synthetic", "source":"test fixture", "license":"CC0-1.0"},
            "instruction":"把听到的数字加一，用中文回答。", "answer":answer,
            "rule":"number-plus-one-v1", "spoken_number":number, "spoken_text":spoken,
            "provenance": {"speaker_id":format!("speaker-{index}"), "session_id":format!("session-{index}"),
                "recording_id":format!("recording-{index}"), "family_id":format!("family-{index}"),
                "teacher_sha256":"1".repeat(64), "verifier_sha256":"2".repeat(64),
                "verified_text":spoken, "created_at":"2026-11-10T00:00:00Z"}
        }));
    }
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&json!({"version":2,"records":records})).unwrap(),
    )
    .unwrap();
    let data = AudioDataset::load_rule_split(root.path(), "dev", &AtomicBool::new(false)).unwrap();
    (root, data)
}

#[test]
fn one_ten_and_one_hundred_match_their_decimal_forms() {
    for (first, second, decimal_first, decimal_second) in [
        ((0, "零", "一"), (9, "九", "十"), "1", "10"),
        ((9, "九", "十"), (99, "九十九", "一百"), "１０", "100"),
    ] {
        let (_root, data) = fixture_with_dev(first, second);
        let mut observations = rows();
        for row in &mut observations {
            row.text = match (row.case_id.as_str(), row.condition.as_str()) {
                ("case-2", "original") | ("case-3", "swapped") => decimal_first,
                ("case-3", "original") | ("case-2", "swapped") => decimal_second,
                _ => "不知道",
            }
            .into();
        }
        let report = summarize(&data, &observations).unwrap();
        assert_eq!(report["baseline"]["original_correct"], 2);
        assert_eq!(report["candidate"]["swapped_correct"], 2);
    }
}

fn rows() -> Vec<NumberObservation> {
    let mut rows = Vec::new();
    for arm in ["baseline", "candidate"] {
        for (case, swapped, answer, other) in [
            ("case-2", "case-3", "13", "16"),
            ("case-3", "case-2", "１６。", "１３。"),
        ] {
            for (condition, audio, text) in [
                ("original", Some(case), answer),
                ("swapped", Some(swapped), other),
                ("silence", None, "不知道"),
            ] {
                rows.push(NumberObservation {
                    arm: arm.into(),
                    case_id: case.into(),
                    condition: condition.into(),
                    audio_case_id: audio.map(str::to_owned),
                    text: text.into(),
                    stopped: true,
                });
            }
        }
    }
    rows
}

#[test]
fn arabic_and_chinese_numbers_are_scored_by_the_actual_audio_answer() {
    let (_root, data) = fixture();
    let observations = rows();
    let report = summarize(&data, &observations).unwrap();
    assert_eq!(report["baseline"]["original_correct"], 2);
    assert_eq!(report["candidate"]["swapped_correct"], 2);
    assert_eq!(report["observations"][0]["text"], "13");
    assert_eq!(report["observations"][3]["text"], "１６。");
    assert_eq!(report["candidate_eligible"], false);
}

#[test]
fn standalone_quoted_answer_is_accepted_but_wrong_or_ambiguous_numbers_are_not() {
    let (_root, data) = fixture();
    let mut observations = rows();
    for row in &mut observations {
        if row.arm == "candidate" {
            row.text = match (row.case_id.as_str(), row.condition.as_str()) {
                ("case-2", "original") => "“十三”。",
                ("case-3", "original") => "十五",
                ("case-2", "swapped") => "十六和十五",
                ("case-3", "swapped") => "十三点五",
                _ => row.text.as_str(),
            }
            .into();
        }
    }
    let report = summarize(&data, &observations).unwrap();
    assert_eq!(report["candidate"]["original_correct"], 1);
    assert_eq!(report["candidate"]["swapped_correct"], 0);
    assert_eq!(report["status"], "insufficient_evidence");
}

#[test]
fn thirteen_and_fifteen_are_not_the_same_rule_answer() {
    let (_root, data) = fixture();
    let mut observations = rows();
    for row in &mut observations {
        if row.arm == "candidate" && row.case_id == "case-2" && row.condition == "original" {
            row.text = "十五".into();
        }
    }
    let report = summarize(&data, &observations).unwrap();
    assert_eq!(report["candidate"]["original_correct"], 1);
}

#[test]
fn explicit_answer_labels_are_scored_without_accepting_negations_or_alternatives() {
    let (_root, data) = fixture();
    let mut observations = rows();
    for row in &mut observations {
        if row.arm == "candidate" {
            row.text = match (row.case_id.as_str(), row.condition.as_str()) {
                ("case-2", "original") => "答案是13。",
                ("case-3", "original") => "结果为“十六”。",
                ("case-2", "swapped") => "答案不是十六。",
                ("case-3", "swapped") => "结果为十三或十五。",
                _ => row.text.as_str(),
            }
            .into();
        }
    }
    let report = summarize(&data, &observations).unwrap();
    assert_eq!(report["candidate"]["original_correct"], 2);
    assert_eq!(report["candidate"]["swapped_correct"], 0);
}

#[test]
fn a_correct_number_without_normal_end_is_still_wrong() {
    let (_root, data) = fixture();
    let mut observations = rows();
    for row in &mut observations {
        if row.arm == "candidate" && row.case_id == "case-2" && row.condition == "original" {
            row.stopped = false;
        }
    }
    let report = summarize(&data, &observations).unwrap();
    assert_eq!(report["candidate"]["original_correct"], 1);
    assert_eq!(report["baseline"]["normal_end"], 6);
    assert_eq!(report["candidate"]["normal_end"], 5);
}

#[test]
fn silence_reports_numbers_anywhere_in_raw_text_even_when_generation_did_not_end() {
    let (_root, data) = fixture();
    let mut observations = rows();
    for row in &mut observations {
        if row.arm == "baseline" && row.condition == "silence" {
            row.text = if row.case_id == "case-2" {
                "我听到的是0，加一就是1。"
            } else {
                "我听到的是“零”，加一就是“一”。"
            }
            .into();
            if row.case_id == "case-2" {
                row.stopped = false;
            }
        }
    }
    let report = summarize(&data, &observations).unwrap();
    assert_eq!(report["baseline"]["silence_numeric_responses"], 2);
    assert_eq!(report["baseline"]["silence_numeric_rate"], 1.0);
    assert_eq!(report["candidate"]["silence_numeric_responses"], 0);
    assert!(
        report["baseline"]
            .get("silence_matching_original")
            .is_none()
    );
    assert_eq!(
        report["observations"][2]["text"],
        "我听到的是0，加一就是1。"
    );
    assert_eq!(
        report["observations"][5]["text"],
        "我听到的是“零”，加一就是“一”。"
    );
}

#[test]
fn repeating_the_input_and_inventing_silence_numbers_cannot_look_like_a_gain() {
    let (_root, data) = fixture();
    let mut observations = rows();
    for row in &mut observations {
        row.text = match (row.case_id.as_str(), row.condition.as_str()) {
            ("case-2", "original") | ("case-3", "swapped") => "十二。",
            ("case-3", "original") | ("case-2", "swapped") => "十五。",
            ("case-2", "silence") => "我听到的是0，加一就是1。",
            ("case-3", "silence") => "我听到的是“零”，加一就是“一”。",
            _ => unreachable!(),
        }
        .into();
    }
    let report = summarize(&data, &observations).unwrap();
    for arm in ["baseline", "candidate"] {
        assert_eq!(report[arm]["original_correct"], 0);
        assert_eq!(report[arm]["swapped_correct"], 0);
        assert_eq!(report[arm]["silence_numeric_responses"], 2);
    }
    assert_eq!(report["candidate_eligible"], false);
}

#[test]
fn silence_recognizes_english_number_words_without_matching_word_fragments() {
    let (_root, data) = fixture();
    let mut observations = rows();
    for row in &mut observations {
        if row.arm == "baseline" && row.condition == "silence" {
            row.text = if row.case_id == "case-2" {
                "I heard One."
            } else {
                "I heard someone."
            }
            .into();
        }
    }
    let report = summarize(&data, &observations).unwrap();
    assert_eq!(report["baseline"]["silence_numeric_responses"], 1);
}

#[test]
fn caller_supplied_rows_record_the_scoring_protocol_but_cannot_claim_a_verified_run() {
    let (_root, data) = fixture();
    let report = summarize(&data, &rows()).unwrap();
    assert_eq!(
        report["scoring_protocol"]["answer"],
        "single-integer-or-answer-label-0-100-v1"
    );
    assert_eq!(report["scoring_protocol"]["silence"], "numeric-mention-v1");
    assert_eq!(
        report["scoring_protocol"]["normal_end_required_for_correct_answer"],
        true
    );
    assert_eq!(
        report["scoring_protocol"]["silence_includes_unfinished"],
        true
    );
    let hash = report["scoring_protocol"]["implementation_sha256"]
        .as_str()
        .unwrap();
    assert_eq!(hash.len(), 64);
    assert!(
        hash.bytes()
            .all(|ch| ch.is_ascii_digit() || (b'a'..=b'f').contains(&ch))
    );
    assert_eq!(report["status"], "insufficient_evidence");
    assert_eq!(report["candidate_eligible"], false);
}

#[test]
fn ambiguous_or_decorated_numerals_do_not_become_a_different_integer() {
    let (_root, data) = fixture();
    for response in ["1³", "①③", "13.0", "1,3", "-13", "不是十三", "十三和十五"] {
        let mut observations = rows();
        for row in &mut observations {
            if row.arm == "candidate" && row.case_id == "case-2" && row.condition == "original" {
                row.text = response.into();
            }
        }
        let report = summarize(&data, &observations).unwrap();
        assert_eq!(report["candidate"]["original_correct"], 1, "{response}");
    }
}

#[test]
fn evaluation_binary_binding_detects_file_changes_and_never_trusts_a_supplied_hash() {
    let root = tempfile::tempdir().unwrap();
    let binary = root.path().join("evaluator");
    fs::write(&binary, b"abc").unwrap();
    let binding = EvaluationBinary::capture(&binary).unwrap();
    assert_eq!(
        binding.sha256(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    binding.verify().unwrap();
    fs::write(&binary, b"abd").unwrap();
    assert!(binding.verify().is_err());
    assert!(EvaluationBinary::capture(root.path()).is_err());
}

#[cfg(unix)]
#[test]
fn evaluation_binary_binding_rejects_pipes_without_waiting_for_a_writer() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("not-a-binary");
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(EvaluationBinary::capture(&path).is_err());
}
