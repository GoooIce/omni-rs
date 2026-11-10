//! Summarize a completed native development comparison without loading models.

use omni_lfm3::speech_adapter::SpeechObservation;
use omni_rsi::speech_evaluation::SpeechEvaluation;
use serde_json::Value;
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let path = PathBuf::from(arguments.next().ok_or("provide an epoch report path")?);
    if arguments.next().is_some() {
        return Err("provide exactly one epoch report path".into());
    }
    if fs::metadata(&path)?.len() > 16 * 1024 * 1024 {
        return Err("epoch report is oversized".into());
    }
    let report: Value = serde_json::from_slice(&fs::read(path)?)?;
    let evaluation = SpeechEvaluation::fixed()?;
    if report
        .pointer("/protocol/evaluation_hash")
        .and_then(Value::as_str)
        != Some(evaluation.fingerprint().as_str())
    {
        return Err("report uses a different fixed evaluation set".into());
    }
    let observations: Vec<SpeechObservation> = serde_json::from_value(
        report
            .pointer("/generation_gate/results")
            .cloned()
            .ok_or("missing generation results")?,
    )?;
    let passed = evaluation.passes_development(&observations)?;
    if report
        .pointer("/generation_gate/passed")
        .and_then(Value::as_bool)
        != Some(passed)
    {
        return Err("recorded gate result does not match the observations".into());
    }
    let mut summary = evaluation.summarize(&evaluation.dev, &observations)?;
    summary["passed"] = Value::Bool(passed);
    summary["epoch"] = report.get("epoch").cloned().ok_or("missing epoch")?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
