//! Idle, verified chart-reader adaptation for the local LFM3 model.
#[cfg(all(test, unix, feature = "cpu"))]
extern crate self as omni_rsi;

#[cfg(all(test, unix, feature = "cpu"))]
#[path = "../tests/support/number_v3_fixture.rs"]
mod number_v3_fixture;

pub mod chart;
pub mod child;
pub mod evolution;
pub mod fingerprint;
pub mod humming;
pub mod humming_audio;
pub mod idle;
pub mod lease;
pub mod model;
pub mod number_audio_v3;
pub mod number_blind_holdout;
pub mod number_development;
pub mod number_formal_holdout;
pub mod number_gate;
pub mod number_holdout_ledger;
pub mod number_v3_evaluation;
mod number_v3_target;
pub mod platform;
mod private_acl;
pub mod reader;
pub mod speech_asr;
pub mod speech_audio;
pub mod speech_audio_checkpoint;
pub mod speech_audio_data;
pub mod speech_audio_round;
pub mod speech_audio_target;
pub mod speech_checkpoint;
pub mod speech_data;
pub mod speech_evaluation;
pub mod speech_generate;
pub mod speech_holdout;
pub mod speech_ingest;
pub mod speech_number_evaluation;
mod speech_pcm;
pub mod speech_publish;
pub mod speech_round;
pub mod speech_seed;
mod speech_target_format;
pub mod speech_tone;
pub mod speech_tone_input;
pub mod speech_training;
pub mod storage;
pub mod watchdog;
