//! Opt-in stage timings. `sync` drains the backend between stages and therefore
//! diagnoses costs, not end-to-end throughput; `host` preserves normal scheduling.
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use std::{collections::BTreeMap, time::Instant};

#[derive(Debug, serde::Serialize)]
pub struct Profile {
    pub synchronized: bool,
    pub stage_seconds: BTreeMap<&'static str, Vec<f64>>,
}

pub(crate) struct Profiler {
    report: Option<Profile>,
    previous: Instant,
}

impl Profiler {
    pub fn new<B: Backend>(device: &B::Device) -> Result<Self> {
        let mode = std::env::var("BREEZE_TTS_PROFILE").unwrap_or_default();
        let report = if matches!(mode.as_str(), "host" | "sync") {
            if mode == "sync" {
                B::sync(device).map_err(|e| Error::Inference(e.to_string()))?;
            }
            Some(Profile {
                synchronized: mode == "sync",
                stage_seconds: BTreeMap::new(),
            })
        } else {
            None
        };
        Ok(Self {
            report,
            previous: Instant::now(),
        })
    }

    pub fn mark<B: Backend>(&mut self, name: &'static str, device: &B::Device) -> Result<()> {
        if let Some(report) = &mut self.report {
            if report.synchronized {
                B::sync(device).map_err(|e| Error::Inference(e.to_string()))?;
            }
            let now = Instant::now();
            report
                .stage_seconds
                .entry(name)
                .or_default()
                .push(now.duration_since(self.previous).as_secs_f64());
            self.previous = now;
        }
        Ok(())
    }

    pub fn finish(self) -> Option<Profile> {
        self.report
    }
}
