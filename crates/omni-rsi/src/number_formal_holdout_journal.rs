//! Private, exclusive, incrementally synchronized raw attempt observations.
//! This journal is never a successful holdout report.

use crate::number_v3_evaluation::TextEnd;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, path::Path};

const MAX_JOURNAL_BYTES: usize = 128 * 1024 * 1024;
pub(super) const MAX_RECORDED_TEXT_BYTES: usize = 4096;
const MAX_STREAM_BYTES: usize = 64 * 1024;
const MAX_STREAM_FRAGMENTS: usize = 256;

pub(super) fn bounded_text(text: &str) -> (&str, bool) {
    if text.len() <= MAX_RECORDED_TEXT_BYTES {
        return (text, true);
    }
    let mut end = MAX_RECORDED_TEXT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (&text[..end], false)
}

pub(super) struct Journal {
    file: File,
    hash: Sha256,
    written: usize,
}

/// This value can be created only after every returned fragment and its end
/// marker have been synchronized by `ObservationStream`.
pub(super) struct CommittedText {
    text: String,
    end: TextEnd,
}

impl CommittedText {
    pub(super) fn into_parts(self) -> (String, TextEnd) {
        (self.text, self.end)
    }
}

pub(super) struct ObservationStream<'a> {
    journal: &'a mut Journal,
    observation_id: usize,
    text: String,
    fragments: usize,
    failed: bool,
}

pub(super) fn end_value(end: &TextEnd) -> serde_json::Value {
    match end {
        TextEnd::Stop(token) => serde_json::json!({"stop_token":token}),
        TextEnd::Limit => serde_json::json!({"kind":"limit"}),
        TextEnd::Cancelled => serde_json::json!({"kind":"cancelled"}),
    }
}

// A future trusted executor must use this seam; today it is exercised only by
// the synthetic executor and cannot be reached through a production permit.
#[cfg_attr(not(test), allow(dead_code))]
impl ObservationStream<'_> {
    pub(super) fn push(&mut self, fragment: &str) -> Result<(), String> {
        if self.failed {
            return Err("holdout stream is already incomplete".into());
        }
        if fragment.is_empty() {
            return Ok(());
        }
        if self.fragments >= MAX_STREAM_FRAGMENTS {
            self.failed = true;
            self.journal.append(&serde_json::json!({
                "record":"stream_incomplete", "observation_id":self.observation_id,
                "reason":"fragment_limit", "captured_bytes":self.text.len(),
                "offered_bytes":fragment.len()
            }))?;
            return Err("holdout stream fragment limit exceeded".into());
        }
        let available = MAX_STREAM_BYTES - self.text.len();
        let mut end = fragment.len().min(available);
        while !fragment.is_char_boundary(end) {
            end -= 1;
        }
        if end > 0 {
            let part = &fragment[..end];
            self.failed = true;
            self.journal.append(&serde_json::json!({
                "record":"stream_fragment", "observation_id":self.observation_id,
                "sequence":self.fragments, "offset_bytes":self.text.len(), "text":part
            }))?;
            self.text.push_str(part);
            self.fragments += 1;
            self.failed = false;
        }
        if end != fragment.len() {
            self.failed = true;
            self.journal.append(&serde_json::json!({
                "record":"stream_incomplete", "observation_id":self.observation_id,
                "reason":"capture_limit", "captured_bytes":self.text.len(),
                "offered_bytes":fragment.len(), "omitted_bytes":fragment.len() - end
            }))?;
            return Err("holdout stream capture limit exceeded".into());
        }
        Ok(())
    }

    pub(super) fn finish(self, end: TextEnd) -> Result<CommittedText, String> {
        if self.failed {
            return Err("holdout stream is incomplete".into());
        }
        self.journal.append(&serde_json::json!({
            "record":"stream_end", "observation_id":self.observation_id,
            "text_bytes":self.text.len(), "text_sha256":format!("{:x}", Sha256::digest(self.text.as_bytes())),
            "end":end_value(&end)
        }))?;
        Ok(CommittedText {
            text: self.text,
            end,
        })
    }
}

impl Journal {
    pub(super) fn stream(&mut self, observation_id: usize) -> ObservationStream<'_> {
        ObservationStream {
            journal: self,
            observation_id,
            text: String::new(),
            fragments: 0,
            failed: false,
        }
    }
    #[cfg(unix)]
    pub(super) fn create(directory: &Path, binding: &serde_json::Value) -> Result<Self, String> {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        let dir = crate::private_acl::checked_directory(directory)?;
        let dir_metadata = dir.metadata().map_err(|e| e.to_string())?;
        crate::private_acl::ensure_current_owner(&dir_metadata)?;
        if dir_metadata.permissions().mode() & 0o077 != 0 {
            return Err("attempt journal directory lacks private permissions".into());
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(directory.join("raw-observations.jsonl"))
            .map_err(|error| format!("create exclusive attempt journal: {error}"))?;
        crate::private_acl::ensure_safe_acl(&file)?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        crate::private_acl::ensure_current_owner(&metadata)?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o777 != 0o600 {
            return Err("attempt journal lacks private permissions".into());
        }
        let mut journal = Self {
            file,
            hash: Sha256::new(),
            written: 0,
        };
        journal.append(&serde_json::json!({
            "record":"incomplete_attempt", "candidate_eligible":false,
            "binding":binding, "not_a_success_report":true
        }))?;
        dir.sync_all()
            .map_err(|error| format!("sync attempt journal directory: {error}"))?;
        Ok(journal)
    }

    #[cfg(not(unix))]
    pub(super) fn create(_: &Path, _: &serde_json::Value) -> Result<Self, String> {
        Err("private holdout ACL verification is unsupported on this platform".into())
    }

    pub(super) fn append(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let mut line = serde_json::to_vec(value).map_err(|error| error.to_string())?;
        line.push(b'\n');
        let total = self
            .written
            .checked_add(line.len())
            .filter(|size| *size <= MAX_JOURNAL_BYTES)
            .ok_or("attempt journal byte limit exceeded")?;
        self.file
            .write_all(&line)
            .and_then(|_| self.file.sync_all())
            .map_err(|error| format!("persist attempt journal: {error}"))?;
        self.hash.update(&line);
        self.written = total;
        Ok(())
    }

    pub(super) fn failure(&mut self, error: &str) -> Result<(), String> {
        let (message, complete) = bounded_text(error);
        self.append(&serde_json::json!({
            "record":"failure", "error":message, "error_complete":complete,
            "candidate_eligible":false
        }))
    }

    pub(super) fn sha256(&self) -> String {
        format!("{:x}", self.hash.clone().finalize())
    }
}
