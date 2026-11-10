//! Private, exclusive, incrementally synchronized raw attempt observations.
//! This journal is never a successful holdout report.

use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, path::Path};

const MAX_JOURNAL_BYTES: usize = 128 * 1024 * 1024;
pub(super) const MAX_RECORDED_TEXT_BYTES: usize = 4096;

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

impl Journal {
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
        crate::private_acl::ensure_no_acl(&file)?;
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
