//! Deterministic, build-time identity for a local dependency's complete source tree.
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
};

const MAX_FILES: usize = 100_000;
const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TREE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub fn uncomment(line: &str) -> Result<&str, String> {
    let mut quote = None;
    let mut escaped = false;
    for (i, byte) in line.bytes().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some(b'"') && byte == b'\\' {
            escaped = true;
            continue;
        }
        if let Some(open) = quote {
            if byte == open {
                quote = None;
            }
        } else if byte == b'"' || byte == b'\'' {
            if line
                .as_bytes()
                .get(i..i + 3)
                .is_some_and(|next| next == [byte; 3])
            {
                return Err("multiline TOML strings are unsupported here".into());
            }
            quote = Some(byte);
        } else if byte == b'#' {
            return Ok(&line[..i]);
        }
    }
    if quote.is_some() {
        return Err("unterminated TOML string".into());
    }
    Ok(line)
}

pub fn keys(mut input: &str) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    loop {
        input = input.trim_start();
        let (key, rest) = if input.starts_with(['"', '\'']) {
            let quote = input.as_bytes()[0] as char;
            let mut chars = input[1..].char_indices();
            let mut key = String::new();
            let mut end = None;
            while let Some((offset, ch)) = chars.next() {
                if ch == quote {
                    end = Some(offset + 1);
                    break;
                }
                if quote == '"' && ch == '\\' {
                    let escape = chars.next().ok_or("invalid TOML key escape")?.1;
                    let decoded = match escape {
                        'b' => '\u{0008}',
                        't' => '\t',
                        'n' => '\n',
                        'f' => '\u{000c}',
                        'r' => '\r',
                        '"' => '"',
                        '\\' => '\\',
                        'u' | 'U' => {
                            let mut value = 0_u32;
                            for _ in 0..if escape == 'u' { 4 } else { 8 } {
                                let digit =
                                    chars.next().ok_or("invalid TOML key unicode escape")?.1;
                                if !digit.is_ascii_hexdigit() {
                                    return Err("invalid TOML key unicode escape".into());
                                }
                                value = value
                                    .checked_mul(16)
                                    .and_then(|v| v.checked_add(digit.to_digit(16)?))
                                    .ok_or("invalid TOML key unicode escape")?;
                            }
                            char::from_u32(value).ok_or("invalid TOML key unicode scalar")?
                        }
                        _ => return Err("unsupported TOML key escape".into()),
                    };
                    key.push(decoded);
                } else {
                    key.push(ch);
                }
            }
            let end = end.ok_or("invalid quoted TOML key")?;
            (key, &input[end + 1..])
        } else {
            let len = input
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
                .count();
            if len == 0 {
                return Err("invalid TOML key".into());
            }
            (input[..len].to_owned(), &input[len..])
        };
        if key.is_empty() {
            return Err("empty TOML key".into());
        }
        result.push(key);
        let rest = rest.trim_start();
        if rest.is_empty() {
            return Ok(result);
        }
        input = rest.strip_prefix('.').ok_or("invalid TOML dotted key")?;
    }
}

pub fn assignment(line: &str) -> Result<(&str, &str), String> {
    let mut quote = None;
    let mut escaped = false;
    for (i, byte) in line.bytes().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some(b'"') && byte == b'\\' {
            escaped = true;
            continue;
        }
        if let Some(open) = quote {
            if byte == open {
                quote = None;
            }
        } else if byte == b'"' || byte == b'\'' {
            quote = Some(byte);
        } else if byte == b'=' {
            return Ok((line[..i].trim(), line[i + 1..].trim()));
        }
    }
    Err("missing TOML assignment".into())
}

pub fn dependency_assignment(line: &str, paths: &mut Vec<String>) -> Result<(), String> {
    let (lhs, rhs) = assignment(line)?;
    let key = keys(lhs)?;
    if key.iter().any(|part| part == "path") {
        if key.last().map(String::as_str) != Some("path") {
            return Err("unsupported path key".into());
        }
        let quote = rhs
            .chars()
            .next()
            .filter(|c| *c == '"' || *c == '\'')
            .ok_or("unsupported local dependency path")?;
        let value = rhs
            .strip_prefix(quote)
            .and_then(|s| s.strip_suffix(quote))
            .ok_or("unsupported local dependency path")?;
        if value.is_empty() || value.contains(['\\', '\n', '\r', '"', '\'']) {
            return Err("unsupported local dependency path".into());
        }
        paths.push(value.to_owned());
    }
    if key.last().map(String::as_str) == Some("workspace") && rhs == "true" {
        return Err("inherited workspace dependencies are unsupported".into());
    }
    if rhs.starts_with('{') {
        let body = rhs
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .ok_or("multiline inline dependency is unsupported")?;
        let mut quote = None;
        let mut escaped = false;
        let mut depth = 0usize;
        let mut start = 0usize;
        for (i, byte) in body.bytes().enumerate() {
            if escaped {
                escaped = false;
                continue;
            }
            if quote == Some(b'"') && byte == b'\\' {
                escaped = true;
                continue;
            }
            if let Some(open) = quote {
                if byte == open {
                    quote = None;
                }
            } else {
                match byte {
                    b'"' | b'\'' => quote = Some(byte),
                    b'[' | b'{' => depth += 1,
                    b']' | b'}' => {
                        depth = depth.checked_sub(1).ok_or("invalid inline dependency")?
                    }
                    b',' if depth == 0 => {
                        dependency_assignment(body[start..i].trim(), paths)?;
                        start = i + 1;
                    }
                    _ => (),
                }
            }
        }
        if quote.is_some() || depth != 0 {
            return Err("invalid inline dependency".into());
        }
        let last = body[start..].trim();
        if !last.is_empty() {
            dependency_assignment(last, paths)?;
        }
    }
    Ok(())
}

pub fn validate_config(config: &str) -> Result<(), String> {
    let mut section = Vec::new();
    for original in config.lines() {
        let line = uncomment(original)?.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            let header = if line.starts_with("[[") {
                line.strip_prefix("[[").and_then(|s| s.strip_suffix("]]"))
            } else {
                line.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
            }
            .ok_or("invalid Cargo configuration table")?;
            section = keys(header)?;
            if section
                .first()
                .is_some_and(|k| k == "patch" || k == "paths")
            {
                return Err("Cargo configuration path overrides are unsupported".into());
            }
        } else if section.is_empty()
            && let Ok((lhs, _)) = assignment(line)
        {
            let key = keys(lhs)?;
            if key.first().is_some_and(|k| k == "patch" || k == "paths") {
                return Err("Cargo configuration path overrides are unsupported".into());
            }
        }
    }
    Ok(())
}

/// The repository supports single-line path patches. Reject other patch syntax
/// instead of silently omitting a source tree from an audio identity.
pub fn patch_paths(manifest: &str) -> Result<Vec<(String, String)>, String> {
    let mut active = false;
    let mut seen = false;
    let mut result = Vec::new();
    for original in manifest.lines() {
        let line = uncomment(original)?.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            let section = if line.starts_with("[[") {
                line.strip_prefix("[[")
                    .and_then(|s| s.strip_suffix("]]"))
                    .ok_or("invalid Cargo table")?
            } else {
                line.strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or("invalid Cargo table")?
            };
            let section_keys = keys(section)?;
            if section_keys.first().is_some_and(|key| key == "patch") && line != "[patch.crates-io]"
            {
                return Err("unsupported Cargo patch table".into());
            }
            active = line == "[patch.crates-io]";
            seen |= active;
            continue;
        }
        if !active {
            if let Ok((lhs, _)) = assignment(line)
                && keys(lhs)?.first().is_some_and(|key| key == "patch")
            {
                return Err("unsupported dotted Cargo patch".into());
            }
            continue;
        }
        let (name, value) = line.split_once('=').ok_or("invalid Cargo patch")?;
        let name = name.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err("unsupported Cargo patch name".into());
        }
        let value = value
            .trim()
            .strip_prefix('{')
            .and_then(|v| v.strip_suffix('}'))
            .ok_or("unsupported Cargo patch")?;
        let (key, path) = value
            .trim()
            .split_once('=')
            .ok_or("unsupported Cargo patch")?;
        if key.trim() != "path" {
            return Err("Cargo patch must use a local path".into());
        }
        let path = path
            .trim()
            .strip_prefix('"')
            .and_then(|p| p.strip_suffix('"'))
            .ok_or("unsupported Cargo patch path")?;
        if path.is_empty() || path.contains(['"', '\\', '\n', '\r']) {
            return Err("unsupported Cargo patch path".into());
        }
        if result.iter().any(|(existing, _)| existing == name) {
            return Err("duplicate Cargo patch".into());
        }
        result.push((name.to_owned(), path.to_owned()));
    }
    if !seen || result.is_empty() {
        return Err("missing local Cargo patches".into());
    }
    result.sort();
    Ok(result)
}

pub fn source_tree_digest(root: &Path) -> Result<(String, Vec<PathBuf>), String> {
    fn field(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    fn walk(
        root: &Path,
        relative: &Path,
        hash: &mut Sha256,
        watches: &mut Vec<PathBuf>,
        total: &mut u64,
        entries: &mut usize,
    ) -> Result<(), String> {
        *entries += 1;
        if *entries > MAX_FILES {
            return Err("backend source tree has too many entries".into());
        }
        let path = root.join(relative);
        let metadata =
            fs::symlink_metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let name = relative
            .to_str()
            .ok_or("backend source path is not UTF-8")?;
        if name.contains(['\n', '\r']) {
            return Err("backend source path contains a newline".into());
        }
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "backend source contains symlink: {}",
                path.display()
            ));
        }
        watches.push(path.clone());
        if metadata.is_dir() {
            field(hash, b"dir");
            field(hash, name.as_bytes());
            let mut children = fs::read_dir(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .map(|entry| {
                    entry
                        .map(|entry| entry.file_name())
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            children.sort();
            for child in children {
                if child == ".git"
                    || (relative.as_os_str().is_empty()
                        && (child == "target" || child == "artifacts"))
                {
                    continue;
                }
                walk(root, &relative.join(child), hash, watches, total, entries)?;
            }
        } else if metadata.is_file() {
            let mut options = OpenOptions::new();
            options.read(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
            }
            let mut file = options
                .open(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let actual = file.metadata().map_err(|e| e.to_string())?;
            if !actual.is_file() || actual.len() > MAX_FILE_BYTES {
                return Err(format!("invalid backend source file: {}", path.display()));
            }
            *total = total
                .checked_add(actual.len())
                .filter(|n| *n <= MAX_TREE_BYTES)
                .ok_or("backend source tree exceeds size limit")?;
            field(hash, b"file");
            field(hash, name.as_bytes());
            hash.update(actual.len().to_le_bytes());
            let mut seen = 0u64;
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                seen = seen.checked_add(n as u64).ok_or("source size overflow")?;
                if seen > actual.len() {
                    return Err("backend source changed while hashing".into());
                }
                hash.update(&buffer[..n]);
            }
            if seen != actual.len() {
                return Err("backend source changed while hashing".into());
            }
        } else {
            return Err(format!(
                "backend source contains a special file: {}",
                path.display()
            ));
        }
        Ok(())
    }

    let mut hash = Sha256::new();
    hash.update(b"omni-local-backend-source-tree-v1\0");
    let mut watches = Vec::new();
    walk(root, Path::new(""), &mut hash, &mut watches, &mut 0, &mut 0)?;
    Ok((format!("{:x}", hash.finalize()), watches))
}
