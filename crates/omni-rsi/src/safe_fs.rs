//! Descriptor-anchored no-follow file access and selected-root alias verification.
//! Single source of truth for `openat(O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)`
//! traversal, bounded single-link reads, no-clobber publication, and alias-safe
//! selected-root resolution across `omni-rsi`.

use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    ffi::{OsStr, OsString},
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

fn check_stop(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

/// An open directory descriptor opened step-by-step without following symlinks.
pub(crate) struct AnchoredDir(File);

impl AnchoredDir {
    pub(crate) fn as_file(&self) -> &File {
        &self.0
    }

    pub(crate) fn into_file(self) -> File {
        self.0
    }

    #[cfg(unix)]
    pub(crate) fn open_with_parent(path: &Path) -> Result<(Self, Self), String> {
        if path.as_os_str().is_empty() {
            return Err("empty directory path".into());
        }
        let mut directory =
            File::open(if path.is_absolute() { "/" } else { "." }).map_err(|e| e.to_string())?;
        let mut parent = directory.try_clone().map_err(|e| e.to_string())?;
        for component in path.components() {
            match component {
                Component::RootDir | Component::CurDir => (),
                Component::Normal(name) => {
                    let child = open_child_os(&directory, name, true)?;
                    parent = directory;
                    directory = child;
                }
                Component::ParentDir | Component::Prefix(_) => {
                    return Err("parent traversal or path prefix is unsupported".into());
                }
            }
        }
        if !directory.metadata().map_err(|e| e.to_string())?.is_dir() {
            return Err("path is not a directory".into());
        }
        Ok((Self(directory), Self(parent)))
    }

    #[cfg(not(unix))]
    pub(crate) fn open_with_parent(_: &Path) -> Result<(Self, Self), String> {
        Err("descriptor-anchored no-follow access requires Unix".into())
    }

    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        Self::open_with_parent(path).map(|(dir, _)| dir)
    }

    /// Resolve aliases in ancestors of a caller-selected root, then open the
    /// resolved path step-by-step without following symlinks.
    pub(crate) fn open_selected(path: &Path) -> Result<Self, String> {
        Self::open(&resolve_selected_path(path, false)?)
    }

    pub(crate) fn open_dir(&self, name: &OsStr) -> Result<Self, String> {
        open_child_os(&self.0, name, true).map(Self)
    }

    #[cfg(test)]
    pub(crate) fn open_file(&self, name: &OsStr) -> Result<File, String> {
        open_child_os(&self.0, name, false)
    }

    pub(crate) fn same_dir(&self, other: &Self) -> Result<bool, String> {
        same_directory(&self.0, &other.0)
    }
}

#[cfg(unix)]
pub(crate) fn open_child_io(
    directory: &File,
    name: &OsStr,
    is_directory: bool,
) -> std::io::Result<File> {
    use std::{
        ffi::CString,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::{ffi::OsStrExt, fs::MetadataExt},
        },
    };
    let name = CString::new(name.as_bytes())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid filename"))?;
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe {
        libc::fstatat(
            directory.as_raw_fd(),
            name.as_ptr(),
            &mut stat,
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error());
    }
    let expected = if is_directory {
        libc::S_IFDIR
    } else {
        libc::S_IFREG
    };
    if stat.st_mode & libc::S_IFMT != expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            if is_directory {
                "symlink, special file, or unexpected entry type"
            } else {
                "symlink, special file, or not a regular file"
            },
        ));
    }
    let flags = libc::O_RDONLY
        | libc::O_NOFOLLOW
        | libc::O_NONBLOCK
        | libc::O_CLOEXEC
        | if is_directory { libc::O_DIRECTORY } else { 0 };
    // The returned descriptor is transferred into File exactly once.
    let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata()?;
    if (is_directory && !metadata.is_dir())
        || (!is_directory && !metadata.is_file())
        || metadata.dev() != stat.st_dev as u64
        || metadata.ino() != stat.st_ino as u64
    {
        return Err(std::io::Error::other("path changed while opening"));
    }
    Ok(file)
}

#[cfg(not(unix))]
pub(crate) fn open_child_io(_: &File, _: &OsStr, _: bool) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "descriptor-anchored no-follow access requires Unix",
    ))
}

pub(crate) fn open_child_os(
    directory: &File,
    name: &OsStr,
    is_directory: bool,
) -> Result<File, String> {
    open_child_io(directory, name, is_directory)
        .map_err(|error| format!("cannot open path without following links: {error}"))
}

/// Walk a slash-separated relative path beneath an anchored directory without
/// following symlinks, returning the final regular file descriptor.
pub(crate) fn open_relative_file(
    root: &File,
    relative: &str,
    stop: &AtomicBool,
) -> Result<File, String> {
    let mut directory = root.try_clone().map_err(|e| e.to_string())?;
    let mut components = relative.split('/').peekable();
    while let Some(component) = components.next() {
        check_stop(stop)?;
        if component.is_empty() || component == "." || component == ".." {
            return Err("invalid relative path component".into());
        }
        let is_dir = components.peek().is_some();
        directory = open_child_os(&directory, OsStr::new(component), is_dir)?;
    }
    if !directory.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("target path must be a regular file".into());
    }
    Ok(directory)
}

#[cfg(unix)]
pub(crate) fn same_directory(first: &File, second: &File) -> Result<bool, String> {
    use std::os::unix::fs::MetadataExt;
    let a = first.metadata().map_err(|e| e.to_string())?;
    let b = second.metadata().map_err(|e| e.to_string())?;
    Ok(a.is_dir() && b.is_dir() && a.dev() == b.dev() && a.ino() == b.ino())
}

#[cfg(not(unix))]
pub(crate) fn same_directory(_: &File, _: &File) -> Result<bool, String> {
    Ok(false)
}

#[cfg(unix)]
pub(crate) fn same_single_link_file(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    before.is_file()
        && after.is_file()
        && before.nlink() == 1
        && after.nlink() == 1
        && before.dev() == after.dev()
        && before.ino() == after.ino()
        && before.len() == after.len()
        && before.mtime() == after.mtime()
        && before.mtime_nsec() == after.mtime_nsec()
        && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec()
}

#[cfg(not(unix))]
pub(crate) fn same_single_link_file(_: &std::fs::Metadata, _: &std::fs::Metadata) -> bool {
    false
}

/// Read a single-link regular file relative to an anchored directory while
/// enforcing byte limits, optional cumulative budget, cancellation, and
/// before/after metadata stability.
pub(crate) fn read_single_link_child(
    directory: &File,
    name: &OsStr,
    limit: u64,
    remaining: Option<&mut u64>,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    check_stop(stop)?;
    let mut file = open_child_os(directory, name, false)?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !same_single_link_file(&before, &before) {
        return Err("artifact must be a single-link regular file".into());
    }
    if before.len() == 0 || before.len() > limit {
        return Err("invalid or oversized artifact".into());
    }
    if let Some(budget) = remaining {
        *budget = budget
            .checked_sub(before.len())
            .ok_or("artifact byte budget exceeded")?;
    }
    let mut bytes =
        Vec::with_capacity(usize::try_from(before.len()).map_err(|_| "artifact size overflow")?);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        check_stop(stop)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        if (bytes.len() as u64)
            .checked_add(count as u64)
            .is_none_or(|n| n > limit || n > before.len())
        {
            return Err("artifact changed or exceeds its limit".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if bytes.len() as u64 != before.len() || !same_single_link_file(&before, &after) {
        return Err("artifact changed while reading".into());
    }
    check_stop(stop)?;
    Ok(bytes)
}

pub(crate) fn read_regular_file(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path).map_err(|e| e.to_string())?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit {
        return Err("invalid or oversized file".into());
    }
    let mut bytes = Vec::new();
    (&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let after = file.metadata().map_err(|e| e.to_string())?;
    if bytes.len() as u64 != metadata.len()
        || metadata.len() != after.len()
        || metadata.modified().ok() != after.modified().ok()
        || !after.is_file()
    {
        return Err("file changed while reading".into());
    }
    Ok(bytes)
}

pub(crate) fn hash_regular_file(
    path: &Path,
    limit: u64,
    stop: &AtomicBool,
) -> Result<String, String> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path).map_err(|e| e.to_string())?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() || before.len() == 0 || before.len() > limit {
        return Err("invalid artifact".into());
    }
    let mut hash = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        check_stop(stop)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or("artifact size overflow")?;
        if total > limit {
            return Err("artifact exceeds limit".into());
        }
        hash.update(&buffer[..count]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if total != before.len()
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Err("artifact changed while hashing".into());
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn write_noclobber(
    path: &Path,
    prefix: &str,
    bytes: &[u8],
    stop: &AtomicBool,
) -> Result<(), String> {
    check_stop(stop)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut file = tempfile::Builder::new()
        .prefix(prefix)
        .tempfile_in(parent)
        .map_err(|e| e.to_string())?;
    file.write_all(bytes)
        .and_then(|_| file.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    check_stop(stop)?;
    file.persist_noclobber(path)
        .map_err(|e| e.error.to_string())?;
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| {
            format!(
                "file published at {} but directory durability is unconfirmed: {e}",
                path.display()
            )
        })?;
    Ok(())
}

/// The caller selects a root; aliases in its parent are trusted, while its last
/// component and everything opened beneath it remain no-follow.
pub(crate) fn resolve_selected_path(path: &Path, missing_final: bool) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() {
        return Err("invalid selected path".into());
    }
    if path
        .components()
        .any(|part| matches!(part, Component::ParentDir))
    {
        return Err("parent traversal is not allowed".into());
    }
    let Some(name) = path.file_name() else {
        return fs::canonicalize(path).map_err(|e| e.to_string());
    };
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let resolved = fs::canonicalize(parent)
        .map_err(|e| e.to_string())?
        .join(name);
    match fs::symlink_metadata(&resolved) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err("selected path final component is a symlink".into());
        }
        Ok(_) => (),
        Err(error) if missing_final && error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.to_string()),
    }
    Ok(resolved)
}

/// Validate that no component of a physical path is a symlink or non-directory interior.
pub(crate) fn no_symlinks_on_path(path: &Path, missing_final: bool) -> Result<(), String> {
    let components: Vec<_> = path.components().collect();
    if components.is_empty() {
        return Err("empty selected path".into());
    }
    let mut current = PathBuf::new();
    for (index, component) in components.iter().enumerate() {
        if matches!(component, Component::ParentDir) {
            return Err("parent traversal is not allowed".into());
        }
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink()
                    || (index + 1 < components.len() && !metadata.is_dir())
                {
                    return Err("linked or non-directory path component".into());
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::NotFound
                    && missing_final
                    && index + 1 == components.len() => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}

/// A caller-selected path paired with its normalized physical identity.
pub(crate) struct SelectedPath {
    pub(crate) original: PathBuf,
    pub(crate) resolved: PathBuf,
    pub(crate) missing_final: bool,
}

fn normalize_selected_path(path: &Path, missing_final: bool) -> Result<PathBuf, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err("selected path final component is a symlink".into())
        }
        Ok(_) => fs::canonicalize(path).map_err(|error| error.to_string()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && missing_final => {
            let parent = path.parent().ok_or("selected path has no parent")?;
            let name = path
                .file_name()
                .ok_or("selected path has no final component")?;
            Ok(fs::canonicalize(parent)
                .map_err(|error| error.to_string())?
                .join(name))
        }
        Err(error) => Err(error.to_string()),
    }
}

fn resolve_below_selected_root(
    path: &Path,
    roots: &[&SelectedPath],
    missing_final: bool,
) -> Result<Option<PathBuf>, String> {
    fn prepend(path: &Path, pending: &mut VecDeque<OsString>) -> Result<(), String> {
        for component in path.components().rev() {
            match component {
                Component::Normal(name) => pending.push_front(name.to_os_string()),
                Component::ParentDir => pending.push_front(OsString::from("..")),
                Component::CurDir | Component::RootDir => {}
                Component::Prefix(_) => return Err("unsupported selected path prefix".into()),
            }
        }
        Ok(())
    }

    let mut current = if path.is_absolute() {
        PathBuf::from(std::path::MAIN_SEPARATOR.to_string())
    } else {
        fs::canonicalize(".").map_err(|error| error.to_string())?
    };
    let mut pending = VecDeque::new();
    prepend(path, &mut pending)?;
    let mut links = 0;
    loop {
        if roots.iter().any(|root| current.starts_with(&root.resolved)) {
            for part in pending {
                current.push(part);
            }
            no_symlinks_on_path(&current, missing_final)?;
            return Ok(Some(current));
        }
        let Some(part) = pending.pop_front() else {
            return Ok(None);
        };
        if part == ".." {
            current.pop();
            continue;
        }
        let candidate = current.join(&part);
        match fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                if pending.is_empty() {
                    return Err("selected path final component is a symlink".into());
                }
                links += 1;
                if links > 40 {
                    return Err("too many linked path components".into());
                }
                let target = fs::read_link(&candidate).map_err(|error| error.to_string())?;
                if target.is_absolute() {
                    current = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
                }
                prepend(&target, &mut pending)?;
            }
            Ok(metadata) => {
                if !pending.is_empty() && !metadata.is_dir() {
                    return Err("non-directory selected path component".into());
                }
                // Normalize case-insensitive spelling before the next boundary check.
                current = fs::canonicalize(&candidate).map_err(|error| error.to_string())?;
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && missing_final
                    && pending.is_empty() =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}

impl SelectedPath {
    pub(crate) fn new(path: &Path, missing_final: bool, roots: &[&Self]) -> Result<Self, String> {
        if path.as_os_str().is_empty() {
            return Err("invalid selected path".into());
        }
        if path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
        {
            return Err("parent traversal is not allowed".into());
        }
        let resolved = if !roots.is_empty()
            && let Some(resolved) = resolve_below_selected_root(path, roots, missing_final)?
        {
            resolved
        } else {
            resolve_selected_path(path, missing_final)?
        };
        Ok(Self {
            original: path.into(),
            resolved: normalize_selected_path(&resolved, missing_final)?,
            missing_final,
        })
    }
}

/// Ensure no later-selected root contains an earlier-selected path through a symlink.
pub(crate) fn verify_reverse_containment(roots: &[&SelectedPath]) -> Result<(), String> {
    for root in roots {
        for candidate in roots {
            let checked = SelectedPath::new(&candidate.original, candidate.missing_final, &[root])?;
            if checked.resolved != candidate.resolved {
                return Err("linked nested selected path".into());
            }
        }
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn anchored_dir_rejects_symlinks_hardlinks_and_fifos() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let file_path = root.join("ok.txt");
        fs::write(&file_path, b"hello").unwrap();
        symlink(&file_path, root.join("link.txt")).unwrap();
        fs::hard_link(&file_path, root.join("hard.txt")).unwrap();
        let fifo = root.join("pipe");
        let c_fifo = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_fifo.as_ptr(), 0o600) }, 0);

        let anchored = AnchoredDir::open(&root).unwrap();
        let stop = AtomicBool::new(false);
        assert!(
            read_single_link_child(anchored.as_file(), OsStr::new("link.txt"), 64, None, &stop)
                .is_err()
        );
        assert!(
            read_single_link_child(anchored.as_file(), OsStr::new("hard.txt"), 64, None, &stop)
                .is_err()
        );
        // FIFO must fail closed immediately without blocking on open.
        assert!(anchored.open_file(OsStr::new("pipe")).is_err());
        fs::remove_file(root.join("hard.txt")).unwrap();
        assert_eq!(
            read_single_link_child(anchored.as_file(), OsStr::new("ok.txt"), 64, None, &stop)
                .unwrap(),
            b"hello"
        );
    }

    #[test]
    fn selected_path_rejects_symlinks_inside_selected_root_even_through_ancestor_alias() {
        let dir = tempfile::tempdir().unwrap();
        let parent = fs::canonicalize(dir.path()).unwrap().join("parent");
        let real = parent.join("real");
        let inner = real.join("inner");
        fs::create_dir_all(&inner).unwrap();
        fs::create_dir_all(real.join("target")).unwrap();
        symlink(real.join("target"), inner.join("linked")).unwrap();
        let alias = fs::canonicalize(dir.path()).unwrap().join("alias");
        symlink(&parent, &alias).unwrap();

        // Final component symlink is rejected; ancestor alias is allowed.
        assert!(SelectedPath::new(&alias, false, &[]).is_err());
        let aliased_root = alias.join("real");
        let root = SelectedPath::new(&aliased_root, false, &[]).unwrap();
        assert_eq!(root.resolved, real);
        // Entering the selected root through its ancestor alias must still reject symlinks inside it.
        assert!(
            SelectedPath::new(&aliased_root.join("inner/linked/file.json"), true, &[&root])
                .is_err()
        );
        let ok = SelectedPath::new(&aliased_root.join("inner/file.json"), true, &[&root]).unwrap();
        assert_eq!(ok.resolved, real.join("inner/file.json"));
    }
}
