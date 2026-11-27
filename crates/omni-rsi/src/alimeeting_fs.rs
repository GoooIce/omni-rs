//! Descriptor-relative, bounded input enumeration and no-clobber publication.

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod platform {
    use std::{
        collections::BTreeSet,
        ffi::{CStr, CString, OsStr},
        fs::File,
        io::{Read, Write},
        os::{
            fd::{AsRawFd, FromRawFd, IntoRawFd},
            unix::{ffi::OsStrExt, fs::MetadataExt},
        },
        path::Path,
    };

    const MAX_DEPTH: usize = 16;
    const MAX_ENTRIES: usize = 20_000;
    const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;
    const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;

    fn cstring(name: &OsStr) -> Result<CString, String> {
        CString::new(name.as_bytes()).map_err(|_| "NUL in path component".into())
    }

    fn entry_stat(dir: &File, name: &OsStr) -> Result<libc::stat, String> {
        let name = cstring(name)?;
        // fstatat does not follow the directory entry, including symlinks.
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        let status = unsafe {
            libc::fstatat(
                dir.as_raw_fd(),
                name.as_ptr(),
                &mut stat,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if status != 0 {
            return Err(format!(
                "cannot inspect input entry: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(stat)
    }

    fn open_child(dir: &File, name: &OsStr, directory: bool) -> Result<File, String> {
        crate::safe_fs::open_child_os(dir, name, directory)
    }

    pub(crate) fn open_directory(path: &Path) -> Result<File, String> {
        crate::safe_fs::AnchoredDir::open(path).map(crate::safe_fs::AnchoredDir::into_file)
    }

    struct DirectoryStream(*mut libc::DIR);

    impl Drop for DirectoryStream {
        fn drop(&mut self) {
            unsafe { libc::closedir(self.0) };
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn errno() -> *mut libc::c_int {
        unsafe { libc::__error() }
    }

    #[cfg(target_os = "linux")]
    unsafe fn errno() -> *mut libc::c_int {
        unsafe { libc::__errno_location() }
    }

    fn names(directory: &File) -> Result<Vec<String>, String> {
        // Open a fresh file description so readdir never changes the caller's offset.
        let duplicate = open_child(directory, OsStr::new("."), true)?;
        let fd = duplicate.into_raw_fd();
        let stream = unsafe { libc::fdopendir(fd) };
        if stream.is_null() {
            unsafe { libc::close(fd) };
            return Err(std::io::Error::last_os_error().to_string());
        }
        let stream = DirectoryStream(stream);
        let mut output = Vec::new();
        loop {
            unsafe { *errno() = 0 };
            let entry = unsafe { libc::readdir(stream.0) };
            if entry.is_null() {
                if unsafe { *errno() } != 0 {
                    return Err(std::io::Error::last_os_error().to_string());
                }
                break;
            }
            let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }
                .to_str()
                .map_err(|_| "non-UTF-8 input filename")?;
            if name == "." || name == ".." {
                continue;
            }
            if name.len() > 255 || name.chars().any(char::is_control) {
                return Err("invalid input filename".into());
            }
            output.push(name.to_owned());
            if output.len() > MAX_ENTRIES {
                return Err("too many input entries".into());
            }
        }
        output.sort();
        Ok(output)
    }

    fn same_observation(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
        a.dev() == b.dev()
            && a.ino() == b.ino()
            && a.len() == b.len()
            && a.mtime() == b.mtime()
            && a.mtime_nsec() == b.mtime_nsec()
            && a.ctime() == b.ctime()
            && a.ctime_nsec() == b.ctime_nsec()
    }

    struct Budget {
        entries: usize,
        files: usize,
        bytes: u64,
        file_identities: BTreeSet<(u64, u64)>,
    }

    fn walk(
        directory: &File,
        prefix: &str,
        depth: usize,
        budget: &mut Budget,
        visitor: &mut impl FnMut(String, Vec<u8>) -> Result<(), String>,
    ) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Err("input directory depth exceeds limit".into());
        }
        let before_dir = directory.metadata().map_err(|error| error.to_string())?;
        for name in names(directory)? {
            budget.entries = budget
                .entries
                .checked_add(1)
                .ok_or("entry count overflow")?;
            if budget.entries > MAX_ENTRIES {
                return Err("too many input entries".into());
            }
            let relative = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            let kind = entry_stat(directory, OsStr::new(&name))?.st_mode & libc::S_IFMT;
            if kind == libc::S_IFDIR {
                let child = open_child(directory, OsStr::new(&name), true)?;
                walk(&child, &relative, depth + 1, budget, visitor)?;
            } else if kind == libc::S_IFREG {
                if !name.to_ascii_lowercase().ends_with(".textgrid") {
                    return Err(format!("unexpected non-TextGrid file: {relative}"));
                }
                let mut file = open_child(directory, OsStr::new(&name), false)?;
                let before = file.metadata().map_err(|error| error.to_string())?;
                if before.nlink() != 1 {
                    return Err(format!("hardlinked input file: {relative}"));
                }
                if !budget.file_identities.insert((before.dev(), before.ino())) {
                    return Err(format!("hardlinked input alias: {relative}"));
                }
                if before.len() > MAX_FILE_BYTES {
                    return Err(format!("TextGrid exceeds file limit: {relative}"));
                }
                budget.bytes = budget
                    .bytes
                    .checked_add(before.len())
                    .ok_or("input size overflow")?;
                if budget.bytes > MAX_TOTAL_BYTES {
                    return Err("total TextGrid bytes exceed limit".into());
                }
                let mut bytes = Vec::new();
                Read::by_ref(&mut file)
                    .take(MAX_FILE_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|error| error.to_string())?;
                let after = file.metadata().map_err(|error| error.to_string())?;
                if bytes.len() as u64 != before.len() || !same_observation(&before, &after) {
                    return Err(format!("TextGrid changed while reading: {relative}"));
                }
                budget.files += 1;
                visitor(relative, bytes)?;
            } else {
                return Err(format!("symlink or special file in input: {relative}"));
            }
        }
        let after_dir = directory.metadata().map_err(|error| error.to_string())?;
        if !same_observation(&before_dir, &after_dir) {
            return Err("input directory changed during enumeration".into());
        }
        Ok(())
    }

    pub(crate) fn visit_inputs(
        root: &File,
        mut visitor: impl FnMut(String, Vec<u8>) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut budget = Budget {
            entries: 0,
            files: 0,
            bytes: 0,
            file_identities: BTreeSet::new(),
        };
        walk(root, "", 0, &mut budget, &mut visitor)?;
        if budget.files == 0 {
            return Err("no TextGrid files in input".into());
        }
        Ok(())
    }

    fn ensure_output_absent(directory: &File, name: &CStr) -> Result<(), String> {
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        let status = unsafe {
            libc::fstatat(
                directory.as_raw_fd(),
                name.as_ptr(),
                &mut stat,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if status == 0 {
            return Err("output already exists".into());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(format!("cannot inspect output destination: {error}"))
        }
    }

    pub(crate) fn output_parent(root: &File, output: &Path) -> Result<(File, CString), String> {
        let name = output.file_name().ok_or("output needs a filename")?;
        if name == OsStr::new(".") || name == OsStr::new("..") || name.as_bytes().is_empty() {
            return Err("invalid output filename".into());
        }
        let name = cstring(name)?;
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let output_dir = open_directory(parent)?;
        let output_metadata = output_dir.metadata().map_err(|error| error.to_string())?;
        if output_metadata.mode() & 0o022 != 0 {
            return Err("output directory must not be writable by group or other users".into());
        }
        let source = root.metadata().map_err(|error| error.to_string())?;
        let mut ancestor = output_dir.try_clone().map_err(|error| error.to_string())?;
        for _ in 0..128 {
            let here = ancestor.metadata().map_err(|error| error.to_string())?;
            if here.dev() == source.dev() && here.ino() == source.ino() {
                return Err("output must be outside the input tree".into());
            }
            let next = open_child(&ancestor, OsStr::new(".."), true)?;
            let above = next.metadata().map_err(|error| error.to_string())?;
            if here.dev() == above.dev() && here.ino() == above.ino() {
                ensure_output_absent(&output_dir, &name)?;
                return Ok((output_dir, name));
            }
            ancestor = next;
        }
        Err("output parent depth exceeds limit".into())
    }

    pub(crate) fn publish(directory: &File, name: &CStr, contents: &[u8]) -> Result<(), String> {
        let mut temporary = None;
        for attempt in 0..128_u32 {
            let candidate = CString::new(format!(
                ".alimeeting-screen-{:032x}-{attempt}.tmp",
                rand::random::<u128>()
            ))
            .map_err(|error| error.to_string())?;
            let flags =
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC;
            let fd =
                unsafe { libc::openat(directory.as_raw_fd(), candidate.as_ptr(), flags, 0o600) };
            if fd >= 0 {
                temporary = Some((candidate, unsafe { File::from_raw_fd(fd) }));
                break;
            }
            if std::io::Error::last_os_error().kind() != std::io::ErrorKind::AlreadyExists {
                return Err(std::io::Error::last_os_error().to_string());
            }
        }
        let (temp_name, mut file) = temporary.ok_or("unable to create temporary output")?;
        let result = (|| {
            file.write_all(contents)
                .map_err(|error| error.to_string())?;
            file.sync_all().map_err(|error| error.to_string())?;
            let written = file.metadata().map_err(|error| error.to_string())?;
            let current = entry_stat(directory, OsStr::from_bytes(temp_name.as_bytes()))?;
            if current.st_mode & libc::S_IFMT != libc::S_IFREG
                || written.dev() != current.st_dev as u64
                || written.ino() != current.st_ino
            {
                return Err("temporary output changed before publication".into());
            }
            // linkat creates the destination only if it does not already exist.
            let status = unsafe {
                libc::linkat(
                    directory.as_raw_fd(),
                    temp_name.as_ptr(),
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    0,
                )
            };
            if status != 0 {
                return Err(format!(
                    "cannot publish new output: {}",
                    std::io::Error::last_os_error()
                ));
            }
            Ok(())
        })();
        unsafe { libc::unlinkat(directory.as_raw_fd(), temp_name.as_ptr(), 0) };
        result
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) use platform::{open_directory, output_parent, publish, visit_inputs};

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod unsupported {
    use std::{
        ffi::{CStr, CString},
        fs::File,
        path::Path,
    };

    pub(crate) fn open_directory(_: &Path) -> Result<File, String> {
        Err("secure TextGrid screening is supported only on macOS and Linux".into())
    }
    pub(crate) fn output_parent(_: &File, _: &Path) -> Result<(File, CString), String> {
        Err("secure TextGrid screening is unsupported".into())
    }
    pub(crate) fn visit_inputs(
        _: &File,
        _: impl FnMut(String, Vec<u8>) -> Result<(), String>,
    ) -> Result<(), String> {
        Err("secure TextGrid screening is unsupported".into())
    }
    pub(crate) fn publish(_: &File, _: &CStr, _: &[u8]) -> Result<(), String> {
        Err("secure TextGrid screening is unsupported".into())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) use unsupported::{open_directory, output_parent, publish, visit_inputs};
