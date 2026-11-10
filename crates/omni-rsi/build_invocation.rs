//! Verify the direct Cargo process and its actual invocation directory.
use std::path::{Path, PathBuf};

/// The build script runs in the manifest directory, not Cargo's launch directory.
/// Trust only an inspectable direct Cargo parent; PWD can be absent or stale.
pub fn cargo_invocation_cwd(_repo: &Path) -> Option<PathBuf> {
    let expected = PathBuf::from(std::env::var_os("CARGO")?);
    if !expected.is_absolute() {
        return None;
    }
    let (executable, cwd) = parent_process()?;
    verified_parent_cwd(&expected, &executable, &cwd)
}

pub fn verified_parent_cwd(expected: &Path, executable: &Path, cwd: &Path) -> Option<PathBuf> {
    if !expected.is_absolute() {
        return None;
    }
    let executable = std::fs::canonicalize(executable).ok()?;
    if !executable.is_file() || std::fs::canonicalize(expected).ok()? != executable {
        return None;
    }
    let cwd = std::fs::canonicalize(cwd).ok()?;
    cwd.is_dir().then_some(cwd)
}

pub fn proc_self_matches_pid(link: &Path, pid: u32) -> bool {
    link == Path::new(&pid.to_string())
}

#[cfg(target_os = "macos")]
fn parent_process() -> Option<(PathBuf, PathBuf)> {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};

    let parent = unsafe { libc::getppid() };
    if parent <= 1 {
        return None;
    }
    let mut info: libc::proc_vnodepathinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of_val(&info);
    let result = unsafe {
        libc::proc_pidinfo(
            parent,
            libc::PROC_PIDVNODEPATHINFO,
            0,
            (&mut info as *mut libc::proc_vnodepathinfo).cast(),
            i32::try_from(size).ok()?,
        )
    };
    if usize::try_from(result).ok()? != size {
        return None;
    }
    let mut executable = vec![0_u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    let result = unsafe {
        libc::proc_pidpath(
            parent,
            executable.as_mut_ptr().cast(),
            u32::try_from(executable.len()).ok()?,
        )
    };
    if result <= 0 {
        return None;
    }
    let cwd: Vec<u8> = info
        .pvi_cdir
        .vip_path
        .iter()
        .flatten()
        .map(|&byte| byte as u8)
        .collect();
    let path = |bytes: &[u8]| {
        let end = bytes.iter().position(|byte| *byte == 0)?;
        (end > 0).then(|| PathBuf::from(OsString::from_vec(bytes[..end].to_vec())))
    };
    Some((path(&executable)?, path(&cwd)?))
}

#[cfg(target_os = "linux")]
fn parent_process() -> Option<(PathBuf, PathBuf)> {
    // A container may expose a procfs mounted for a different PID namespace.
    // Its parent PID would then name a different process; do not trust it.
    if !proc_self_matches_pid(&std::fs::read_link("/proc/self").ok()?, std::process::id()) {
        return None;
    }
    let parent = unsafe { libc::getppid() };
    if parent <= 1 {
        return None;
    }
    let proc = PathBuf::from("/proc").join(parent.to_string());
    Some((
        std::fs::read_link(proc.join("exe")).ok()?,
        std::fs::read_link(proc.join("cwd")).ok()?,
    ))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn parent_process() -> Option<(PathBuf, PathBuf)> {
    None
}
