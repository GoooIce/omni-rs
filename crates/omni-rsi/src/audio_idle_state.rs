//! Shared idle reservations and cooperative worker supervision.
use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

const LEGACY: &str = "audio-idle-legacy-started";
const V3: &str = "audio-idle-v3-started";
const V3_NAMESPACE: &str = "audio-idle-v3-namespace-started";

pub fn namespace(root: &Path, v3: bool) -> PathBuf {
    if v3 {
        root.join("number-audio-v3")
    } else {
        root.to_path_buf()
    }
}

fn exists(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.to_string()),
    }
}

fn alias_has_v3_marker(root: &Path) -> Result<bool, String> {
    let canonical = root.canonicalize().map_err(|error| error.to_string())?;
    for path in [root, canonical.as_path()] {
        if path
            .file_name()
            .is_some_and(|name| name == "number-audio-v3")
            && let Some(parent) = path.parent()
            && exists(&parent.join(V3))?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check all known reservations for the physical checkpoint namespace.
pub fn check_fresh(root: &Path, v3: bool) -> Result<(), String> {
    let state = namespace(root, v3);
    let occupied = exists(&state.join("audio-conditioned-v1"))?
        || exists(&state.join(LEGACY))?
        || exists(&state.join(V3_NAMESPACE))?
        || if v3 {
            exists(&root.join(V3))?
        } else {
            alias_has_v3_marker(root)?
        };
    if occupied {
        return Err("audio training state already exists or is reserved; preserve it and explicitly resume a complete checkpoint".into());
    }
    Ok(())
}

#[cfg(unix)]
fn write_marker(path: &Path, content: &[u8]) -> Result<(), String> {
    use std::{io::Write, os::unix::fs::OpenOptionsExt};
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| format!("cannot reserve fresh audio state: {error}"))?;
    file.write_all(content)
        .and_then(|()| file.sync_all())
        .map_err(|error| error.to_string())?;
    fs::File::open(path.parent().ok_or("reservation has no parent")?)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| error.to_string())
}

/// Reserve before spawning a worker, serialized with foreground trainers and aliases.
#[cfg(unix)]
pub fn reserve_fresh(root: &Path, v3: bool) -> Result<(), String> {
    let state = namespace(root, v3);
    fs::create_dir_all(&state).map_err(|error| error.to_string())?;
    let _lease = crate::lease::Lease::try_acquire(&state.join("audio-training.lock"))?
        .ok_or("another audio trainer owns this state")?;
    check_fresh(root, v3)?;
    if v3 {
        // Write inside the actual checkpoint namespace first, so an interrupted
        // reservation is visible even through a differently named symlink.
        write_marker(&state.join(V3_NAMESPACE), b"audio-idle-v3-namespace-v1\n")?;
    }
    write_marker(
        &root.join(if v3 { V3 } else { LEGACY }),
        b"audio-idle-started-v1\n",
    )
}

#[cfg(unix)]
fn check_worker_fresh(root: &Path, v3: bool) -> Result<(), String> {
    let state = namespace(root, v3);
    let valid = if v3 {
        exists(&root.join(V3))?
            && exists(&state.join(V3_NAMESPACE))?
            && !exists(&state.join(LEGACY))?
    } else {
        exists(&root.join(LEGACY))?
            && !exists(&root.join(V3_NAMESPACE))?
            && !alias_has_v3_marker(root)?
    };
    if !valid {
        return Err("audio idle reservation is missing or conflicts with another namespace".into());
    }
    Ok(())
}

#[cfg(unix)]
fn verify_lease(fd: i32, path: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if fd < 3 || unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return Err("missing inherited lease descriptor".into());
    }
    let metadata = fs::metadata(path).map_err(|error| error.to_string())?;
    if !metadata.is_file()
        || metadata.dev() != stat.st_dev as u64
        || metadata.ino() != stat.st_ino as u64
    {
        return Err("inherited lease does not match the scheduler state".into());
    }
    if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err("inherited cooperative lease is not held".into());
    }
    Ok(())
}

#[cfg(unix)]
fn supervisor_alive(stream: &std::os::unix::net::UnixStream) -> bool {
    use std::os::fd::AsRawFd;
    let mut poll = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe { libc::poll(&mut poll, 1, 0) == 0 }
}

/// Evidence of the existing cooperative scheduler connection, not a security
/// boundary against another process running as the same user.
#[cfg(unix)]
pub struct AudioSupervision {
    root: PathBuf,
    runner_fd: i32,
    device_fd: i32,
    stream: std::os::unix::net::UnixStream,
}

#[cfg(unix)]
impl AudioSupervision {
    /// Validate RSI's own leases and supervisor socket.
    pub fn validate(
        root: &Path,
        runner_fd: i32,
        device_fd: i32,
        supervisor_fd: i32,
    ) -> Result<Self, String> {
        use std::{
            collections::HashSet,
            os::{fd::FromRawFd, unix::net::UnixStream},
        };
        let fds = [runner_fd, device_fd, supervisor_fd];
        let mut distinct = HashSet::new();
        if fds.into_iter().any(|fd| fd < 3 || !distinct.insert(fd)) {
            return Err("invalid inherited descriptors".into());
        }
        verify_lease(runner_fd, &root.join("runner.lock"))?;
        verify_lease(device_fd, &root.join("device.lock"))?;
        let mut socket_type: libc::c_int = 0;
        let mut len = std::mem::size_of_val(&socket_type) as libc::socklen_t;
        if unsafe {
            libc::getsockopt(
                supervisor_fd,
                libc::SOL_SOCKET,
                libc::SO_TYPE,
                (&mut socket_type as *mut libc::c_int).cast(),
                &mut len,
            )
        } != 0
            || socket_type != libc::SOCK_STREAM
        {
            return Err("missing supervisor socket".into());
        }
        let mut address: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut address_len = std::mem::size_of_val(&address) as libc::socklen_t;
        if unsafe {
            libc::getpeername(
                supervisor_fd,
                (&mut address as *mut libc::sockaddr_storage).cast(),
                &mut address_len,
            )
        } != 0
            || i32::from(address.ss_family) != libc::AF_UNIX
        {
            return Err("invalid supervisor socket".into());
        }
        let duplicate = unsafe { libc::fcntl(supervisor_fd, libc::F_DUPFD_CLOEXEC, 3) };
        if duplicate < 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let stream = unsafe { UnixStream::from_raw_fd(duplicate) };
        if !supervisor_alive(&stream) {
            return Err("supervisor is no longer present".into());
        }
        let foreground = crate::lease::Lease::try_acquire(&root.join("foreground.lock"))?
            .ok_or("foreground RSI request is active")?;
        drop(foreground);
        Ok(Self {
            root: root.canonicalize().map_err(|error| error.to_string())?,
            runner_fd,
            device_fd,
            stream,
        })
    }

    pub fn monitor(
        &self,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<crate::watchdog::Watchdog, String> {
        let stream = self.stream.try_clone().map_err(|error| error.to_string())?;
        Ok(crate::watchdog::Watchdog::start(
            stop,
            std::time::Duration::from_millis(100),
            move || supervisor_alive(&stream),
        ))
    }

    /// Keep cancellation tied to the supervisor for the entire public training call.
    pub(crate) fn while_connected<R>(
        &self,
        stop: &std::sync::atomic::AtomicBool,
        work: impl FnOnce() -> R,
    ) -> Result<R, String> {
        let stream = self.stream.try_clone().map_err(|error| error.to_string())?;
        Ok(crate::watchdog::Watchdog::scoped(
            stop,
            std::time::Duration::from_millis(100),
            move || supervisor_alive(&stream),
            work,
        ))
    }

    pub(crate) fn check_scope(&self, root: &Path) -> Result<(), String> {
        if root.canonicalize().map_err(|error| error.to_string())? != self.root
            || !supervisor_alive(&self.stream)
        {
            return Err("supervisor or state changed before audio training".into());
        }
        verify_lease(self.runner_fd, &root.join("runner.lock"))?;
        verify_lease(self.device_fd, &root.join("device.lock"))?;
        Ok(())
    }

    pub(crate) fn check_fresh(&self, root: &Path, v3: bool) -> Result<(), String> {
        self.check_scope(root)?;
        check_worker_fresh(root, v3)
    }
}

#[cfg(all(test, unix))]
mod lifecycle_tests {
    use super::AudioSupervision;
    use std::{
        os::unix::net::UnixStream,
        path::PathBuf,
        sync::atomic::{AtomicBool, Ordering},
        time::{Duration, Instant},
    };

    #[test]
    fn supervised_work_cancels_on_disconnect_and_stops_monitor_on_return() {
        let (parent, stream) = UnixStream::pair().unwrap();
        let supervision = AudioSupervision {
            root: PathBuf::new(),
            runner_fd: -1,
            device_fd: -1,
            stream,
        };
        let stop = AtomicBool::new(false);
        supervision
            .while_connected(&stop, || {
                drop(parent);
                let deadline = Instant::now() + Duration::from_secs(2);
                while !stop.load(Ordering::Acquire) {
                    assert!(Instant::now() < deadline, "disconnect did not cancel work");
                    std::thread::sleep(Duration::from_millis(5));
                }
            })
            .unwrap();

        let (_parent, stream) = UnixStream::pair().unwrap();
        let supervision = AudioSupervision {
            stream,
            ..supervision
        };
        let stop = AtomicBool::new(false);
        supervision.while_connected(&stop, || ()).unwrap();
        assert!(!stop.load(Ordering::Acquire));
    }
}
