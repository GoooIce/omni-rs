//! Supervise a cancellable training process and its process group.
use std::{
    process::{Command, ExitStatus},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
struct ChildGuard {
    child: std::process::Child,
    id: u32,
    finished: bool,
}
#[cfg(unix)]
struct InheritedChildGuard {
    child: std::process::Child,
    finished: bool,
}
#[cfg(unix)]
impl Drop for InheritedChildGuard {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
#[cfg(unix)]
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.finished {
            let _ = group_signal(self.id, libc::SIGKILL);
            let _ = self.child.wait();
            while group_exists(self.id) {
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

#[cfg(unix)]
fn group_signal(id: u32, signal: i32) -> Result<(), String> {
    let pid = i32::try_from(id).map_err(|e| e.to_string())?;
    if unsafe { libc::kill(-pid, signal) } == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error.to_string())
    }
}

#[cfg(unix)]
fn group_exists(id: u32) -> bool {
    let Ok(pid) = i32::try_from(id) else {
        return true;
    };
    if unsafe { libc::kill(-pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// Stop the entire child process group before returning the device lease.
#[cfg(unix)]
pub fn run(command: Command, stop: &AtomicBool, grace: Duration) -> Result<ExitStatus, String> {
    run_inner(command, stop, grace, None)
}

/// Like `run`, but also cancel the complete process group after a deadline.
#[cfg(unix)]
pub fn run_with_timeout(
    command: Command,
    stop: &AtomicBool,
    grace: Duration,
    timeout: Duration,
) -> Result<ExitStatus, String> {
    run_inner(command, stop, grace, Some(timeout))
}

/// Run an R2T2 child inside an already supervised worker process group. The
/// outer scheduler can then terminate the whole tree if the worker stalls.
#[cfg(unix)]
pub fn run_inherited_with_timeout(
    mut command: Command,
    stop: &AtomicBool,
    grace: Duration,
    timeout: Duration,
) -> Result<ExitStatus, String> {
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    let child = command.spawn().map_err(|error| error.to_string())?;
    let mut guard = InheritedChildGuard {
        child,
        finished: false,
    };
    let started = Instant::now();
    let mut sent = None;
    loop {
        let timed_out = started.elapsed() >= timeout;
        if let Some(status) = guard.child.try_wait().map_err(|error| error.to_string())? {
            guard.finished = true;
            return if stop.load(Ordering::Acquire) {
                Err("interrupted".into())
            } else if timed_out {
                Err("training process timed out".into())
            } else {
                Ok(status)
            };
        }
        if sent.is_none() && (stop.load(Ordering::Acquire) || timed_out) {
            let pid = i32::try_from(guard.child.id()).map_err(|error| error.to_string())?;
            if unsafe { libc::kill(pid, libc::SIGINT) } != 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(error.to_string());
                }
            }
            sent = Some(Instant::now());
        }
        if sent.is_some_and(|time| time.elapsed() >= grace) {
            let _ = guard.child.kill();
            let _ = guard.child.wait();
            guard.finished = true;
            return Err(if stop.load(Ordering::Acquire) {
                "interrupted"
            } else {
                "training process timed out"
            }
            .into());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(unix)]
fn run_inner(
    mut command: Command,
    stop: &AtomicBool,
    grace: Duration,
    timeout: Option<Duration>,
) -> Result<ExitStatus, String> {
    use std::os::unix::process::CommandExt;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    command.process_group(0);
    let child = command.spawn().map_err(|e| e.to_string())?;
    let id = child.id();
    let mut guard = ChildGuard {
        child,
        id,
        finished: false,
    };
    let mut sent = None;
    let mut status = None;
    let started = Instant::now();
    loop {
        let timed_out = timeout.is_some_and(|limit| started.elapsed() >= limit);
        if status.is_none() {
            status = guard.child.try_wait().map_err(|e| e.to_string())?;
        }
        if let Some(exit) = status
            && !group_exists(id)
        {
            guard.finished = true;
            return if stop.load(Ordering::Acquire) {
                Err("interrupted".into())
            } else if timed_out {
                Err("training process timed out".into())
            } else {
                Ok(exit)
            };
        }
        if sent.is_none() && (stop.load(Ordering::Acquire) || timed_out || status.is_some()) {
            group_signal(id, libc::SIGINT)?;
            sent = Some(Instant::now());
        }
        if sent.is_some_and(|at| at.elapsed() >= grace) {
            group_signal(id, libc::SIGKILL)?;
            let _ = guard.child.wait();
            if group_exists(id) {
                while group_exists(id) {
                    thread::sleep(Duration::from_millis(100));
                }
            }
            guard.finished = true;
            return Err(if stop.load(Ordering::Acquire) {
                "interrupted".into()
            } else if timed_out {
                "training process timed out".into()
            } else {
                "training process left a child running".into()
            });
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(not(unix))]
pub fn run(_command: Command, _stop: &AtomicBool, _grace: Duration) -> Result<ExitStatus, String> {
    Err("training worker process groups are unsupported on this platform".into())
}

#[cfg(not(unix))]
pub fn run_with_timeout(
    _command: Command,
    _stop: &AtomicBool,
    _grace: Duration,
    _timeout: Duration,
) -> Result<ExitStatus, String> {
    Err("training worker process groups are unsupported on this platform".into())
}

#[cfg(not(unix))]
pub fn run_inherited_with_timeout(
    _command: Command,
    _stop: &AtomicBool,
    _grace: Duration,
    _timeout: Duration,
) -> Result<ExitStatus, String> {
    Err("training worker process groups are unsupported on this platform".into())
}
