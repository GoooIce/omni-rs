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
pub fn run(mut command: Command, stop: &AtomicBool, grace: Duration) -> Result<ExitStatus, String> {
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
    loop {
        if status.is_none() {
            status = guard.child.try_wait().map_err(|e| e.to_string())?;
        }
        if let Some(exit) = status
            && !group_exists(id)
        {
            guard.finished = true;
            return if stop.load(Ordering::Acquire) {
                Err("interrupted".into())
            } else {
                Ok(exit)
            };
        }
        if sent.is_none() && (stop.load(Ordering::Acquire) || status.is_some()) {
            group_signal(id, libc::SIGINT)?;
            sent = Some(Instant::now());
        }
        if sent.is_some_and(|at| at.elapsed() >= grace) {
            group_signal(id, libc::SIGKILL)?;
            let _ = guard.child.wait();
            if group_exists(id) {
                eprintln!("omni-rsi: waiting for the training process group to exit");
                while group_exists(id) {
                    thread::sleep(Duration::from_millis(100));
                }
            }
            guard.finished = true;
            return Err(if stop.load(Ordering::Acquire) {
                "interrupted".into()
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
