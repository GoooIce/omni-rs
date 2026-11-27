//! Audio-conditioned idle work; checkpoint state remains separate from self-reading.
use super::{AudioJob, WatchConfig};
use clap::Args as ClapArgs;
use omni_rsi::speech_audio_checkpoint::{AudioCheckpoint, AudioCheckpointSummary};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Clone, Copy, ClapArgs)]
pub(super) struct AudioProof {
    #[arg(long)]
    runner_fd: i32,
    #[arg(long)]
    device_fd: i32,
    #[arg(long)]
    supervisor_fd: i32,
}

fn namespace(root: &Path, job: &AudioJob) -> PathBuf {
    omni_rsi::audio_idle_state::namespace(root, job.number_v3)
}

fn checkpoint(root: &Path, job: &AudioJob) -> Result<Option<AudioCheckpointSummary>, String> {
    let saved = AudioCheckpoint::inspect(&namespace(root, job))
        .map_err(|error| format!("invalid or incompatible audio checkpoint: {error}"))?;
    if saved
        .as_ref()
        .is_some_and(|value| value.protocol_version != if job.number_v3 { 2 } else { 1 })
    {
        return Err("audio checkpoint belongs to another training namespace".into());
    }
    Ok(saved)
}

pub(super) fn preflight(root: &Path, job: &AudioJob) -> Result<(), String> {
    if !(1..=100).contains(&job.epochs) || !(1..=100).contains(&job.updates_per_round) {
        return Err("audio epochs and update budget must be between 1 and 100".into());
    }
    if job.start.fresh {
        omni_rsi::audio_idle_state::check_fresh(root, job.number_v3)?;
    } else if checkpoint(root, job)?.is_none() {
        return Err("no complete audio checkpoint to resume; existing state is untouched".into());
    }
    Ok(())
}

#[cfg(unix)]
fn mark_started(root: &Path, job: &AudioJob) -> Result<(), String> {
    omni_rsi::audio_idle_state::reserve_fresh(root, job.number_v3)
}

#[cfg(unix)]
pub(super) fn watch_audio(
    mut config: WatchConfig<'_>,
    model: &Path,
    job: &AudioJob,
    executable: &Path,
) -> Result<(), String> {
    use std::{cell::Cell, sync::atomic::Ordering};
    let root = config.root;
    let worker_grace = config.worker_grace;
    preflight(root, job)?;
    if !model.is_dir() || !job.data.is_dir() {
        return Err("audio model or authorized data directory is missing".into());
    }
    config.require_work_completion = true;
    let attempted = Cell::new(false);
    let output_exceeded = AtomicBool::new(false);
    let worker = WorkerSpec {
        executable,
        model,
        root,
        job,
        grace: worker_grace,
    };
    super::watch_with_leases(
        config,
        || Ok(None),
        |stop, leases| {
            if stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            let resume = job.start.resume || attempted.get();
            if resume {
                if checkpoint(root, job)?.is_none() {
                    return Err(
                        "no complete audio checkpoint to resume; existing state is untouched"
                            .into(),
                    );
                }
            } else {
                preflight(root, job)?;
                mark_started(root, job)?;
            }
            attempted.set(true);
            let result = run_worker(&worker, resume, leases, stop, &output_exceeded);
            match result {
                Err(error) if !stop.load(Ordering::Acquire) && checkpoint(root, job)?.is_none() => {
                    Err(format!(
                        "{error}; no complete audio checkpoint exists; the started state cannot be resumed"
                    ))
                }
                other => other,
            }
        },
        || {
            if output_exceeded.load(Ordering::Acquire) {
                return Err("audio worker output limit exceeded".into());
            }
            if attempted.get() && checkpoint(root, job)?.is_none() {
                return Err("no complete audio checkpoint after interruption; existing state is untouched and cannot be resumed".into());
            }
            Ok(())
        },
    )
}

#[cfg(unix)]
fn collect_worker_output(
    stream: std::os::unix::net::UnixStream,
    limit: usize,
    stop: &AtomicBool,
    finished: &AtomicBool,
    exceeded: &AtomicBool,
) -> Result<Vec<u8>, String> {
    omni_rsi::idle_supervisor::collect_bounded_stream(stream, limit, stop, finished, exceeded)
}

#[cfg(not(unix))]
pub(super) fn watch_audio(
    _: WatchConfig<'_>,
    _: &Path,
    _: &AudioJob,
    _: &Path,
) -> Result<(), String> {
    Err("supervised audio idle training is unsupported on this platform".into())
}

#[cfg(unix)]
fn configure_inherited(command: &mut std::process::Command, proof: AudioProof) {
    use std::os::unix::process::CommandExt;
    let fds = [
        Some(proof.runner_fd),
        Some(proof.device_fd),
        Some(proof.supervisor_fd),
    ];
    // Only the forked child changes descriptor flags; the scheduler keeps its leases.
    unsafe {
        command.pre_exec(move || {
            for fd in fds.into_iter().flatten() {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct WorkerSpec<'a> {
    executable: &'a Path,
    model: &'a Path,
    root: &'a Path,
    job: &'a AudioJob,
    grace: std::time::Duration,
}

#[cfg(unix)]
fn run_worker(
    worker: &WorkerSpec<'_>,
    resume: bool,
    leases: super::HeldLeases<'_>,
    stop: &AtomicBool,
    exceeded: &AtomicBool,
) -> Result<serde_json::Value, String> {
    use std::{
        os::{
            fd::{AsRawFd, OwnedFd},
            unix::net::UnixStream,
        },
        process::{Command, Stdio},
        sync::atomic::Ordering,
        time::Duration,
    };
    let WorkerSpec {
        executable,
        model,
        root,
        job,
        grace,
    } = *worker;
    let (keepalive, worker_socket) = UnixStream::pair().map_err(|error| error.to_string())?;
    let proof = AudioProof {
        runner_fd: leases.runner.as_raw_fd(),
        device_fd: leases.device.as_raw_fd(),
        supervisor_fd: worker_socket.as_raw_fd(),
    };
    let (output_reader, output_writer) = UnixStream::pair().map_err(|error| error.to_string())?;
    let (log_reader, log_writer) = UnixStream::pair().map_err(|error| error.to_string())?;
    output_reader
        .set_nonblocking(true)
        .map_err(|error| error.to_string())?;
    log_reader
        .set_nonblocking(true)
        .map_err(|error| error.to_string())?;
    let mut command = Command::new(executable);
    command
        .arg("--model")
        .arg(model)
        .arg("--state")
        .arg(root)
        .arg("audio-work")
        .arg("--data")
        .arg(&job.data)
        .arg("--epochs")
        .arg(job.epochs.to_string())
        .arg("--updates-per-round")
        .arg(job.updates_per_round.to_string())
        .arg(if resume { "--resume" } else { "--fresh" })
        .arg("--runner-fd")
        .arg(proof.runner_fd.to_string())
        .arg("--device-fd")
        .arg(proof.device_fd.to_string())
        .arg("--supervisor-fd")
        .arg(proof.supervisor_fd.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(OwnedFd::from(output_writer)))
        .stderr(Stdio::from(OwnedFd::from(log_writer)));
    if job.number_v3 {
        command.arg("--number-v3");
    }
    configure_inherited(&mut command, proof);
    let local_stop = AtomicBool::new(false);
    let finished = AtomicBool::new(false);
    let result: Result<_, String> = std::thread::scope(|scope| {
        let stdout = scope.spawn(|| {
            collect_worker_output(output_reader, 1_048_576, &local_stop, &finished, exceeded)
        });
        let stderr = scope
            .spawn(|| collect_worker_output(log_reader, 65_536, &local_stop, &finished, exceeded));
        let relay = scope.spawn(|| {
            while !finished.load(Ordering::Acquire) {
                if stop.load(Ordering::Acquire) {
                    local_stop.store(true, Ordering::Release);
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        let status = omni_rsi::child::run(command, &local_stop, grace);
        finished.store(true, Ordering::Release);
        relay
            .join()
            .map_err(|_| "audio cancellation relay panicked")?;
        let stdout = stdout
            .join()
            .map_err(|_| "audio stdout reader panicked")??;
        let stderr = stderr
            .join()
            .map_err(|_| "audio stderr reader panicked")??;
        Ok((status, stdout, stderr))
    });
    drop(worker_socket);
    drop(keepalive);
    let (status, bytes, log) = result?;
    if exceeded.load(Ordering::Acquire) {
        return Err("audio worker output limit exceeded".into());
    }
    let status = status?;
    if !status.success() {
        return Err(format!(
            "audio worker exited {status}: {}",
            String::from_utf8_lossy(&log).trim()
        ));
    }
    let report: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid audio worker report: {error}"))?;
    let status = report
        .get("status")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing audio training status")?;
    let progress: omni_rsi::speech_audio_checkpoint::AudioProgress = serde_json::from_value(
        report
            .get("progress")
            .cloned()
            .ok_or("missing audio training progress")?,
    )
    .map_err(|error| error.to_string())?;
    let saved =
        checkpoint(root, job)?.ok_or("audio worker returned without a complete checkpoint")?;
    if report.get("checkpoint").and_then(serde_json::Value::as_str)
        != Some(saved.checkpoint_sha256.as_str())
        || progress != saved.progress
        || status
            != if progress.epoch >= job.epochs {
                "complete"
            } else {
                "partial"
            }
    {
        return Err("audio worker result does not match its complete checkpoint".into());
    }
    Ok(
        serde_json::json!({"complete":status == "complete", "training_status":status,
        "progress":progress, "checkpoint":saved.checkpoint_sha256, "quality":"not_assessed"}),
    )
}

#[cfg(unix)]
fn validate_supervision(
    root: &Path,
    proof: AudioProof,
) -> Result<omni_rsi::audio_idle_state::AudioSupervision, String> {
    omni_rsi::audio_idle_state::AudioSupervision::validate(
        root,
        proof.runner_fd,
        proof.device_fd,
        proof.supervisor_fd,
    )
}

#[cfg(unix)]
pub(super) fn work(
    model: &Path,
    root: &Path,
    job: &AudioJob,
    proof: AudioProof,
    stop: Arc<AtomicBool>,
) -> Result<(), String> {
    use omni_rsi::speech_audio_round::{AudioRoundOptions, train_supervised_audio_round};
    let supervision = validate_supervision(root, proof)
        .map_err(|error| format!("supervised audio worker rejected: {error}"))?;
    let options = AudioRoundOptions {
        model,
        data: &job.data,
        state: root,
        epochs: job.epochs,
        max_updates: job.updates_per_round,
        resume: job.start.resume,
    };
    let result = train_supervised_audio_round::<super::TrainingBackend>(
        &options,
        &Default::default(),
        &stop,
        job.number_v3,
        &supervision,
    )?;
    println!(
        "{}",
        serde_json::to_string(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}

#[cfg(not(unix))]
pub(super) fn work(
    _: &Path,
    _: &Path,
    _: &AudioJob,
    _: AudioProof,
    _: Arc<AtomicBool>,
) -> Result<(), String> {
    Err("supervised audio worker is unsupported on this platform".into())
}

#[cfg(all(test, unix, feature = "cpu"))]
#[path = "audio_idle_tests.rs"]
mod tests;

#[cfg(all(test, not(unix)))]
mod unsupported_tests {
    use super::*;
    use crate::AudioStart;

    #[test]
    fn unsupported_platform_cannot_start_an_audio_worker() {
        let root = std::path::Path::new("unused");
        let job = AudioJob {
            data: root.into(),
            number_v3: false,
            epochs: 1,
            updates_per_round: 1,
            start: AudioStart {
                fresh: true,
                resume: false,
            },
        };
        let config = WatchConfig::new(
            root,
            1,
            omni_rsi::idle::IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "audio",
        );
        assert!(
            watch_audio(config, root, &job, root)
                .unwrap_err()
                .contains("unsupported")
        );
        let proof = AudioProof {
            runner_fd: 3,
            device_fd: 4,
            supervisor_fd: 5,
        };
        assert!(
            work(root, root, &job, proof, Arc::new(AtomicBool::new(false)))
                .unwrap_err()
                .contains("unsupported")
        );
    }
}
