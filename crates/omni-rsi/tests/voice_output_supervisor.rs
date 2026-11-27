//! Synthetic subprocess checks for the public supervisor, separate from CLI tests.

#[cfg(unix)]
mod unix {
    use omni_rsi::child::run_with_timeout;
    use std::{
        fs::{self, File},
        io::{Read, Write},
        os::{
            fd::{AsRawFd, FromRawFd, IntoRawFd},
            unix::net::UnixStream,
        },
        path::Path,
        process::{Command, Stdio},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    const OUTPUT_BYTES: usize = 8 * 1024 * 1024;

    #[test]
    fn synthetic_writer_worker() {
        let Some(marker) = std::env::var_os("VOICE_RECHECK_SYNTHETIC_WRITER") else {
            return;
        };
        fs::write(marker, b"started").unwrap();
        std::io::stdout()
            .write_all(&vec![b'x'; OUTPUT_BYTES])
            .unwrap();
    }

    fn synthetic_command(marker: &Path) -> (Command, UnixStream) {
        let (reader, writer) = UnixStream::pair().unwrap();
        reader
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let writer = unsafe { File::from_raw_fd(writer.into_raw_fd()) };
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("voice_output_supervisor::unix::synthetic_writer_worker")
            .arg("--nocapture")
            .env("VOICE_RECHECK_SYNTHETIC_WRITER", marker)
            .stdin(Stdio::null())
            .stdout(Stdio::from(writer))
            .stderr(Stdio::null());
        (command, reader)
    }

    #[test]
    fn synthetic_supervisor_interrupts_a_blocked_writer() {
        let temporary = tempfile::tempdir().unwrap();
        let marker = temporary.path().join("writer-started");
        let (command, mut reader) = synthetic_command(&marker);
        let observed = reader.try_clone().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&stop);
        let notify = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                let mut available: libc::c_int = 0;
                let ready =
                    unsafe { libc::ioctl(observed.as_raw_fd(), libc::FIONREAD, &mut available) }
                        == 0;
                if marker.exists() && ready && available > 0 {
                    thread::sleep(Duration::from_millis(50));
                    signal.store(true, Ordering::Release);
                    return true;
                }
                thread::sleep(Duration::from_millis(10));
            }
            signal.store(true, Ordering::Release);
            false
        });
        let started = Instant::now();
        let error = run_with_timeout(
            command,
            &stop,
            Duration::from_millis(250),
            Duration::from_secs(6),
        )
        .unwrap_err();
        assert!(
            notify.join().unwrap(),
            "synthetic writer never began output"
        );
        assert!(error.contains("interrupted"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(3));
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).unwrap();
        assert!(!bytes.is_empty() && bytes.len() < OUTPUT_BYTES);
    }

    #[test]
    fn synthetic_supervisor_times_out_if_the_reader_never_drains() {
        let temporary = tempfile::tempdir().unwrap();
        let marker = temporary.path().join("writer-started");
        let (command, mut reader) = synthetic_command(&marker);
        let started = Instant::now();
        let error = run_with_timeout(
            command,
            &AtomicBool::new(false),
            Duration::from_millis(250),
            Duration::from_millis(750),
        )
        .unwrap_err();
        assert!(marker.exists(), "synthetic writer never started");
        assert!(error.contains("timed out"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(3));
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).unwrap();
        assert!(!bytes.is_empty() && bytes.len() < OUTPUT_BYTES);
    }
}
