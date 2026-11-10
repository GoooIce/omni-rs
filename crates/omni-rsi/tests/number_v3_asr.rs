#![cfg(all(unix, feature = "cpu"))]

#[test]
fn cpu_r2t2_uses_explicit_frozen_command_options() {
    use omni_rsi::number_v3_evaluation::{V3AsrMode, transcribe_v3_r2t2};
    use std::{fs, os::unix::fs::PermissionsExt, sync::atomic::AtomicBool};

    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    fs::create_dir(&model).unwrap();
    let audio = root.path().join("audio.wav");
    fs::write(&audio, b"fixture").unwrap();
    let binary = root.path().join("fake-r2t2");
    fs::write(&binary, r##"#!/bin/sh
backend=0
precision=0
chunk=0
budget=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --backend) shift; [ "$1" = cpu ] || exit 3; backend=1 ;;
    --precision) shift; [ "$1" = f32 ] || exit 4; precision=1 ;;
    --chunk-ms) shift; [ "$1" = 2000 ] || exit 5; chunk=1 ;;
    --max-new-tokens) shift; [ "$1" = 32 ] || exit 7; budget=1 ;;
  esac
  shift
done
[ "$backend$precision$chunk$budget" = 1111 ] || exit 6
printf '%s\n' '{"text":"十三","language":"zh","is_final":true,"cancelled":false,"token_limit_reached":false}'
"##).unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    let result = transcribe_v3_r2t2(
        &audio,
        &model,
        &binary,
        V3AsrMode::CpuF32,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(result.text, "十三");
    assert_eq!(result.language, "zh");
}

#[test]
fn cpu_r2t2_rejects_a_truncated_final_event() {
    use omni_rsi::number_v3_evaluation::{V3AsrMode, transcribe_v3_r2t2};
    use std::{fs, os::unix::fs::PermissionsExt, sync::atomic::AtomicBool};
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    fs::create_dir(&model).unwrap();
    let audio = root.path().join("audio.wav");
    fs::write(&audio, b"fixture").unwrap();
    let binary = root.path().join("fake-r2t2");
    fs::write(&binary, "#!/bin/sh\nprintf '%s\\n' '{\"text\":\"十三\",\"is_final\":true,\"cancelled\":false,\"token_limit_reached\":true}'\n").unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    let error = transcribe_v3_r2t2(
        &audio,
        &model,
        &binary,
        V3AsrMode::CpuF32,
        &AtomicBool::new(false),
    )
    .err()
    .unwrap();
    assert!(error.contains("token limit"), "{error}");
}

#[test]
fn both_v3_asr_modes_pass_every_frozen_option_explicitly() {
    use omni_rsi::number_v3_evaluation::{V3AsrMode, transcribe_v3_r2t2};
    use std::{fs, os::unix::fs::PermissionsExt, sync::atomic::AtomicBool};
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    fs::create_dir(&model).unwrap();
    let audio = root.path().join("audio.wav");
    fs::write(&audio, b"fixture").unwrap();
    let binary = root.path().join("fake-r2t2");
    fs::write(&binary, "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$(dirname \"$0\")/args\"\nprintf '%s\\n' '{\"text\":\"十三\",\"is_final\":true,\"cancelled\":false,\"token_limit_reached\":false}'\n").unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    for (mode, backend, precision, tokens) in [
        (V3AsrMode::MetalF16, "metal", "f16", "4"),
        (V3AsrMode::CpuF32, "cpu", "f32", "32"),
    ] {
        transcribe_v3_r2t2(&audio, &model, &binary, mode, &AtomicBool::new(false)).unwrap();
        let captured = fs::read_to_string(root.path().join("args")).unwrap();
        let args: Vec<_> = captured.lines().collect();
        for (name, value) in [
            ("--language", "auto"),
            ("--backend", backend),
            ("--precision", precision),
            ("--chunk-ms", "2000"),
            ("--lookahead-ms", "160"),
            ("--unfixed-tokens", "1"),
            ("--max-new-tokens", tokens),
            ("--final-max-tokens", "128"),
            ("--window-seconds", "16"),
            ("--max-context-tokens", "4096"),
            ("--text-postprocess", "legacy"),
            ("--stable-tail-ms", "0"),
            ("--final-padding-ms", "0"),
            ("--context", ""),
        ] {
            assert!(
                args.windows(2).any(|pair| pair == [name, value]),
                "missing {name}={value}: {args:?}"
            );
        }
        for flag in [
            "--json=true",
            "--realtime=false",
            "--warmup=false",
            "--no-cache=false",
        ] {
            assert!(args.contains(&flag), "missing explicit {flag}: {args:?}");
        }
    }
}

#[test]
fn v3_asr_rejects_cancelled_intermediate_or_final_events() {
    use omni_rsi::number_v3_evaluation::{V3AsrMode, transcribe_v3_r2t2};
    use std::{fs, os::unix::fs::PermissionsExt, sync::atomic::AtomicBool};
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    fs::create_dir(&model).unwrap();
    let audio = root.path().join("audio.wav");
    fs::write(&audio, b"fixture").unwrap();
    let binary = root.path().join("fake-r2t2");
    for (first, last) in [(true, false), (false, true)] {
        let first = format!(
            "{{\"text\":\"十\",\"is_final\":false,\"cancelled\":{first},\"token_limit_reached\":false}}"
        );
        let last = format!(
            "{{\"text\":\"十三\",\"is_final\":true,\"cancelled\":{last},\"token_limit_reached\":false}}"
        );
        fs::write(
            &binary,
            format!("#!/bin/sh\nprintf '%s\\n' '{first}' '{last}'\n"),
        )
        .unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
        for mode in [V3AsrMode::MetalF16, V3AsrMode::CpuF32] {
            let error = transcribe_v3_r2t2(&audio, &model, &binary, mode, &AtomicBool::new(false))
                .expect_err("cancelled event accepted");
            assert!(error.contains("cancel"), "{error}");
        }
    }
}

#[test]
fn v3_cpu_asr_can_be_cancelled_promptly_despite_longer_timeout() {
    use omni_rsi::number_v3_evaluation::{V3AsrMode, transcribe_v3_r2t2};
    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::{Duration, Instant},
    };
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    fs::create_dir(&model).unwrap();
    let audio = root.path().join("audio.wav");
    fs::write(&audio, b"fixture").unwrap();
    let binary = root.path().join("fake-r2t2");
    fs::write(&binary, "#!/bin/sh\nexec sleep 30\n").unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let cancel = Arc::clone(&stop);
    let worker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(250));
        cancel.store(true, Ordering::Release);
    });
    let started = Instant::now();
    let result = transcribe_v3_r2t2(&audio, &model, &binary, V3AsrMode::CpuF32, &stop);
    worker.join().unwrap();
    assert!(result.unwrap_err().contains("interrupt"));
    assert!(started.elapsed() < Duration::from_secs(5));
}
