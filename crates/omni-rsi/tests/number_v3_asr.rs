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
