use omni_lfm3::{
    media_io::{decode_video_bytes, decode_wav},
    video::VideoConfig,
};
use std::{io::Cursor, sync::atomic::AtomicBool};

#[test]
fn wav_input_is_downmixed_resampled_and_rejects_invalid_data() {
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 2,
                sample_rate: 8_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..800 {
            writer.write_sample(10_000i16).unwrap();
            writer.write_sample(-2_000i16).unwrap();
        }
        writer.finalize().unwrap();
    }
    let audio = decode_wav(bytes.get_ref(), &AtomicBool::new(false)).unwrap();
    assert_eq!(audio.len(), 1600);
    assert!(audio.iter().all(|v| v.is_finite()) && (audio[800] - 4000.0 / 32768.0).abs() < 0.001);
    assert!(decode_wav(b"not a wav", &AtomicBool::new(false)).is_err());
}

#[test]
#[ignore = "requires ffmpeg and ffprobe"]
fn video_file_decoding_keeps_frame_clock_and_soundtrack() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("video.mkv");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=2:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=2",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            "-y",
        ])
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    let input = std::fs::read(path).unwrap();
    let video =
        decode_video_bytes(&input, &VideoConfig::default(), &AtomicBool::new(false)).unwrap();
    assert_eq!(video.indices, vec![0, 1, 2, 3]);
    assert_eq!(video.fps, 2.0);
    assert_eq!(video.frames.len(), 4);
    assert!(video.audio.unwrap().len() >= 30_000);
}
