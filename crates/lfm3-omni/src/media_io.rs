//! Native byte-based media decoding. Video uses local ffmpeg/ffprobe, never a host path supplied by a client.
use crate::video::{RgbFrame, VideoConfig, sample_indices};
use omni_r2t2::{
    Error, Result,
    audio::Resampler,
    input::{AudioSource, WavSource},
};
use serde::Deserialize;
use std::{
    io::{Cursor, Read, Write},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

const MAX_INPUT: usize = 512 * 1024 * 1024;
const MAX_VIDEO_RGB: usize = 512 * 1024 * 1024;
const MAX_AUDIO_PCM: usize = 256 * 1024 * 1024;

/// Decode, downmix and resample a WAV upload to mono 16 kHz.
pub fn decode_wav(bytes: &[u8], cancelled: &AtomicBool) -> Result<Vec<f32>> {
    if bytes.is_empty() || bytes.len() > MAX_INPUT {
        return Err(Error::Audio("invalid or oversized WAV input".into()));
    }
    let mut source = WavSource::new(Cursor::new(bytes))?;
    let mut resampler =
        Resampler::new(source.sample_rate(), 16_000).map_err(|e| Error::Audio(e.into()))?;
    let mut audio = Vec::new();
    while let Some(packet) = source.next_packet(cancelled)? {
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Interrupted);
        }
        audio.extend(
            resampler
                .push(&packet, false)
                .map_err(|e| Error::Audio(e.into()))?,
        );
        if audio.len() > MAX_AUDIO_PCM / 4 {
            return Err(Error::Audio("decoded audio exceeds size limit".into()));
        }
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Interrupted);
    }
    audio.extend(
        resampler
            .push(&[], true)
            .map_err(|e| Error::Audio(e.into()))?,
    );
    if audio.is_empty() {
        return Err(Error::Audio("audio input is empty".into()));
    }
    Ok(audio)
}

/// Decode common audio formats to mono 16 kHz. WAV uses the native adapter;
/// other formats use the local ffmpeg executable.
pub fn decode_audio_bytes(bytes: &[u8], cancelled: &AtomicBool) -> Result<Vec<f32>> {
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        return decode_wav(bytes, cancelled);
    }
    if bytes.is_empty() || bytes.len() > MAX_INPUT {
        return Err(Error::Audio("invalid or oversized audio input".into()));
    }
    let mut file = tempfile::Builder::new().prefix("omni-audio-").tempfile()?;
    file.write_all(bytes)?;
    file.flush()?;
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-i"]).arg(file.path()).args([
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
    ]);
    let bytes = run_bounded(command, "ffmpeg audio", MAX_AUDIO_PCM, cancelled)?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error::Audio("invalid decoded audio length".into()));
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .collect();
    if samples.iter().any(|v| !v.is_finite()) {
        return Err(Error::Audio("non-finite decoded audio".into()));
    }
    Ok(samples)
}

pub struct DecodedVideo {
    pub frames: Vec<RgbFrame>,
    pub indices: Vec<usize>,
    pub fps: f64,
    pub audio: Option<Vec<f32>>,
}
#[derive(Deserialize)]
struct Probe {
    streams: Vec<Stream>,
}
#[derive(Deserialize)]
struct Stream {
    codec_type: String,
    width: Option<usize>,
    height: Option<usize>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    nb_read_frames: Option<String>,
    start_time: Option<String>,
}
fn rate(value: &str) -> Option<f64> {
    let (n, d) = value.split_once('/')?;
    let (n, d) = (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?);
    let rate = n / d;
    (rate.is_finite() && rate > 0.0).then_some(rate)
}
fn start_time(value: &Option<String>) -> f64 {
    value
        .as_deref()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
}

/// Decode video bytes and a soundtrack, retaining sampled source indices and clock.
/// This requires `ffprobe` and `ffmpeg` on PATH; each process is cancellable.
pub fn decode_video_bytes(
    bytes: &[u8],
    config: &VideoConfig,
    cancelled: &AtomicBool,
) -> Result<DecodedVideo> {
    if bytes.is_empty() || bytes.len() > MAX_INPUT {
        return Err(Error::Inference("invalid or oversized video input".into()));
    }
    let mut file = tempfile::Builder::new().prefix("omni-video-").tempfile()?;
    file.write_all(bytes)?;
    file.flush()?;
    let mut probe = Command::new("ffprobe");
    probe
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,start_time",
            "-of",
            "json",
        ])
        .arg(file.path());
    let probe: Probe =
        serde_json::from_slice(&run_bounded(probe, "ffprobe", 2 * 1024 * 1024, cancelled)?)
            .map_err(|e| Error::Inference(format!("invalid video metadata: {e}")))?;
    let video = probe
        .streams
        .iter()
        .find(|s| s.codec_type == "video")
        .ok_or_else(|| Error::Inference("video has no video stream".into()))?;
    let width = video
        .width
        .filter(|v| *v > 0 && *v <= 16_384)
        .ok_or_else(|| Error::Inference("invalid video width".into()))?;
    let height = video
        .height
        .filter(|v| *v > 0 && *v <= 16_384)
        .ok_or_else(|| Error::Inference("invalid video height".into()))?;
    let fps = video
        .avg_frame_rate
        .as_deref()
        .and_then(rate)
        .ok_or_else(|| Error::Inference("video is missing a frame rate".into()))?;
    if let Some(nominal) = video.r_frame_rate.as_deref().and_then(rate) {
        if ((nominal - fps) / fps).abs() > 0.005 {
            return Err(Error::Inference(
                "variable-frame-rate video is not supported without explicit frame timestamps"
                    .into(),
            ));
        }
    }
    let total = video
        .nb_read_frames
        .as_deref()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Inference("video frame count is unavailable".into()))?;
    let indices = sample_indices(total, fps, config)?;
    if indices.len() < config.temporal_patch_size {
        return Err(Error::Inference("video has too few frames".into()));
    }
    let frame_bytes = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Inference("video dimensions overflow".into()))?;
    let expected = frame_bytes
        .checked_mul(indices.len())
        .ok_or_else(|| Error::Inference("video size overflow".into()))?;
    if expected > MAX_VIDEO_RGB {
        return Err(Error::Inference("decoded video exceeds size limit".into()));
    }
    let filter = indices
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-i"]).arg(file.path()).args([
        "-map",
        "0:v:0",
        "-vf",
        &format!("select={filter}"),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]);
    let raw = run_bounded(command, "ffmpeg video", expected, cancelled)?;
    if raw.len() != expected {
        return Err(Error::Inference(format!(
            "video decoder returned {} bytes; expected {expected}",
            raw.len()
        )));
    }
    let frames = raw
        .chunks_exact(frame_bytes)
        .map(|rgb| RgbFrame {
            width,
            height,
            rgb: rgb.to_vec(),
        })
        .collect();
    let audio = if let Some(stream) = probe.streams.iter().find(|s| s.codec_type == "audio") {
        let mut command = Command::new("ffmpeg");
        command.args(["-v", "error", "-i"]).arg(file.path()).args([
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "asetpts=PTS-STARTPTS",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ]);
        let bytes = run_bounded(command, "ffmpeg audio", MAX_AUDIO_PCM, cancelled)?;
        if !bytes.len().is_multiple_of(4) {
            return Err(Error::Audio("invalid decoded soundtrack length".into()));
        }
        let mut pcm: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect();
        if pcm.iter().any(|v| !v.is_finite()) {
            return Err(Error::Audio("non-finite soundtrack".into()));
        }
        let offset = ((start_time(&stream.start_time) - start_time(&video.start_time)) * 16_000.0)
            .round() as i64;
        if offset > 0 {
            let pad = offset as usize;
            if pad > MAX_AUDIO_PCM / 4 || pcm.len() > MAX_AUDIO_PCM / 4 - pad {
                return Err(Error::Audio("soundtrack offset exceeds size limit".into()));
            }
            let mut aligned = vec![0.0; pad];
            aligned.extend(pcm);
            pcm = aligned;
        } else if offset < 0 {
            let skip = offset.unsigned_abs().min(pcm.len() as u64) as usize;
            pcm.drain(..skip);
        }
        if pcm.is_empty() {
            return Err(Error::Audio("video soundtrack is empty".into()));
        }
        Some(pcm)
    } else {
        None
    };
    Ok(DecodedVideo {
        frames,
        indices,
        fps,
        audio,
    })
}

fn run_bounded(
    mut command: Command,
    label: &str,
    limit: usize,
    cancelled: &AtomicBool,
) -> Result<Vec<u8>> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Interrupted);
    }
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::Inference(format!("cannot start {label}: {e}")))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::Inference(format!("cannot read {label} output")))?;
    let exceeded = Arc::new(AtomicBool::new(false));
    let marker = exceeded.clone();
    let reader = thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        stdout.take(limit as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > limit {
            marker.store(true, Ordering::Relaxed);
        }
        Ok(bytes)
    });
    let status = loop {
        if cancelled.load(Ordering::Relaxed) || exceeded.load(Ordering::Relaxed) {
            let _ = child.kill();
            break child.wait()?;
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        thread::sleep(Duration::from_millis(20));
    };
    let output = reader
        .join()
        .map_err(|_| Error::Inference(format!("{label} output reader failed")))??;
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Interrupted);
    }
    if exceeded.load(Ordering::Relaxed) || output.len() > limit {
        return Err(Error::Inference(format!(
            "{label} output exceeds size limit"
        )));
    }
    if !status.success() {
        return Err(Error::Inference(format!("{label} failed with {status}")));
    }
    Ok(output)
}
