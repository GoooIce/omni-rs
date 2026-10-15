//! Export immutable completed output without occupying a model worker.
use std::{
    path::{Path, PathBuf},
    sync::{
        mpsc::{self, Receiver},
        Arc,
    },
};
#[derive(Clone)]
pub enum Snapshot {
    Audio(Arc<[f32]>),
    Image(Arc<omni_qwen_image::GeneratedImage>),
    Music(Arc<[f32]>),
    Bytes(Arc<[u8]>),
}
impl Snapshot {
    pub fn save(
        self,
        path: PathBuf,
        notify: impl FnOnce() + Send + 'static,
    ) -> Receiver<Result<String, String>> {
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match self {
                Self::Audio(samples) => save_wav(&samples, &path),
                Self::Image(image) => image.save_png(&path).map_err(|e| e.to_string()),
                Self::Music(samples) => atomic_write(&path, |temporary| {
                    omni_music::audio::write_wav(temporary, &samples, 48000, 2)
                        .map_err(|e| e.to_string())
                }),
                Self::Bytes(bytes) => atomic_write(&path, |temporary| {
                    std::fs::write(temporary, &bytes).map_err(|e| e.to_string())
                }),
            }))
            .unwrap_or_else(|_| Err("保存失败，请重试".into()))
            .map(|_| format!("已保存：{}", path.display()));
            let _ = sender.send(result);
            notify();
        });
        receiver
    }
}
fn save_wav(samples: &[f32], path: &Path) -> Result<(), String> {
    atomic_write(path, |temporary| {
        let mut writer = hound::WavWriter::create(
            temporary,
            hound::WavSpec {
                channels: 1,
                sample_rate: omni_breeze_tts::SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .map_err(|e| e.to_string())?;
        for value in samples {
            writer
                .write_sample((value.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
                .map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())
    })
}
fn atomic_write(
    path: &Path,
    write: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let mut name = path.file_name().ok_or("无效的保存位置")?.to_os_string();
    name.push(format!(
        ".{}.part",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let temporary = Temporary(path.with_file_name(name));
    write(&temporary.0)?;
    std::fs::rename(&temporary.0, path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn music_export_preserves_stereo_rate_and_failed_save_keeps_the_previous_file() {
        let directory =
            std::env::temp_dir().join(format!("omni-music-export-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("song.wav");
        let pcm: Arc<[f32]> = Arc::from([0.25, -0.5, 0.125, 0.75]);
        Snapshot::Music(pcm.clone())
            .save(path.clone(), || {})
            .recv()
            .unwrap()
            .unwrap();
        let before = std::fs::read(&path).unwrap();
        let mut wav = hound::WavReader::open(&path).unwrap();
        assert_eq!((wav.spec().sample_rate, wav.spec().channels), (48000, 2));
        assert_eq!(
            wav.samples::<f32>().collect::<Result<Vec<_>, _>>().unwrap(),
            pcm.as_ref()
        );
        assert!(Snapshot::Music(Arc::from([f32::NAN, 0.0]))
            .save(path.clone(), || {})
            .recv()
            .unwrap()
            .is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        let text = directory.join("score.abc");
        Snapshot::Bytes(Arc::from(b"X:1\nK:C\nCDEF".as_slice()))
            .save(text.clone(), || {})
            .recv()
            .unwrap()
            .unwrap();
        assert_eq!(std::fs::read(&text).unwrap(), b"X:1\nK:C\nCDEF");
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn a_completed_snapshot_can_be_exported_while_the_runtime_is_occupied() {
        let runtime = crate::desktop_runtime::Runtime::default();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let _running = runtime.acquire("another model", stop).unwrap();
        let path = std::env::temp_dir().join(format!(
            "omni-export-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let snapshot = Snapshot::Audio(Arc::from(vec![0.25, -0.5]));
        let saved = snapshot
            .save(path.clone(), || {})
            .recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        assert!(saved.is_ok());
        assert!(runtime.busy());
        let mut wav = hound::WavReader::open(&path).unwrap();
        let samples: Vec<i16> = wav.samples().collect::<Result<_, _>>().unwrap();
        assert_eq!(samples, [8192, -16384]);
        std::fs::remove_file(path).unwrap();
    }
}
