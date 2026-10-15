//! Complete files outlive the job history that produced them.
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};
#[derive(Clone)]
pub(crate) struct Results(Arc<PathBuf>);
pub(crate) struct PreparedFile {
    staging: PathBuf,
    published: PathBuf,
}
impl Results {
    pub fn new(root: PathBuf) -> Self {
        Self(Arc::new(root))
    }
    pub fn prepare<E: From<io::Error>>(
        &self,
        name: &str,
        encode: impl FnOnce(&Path) -> Result<(), E>,
    ) -> Result<PreparedFile, E> {
        let (id, _) = parse_name(name)?;
        fs::create_dir_all(self.0.join(".staging"))?;
        fs::create_dir_all(self.0.join("published"))?;
        let staging = self.0.join(".staging").join(id);
        fs::create_dir(&staging)?;
        let prepared = PreparedFile {
            staging,
            published: self.0.join("published").join(id),
        };
        encode(&prepared.staging.join(name))?;
        Ok(prepared)
    }
    pub fn read(&self, name: &str) -> io::Result<(Vec<u8>, &'static str)> {
        let (id, content_type) = parse_name(name)?;
        Ok((
            fs::read(self.0.join("published").join(id).join(name))?,
            content_type,
        ))
    }
}
fn parse_name(name: &str) -> io::Result<(&str, &'static str)> {
    let invalid = || io::Error::new(io::ErrorKind::NotFound, "unknown published result");
    let (id, ext) = name.rsplit_once('.').ok_or_else(invalid)?;
    let (nonce, sequence) = id.split_once('-').ok_or_else(invalid)?;
    if id.len() > 80
        || nonce.is_empty()
        || sequence.is_empty()
        || !nonce.bytes().all(|b| b.is_ascii_hexdigit())
        || !sequence.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(invalid());
    }
    Ok((
        id,
        match ext {
            "wav" => "audio/wav",
            "png" => "image/png",
            "json" => "application/json",
            "abc" => "text/vnd.abc; charset=utf-8",
            "mid" => "audio/midi",
            _ => return Err(invalid()),
        },
    ))
}
impl PreparedFile {
    /// The directory rename is the publication point, including after restart.
    pub fn publish(self) -> io::Result<()> {
        fs::rename(&self.staging, &self.published)
    }
}
impl Drop for PreparedFile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.staging);
    }
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub struct Directory(pub PathBuf);
    impl Directory {
        pub fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            Self(std::env::temp_dir().join(format!(
                    "omni-results-{}-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos(),
                    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )))
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn publication_is_atomic_and_survives_reopening() {
        let dir = Directory::new();
        let files = Results::new(dir.0.clone());
        let prepared = files
            .prepare::<io::Error>("abcd-1.wav", |path| {
                let mut writer = hound::WavWriter::create(
                    path,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: 24000,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )
                .unwrap();
                writer.write_sample(123i16).unwrap();
                assert!(files.read("abcd-1.wav").is_err());
                writer.finalize().unwrap();
                Ok(())
            })
            .unwrap();
        assert!(files.read("abcd-1.wav").is_err());
        prepared.publish().unwrap();
        let reopened = Results::new(dir.0.clone());
        let (bytes, mime) = reopened.read("abcd-1.wav").unwrap();
        assert_eq!(mime, "audio/wav");
        let mut wav = hound::WavReader::new(io::Cursor::new(bytes)).unwrap();
        assert_eq!(wav.samples::<i16>().next().unwrap().unwrap(), 123);
        fs::remove_file(dir.0.join("published/abcd-1/abcd-1.wav")).unwrap();
        assert!(reopened.read("abcd-1.wav").is_err());
    }
    #[test]
    fn errors_and_unwinds_remove_the_whole_staging_directory() {
        let dir = Directory::new();
        let files = Results::new(dir.0.clone());
        let result = files.prepare::<io::Error>("abcd-2.png", |path| {
            fs::write(path.with_extension("png.part"), b"unfinished")?;
            Err(io::Error::other("disk error"))
        });
        assert!(result.is_err());
        let panic = std::panic::catch_unwind(|| {
            files.prepare::<io::Error>("abcd-3.png", |path| {
                fs::write(path, b"unfinished")?;
                panic!("encoder panic");
            })
        });
        assert!(panic.is_err());
        assert_eq!(fs::read_dir(dir.0.join(".staging")).unwrap().count(), 0);
        for name in [
            "abcd-2.png",
            "abcd-3.png",
            "../abcd-3.png",
            "abcd-3.png.part",
        ] {
            assert!(files.read(name).is_err());
        }
    }
}
