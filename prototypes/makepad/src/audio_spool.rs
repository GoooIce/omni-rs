//! One writer and one reader. PCM is temporary, segmented, and deleted as read.
//! Disk I/O belongs to the capture supervisor, never the device callback.
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
};

pub struct Spool {
    directory: PathBuf,
    segment_samples: usize,
    pub written: AtomicUsize,
    pub read: AtomicUsize,
    ready: (Mutex<()>, Condvar),
}
impl Spool {
    pub fn new(segment_samples: usize) -> io::Result<Arc<Self>> {
        static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "r2t2-audio-{}-{stamp}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&directory)?;
        Ok(Arc::new(Self {
            directory,
            segment_samples: segment_samples.max(1),
            written: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
            ready: (Mutex::new(()), Condvar::new()),
        }))
    }
    fn path(&self, sample: usize) -> PathBuf {
        self.directory
            .join(format!("{}.pcm", sample / self.segment_samples))
    }
    pub fn pending(&self) -> usize {
        let read = self.read.load(Ordering::Acquire);
        self.written.load(Ordering::Acquire).saturating_sub(read)
    }

    /// Called by the spool writer/supervisor, never the real-time callback.
    pub fn notify_reader(&self) {
        let _guard = self.ready.0.lock().unwrap();
        self.ready.1.notify_one();
    }

    pub fn wait_readable(&self, stop: &AtomicBool, finished: &AtomicBool) {
        let guard = self.ready.0.lock().unwrap();
        let _ = self
            .ready
            .1
            .wait_timeout_while(guard, std::time::Duration::from_millis(100), |_| {
                self.pending() == 0
                    && !stop.load(Ordering::Acquire)
                    && !finished.load(Ordering::Acquire)
            })
            .unwrap();
    }
}
impl Drop for Spool {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

pub struct Writer {
    file: Option<File>,
    spool: Arc<Spool>,
    position: usize,
    bytes: Vec<u8>,
}
impl Writer {
    pub fn new(spool: Arc<Spool>) -> Self {
        Self {
            file: None,
            spool,
            position: 0,
            bytes: Vec::new(),
        }
    }
    pub fn push(&mut self, mut pcm: &[f32]) -> io::Result<()> {
        while !pcm.is_empty() {
            if self.file.is_none() {
                self.file = Some(
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(self.spool.path(self.position))?,
                );
            }
            let count = pcm
                .len()
                .min(self.spool.segment_samples - self.position % self.spool.segment_samples);
            self.bytes.clear();
            for sample in &pcm[..count] {
                self.bytes.extend_from_slice(&sample.to_le_bytes());
            }
            self.file.as_mut().unwrap().write_all(&self.bytes)?;
            self.position += count;
            // Close before publishing a complete segment so Windows can delete it.
            if self.position % self.spool.segment_samples == 0 {
                self.file = None;
            }
            self.spool.written.store(self.position, Ordering::Release);
            self.spool.notify_reader();
            pcm = &pcm[count..];
        }
        Ok(())
    }
}

pub struct Reader {
    file: Option<File>,
    spool: Arc<Spool>,
    position: usize,
    bytes: Vec<u8>,
}
impl Reader {
    pub fn new(spool: Arc<Spool>) -> Self {
        Self {
            file: None,
            spool,
            position: 0,
            bytes: Vec::new(),
        }
    }
    pub fn pop(&mut self, output: &mut [f32]) -> io::Result<usize> {
        let available = self
            .spool
            .written
            .load(Ordering::Acquire)
            .saturating_sub(self.position);
        let count = available
            .min(output.len())
            .min(self.spool.segment_samples - self.position % self.spool.segment_samples);
        if count == 0 {
            return Ok(0);
        }
        if self.file.is_none() {
            self.file = Some(File::open(self.spool.path(self.position))?);
        }
        self.bytes.resize(count * 4, 0);
        self.file.as_mut().unwrap().read_exact(&mut self.bytes)?;
        for (sample, bytes) in output.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *sample = f32::from_le_bytes(bytes.try_into().unwrap());
        }
        let old_path = self.spool.path(self.position);
        self.position += count;
        if self.position % self.spool.segment_samples == 0 {
            self.file = None;
            fs::remove_file(old_path)?;
        }
        self.spool.read.store(self.position, Ordering::Release);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stalled_consumer_has_no_duration_limit_and_segments_are_reclaimed() {
        let spool = Spool::new(100).unwrap();
        let path = spool.directory.clone();
        let mut writer = Writer::new(spool.clone());
        for block in 0..500 {
            writer
                .push(&(0..37).map(|i| (block * 37 + i) as f32).collect::<Vec<_>>())
                .unwrap();
        }
        assert_eq!(spool.pending(), 18500);
        let mut reader = Reader::new(spool.clone());
        let mut out = [0.0; 79];
        let mut received = 0;
        loop {
            let n = reader.pop(&mut out).unwrap();
            if n == 0 {
                break;
            }
            for &sample in &out[..n] {
                assert_eq!(sample, received as f32);
                received += 1;
            }
        }
        assert_eq!(received, 18500);
        assert_eq!(fs::read_dir(&path).unwrap().count(), 0);
        drop(reader);
        drop(writer);
        drop(spool);
        assert!(!path.exists());
    }
    #[test]
    fn partial_segment_can_be_read_while_producer_appends() {
        let spool = Spool::new(5).unwrap();
        let mut writer = Writer::new(spool.clone());
        let mut reader = Reader::new(spool);
        let mut out = [0.0; 4];
        writer.push(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(reader.pop(&mut out).unwrap(), 3);
        assert_eq!(&out[..3], &[1.0, 2.0, 3.0]);
        writer.push(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(reader.pop(&mut out).unwrap(), 2);
        assert_eq!(&out[..2], &[4.0, 5.0]);
        assert_eq!(reader.pop(&mut out).unwrap(), 1);
        assert_eq!(out[0], 6.0);
    }

    #[test]
    fn notification_and_completion_cannot_lose_a_partial_tail() {
        let spool = Spool::new(5).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let worker_spool = spool.clone();
        let worker_stop = stop.clone();
        let worker_finished = finished.clone();
        let reader = std::thread::spawn(move || {
            let mut reader = Reader::new(worker_spool.clone());
            let mut samples = Vec::new();
            let mut packet = [0.0; 4];
            loop {
                let n = reader.pop(&mut packet).unwrap();
                samples.extend_from_slice(&packet[..n]);
                if n == 0 {
                    if worker_finished.load(Ordering::Acquire) && worker_spool.pending() == 0 {
                        return samples;
                    }
                    worker_spool.wait_readable(&worker_stop, &worker_finished);
                }
            }
        });
        let mut writer = Writer::new(spool.clone());
        writer.push(&[1.0, 2.0, 3.0]).unwrap();
        finished.store(true, Ordering::Release);
        spool.notify_reader();
        assert_eq!(reader.join().unwrap(), [1.0, 2.0, 3.0]);
        stop.store(true, Ordering::Release);
        spool.wait_readable(&stop, &AtomicBool::new(false));
    }
}
