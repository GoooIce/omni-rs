//! Single producer (audio callback), single consumer (inference thread).
//! Atomic sample slots avoid unsafe code; neither side waits for the other.
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub struct CaptureBuffer {
    samples: Box<[AtomicU32]>,
    written: AtomicUsize,
    read: AtomicUsize,
}

impl CaptureBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            samples: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            written: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
        }
    }

    /// Commit a whole callback, or nothing. No allocation, locks, or partial gaps.
    pub fn push<T: Copy>(
        &self,
        input: &[T],
        channels: usize,
        convert: impl Fn(T) -> f32,
    ) -> Result<f32, &'static str> {
        if channels == 0 || input.len() % channels != 0 {
            return Err("invalid audio frame");
        }
        let head = self.written.load(Ordering::Relaxed);
        let tail = self.read.load(Ordering::Acquire);
        let frames = input.len() / channels;
        if frames > self.samples.len() - head.wrapping_sub(tail) {
            return Err("overflow");
        }
        let mut peak = 0.0_f32;
        for (i, frame) in input.chunks_exact(channels).enumerate() {
            let sample = frame.iter().map(|&s| convert(s)).sum::<f32>() / channels as f32;
            if !sample.is_finite() {
                return Err("non-finite audio");
            }
            peak = peak.max(sample.abs());
            self.samples[head.wrapping_add(i) % self.samples.len()]
                .store(sample.to_bits(), Ordering::Relaxed);
        }
        self.written
            .store(head.wrapping_add(frames), Ordering::Release);
        Ok(peak)
    }

    pub fn pop(&self, output: &mut [f32]) -> usize {
        let tail = self.read.load(Ordering::Relaxed);
        let head = self.written.load(Ordering::Acquire);
        let count = output.len().min(head.wrapping_sub(tail));
        for (i, sample) in output.iter_mut().take(count).enumerate() {
            *sample = f32::from_bits(
                self.samples[tail.wrapping_add(i) % self.samples.len()].load(Ordering::Relaxed),
            );
        }
        self.read.store(tail.wrapping_add(count), Ordering::Release);
        count
    }

    pub fn captured(&self) -> usize {
        self.written.load(Ordering::Acquire)
    }

    pub fn pending(&self) -> usize {
        // Read the consumer first so a concurrent producer cannot appear behind it.
        let tail = self.read.load(Ordering::Acquire);
        self.written.load(Ordering::Acquire).wrapping_sub(tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_wraparound_and_overflow_preserve_the_accepted_prefix() {
        let ring = CaptureBuffer::new(4);
        assert_eq!(ring.push(&[1_i16, 3, 2, 4, 3, 5], 2, |x| x as f32), Ok(4.0));
        assert_eq!(ring.push(&[9.0, 9.0], 1, |x| x), Err("overflow"));
        let mut out = [0.0; 2];
        assert_eq!(ring.pop(&mut out), 2);
        assert_eq!(out, [2.0, 3.0]);
        ring.push(&[5.0, 6.0, 7.0], 1, |x| x).unwrap();
        let mut tail = [0.0; 5];
        assert_eq!(ring.pop(&mut tail), 4);
        assert_eq!(&tail[..4], &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(ring.pending(), 0);
        assert_eq!(ring.captured(), 6);
    }

    #[test]
    fn invalid_callback_never_commits_partial_audio() {
        let ring = CaptureBuffer::new(8);
        assert!(ring.push(&[1.0, f32::NAN], 1, |x| x).is_err());
        assert!(ring.push(&[1.0], 2, |x| x).is_err());
        assert_eq!(ring.pending(), 0);
        ring.push(&[2.0], 1, |x| x).unwrap();
        let mut out = [0.0; 2];
        assert_eq!(ring.pop(&mut out), 1);
        assert_eq!(out[0], 2.0);
    }

    #[test]
    fn concurrent_callbacks_and_drain_keep_every_sample_in_order() {
        let ring = std::sync::Arc::new(CaptureBuffer::new(257));
        let producer = ring.clone();
        let task = std::thread::spawn(move || {
            for i in 0..100_000 {
                while producer.push(&[i as f32], 1, |x| x).is_err() {
                    std::thread::yield_now();
                }
            }
        });
        let mut count = 0;
        let mut block = [0.0; 137];
        while count < 100_000 {
            let n = ring.pop(&mut block);
            for &sample in &block[..n] {
                assert_eq!(sample, count as f32);
                count += 1;
            }
            std::thread::yield_now();
        }
        task.join().unwrap();
    }
}
