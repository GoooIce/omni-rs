use crate::engine_harness::Publisher;
use crate::{
    audio_spool::{Spool, Writer},
    capture_buffer::CaptureBuffer,
    protocol::Update,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering},
        mpsc::{self, SyncSender},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

// Device callback -> disk writer handoff only; inference never consumes this ring.
const DEVICE_BUFFER_SECONDS: usize = 8;

pub struct Shared {
    pub buffer: CaptureBuffer,
    pub spool: Arc<Spool>,
    pub finished: AtomicBool,
    peak: AtomicU32,
    failure: AtomicU8,
    stop: Arc<AtomicBool>,
}

impl Shared {
    fn new(rate: u32, stop: Arc<AtomicBool>) -> std::io::Result<Self> {
        Ok(Self {
            buffer: CaptureBuffer::new(rate as usize * DEVICE_BUFFER_SECONDS),
            spool: Spool::new(rate as usize * 2)?,
            finished: AtomicBool::new(false),
            peak: AtomicU32::new(0),
            failure: AtomicU8::new(0),
            stop,
        })
    }

    fn halted(&self) -> bool {
        self.stop.load(Ordering::Acquire) || self.failure.load(Ordering::Acquire) != 0
    }

    fn enqueue<T: Copy>(&self, data: &[T], channels: usize, convert: impl Fn(T) -> f32) {
        if self.halted() {
            return;
        }
        match self.buffer.push(data, channels, convert) {
            Ok(peak) => {
                self.peak.fetch_max(peak.to_bits(), Ordering::Relaxed);
            }
            Err(error) => {
                self.failure
                    .store(if error == "overflow" { 1 } else { 3 }, Ordering::Release);
            }
        }
    }

    pub fn warning(&self) -> &'static str {
        match self.failure.load(Ordering::Acquire) {
            1 => "输入设备缓冲溢出，临时存储写入未能跟上录音；请检查磁盘状态",
            2 => "输入设备中断，已停录并保留已接收音频的转写",
            3 => "输入设备返回无效音频，已停录并保留此前的转写",
            4 => "连续 5 秒未收到录音数据，已停录；请检查系统麦克风权限和输入设备",
            5 => "无法写入录音临时缓存，请检查磁盘剩余空间；已保留此前的转写",
            _ => "",
        }
    }
}

pub struct Microphone {
    pub shared: Arc<Shared>,
    pub rate: u32,
    thread: Option<JoinHandle<()>>,
}

impl Microphone {
    pub fn start(stop: Arc<AtomicBool>, updates: Publisher<Update>) -> Result<Self, String> {
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        // CPAL streams are owned and dropped here, independently of a busy GPU.
        let thread = thread::spawn(move || {
            let result = start_source(stop, &updates, &ready_tx);
            if let Err(error) = result {
                let _ = ready_tx.send(Err(error));
            }
        });
        match ready_rx.recv().map_err(|e| e.to_string())? {
            Ok((shared, rate)) => Ok(Self {
                shared,
                rate,
                thread: Some(thread),
            }),
            Err(error) => {
                let _ = thread.join();
                Err(error)
            }
        }
    }
}

impl Drop for Microphone {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

type Ready = SyncSender<Result<(Arc<Shared>, u32), String>>;

fn meter(shared: &Shared, rate: u32, name: &str, updates: &Publisher<Update>, live: bool) {
    let update = Update {
        kind: "capture".into(),
        message: if live {
            format!("录音中 · {name}")
        } else if shared.stop.load(Ordering::Acquire) {
            "麦克风已释放 · 正在打断识别…".into()
        } else if shared.warning().is_empty() {
            "输入已结束 · 正在处理尾音，仍可点击停止打断".into()
        } else {
            format!(
                "输入已停止 · 正在处理剩余音频… {}",
                shared.warning().split('；').next().unwrap_or_default()
            )
        },
        capturing: live,
        captured_seconds: shared.buffer.captured() as f64 / rate as f64,
        backlog_seconds: (shared.spool.pending() + shared.buffer.pending()) as f64 / rate as f64,
        level: f32::from_bits(shared.peak.swap(0, Ordering::Relaxed)) as f64,
        ..Default::default()
    };
    // A meter tick may be coalesced; PCM never travels in this queue.
    updates.try_publish(update);
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels as usize;
    let failure = shared.clone();
    device
        .build_input_stream(
            config,
            move |data: &[T], _| shared.enqueue(data, channels, |s| s.to_sample::<f32>()),
            move |error| {
                eprintln!("[microphone] {error}");
                failure.failure.store(2, Ordering::Release);
            },
            None,
        )
        .map_err(|e| format!("无法打开麦克风，请检查系统录音权限及默认输入设备：{e}"))
}

fn start_source(
    stop: Arc<AtomicBool>,
    updates: &Publisher<Update>,
    ready: &Ready,
) -> Result<(), String> {
    // Explicit reproducible test mode; never requests device access and labels the UI.
    let args: Vec<_> = std::env::args().collect();
    if let Some(index) = args.iter().position(|a| a == "--mic-test-wav") {
        let path = args.get(index + 1).ok_or("--mic-test-wav 缺少文件路径")?;
        return test_source(path, stop, updates, ready);
    }
    let device = cpal::default_host()
        .default_input_device()
        .ok_or("没有默认麦克风，请在系统设置中选择输入设备")?;
    let supported = device
        .default_input_config()
        .map_err(|e| format!("无法读取麦克风配置：{e}"))?;
    let config: cpal::StreamConfig = supported.clone().into();
    let rate = config.sample_rate.0;
    if config.channels == 0 || !(8000..=384000).contains(&rate) {
        return Err("不支持此麦克风采样率或通道数".into());
    }
    let name = device.name().unwrap_or_else(|_| "默认输入设备".into());
    let shared = Arc::new(Shared::new(rate, stop).map_err(|e| format!("无法创建录音缓存：{e}"))?);
    let mut writer = Writer::new(shared.spool.clone());
    let mut packet = vec![0.0; rate as usize / 10];
    macro_rules! stream {
        ($t:ty) => {
            build_stream::<$t>(&device, &config, shared.clone())?
        };
    }
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => stream!(f32),
        cpal::SampleFormat::F64 => stream!(f64),
        cpal::SampleFormat::I8 => stream!(i8),
        cpal::SampleFormat::I16 => stream!(i16),
        cpal::SampleFormat::I32 => stream!(i32),
        cpal::SampleFormat::I64 => stream!(i64),
        cpal::SampleFormat::U8 => stream!(u8),
        cpal::SampleFormat::U16 => stream!(u16),
        cpal::SampleFormat::U32 => stream!(u32),
        cpal::SampleFormat::U64 => stream!(u64),
        format => return Err(format!("不支持的麦克风采样格式：{format:?}")),
    };
    // A stop during the system permission request must not begin capture later.
    if !shared.halted() {
        stream.play().map_err(|e| format!("无法启动麦克风：{e}"))?;
    }
    if ready.send(Ok((shared.clone(), rate))).is_err() {
        return Ok(());
    }
    let mut last_meter = Instant::now() - Duration::from_secs(1);
    let mut last_audio = Instant::now();
    let mut last_count = 0;
    while !shared.halted() {
        drain(&shared, &mut writer, &mut packet);
        let count = shared.buffer.captured();
        if count != last_count {
            last_count = count;
            last_audio = Instant::now();
        } else if last_audio.elapsed() > Duration::from_secs(5) {
            shared.failure.store(4, Ordering::Release);
            break;
        }
        if last_meter.elapsed() >= Duration::from_millis(100) {
            meter(&shared, rate, &name, updates, true);
            last_meter = Instant::now();
        }
        thread::sleep(Duration::from_millis(10));
    }
    drop(stream);
    drain(&shared, &mut writer, &mut packet);
    meter(&shared, rate, &name, updates, false);
    shared.finished.store(true, Ordering::Release);
    shared.spool.notify_reader();
    Ok(())
}

fn test_source(
    path: &str,
    stop: Arc<AtomicBool>,
    updates: &Publisher<Update>,
    ready: &Ready,
) -> Result<(), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.channels == 0
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
        || !(8000..=384000).contains(&spec.sample_rate)
        || reader.duration() > spec.sample_rate * 600
    {
        return Err("测试音源需要 16-bit PCM WAV，最长 10 分钟".into());
    }
    let pcm: Vec<_> = reader
        .samples::<i16>()
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let shared = Arc::new(Shared::new(spec.sample_rate, stop).map_err(|e| e.to_string())?);
    let mut writer = Writer::new(shared.spool.clone());
    let mut drain_packet = vec![0.0; spec.sample_rate as usize / 10];
    if ready.send(Ok((shared.clone(), spec.sample_rate))).is_err() {
        return Ok(());
    }
    let start = Instant::now();
    let frames = (spec.sample_rate / 100) as usize;
    for (i, packet) in pcm.chunks(frames * spec.channels as usize).enumerate() {
        if shared.halted() {
            break;
        }
        shared.enqueue(packet, spec.channels as usize, |s| s as f32 / 32768.0);
        drain(&shared, &mut writer, &mut drain_packet);
        if i % 10 == 0 {
            meter(
                &shared,
                spec.sample_rate,
                "测试音源（非麦克风）",
                updates,
                true,
            );
        }
        thread::sleep(
            Duration::from_secs_f64((i + 1) as f64 * frames as f64 / spec.sample_rate as f64)
                .saturating_sub(start.elapsed()),
        );
    }
    meter(
        &shared,
        spec.sample_rate,
        "测试音源（非麦克风）",
        updates,
        false,
    );
    shared.finished.store(true, Ordering::Release);
    shared.spool.notify_reader();
    Ok(())
}

fn drain(shared: &Shared, writer: &mut Writer, packet: &mut [f32]) {
    if shared.failure.load(Ordering::Acquire) == 5 {
        return;
    }
    loop {
        let n = shared.buffer.pop(packet);
        if n == 0 {
            break;
        }
        if let Err(error) = writer.push(&packet[..n]) {
            eprintln!("[microphone] temporary audio cache: {error}");
            shared.failure.store(5, Ordering::Release);
            break;
        }
    }
}
