//! Real-model serial vs continuous speech batches, including join/cancel/retry.
use omni_breeze_tts::{Model, Precision, Sampling, SynthesisBatch, SynthesisOptions};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::wgpu::{Metal, graphics, init_setup};
    let device = Default::default();
    init_setup::<graphics::Metal>(&device, Default::default());
    let mut model = Model::<Metal>::load(
        std::path::Path::new("models/Breeze-TTS-2"),
        if std::env::args().any(|a| a == "--fp32") {
            Precision::F32
        } else {
            Precision::F16
        },
        &device,
        |_, _| {},
    )?;
    if std::env::args().any(|a| a == "--stock") {
        model.set_depth_kernel_mode(omni_breeze_tts::DepthKernelMode::Stock);
    }
    let cases: Vec<_> = [
        ("你好，欢迎使用本地语音合成。", ""),
        (
            "请带上更新后的项目报告，明天上午九点开会。",
            "用温和自然的语气朗读。",
        ),
        ("今天天气很好，我们一起去公园散步吧。", ""),
    ]
    .into_iter()
    .map(|(text, instruction)| SynthesisOptions {
        text: text.into(),
        instruction: instruction.into(),
        paged_attention: std::env::args().any(|a| a == "--paged"),
        max_frames: if std::env::args().any(|a| a == "--short") {
            2
        } else if std::env::args().any(|a| a == "--long") {
            48
        } else {
            12
        },
        cpu_depth_sampling: std::env::args().any(|a| a == "--cpu-depth"),
        sampling: Sampling {
            temperature: if std::env::args().any(|a| a == "--sampled") {
                0.9
            } else {
                0.
            },
            top_p: if std::env::args().any(|a| a == "--nucleus") {
                0.9
            } else {
                1.0
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .collect();
    let mut engine = SynthesisBatch::new(model, 4)?;
    for width in 1..=3 {
        for c in &cases[..width] {
            let mut warm = c.clone();
            warm.max_frames = if std::env::args().any(|a| a == "--warm-full") {
                c.max_frames
            } else {
                2
            };
            engine
                .try_submit(&warm, Arc::new(AtomicBool::new(false)))?
                .unwrap();
        }
        while !engine.is_idle() {
            for (_, result) in engine.step(|_, _| Ok(())) {
                result?;
            }
        }
    }
    let mut serial = Vec::new();
    let serial_start = Instant::now();
    for c in &cases {
        let mut audio = Vec::new();
        let report = engine
            .model()
            .synthesize(c, &AtomicBool::new(false), |samples| {
                audio.extend_from_slice(samples);
                Ok(())
            })?;
        serial.push((report, audio));
    }
    let serial_seconds = serial_start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut ids = Vec::new();
    let mut audio: BTreeMap<u64, Vec<f32>> = BTreeMap::new();
    let mut reports = BTreeMap::new();
    for c in &cases {
        ids.push(
            engine
                .try_submit(c, Arc::new(AtomicBool::new(false)))?
                .unwrap(),
        );
    }
    while !engine.is_idle() {
        for (id, result) in engine.step(|id, samples| {
            audio.entry(id).or_default().extend_from_slice(samples);
            Ok(())
        }) {
            reports.insert(id, result?);
        }
    }
    let batch_seconds = start.elapsed().as_secs_f64();
    let mut errors = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let report = &reports[id];
        assert_eq!(report.codes, serial[i].0.codes, "codes {i}");
        let actual = &audio[id];
        let expected = &serial[i].1;
        assert_eq!(actual.len(), expected.len());
        let rms = (actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).powi(2) as f64)
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        assert!(rms < 0.005, "PCM RMS {rms}");
        errors.push(rms);
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let cancelled = engine.try_submit(&cases[0], cancel.clone())?.unwrap();
    engine.step(|_, _| Ok(()));
    let survivor = engine
        .try_submit(&cases[1], Arc::new(AtomicBool::new(false)))?
        .unwrap();
    cancel.store(true, Ordering::Release);
    let mut finished = BTreeMap::new();
    while !engine.is_idle() {
        for (id, result) in engine.step(|_, _| Ok(())) {
            assert!(finished.insert(id, result?).is_none());
        }
    }
    assert!(finished[&cancelled].stopped);
    assert_eq!(finished[&survivor].codes, serial[1].0.codes);
    let retry = engine
        .try_submit(&cases[0], Arc::new(AtomicBool::new(false)))?
        .unwrap();
    let mut retried = None;
    while !engine.is_idle() {
        for (id, result) in engine.step(|_, _| Ok(())) {
            assert_eq!(id, retry);
            retried = Some(result?);
        }
    }
    assert_eq!(retried.unwrap().codes, serial[0].0.codes);
    let mut short = cases[0].clone();
    short.max_frames = 2;
    short.chunk_frames = 1;
    let failed = engine
        .try_submit(&short, Arc::new(AtomicBool::new(false)))?
        .unwrap();
    let surviving = engine
        .try_submit(&short, Arc::new(AtomicBool::new(false)))?
        .unwrap();
    let mut outcomes = BTreeMap::new();
    while !engine.is_idle() {
        for (id, result) in engine.step(|id, _| {
            if id == failed {
                Err(omni_r2t2::Error::Audio("sink failed".into()))
            } else {
                Ok(())
            }
        }) {
            assert!(outcomes.insert(id, result).is_none());
        }
    }
    assert!(outcomes[&failed].is_err());
    assert_eq!(outcomes[&surviving].as_ref().unwrap().frames, 2);
    if std::env::args().any(|a| a == "--lifecycle") {
        check_lifecycle(&mut engine)?;
    }
    println!(
        "{}",
        serde_json::json!({"passed":true,"serial_seconds":serial_seconds,"batch_seconds":batch_seconds,
        "speedup":serial_seconds/batch_seconds,"pcm_rms_errors":errors,"batch":engine.stats(),"reports":reports,
        "join_cancel_retry":true,"lifecycle":std::env::args().any(|a| a == "--lifecycle"),
        "serial_profiles":serial.iter().map(|(r,_)| &r.profile).collect::<Vec<_>>() })
    );
    Ok(())
}

/// Full-weight checks use the production public interfaces; callbacks exercise
/// cancellation in both a complete chunk and a final one-frame tail.
fn check_lifecycle<B: burn::tensor::backend::Backend>(
    engine: &mut SynthesisBatch<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = SynthesisOptions {
        text: "你好，欢迎使用本地语音合成。".into(),
        max_frames: 3,
        chunk_frames: 2,
        sampling: Sampling {
            temperature: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let stopped = engine
        .model()
        .synthesize(&options, &AtomicBool::new(true), |_| {
            panic!("pre-cancel emitted audio")
        })?;
    assert!(stopped.stopped && stopped.codes.is_empty() && stopped.first_audio_seconds.is_none());
    assert!(matches!(
        engine.try_submit(&options, Arc::new(AtomicBool::new(true))),
        Err(omni_r2t2::Error::Interrupted)
    ));
    let mut invalid = options.clone();
    invalid.max_frames = 0;
    assert!(
        engine
            .model()
            .synthesize(&invalid, &AtomicBool::new(false), |_| Ok(()))
            .is_err()
    );
    assert!(
        engine
            .try_submit(&invalid, Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    for cancel_after in [None, Some(1), Some(2)] {
        let stop = AtomicBool::new(false);
        let mut serial_audio = Vec::new();
        let mut serial_chunks = Vec::new();
        let serial_start = Instant::now();
        let mut serial_handoff = 0.0;
        let serial = engine.model().synthesize(&options, &stop, |pcm| {
            serial_chunks.push(pcm.len());
            serial_audio.extend_from_slice(pcm);
            if serial_chunks.len() == 1 {
                serial_handoff = serial_start.elapsed().as_secs_f64();
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            if cancel_after == Some(serial_chunks.len()) {
                stop.store(true, Ordering::Release);
            }
            Ok(())
        })?;
        assert!(serial.first_audio_seconds.unwrap() <= serial_handoff);
        assert_eq!(serial.stopped, cancel_after.is_some());
        assert_eq!(serial.reached_limit, cancel_after.is_none());
        assert_eq!(serial.audio_seconds, serial_audio.len() as f64 / 24000.0);
        assert_eq!(
            serial_chunks.len(),
            if cancel_after == Some(1) { 1 } else { 2 }
        );
        if let Some(profile) = &serial.profile {
            for name in ["prepare", "prefill", "tail_and_codes_readback"] {
                assert_eq!(profile.stage_seconds[name].len(), 1, "{name}");
            }
            for name in [
                "semantic_sample",
                "depth",
                "codes_device",
                "codec_and_callback",
            ] {
                assert_eq!(profile.stage_seconds[name].len(), serial.frames, "{name}");
            }
            let continued = serial.frames - usize::from(serial.reached_limit);
            for name in ["audio_embedding", "backbone"] {
                assert_eq!(profile.stage_seconds[name].len(), continued, "{name}");
            }
        }
        let stop = Arc::new(AtomicBool::new(false));
        let start = Instant::now();
        let id = engine.try_submit(&options, stop.clone())?.unwrap();
        let mut audio = Vec::new();
        let mut chunks = Vec::new();
        let mut report = None;
        let mut handoff = 0.0;
        while !engine.is_idle() {
            for (completed, result) in engine.step(|request, pcm| {
                assert_eq!(request, id);
                chunks.push(pcm.len());
                audio.extend_from_slice(pcm);
                if chunks.len() == 1 {
                    handoff = start.elapsed().as_secs_f64();
                    std::thread::sleep(std::time::Duration::from_millis(30));
                }
                if cancel_after == Some(chunks.len()) {
                    stop.store(true, Ordering::Release);
                }
                Ok(())
            }) {
                assert_eq!(completed, id);
                assert!(report.replace(result?).is_none());
            }
        }
        let report = report.unwrap();
        assert!(report.first_audio_seconds.unwrap() <= handoff);
        assert_eq!(report.stopped, serial.stopped);
        assert_eq!(report.reached_limit, serial.reached_limit);
        assert_eq!(report.codes, serial.codes);
        assert_eq!(chunks, serial_chunks);
        assert_eq!(audio, serial_audio);
        assert_eq!(report.audio_seconds, serial.audio_seconds);
        assert!(engine.step(|_, _| panic!("late output")).is_empty());
        // Late cancellation cannot change an already returned completion.
        stop.store(true, Ordering::Release);
        assert_eq!(report.stopped, cancel_after.is_some());
    }
    assert!(
        engine
            .model()
            .synthesize(&options, &AtomicBool::new(false), |_| Err(
                omni_r2t2::Error::Audio("serial sink".into())
            ))
            .is_err()
    );
    Ok(())
}
