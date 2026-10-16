//! Exercise Models through the production Jobs loop and real result storage.
use super::driver::{Product, Progress};
use super::*;
use crate::{
    jobs::{Jobs, View},
    results::{Results, tests::Directory},
};
use std::{
    sync::{
        Arc, Mutex,
        atomic::AtomicBool,
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};

#[derive(Debug, PartialEq, Eq)]
enum Call {
    Load(Kind),
    Admit(Kind),
    Step(Kind, Vec<u64>),
}
enum Reply {
    Load(Result<(), Failure>),
    Admit(Result<Option<u64>, Failure>),
    Step(Step),
    Panic,
}
struct Loaded {
    kind: Kind,
    ids: Vec<u64>,
    dropped: Arc<Mutex<Vec<Kind>>>,
}
impl Drop for Loaded {
    fn drop(&mut self) {
        self.dropped.lock().unwrap().push(self.kind);
    }
}
struct ScriptedDriver {
    calls: Sender<Call>,
    replies: Receiver<Reply>,
    dropped: Arc<Mutex<Vec<Kind>>>,
}
impl ScriptedDriver {
    fn reply(&self) -> Reply {
        self.replies
            .recv_timeout(Duration::from_secs(5))
            .expect("test driver reply")
    }
}
impl Driver for ScriptedDriver {
    type Loaded = Loaded;
    fn load(
        &mut self,
        kind: Kind,
        _: &AtomicBool,
        progress: &dyn Fn(Progress),
    ) -> Result<Loaded, Failure> {
        progress(Progress::Loading {
            progress: 0.25,
            message: "weights".into(),
        });
        self.calls.send(Call::Load(kind)).unwrap();
        let Reply::Load(result) = self.reply() else {
            panic!("expected load reply")
        };
        result?;
        Ok(Loaded {
            kind,
            ids: Vec::new(),
            dropped: self.dropped.clone(),
        })
    }
    fn admit(
        &mut self,
        engine: &mut Loaded,
        command: &Command,
        _: Arc<AtomicBool>,
        progress: &dyn Fn(Progress),
    ) -> Result<Option<u64>, Failure> {
        assert_eq!(command.kind(), engine.kind);
        progress(Progress::Loading {
            progress: 0.75,
            message: "preparing".into(),
        });
        self.calls.send(Call::Admit(engine.kind)).unwrap();
        let Reply::Admit(result) = self.reply() else {
            panic!("expected admission reply")
        };
        let id = result?;
        if let Some(id) = id {
            engine.ids.push(id);
        }
        Ok(id)
    }
    fn step(&mut self, engine: &mut Loaded) -> Step {
        self.calls
            .send(Call::Step(engine.kind, engine.ids.clone()))
            .unwrap();
        match self.reply() {
            Reply::Panic => panic!("scripted shared engine panic"),
            Reply::Step(step) => {
                engine
                    .ids
                    .retain(|id| !step.completed.iter().any(|(done, _)| done == id));
                step
            }
            _ => panic!("expected step reply"),
        }
    }
}
struct Harness {
    jobs: Jobs,
    calls: Receiver<Call>,
    replies: Option<Sender<Reply>>,
    dropped: Arc<Mutex<Vec<Kind>>>,
}
impl Harness {
    fn new(dir: &Directory) -> Self {
        let (calls, seen) = mpsc::channel();
        let (replies, receive) = mpsc::channel();
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let drops = dropped.clone();
        let jobs = Jobs::with_executor(Results::new(dir.0.clone()), 8, move || {
            Models::with_driver(ScriptedDriver {
                calls,
                replies: receive,
                dropped: drops,
            })
        });
        Self {
            jobs,
            calls: seen,
            replies: Some(replies),
            dropped,
        }
    }
    fn expect(&self, call: Call) {
        assert_eq!(
            self.calls.recv_timeout(Duration::from_secs(5)).unwrap(),
            call
        );
    }
    fn reply(&self, reply: Reply) {
        self.replies.as_ref().unwrap().send(reply).unwrap();
    }
    fn load(&self, kind: Kind) {
        self.expect(Call::Load(kind));
        self.reply(Reply::Load(Ok(())));
    }
    fn admit(&self, kind: Kind, id: u64) {
        self.expect(Call::Admit(kind));
        self.reply(Reply::Admit(Ok(Some(id))));
    }
    fn step(&self, kind: Kind, ids: &[u64]) {
        self.expect(Call::Step(kind, ids.to_vec()));
    }
    fn terminal(&self, id: &str) -> View {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let view = self.jobs.get(id).unwrap();
            if matches!(view.status, "done" | "cancelled" | "error") {
                return view;
            }
            assert!(Instant::now() < deadline, "job {id} stuck {}", view.status);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        self.jobs.close();
        self.replies.take(); // unblock the driver even when an assertion unwinds
        self.jobs.shutdown();
    }
}
fn speech() -> Command {
    Command::Speech(Speech {
        text: "test".into(),
        ..Default::default()
    })
}
fn asr() -> Command {
    Command::Transcribe {
        audio: Vec::new(),
        config: StreamConfig::default(),
    }
}
fn image() -> Command {
    Command::Image(Image {
        prompt: "test".into(),
        ..Default::default()
    })
}
fn speech_product(samples: usize, cancelled: bool) -> Product {
    Product::speech(
        omni_breeze_tts::SynthesisReport {
            frames: 1,
            codes: vec![[0; 16]],
            audio_seconds: samples as f64 / 24000.0,
            elapsed_seconds: 0.125,
            first_audio_seconds: Some(0.025),
            stopped: cancelled,
            reached_limit: !cancelled,
            profile: None,
        },
        Default::default(),
    )
}
fn speech_step(id: u64, audio: Vec<f32>, cancelled: bool) -> Step {
    Step {
        completed: vec![(id, Ok(speech_product(audio.len(), cancelled)))],
        updates: vec![(id, Update::Audio(audio))],
    }
}
fn done(id: u64, product: Product) -> Step {
    Step {
        completed: vec![(id, Ok(product))],
        ..Default::default()
    }
}
fn wav(h: &Harness, id: &str) -> Vec<i16> {
    let (bytes, mime) = h.jobs.read_file(&format!("{id}.wav")).unwrap();
    assert_eq!(mime, "audio/wav");
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
    assert_eq!(reader.spec().sample_rate, 24000);
    reader.samples::<i16>().map(Result::unwrap).collect()
}
fn transcription() -> Product {
    Product::transcription(
        omni_r2t2::stream::BatchReport {
            text: "hello".into(),
            cancelled: false,
            statistics: Default::default(),
        },
        Default::default(),
    )
}

#[test]
fn music_artifacts_publish_as_one_bundle_with_stereo_sample_rate() {
    let dir = Directory::new();
    let h = Harness::new(&dir);
    let request = omni_music::SongRequest {
        style: "piano".into(),
        ..Default::default()
    };
    let id = h
        .jobs
        .submit(Command::Music(omni_music::pipeline::Request::Generate(
            request.clone(),
        )))
        .unwrap()
        .id;
    h.load(Kind::Music);
    h.admit(Kind::Music, 1);
    h.step(Kind::Music, &[1]);
    for extension in ["json", "abc", "wav"] {
        assert!(h.jobs.read_file(&format!("{id}.{extension}")).is_err());
    }
    let product = omni_music::pipeline::Product::Song {
        audio: vec![0.25, -0.25, 0.5, -0.5],
        elapsed_seconds: 0.2,
        semantic: omni_music::yue::Semantic {
            plan: omni_music::yue::Plan {
                request,
                abc: Some("X:1\nK:C\nC |\n".into()),
                abc_tokens: vec![],
                prefix: vec![],
                truncated: false,
            },
            tokens: vec![1],
            truncated: true,
        },
    };
    h.reply(Reply::Step(done(1, Product::music(product))));
    let result = h.terminal(&id);
    assert_eq!(result.status, "done");
    let value = result.result.unwrap();
    assert_eq!(value["truncated"], true);
    assert_eq!(value["score_truncated"], false);
    assert_eq!(value["music_truncated"], true);
    let (bytes, mime) = h.jobs.read_file(&format!("{id}.wav")).unwrap();
    assert_eq!(mime, "audio/wav");
    let mut wav = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
    assert_eq!((wav.spec().channels, wav.spec().sample_rate), (2, 48000));
    assert_eq!(
        wav.samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        [0.25, -0.25, 0.5, -0.5]
    );
    let (json, _) = h.jobs.read_file(&format!("{id}.json")).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&json).unwrap(),
        value
    );
    assert_eq!(
        h.jobs.read_file(&format!("{id}.abc")).unwrap().0,
        b"X:1\nK:C\nC |\n"
    );
}

#[test]
fn resident_reuse_fifo_and_progress_are_visible_during_load_and_admission() {
    let dir = Directory::new();
    let h = Harness::new(&dir);
    let a = h.jobs.submit(speech()).unwrap().id;
    h.expect(Call::Load(Kind::Speech));
    assert_eq!(h.jobs.get(&a).unwrap().progress["progress"], 0.25);
    h.reply(Reply::Load(Ok(())));
    h.expect(Call::Admit(Kind::Speech));
    assert_eq!(h.jobs.get(&a).unwrap().progress["progress"], 0.75);
    h.reply(Reply::Admit(Ok(Some(1))));
    h.step(Kind::Speech, &[1]);
    let b = h.jobs.submit(asr()).unwrap().id;
    let c = h.jobs.submit(speech()).unwrap().id;
    h.reply(Reply::Step(Step::default()));
    h.step(Kind::Speech, &[1]);
    assert_eq!(h.jobs.get(&b).unwrap().status, "queued");
    assert_eq!(h.jobs.get(&c).unwrap().status, "queued");
    h.reply(Reply::Step(speech_step(1, vec![0.25], false)));
    h.load(Kind::Transcription);
    h.admit(Kind::Transcription, 1);
    h.step(Kind::Transcription, &[1]);
    h.reply(Reply::Step(done(1, transcription())));
    h.admit(Kind::Speech, 2);
    h.step(Kind::Speech, &[2]); // no second load
    h.reply(Reply::Step(speech_step(2, vec![-0.25], false)));
    assert_eq!(h.terminal(&a).status, "done");
    assert_eq!(h.terminal(&b).result.unwrap()["text"], "hello");
    assert_eq!(h.terminal(&c).status, "done");
    assert_eq!(wav(&h, &a), vec![8192]);
    assert_eq!(wav(&h, &c), vec![-8192]);
    assert!(h.dropped.lock().unwrap().is_empty());
    h.jobs.shutdown();
    assert_eq!(h.dropped.lock().unwrap().len(), 2);
}

#[test]
fn full_admission_cancelled_partial_audio_and_out_of_order_completion_keep_owners() {
    let dir = Directory::new();
    let h = Harness::new(&dir);
    let a = h.jobs.submit(speech()).unwrap().id;
    h.load(Kind::Speech);
    h.admit(Kind::Speech, 11);
    h.step(Kind::Speech, &[11]);
    let b = h.jobs.submit(speech()).unwrap().id;
    h.reply(Reply::Step(Step {
        updates: vec![(11, Update::Audio(vec![0.1]))],
        ..Default::default()
    }));
    h.expect(Call::Admit(Kind::Speech));
    h.reply(Reply::Admit(Ok(None)));
    h.step(Kind::Speech, &[11]);
    assert_eq!(h.jobs.get(&b).unwrap().status, "queued");
    h.reply(Reply::Step(Step::default()));
    h.admit(Kind::Speech, 12);
    h.step(Kind::Speech, &[11, 12]);
    assert!(h.jobs.read_file(&format!("{a}.wav")).is_err());
    h.jobs.cancel(&a);
    h.reply(Reply::Step(Step {
        updates: vec![
            (12, Update::Audio(vec![-0.5, -0.25])),
            (11, Update::Audio(vec![0.2])),
        ],
        completed: vec![
            (12, Ok(speech_product(2, false))),
            (11, Ok(speech_product(2, true))),
        ],
    }));
    assert_eq!(h.terminal(&a).status, "cancelled");
    assert_eq!(h.terminal(&b).status, "done");
    assert_eq!(wav(&h, &a), vec![3277, 6553]);
    assert_eq!(wav(&h, &b), vec![-16384, -8192]);
    assert_eq!(
        h.jobs.get(&a).unwrap().progress["audio_seconds"],
        2.0 / 24000.0
    );
}

#[test]
fn load_and_admission_errors_leave_no_ghost_request_and_allow_retry() {
    let dir = Directory::new();
    let h = Harness::new(&dir);
    for failure in [Failure::Cancelled, Failure::Error("bad weights".into())] {
        let id = h.jobs.submit(speech()).unwrap().id;
        h.expect(Call::Load(Kind::Speech));
        h.reply(Reply::Load(Err(failure)));
        assert!(matches!(h.terminal(&id).status, "cancelled" | "error"));
    }
    let failed = h.jobs.submit(speech()).unwrap().id;
    h.load(Kind::Speech);
    h.expect(Call::Admit(Kind::Speech));
    h.reply(Reply::Admit(Err(Failure::Error("bad input".into()))));
    assert_eq!(h.terminal(&failed).status, "error");
    let retry = h.jobs.submit(asr()).unwrap().id;
    h.load(Kind::Transcription);
    h.admit(Kind::Transcription, 1);
    h.step(Kind::Transcription, &[1]);
    h.reply(Reply::Step(done(1, transcription())));
    assert_eq!(h.terminal(&retry).status, "done");
}

#[test]
fn duplicate_unknown_and_wrong_modality_events_fail_the_whole_batch_then_reload() {
    for scenario in 0..5 {
        let dir = Directory::new();
        let h = Harness::new(&dir);
        let a = h.jobs.submit(speech()).unwrap().id;
        h.load(Kind::Speech);
        h.admit(Kind::Speech, 1);
        h.step(Kind::Speech, &[1]);
        let b = h.jobs.submit(speech()).unwrap().id;
        h.reply(Reply::Step(Step::default()));
        h.admit(Kind::Speech, if scenario == 4 { 1 } else { 2 });
        if scenario != 4 {
            h.step(Kind::Speech, &[1, 2]);
            let step = match scenario {
                0 => Step {
                    completed: vec![
                        (1, Ok(speech_product(0, false))),
                        (1, Ok(speech_product(0, false))),
                    ],
                    ..Default::default()
                },
                1 => done(99, speech_product(0, false)),
                2 => done(1, transcription()),
                3 => Step {
                    updates: vec![(99, Update::Audio(vec![1.0]))],
                    ..Default::default()
                },
                _ => unreachable!(),
            };
            h.reply(Reply::Step(step));
        }
        // A failed admission must release its engine before its own terminal.
        assert_eq!(h.terminal(&b).status, "error");
        assert_eq!(*h.dropped.lock().unwrap(), vec![Kind::Speech]);
        for id in [&a, &b] {
            let view = h.terminal(id);
            assert_eq!(view.status, "error");
            assert!(view.error.unwrap().contains("protocol error"));
            assert!(h.jobs.read_file(&format!("{id}.wav")).is_err());
        }
        assert_eq!(*h.dropped.lock().unwrap(), vec![Kind::Speech]);
        let retry = h.jobs.submit(speech()).unwrap().id;
        h.load(Kind::Speech);
        h.admit(Kind::Speech, 1);
        h.step(Kind::Speech, &[1]);
        h.reply(Reply::Step(speech_step(1, vec![0.1], false)));
        assert_eq!(h.terminal(&retry).status, "done");
    }
}

#[test]
fn panic_unloads_resident_models_and_publication_failure_can_retry() {
    let dir = Directory::new();
    let h = Harness::new(&dir);
    let a = h.jobs.submit(asr()).unwrap().id;
    h.load(Kind::Transcription);
    h.admit(Kind::Transcription, 1);
    h.step(Kind::Transcription, &[1]);
    h.reply(Reply::Step(done(1, transcription())));
    assert_eq!(h.terminal(&a).status, "done");
    let b = h.jobs.submit(speech()).unwrap().id;
    h.load(Kind::Speech);
    h.admit(Kind::Speech, 1);
    h.step(Kind::Speech, &[1]);
    h.reply(Reply::Panic);
    assert_eq!(h.terminal(&b).status, "error");
    assert_eq!(h.dropped.lock().unwrap().len(), 2);
    let c = h.jobs.submit(speech()).unwrap().id;
    h.load(Kind::Speech);
    h.admit(Kind::Speech, 1);
    h.step(Kind::Speech, &[1]);
    let collision = dir.0.join("published").join(&c);
    std::fs::create_dir_all(&collision).unwrap();
    std::fs::write(collision.join("occupied"), b"keep").unwrap();
    h.reply(Reply::Step(speech_step(1, vec![0.25], false)));
    assert_eq!(h.terminal(&c).status, "error");
    assert!(!dir.0.join(".staging").join(&c).exists());
    let retry = h.jobs.submit(speech()).unwrap().id;
    h.admit(Kind::Speech, 2);
    h.step(Kind::Speech, &[2]);
    h.reply(Reply::Step(speech_step(2, vec![0.5], false)));
    assert_eq!(h.terminal(&retry).status, "done");
    assert_eq!(wav(&h, &retry), vec![16384]);
}

fn image_report() -> omni_qwen_image::Report {
    omni_qwen_image::Report {
        width: 1,
        height: 1,
        seed: 42,
        steps: 8,
        reference_count: 0,
        reference_resolution: 0,
        guidance_enabled: false,
        true_cfg_scale: 1.0,
        reference_seconds: 0.0,
        text_seconds: 0.0,
        denoise_seconds: 0.0,
        decode_seconds: 0.0,
        elapsed_seconds: 0.0,
    }
}
#[test]
fn image_encoding_uses_staging_and_failed_encoding_cleans_up_before_retry() {
    // A complete 1x1 RGBA PNG, produced independently of the worker encoder.
    const PNG: &[u8] = include_bytes!("test_pixel.png");
    let dir = Directory::new();
    let h = Harness::new(&dir);
    for (index, fail) in [false, true, false].into_iter().enumerate() {
        let job = h.jobs.submit(image()).unwrap().id;
        if index == 0 {
            h.load(Kind::Image);
        }
        let id = index as u64 + 1;
        h.admit(Kind::Image, id);
        h.step(Kind::Image, &[id]);
        let files = Results::new(dir.0.clone());
        let name = format!("{job}.png");
        h.reply(Reply::Step(done(
            id,
            Product::image(
                Box::new(move |path| {
                    assert!(files.read(&name).is_err());
                    std::fs::write(path, PNG)?;
                    assert!(files.read(&name).is_err());
                    if fail {
                        std::fs::write(path.with_extension("png.part"), b"partial")?;
                        return Err(Failure::Error("image encode failed".into()));
                    }
                    Ok(image_report())
                }),
                Default::default(),
            ),
        )));
        let view = h.terminal(&job);
        if fail {
            assert_eq!(view.status, "error");
            assert!(h.jobs.read_file(&format!("{job}.png")).is_err());
        } else {
            assert_eq!(view.status, "done");
            assert_eq!(view.result.unwrap()["report"]["width"], 1);
            let (bytes, mime) = h.jobs.read_file(&format!("{job}.png")).unwrap();
            assert_eq!(mime, "image/png");
            assert_eq!(bytes, PNG);
        }
        assert!(!dir.0.join(".staging").join(&job).exists());
    }
}
#[test]
fn translation_preserves_partial_text_progress_cache_and_batch_metadata() {
    let dir = Directory::new();
    let h = Harness::new(&dir);
    let job = h
        .jobs
        .submit(Command::Translate(omni_hy_mt::Request {
            text: "hello".into(),
            target_language: "Chinese".into(),
            options: Default::default(),
        }))
        .unwrap()
        .id;
    h.load(Kind::Translation);
    h.admit(Kind::Translation, 9);
    h.step(Kind::Translation, &[9]);
    h.jobs.cancel(&job);
    h.reply(Reply::Step(Step {
        updates: vec![(
            9,
            Update::Progress(Progress::Translation(omni_hy_mt::TranslationEvent {
                delta: "你好".into(),
                text: "你好".into(),
                generated_tokens: 2,
            })),
        )],
        completed: vec![(
            9,
            Ok(Product::translation(
                omni_hy_mt::Translation {
                    text: "你好".into(),
                    target_language: "Chinese".into(),
                    prompt_tokens: 12,
                    generated_tokens: 2,
                    cached_tokens: 8,
                    prefill_ms: 1.0,
                    compute_ms: 2.0,
                    finish_reason: "cancelled".into(),
                    cache: Default::default(),
                },
                omni_hy_mt::BatchStats {
                    max_batch_size: 3,
                    ..Default::default()
                },
            )),
        )],
    }));
    let view = h.terminal(&job);
    assert_eq!(view.status, "cancelled");
    assert_eq!(view.progress["text"], "你好");
    let output = view.result.unwrap();
    assert_eq!(output["text"], "你好");
    assert_eq!(output["cached_tokens"], 8);
    assert_eq!(output["batch"]["max_batch_size"], 3);
}
