//! The sole execution seam: production engines and scripted tests both return
//! owned events. Job ownership and result publication stay in Models / Jobs.
use super::{Backend, Command, Failure, Kind, ModelPaths};
use std::{
    io::Cursor,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

pub(crate) enum Progress {
    Loading { progress: f64, message: String },
    Tensors { current: usize, total: usize },
    Translation(omni_hy_mt::TranslationEvent),
    Transcription(omni_r2t2::TranscriptionEvent),
    Image(omni_qwen_image::Progress),
    Music(omni_music::Progress),
}

pub(crate) enum Update {
    Progress(Progress),
    Audio(Vec<f32>),
}
impl Update {
    pub fn kind(&self) -> Option<Kind> {
        Some(match self {
            Self::Audio(_) => Kind::Speech,
            Self::Progress(Progress::Translation(_)) => Kind::Translation,
            Self::Progress(Progress::Transcription(_)) => Kind::Transcription,
            Self::Progress(Progress::Image(_)) => Kind::Image,
            Self::Progress(Progress::Music(_)) => Kind::Music,
            // Loading progress is emitted synchronously, never by a step.
            Self::Progress(Progress::Loading { .. } | Progress::Tensors { .. }) => return None,
        })
    }
}
// Encoding consumes the image only after its engine borrow has ended. Returning
// the report keeps GeneratedImage's source-path protection inside save_png.
pub(crate) type Png = Box<dyn FnOnce(&Path) -> Result<omni_qwen_image::Report, Failure> + Send>;
pub(crate) enum Product {
    Music(omni_music::pipeline::Product),
    Translation {
        report: omni_hy_mt::Translation,
        batch: omni_hy_mt::BatchStats,
    },
    Transcription {
        report: omni_r2t2::stream::BatchReport,
        batch: omni_r2t2::stream::BatchStats,
    },
    Speech {
        report: omni_breeze_tts::SynthesisReport,
        batch: omni_breeze_tts::BatchStats,
    },
    Image {
        encode: Png,
        batch: omni_qwen_image::BatchStats,
    },
}
impl Product {
    pub fn kind(&self) -> Kind {
        match self {
            Self::Translation { .. } => Kind::Translation,
            Self::Transcription { .. } => Kind::Transcription,
            Self::Speech { .. } => Kind::Speech,
            Self::Image { .. } => Kind::Image,
            Self::Music(_) => Kind::Music,
        }
    }
}
#[derive(Default)]
pub(crate) struct Step {
    pub updates: Vec<(u64, Update)>,
    pub completed: Vec<(u64, Result<Product, Failure>)>,
}
pub(crate) trait Driver: 'static {
    type Loaded;
    fn load(
        &mut self,
        kind: Kind,
        cancel: &AtomicBool,
        progress: &dyn Fn(Progress),
    ) -> Result<Self::Loaded, Failure>;
    fn admit(
        &mut self,
        engine: &mut Self::Loaded,
        command: &Command,
        cancel: Arc<AtomicBool>,
        progress: &dyn Fn(Progress),
    ) -> Result<Option<u64>, Failure>;
    fn step(&mut self, engine: &mut Self::Loaded) -> Step;
}
pub(crate) struct NativeDriver {
    paths: ModelPaths,
}
pub(crate) enum NativeLoaded {
    Music(Box<super::music::Music>),
    Translation(Box<omni_hy_mt::Translator<Backend>>),
    Transcription(Box<omni_r2t2::stream::TranscriptionBatch<Backend>>),
    Speech(Box<omni_breeze_tts::SynthesisBatch<Backend>>),
    Image(Box<omni_qwen_image::ImageBatch<Backend>>),
}
impl NativeDriver {
    pub fn new(paths: ModelPaths) -> Self {
        Self { paths }
    }
}
impl Driver for NativeDriver {
    type Loaded = NativeLoaded;
    fn load(
        &mut self,
        kind: Kind,
        cancel: &AtomicBool,
        progress: &dyn Fn(Progress),
    ) -> Result<NativeLoaded, Failure> {
        let paths = &self.paths;
        let device = Default::default();
        Ok(match kind {
            Kind::Music => NativeLoaded::Music(Box::new(super::music::Music::new(paths.clone()))),
            Kind::Translation => {
                let model = omni_hy_mt::Model::from_directory(
                    &paths.translation,
                    paths.precision,
                    &device,
                    cancel,
                    |current, total| progress(Progress::Tensors { current, total }),
                )?;
                NativeLoaded::Translation(Box::new(omni_hy_mt::Translator::with_config(
                    model,
                    paths.cache_tokens,
                    paths.translation_batch,
                )?))
            }
            Kind::Transcription => {
                let model = omni_r2t2::Model::from_directory_cancellable(
                    &paths.asr,
                    paths.precision,
                    &device,
                    cancel,
                    |value, message| {
                        progress(Progress::Loading {
                            progress: value,
                            message: message.into(),
                        })
                    },
                )?;
                NativeLoaded::Transcription(Box::new(omni_r2t2::stream::TranscriptionBatch::new(
                    Arc::new(model),
                    paths.asr_batch_size,
                )?))
            }
            Kind::Speech => {
                let model = omni_breeze_tts::Model::load_cancellable(
                    &paths.tts,
                    paths.precision,
                    &device,
                    cancel,
                    |value, message| {
                        progress(Progress::Loading {
                            progress: value,
                            message: message.into(),
                        })
                    },
                )?;
                NativeLoaded::Speech(Box::new(omni_breeze_tts::SynthesisBatch::new(
                    model,
                    paths.tts_batch_size,
                )?))
            }
            Kind::Image => {
                let model = omni_qwen_image::Model::load(
                    &paths.image,
                    paths.precision,
                    &device,
                    cancel,
                    |event| progress(Progress::Image(event)),
                )?;
                NativeLoaded::Image(Box::new(omni_qwen_image::ImageBatch::new(
                    model,
                    paths.image_batch_size,
                )?))
            }
        })
    }
    fn admit(
        &mut self,
        engine: &mut NativeLoaded,
        command: &Command,
        cancel: Arc<AtomicBool>,
        progress: &dyn Fn(Progress),
    ) -> Result<Option<u64>, Failure> {
        Ok(match (engine, command) {
            (NativeLoaded::Music(engine), Command::Music(request)) => {
                engine.admit(request, cancel)?
            }
            (NativeLoaded::Translation(engine), Command::Translate(request)) => {
                let id = engine.try_submit(request, cancel)?;
                if id.is_some() {
                    progress(Progress::Translation(omni_hy_mt::TranslationEvent {
                        delta: String::new(),
                        text: String::new(),
                        generated_tokens: 0,
                    }));
                }
                id
            }
            (NativeLoaded::Transcription(engine), Command::Transcribe { audio, config }) => {
                let source = omni_r2t2::input::WavSource::new(Cursor::new(audio.clone()))?;
                let mut config = config.clone();
                config.paged_attention = self.paths.asr_paged_attention;
                engine.try_submit(config, source, cancel)?
            }
            (NativeLoaded::Speech(engine), Command::Speech(request)) => {
                let mut options = omni_breeze_tts::SynthesisOptions {
                    text: request.text.clone(),
                    instruction: request.instruction.clone(),
                    max_frames: request.max_frames,
                    guidance_scale: request.guidance_scale,
                    paged_attention: self.paths.tts_paged_attention,
                    ..Default::default()
                };
                options.sampling.seed = request.seed;
                engine.try_submit(&options, cancel)?
            }
            (NativeLoaded::Image(engine), Command::Image(request)) => {
                engine.try_submit(&request.options(), cancel, |event| {
                    progress(Progress::Image(event))
                })?
            }
            _ => {
                return Err(Failure::Error(
                    "model driver protocol error: wrong resident model".into(),
                ));
            }
        })
    }
    fn step(&mut self, engine: &mut NativeLoaded) -> Step {
        let mut step = Step::default();
        match engine {
            NativeLoaded::Music(engine) => return engine.step(),
            NativeLoaded::Translation(engine) => {
                let results = pollster::block_on(engine.step(|id, event| {
                    step.updates
                        .push((id, Update::Progress(Progress::Translation(event))));
                }));
                let batch = engine.batch_stats();
                step.completed = results
                    .into_iter()
                    .map(|(id, result)| {
                        (
                            id,
                            result
                                .map(|report| Product::Translation { report, batch })
                                .map_err(Failure::from),
                        )
                    })
                    .collect();
            }
            NativeLoaded::Transcription(engine) => {
                let results = engine.step(|id, event| {
                    step.updates
                        .push((id, Update::Progress(Progress::Transcription(event))));
                    Ok(())
                });
                let batch = engine.stats();
                step.completed = results
                    .into_iter()
                    .map(|(id, result)| {
                        (
                            id,
                            result
                                .map(|report| Product::Transcription { report, batch })
                                .map_err(Failure::from),
                        )
                    })
                    .collect();
            }
            NativeLoaded::Speech(engine) => {
                let results = engine.step(|id, audio| {
                    step.updates.push((id, Update::Audio(audio.to_vec())));
                    Ok(())
                });
                let batch = engine.stats();
                step.completed = results
                    .into_iter()
                    .map(|(id, result)| {
                        (
                            id,
                            result
                                .map(|report| Product::Speech { report, batch })
                                .map_err(Failure::from),
                        )
                    })
                    .collect();
            }
            NativeLoaded::Image(engine) => {
                let results = engine.step(|id, event| {
                    step.updates
                        .push((id, Update::Progress(Progress::Image(event))))
                });
                let batch = engine.stats();
                step.completed = results
                    .into_iter()
                    .map(|(id, result)| {
                        (
                            id,
                            result
                                .map(|image| Product::Image {
                                    encode: Box::new(move |path| {
                                        image.save_png(path)?;
                                        Ok(image.report)
                                    }),
                                    batch,
                                })
                                .map_err(Failure::from),
                        )
                    })
                    .collect();
            }
        }
        step
    }
}
