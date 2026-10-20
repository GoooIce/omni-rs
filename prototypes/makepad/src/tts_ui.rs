use crate::ActionButtonState;
use crate::{
    model_location,
    tts_engine::{Command, Engine, ReferenceInput},
    App,
};
use makepad_widgets::*;
use std::path::{Path, PathBuf};

enum ReferenceSelection {
    File(PathBuf),
    Designed,
}

enum Pending {
    Idle,
    Sampling {
        reference: Option<crate::tts_engine::ReferenceInput>,
        design: bool,
        speech_seed: u64,
    },
    Holding {
        next: usize,
        pairs: [omni_breeze_tts::enhancement::Pair; 3],
        reference: Option<crate::tts_engine::ReferenceInput>,
        design: bool,
        speech_seed: u64,
    },
}

pub struct State {
    engine: Engine,
    dialog: Option<crate::desktop_runtime::Dialog>,
    needs_directory: bool,
    output: Option<std::sync::Arc<[f32]>>,
    saving_snapshot: Option<crate::export::Snapshot>,
    exporting: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    path: PathBuf,
    settings: Option<PathBuf>,
    initialized: bool,
    ready: bool,
    busy: bool,
    playing: bool,
    loading: bool,
    stopping: bool,
    picking: bool,
    saving: bool,
    remember: bool,
    has_audio: bool,
    reference: Option<ReferenceSelection>,
    picking_reference: bool,
    design: bool,
    reusable: bool,
    candidate: u64,
    controls: Option<[bool; 17]>,
    session: omni_breeze_tts::enhancement::Session,
    checkpoint: PathBuf,
    enhancement_settings: Option<PathBuf>,
    pending: Pending,
    hearing_audio: [Option<std::sync::Arc<[f32]>>; 3],
    hearing_play: Option<usize>,
}
impl State {
    pub fn playing(&self) -> bool {
        self.playing
    }
    pub fn stop_playback(&mut self) {
        if self.playing {
            self.stopping = true;
            self.engine.cancel();
        }
    }
    pub fn needs_model(&self) -> bool {
        !self.ready && !self.loading
    }

    pub fn new(root: &Path, runtime: crate::desktop_runtime::Runtime) -> Self {
        let settings =
            model_location::settings_path_for("BREEZE_TTS_SETTINGS_PATH", "breeze-tts.json");
        let enhancement_settings = settings
            .as_ref()
            .map(|path| path.with_file_name("breeze-tts-enhancement.json"));
        let session = enhancement_settings
            .as_ref()
            .and_then(|path| std::fs::read(path).ok())
            .map(|bytes| omni_breeze_tts::enhancement::Session::load(&bytes))
            .unwrap_or_default();
        let path = model_location::initial_directory_for(
            root,
            settings.as_deref(),
            "BREEZE_TTS_MODEL_DIR",
            "models/Breeze-TTS-2",
        );
        Self {
            engine: Engine::spawn_shared(SignalToUI::set_ui_signal, runtime),
            dialog: None,
            needs_directory: false,
            output: None,
            saving_snapshot: None,
            exporting: None,
            path,
            settings,
            initialized: false,
            ready: false,
            busy: false,
            playing: false,
            loading: false,
            stopping: false,
            picking: false,
            saving: false,
            remember: false,
            has_audio: false,
            reference: None,
            picking_reference: false,
            design: false,
            reusable: false,
            candidate: 1,
            controls: None,
            session,
            checkpoint: root.join(omni_breeze_tts::enhancement::DEFAULT_CHECKPOINT),
            enhancement_settings,
            pending: Pending::Idle,
            hearing_audio: [None, None, None],
            hearing_play: None,
        }
    }
}
impl App {
    pub fn activate_tts(&mut self, cx: &mut Cx) {
        if self.inference_busy() || self.runtime.dialog_busy() {
            return;
        }
        if self
            .tts
            .as_ref()
            .is_some_and(|t| t.needs_directory && !t.busy)
        {
            self.choose_tts(cx);
            return;
        }
        if self.tts.as_ref().is_some_and(|t| !t.ready && !t.busy) {
            self.load_tts(cx);
        }
    }
    fn set_tts_design_mode(&mut self, cx: &mut Cx, design: bool) {
        let Some(t) = self.tts.as_mut() else {
            return;
        };
        t.design = design;
        t.controls = None;
        self.ui
            .view(cx, ids!(tts_reference_group))
            .set_visible(cx, !design);
        self.ui
            .view(cx, ids!(tts_design_presets))
            .set_visible(cx, design);
        self.ui
            .view(cx, ids!(tts_design_results))
            .set_visible(cx, design);
        self.ui.button(cx, ids!(tts_design_mode)).set_text(
            cx,
            if design {
                "● 音色设计"
            } else {
                "音色设计"
            },
        );
        self.ui.button(cx, ids!(tts_speech_mode)).set_text(
            cx,
            if design {
                "朗读与克隆"
            } else {
                "● 朗读与克隆"
            },
        );
        self.ui.label(cx, ids!(tts_text_title)).set_text(
            cx,
            if design {
                "试听文字"
            } else {
                "要朗读的文字"
            },
        );
        self.ui.label(cx, ids!(tts_instruction_title)).set_text(
            cx,
            if design {
                "想要的声音（必填）"
            } else {
                "声音描述（可选）"
            },
        );
        self.ui.button(cx, ids!(tts_synthesize)).set_text(
            cx,
            if design {
                "生成并试听"
            } else {
                "合成并播放"
            },
        );
        if design
            && self
                .ui
                .text_input(cx, ids!(tts_text))
                .text()
                .trim()
                .is_empty()
        {
            self.ui.text_input(cx, ids!(tts_text)).set_text(
                cx,
                "你好，很高兴认识你。今天一起听听这个声音，感受它的语气和节奏。",
            );
        }
    }
    fn tts_status(&self, cx: &mut Cx, text: &str) {
        self.ui.label(cx, ids!(tts_status)).set_text(cx, text);
    }
    fn load_tts(&mut self, cx: &mut Cx) {
        if self.runtime.busy() {
            return;
        }
        let Some(t) = self.tts.as_mut() else {
            return;
        };
        t.needs_directory = false;
        t.initialized = true;
        t.ready = false;
        t.busy = true;
        t.loading = true;
        t.reusable = false;
        if matches!(t.reference, Some(ReferenceSelection::Designed)) {
            t.reference = None;
            self.ui
                .text_input(cx, ids!(tts_reference_text))
                .set_is_read_only(cx, false);
            self.ui
                .text_input(cx, ids!(tts_reference_text))
                .set_text(cx, "");
            self.ui
                .label(cx, ids!(tts_reference_name))
                .set_text(cx, "可选 · 建议 3–10 秒清晰人声");
        }
        let result = t.engine.submit(Command::Load(t.path.clone()));
        self.ui.view(cx, ids!(tts_recovery)).set_visible(cx, false);
        self.ui.view(cx, ids!(tts_progress)).set_visible(cx, true);
        self.tts_progress(cx, 0.0);
        self.tts_status(cx, "正在加载 Breeze-TTS-2…");
        if let Err(error) = result {
            self.tts_error(cx, &error.to_string(), true);
        }
    }
    fn choose_tts(&mut self, cx: &mut Cx) {
        let Some(t) = self.tts.as_mut() else {
            return;
        };
        if t.picking || self.transcription.choosing_model() {
            return;
        }
        let Some(dialog) = self.runtime.dialog("tts") else {
            return;
        };
        t.dialog = Some(dialog);
        t.needs_directory = false;
        t.picking = true;
        t.busy = true;
        t.loading = true;
        crate::engine_ui::choose_directory(
            cx,
            &t.path,
            "选择 Breeze-TTS-2 模型目录（需包含 audio_tokenizer 子目录）",
        );
    }
    fn tts_progress(&self, cx: &mut Cx, progress: f64) {
        crate::engine_ui::set_progress(
            &self.ui,
            cx,
            ids!(tts_progress),
            ids!(tts_progress_fill),
            progress,
        );
    }
    fn tts_error(&mut self, cx: &mut Cx, error: &str, fatal: bool) {
        let Some(t) = self.tts.as_mut() else {
            return;
        };
        if t.loading || fatal {
            t.ready = false;
        }
        t.busy = false;
        t.loading = false;
        t.stopping = false;
        t.playing = false;
        self.ui.view(cx, ids!(tts_progress)).set_visible(cx, false);
        self.ui
            .view(cx, ids!(tts_recovery))
            .set_visible(cx, !t.ready);
        self.tts_status(cx, &format!("错误：{error}"));
    }
    pub fn refresh_tts_controls(&mut self, cx: &mut Cx) {
        let other_busy = self.inference_busy();
        let dialog_busy = self.runtime.dialog_busy();
        let Some(t) = self.tts.as_mut() else {
            return;
        };
        let state = [
            t.busy,
            t.loading,
            t.ready,
            t.stopping,
            t.has_audio,
            other_busy,
            dialog_busy,
            self.workspace.language(),
            t.session.enabled(),
            t.session.hearing(),
            t.session.can_choose(),
            !t.session.choices().is_empty(),
            t.session.draft().is_some(),
            t.session.previous_prompt().is_some(),
            t.hearing_audio[0].is_some(),
            t.hearing_audio[1].is_some(),
            t.hearing_audio[2].is_some(),
        ];
        if t.controls == Some(state) {
            return;
        }
        t.controls = Some(state);
        let idle = !t.busy && !other_busy && !dialog_busy;
        self.ui.button(cx, ids!(tts_load)).set_visible(cx, false);
        self.ui.button(cx, ids!(tts_synthesize)).set_text(
            cx,
            if t.loading {
                "正在载入…"
            } else if !t.ready {
                "载入语音模型"
            } else if t.design {
                "生成并试听"
            } else {
                "合成并播放"
            },
        );
        self.ui
            .button(cx, ids!(tts_synthesize))
            .set_action_enabled(cx, idle && !t.loading);
        self.ui.button(cx, ids!(tts_stop)).set_visible(cx, t.busy);
        self.ui.button(cx, ids!(tts_stop)).set_action_enabled(
            cx,
            t.busy && !t.stopping && !t.picking && !t.picking_reference,
        );
        self.ui.button(cx, ids!(tts_save)).set_action_enabled(
            cx,
            t.output.is_some() && !t.saving && !self.runtime.dialog_busy(),
        );
        self.ui
            .button(cx, ids!(tts_choose))
            .set_action_enabled(cx, idle);
        self.ui
            .button(cx, ids!(tts_retry))
            .set_action_enabled(cx, idle);
        self.ui.button(cx, ids!(tts_replay)).set_action_enabled(
            cx,
            !t.busy && t.output.is_some() && !self.runtime.dialog_busy(),
        );
        self.ui
            .button(cx, ids!(tts_design_mode))
            .set_action_enabled(cx, idle);
        self.ui
            .button(cx, ids!(tts_speech_mode))
            .set_action_enabled(cx, idle);
        self.ui
            .button(cx, ids!(tts_design_next))
            .set_action_enabled(cx, idle && t.ready && t.design);
        self.ui
            .button(cx, ids!(tts_design_use))
            .set_action_enabled(cx, idle && t.ready && t.design && t.reusable);
        for id in [
            ids!(tts_voice_gentle),
            ids!(tts_voice_warm),
            ids!(tts_voice_bright),
            ids!(tts_voice_english),
        ] {
            self.ui.button(cx, id).set_action_enabled(cx, idle);
        }
        self.ui
            .button(cx, ids!(tts_reference_choose))
            .set_action_enabled(cx, idle);
        self.ui
            .button(cx, ids!(tts_reference_clear))
            .set_action_enabled(cx, idle && t.reference.is_some());
        self.ui.button(cx, ids!(tts_enhance)).set_text(
            cx,
            if t.session.enabled() {
                "● 提示词增强"
            } else {
                "提示词增强"
            },
        );
        self.ui
            .button(cx, ids!(tts_enhance))
            .set_action_enabled(cx, idle);
        self.ui
            .view(cx, ids!(tts_hearing))
            .set_visible(cx, t.session.hearing());
        for (index, id) in [ids!(tts_hear_0), ids!(tts_hear_1), ids!(tts_hear_2)]
            .into_iter()
            .enumerate()
        {
            self.ui
                .button(cx, id)
                .set_action_enabled(cx, idle && t.hearing_audio[index].is_some());
        }
        for id in [ids!(tts_pick_0), ids!(tts_pick_1), ids!(tts_pick_2)] {
            self.ui
                .button(cx, id)
                .set_action_enabled(cx, idle && t.session.can_choose());
        }
        self.ui
            .button(cx, ids!(tts_draft))
            .set_action_enabled(cx, idle && !t.session.choices().is_empty());
        self.ui
            .button(cx, ids!(tts_draft_accept))
            .set_action_enabled(cx, idle && t.session.draft().is_some());
        self.ui
            .button(cx, ids!(tts_draft_discard))
            .set_action_enabled(cx, idle && t.session.draft().is_some());
        self.ui
            .button(cx, ids!(tts_draft_restore))
            .set_action_enabled(cx, idle && t.session.previous_prompt().is_some());
    }
    pub fn poll_tts(&mut self, cx: &mut Cx) {
        if let Some(t) = self.tts.as_mut() {
            if let Some(result) = t.exporting.as_ref().and_then(|rx| rx.try_recv().ok()) {
                t.exporting = None;
                t.saving = false;
                t.controls = None;
                self.tts_status(cx, &result.unwrap_or_else(|e| format!("保存失败：{e}")));
            }
        }
        let updates: Vec<_> = self
            .tts
            .as_ref()
            .map(|t| t.engine.updates.try_iter().collect())
            .unwrap_or_default();
        for mut update in updates {
            if update.kind == "enhanced" {
                self.enhanced_ready(cx, update.consideration.take());
                continue;
            }
            if update.kind == "held" {
                self.held_ready(cx, update.audio.take());
                continue;
            }
            if update.kind == "drafted" {
                if let Some(text) = update.draft_text.take() {
                    if let Some(t) = self.tts.as_mut() {
                        t.session.stage_draft(text.clone());
                        t.busy = false;
                        t.stopping = false;
                        t.controls = None;
                    }
                    self.ui
                        .text_input(cx, ids!(tts_draft_text))
                        .set_text(cx, &text);
                    self.reveal_tts_details(cx);
                    self.tts_status(cx, "增强提示草稿已显示，可修改后接受");
                }
                continue;
            }
            let Some(t) = self.tts.as_mut() else {
                return;
            };
            match update.kind {
                "generated" => {
                    if let Some(audio) = update.audio {
                        t.output = Some(audio);
                    }
                    t.has_audio = t.output.is_some();
                    t.playing = true;
                    t.controls = None;
                }
                "replaying" => {
                    t.playing = true;
                    self.tts_status(cx, &update.message);
                }
                "loading" => {
                    self.tts_progress(cx, update.progress);
                    self.tts_status(cx, &update.message);
                }
                "ready" => {
                    t.ready = true;
                    t.busy = false;
                    t.loading = false;
                    let result = if t.remember {
                        t.remember = false;
                        t.settings
                            .as_deref()
                            .map(|s| model_location::remember(s, &t.path))
                            .transpose()
                    } else {
                        Ok(None)
                    };
                    self.ui.view(cx, ids!(tts_progress)).set_visible(cx, false);
                    self.ui.view(cx, ids!(tts_recovery)).set_visible(cx, false);
                    self.tts_status(cx, &update.message);
                    if let Err(error) = result {
                        self.tts_status(cx, &format!("模型已就绪 · 目录未保存：{error}"));
                    }
                }
                "missing" => {
                    t.ready = false;
                    self.ui.view(cx, ids!(tts_progress)).set_visible(cx, false);
                    self.ui.view(cx, ids!(tts_recovery)).set_visible(cx, true);
                    t.busy = false;
                    t.loading = false;
                    t.needs_directory = true;
                    self.tts_status(cx, &update.message);
                }
                "remembered" => {
                    if let Some(t) = self.tts.as_mut() {
                        t.reusable = true;
                        t.busy = false;
                        t.stopping = false;
                        t.controls = None;
                    }
                    self.tts_status(cx, "已选定这一版");
                }
                "audio" | "draining" | "done" => {
                    t.has_audio = t.output.is_some();
                    if update.kind == "done" {
                        let unfinished = !matches!(t.pending, Pending::Idle);
                        if unfinished {
                            t.session.abandon();
                            t.pending = Pending::Idle;
                            t.hearing_audio = [None, None, None];
                            t.hearing_play = None;
                            t.controls = None;
                            self.ui.view(cx, ids!(tts_hearing)).set_visible(cx, false);
                        }
                        t.playing = false;
                        t.busy = false;
                        t.stopping = false;
                        t.reusable = update.reusable;
                        if let Some(seed) = update.design_seed {
                            self.ui.label(cx, ids!(tts_design_candidate)).set_text(
                                cx,
                                &format!("候选 {} · 已完成", seed.saturating_sub(41)),
                            );
                        } else if t.design {
                            self.ui
                                .label(cx, ids!(tts_design_candidate))
                                .set_text(cx, &format!("候选 {} · 未完成", t.candidate));
                        }
                    }
                    if !t.stopping {
                        self.tts_status(cx, &update.message);
                    }
                    if let Some(playback) = &update.playback {
                        let first = playback
                            .first_playback_seconds
                            .map(|s| format!("{s:.2} s"))
                            .unwrap_or_else(|| "等待播放".into());
                        self.ui.label(cx, ids!(tts_metrics)).set_text(
                            cx,
                            &format!(
                                "音频 {:.2} s  ·  首声 {}  ·  缓冲 {:.2} s  ·  生成 {:.2} s",
                                update.audio_seconds,
                                first,
                                playback.buffered_seconds,
                                update.elapsed
                            ),
                        );
                    } else {
                        self.ui.label(cx, ids!(tts_metrics)).set_text(
                            cx,
                            &format!(
                                "音频 {:.2} s  ·  首段 {:.2} s  ·  生成 {:.2} s  ·  RTF {:.3}",
                                update.audio_seconds,
                                update.first_audio.unwrap_or(0.0),
                                update.elapsed,
                                update.elapsed / update.audio_seconds.max(0.001)
                            ),
                        );
                    }
                }
                "replayed" | "voice_cancelled" => {
                    let played_to_end = update.message == "试听结束";
                    let hearing = t.session.hearing();
                    let index = t.hearing_play.take();
                    let keep =
                        omni_breeze_tts::enhancement::candidate_is_choosable(if played_to_end {
                            omni_breeze_tts::enhancement::CandidateEnd::Normal {
                                delivered_audio: true,
                            }
                        } else {
                            omni_breeze_tts::enhancement::CandidateEnd::Stopped
                        });
                    if keep {
                        if let Some(index) = index {
                            let _ = t.session.mark_heard(index);
                        }
                    }
                    t.playing = false;
                    t.busy = false;
                    t.stopping = false;
                    t.controls = None;
                    let abandon = hearing && !keep;
                    let ready = t.session.can_choose();
                    if abandon {
                        t.session.abandon();
                        t.pending = Pending::Idle;
                        t.hearing_audio = [None, None, None];
                        t.controls = None;
                        self.ui.view(cx, ids!(tts_hearing)).set_visible(cx, false);
                    }
                    let message: &str = if abandon {
                        "已停止 · 未完成的盲选已作废"
                    } else if ready {
                        "三段都已听完，可以选择"
                    } else {
                        update.message.as_str()
                    };
                    self.tts_status(cx, message);
                }
                "voice_ready" => {
                    t.busy = false;
                    t.stopping = false;
                    t.reference = Some(ReferenceSelection::Designed);
                    t.controls = None;
                    self.ui.label(cx, ids!(tts_reference_name)).set_text(
                        cx,
                        &format!(
                            "已设计音色 · 候选 {}",
                            update.design_seed.unwrap_or(42).saturating_sub(41)
                        ),
                    );
                    self.ui
                        .text_input(cx, ids!(tts_reference_text))
                        .set_text(cx, update.reference_text.as_deref().unwrap_or_default());
                    self.ui
                        .text_input(cx, ids!(tts_reference_text))
                        .set_is_read_only(cx, true);
                    self.ui
                        .text_input(cx, ids!(tts_instruction))
                        .set_text(cx, "");
                    self.set_tts_design_mode(cx, false);
                    self.tts_status(cx, &update.message);
                }
                "load_cancelled" | "cancelled" => {
                    self.drop_unfinished_hearing(cx);
                    self.tts_error(cx, &update.message, false);
                    self.tts_status(cx, &update.message);
                }
                "error" | "fatal" => {
                    self.drop_unfinished_hearing(cx);
                    self.tts_error(cx, &update.message, update.kind == "fatal");
                }
                _ => self.tts_status(cx, &update.message),
            }
        }
    }
    pub fn handle_tts_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        if self.tts.is_none() {
            return;
        }
        let inference_busy = self.inference_busy();
        for action in actions {
            let t = self.tts.as_mut().unwrap();
            match action.downcast_ref::<FileDialogAction>() {
                Some(FileDialogAction::FolderSelected(path)) if t.picking => {
                    t.dialog.take();
                    t.picking = false;
                    t.path = model_location::selected_directory(path);
                    t.remember = true;
                    t.ready = false;
                    t.initialized = false;
                    t.busy = false;
                    t.loading = false;
                    if self.workspace.language() && !self.runtime.busy() {
                        self.load_tts(cx);
                    }
                }
                Some(FileDialogAction::FolderCancelled) if t.picking => {
                    t.dialog.take();
                    t.picking = false;
                    t.busy = false;
                    t.loading = false;
                    self.tts_status(cx, "尚未选择模型 · 选择目录后自动加载");
                }
                Some(FileDialogAction::SaveFileSelected { path, .. })
                    if t.saving && t.dialog.is_some() =>
                {
                    t.dialog.take();
                    t.exporting = t
                        .saving_snapshot
                        .take()
                        .map(|snapshot| snapshot.save(path.clone(), SignalToUI::set_ui_signal));
                    t.controls = None;
                }
                Some(FileDialogAction::SaveFileCancelled { .. })
                    if t.saving && t.dialog.is_some() =>
                {
                    t.dialog.take();
                    t.saving_snapshot = None;
                    t.saving = false;
                    t.controls = None;
                }
                Some(FileDialogAction::FileSelected { paths, .. }) if t.picking_reference => {
                    t.dialog.take();
                    t.picking_reference = false;
                    t.busy = false;
                    t.controls = None;
                    if let Some(path) = paths.first() {
                        t.reference = Some(ReferenceSelection::File(path.clone()));
                        self.ui
                            .text_input(cx, ids!(tts_reference_text))
                            .set_is_read_only(cx, false);
                        self.ui
                            .label(cx, ids!(tts_reference_name))
                            .set_text(cx, &path.file_name().unwrap_or_default().to_string_lossy());
                        self.tts_status(cx, "已选择参考音频 · 请填写音频中实际说出的文字");
                    }
                }
                Some(FileDialogAction::FileCancelled { .. }) if t.picking_reference => {
                    t.dialog.take();
                    t.picking_reference = false;
                    t.busy = false;
                    self.tts_status(cx, "已取消选择参考音频");
                }
                _ => {}
            }
        }
        if !self.workspace.language() {
            return;
        }
        let t = self.tts.as_mut().unwrap();
        if self.ui.button(cx, ids!(tts_stop)).clicked(actions)
            && t.busy
            && !t.picking
            && !t.picking_reference
        {
            t.stopping = true;
            t.engine.cancel();
            self.tts_status(cx, "正在停止合成 · 扬声器已静音");
            return;
        }
        if self.ui.button(cx, ids!(tts_save)).clicked(actions) && t.output.is_some() && !t.saving {
            let Some(dialog) = self.runtime.dialog("tts") else {
                return;
            };
            t.dialog = Some(dialog);
            t.saving_snapshot = t.output.clone().map(crate::export::Snapshot::Audio);
            t.saving = true;
            t.controls = None;
            cx.open_save_file_dialog(
                FileDialog::new()
                    .set_title("保存合成音频".into())
                    .set_filename("breeze.wav".into()),
            );
            return;
        }
        if self.ui.button(cx, ids!(tts_replay)).clicked(actions)
            && !t.busy
            && t.output.is_some()
            && !self.runtime.dialog_busy()
        {
            if let Some(m) = &mut self.music {
                m.stop_playback();
            }
            t.playing = true;
            t.busy = true;
            t.stopping = false;
            if let Err(error) = t
                .engine
                .submit(Command::Replay(t.output.as_ref().unwrap().clone()))
            {
                self.tts_error(cx, &error.to_string(), false);
            }
            return;
        }
        if t.busy || inference_busy || self.runtime.dialog_busy() {
            return;
        }
        if self.ui.button(cx, ids!(tts_design_mode)).clicked(actions) {
            self.reveal_tts_details(cx);
            self.set_tts_design_mode(cx, true);
            self.tts_status(cx, "描述想要的声音，生成试听后可换个候选或直接选用");
            return;
        }
        if self.ui.button(cx, ids!(tts_speech_mode)).clicked(actions) {
            self.set_tts_design_mode(cx, false);
            return;
        }
        for (id, description) in [
            (ids!(tts_voice_gentle), "一位温柔自信的年轻女性，声音清晰，音色温暖柔和，语气亲切，语速自然。"),
            (ids!(tts_voice_warm), "一位沉稳的成年男性，声音低沉温暖，吐字清晰，语气从容可靠，以平稳自然的节奏讲述。"),
            (ids!(tts_voice_bright), "一位清朗的年轻男性，声音明亮有活力，吐字干净，语气轻快友好，语速自然。"),
            (ids!(tts_voice_english), "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."),
        ] {
            if self.ui.button(cx, id).clicked(actions) && t.design {
                self.ui.text_input(cx, ids!(tts_instruction)).set_text(cx, description);
                return;
            }
        }

        if self.ui.button(cx, ids!(tts_design_use)).clicked(actions) && t.design && t.reusable {
            t.busy = true;
            t.stopping = false;
            if let Err(error) = t.engine.submit(Command::UseDesign) {
                self.tts_error(cx, &error.to_string(), false);
            }
            return;
        }
        if self
            .ui
            .button(cx, ids!(tts_reference_choose))
            .clicked(actions)
        {
            if t.design {
                return;
            }
            let Some(dialog) = self.runtime.dialog("tts") else {
                return;
            };
            t.dialog = Some(dialog);
            t.picking_reference = true;
            t.busy = true;
            cx.open_select_file_dialog(
                FileDialog::new()
                    .set_title("选择参考 WAV（建议 3–10 秒清晰人声）".into())
                    .add_filter("WAV 音频".into(), vec!["wav".into()]),
            );
            return;
        }
        if self
            .ui
            .button(cx, ids!(tts_reference_clear))
            .clicked(actions)
        {
            t.reference = None;
            self.ui
                .text_input(cx, ids!(tts_reference_text))
                .set_is_read_only(cx, false);
            t.controls = None;
            self.ui
                .text_input(cx, ids!(tts_reference_text))
                .set_text(cx, "");
            self.ui
                .label(cx, ids!(tts_reference_name))
                .set_text(cx, "可选 · 建议 3–10 秒清晰人声");
            self.tts_status(cx, "已清除参考声音");
            return;
        }
        if self.ui.button(cx, ids!(tts_choose)).clicked(actions) {
            self.choose_tts(cx);
            return;
        }
        if self.ui.button(cx, ids!(tts_retry)).clicked(actions) {
            self.load_tts(cx);
            return;
        }
        if self.ui.button(cx, ids!(tts_enhance)).clicked(actions) {
            let enabled = !t.session.enabled();
            t.session.set_enabled(enabled);
            t.controls = None;
            if !enabled {
                t.session.abandon();
                t.pending = Pending::Idle;
                t.hearing_audio = [None, None, None];
                t.hearing_play = None;
                self.ui.view(cx, ids!(tts_hearing)).set_visible(cx, false);
            }
            self.remember_enhancement();
            self.tts_status(
                cx,
                if enabled {
                    "提示词增强已打开。合成前会先改写朗读文字和声音描述"
                } else {
                    "提示词增强已关闭"
                },
            );
            return;
        }
        for (index, id) in [ids!(tts_hear_0), ids!(tts_hear_1), ids!(tts_hear_2)]
            .into_iter()
            .enumerate()
        {
            if self.ui.button(cx, id).clicked(actions) {
                let Some(audio) = t.hearing_audio[index].clone() else {
                    return;
                };
                if let Some(m) = &mut self.music {
                    m.stop_playback();
                }
                t.hearing_play = Some(index);
                t.playing = true;
                t.busy = true;
                t.stopping = false;
                t.controls = None;
                if let Err(error) = t.engine.submit(Command::Replay(audio)) {
                    self.tts_error(cx, &error.to_string(), false);
                }
                return;
            }
        }
        for (index, id) in [ids!(tts_pick_0), ids!(tts_pick_1), ids!(tts_pick_2)]
            .into_iter()
            .enumerate()
        {
            if self.ui.button(cx, id).clicked(actions) && t.session.can_choose() {
                let design = t.design;
                let seed = t.candidate.wrapping_add(41);
                match t.session.choose(index) {
                    Ok(pair) => {
                        let audio = t.hearing_audio[index].clone();
                        t.output = audio.clone();
                        t.has_audio = audio.is_some();
                        t.hearing_audio = [None, None, None];
                        t.hearing_play = None;
                        t.controls = None;
                        let submit = if design {
                            audio.map(|audio| {
                                t.busy = true;
                                t.engine.submit(Command::Remember {
                                    audio,
                                    text: pair.speech_text,
                                    seed,
                                })
                            })
                        } else {
                            None
                        };
                        self.ui.view(cx, ids!(tts_hearing)).set_visible(cx, false);
                        self.remember_enhancement();
                        if let Some(Err(error)) = submit {
                            self.tts_error(cx, &error.to_string(), false);
                        }
                        self.tts_status(cx, "已选定这一版");
                    }
                    Err(error) => self.tts_status(cx, error),
                }
                return;
            }
        }
        if self.ui.button(cx, ids!(tts_draft)).clicked(actions) {
            match t.session.draft_request() {
                Ok(request) => {
                    let seed = t.session.next_sample_seeds()[0];
                    let checkpoint = t.checkpoint.clone();
                    t.busy = true;
                    t.stopping = false;
                    t.controls = None;
                    if let Err(error) = t.engine.submit(Command::Draft {
                        request,
                        seed,
                        checkpoint,
                    }) {
                        self.tts_error(cx, &error.to_string(), false);
                    } else {
                        self.tts_status(cx, "正在根据语音选择起草增强提示");
                    }
                }
                Err(error) => self.tts_status(cx, error),
            }
            return;
        }
        if self.ui.button(cx, ids!(tts_draft_accept)).clicked(actions) {
            let displayed = self.ui.text_input(cx, ids!(tts_draft_text)).text();
            match t.session.accept_displayed(&displayed) {
                Ok(()) => {
                    t.controls = None;
                    self.remember_enhancement();
                    self.tts_status(cx, "已换上增强提示");
                }
                Err(error) => self.tts_status(cx, error),
            }
            return;
        }
        if self.ui.button(cx, ids!(tts_draft_discard)).clicked(actions) {
            t.session.discard_draft();
            t.controls = None;
            self.ui
                .text_input(cx, ids!(tts_draft_text))
                .set_text(cx, "");
            self.tts_status(cx, "未接受草稿，当前增强提示不变");
            return;
        }
        if self.ui.button(cx, ids!(tts_draft_restore)).clicked(actions) {
            match t.session.restore_prompt() {
                Ok(()) => {
                    t.controls = None;
                    self.remember_enhancement();
                    self.tts_status(cx, "已恢复上一版增强提示");
                }
                Err(error) => self.tts_status(cx, error),
            }
            return;
        }

        let next_candidate = t.design && self.ui.button(cx, ids!(tts_design_next)).clicked(actions);
        if (self.ui.button(cx, ids!(tts_synthesize)).clicked(actions) || next_candidate) && t.ready
        {
            let text = self.ui.text_input(cx, ids!(tts_text)).text();
            if text.trim().is_empty() {
                self.tts_status(cx, "请输入要合成的文本");
                return;
            }
            let instruction = self.ui.text_input(cx, ids!(tts_instruction)).text();
            if t.design && instruction.trim().is_empty() {
                self.tts_status(cx, "请描述想要的声音，或选择一个描述示例");
                return;
            }
            let reference_text = self.ui.text_input(cx, ids!(tts_reference_text)).text();
            let has_reference = t.reference.is_some();
            if !t.design && has_reference == reference_text.trim().is_empty() {
                self.tts_status(
                    cx,
                    if has_reference {
                        "请填写参考音频中实际说出的文字"
                    } else {
                        "参考文字需要配套参考 WAV，请选择音频或清空参考文字"
                    },
                );
                return;
            }
            let reference = if t.design {
                None
            } else {
                t.reference.as_ref().map(|r| match r {
                    ReferenceSelection::File(path) => ReferenceInput::File {
                        path: path.clone(),
                        text: reference_text,
                    },
                    ReferenceSelection::Designed => ReferenceInput::Designed,
                })
            };
            if next_candidate {
                t.candidate = t.candidate.saturating_add(1);
            }
            let seed = if t.design {
                t.candidate.wrapping_add(41)
            } else {
                42
            };
            if t.design {
                self.ui
                    .label(cx, ids!(tts_design_candidate))
                    .set_text(cx, &format!("候选 {} · 正在生成", t.candidate));
            }
            let design = t.design;
            let enhance = t.session.enabled();
            if enhance {
                t.session.abandon();
                t.hearing_audio = [None, None, None];
                t.hearing_play = None;
                t.pending = Pending::Sampling {
                    reference: reference.clone(),
                    design,
                    speech_seed: seed,
                };
                self.ui.view(cx, ids!(tts_hearing)).set_visible(cx, false);
            }
            t.busy = true;
            t.reusable = false;
            t.stopping = false;
            t.controls = None;
            if let Some(m) = &mut self.music {
                m.stop_playback();
            }
            let result = if enhance {
                let seeds = t.session.next_sample_seeds();
                let prompt = t.session.enhancer_prompt().to_string();
                let checkpoint = t.checkpoint.clone();
                t.engine.submit(Command::Enhance {
                    source: omni_breeze_tts::enhancement::Source {
                        speech_text: text,
                        voice_instruction: instruction,
                    },
                    prompt,
                    seeds,
                    checkpoint,
                })
            } else {
                t.engine.submit(Command::Synthesize {
                    options: omni_breeze_tts::SynthesisOptions {
                        text,
                        instruction,
                        sampling: omni_breeze_tts::Sampling {
                            seed,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    reference,
                    design,
                    hold: false,
                })
            };
            self.tts_status(
                cx,
                if enhance {
                    "正在用 35B 增强朗读文字和声音描述"
                } else {
                    "正在生成首段音频 · 自动缓冲，优先保持播放连贯"
                },
            );
            if let Err(error) = result {
                self.tts_error(cx, &error.to_string(), false);
            }
        }
    }

    fn remember_enhancement(&self) {
        let Some(t) = self.tts.as_ref() else {
            return;
        };
        let Some(path) = t.enhancement_settings.as_ref() else {
            return;
        };
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, t.session.store());
    }

    fn drop_unfinished_hearing(&mut self, cx: &mut Cx) {
        let Some(t) = self.tts.as_mut() else {
            return;
        };
        if matches!(t.pending, Pending::Idle) && !t.session.hearing() {
            return;
        }
        t.session.abandon();
        t.pending = Pending::Idle;
        t.hearing_audio = [None, None, None];
        t.hearing_play = None;
        t.controls = None;
        self.ui.view(cx, ids!(tts_hearing)).set_visible(cx, false);
    }

    fn enhanced_ready(
        &mut self,
        cx: &mut Cx,
        consideration: Option<omni_breeze_tts::enhancement::Consideration>,
    ) {
        let step = {
            let Some(t) = self.tts.as_mut() else {
                return;
            };
            let pending = std::mem::replace(&mut t.pending, Pending::Idle);
            let Pending::Sampling {
                reference,
                design,
                speech_seed,
            } = pending
            else {
                t.busy = false;
                t.controls = None;
                return;
            };
            match consideration {
                Some(omni_breeze_tts::enhancement::Consideration::Failed(reason)) => {
                    t.busy = false;
                    t.stopping = false;
                    t.controls = None;
                    Err(reason)
                }
                Some(omni_breeze_tts::enhancement::Consideration::Speak(pair)) => {
                    Ok((pair, reference, design, speech_seed, false))
                }
                Some(omni_breeze_tts::enhancement::Consideration::Choose {
                    alternatives,
                    enhancer_prompt,
                }) => {
                    t.session
                        .start_hearing(alternatives.clone(), &enhancer_prompt);
                    t.hearing_audio = [None, None, None];
                    let pair = alternatives[0].clone();
                    t.pending = Pending::Holding {
                        next: 0,
                        pairs: alternatives,
                        reference: reference.clone(),
                        design,
                        speech_seed,
                    };
                    t.controls = None;
                    Ok((pair, reference, design, speech_seed, true))
                }
                None => {
                    t.busy = false;
                    t.controls = None;
                    return;
                }
            }
        };
        match step {
            Err(reason) => self.tts_status(cx, &reason),
            Ok((pair, reference, design, seed, hold)) => {
                if hold {
                    self.ui.view(cx, ids!(tts_hearing)).set_visible(cx, true);
                    self.tts_status(cx, "正在准备甲、乙、丙");
                } else {
                    self.tts_status(cx, "增强结果一致，开始合成");
                }
                self.submit_pair(
                    cx,
                    pair.speech_text,
                    pair.voice_instruction,
                    seed,
                    reference,
                    design,
                    hold,
                );
            }
        }
    }

    fn held_ready(&mut self, cx: &mut Cx, audio: Option<std::sync::Arc<[f32]>>) {
        let next = {
            let Some(t) = self.tts.as_mut() else {
                return;
            };
            let Pending::Holding {
                next,
                pairs,
                reference,
                design,
                speech_seed,
            } = &mut t.pending
            else {
                t.busy = false;
                t.controls = None;
                return;
            };
            if let Some(audio) = audio {
                if *next < 3 {
                    t.hearing_audio[*next] = Some(audio);
                }
            }
            *next += 1;
            if *next >= 3 {
                t.pending = Pending::Idle;
                t.busy = false;
                t.stopping = false;
                t.controls = None;
                None
            } else {
                let index = *next;
                Some((
                    pairs[index].clone(),
                    reference.clone(),
                    *design,
                    speech_seed.wrapping_add(index as u64),
                ))
            }
        };
        match next {
            None => self.tts_status(cx, "请把甲、乙、丙都听完再选择"),
            Some((pair, reference, design, seed)) => self.submit_pair(
                cx,
                pair.speech_text,
                pair.voice_instruction,
                seed,
                reference,
                design,
                true,
            ),
        }
    }

    fn submit_pair(
        &mut self,
        cx: &mut Cx,
        text: String,
        instruction: String,
        seed: u64,
        reference: Option<ReferenceInput>,
        design: bool,
        hold: bool,
    ) {
        let Some(t) = self.tts.as_mut() else {
            return;
        };
        t.busy = true;
        t.reusable = false;
        t.stopping = false;
        t.controls = None;
        if let Err(error) = t.engine.submit(Command::Synthesize {
            options: omni_breeze_tts::SynthesisOptions {
                text,
                instruction,
                sampling: omni_breeze_tts::Sampling {
                    seed,
                    ..Default::default()
                },
                ..Default::default()
            },
            reference,
            design,
            hold,
        }) {
            self.tts_error(cx, &error.to_string(), false);
        }
    }
}
