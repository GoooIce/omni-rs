use crate::{
    omni_engine::{Command, Engine, Part},
    ActionButtonState, App,
};
use makepad_widgets::*;
use omni_lfm3::generate::OutputMode;
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Receiver,
        Arc,
    },
};

enum Picking {
    Model,
    Media,
    Save(crate::export::Snapshot),
}
pub struct State {
    engine: Engine,
    settings: Option<PathBuf>,
    dialog: Option<crate::desktop_runtime::Dialog>,
    picking: Option<Picking>,
    exporting: Option<Receiver<Result<String, String>>>,
    visible: bool,
    ready: bool,
    loaded_path: Option<PathBuf>,
    loading_path: Option<PathBuf>,
    busy: bool,
    playing: bool,
    stopping: bool,
    listening: Option<Arc<AtomicBool>>,
    stable_listen: bool,
    heard: String,
    can_continue: bool,
    mode: usize,
    parts: Vec<Part>,
    submitted: Option<String>,
    pending_user: String,
    response: String,
    transcript: String,
    audio: Option<Arc<[f32]>>,
}
impl State {
    fn new(runtime: crate::desktop_runtime::Runtime) -> Self {
        Self {
            engine: Engine::spawn_shared(SignalToUI::set_ui_signal, runtime),
            settings: crate::model_location::settings_path_for(
                "LFM3_SETTINGS_PATH",
                "lfm3-omni.json",
            ),
            dialog: None,
            picking: None,
            exporting: None,
            visible: false,
            ready: false,
            loaded_path: None,
            loading_path: None,
            busy: false,
            playing: false,
            stopping: false,
            listening: None,
            stable_listen: false,
            heard: String::new(),
            can_continue: true,
            mode: 0,
            parts: Vec::new(),
            submitted: None,
            pending_user: String::new(),
            response: String::new(),
            transcript: String::new(),
            audio: None,
        }
    }
    pub fn visible(&self) -> bool {
        self.visible
    }
    pub fn playing(&self) -> bool {
        self.playing
    }
    pub fn stop_playback(&mut self) {
        if self.playing {
            self.stopping = true;
            self.engine.cancel();
        }
    }
}
fn part_label(part: &Part) -> String {
    match part {
        Part::Text(text) => format!("文字：{}", text.chars().take(24).collect::<String>()),
        Part::File(path) => path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
    }
}
impl App {
    pub fn initialize_omni(&mut self, cx: &mut Cx, root: &Path) {
        let state = State::new(self.runtime.clone());
        let path = crate::model_location::initial_directory_for(
            root,
            state.settings.as_deref(),
            "LFM3_MODEL_DIR",
            "models/LFM3-Omni-2B",
        );
        self.ui
            .text_input(cx, ids!(omni_model))
            .set_text(cx, &path.to_string_lossy());
        self.omni = Some(state);
    }
    fn omni_status(&self, cx: &mut Cx, message: &str) {
        self.ui.label(cx, ids!(omni_status)).set_text(cx, message);
    }
    fn omni_parts(&self, cx: &mut Cx) {
        let Some(state) = &self.omni else {
            return;
        };
        let label = if state.parts.is_empty() {
            "当前轮次尚未加入内容".to_owned()
        } else {
            format!(
                "已排入：{}",
                state
                    .parts
                    .iter()
                    .map(part_label)
                    .collect::<Vec<_>>()
                    .join(" → ")
            )
        };
        self.ui.label(cx, ids!(omni_parts)).set_text(cx, &label);
    }
    fn omni_transcript(&self, cx: &mut Cx) {
        let Some(state) = &self.omni else {
            return;
        };
        let tail = if state.pending_user.is_empty() {
            String::new()
        } else {
            format!("\n你：{}\n助手：{}", state.pending_user, state.response)
        };
        self.ui
            .text_input(cx, ids!(omni_transcript))
            .set_text(cx, &format!("{}{}", state.transcript, tail));
    }
    pub fn refresh_omni_controls(&mut self, cx: &mut Cx) {
        let Some(state) = &self.omni else {
            return;
        };
        let idle = !state.busy && !self.runtime.busy() && !self.runtime.dialog_busy();
        for id in [
            ids!(omni_choose_model),
            ids!(omni_load),
            ids!(omni_add_text),
            ids!(omni_add_media),
            ids!(omni_clear_parts),
            ids!(omni_mode),
            ids!(omni_listen_route),
        ] {
            self.ui.button(cx, id).set_action_enabled(cx, idle);
        }
        self.ui
            .button(cx, ids!(omni_send))
            .set_action_enabled(cx, idle && state.ready && state.can_continue);
        self.ui
            .button(cx, ids!(omni_new))
            .set_action_enabled(cx, idle && state.ready);
        self.ui
            .button(cx, ids!(omni_stop))
            .set_action_enabled(cx, state.busy && !state.stopping);
        self.ui.button(cx, ids!(omni_microphone)).set_text(
            cx,
            if state.listening.is_some() {
                "结束录音"
            } else {
                "中文听写"
            },
        );
        self.ui
            .button(cx, ids!(omni_microphone))
            .set_action_enabled(
                cx,
                state
                    .listening
                    .as_ref()
                    .map_or(idle && state.ready, |finish| {
                        !state.stopping && !finish.load(Ordering::Acquire)
                    }),
            );
        self.ui
            .button(cx, ids!(omni_play))
            .set_action_enabled(cx, idle && state.audio.is_some());
        self.ui.button(cx, ids!(omni_save)).set_action_enabled(
            cx,
            idle && state.audio.is_some() && state.exporting.is_none(),
        );
        self.ui.button(cx, ids!(omni_mode)).set_text(
            cx,
            [
                "输出：自动",
                "输出：文字",
                "输出：语音",
                "输出：文字语音交错",
            ][state.mode],
        );
        self.ui.button(cx, ids!(omni_listen_route)).set_text(
            cx,
            if state.stable_listen {
                "听写：稳定出字"
            } else {
                "听写：预览校正"
            },
        );
    }
    pub fn poll_omni(&mut self, cx: &mut Cx) {
        if let Some(state) = self.omni.as_mut() {
            if let Some(result) = state.exporting.as_ref().and_then(|rx| rx.try_recv().ok()) {
                state.exporting = None;
                self.omni_status(
                    cx,
                    &result.unwrap_or_else(|error| format!("保存失败：{error}")),
                );
            }
        }
        let updates: Vec<_> = self
            .omni
            .as_ref()
            .map(|s| s.engine.updates.try_iter().collect())
            .unwrap_or_default();
        for update in updates {
            let state = self.omni.as_mut().unwrap();
            if matches!(
                update.kind,
                "loading" | "preparing" | "playing" | "speech" | "listening" | "refining"
            ) {
                if !state.stopping {
                    self.omni_status(cx, &update.message);
                }
                continue;
            }
            if update.kind == "text" {
                state.response = update.text;
                self.omni_transcript(cx);
                continue;
            }
            if matches!(update.kind, "heard" | "heard_final" | "heard_stable") {
                state.heard = update.text;
                self.ui.label(cx, ids!(omni_live)).set_text(
                    cx,
                    &format!(
                        "{}：{}",
                        match update.kind {
                            "heard" => "实时预览",
                            "heard_stable" => "已确认",
                            _ => "完整转写",
                        },
                        state.heard
                    ),
                );
                continue;
            }
            state.busy = false;
            state.playing = false;
            state.stopping = false;
            if state.listening.take().is_some() {
                let text = if update.text.is_empty() {
                    std::mem::take(&mut state.heard)
                } else {
                    update.text.clone()
                };
                state.heard.clear();
                if !text.is_empty() {
                    let current = self.ui.text_input(cx, ids!(omni_draft)).text();
                    let separator = if current.is_empty() { "" } else { "\n" };
                    self.ui
                        .text_input(cx, ids!(omni_draft))
                        .set_text(cx, &format!("{current}{separator}{text}"));
                }
                self.omni_status(cx, &update.message);
                continue;
            }
            if update.kind == "ready" {
                state.ready = true;
                state.can_continue = true;
                state.loaded_path = state.loading_path.take();
                if let (Some(settings), Some(path)) = (&state.settings, &state.loaded_path) {
                    let _ = crate::model_location::remember(settings, path);
                }
                state.transcript.clear();
                state.audio = None;
            } else if update.kind == "cleared" {
                state.transcript.clear();
                state.response.clear();
                state.pending_user.clear();
                state.audio = None;
                state.can_continue = true;
            } else if matches!(update.kind, "done" | "cancelled") && state.submitted.is_some() {
                if !update.text.is_empty() {
                    state.response = update.text;
                }
                if let Some(audio) = update.audio {
                    state.audio = Some(audio);
                }
                state.can_continue = update.kind == "done" && update.can_continue;
                state.transcript.push_str(&format!(
                    "\n你：{}\n助手：{}\n",
                    state.pending_user, state.response
                ));
                state.pending_user.clear();
                state.response.clear();
                if update.kind == "done"
                    && state.submitted.as_deref()
                        == Some(self.ui.text_input(cx, ids!(omni_draft)).text().as_str())
                {
                    self.ui.text_input(cx, ids!(omni_draft)).set_text(cx, "");
                }
                if update.kind == "done" {
                    state.parts.clear();
                }
                state.submitted = None;
            } else if matches!(update.kind, "missing" | "load_cancelled" | "fatal") {
                state.ready = false;
                state.loaded_path = None;
                state.loading_path = None;
                if update.kind == "missing" {
                    self.notify_panel_missing(cx, omni_hub::catalog::CheckpointId::Lfm3Omni2b);
                }
            } else if update.kind == "error" {
                if state.submitted.is_some() {
                    state.can_continue = false;
                }
                state.submitted = None;
            }
            self.omni_status(cx, &update.message);
            self.omni_parts(cx);
            self.omni_transcript(cx);
        }
    }
    pub fn open_omni_model_picker(&mut self, cx: &mut Cx) {
        let Some(dialog) = self.runtime.dialog("lfm3-omni") else {
            return;
        };
        let Some(state) = self.omni.as_mut() else {
            return;
        };
        state.dialog = Some(dialog);
        state.picking = Some(Picking::Model);
        crate::engine_ui::choose_directory(
            cx,
            Path::new(&self.ui.text_input(cx, ids!(omni_model)).text()),
            "选择 LFM3-Omni-2B 模型目录",
        );
    }
    pub fn handle_omni_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        if self.omni.is_none() {
            return;
        }
        let mut chosen_omni_dir: Option<PathBuf> = None;
        for action in actions {
            let Some(action) = action.downcast_ref::<FileDialogAction>() else {
                continue;
            };
            let state = self.omni.as_mut().unwrap();
            match action {
                FileDialogAction::FolderSelected(path)
                    if matches!(state.picking, Some(Picking::Model)) =>
                {
                    let dir = crate::model_location::selected_directory(path);
                    self.ui
                        .text_input(cx, ids!(omni_model))
                        .set_text(cx, &dir.to_string_lossy());
                    chosen_omni_dir = Some(dir);
                }
                FileDialogAction::FileSelected { paths, .. }
                    if matches!(state.picking, Some(Picking::Media)) =>
                {
                    for path in paths {
                        state.parts.push(Part::File(path.clone()));
                    }
                }
                FileDialogAction::SaveFileSelected { path, .. }
                    if matches!(state.picking, Some(Picking::Save(_))) =>
                {
                    if let Some(Picking::Save(snapshot)) = state.picking.take() {
                        state.exporting =
                            Some(snapshot.save(path.clone(), SignalToUI::set_ui_signal));
                    }
                }
                FileDialogAction::FolderCancelled
                    if matches!(state.picking, Some(Picking::Model)) => {}
                FileDialogAction::FileCancelled { .. }
                    if matches!(state.picking, Some(Picking::Media)) => {}
                FileDialogAction::SaveFileCancelled { .. }
                    if matches!(state.picking, Some(Picking::Save(_))) => {}
                _ => continue,
            }
            state.picking = None;
            state.dialog = None;
            self.omni_parts(cx);
        }
        if let Some(dir) = chosen_omni_dir {
            self.sync_checkpoint_directory(cx, omni_hub::catalog::CheckpointId::Lfm3Omni2b, dir);
        }
        if self.ui.button(cx, ids!(omni_toggle)).clicked(actions)
            || self.ui.button(cx, ids!(omni_back)).clicked(actions)
        {
            let state = self.omni.as_mut().unwrap();
            state.visible = !state.visible;
            self.ui
                .view(cx, ids!(omni_panel))
                .set_visible(cx, state.visible);
            self.ui
                .view(cx, ids!(language_columns))
                .set_visible(cx, !state.visible);
        }
        if !self.omni.as_ref().unwrap().visible || !self.workspace.language() {
            return;
        }
        if self.ui.button(cx, ids!(omni_listen_route)).clicked(actions) {
            let state = self.omni.as_mut().unwrap();
            state.stable_listen = !state.stable_listen;
            self.refresh_omni_controls(cx);
            return;
        }
        if self.ui.button(cx, ids!(omni_microphone)).clicked(actions) {
            let state = self.omni.as_mut().unwrap();
            if let Some(finish) = &state.listening {
                finish.store(true, Ordering::Release);
                self.omni_status(cx, "正在结束录音并处理尾音…");
            } else {
                let path = PathBuf::from(self.ui.text_input(cx, ids!(omni_model)).text());
                if state.loaded_path.as_ref() != Some(&path) {
                    self.omni_status(cx, "模型目录已改变，请重新载入");
                    return;
                }
                let finish = Arc::new(AtomicBool::new(false));
                let command = if state.stable_listen {
                    Command::ListenStable(finish.clone())
                } else {
                    Command::Listen(finish.clone())
                };
                match state.engine.submit(command) {
                    Ok(()) => {
                        state.listening = Some(finish);
                        state.heard.clear();
                        state.busy = true;
                        state.stopping = false;
                        self.ui.label(cx, ids!(omni_live)).set_text(cx, "");
                        self.omni_status(cx, "正在准备麦克风…");
                    }
                    Err(error) => self.omni_status(cx, &error.to_string()),
                }
            }
            return;
        }
        if self.ui.button(cx, ids!(omni_stop)).clicked(actions) {
            let state = self.omni.as_mut().unwrap();
            if state.busy {
                state.engine.cancel();
                state.stopping = true;
                self.omni_status(cx, "正在停止…");
            }
            return;
        }
        if self.ui.button(cx, ids!(omni_choose_model)).clicked(actions)
            || self.ui.button(cx, ids!(omni_add_media)).clicked(actions)
        {
            let model = self.ui.button(cx, ids!(omni_choose_model)).clicked(actions);
            let Some(dialog) = self.runtime.dialog("lfm3-omni") else {
                return;
            };
            let state = self.omni.as_mut().unwrap();
            state.dialog = Some(dialog);
            if model {
                state.picking = Some(Picking::Model);
                crate::engine_ui::choose_directory(
                    cx,
                    Path::new(&self.ui.text_input(cx, ids!(omni_model)).text()),
                    "选择 LFM3-Omni-2B 模型目录",
                );
            } else {
                state.picking = Some(Picking::Media);
                cx.open_select_file_dialog(
                    FileDialog::new()
                        .set_title("选择图片、音频或视频".into())
                        .add_filter(
                            "媒体".into(),
                            vec![
                                "png".into(),
                                "jpg".into(),
                                "jpeg".into(),
                                "webp".into(),
                                "wav".into(),
                                "mp3".into(),
                                "m4a".into(),
                                "flac".into(),
                                "ogg".into(),
                                "mp4".into(),
                                "mov".into(),
                                "mkv".into(),
                                "webm".into(),
                            ],
                        ),
                );
            }
            return;
        }
        if self.ui.button(cx, ids!(omni_save)).clicked(actions) {
            let Some(audio) = self.omni.as_ref().and_then(|s| s.audio.clone()) else {
                return;
            };
            let Some(dialog) = self.runtime.dialog("lfm3-omni") else {
                return;
            };
            let state = self.omni.as_mut().unwrap();
            state.dialog = Some(dialog);
            state.picking = Some(Picking::Save(crate::export::Snapshot::Audio(audio)));
            cx.open_save_file_dialog(
                FileDialog::new()
                    .set_title("保存对话语音".into())
                    .set_filename("omni.wav".into()),
            );
            return;
        }
        if self.ui.button(cx, ids!(omni_add_text)).clicked(actions) {
            let text = self.ui.text_input(cx, ids!(omni_draft)).text();
            if !text.trim().is_empty() {
                self.omni.as_mut().unwrap().parts.push(Part::Text(text));
                self.ui.text_input(cx, ids!(omni_draft)).set_text(cx, "");
                self.omni_parts(cx);
            }
        }
        if self.ui.button(cx, ids!(omni_clear_parts)).clicked(actions) {
            self.omni.as_mut().unwrap().parts.clear();
            self.omni_parts(cx);
        }
        if self.ui.button(cx, ids!(omni_mode)).clicked(actions) {
            let state = self.omni.as_mut().unwrap();
            state.mode = (state.mode + 1) % 4;
        }
        let command = if self.ui.button(cx, ids!(omni_load)).clicked(actions) {
            let path = PathBuf::from(self.ui.text_input(cx, ids!(omni_model)).text());
            let state = self.omni.as_mut().unwrap();
            state.ready = false;
            state.loaded_path = None;
            state.loading_path = Some(path.clone());
            Some(Command::Load(path))
        } else if self.ui.button(cx, ids!(omni_new)).clicked(actions) {
            Some(Command::Clear)
        } else if self.ui.button(cx, ids!(omni_play)).clicked(actions) {
            self.omni
                .as_ref()
                .and_then(|s| s.audio.clone())
                .map(Command::Replay)
        } else if self.ui.button(cx, ids!(omni_send)).clicked(actions) {
            let draft = self.ui.text_input(cx, ids!(omni_draft)).text();
            let temperature_text = self.ui.text_input(cx, ids!(omni_temperature)).text();
            let audio_temperature = if temperature_text.trim().is_empty() {
                None
            } else {
                match temperature_text.trim().parse::<f32>() {
                    Ok(value) if value.is_finite() && value >= 0.0 => Some(value),
                    _ => {
                        self.omni_status(cx, "语音温度必须是非负有限数值");
                        return;
                    }
                }
            };
            let top_k_text = self.ui.text_input(cx, ids!(omni_top_k)).text();
            let audio_top_k = if top_k_text.trim().is_empty() {
                None
            } else {
                match top_k_text.trim().parse::<usize>() {
                    Ok(value) => Some(value),
                    Err(_) => {
                        self.omni_status(cx, "top-k 必须是非负整数");
                        return;
                    }
                }
            };
            let state = self.omni.as_mut().unwrap();
            if state.loaded_path.as_ref()
                != Some(&PathBuf::from(
                    self.ui.text_input(cx, ids!(omni_model)).text(),
                ))
            {
                self.omni_status(cx, "模型目录已改变，请重新载入");
                return;
            }
            let mut parts = state.parts.clone();
            if !draft.trim().is_empty() {
                parts.push(Part::Text(draft.clone()));
            }
            if parts.is_empty() {
                self.omni_status(cx, "请输入文字或加入媒体");
                return;
            }
            state.pending_user = parts.iter().map(part_label).collect::<Vec<_>>().join(" → ");
            state.response.clear();
            state.submitted = Some(draft);
            Some(Command::Generate {
                parts,
                audio_temperature,
                audio_top_k,
                mode: [
                    OutputMode::Auto,
                    OutputMode::Text,
                    OutputMode::Audio,
                    OutputMode::Interleaved,
                ][state.mode],
            })
        } else {
            None
        };
        if let Some(command) = command {
            let state = self.omni.as_mut().unwrap();
            state.playing = matches!(command, Command::Replay(_));
            match state.engine.submit(command) {
                Ok(()) => {
                    state.busy = true;
                    state.stopping = false;
                    self.omni_status(cx, "正在准备…");
                    self.omni_transcript(cx);
                }
                Err(error) => {
                    state.playing = false;
                    state.submitted = None;
                    state.pending_user.clear();
                    self.omni_status(cx, &error.to_string());
                }
            }
        }
    }
}
