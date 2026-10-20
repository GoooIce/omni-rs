use crate::ActionButtonState;
use crate::{
    model_location,
    translation_engine::{Command, Engine},
    App,
};
use makepad_widgets::*;
use omni_hy_mt::{Options, Request};
use std::path::{Path, PathBuf};

pub struct State {
    engine: Engine,
    dialog: Option<crate::desktop_runtime::Dialog>,
    path: PathBuf,
    settings: Option<PathBuf>,
    ready: bool,
    pub busy: bool,
    picking: bool,
    stopping: bool,
    next_id: u64,
    current: Option<(u64, usize)>,
    pending: Option<Request>,
    results: [String; 2],
}
impl State {
    pub fn ready(&self) -> bool {
        self.ready
    }
    pub fn running(&self) -> Option<bool> {
        (self.busy && !self.picking).then_some(self.stopping)
    }
    pub fn new(root: &Path, runtime: crate::desktop_runtime::Runtime) -> Self {
        let settings = model_location::settings_path_for("HY_MT_SETTINGS_PATH", "hy-mt.json");
        let path = model_location::initial_directory_for(
            root,
            settings.as_deref(),
            "HY_MT_MODEL_DIR",
            "models/Hy-MT2-7B",
        );
        Self {
            engine: Engine::spawn_shared(SignalToUI::set_ui_signal, runtime),
            dialog: None,
            path,
            settings,
            ready: false,
            busy: false,
            picking: false,
            stopping: false,
            next_id: 0,
            current: None,
            pending: None,
            results: Default::default(),
        }
    }
}
impl App {
    pub fn stop_translation(&mut self, cx: &mut Cx) {
        let Some(state) = self.translation.as_mut() else {
            return;
        };
        if state.running().is_none() {
            return;
        }
        state.stopping = true;
        let pending = state.pending.take().is_some();
        state.engine.cancel();
        if pending && state.ready {
            state.busy = false;
            state.stopping = false;
            state.current = None;
        }
        self.translation_status(cx, "正在停止翻译…");
    }
    fn translation_status(&self, cx: &mut Cx, message: &str) {
        self.ui
            .view(cx, ids!(asr_translation_note))
            .set_visible(cx, !message.is_empty());
        for id in [ids!(asr_translation_status), ids!(tts_translation_status)] {
            self.ui.label(cx, id).set_text(cx, message);
        }
    }
    pub fn refresh_translation_controls(&mut self, cx: &mut Cx) {
        let other_busy = self.runtime.busy() || self.runtime.dialog_busy();
        let Some(state) = self.translation.as_ref() else {
            return;
        };
        let idle = !state.busy && !other_busy;
        self.ui.button(cx, ids!(asr_translate)).set_text(
            cx,
            if state.ready {
                "翻译"
            } else {
                "载入翻译模型"
            },
        );
        self.ui.label(cx, ids!(asr_translation_readiness)).set_text(
            cx,
            if state.ready {
                "翻译上面的转写 · 模型已就绪"
            } else {
                "翻译上面的转写 · 模型尚未载入"
            },
        );
        self.ui.button(cx, ids!(asr_translate)).set_action_enabled(
            cx,
            idle && (!state.ready || !self.transcription.transcript().is_empty()),
        );
        self.ui
            .button(cx, ids!(tts_translate))
            .set_action_enabled(cx, idle);
        for id in [ids!(asr_translation_stop), ids!(tts_translation_stop)] {
            self.ui.button(cx, id).set_visible(cx, state.busy);
            self.ui
                .button(cx, id)
                .set_action_enabled(cx, state.busy && !state.stopping && !state.picking);
        }
        for id in [ids!(asr_translation_model), ids!(tts_translation_model)] {
            self.ui.button(cx, id).set_visible(cx, state.ready);
            self.ui.button(cx, id).set_action_enabled(cx, idle);
        }
        let asr_result = !state.results[0].is_empty();
        self.ui
            .view(cx, ids!(asr_translation_well))
            .set_visible(cx, asr_result);
        self.ui
            .button(cx, ids!(asr_translation_copy))
            .set_visible(cx, asr_result);
        self.ui
            .button(cx, ids!(asr_translation_to_tts))
            .set_visible(cx, asr_result);
        self.ui
            .button(cx, ids!(asr_translation_copy))
            .set_action_enabled(cx, asr_result);
        let tts_result = !state.results[1].is_empty();
        self.ui
            .button(cx, ids!(tts_translation_use))
            .set_visible(cx, tts_result);
        self.ui
            .button(cx, ids!(tts_translation_use))
            .set_action_enabled(cx, idle && tts_result);
    }
    fn load_translation_model(&mut self, cx: &mut Cx) {
        let state = self.translation.as_mut().unwrap();
        if state.ready || state.busy {
            return;
        }
        state.next_id += 1;
        state.current = Some((state.next_id, 0));
        state.busy = true;
        state.stopping = false;
        state.pending = None;
        let command = Command::Load(state.path.clone());
        if let Err(error) = state.engine.submit(command) {
            state.busy = false;
            state.current = None;
            self.translation_status(cx, &error.to_string());
        } else {
            self.translation_status(cx, "正在载入翻译模型…");
        }
    }
    fn submit_translation(&mut self, cx: &mut Cx, origin: usize, request: Request) {
        if let Err(error) = request.validate() {
            self.translation_status(cx, &error.to_string());
            return;
        }
        let state = self.translation.as_mut().unwrap();
        state.next_id += 1;
        state.current = Some((state.next_id, origin));
        state.busy = true;
        state.stopping = false;
        let command = if state.ready {
            Command::Translate {
                job_id: state.next_id,
                request,
            }
        } else {
            state.pending = Some(request);
            Command::Load(state.path.clone())
        };
        if let Err(error) = state.engine.submit(command) {
            state.busy = false;
            state.pending = None;
            state.current = None;
            self.translation_status(cx, &error.to_string());
        } else {
            self.translation_status(cx, "正在准备翻译 · 首次使用会载入 Hy-MT2-7B…");
        }
    }
    pub fn handle_translation_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        if self.translation.is_none() {
            return;
        }
        for action in actions {
            if !self.translation.as_ref().unwrap().picking {
                break;
            }
            match action.downcast_ref::<FileDialogAction>() {
                Some(FileDialogAction::FolderSelected(path)) => {
                    let state = self.translation.as_mut().unwrap();
                    state.dialog.take();
                    state.picking = false;
                    state.busy = false;
                    state.path = model_location::selected_directory(path);
                    state.ready = false;
                    self.translation_status(cx, "翻译目录已选择 · 点击翻译以载入模型");
                }
                Some(FileDialogAction::FolderCancelled) => {
                    let state = self.translation.as_mut().unwrap();
                    state.dialog.take();
                    state.picking = false;
                    state.busy = false;
                }
                _ => {}
            }
        }
        if self
            .ui
            .button(cx, ids!(asr_translation_copy))
            .clicked(actions)
        {
            cx.copy_to_clipboard(&self.translation.as_ref().unwrap().results[0]);
        }
        if self
            .ui
            .button(cx, ids!(asr_translation_stop))
            .clicked(actions)
            || self
                .ui
                .button(cx, ids!(tts_translation_stop))
                .clicked(actions)
        {
            self.stop_translation(cx);
            return;
        }
        if self.runtime.busy()
            || self.runtime.dialog_busy()
            || self.translation.as_ref().is_some_and(|s| s.busy)
        {
            return;
        }
        if self
            .ui
            .button(cx, ids!(asr_translation_model))
            .clicked(actions)
            || self
                .ui
                .button(cx, ids!(tts_translation_model))
                .clicked(actions)
        {
            let state = self.translation.as_mut().unwrap();
            let Some(dialog) = self.runtime.dialog("translation") else {
                return;
            };
            state.dialog = Some(dialog);
            state.picking = true;
            state.busy = true;
            crate::engine_ui::choose_directory(cx, &state.path, "选择 Hy-MT2-7B 模型目录");
            return;
        }
        if self
            .ui
            .button(cx, ids!(tts_translation_use))
            .clicked(actions)
        {
            let text = &self.translation.as_ref().unwrap().results[1];
            if !text.is_empty() {
                self.ui.text_input(cx, ids!(tts_text)).set_text(cx, text);
            }
        }
        if self.ui.button(cx, ids!(asr_translate)).clicked(actions)
            && !self.translation.as_ref().is_some_and(|state| state.ready())
        {
            self.load_translation_model(cx);
            return;
        }
        let origin = if self.ui.button(cx, ids!(asr_translate)).clicked(actions) {
            0
        } else if self.ui.button(cx, ids!(tts_translate)).clicked(actions) {
            1
        } else {
            return;
        };
        let text = if origin == 0 {
            self.transcription.transcript().to_owned()
        } else {
            self.ui.text_input(cx, ids!(tts_text)).text()
        };
        let target_language = self
            .ui
            .text_input(
                cx,
                if origin == 0 {
                    ids!(asr_translation_language)
                } else {
                    ids!(tts_translation_language)
                },
            )
            .text();
        self.submit_translation(
            cx,
            origin,
            Request {
                text,
                target_language,
                options: Options::default(),
            },
        );
    }
    pub fn poll_translation(&mut self, cx: &mut Cx) {
        if self.translation.is_none() {
            return;
        }
        let updates: Vec<_> = self
            .translation
            .as_ref()
            .map(|s| s.engine.updates.try_iter().collect())
            .unwrap_or_default();
        for update in updates {
            let state = self.translation.as_mut().unwrap();
            if update.job_id != 0 && state.current.is_none_or(|(id, _)| id != update.job_id) {
                continue;
            }
            match update.kind {
                "ready" => {
                    state.ready = true;
                    if let Some(settings) = &state.settings {
                        let _ = model_location::remember(settings, &state.path);
                    }
                    if state.pending.is_none() {
                        state.busy = false;
                        state.current = None;
                    }
                    if state.stopping {
                        state.stopping = false;
                        self.translation_status(cx, "翻译已停止 · 模型已就绪");
                        continue;
                    }
                }
                "translation" | "done" | "cancelled" => {
                    if let Some((_, origin)) = state.current {
                        if update.job_id != 0 {
                            state.results[origin] = update.text.clone();
                            self.ui
                                .text_input(
                                    cx,
                                    if origin == 0 {
                                        ids!(asr_translation)
                                    } else {
                                        ids!(tts_translation)
                                    },
                                )
                                .set_text(cx, &update.text);
                        }
                    }
                    if update.kind != "translation" {
                        state.busy = false;
                        state.stopping = false;
                        state.current = None;
                        state.pending = None;
                    }
                }
                "missing" | "error" | "fatal" | "load_cancelled" => {
                    state.busy = false;
                    state.stopping = false;
                    state.current = None;
                    state.pending = None;
                    if update.kind != "error" {
                        state.ready = false;
                    }
                }
                _ => {}
            }
            self.translation_status(cx, &update.message);
        }
        if !self.runtime.busy() {
            let state = self.translation.as_mut().unwrap();
            if state.ready && !state.stopping {
                if let (Some(request), Some((job_id, _))) = (state.pending.as_ref(), state.current)
                {
                    match state.engine.submit(Command::Translate {
                        job_id,
                        request: request.clone(),
                    }) {
                        Ok(()) => {
                            state.pending = None;
                        }
                        Err(crate::engine_harness::SubmitError::Busy) => {}
                        Err(error) => {
                            state.pending = None;
                            state.current = None;
                            state.busy = false;
                            self.translation_status(cx, &error.to_string());
                        }
                    }
                }
            }
        }
    }
}
