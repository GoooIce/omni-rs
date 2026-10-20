use crate::protocol::{self, Update};
use crate::{ActionButtonState, App};
use makepad_widgets::*;

#[cfg(target_arch = "wasm32")]
use crate::web_bridge;
#[cfg(not(target_arch = "wasm32"))]
use crate::{engine, model_location};

#[derive(Default)]
pub struct State {
    transcript: String,
    ready: bool,
    busy: bool,
    loading: bool,
    microphone: bool,
    stopping: bool,
    mode: usize,
    language_choice: String,
    controls: Option<(bool, bool, bool, bool, bool, bool)>,
    directory: String,
    #[cfg(not(target_arch = "wasm32"))]
    engine: Option<engine::Engine>,
    #[cfg(not(target_arch = "wasm32"))]
    model_dialog: Option<crate::desktop_runtime::Dialog>,
    #[cfg(not(target_arch = "wasm32"))]
    needs_directory: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pending_model_load: bool,
    #[cfg(not(target_arch = "wasm32"))]
    picking_model: bool,
    #[cfg(not(target_arch = "wasm32"))]
    picking_audio: bool,
    #[cfg(not(target_arch = "wasm32"))]
    remember_model: bool,
    #[cfg(not(target_arch = "wasm32"))]
    settings_path: Option<std::path::PathBuf>,
}

impl State {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn transcript(&self) -> &str {
        &self.transcript
    }

    #[cfg(target_arch = "wasm32")]
    pub fn busy(&self) -> bool {
        self.busy
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn choosing_model(&self) -> bool {
        self.picking_model
    }
}

impl App {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_transcription(&mut self, cx: &mut Cx, root: &std::path::Path) {
        self.transcription.settings_path = model_location::settings_path();
        let directory =
            model_location::initial_directory(root, self.transcription.settings_path.as_deref());
        self.transcription.directory = directory.to_string_lossy().into_owned();
        self.ui
            .text_input(cx, ids!(model_path))
            .set_text(cx, &self.transcription.directory);
        self.ui
            .label(cx, ids!(model_title))
            .set_text(cx, "转写模型");
        self.ui
            .label(cx, ids!(transcript_title))
            .set_text(cx, "转写文字");
        let sample = root.join("artifacts/Confucius4-R2T2-upstream/resources/test.wav");
        let sample = std::fs::canonicalize(&sample).unwrap_or(sample);
        self.set_audio_path_text(cx, &sample.to_string_lossy());
        self.refresh_audio_file(cx);
        self.transcription.language_choice = "自动".to_owned();
        self.refresh_language_choices(cx);
        self.refresh_speed_choices(cx);
        self.transcription.engine = Some(engine::Engine::spawn_shared(
            SignalToUI::set_ui_signal,
            self.runtime.clone(),
        ));
        self.ui.label(cx, ids!(backend)).set_text(
            cx,
            if cfg!(target_os = "macos") {
                "Metal · 本机推理"
            } else {
                "WGPU · 本机推理"
            },
        );
        self.start_model_load(cx);
    }

    #[cfg(target_arch = "wasm32")]
    pub fn open_transcription(&mut self, cx: &mut Cx) {
        self.transcription.mode = 1;
        self.transcription.directory = "/models/web/manifest.json".to_owned();
        self.ui
            .button(cx, ids!(speed_balanced))
            .set_visible(cx, false);
        self.ui
            .text_input(cx, ids!(model_path))
            .set_text(cx, &self.transcription.directory);
        self.set_audio_path_text(cx, "/artifacts/Confucius4-R2T2-upstream/resources/test.wav");
        self.ui
            .label(cx, ids!(backend))
            .set_text(cx, "WebGPU · 浏览器推理");
        self.ui.button(cx, ids!(sample)).set_text(cx, "选择音频");
        self.ui
            .button(cx, ids!(choose_audio))
            .set_visible(cx, false);
        self.refresh_audio_file(cx);
        self.transcription.language_choice = "自动".to_owned();
        self.refresh_language_choices(cx);
        self.refresh_speed_choices(cx);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn resume_transcription_model(&mut self, cx: &mut Cx) {
        if self.transcription.needs_directory {
            self.choose_model(cx);
        } else if self.transcription.pending_model_load {
            self.start_model_load(cx);
        }
    }

    pub fn cancel_transcription(&mut self, cx: &mut Cx) {
        if !self.transcription.busy || self.transcription.stopping {
            return;
        }
        self.transcription.stopping = true;
        self.submit_transcription(cx, "stop");
        self.status(cx, "正在打断识别 · 保留已确认文字…");
    }

    pub fn poll_transcription(&mut self, cx: &mut Cx, event: &Event) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = event;
            let updates: Vec<_> = self
                .transcription
                .engine
                .as_ref()
                .map(|engine| engine.updates.try_iter().collect())
                .unwrap_or_default();
            for update in updates {
                self.apply_update(cx, update);
            }
        }
        #[cfg(target_arch = "wasm32")]
        if let Event::ToWasmMsg(message) = event {
            use makepad_widgets::makepad_platform::makepad_wasm_bridge::ToWasm;
            if message.id == web_bridge::ToWasmUpdate::live_id() {
                let message = web_bridge::ToWasmUpdate::read_to_wasm(&mut message.as_ref());
                match serde_json::from_str::<Update>(&message.json) {
                    Ok(update) if update.kind == "file" => {
                        self.set_audio_path_text(cx, &update.text);
                        self.refresh_audio_file(cx);
                    }
                    Ok(update) => self.apply_update(cx, update),
                    Err(error) => self.apply_update(
                        cx,
                        Update::status("error", format!("消息解析失败：{error}")),
                    ),
                }
            }
        }
    }

    pub fn refresh_transcription_controls(&mut self, cx: &mut Cx) {
        let busy = self.transcription.busy || self.inference_busy();
        let state = (
            self.transcription.busy,
            self.transcription.loading,
            self.transcription.ready,
            self.transcription.transcript.is_empty(),
            self.transcription.stopping,
            self.inference_busy(),
        );
        if self.transcription.controls == Some(state) {
            return;
        }
        self.transcription.controls = Some(state);
        self.ui.button(cx, ids!(load)).set_action_enabled(cx, !busy);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.ui
                .button(cx, ids!(choose_model))
                .set_action_enabled(cx, !busy);
            self.ui
                .button(cx, ids!(retry_model))
                .set_action_enabled(cx, !busy);
        }
        self.ui
            .button(cx, ids!(transcribe))
            .set_action_enabled(cx, self.transcription.ready && !busy);
        self.ui
            .button(cx, ids!(sample))
            .set_action_enabled(cx, !busy);
        self.ui
            .button(cx, ids!(choose_audio))
            .set_action_enabled(cx, !busy);
        for id in [
            ids!(asr_lang_auto),
            ids!(asr_lang_zh),
            ids!(asr_lang_en),
            ids!(asr_lang_yue),
            ids!(asr_lang_ja),
            ids!(asr_lang_ko),
        ] {
            self.ui.button(cx, id).set_action_enabled(cx, !busy);
        }
        for id in [
            ids!(speed_balanced),
            ids!(speed_low),
            ids!(speed_fast),
            ids!(speed_save),
        ] {
            self.ui.button(cx, id).set_action_enabled(cx, !busy);
        }
        self.ui
            .button(cx, ids!(microphone))
            .set_action_enabled(cx, self.transcription.ready && !busy);
        self.ui
            .button(cx, ids!(stop))
            .set_visible(cx, self.transcription.busy);
        self.ui
            .button(cx, ids!(stop))
            .set_action_enabled(cx, self.transcription.busy && !self.transcription.stopping);
        self.ui
            .button(cx, ids!(copy))
            .set_action_enabled(cx, !self.transcription.transcript.is_empty());
    }

    fn status(&self, cx: &mut Cx, text: &str) {
        let shown = text.replacen("模型已就绪", "转写已就绪", 1);
        self.ui.label(cx, ids!(status)).set_text(cx, &shown);
    }

    fn show_input_status(&self, cx: &mut Cx, text: &str) {
        self.ui
            .view(cx, ids!(input_status_row))
            .set_visible(cx, true);
        self.ui.label(cx, ids!(input_status)).set_text(cx, text);
    }

    fn audio_path_value(&self, cx: &mut Cx) -> String {
        self.ui
            .text_input(cx, ids!(audio_path))
            .text()
            .replace(['\n', '\r'], "")
    }

    fn set_audio_path_text(&self, cx: &mut Cx, path: &str) {
        let clean: Vec<char> = path
            .chars()
            .filter(|ch| *ch != '\n' && *ch != '\r')
            .collect();
        let mut split_at = None;
        let mut best_tail = 0usize;
        for (index, ch) in clean.iter().enumerate() {
            if *ch != '/' || index == 0 {
                continue;
            }
            let tail = clean.len() - index - 1;
            if tail > best_tail && tail <= 48 {
                best_tail = tail;
                split_at = Some(index);
            }
        }
        let shown = if let Some(index) = split_at {
            let head: String = clean[..=index].iter().collect();
            let tail: String = clean[index + 1..].iter().collect();
            format!("{head}\n{tail}")
        } else {
            clean.into_iter().collect()
        };
        self.ui
            .text_input(cx, ids!(audio_path))
            .set_text(cx, &shown);
    }

    fn refresh_audio_file(&self, cx: &mut Cx) {
        let path = self.audio_path_value(cx);
        let trimmed = path.trim();
        let name = if trimmed.is_empty() {
            "尚未选择音频".to_owned()
        } else {
            std::path::Path::new(trimmed)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| trimmed.to_owned())
        };
        self.ui.label(cx, ids!(audio_file)).set_text(cx, &name);
        self.ui.label(cx, ids!(audio_path_name)).set_text(cx, &name);
    }

    fn refresh_speed_choices(&self, cx: &mut Cx) {
        #[cfg(not(target_arch = "wasm32"))]
        let choices = [
            (ids!(speed_balanced), 0usize),
            (ids!(speed_low), 1),
            (ids!(speed_fast), 2),
            (ids!(speed_save), 3),
        ];
        #[cfg(target_arch = "wasm32")]
        let choices = [
            (ids!(speed_low), 0usize),
            (ids!(speed_fast), 1),
            (ids!(speed_save), 2),
        ];
        for (id, mode) in choices {
            let name = protocol::mode_label(mode);
            let label = if self.transcription.mode == mode {
                format!("● {name}")
            } else {
                name.to_owned()
            };
            self.ui.button(cx, id).set_text(cx, &label);
        }
        self.ui
            .label(cx, ids!(speed_hint))
            .set_text(cx, protocol::mode_hint(self.transcription.mode));
    }

    fn asr_language_raw(&self, cx: &mut Cx) -> String {
        let custom = self.ui.text_input(cx, ids!(asr_language)).text();
        let custom = custom.trim();
        if !custom.is_empty() {
            return custom.to_string();
        }
        if self.transcription.language_choice.is_empty() {
            "自动".to_owned()
        } else {
            self.transcription.language_choice.clone()
        }
    }

    fn refresh_language_choices(&self, cx: &mut Cx) {
        let raw = self.asr_language_raw(cx);
        let selected = match protocol::asr_language(&raw) {
            Ok(None) => Some("自动"),
            Ok(Some(name)) => match name.as_str() {
                "Chinese" => Some("中文"),
                "English" => Some("英文"),
                "Cantonese" => Some("粤语"),
                "Japanese" => Some("日语"),
                "Korean" => Some("韩语"),
                _ => None,
            },
            Err(_) => None,
        };
        let caption = match selected {
            Some(_) => "语言".to_owned(),
            None if protocol::asr_language(&raw).is_err() => "语言 · 无法识别".to_owned(),
            None => format!("语言 · {}", raw.trim()),
        };
        self.ui
            .label(cx, ids!(asr_language_caption))
            .set_text(cx, &caption);
        for (id, name) in [
            (ids!(asr_lang_auto), "自动"),
            (ids!(asr_lang_zh), "中文"),
            (ids!(asr_lang_en), "英文"),
            (ids!(asr_lang_yue), "粤语"),
            (ids!(asr_lang_ja), "日语"),
            (ids!(asr_lang_ko), "韩语"),
        ] {
            let label = if selected == Some(name) {
                format!("● {name}")
            } else {
                name.to_owned()
            };
            self.ui.button(cx, id).set_text(cx, &label);
        }
    }

    fn set_transcript_language(&self, cx: &mut Cx, language: &str) {
        let title = if language.is_empty() {
            "转写文字".to_owned()
        } else {
            format!("转写文字 · {language}")
        };
        self.ui
            .label(cx, ids!(transcript_title))
            .set_text(cx, &title);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn set_model_progress(&self, cx: &mut Cx, progress: f64) {
        crate::engine_ui::set_progress(
            &self.ui,
            cx,
            ids!(model_progress),
            ids!(model_progress_fill),
            progress,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_model_load(&mut self, cx: &mut Cx) {
        if self.runtime.busy() {
            self.transcription.pending_model_load = true;
            self.transcription.busy = false;
            self.transcription.loading = false;
            return;
        }
        self.transcription.pending_model_load = false;
        self.transcription.needs_directory = false;
        self.transcription.busy = true;
        self.transcription.loading = true;
        self.transcription.ready = false;
        self.ui
            .view(cx, ids!(model_recovery))
            .set_visible(cx, false);
        self.set_model_progress(cx, 0.0);
        self.ui.view(cx, ids!(model_progress)).set_visible(cx, true);
        self.status(cx, "正在加载 Confucius4-R2T2…");
        self.submit_transcription(cx, "load");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn choose_audio(&mut self, cx: &mut Cx) {
        if self.transcription.picking_model || self.transcription.picking_audio {
            return;
        }
        let Some(dialog) = self.runtime.dialog("asr") else {
            return;
        };
        self.transcription.model_dialog = Some(dialog);
        self.transcription.picking_audio = true;
        self.transcription.busy = true;
        let current = std::path::PathBuf::from(self.audio_path_value(cx));
        let mut dialog = FileDialog::new()
            .set_title("选择要转写的 WAV".into())
            .add_filter("WAV 音频".into(), vec!["wav".into()]);
        if let Some(parent) = current.parent().filter(|path| path.is_dir()) {
            dialog = dialog.set_location(parent.to_owned());
        }
        cx.open_select_file_dialog(dialog);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn choose_model(&mut self, cx: &mut Cx) {
        if self.transcription.picking_model || self.transcription.picking_audio {
            return;
        }
        let Some(dialog) = self.runtime.dialog("asr") else {
            return;
        };
        self.transcription.model_dialog = Some(dialog);
        self.transcription.needs_directory = false;
        self.transcription.picking_model = true;
        self.transcription.busy = true;
        self.transcription.loading = true;
        let current = std::path::PathBuf::from(self.ui.text_input(cx, ids!(model_path)).text());
        crate::engine_ui::choose_directory(
            cx,
            &current,
            "选择 Confucius4-R2T2 模型目录（包含 config.json、tokenizer.json 和 safetensors 权重）",
        );
    }

    fn apply_update(&mut self, cx: &mut Cx, update: Update) {
        match update.kind.as_str() {
            "ready" => {
                #[cfg(not(target_arch = "wasm32"))]
                let asr_visible = self.workspace.language();
                #[cfg(target_arch = "wasm32")]
                let asr_visible = true;
                if asr_visible && !update.backend.is_empty() {
                    self.ui
                        .label(cx, ids!(backend))
                        .set_text(cx, &update.backend);
                }
                self.transcription.ready = true;
                self.transcription.busy = false;
                self.transcription.loading = false;
                self.status(cx, &update.message);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.ui
                        .view(cx, ids!(model_progress))
                        .set_visible(cx, false);
                    self.ui
                        .view(cx, ids!(model_recovery))
                        .set_visible(cx, false);
                    if self.transcription.remember_model {
                        self.transcription.remember_model = false;
                        let path = self.ui.text_input(cx, ids!(model_path)).text();
                        let result = self
                            .transcription
                            .settings_path
                            .as_deref()
                            .ok_or_else(|| "找不到用户设置目录".to_owned())
                            .and_then(|settings| {
                                model_location::remember(settings, std::path::Path::new(&path))
                            });
                        if let Err(error) = result {
                            self.status(cx, &format!("模型已就绪 · 目录未能保存：{error}"));
                        }
                    }
                }
            }
            "loading" | "progress" | "notice" => {
                self.status(cx, &update.message);
                #[cfg(not(target_arch = "wasm32"))]
                if update.kind == "loading" {
                    self.set_model_progress(cx, update.progress);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            "missing" => {
                self.transcription.ready = false;
                self.status(cx, &update.message);
                self.ui
                    .view(cx, ids!(model_progress))
                    .set_visible(cx, false);
                self.ui.view(cx, ids!(model_recovery)).set_visible(cx, true);
                self.transcription.busy = false;
                self.transcription.loading = false;
                self.transcription.needs_directory = true;
            }
            "capture" => {
                if self.transcription.microphone && self.transcription.busy {
                    let bars = (update.level.clamp(0.0, 1.0).sqrt() * 12.0).round() as usize;
                    self.show_input_status(
                        cx,
                        &format!(
                            "{} · 已采集 {:.1}s · 待处理 {:.1}s · [{}{}]",
                            update.message,
                            update.captured_seconds,
                            update.backlog_seconds,
                            "|".repeat(bars),
                            "·".repeat(12 - bars)
                        ),
                    );
                }
            }
            "event" | "done" => {
                // Avoid relaying out the full transcript on empty deltas.
                // Engine messages still preserve every committed delta.
                if cfg!(target_arch = "wasm32") || self.transcription.transcript != update.text {
                    self.transcription.transcript = update.text;
                    self.ui
                        .text_input(cx, ids!(transcript))
                        .set_text(cx, &self.transcription.transcript);
                }
                if !update.language.is_empty() {
                    self.set_transcript_language(cx, &update.language);
                }
                self.ui.label(cx, ids!(metrics)).set_text(
                    cx,
                    &format!(
                        "音频 {:.2} s  ·  推理 {:.2} s  ·  RTF {:.3}  ·  {} 段  ·  本段 {:.0} ms",
                        update.audio_seconds,
                        update.total_compute_ms / 1000.0,
                        update.rtf,
                        update.event_count,
                        update.compute_ms
                    ),
                );
                if update.kind == "done" {
                    self.transcription.busy = false;
                    self.transcription.stopping = false;
                    if self.transcription.microphone {
                        self.show_input_status(
                            cx,
                            &format!("录音已结束 · 已采集 {:.2} s", update.captured_seconds),
                        );
                    }
                    self.transcription.microphone = false;
                    self.status(
                        cx,
                        if !update.message.is_empty() {
                            &update.message
                        } else if update.cancelled {
                            "已停止 · 保留已识别内容"
                        } else {
                            "转写完成 · 可再次运行观察预热后的速度"
                        },
                    );
                } else {
                    self.status(
                        cx,
                        if self.transcription.stopping {
                            "正在打断识别 · 保留已确认文字…"
                        } else if self.transcription.microphone {
                            "正在录音并转写 · 点击停止结束录音"
                        } else {
                            "正在转写 · 界面可以继续操作"
                        },
                    );
                }
            }
            "error" | "fatal" | "load_cancelled" => {
                self.transcription.busy = false;
                if self.transcription.loading || update.kind == "fatal" {
                    self.transcription.ready = false;
                }
                self.transcription.loading = false;
                self.transcription.stopping = false;
                if self.transcription.microphone {
                    self.show_input_status(cx, "麦克风已关闭");
                }
                self.transcription.microphone = false;
                self.status(cx, &format!("错误：{}", update.message));
                #[cfg(not(target_arch = "wasm32"))]
                if !self.transcription.ready {
                    self.ui
                        .view(cx, ids!(model_progress))
                        .set_visible(cx, false);
                    self.ui.view(cx, ids!(model_recovery)).set_visible(cx, true);
                }
            }
            _ => (),
        }
    }

    pub fn submit_transcription(&mut self, cx: &mut Cx, command: &str) {
        self.transcription.directory = self.ui.text_input(cx, ids!(model_path)).text();
        let path = self.transcription.directory.clone();
        let audio = self.audio_path_value(cx);
        let context = self.ui.text_input(cx, ids!(context)).text();
        let language = if command == "transcribe" || command == "microphone" {
            match protocol::asr_language(&self.asr_language_raw(cx)) {
                Ok(language) => language,
                Err(error) => {
                    self.transcription.busy = false;
                    self.transcription.microphone = false;
                    self.status(cx, &error);
                    return;
                }
            }
        } else {
            None
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(engine) = &self.transcription.engine else {
                return;
            };
            if command == "stop" {
                engine.cancel();
                return;
            }
            let request = if command == "load" {
                engine::Command::Load(path)
            } else if command == "microphone" {
                engine::Command::Microphone {
                    context,
                    options: protocol::decode_options(self.transcription.mode),
                    language,
                }
            } else {
                engine::Command::Transcribe {
                    path: audio,
                    context,
                    options: protocol::decode_options(self.transcription.mode),
                    language,
                }
            };
            if let Err(error) = engine.submit(request) {
                self.apply_update(
                    cx,
                    Update::status("error", format!("推理线程不可用：{error}")),
                );
            }
        }
        #[cfg(target_arch = "wasm32")]
        cx.os.from_wasm(web_bridge::FromWasmCommand {
            json: serde_json::json!({
                "type": command, "url": path, "audio": audio, "context": context,
                "config": protocol::decode_options(self.transcription.mode),
                "language": language,
            })
            .to_string(),
        });
    }

    pub fn handle_transcription_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            for action in actions {
                if !self.transcription.picking_model && !self.transcription.picking_audio {
                    continue;
                }
                match action.downcast_ref::<FileDialogAction>() {
                    Some(FileDialogAction::FileSelected { paths, .. })
                        if self.transcription.picking_audio =>
                    {
                        self.transcription.model_dialog.take();
                        self.transcription.picking_audio = false;
                        self.transcription.busy = false;
                        if let Some(path) = paths.first() {
                            self.set_audio_path_text(cx, &path.to_string_lossy());
                            self.refresh_audio_file(cx);
                            self.status(
                                cx,
                                &format!(
                                    "已选择 {} · 可以开始转写",
                                    self.ui.label(cx, ids!(audio_file)).text()
                                ),
                            );
                        }
                    }
                    Some(FileDialogAction::FileCancelled { .. })
                        if self.transcription.picking_audio =>
                    {
                        self.transcription.model_dialog.take();
                        self.transcription.picking_audio = false;
                        self.transcription.busy = false;
                    }
                    Some(FileDialogAction::FolderSelected(path))
                        if self.transcription.picking_model =>
                    {
                        self.transcription.model_dialog.take();
                        self.transcription.picking_model = false;
                        let directory = model_location::selected_directory(path);
                        self.transcription.directory = directory.to_string_lossy().into_owned();
                        self.ui
                            .text_input(cx, ids!(model_path))
                            .set_text(cx, &self.transcription.directory);
                        self.transcription.remember_model = true;
                        self.transcription.ready = false;
                        self.transcription.busy = false;
                        self.transcription.loading = false;
                        self.transcription.pending_model_load = true;
                        if self.workspace.language() && !self.inference_busy() {
                            self.start_model_load(cx);
                        }
                    }
                    Some(FileDialogAction::FolderCancelled) if self.transcription.picking_model => {
                        self.transcription.model_dialog.take();
                        self.transcription.picking_model = false;
                        self.transcription.busy = false;
                        self.transcription.loading = false;
                        self.status(cx, "尚未选择模型 · 选择模型目录后自动加载");
                    }
                    _ => (),
                }
            }
            if !self.workspace.language() {
                return;
            }
            if self.ui.button(cx, ids!(choose_audio)).clicked(actions) && !self.transcription.busy {
                self.choose_audio(cx);
            }
            if self.ui.button(cx, ids!(choose_model)).clicked(actions) && !self.transcription.busy {
                self.choose_model(cx);
            }
            if self.ui.button(cx, ids!(retry_model)).clicked(actions) && !self.transcription.busy {
                self.start_model_load(cx);
            }
        }
        let speed_choice = [
            (ids!(speed_balanced), 0usize),
            (
                ids!(speed_low),
                if cfg!(target_arch = "wasm32") { 0 } else { 1 },
            ),
            (
                ids!(speed_fast),
                if cfg!(target_arch = "wasm32") { 1 } else { 2 },
            ),
            (
                ids!(speed_save),
                if cfg!(target_arch = "wasm32") { 2 } else { 3 },
            ),
        ];
        for (id, mode) in speed_choice {
            if self.ui.button(cx, id).clicked(actions)
                && !self.transcription.busy
                && !self.inference_busy()
            {
                if mode < protocol::mode_count() {
                    self.transcription.mode = mode;
                    self.refresh_speed_choices(cx);
                }
            }
        }
        if self.ui.button(cx, ids!(ping)).clicked(actions) {
            self.pings += 1;
            self.ui.label(cx, ids!(heartbeat)).set_text(
                cx,
                &format!("界面响应 {} 次 · 心跳 {}", self.pings, self.ticks),
            );
        }
        if self.ui.button(cx, ids!(copy)).clicked(actions) {
            #[cfg(not(target_arch = "wasm32"))]
            cx.copy_to_clipboard(&self.transcription.transcript);
            // The pinned upstream Web backend leaves CopyToClipboard unhandled.
            #[cfg(target_arch = "wasm32")]
            cx.os.from_wasm(web_bridge::FromWasmCommand {
                json: serde_json::json!({"type": "copy", "text": self.transcription.transcript})
                    .to_string(),
            });
        }
        if self.ui.button(cx, ids!(stop)).clicked(actions) {
            self.cancel_transcription(cx);
        }
        if self
            .ui
            .text_input(cx, ids!(audio_path))
            .changed(actions)
            .is_some()
        {
            self.refresh_audio_file(cx);
        }
        if let Some(text) = self.ui.text_input(cx, ids!(asr_language)).changed(actions) {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                self.refresh_language_choices(cx);
            } else if let Some(choice) = ["自动", "中文", "英文", "粤语", "日语", "韩语"]
                .into_iter()
                .find(|choice| *choice == trimmed)
            {
                self.transcription.language_choice = choice.to_owned();
                self.ui.text_input(cx, ids!(asr_language)).set_text(cx, "");
                self.refresh_language_choices(cx);
            } else {
                self.refresh_language_choices(cx);
                if let Err(error) = protocol::asr_language(trimmed) {
                    self.status(cx, &error);
                }
            }
        }
        for (id, value) in [
            (ids!(asr_lang_auto), "自动"),
            (ids!(asr_lang_zh), "中文"),
            (ids!(asr_lang_en), "英文"),
            (ids!(asr_lang_yue), "粤语"),
            (ids!(asr_lang_ja), "日语"),
            (ids!(asr_lang_ko), "韩语"),
        ] {
            if self.ui.button(cx, id).clicked(actions)
                && !self.transcription.busy
                && !self.inference_busy()
            {
                self.transcription.language_choice = value.to_owned();
                self.ui.text_input(cx, ids!(asr_language)).set_text(cx, "");
                self.refresh_language_choices(cx);
            }
        }
        if self.ui.button(cx, ids!(sample)).clicked(actions) && !self.transcription.busy {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let sample = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../artifacts/Confucius4-R2T2-upstream/resources/test.wav");
                let sample = std::fs::canonicalize(&sample).unwrap_or(sample);
                self.set_audio_path_text(cx, &sample.to_string_lossy());
                self.refresh_audio_file(cx);
                self.status(cx, "已填入官方示例 · 可以开始转写");
            }
            #[cfg(target_arch = "wasm32")]
            self.submit_transcription(cx, "choose");
        }
        #[cfg(target_arch = "wasm32")]
        if self.ui.button(cx, ids!(load)).clicked(actions) && !self.transcription.busy {
            self.transcription.busy = true;
            self.transcription.loading = true;
            self.transcription.ready = false;
            self.status(cx, "正在载入模型…");
            self.submit_transcription(cx, "load");
        }
        if self.ui.button(cx, ids!(transcribe)).clicked(actions)
            && !self.transcription.busy
            && !self.inference_busy()
        {
            if !self.transcription.ready {
                self.status(cx, "请先载入模型");
                return;
            }
            if let Err(error) = protocol::asr_language(&self.asr_language_raw(cx)) {
                self.status(cx, &error);
                return;
            }
            self.transcription.busy = true;
            self.transcription.transcript.clear();
            self.ui.text_input(cx, ids!(transcript)).set_text(cx, "");
            self.set_transcript_language(cx, "");
            self.status(cx, "开始转写 · 首次运行可能需要较长时间");
            self.submit_transcription(cx, "transcribe");
        }
        if self.ui.button(cx, ids!(microphone)).clicked(actions)
            && !self.transcription.busy
            && !self.inference_busy()
            && self.transcription.ready
        {
            if let Err(error) = protocol::asr_language(&self.asr_language_raw(cx)) {
                self.status(cx, &error);
                return;
            }
            self.transcription.busy = true;
            self.transcription.microphone = true;
            self.transcription.stopping = false;
            self.transcription.transcript.clear();
            self.ui.text_input(cx, ids!(transcript)).set_text(cx, "");
            self.set_transcript_language(cx, "");
            self.show_input_status(cx, "正在打开默认输入设备…");
            self.status(cx, "正在开启麦克风 · 首次使用请允许录音权限");
            self.submit_transcription(cx, "microphone");
        }
    }
}
