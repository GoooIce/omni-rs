use crate::ActionButtonState;
use crate::{
    music_engine::{Command, Engine, Job, Output, Song},
    App,
};
use makepad_widgets::*;
use omni_music::{Planning, Sampling, SongRequest};
use std::{
    path::{Path, PathBuf},
    sync::{mpsc::Receiver, Arc},
};

enum Picking {
    Audio,
    Models,
    Save(crate::export::Snapshot),
}
#[derive(Clone, PartialEq, Eq)]
struct Draft {
    source: String,
    abc: String,
    melody_only: bool,
    use_score: bool,
}
pub struct State {
    engine: Engine,
    settings: Option<PathBuf>,
    dialog: Option<crate::desktop_runtime::Dialog>,
    picking: Option<Picking>,
    exporting: Option<Receiver<Result<String, String>>>,
    busy: bool,
    playing: bool,
    stopping: bool,
    melody_only: bool,
    use_score: bool,
    source: String,
    submitted: Option<Draft>,
    submitted_root: Option<PathBuf>,
    analysis: Option<Arc<[u8]>>,
    score: Option<omni_music::score::Score>,
    song: Option<Song>,
    controls: Option<[bool; 10]>,
}
impl State {
    pub fn new(runtime: crate::desktop_runtime::Runtime) -> Self {
        Self {
            engine: Engine::spawn_shared(SignalToUI::set_ui_signal, runtime),
            settings: crate::model_location::settings_path_for("MUSIC_SETTINGS_PATH", "music.json"),
            dialog: None,
            picking: None,
            exporting: None,
            busy: false,
            playing: false,
            stopping: false,
            melody_only: true,
            use_score: false,
            source: String::new(),
            submitted: None,
            submitted_root: None,
            analysis: None,
            score: None,
            song: None,
            controls: None,
        }
    }
    pub fn playing(&self) -> bool {
        self.playing
    }
    pub fn stop_playback(&mut self) {
        if self.playing {
            self.engine.cancel();
            self.stopping = true;
        }
    }
}

impl App {
    pub fn initialize_music(&mut self, cx: &mut Cx, root: &Path) {
        let state = State::new(self.runtime.clone());
        let path = crate::model_location::initial_directory_for(
            root,
            state.settings.as_deref(),
            "MUSIC_MODELS_DIR",
            "models",
        );
        self.ui
            .text_input(cx, ids!(music_models))
            .set_text(cx, &path.to_string_lossy());
        self.music = Some(state);
    }
    fn music_draft(&self, cx: &mut Cx) -> Draft {
        let m = self.music.as_ref().unwrap();
        Draft {
            source: self.ui.text_input(cx, ids!(music_source)).text(),
            abc: self.ui.text_input(cx, ids!(music_score)).text(),
            melody_only: m.melody_only,
            use_score: m.use_score,
        }
    }
    fn music_status(&self, cx: &mut Cx, text: &str) {
        self.ui.label(cx, ids!(music_status)).set_text(cx, text);
    }
    pub fn refresh_music_controls(&mut self, cx: &mut Cx) {
        let inference_busy = self.inference_busy();
        let dialog_busy = self.runtime.dialog_busy();
        let abc = !self
            .ui
            .text_input(cx, ids!(music_score))
            .text()
            .trim()
            .is_empty();
        let source = !self
            .ui
            .text_input(cx, ids!(music_source))
            .text()
            .trim()
            .is_empty();
        let Some(m) = self.music.as_mut() else {
            return;
        };
        let idle = !m.busy && !inference_busy && !dialog_busy;
        let saving = m.exporting.is_some() || dialog_busy;
        let state = [
            idle,
            m.busy,
            m.stopping,
            saving,
            abc,
            source,
            m.analysis.is_some(),
            m.score.is_some(),
            m.song.is_some(),
            m.playing,
        ];
        if m.controls == Some(state) {
            return;
        }
        m.controls = Some(state);
        self.ui
            .button(cx, ids!(music_generate))
            .set_action_enabled(cx, idle);
        for id in [ids!(music_analyze), ids!(music_transcribe)] {
            self.ui
                .button(cx, id)
                .set_action_enabled(cx, idle && source);
        }
        for id in [ids!(music_choose_audio), ids!(music_choose_models)] {
            self.ui.button(cx, id).set_action_enabled(cx, !dialog_busy);
        }
        self.ui.button(cx, ids!(music_stop)).set_visible(cx, m.busy);
        self.ui
            .button(cx, ids!(music_stop))
            .set_action_enabled(cx, m.busy && !m.stopping);
        self.ui
            .button(cx, ids!(music_save_analysis))
            .set_action_enabled(cx, m.analysis.is_some() && !saving);
        self.ui
            .button(cx, ids!(music_save_abc))
            .set_action_enabled(cx, abc && !saving);
        for id in [ids!(music_save_midi), ids!(music_save_score)] {
            self.ui
                .button(cx, id)
                .set_action_enabled(cx, m.score.is_some() && !saving);
        }
        self.ui
            .button(cx, ids!(music_restore_score))
            .set_action_enabled(cx, m.score.as_ref().is_some_and(|s| s.abc.is_some()));
        for id in [ids!(music_save_audio), ids!(music_save_song)] {
            self.ui
                .button(cx, id)
                .set_action_enabled(cx, m.song.is_some() && !saving);
        }
        self.ui
            .button(cx, ids!(music_replay))
            .set_action_enabled(cx, m.song.is_some() && !m.busy && !dialog_busy);
    }
    fn music_mode_labels(&self, cx: &mut Cx) {
        let m = self.music.as_ref().unwrap();
        self.ui.button(cx, ids!(music_melody)).set_text(
            cx,
            if m.melody_only {
                "旋律"
            } else {
                "旋律与和弦"
            },
        );
        self.ui.button(cx, ids!(music_use_score)).set_text(
            cx,
            if m.use_score {
                "使用此乐谱：开"
            } else {
                "使用此乐谱：关"
            },
        );
    }
    pub fn poll_music(&mut self, cx: &mut Cx) {
        if let Some(m) = self.music.as_mut() {
            if let Some(result) = m.exporting.as_ref().and_then(|rx| rx.try_recv().ok()) {
                m.exporting = None;
                self.ui
                    .label(cx, ids!(music_export_status))
                    .set_text(cx, &result.unwrap_or_else(|e| format!("保存失败：{e}")));
            }
        }
        let updates: Vec<_> = self
            .music
            .as_ref()
            .map(|m| m.engine.updates.try_iter().collect())
            .unwrap_or_default();
        for update in updates {
            let draft = self.music_draft(cx);
            let m = self.music.as_mut().unwrap();
            if matches!(update.kind, "progress" | "playing") {
                if !m.stopping {
                    self.music_status(cx, &update.message);
                }
                crate::engine_ui::set_progress(
                    &self.ui,
                    cx,
                    ids!(music_progress),
                    ids!(music_progress_fill),
                    update.progress,
                );
                continue;
            }
            m.busy = false;
            m.playing = false;
            m.stopping = false;
            m.controls = None;
            self.ui
                .view(cx, ids!(music_progress))
                .set_visible(cx, false);
            let input = m.submitted.take();
            let current = input.as_ref() == Some(&draft);
            let source = input
                .as_ref()
                .map(|d| {
                    Path::new(&d.source)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default();
            if let Some(output) = update.output {
                match output {
                    Output::Analysis(a) => {
                        self.ui.label(cx, ids!(music_analysis)).set_text(
                            cx,
                            &format!(
                                "{source} · {:.2} 秒 · {} 帧 × {} 维",
                                a.audio_seconds, a.frames, a.dimensions
                            ),
                        );
                        m.analysis = Some(
                            serde_json::to_vec_pretty(&a)
                                .expect("serializable analysis")
                                .into(),
                        );
                    }
                    Output::Score(score) => {
                        let mut notes = vec![format!(
                            "{source} · {} 个音乐事件{}",
                            score.events.len(),
                            if score.truncated {
                                " · 达到长度上限"
                            } else {
                                ""
                            }
                        )];
                        if let Some(abc) = &score.abc {
                            if current {
                                self.ui.text_input(cx, ids!(music_score)).set_text(cx, abc);
                                m.use_score = true;
                                notes.push("乐谱已填入，请检查后生成歌曲。".into());
                            } else {
                                notes.push(
                                    "输入已改变，当前编辑已保留。可点“载入转谱结果”。".into(),
                                );
                            }
                        } else {
                            notes.push(format!(
                                "ABC 转换未完成：{}。可保存转谱 MIDI 和事件。",
                                score.abc_error.as_deref().unwrap_or("缺少有效节拍或调性")
                            ));
                        }
                        notes.extend(score.warnings.iter().cloned());
                        self.ui
                            .text_input(cx, ids!(music_score_status))
                            .set_text(cx, &notes.join("\n"));
                        m.score = Some(score);
                    }
                    Output::Song(song) => {
                        if current && !draft.use_score {
                            if let Some(abc) = &song.semantic.plan.abc {
                                self.ui.text_input(cx, ids!(music_score)).set_text(cx, abc);
                                m.melody_only = false;
                                self.ui.text_input(cx, ids!(music_score_status)).set_text(
                                    cx,
                                    if song.semantic.plan.truncated {
                                        "创作乐谱已达上限，请补全后再打开“使用此乐谱”。"
                                    } else {
                                        "已填入创作乐谱，可编辑后打开“使用此乐谱”再次生成。"
                                    },
                                );
                            }
                        }
                        self.ui.label(cx, ids!(music_song_status)).set_text(
                            cx,
                            &format!("{} · 耗时 {:.1} 秒", update.message, song.elapsed_seconds),
                        );
                        m.song = Some(song);
                    }
                }
                if let Some(root) = m.submitted_root.take() {
                    if let Some(settings) = &m.settings {
                        if let Err(error) = crate::model_location::remember(settings, &root) {
                            self.ui
                                .label(cx, ids!(music_export_status))
                                .set_text(cx, &format!("模型目录未能保存：{error}"));
                        }
                    }
                }
            } else {
                m.submitted_root = None;
            }
            self.music_mode_labels(cx);
            self.music_status(
                cx,
                &if matches!(update.kind, "error" | "fatal" | "missing") {
                    format!("失败：{}", update.message)
                } else if update.kind == "cancelled" {
                    "已停止 · 保留已完成的结果".into()
                } else {
                    update.message
                },
            );
        }
    }
    fn submit_music(&mut self, cx: &mut Cx, job: Job) {
        let draft = self.music_draft(cx);
        let root = PathBuf::from(self.ui.text_input(cx, ids!(music_models)).text().trim());
        let m = self.music.as_mut().unwrap();
        match m.engine.submit(Command::Run {
            root: root.clone(),
            job: Box::new(job),
        }) {
            Ok(()) => {
                m.busy = true;
                m.stopping = false;
                m.submitted = Some(draft);
                m.submitted_root = Some(root);
                self.ui.view(cx, ids!(music_progress)).set_visible(cx, true);
                crate::engine_ui::set_progress(
                    &self.ui,
                    cx,
                    ids!(music_progress),
                    ids!(music_progress_fill),
                    0.0,
                );
                self.music_status(cx, "准备音乐任务 · 可以继续编辑或切换页面");
            }
            Err(error) => self.music_status(cx, &error.to_string()),
        }
    }
    fn save_music(&mut self, cx: &mut Cx, snapshot: crate::export::Snapshot, name: &str) {
        let m = self.music.as_mut().unwrap();
        if m.exporting.is_some() {
            return;
        }
        let Some(dialog) = self.runtime.dialog("music") else {
            return;
        };
        m.dialog = Some(dialog);
        m.picking = Some(Picking::Save(snapshot));
        cx.open_save_file_dialog(
            FileDialog::new()
                .set_title("保存音乐结果".into())
                .set_filename(name.into()),
        );
    }
    pub fn handle_music_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        if self.music.is_none() {
            return;
        }
        for action in actions {
            let m = self.music.as_mut().unwrap();
            let Some(action) = action.downcast_ref::<FileDialogAction>() else {
                continue;
            };
            match action {
                FileDialogAction::FileSelected { paths, .. }
                    if matches!(m.picking, Some(Picking::Audio)) =>
                {
                    if let Some(path) = paths.first() {
                        self.ui
                            .text_input(cx, ids!(music_source))
                            .set_text(cx, &path.to_string_lossy());
                    }
                }
                FileDialogAction::FolderSelected(path)
                    if matches!(m.picking, Some(Picking::Models)) =>
                {
                    self.ui
                        .text_input(cx, ids!(music_models))
                        .set_text(cx, &path.to_string_lossy());
                }
                FileDialogAction::SaveFileSelected { path, .. }
                    if matches!(m.picking, Some(Picking::Save(_))) =>
                {
                    if let Some(Picking::Save(snapshot)) = m.picking.take() {
                        m.exporting = Some(snapshot.save(path.clone(), SignalToUI::set_ui_signal));
                    }
                }
                FileDialogAction::FileCancelled { .. }
                    if matches!(m.picking, Some(Picking::Audio)) => {}
                FileDialogAction::FolderCancelled if matches!(m.picking, Some(Picking::Models)) => {
                }
                FileDialogAction::SaveFileCancelled { .. }
                    if matches!(m.picking, Some(Picking::Save(_))) => {}
                _ => continue,
            }
            m.picking = None;
            m.dialog = None;
        }
        let source = self.ui.text_input(cx, ids!(music_source)).text();
        let m = self.music.as_mut().unwrap();
        if source != m.source {
            m.source = source;
            m.use_score = false;
        }
        self.music_mode_labels(cx);
        if !self.workspace.creative() {
            return;
        }
        if self.ui.button(cx, ids!(music_stop)).clicked(actions) {
            let m = self.music.as_mut().unwrap();
            if m.busy {
                m.engine.cancel();
                m.stopping = true;
                self.music_status(cx, "正在停止 · 等待当前操作结束…");
            }
            return;
        }
        for (id, models) in [
            (ids!(music_choose_audio), false),
            (ids!(music_choose_models), true),
        ] {
            if self.ui.button(cx, id).clicked(actions) {
                let Some(dialog) = self.runtime.dialog("music") else {
                    return;
                };
                let m = self.music.as_mut().unwrap();
                m.dialog = Some(dialog);
                if models {
                    m.picking = Some(Picking::Models);
                    let path = self.ui.text_input(cx, ids!(music_models)).text();
                    crate::engine_ui::choose_directory(
                        cx,
                        Path::new(&path),
                        "选择包含音乐模型的父目录",
                    );
                } else {
                    m.picking = Some(Picking::Audio);
                    cx.open_select_file_dialog(
                        FileDialog::new()
                            .set_title("选择参考音乐 WAV".into())
                            .add_filter("WAV 音频".into(), vec!["wav".into()]),
                    );
                }
                return;
            }
        }
        let m = self.music.as_mut().unwrap();
        if self.ui.button(cx, ids!(music_melody)).clicked(actions) {
            m.melody_only = !m.melody_only;
            m.use_score = false;
        }
        if self.ui.button(cx, ids!(music_use_score)).clicked(actions) {
            m.use_score = !m.use_score;
        }
        if self
            .ui
            .button(cx, ids!(music_restore_score))
            .clicked(actions)
        {
            if let Some(score) = &m.score {
                if let Some(abc) = &score.abc {
                    self.ui.text_input(cx, ids!(music_score)).set_text(cx, abc);
                    m.melody_only = score.melody_only;
                    m.use_score = true;
                }
            }
        }
        self.music_mode_labels(cx);
        let m = self.music.as_ref().unwrap();
        let save = if self
            .ui
            .button(cx, ids!(music_save_analysis))
            .clicked(actions)
        {
            m.analysis
                .clone()
                .map(|b| (crate::export::Snapshot::Bytes(b), "analysis.json"))
        } else if self.ui.button(cx, ids!(music_save_abc)).clicked(actions) {
            let text = self.ui.text_input(cx, ids!(music_score)).text();
            (!text.trim().is_empty()).then(|| {
                (
                    crate::export::Snapshot::Bytes(text.into_bytes().into()),
                    "score.abc",
                )
            })
        } else if self.ui.button(cx, ids!(music_save_midi)).clicked(actions) {
            m.score.as_ref().map(|s| {
                (
                    crate::export::Snapshot::Bytes(s.midi().into()),
                    "transcription.mid",
                )
            })
        } else if self.ui.button(cx, ids!(music_save_score)).clicked(actions) {
            m.score.as_ref().map(|s| {
                (
                    crate::export::Snapshot::Bytes(serde_json::to_vec_pretty(s).unwrap().into()),
                    "transcription.json",
                )
            })
        } else if self.ui.button(cx, ids!(music_save_audio)).clicked(actions) {
            m.song
                .as_ref()
                .map(|s| (crate::export::Snapshot::Music(s.audio.clone()), "song.wav"))
        } else if self.ui.button(cx, ids!(music_save_song)).clicked(actions) {
            m.song.as_ref().map(|s| {
                let value = serde_json::json!({"sample_rate":48000,"channels":2,"audio_seconds":s.audio.len() as f64 / 96000.0,
                    "elapsed_seconds":s.elapsed_seconds,"score_truncated":s.semantic.plan.truncated,"music_truncated":s.semantic.truncated,
                    "semantic":s.semantic});
                (crate::export::Snapshot::Bytes(serde_json::to_vec_pretty(&value).unwrap().into()), "song.json")
            })
        } else {
            None
        };
        if let Some((snapshot, name)) = save {
            self.save_music(cx, snapshot, name);
            return;
        }
        if self.ui.button(cx, ids!(music_replay)).clicked(actions)
            && !m.busy
            && !self.runtime.dialog_busy()
        {
            if let Some(song) = &m.song {
                if let Some(t) = &mut self.tts {
                    t.stop_playback();
                }
                let audio = song.audio.clone();
                let m = self.music.as_mut().unwrap();
                match m.engine.submit(Command::Replay(audio)) {
                    Ok(()) => {
                        m.busy = true;
                        m.playing = true;
                        m.stopping = false;
                    }
                    Err(error) => self.music_status(cx, &error.to_string()),
                }
            }
            return;
        }
        if self.inference_busy() || self.runtime.dialog_busy() || self.music.as_ref().unwrap().busy
        {
            return;
        }
        let draft = self.music_draft(cx);
        if self.ui.button(cx, ids!(music_analyze)).clicked(actions) {
            self.submit_music(cx, Job::Analyze(draft.source.trim().into()));
        } else if self.ui.button(cx, ids!(music_transcribe)).clicked(actions) {
            self.submit_music(
                cx,
                Job::Transcribe {
                    path: draft.source.trim().into(),
                    melody_only: draft.melody_only,
                },
            );
        } else if self.ui.button(cx, ids!(music_generate)).clicked(actions) {
            let request = song_request(
                self.ui.text_input(cx, ids!(music_style)).text(),
                self.ui.text_input(cx, ids!(music_lyrics)).text(),
                &draft,
                &self.ui.text_input(cx, ids!(music_seconds)).text(),
                &self.ui.text_input(cx, ids!(music_seed)).text(),
            );
            match request {
                Ok(request) => self.submit_music(cx, Job::Generate(request)),
                Err(error) => self.music_status(cx, &error),
            }
        }
    }
}

fn song_request(
    style: String,
    lyrics: String,
    draft: &Draft,
    seconds: &str,
    seed: &str,
) -> Result<SongRequest, String> {
    if style.trim().is_empty() {
        return Err("请填写歌曲风格".into());
    }
    if draft.use_score && draft.abc.trim().is_empty() {
        return Err("请先转谱或填写 ABC 乐谱，再打开使用乐谱".into());
    }
    let seconds: usize = seconds
        .trim()
        .parse()
        .ok()
        .filter(|s| (1..=480).contains(s))
        .ok_or("时长上限请输入 1–480 秒的整数")?;
    let seed: u64 = seed
        .trim()
        .parse()
        .ok()
        .filter(|&s| s < 1 << 63)
        .ok_or("随机种子请输入 0–9223372036854775807 的整数")?;
    let request = SongRequest {
        style,
        lyrics,
        seed,
        abc: draft.use_score.then(|| draft.abc.clone()),
        cot: if draft.use_score && draft.melody_only {
            Planning::Melody
        } else {
            Planning::Full
        },
        semantic_sampling: Sampling {
            max_tokens: seconds * 25,
            min_tokens: 200.min(seconds * 25),
            ..Sampling::default()
        },
        ..SongRequest::default()
    };
    request.validate().map_err(|e| e.to_string())?;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn requests_snapshot_the_selected_score_mode_and_bound_duration() {
        let mut draft = Draft {
            source: "in.wav".into(),
            abc: "X:1\nK:C\nCDEF".into(),
            melody_only: true,
            use_score: false,
        };
        let request = song_request("piano".into(), "".into(), &draft, "2", "41").unwrap();
        assert_eq!(request.abc, None);
        assert_eq!(request.cot, Planning::Full);
        assert_eq!(request.semantic_sampling.min_tokens, 50);
        draft.use_score = true;
        let request = song_request("piano".into(), "".into(), &draft, "60", "41").unwrap();
        draft.abc.clear();
        assert!(request.abc.unwrap().starts_with("X:1"));
        assert_eq!(request.cot, Planning::Melody);
        assert!(song_request("piano".into(), "".into(), &draft, "60", "41").is_err());
        for seconds in ["0", "481", "NaN", "2.5"] {
            assert!(song_request("piano".into(), "".into(), &draft, seconds, "41").is_err());
        }
    }
}
