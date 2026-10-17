use crate::ActionButtonState;
use crate::{
    image_engine::{Command, Engine},
    model_location, App,
};
use makepad_widgets::*;
use omni_qwen_image::{dimensions, Options};
use std::path::{Path, PathBuf};

pub struct State {
    engine: Engine,
    dialog: Option<crate::desktop_runtime::Dialog>,
    needs_directory: bool,
    output: Option<std::sync::Arc<omni_qwen_image::GeneratedImage>>,
    saving_snapshot: Option<crate::export::Snapshot>,
    exporting: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    path: PathBuf,
    settings: Option<PathBuf>,
    initialized: bool,
    ready: bool,
    busy: bool,
    loading: bool,
    stopping: bool,
    picking: bool,
    picking_reference: bool,
    references: Vec<PathBuf>,
    saving: bool,
    remember: bool,
    has_image: bool,
    advanced: bool,
    cfg_enabled: bool,
    observation: bool,
    observation_board: ObservationBoard,
    controls: Option<(
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
    )>,
}
impl State {
    pub fn new(root: &Path, runtime: crate::desktop_runtime::Runtime) -> Self {
        let settings =
            model_location::settings_path_for("QWEN_IMAGE_SETTINGS_PATH", "qwen-image.json");
        let path = model_location::initial_directory_for(
            root,
            settings.as_deref(),
            "QWEN_IMAGE_MODEL_DIR",
            "models/Qwen-Image-2.1-Turbo",
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
            loading: false,
            stopping: false,
            picking: false,
            picking_reference: false,
            references: Vec::new(),
            saving: false,
            remember: false,
            has_image: false,
            advanced: false,
            cfg_enabled: false,
            observation: false,
            observation_board: ObservationBoard::new(0, 0),
            controls: None,
        }
    }
}

impl App {
    pub fn activate_image(&mut self, cx: &mut Cx) {
        if self.inference_busy() || self.runtime.dialog_busy() {
            return;
        }
        if self
            .image
            .as_ref()
            .is_some_and(|i| i.needs_directory && !i.busy)
        {
            self.choose_image(cx);
        } else if self.image.as_ref().is_some_and(|i| !i.ready && !i.busy) {
            self.load_image(cx);
        }
    }
    fn image_status(&self, cx: &mut Cx, message: &str) {
        self.ui.label(cx, ids!(image_status)).set_text(cx, message);
    }
    fn update_image_dimensions_hint(&self, cx: &mut Cx) {
        let width = self.ui.text_input(cx, ids!(image_width)).text();
        let height = self.ui.text_input(cx, ids!(image_height)).text();
        let message = match (width.trim().parse(), height.trim().parse()) {
            (Ok(w), Ok(h)) => match dimensions::resolve(w, h) {
                Ok((rw, rh)) if (w, h) != (rw, rh) => {
                    format!("按官方规则向下对齐：{rw} × {rh}")
                }
                _ => "自定义 256–2752 · 总像素不超过 430 万".into(),
            },
            _ => "自定义 256–2752 · 总像素不超过 430 万".into(),
        };
        self.ui
            .label(cx, ids!(image_dimensions_hint))
            .set_text(cx, &message);
    }
    fn image_progress(&self, cx: &mut Cx, progress: f64) {
        crate::engine_ui::set_progress(
            &self.ui,
            cx,
            ids!(image_progress),
            ids!(image_progress_fill),
            progress,
        );
    }
    fn image_error(&mut self, cx: &mut Cx, message: &str, fatal: bool) {
        let i = self.image.as_mut().unwrap();
        if fatal || i.loading {
            i.ready = false;
        }
        i.busy = false;
        i.loading = false;
        i.stopping = false;
        self.ui
            .view(cx, ids!(image_progress))
            .set_visible(cx, false);
        self.ui
            .view(cx, ids!(image_recovery))
            .set_visible(cx, !i.ready);
        self.image_status(cx, message);
    }
    fn load_image(&mut self, cx: &mut Cx) {
        if self.runtime.busy() {
            return;
        }
        let i = self.image.as_mut().unwrap();
        i.needs_directory = false;
        i.initialized = true;
        i.ready = false;
        i.busy = true;
        i.loading = true;
        i.stopping = false;
        let result = i.engine.submit(Command::Load(i.path.clone()));
        self.ui.view(cx, ids!(image_progress)).set_visible(cx, true);
        self.ui
            .view(cx, ids!(image_recovery))
            .set_visible(cx, false);
        self.image_progress(cx, 0.0);
        self.image_status(cx, "正在加载 Qwen-Image-2.1-Turbo…");
        if let Err(error) = result {
            self.image_error(cx, &error.to_string(), true);
        }
    }
    fn choose_image(&mut self, cx: &mut Cx) {
        let i = self.image.as_mut().unwrap();
        if i.picking {
            return;
        }
        let Some(dialog) = self.runtime.dialog("image") else {
            return;
        };
        i.dialog = Some(dialog);
        i.needs_directory = false;
        i.picking = true;
        i.busy = true;
        crate::engine_ui::choose_directory(cx, &i.path, "选择 Qwen-Image-2.1-Turbo 完整模型目录");
    }
    pub fn refresh_image_controls(&mut self, cx: &mut Cx) {
        let other_busy = self.inference_busy();
        let dialog_busy = self.runtime.dialog_busy();
        let Some(i) = self.image.as_mut() else {
            return;
        };
        let state = (
            i.busy,
            i.loading,
            i.stopping,
            i.ready,
            i.has_image,
            other_busy,
            dialog_busy,
            self.workspace.creative(),
            self.workspace.language(),
            i.picking || i.saving || i.picking_reference,
            i.observation,
        );
        if i.controls == Some(state) {
            return;
        }
        i.controls = Some(state);
        let idle = !other_busy && !i.busy && !dialog_busy;
        self.ui
            .button(cx, ids!(image_load))
            .set_action_enabled(cx, idle && !i.ready);
        self.ui
            .button(cx, ids!(image_generate))
            .set_action_enabled(cx, idle && i.ready);
        self.ui.button(cx, ids!(image_stop)).set_action_enabled(
            cx,
            i.busy && !i.stopping && !i.picking && !i.picking_reference,
        );
        self.ui.button(cx, ids!(image_save)).set_action_enabled(
            cx,
            i.output.is_some() && !i.saving && !self.runtime.dialog_busy(),
        );
        for id in [
            ids!(image_choose),
            ids!(image_retry),
            ids!(image_example),
            ids!(image_reference_choose),
            ids!(image_settings_toggle),
            ids!(image_cfg_toggle),
            ids!(image_observe_toggle),
            ids!(image_random_seed),
            ids!(image_size_square),
            ids!(image_size_portrait),
            ids!(image_size_landscape),
            ids!(image_size_photo_landscape),
            ids!(image_size_photo_portrait),
            ids!(image_size_widescreen),
            ids!(image_size_vertical),
        ] {
            self.ui.button(cx, id).set_action_enabled(cx, idle);
        }
        self.ui
            .button(cx, ids!(image_reference_clear))
            .set_action_enabled(cx, idle && !i.references.is_empty());
        self.ui.button(cx, ids!(image_generate)).set_text(
            cx,
            if i.observation {
                "观察去噪"
            } else if i.references.is_empty() {
                "生成图像"
            } else {
                "编辑图像"
            },
        );
    }
    pub fn poll_image(&mut self, cx: &mut Cx) {
        if let Some(i) = self.image.as_mut() {
            if let Some(result) = i.exporting.as_ref().and_then(|rx| rx.try_recv().ok()) {
                i.exporting = None;
                i.saving = false;
                i.controls = None;
                self.image_status(cx, &result.unwrap_or_else(|e| format!("保存失败：{e}")));
            }
        }
        let updates: Vec<_> = self
            .image
            .as_ref()
            .map(|i| i.engine.updates.try_iter().collect())
            .unwrap_or_default();
        for update in updates {
            let i = self.image.as_mut().unwrap();
            match update.kind {
                "reference_ready" => {
                    i.busy = false;
                    i.controls = None;
                    if let Some(reference) = update.reference {
                        i.references = reference.paths.clone();
                        let name = reference.paths[0]
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy();
                        let short: String = name.chars().take(22).collect();
                        self.ui.label(cx, ids!(image_reference_name)).set_text(
                            cx,
                            &format!(
                                "{} 张 · {}{}",
                                reference.paths.len(),
                                short,
                                if name.chars().count() > 22 { "…" } else { "" }
                            ),
                        );
                        let texture = Texture::new_with_format(
                            cx,
                            TextureFormat::VecBGRAu8_32 {
                                width: reference.width,
                                height: reference.height,
                                data: Some(reference.pixels.clone()),
                                updated: TextureUpdated::Full,
                            },
                        );
                        self.ui
                            .image(cx, ids!(image_reference_preview))
                            .set_texture(cx, Some(texture));
                    }
                    self.image_status(cx, &update.message);
                }
                "loading" | "progress" => {
                    if !i.stopping {
                        self.image_status(cx, &update.message);
                    }
                    self.image_progress(cx, update.progress);
                }
                "ready" => {
                    i.ready = true;
                    i.busy = false;
                    i.loading = false;
                    self.ui
                        .view(cx, ids!(image_progress))
                        .set_visible(cx, false);
                    let remembered = if i.remember {
                        i.remember = false;
                        i.settings
                            .as_deref()
                            .map(|p| model_location::remember(p, &i.path))
                            .transpose()
                    } else {
                        Ok(None)
                    };
                    self.image_status(
                        cx,
                        &match remembered {
                            Ok(_) => update.message,
                            Err(error) => format!("{} · 目录未能保存：{error}", update.message),
                        },
                    );
                }
                "missing" => {
                    i.ready = false;
                    i.loading = false;
                    self.ui
                        .view(cx, ids!(image_progress))
                        .set_visible(cx, false);
                    self.ui.view(cx, ids!(image_recovery)).set_visible(cx, true);
                    i.busy = false;
                    i.loading = false;
                    i.needs_directory = true;
                    self.image_status(cx, &update.message);
                }
                "observation" => {
                    let rejected = update.observation.as_ref().and_then(|cell| {
                        i.observation_board
                            .show(
                                cell.row,
                                cell.step,
                                cell.width as u32,
                                cell.height as u32,
                                &cell.rgba,
                            )
                            .err()
                    });
                    let stopping = i.stopping;
                    if !stopping {
                        self.image_status(cx, &update.message);
                    }
                    self.image_progress(cx, update.progress);
                    if let Some(error) = rejected {
                        self.image_status(cx, &error);
                    } else if update.observation.is_some() {
                        self.ui
                            .view(cx, ids!(image_preview_single))
                            .set_visible(cx, false);
                        self.ui
                            .view(cx, ids!(image_observation))
                            .set_visible(cx, true);
                        self.paint_observation(cx);
                    }
                }
                "done" => {
                    i.busy = false;
                    i.stopping = false;
                    i.has_image = true;
                    let observed = if update.image.is_none() {
                        let (width, height) = i.observation_board.canvas();
                        let official = i.observation_board.rows()[0].cells().len();
                        let minicpm = i.observation_board.rows()[1].cells().len();
                        (official + minicpm > 0).then_some((official, minicpm, width, height))
                    } else {
                        None
                    };
                    if let Some(result) = update.image {
                        i.output = Some(result.clone());
                        let r = &result.report;
                        self.ui.label(cx, ids!(image_metrics)).set_text(
                            cx,
                            &format!(
                                "{}文本 {:.2} s · 生成 {:.2} s · 解码 {:.2} s · 种子 {}{}",
                                if r.reference_count > 0 {
                                    format!("参考图 {:.2} s · ", r.reference_seconds)
                                } else {
                                    String::new()
                                },
                                r.text_seconds,
                                r.denoise_seconds,
                                r.decode_seconds,
                                r.seed,
                                if r.guidance_enabled {
                                    format!(" · CFG {}", r.true_cfg_scale)
                                } else {
                                    String::new()
                                }
                            ),
                        );
                        // Checkered matte only affects the preview; PNG retains its original alpha.
                        let pixels = result
                            .image
                            .enumerate_pixels()
                            .map(|(x, y, p)| {
                                let matte = if ((x / 16) + (y / 16)) % 2 == 0 {
                                    225u32
                                } else {
                                    195
                                };
                                let a = p[3] as u32;
                                let c = |v: u8| (v as u32 * a + matte * (255 - a) + 127) / 255;
                                0xff000000 | (c(p[0]) << 16) | (c(p[1]) << 8) | c(p[2])
                            })
                            .collect();
                        let texture = Texture::new_with_format(
                            cx,
                            TextureFormat::VecBGRAu8_32 {
                                width: r.width,
                                height: r.height,
                                data: Some(pixels),
                                updated: TextureUpdated::Full,
                            },
                        );
                        self.ui
                            .image(cx, ids!(image_preview))
                            .set_texture(cx, Some(texture));
                        self.ui
                            .view(cx, ids!(image_observation))
                            .set_visible(cx, false);
                        self.ui
                            .view(cx, ids!(image_preview_single))
                            .set_visible(cx, true);
                        self.ui
                            .label(cx, ids!(image_preview_caption))
                            .set_text(cx, "图像预览");
                        self.ui.view(cx, ids!(image_preview_group)).redraw(cx);
                    }
                    if let Some((official, minicpm, width, height)) = observed {
                        self.ui.label(cx, ids!(image_metrics)).set_text(
                            cx,
                            &format!(
                                "官方 {official} 格 · MiniCPM-V-4.7-1B {minicpm} 格 · 解码 {width}×{height}"
                            ),
                        );
                    }
                    self.ui
                        .view(cx, ids!(image_progress))
                        .set_visible(cx, false);
                    self.image_status(cx, &update.message);
                }
                "cancelled" => {
                    i.busy = false;
                    i.stopping = false;
                    self.ui
                        .view(cx, ids!(image_progress))
                        .set_visible(cx, false);
                    self.image_status(cx, &update.message);
                }
                "error" | "fatal" | "load_cancelled" => {
                    self.image_error(cx, &update.message, update.kind != "error")
                }
                _ => {}
            }
        }
    }
    pub fn handle_image_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        if self.image.is_none() {
            return;
        }
        let inference_busy = self.inference_busy();
        self.update_image_dimensions_hint(cx);
        for action in actions {
            let i = self.image.as_mut().unwrap();
            match action.downcast_ref::<FileDialogAction>() {
                Some(FileDialogAction::FileSelected { paths, .. }) if i.picking_reference => {
                    i.dialog.take();
                    i.picking_reference = false;
                    if let Err(error) = i.engine.submit(Command::References(paths.clone())) {
                        self.image_error(cx, &error.to_string(), false);
                    } else {
                        self.image_status(cx, "正在读取参考图片…");
                    }
                }
                Some(FileDialogAction::FileCancelled { .. }) if i.picking_reference => {
                    i.dialog.take();
                    i.picking_reference = false;
                    i.busy = false;
                    self.image_status(cx, "已取消选择参考图");
                }
                Some(FileDialogAction::FolderSelected(path)) if i.picking => {
                    i.dialog.take();
                    i.picking = false;
                    i.path = model_location::selected_directory(path);
                    i.remember = true;
                    i.ready = false;
                    i.initialized = false;
                    i.busy = false;
                    i.loading = false;
                    if self.workspace.creative() && !self.runtime.busy() {
                        self.load_image(cx);
                    }
                }
                Some(FileDialogAction::FolderCancelled) if i.picking => {
                    i.dialog.take();
                    i.picking = false;
                    i.busy = false;
                    i.loading = false;
                    self.image_status(cx, "尚未选择模型 · 选择目录后自动加载");
                }
                Some(FileDialogAction::SaveFileSelected { path, .. })
                    if i.saving && i.dialog.is_some() =>
                {
                    i.dialog.take();
                    i.exporting = i
                        .saving_snapshot
                        .take()
                        .map(|snapshot| snapshot.save(path.clone(), SignalToUI::set_ui_signal));
                    i.controls = None;
                }
                Some(FileDialogAction::SaveFileCancelled { .. })
                    if i.saving && i.dialog.is_some() =>
                {
                    i.dialog.take();
                    i.saving_snapshot = None;
                    i.saving = false;
                    i.controls = None;
                }
                _ => {}
            }
        }
        if !self.workspace.creative() {
            return;
        }
        let i = self.image.as_mut().unwrap();
        if self.ui.button(cx, ids!(image_stop)).clicked(actions)
            && i.busy
            && !i.picking
            && !i.picking_reference
        {
            i.stopping = true;
            i.engine.cancel();
            self.image_status(cx, "正在停止 · 等待当前 GPU 操作完成…");
            return;
        }
        if self.ui.button(cx, ids!(image_save)).clicked(actions) && i.output.is_some() && !i.saving
        {
            let Some(dialog) = self.runtime.dialog("image") else {
                return;
            };
            i.dialog = Some(dialog);
            i.saving_snapshot = i.output.clone().map(crate::export::Snapshot::Image);
            i.saving = true;
            i.controls = None;
            cx.open_save_file_dialog(
                FileDialog::new()
                    .set_title("保存生成的图像".into())
                    .set_filename("qwen-image.png".into())
                    .add_filter("PNG 图像".into(), vec!["png".into()]),
            );
            return;
        }
        if i.busy || inference_busy || self.runtime.dialog_busy() {
            return;
        }
        if self
            .ui
            .button(cx, ids!(image_settings_toggle))
            .clicked(actions)
        {
            i.advanced = !i.advanced;
            self.ui
                .view(cx, ids!(image_advanced))
                .set_visible(cx, i.advanced);
            self.ui.button(cx, ids!(image_settings_toggle)).set_text(
                cx,
                if i.advanced {
                    "收起参数"
                } else {
                    "更多参数"
                },
            );
            return;
        }
        if self
            .ui
            .button(cx, ids!(image_observe_toggle))
            .clicked(actions)
        {
            i.observation = !i.observation;
            i.controls = None;
            let observation = i.observation;
            self.ui.button(cx, ids!(image_observe_toggle)).set_text(
                cx,
                if observation {
                    "去噪观察 · 开启"
                } else {
                    "去噪观察 · 关闭"
                },
            );
            self.image_status(
                cx,
                if observation {
                    "去噪观察已开启 · 仅纯文本，且引导关闭"
                } else {
                    "去噪观察已关闭 · 只在结束时解码"
                },
            );
            return;
        }
        if self.ui.button(cx, ids!(image_cfg_toggle)).clicked(actions) {
            i.cfg_enabled = !i.cfg_enabled;
            self.ui.button(cx, ids!(image_cfg_toggle)).set_text(
                cx,
                if i.cfg_enabled {
                    "负向引导 · 开启"
                } else {
                    "负向引导 · 关闭"
                },
            );
            self.ui.label(cx, ids!(image_cfg_hint)).set_text(
                cx,
                if i.cfg_enabled {
                    "每步多算一次；留空为无条件引导，效果需试验。"
                } else {
                    "当前关闭，负向词不生效；Turbo 默认 CFG=1。"
                },
            );
            self.ui.label(cx, ids!(image_sampling_hint)).set_text(
                cx,
                if i.cfg_enabled {
                    "8 步 · 32 像素对齐 · 负向引导开启"
                } else {
                    "8 步 · 32 像素对齐 · 负向引导关闭"
                },
            );
            return;
        }
        for (id, ratio) in [
            (ids!(image_size_square), "1:1"),
            (ids!(image_size_portrait), "3:4"),
            (ids!(image_size_landscape), "4:3"),
            (ids!(image_size_photo_landscape), "3:2"),
            (ids!(image_size_photo_portrait), "2:3"),
            (ids!(image_size_widescreen), "16:9"),
            (ids!(image_size_vertical), "9:16"),
        ] {
            if self.ui.button(cx, id).clicked(actions) {
                let (width, height) = dimensions::preset(ratio).expect("official aspect ratio");
                self.ui
                    .text_input(cx, ids!(image_width))
                    .set_text(cx, &width.to_string());
                self.ui
                    .text_input(cx, ids!(image_height))
                    .set_text(cx, &height.to_string());
                self.update_image_dimensions_hint(cx);
                return;
            }
        }
        if self.ui.button(cx, ids!(image_random_seed)).clicked(actions) {
            use std::hash::BuildHasher;
            let seed = std::collections::hash_map::RandomState::new().hash_one(i.references.len());
            self.ui
                .text_input(cx, ids!(image_seed))
                .set_text(cx, &seed.to_string());
            return;
        }
        if self.ui.button(cx, ids!(image_choose)).clicked(actions) {
            self.choose_image(cx);
            return;
        }
        if self.ui.button(cx, ids!(image_retry)).clicked(actions) {
            self.load_image(cx);
            return;
        }
        if self.ui.button(cx, ids!(image_example)).clicked(actions) {
            self.ui.text_input(cx, ids!(image_prompt)).set_text(cx, if i.references.is_empty() {"一只红色陶瓷茶壶放在木桌上，窗外清晨的柔和阳光照进房间，写实摄影，细腻的陶瓷质感，简洁的构图。"} else {"把图中的主体改成蓝色，保持背景、光照和构图不变。"});
            return;
        }
        if self
            .ui
            .button(cx, ids!(image_reference_choose))
            .clicked(actions)
        {
            let Some(dialog) = self.runtime.dialog("image") else {
                return;
            };
            i.dialog = Some(dialog);
            i.picking_reference = true;
            i.busy = true;
            cx.open_select_file_dialog(
                FileDialog::new()
                    .set_title("选择 1–4 张参考图，按顺序用于编辑".into())
                    .set_multiple(true)
                    .add_filter(
                        "PNG / JPEG / WebP 图像".into(),
                        vec!["png".into(), "jpg".into(), "jpeg".into(), "webp".into()],
                    ),
            );
            return;
        }
        if self
            .ui
            .button(cx, ids!(image_reference_clear))
            .clicked(actions)
        {
            i.references.clear();
            i.controls = None;
            self.ui
                .image(cx, ids!(image_reference_preview))
                .set_texture(cx, None);
            self.ui
                .label(cx, ids!(image_reference_name))
                .set_text(cx, "未选择参考图 · 文生图");
            self.image_status(cx, "已清除参考图 · 输入画面描述即可生成");
            return;
        }

        if self.ui.button(cx, ids!(image_generate)).clicked(actions) && i.ready {
            let options = (|| -> Result<Options, String> {
                let mut o = Options {
                    prompt: self.ui.text_input(cx, ids!(image_prompt)).text(),
                    negative_prompt: i
                        .cfg_enabled
                        .then(|| self.ui.text_input(cx, ids!(image_negative_prompt)).text()),
                    observation: i.observation,
                    true_cfg_scale: if i.cfg_enabled {
                        let value: f32 = self
                            .ui
                            .text_input(cx, ids!(image_cfg_scale))
                            .text()
                            .trim()
                            .parse()
                            .map_err(|_| "引导强度须为大于 1 的数字")?;
                        if value <= 1.0 {
                            return Err(
                                "开启负向引导时强度须大于 1；默认快速生成请关闭负向引导".into()
                            );
                        }
                        value
                    } else {
                        1.0
                    },
                    width: self
                        .ui
                        .text_input(cx, ids!(image_width))
                        .text()
                        .trim()
                        .parse()
                        .map_err(|_| "宽度须为整数")?,
                    height: self
                        .ui
                        .text_input(cx, ids!(image_height))
                        .text()
                        .trim()
                        .parse()
                        .map_err(|_| "高度须为整数")?,
                    seed: self
                        .ui
                        .text_input(cx, ids!(image_seed))
                        .text()
                        .trim()
                        .parse()
                        .map_err(|_| "种子须为非负整数")?,
                    images: i.references.clone(),
                    ..Default::default()
                };
                let resolved = o.output_dimensions().map_err(|e| e.to_string())?;
                (o.width, o.height) = resolved;
                // Keep reference quality in step with the requested canvas; cap
                // at the model's default 1024² area for predictable memory use.
                let side = (o.width as f64 * o.height as f64)
                    .sqrt()
                    .min(1024.0)
                    // Area rounding of a non-square 256² reference can fall
                    // below the VL processor's minimum and change its grid.
                    .max(288.0);
                o.reference_resolution = (side / 32.0).round_ties_even() as usize * 32;
                let reference_resolution = self
                    .ui
                    .text_input(cx, ids!(image_reference_resolution))
                    .text();
                if !o.images.is_empty()
                    && !reference_resolution.trim().is_empty()
                    && reference_resolution.trim() != "自动"
                {
                    o.reference_resolution = reference_resolution
                        .trim()
                        .parse()
                        .map_err(|_| "参考图分辨率须为“自动”或 288–1024 内 32 的倍数")?;
                    if o.reference_resolution < 288 {
                        return Err("界面参考图分辨率至少为 288，以适应非正方形图片".into());
                    }
                }
                o.validate().map_err(|e| e.to_string())?;
                if o.observation {
                    omni_qwen_image::ObservationRequest::accept(&o).map_err(|e| e.to_string())?;
                }
                Ok(o)
            })();
            let options = match options {
                Ok(o) => o,
                Err(e) => {
                    self.image_status(cx, &e);
                    return;
                }
            };
            // Show the effective canvas before dispatch; Diffusers rounds down.
            self.ui
                .text_input(cx, ids!(image_width))
                .set_text(cx, &options.width.to_string());
            self.ui
                .text_input(cx, ids!(image_height))
                .set_text(cx, &options.height.to_string());
            let observing = options.observation;
            let canvas = (options.width as u32, options.height as u32);
            if observing {
                i.output = None;
                i.has_image = false;
                i.observation_board = ObservationBoard::new(canvas.0, canvas.1);
            }
            i.busy = true;
            i.stopping = false;
            let result = i.engine.submit(Command::Generate(options));
            self.image_status(
                cx,
                if observing {
                    "开始去噪观察 · 官方文本条件在上，1B 投影在下 · 八步都会留下"
                } else {
                    "开始生成 · 首次运行需编译 GPU 算子…"
                },
            );
            if observing {
                self.ui
                    .view(cx, ids!(image_preview_single))
                    .set_visible(cx, false);
                self.ui
                    .view(cx, ids!(image_observation))
                    .set_visible(cx, true);
                self.paint_observation(cx);
            }
            self.ui.view(cx, ids!(image_progress)).set_visible(cx, true);
            self.image_progress(cx, 0.0);
            if let Err(e) = result {
                self.image_error(cx, &e.to_string(), false);
            }
        }
    }
}

const PREVIEW_EDGE: u32 = 160;

struct ShownCell {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

struct OfficialRow {
    canvas_width: u32,
    canvas_height: u32,
    cells: Vec<ShownCell>,
}

impl OfficialRow {
    fn new(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            canvas_width,
            canvas_height,
            cells: Vec::new(),
        }
    }

    fn canvas(&self) -> (u32, u32) {
        (self.canvas_width, self.canvas_height)
    }

    /// Longest edge used when the preview scales a cell down. The stored decode stays at canvas size.
    fn display_edge(&self) -> u32 {
        self.canvas_width.max(self.canvas_height).min(PREVIEW_EDGE)
    }

    fn cells(&self) -> &[ShownCell] {
        &self.cells
    }

    fn slots(&self) -> [Option<&ShownCell>; 8] {
        let mut slots = [None, None, None, None, None, None, None, None];
        for (index, cell) in self.cells.iter().enumerate() {
            slots[index] = Some(cell);
        }
        slots
    }

    fn show(&mut self, step: usize, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
        if width != self.canvas_width || height != self.canvas_height {
            return Err(format!(
                "观察格须按请求画布 {}×{} 解码",
                self.canvas_width, self.canvas_height
            ));
        }
        if pixels.len() != width as usize * height as usize * 4 {
            return Err("观察格像素与画布不一致".into());
        }
        if step == 0 || step > 8 || step != self.cells.len() + 1 {
            return Err("去噪观察按步留下格子，不能替换已经出现的画面".into());
        }
        self.cells.push(ShownCell {
            width,
            height,
            pixels: pixels.to_vec(),
        });
        Ok(())
    }
}

struct ObservationBoard {
    official: OfficialRow,
    minicpm_1b: OfficialRow,
}

impl ObservationBoard {
    fn new(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            official: OfficialRow::new(canvas_width, canvas_height),
            minicpm_1b: OfficialRow::new(canvas_width, canvas_height),
        }
    }

    fn row_order(&self) -> [omni_qwen_image::ObservationRow; 2] {
        [
            omni_qwen_image::ObservationRow::Official,
            omni_qwen_image::ObservationRow::MiniCpm1b,
        ]
    }

    fn rows(&self) -> [&OfficialRow; 2] {
        [&self.official, &self.minicpm_1b]
    }

    fn canvas(&self) -> (u32, u32) {
        self.official.canvas()
    }

    fn show(
        &mut self,
        row: omni_qwen_image::ObservationRow,
        step: usize,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(), String> {
        match row {
            omni_qwen_image::ObservationRow::Official => {
                self.official.show(step, width, height, pixels)
            }
            omni_qwen_image::ObservationRow::MiniCpm1b => {
                self.minicpm_1b.show(step, width, height, pixels)
            }
        }
    }
}

impl App {
    fn paint_observation(&mut self, cx: &mut Cx) {
        let painted: [Vec<Option<(u32, u32, Vec<u32>)>>; 2] = {
            let board = &self.image.as_ref().unwrap().observation_board;
            std::array::from_fn(|row| {
                (0..8)
                    .map(|index| {
                        board.rows()[row].slots()[index].map(|cell| {
                            (
                                cell.width,
                                cell.height,
                                preview_pixels(cell.width, cell.height, &cell.pixels),
                            )
                        })
                    })
                    .collect()
            })
        };
        for (row, cells) in painted.into_iter().enumerate() {
            for (index, cell) in cells.into_iter().enumerate() {
                let texture = cell.map(|(width, height, pixels)| {
                    Texture::new_with_format(
                        cx,
                        TextureFormat::VecBGRAu8_32 {
                            width: width as usize,
                            height: height as usize,
                            data: Some(pixels),
                            updated: TextureUpdated::Full,
                        },
                    )
                });
                if row == 0 {
                    self.set_official_step(cx, index + 1, texture);
                } else {
                    self.set_minicpm_step(cx, index + 1, texture);
                }
            }
        }
        let board = &self.image.as_ref().unwrap().observation_board;
        let (width, height) = board.canvas();
        let edge = board.rows()[0].display_edge();
        let scale = if edge < width.max(height) {
            " · 画面已缩小"
        } else {
            ""
        };
        self.ui
            .label(cx, ids!(image_preview_caption))
            .set_text(cx, "去噪观察");
        self.ui
            .label(cx, ids!(image_official_caption))
            .set_text(cx, &format!("官方文本条件 · 解码 {width}×{height}{scale}"));
        self.ui.label(cx, ids!(image_1b_caption)).set_text(
            cx,
            &format!("MiniCPM-V-4.7-1B 投影 · 解码 {width}×{height}{scale}"),
        );
        self.ui.view(cx, ids!(image_preview_group)).redraw(cx);
    }

    fn set_official_step(&mut self, cx: &mut Cx, step: usize, texture: Option<Texture>) {
        let image = match step {
            1 => self.ui.image(cx, ids!(image_step_1)),
            2 => self.ui.image(cx, ids!(image_step_2)),
            3 => self.ui.image(cx, ids!(image_step_3)),
            4 => self.ui.image(cx, ids!(image_step_4)),
            5 => self.ui.image(cx, ids!(image_step_5)),
            6 => self.ui.image(cx, ids!(image_step_6)),
            7 => self.ui.image(cx, ids!(image_step_7)),
            _ => self.ui.image(cx, ids!(image_step_8)),
        };
        image.set_texture(cx, texture);
    }

    fn set_minicpm_step(&mut self, cx: &mut Cx, step: usize, texture: Option<Texture>) {
        let image = match step {
            1 => self.ui.image(cx, ids!(image_1b_step_1)),
            2 => self.ui.image(cx, ids!(image_1b_step_2)),
            3 => self.ui.image(cx, ids!(image_1b_step_3)),
            4 => self.ui.image(cx, ids!(image_1b_step_4)),
            5 => self.ui.image(cx, ids!(image_1b_step_5)),
            6 => self.ui.image(cx, ids!(image_1b_step_6)),
            7 => self.ui.image(cx, ids!(image_1b_step_7)),
            _ => self.ui.image(cx, ids!(image_1b_step_8)),
        };
        image.set_texture(cx, texture);
    }
}

fn preview_pixels(width: u32, height: u32, rgba: &[u8]) -> Vec<u32> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let index = ((y * width + x) * 4) as usize;
            let matte = if ((x / 16) + (y / 16)) % 2 == 0 {
                225u32
            } else {
                195
            };
            let alpha = rgba[index + 3] as u32;
            let mix = |value: u8| (value as u32 * alpha + matte * (255 - alpha) + 127) / 255;
            pixels.push(
                0xff000000
                    | (mix(rgba[index]) << 16)
                    | (mix(rgba[index + 1]) << 8)
                    | mix(rgba[index + 2]),
            );
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::{ObservationBoard, OfficialRow};
    use omni_qwen_image::ObservationRow;

    fn paint(step: u8) -> Vec<u8> {
        let mut pixels = vec![0u8; 320 * 240 * 4];
        pixels[0] = step;
        pixels[3] = 255;
        pixels
    }

    #[test]
    fn official_row_keeps_earlier_cells_as_later_steps_arrive() {
        let mut row = OfficialRow::new(320, 240);
        assert!(row.show(1, 64, 64, &vec![0; 64 * 64 * 4]).is_err());
        assert!(row.cells().is_empty());

        for step in 1..=8 {
            row.show(step, 320, 240, &paint(step as u8)).unwrap();
            assert_eq!(row.cells().len(), step);
            assert_eq!(row.slots()[0].unwrap().pixels[0], 1);
            assert_eq!(row.slots()[0].unwrap().width, 320);
            assert_eq!(row.slots()[0].unwrap().height, 240);
        }
        let slots = row.slots();
        assert!(slots.iter().all(|slot| slot.is_some()));
        assert_eq!(slots[7].unwrap().pixels[0], 8);
        assert_eq!(row.canvas(), (320, 240));
        assert_eq!(row.display_edge(), 160);

        assert!(row.show(1, 320, 240, &paint(9)).is_err());
        assert_eq!(row.cells().len(), 8);
        assert_eq!(row.slots()[0].unwrap().pixels[0], 1);
    }

    #[test]
    fn minicpm_1b_decodes_stay_under_the_official_row() {
        let mut board = ObservationBoard::new(320, 240);
        assert_eq!(
            board.row_order(),
            [ObservationRow::Official, ObservationRow::MiniCpm1b]
        );
        for step in 1..=8 {
            board
                .show(ObservationRow::Official, step, 320, 240, &paint(step as u8))
                .unwrap();
            board
                .show(
                    ObservationRow::MiniCpm1b,
                    step,
                    320,
                    240,
                    &paint(step as u8 + 10),
                )
                .unwrap();
            let rows = board.rows();
            assert_eq!(rows[0].cells().len(), step);
            assert_eq!(rows[1].cells().len(), step);
            assert_eq!(rows[0].slots()[0].unwrap().pixels[0], 1);
            assert_eq!(rows[0].slots()[0].unwrap().width, 320);
            assert_eq!(rows[0].slots()[0].unwrap().height, 240);
            assert_eq!(rows[1].slots()[0].unwrap().pixels[0], 11);
            assert_eq!(rows[1].slots()[0].unwrap().width, 320);
            assert_eq!(rows[1].slots()[0].unwrap().height, 240);
        }
        assert!(board
            .rows()
            .iter()
            .all(|row| row.slots().iter().all(|slot| slot.is_some())));
        assert!(board
            .show(ObservationRow::MiniCpm1b, 1, 320, 240, &paint(99))
            .is_err());
        assert_eq!(board.rows()[1].cells().len(), 8);
        assert_eq!(board.rows()[1].slots()[0].unwrap().pixels[0], 11);
        assert!(board
            .show(ObservationRow::MiniCpm1b, 9, 64, 64, &vec![0; 64 * 64 * 4],)
            .is_err());
        assert_eq!(board.rows()[1].canvas(), (320, 240));
        assert_eq!(board.rows()[1].slots()[7].unwrap().pixels[0], 18);
    }
}
