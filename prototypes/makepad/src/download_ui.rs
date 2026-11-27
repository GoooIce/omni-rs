//! Makepad desktop bindings for inline missing-model recovery cards (all 6
//! workbench panels) and the unified Models & Download Settings panel.

use crate::models_state::{
    slot_index, CheckpointLifecycle, FallbackPreset, ModelsController, RecoveryCardModel,
    RoutePreset, SettingsTab, TokenFieldState,
};
use crate::{ActionButtonState, App};
use makepad_widgets::*;
use omni_hub::catalog::CheckpointId;
use omni_hub::credentials::CredentialStore;
use omni_hub::providers::{CapabilitySlot, ExternalProviderTarget};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum TokenTarget {
    Hf,
    ModelScope,
    ProviderApiKey,
}

struct TokenFieldIds {
    target: TokenTarget,
    input: &'static [LiveId],
    toggle: &'static [LiveId],
    save: &'static [LiveId],
    clear: &'static [LiveId],
}

struct RecoveryCardIds {
    container: &'static [LiveId],
    info: &'static [LiveId],
    hint: &'static [LiveId],
    download: &'static [LiveId],
    pause: Option<&'static [LiveId]>,
    cancel: Option<&'static [LiveId]>,
}

struct SlotRowIds {
    slot: CapabilitySlot,
    badge: &'static [LiveId],
    local: &'static [LiveId],
    external: &'static [LiveId],
    provider: &'static [LiveId],
    model: &'static [LiveId],
    use_first: &'static [LiveId],
}

fn slot_rows() -> [SlotRowIds; 4] {
    [
        SlotRowIds {
            slot: CapabilitySlot::PromptEnhancement,
            badge: ids!(slot_enhance_badge),
            local: ids!(slot_enhance_local),
            external: ids!(slot_enhance_external),
            provider: ids!(slot_enhance_provider),
            model: ids!(slot_enhance_model),
            use_first: ids!(slot_enhance_use_first),
        },
        SlotRowIds {
            slot: CapabilitySlot::Asr,
            badge: ids!(slot_asr_badge),
            local: ids!(slot_asr_local),
            external: ids!(slot_asr_external),
            provider: ids!(slot_asr_provider),
            model: ids!(slot_asr_model),
            use_first: ids!(slot_asr_use_first),
        },
        SlotRowIds {
            slot: CapabilitySlot::Tts,
            badge: ids!(slot_tts_badge),
            local: ids!(slot_tts_local),
            external: ids!(slot_tts_external),
            provider: ids!(slot_tts_provider),
            model: ids!(slot_tts_model),
            use_first: ids!(slot_tts_use_first),
        },
        SlotRowIds {
            slot: CapabilitySlot::Translation,
            badge: ids!(slot_translation_badge),
            local: ids!(slot_translation_local),
            external: ids!(slot_translation_external),
            provider: ids!(slot_translation_provider),
            model: ids!(slot_translation_model),
            use_first: ids!(slot_translation_use_first),
        },
    ]
}

struct RowIds {
    id: CheckpointId,
    status: &'static [LiveId],
    progress: &'static [LiveId],
    progress_fill: &'static [LiveId],
    hint: &'static [LiveId],
    action: &'static [LiveId],
    pause: &'static [LiveId],
    cancel: &'static [LiveId],
    choose: &'static [LiveId],
}

fn checkpoint_rows() -> [RowIds; 11] {
    [
        RowIds {
            id: CheckpointId::Confucius4R2t2,
            status: ids!(ckpt_asr_status),
            progress: ids!(ckpt_asr_progress),
            progress_fill: ids!(ckpt_asr_progress_fill),
            hint: ids!(ckpt_asr_hint),
            action: ids!(ckpt_asr_action),
            pause: ids!(ckpt_asr_pause),
            cancel: ids!(ckpt_asr_cancel),
            choose: ids!(ckpt_asr_choose),
        },
        RowIds {
            id: CheckpointId::BreezeTts2,
            status: ids!(ckpt_tts_status),
            progress: ids!(ckpt_tts_progress),
            progress_fill: ids!(ckpt_tts_progress_fill),
            hint: ids!(ckpt_tts_hint),
            action: ids!(ckpt_tts_action),
            pause: ids!(ckpt_tts_pause),
            cancel: ids!(ckpt_tts_cancel),
            choose: ids!(ckpt_tts_choose),
        },
        RowIds {
            id: CheckpointId::HyMt2_7b,
            status: ids!(ckpt_hymt_status),
            progress: ids!(ckpt_hymt_progress),
            progress_fill: ids!(ckpt_hymt_progress_fill),
            hint: ids!(ckpt_hymt_hint),
            action: ids!(ckpt_hymt_action),
            pause: ids!(ckpt_hymt_pause),
            cancel: ids!(ckpt_hymt_cancel),
            choose: ids!(ckpt_hymt_choose),
        },
        RowIds {
            id: CheckpointId::QwenImage21Turbo,
            status: ids!(ckpt_image_status),
            progress: ids!(ckpt_image_progress),
            progress_fill: ids!(ckpt_image_progress_fill),
            hint: ids!(ckpt_image_hint),
            action: ids!(ckpt_image_action),
            pause: ids!(ckpt_image_pause),
            cancel: ids!(ckpt_image_cancel),
            choose: ids!(ckpt_image_choose),
        },
        RowIds {
            id: CheckpointId::MertV2FullSong,
            status: ids!(ckpt_mert_status),
            progress: ids!(ckpt_mert_progress),
            progress_fill: ids!(ckpt_mert_progress_fill),
            hint: ids!(ckpt_mert_hint),
            action: ids!(ckpt_mert_action),
            pause: ids!(ckpt_mert_pause),
            cancel: ids!(ckpt_mert_cancel),
            choose: ids!(ckpt_mert_choose),
        },
        RowIds {
            id: CheckpointId::SheetSage2,
            status: ids!(ckpt_sheetsage_status),
            progress: ids!(ckpt_sheetsage_progress),
            progress_fill: ids!(ckpt_sheetsage_progress_fill),
            hint: ids!(ckpt_sheetsage_hint),
            action: ids!(ckpt_sheetsage_action),
            pause: ids!(ckpt_sheetsage_pause),
            cancel: ids!(ckpt_sheetsage_cancel),
            choose: ids!(ckpt_sheetsage_choose),
        },
        RowIds {
            id: CheckpointId::Yue2_3b,
            status: ids!(ckpt_yue3b_status),
            progress: ids!(ckpt_yue3b_progress),
            progress_fill: ids!(ckpt_yue3b_progress_fill),
            hint: ids!(ckpt_yue3b_hint),
            action: ids!(ckpt_yue3b_action),
            pause: ids!(ckpt_yue3b_pause),
            cancel: ids!(ckpt_yue3b_cancel),
            choose: ids!(ckpt_yue3b_choose),
        },
        RowIds {
            id: CheckpointId::Yue2Vae,
            status: ids!(ckpt_yuevae_status),
            progress: ids!(ckpt_yuevae_progress),
            progress_fill: ids!(ckpt_yuevae_progress_fill),
            hint: ids!(ckpt_yuevae_hint),
            action: ids!(ckpt_yuevae_action),
            pause: ids!(ckpt_yuevae_pause),
            cancel: ids!(ckpt_yuevae_cancel),
            choose: ids!(ckpt_yuevae_choose),
        },
        RowIds {
            id: CheckpointId::MiniCpmV47_1b,
            status: ids!(ckpt_vlm1b_status),
            progress: ids!(ckpt_vlm1b_progress),
            progress_fill: ids!(ckpt_vlm1b_progress_fill),
            hint: ids!(ckpt_vlm1b_hint),
            action: ids!(ckpt_vlm1b_action),
            pause: ids!(ckpt_vlm1b_pause),
            cancel: ids!(ckpt_vlm1b_cancel),
            choose: ids!(ckpt_vlm1b_choose),
        },
        RowIds {
            id: CheckpointId::MiniCpmV47_35bInt4,
            status: ids!(ckpt_vlm35b_status),
            progress: ids!(ckpt_vlm35b_progress),
            progress_fill: ids!(ckpt_vlm35b_progress_fill),
            hint: ids!(ckpt_vlm35b_hint),
            action: ids!(ckpt_vlm35b_action),
            pause: ids!(ckpt_vlm35b_pause),
            cancel: ids!(ckpt_vlm35b_cancel),
            choose: ids!(ckpt_vlm35b_choose),
        },
        RowIds {
            id: CheckpointId::Lfm3Omni2b,
            status: ids!(ckpt_omni_status),
            progress: ids!(ckpt_omni_progress),
            progress_fill: ids!(ckpt_omni_progress_fill),
            hint: ids!(ckpt_omni_hint),
            action: ids!(ckpt_omni_action),
            pause: ids!(ckpt_omni_pause),
            cancel: ids!(ckpt_omni_cancel),
            choose: ids!(ckpt_omni_choose),
        },
    ]
}

pub struct UiState {
    pub ctrl: ModelsController,
    picking: Option<CheckpointId>,
    dialog: Option<crate::desktop_runtime::Dialog>,
    pub asr_missing_shown: bool,
    pub asr_load_error: Option<String>,
    pub tts_missing_shown: bool,
    pub tts_load_error: Option<String>,
    pub translation_missing_shown: bool,
    pub image_missing_shown: bool,
    pub image_load_error: Option<String>,
    pub music_missing_checkpoint: Option<CheckpointId>,
    pub omni_missing_shown: bool,
    last_hf_rendered: String,
    last_ms_rendered: String,
    last_provider_key_rendered: String,
}

impl UiState {
    pub fn new(root: &Path) -> Self {
        let ctrl = ModelsController::new(root.to_path_buf(), SignalToUI::set_ui_signal);
        Self {
            ctrl,
            picking: None,
            dialog: None,
            asr_missing_shown: false,
            asr_load_error: None,
            tts_missing_shown: false,
            tts_load_error: None,
            translation_missing_shown: false,
            image_missing_shown: false,
            image_load_error: None,
            music_missing_checkpoint: None,
            omni_missing_shown: false,
            last_hf_rendered: String::new(),
            last_ms_rendered: String::new(),
            last_provider_key_rendered: String::new(),
        }
    }
}

impl App {
    pub fn initialize_downloads(&mut self, cx: &mut Cx, root: &Path) {
        let mut state = UiState::new(root);

        // Sync existing panel directories (including env overrides used by harnesses)
        // into ModelsController so readiness reflects the exact active paths.
        if let Some(dir) = self.transcription_directory() {
            state.ctrl.set_directory(CheckpointId::Confucius4R2t2, dir);
        }
        if let Some(t) = &self.tts {
            state
                .ctrl
                .set_directory(CheckpointId::BreezeTts2, t.path().to_path_buf());
            state.ctrl.set_directory(
                CheckpointId::MiniCpmV47_35bInt4,
                t.enhancement_checkpoint().to_path_buf(),
            );
        }
        if let Some(tr) = &self.translation {
            state
                .ctrl
                .set_directory(CheckpointId::HyMt2_7b, tr.path().to_path_buf());
        }
        if let Some(img) = &self.image {
            state
                .ctrl
                .set_directory(CheckpointId::QwenImage21Turbo, img.path().to_path_buf());
        }
        let music_root = PathBuf::from(self.ui.text_input(cx, ids!(music_models)).text());
        if !music_root.as_os_str().is_empty() {
            for (id, sub) in [
                (CheckpointId::MertV2FullSong, "MERT-v2-FullSong"),
                (CheckpointId::SheetSage2, "SheetSage2"),
                (CheckpointId::Yue2_3b, "YuE2-3B"),
                (CheckpointId::Yue2Vae, "YuE2-Vae"),
            ] {
                state.ctrl.set_directory(id, music_root.join(sub));
            }
        }
        let omni_dir = PathBuf::from(self.ui.text_input(cx, ids!(omni_model)).text());
        if !omni_dir.as_os_str().is_empty() {
            state.ctrl.set_directory(CheckpointId::Lfm3Omni2b, omni_dir);
        }

        // Seed route & token inputs once without clobbering later unsaved edits.
        self.ui
            .text_input(cx, ids!(mirror_url_input))
            .set_text(cx, &state.ctrl.route_draft.mirror_url);
        self.ui
            .text_input(cx, ids!(proxy_url_input))
            .set_text(cx, &state.ctrl.route_draft.proxy_url);

        state.last_hf_rendered = self.apply_token_input(
            cx,
            ids!(hf_token_input),
            &state.ctrl.hf_token,
            state.ctrl.credentials.as_ref(),
        );
        state.last_ms_rendered = self.apply_token_input(
            cx,
            ids!(ms_token_input),
            &state.ctrl.ms_token,
            state.ctrl.credentials.as_ref(),
        );

        self.seed_provider_inputs_from_state(cx, &mut state);

        self.downloads = Some(state);
        self.refresh_download_controls(cx);
    }

    fn apply_token_input(
        &self,
        cx: &mut Cx,
        input_id: &[LiveId],
        field: &TokenFieldState,
        store: &dyn CredentialStore,
    ) -> String {
        let (text, ro) = field.rendered_input(store);
        let input = self.ui.text_input(cx, input_id);
        input.set_is_password(cx, field.is_masked());
        input.set_text(cx, &text);
        input.set_is_read_only(cx, ro);
        text
    }

    pub fn sync_transcription_download_directory(&mut self, cx: &mut Cx) {
        if let Some(dir) = self.transcription_directory() {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.set_directory(CheckpointId::Confucius4R2t2, dir);
            }
            self.refresh_download_controls(cx);
        }
    }

    fn seed_provider_inputs_from_state(&self, cx: &mut Cx, state: &mut UiState) {
        let draft = &state.ctrl.provider_draft;
        self.ui
            .text_input(cx, ids!(provider_id_input))
            .set_text(cx, &draft.id);
        self.ui
            .text_input(cx, ids!(provider_name_input))
            .set_text(cx, &draft.name);
        self.ui
            .text_input(cx, ids!(provider_base_url_input))
            .set_text(cx, &draft.base_url);
        self.ui
            .text_input(cx, ids!(provider_timeout_input))
            .set_text(cx, &draft.timeout_secs);
        self.ui
            .text_input(cx, ids!(provider_api_key_env_input))
            .set_text(cx, &draft.api_key_env);
        self.ui
            .text_input(cx, ids!(provider_custom_headers_input))
            .set_text(cx, &draft.custom_headers);

        state.last_provider_key_rendered = self.apply_token_input(
            cx,
            ids!(provider_api_key_input),
            &state.ctrl.provider_api_key,
            state.ctrl.credentials.as_ref(),
        );

        for row in slot_rows() {
            self.ui
                .text_input(cx, row.model)
                .set_text(cx, &state.ctrl.slot_drafts[slot_index(row.slot)].model);
        }
        self.ui.text_input(cx, ids!(slot_tts_voice)).set_text(
            cx,
            &state.ctrl.slot_drafts[slot_index(CapabilitySlot::Tts)].voice,
        );
    }

    pub fn slot_is_external(&self, slot: CapabilitySlot) -> bool {
        self.downloads
            .as_ref()
            .is_some_and(|d| d.ctrl.is_slot_external(slot))
    }

    pub fn slot_badge(&self, slot: CapabilitySlot) -> String {
        self.downloads
            .as_ref()
            .map(|d| d.ctrl.slot_badge(slot))
            .unwrap_or_else(|| "本地模型".to_owned())
    }

    pub fn resolve_slot_external(
        &self,
        slot: CapabilitySlot,
    ) -> Result<Option<ExternalProviderTarget>, String> {
        let Some(d) = self.downloads.as_ref() else {
            return Ok(None);
        };
        d.ctrl
            .resolve_slot_external(slot)
            .map_err(|e| e.to_string())
    }

    pub fn switch_slot_to_local(&mut self, cx: &mut Cx, slot: CapabilitySlot) {
        if let Some(d) = self.downloads.as_mut() {
            d.ctrl.switch_slot_to_local(slot);
        }
        self.on_slot_routing_changed(cx, slot);
        self.refresh_download_controls(cx);
    }

    pub fn open_providers_settings(&mut self, cx: &mut Cx) {
        if let Some(d) = self.downloads.as_mut() {
            d.ctrl.settings_tab = SettingsTab::ExternalProviders;
        }
        self.open_settings_panel(cx);
    }

    fn on_slot_routing_changed(&mut self, cx: &mut Cx, slot: CapabilitySlot) {
        match slot {
            CapabilitySlot::Asr => self.on_asr_provider_changed(cx),
            CapabilitySlot::Tts => self.on_tts_provider_changed(cx),
            CapabilitySlot::Translation => self.on_translation_provider_changed(cx),
            CapabilitySlot::PromptEnhancement => {
                self.on_tts_provider_changed(cx);
                self.on_image_provider_changed(cx);
            }
        }
    }

    pub fn settings_panel_open(&self) -> bool {
        self.downloads
            .as_ref()
            .is_some_and(|d| d.ctrl.settings_open)
    }

    pub fn open_settings_panel(&mut self, cx: &mut Cx) {
        if let Some(d) = self.downloads.as_mut() {
            d.ctrl.settings_open = true;
            d.ctrl.refresh_all_readiness();
        }
        self.ui.view(cx, ids!(settings_panel)).set_visible(cx, true);
        self.ui.view(cx, ids!(page_stage)).set_visible(cx, false);
        self.ui
            .view(cx, ids!(pages))
            .set_scroll_pos(cx, Vec2d::default());
        self.refresh_download_controls(cx);
    }

    pub fn close_settings_panel(&mut self, cx: &mut Cx) {
        if let Some(d) = self.downloads.as_mut() {
            if !d.ctrl.settings_open {
                return;
            }
            d.ctrl.settings_open = false;
        }
        self.ui
            .view(cx, ids!(settings_panel))
            .set_visible(cx, false);
        self.ui.view(cx, ids!(page_stage)).set_visible(cx, true);
        self.ui
            .view(cx, ids!(pages))
            .set_scroll_pos(cx, Vec2d::default());
        self.refresh_download_controls(cx);
    }

    pub fn notify_panel_load_error(&mut self, cx: &mut Cx, id: CheckpointId, error: &str) {
        if let Some(d) = self.downloads.as_mut() {
            match id {
                CheckpointId::Confucius4R2t2 => d.asr_load_error = Some(error.to_owned()),
                CheckpointId::BreezeTts2 => d.tts_load_error = Some(error.to_owned()),
                CheckpointId::QwenImage21Turbo => d.image_load_error = Some(error.to_owned()),
                _ => {}
            }
        }
        self.refresh_download_controls(cx);
    }

    pub fn clear_panel_load_error(&mut self, cx: &mut Cx, id: CheckpointId) {
        if let Some(d) = self.downloads.as_mut() {
            match id {
                CheckpointId::Confucius4R2t2 => d.asr_load_error = None,
                CheckpointId::BreezeTts2 => d.tts_load_error = None,
                CheckpointId::QwenImage21Turbo => d.image_load_error = None,
                _ => {}
            }
        }
        self.refresh_download_controls(cx);
    }

    pub fn notify_panel_missing(&mut self, cx: &mut Cx, id: CheckpointId) {
        if let Some(d) = self.downloads.as_mut() {
            d.ctrl.refresh_readiness(id);
            match id {
                CheckpointId::Confucius4R2t2 => d.asr_missing_shown = true,
                CheckpointId::BreezeTts2 => d.tts_missing_shown = true,
                CheckpointId::HyMt2_7b => d.translation_missing_shown = true,
                CheckpointId::QwenImage21Turbo => d.image_missing_shown = true,
                CheckpointId::MertV2FullSong
                | CheckpointId::SheetSage2
                | CheckpointId::Yue2_3b
                | CheckpointId::Yue2Vae => d.music_missing_checkpoint = Some(id),
                CheckpointId::Lfm3Omni2b => d.omni_missing_shown = true,
                _ => {}
            }
        }
        self.refresh_download_controls(cx);
    }

    pub fn sync_checkpoint_directory(&mut self, cx: &mut Cx, id: CheckpointId, dir: PathBuf) {
        if let Some(d) = self.downloads.as_mut() {
            d.ctrl.set_directory(id, dir);
            if d.ctrl.lifecycle(id).is_ready() {
                match id {
                    CheckpointId::Confucius4R2t2 => d.asr_missing_shown = false,
                    CheckpointId::BreezeTts2 => d.tts_missing_shown = false,
                    CheckpointId::HyMt2_7b => d.translation_missing_shown = false,
                    CheckpointId::QwenImage21Turbo => d.image_missing_shown = false,
                    CheckpointId::MertV2FullSong
                    | CheckpointId::SheetSage2
                    | CheckpointId::Yue2_3b
                    | CheckpointId::Yue2Vae => {
                        if d.music_missing_checkpoint == Some(id) {
                            d.music_missing_checkpoint = None;
                        }
                    }
                    CheckpointId::Lfm3Omni2b => d.omni_missing_shown = false,
                    _ => {}
                }
            }
        }
        self.refresh_download_controls(cx);
    }

    fn trigger_checkpoint_primary_action(&mut self, cx: &mut Cx, id: CheckpointId) {
        let Some(d) = self.downloads.as_mut() else {
            return;
        };
        let lc = d.ctrl.lifecycle(id).clone();
        match lc {
            CheckpointLifecycle::Failed {
                can_switch_hf: true,
                ..
            } => {
                let _ = d.ctrl.download_via_hf_fallback(id);
            }
            CheckpointLifecycle::Failed {
                can_clean_retry: true,
                ..
            } => {
                let _ = d.ctrl.clean_and_redownload(id);
            }
            CheckpointLifecycle::Ready => {}
            _ => {
                let _ = d.ctrl.start_download(id, None);
            }
        }
        self.refresh_download_controls(cx);
    }

    fn choose_checkpoint_directory(&mut self, cx: &mut Cx, id: CheckpointId) {
        let Some(dialog) = self.runtime.dialog("models-settings") else {
            return;
        };
        let Some(d) = self.downloads.as_mut() else {
            return;
        };
        d.dialog = Some(dialog);
        d.picking = Some(id);
        let current = d.ctrl.directory(id);
        let title = format!("选择 {} 模型目录", id.descriptor().display_name);
        crate::engine_ui::choose_directory(cx, &current, &title);
    }

    fn apply_ready_checkpoint_to_panels(&mut self, cx: &mut Cx, id: CheckpointId, dir: &Path) {
        match id {
            CheckpointId::Confucius4R2t2 => {
                self.apply_downloaded_asr_directory(cx, dir);
            }
            CheckpointId::BreezeTts2 => {
                if let Some(t) = self.tts.as_mut() {
                    t.set_directory_ready_for_load(dir.to_path_buf());
                }
                self.ui.view(cx, ids!(tts_recovery)).set_visible(cx, false);
                self.ui.label(cx, ids!(tts_status)).set_text(
                    cx,
                    "Breeze-TTS-2 下载并校验完成 · 点击「载入语音模型」即可使用",
                );
            }
            CheckpointId::HyMt2_7b => {
                if let Some(tr) = self.translation.as_mut() {
                    tr.set_directory(dir.to_path_buf());
                }
                self.ui
                    .view(cx, ids!(translation_recovery))
                    .set_visible(cx, false);
                self.ui
                    .view(cx, ids!(asr_translation_note))
                    .set_visible(cx, true);
                for label in [ids!(asr_translation_status), ids!(tts_translation_status)] {
                    self.ui
                        .label(cx, label)
                        .set_text(cx, "Hy-MT2-7B 下载并校验完成 · 点击翻译即可自动载入");
                }
            }
            CheckpointId::QwenImage21Turbo => {
                if let Some(img) = self.image.as_mut() {
                    img.set_directory_ready_for_load(dir.to_path_buf());
                }
                self.ui
                    .view(cx, ids!(image_recovery))
                    .set_visible(cx, false);
                self.ui.label(cx, ids!(image_status)).set_text(
                    cx,
                    "Qwen-Image-2.1-Turbo 下载并校验完成 · 点击「载入图像模型」即可使用",
                );
            }
            CheckpointId::MertV2FullSong
            | CheckpointId::SheetSage2
            | CheckpointId::Yue2_3b
            | CheckpointId::Yue2Vae => {
                if let Some(parent) = dir.parent() {
                    self.ui
                        .text_input(cx, ids!(music_models))
                        .set_text(cx, &parent.to_string_lossy());
                }
                self.ui
                    .view(cx, ids!(music_recovery))
                    .set_visible(cx, false);
                self.ui.label(cx, ids!(music_status)).set_text(
                    cx,
                    &format!(
                        "{} 下载并校验完成 · 可继续创作或转谱",
                        id.descriptor().display_name
                    ),
                );
            }
            CheckpointId::MiniCpmV47_35bInt4 => {
                if let Some(t) = self.tts.as_mut() {
                    t.set_enhancement_checkpoint(dir.to_path_buf());
                }
            }
            CheckpointId::Lfm3Omni2b => {
                self.ui
                    .text_input(cx, ids!(omni_model))
                    .set_text(cx, &dir.to_string_lossy());
                self.ui.view(cx, ids!(omni_recovery)).set_visible(cx, false);
                self.ui
                    .label(cx, ids!(omni_status))
                    .set_text(cx, "LFM3-Omni-2B 目录已就绪 · 点击「载入」即可使用");
            }
            CheckpointId::MiniCpmV47_1b => {}
        }
    }

    pub fn poll_downloads(&mut self, cx: &mut Cx) {
        let ready_items: Vec<(CheckpointId, PathBuf)> = {
            let Some(d) = self.downloads.as_mut() else {
                return;
            };
            let newly_ready = d.ctrl.poll();
            newly_ready
                .into_iter()
                .map(|id| {
                    match id {
                        CheckpointId::Confucius4R2t2 => d.asr_missing_shown = false,
                        CheckpointId::BreezeTts2 => d.tts_missing_shown = false,
                        CheckpointId::HyMt2_7b => d.translation_missing_shown = false,
                        CheckpointId::QwenImage21Turbo => d.image_missing_shown = false,
                        CheckpointId::MertV2FullSong
                        | CheckpointId::SheetSage2
                        | CheckpointId::Yue2_3b
                        | CheckpointId::Yue2Vae => {
                            if d.music_missing_checkpoint == Some(id) {
                                d.music_missing_checkpoint = None;
                            }
                        }
                        CheckpointId::Lfm3Omni2b => d.omni_missing_shown = false,
                        _ => {}
                    }
                    (id, d.ctrl.directory(id))
                })
                .collect()
        };

        for (id, dir) in ready_items {
            self.apply_ready_checkpoint_to_panels(cx, id, &dir);
        }
    }

    pub fn handle_download_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        if self.downloads.is_none() {
            return;
        }

        // Handle directory picker completion from Settings rows.
        let mut chosen_from_settings: Option<(CheckpointId, PathBuf)> = None;
        for action in actions {
            let d = self.downloads.as_mut().unwrap();
            let Some(id) = d.picking else {
                break;
            };
            match action.downcast_ref::<FileDialogAction>() {
                Some(FileDialogAction::FolderSelected(path)) => {
                    let selected = crate::model_location::selected_directory(path);
                    d.picking = None;
                    d.dialog = None;
                    chosen_from_settings = Some((id, selected));
                }
                Some(FileDialogAction::FolderCancelled) => {
                    d.picking = None;
                    d.dialog = None;
                }
                _ => {}
            }
        }
        if let Some((id, dir)) = chosen_from_settings {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.set_directory(id, dir.clone());
                let _ = d.ctrl.settings.save(&d.ctrl.settings_path);
            }
            if self
                .downloads
                .as_ref()
                .is_some_and(|d| d.ctrl.lifecycle(id).is_ready())
            {
                self.apply_ready_checkpoint_to_panels(cx, id, &dir);
            }
        }

        // Top-bar settings toggle & background download banner button.
        if self.ui.button(cx, ids!(settings_toggle)).clicked(actions) {
            if self.settings_panel_open() {
                self.close_settings_panel(cx);
            } else {
                self.open_settings_panel(cx);
            }
            return;
        }
        if self
            .ui
            .button(cx, ids!(download_banner_open))
            .clicked(actions)
        {
            self.open_settings_panel(cx);
            return;
        }
        if self.ui.button(cx, ids!(settings_close)).clicked(actions) {
            self.close_settings_panel(cx);
            return;
        }

        // Inline workbench recovery card actions (all 6 panels).
        if self.ui.button(cx, ids!(asr_open_settings)).clicked(actions)
            || self.ui.button(cx, ids!(tts_open_settings)).clicked(actions)
            || self
                .ui
                .button(cx, ids!(translation_open_settings))
                .clicked(actions)
            || self
                .ui
                .button(cx, ids!(image_open_settings))
                .clicked(actions)
            || self
                .ui
                .button(cx, ids!(music_open_settings))
                .clicked(actions)
            || self
                .ui
                .button(cx, ids!(omni_open_settings))
                .clicked(actions)
        {
            self.open_settings_panel(cx);
            return;
        }

        if self
            .ui
            .button(cx, ids!(asr_download_model))
            .clicked(actions)
        {
            self.trigger_checkpoint_primary_action(cx, CheckpointId::Confucius4R2t2);
        }
        if self
            .ui
            .button(cx, ids!(asr_download_pause))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.pause_download(CheckpointId::Confucius4R2t2);
            }
        }
        if self
            .ui
            .button(cx, ids!(asr_download_cancel))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.cancel_download(CheckpointId::Confucius4R2t2);
            }
        }

        if self
            .ui
            .button(cx, ids!(tts_download_model))
            .clicked(actions)
        {
            self.trigger_checkpoint_primary_action(cx, CheckpointId::BreezeTts2);
        }
        if self
            .ui
            .button(cx, ids!(tts_download_pause))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.pause_download(CheckpointId::BreezeTts2);
            }
        }
        if self
            .ui
            .button(cx, ids!(tts_download_cancel))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.cancel_download(CheckpointId::BreezeTts2);
            }
        }

        if self
            .ui
            .button(cx, ids!(translation_download_model))
            .clicked(actions)
        {
            self.trigger_checkpoint_primary_action(cx, CheckpointId::HyMt2_7b);
        }
        if self
            .ui
            .button(cx, ids!(translation_download_pause))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.pause_download(CheckpointId::HyMt2_7b);
            }
        }
        if self
            .ui
            .button(cx, ids!(translation_download_cancel))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.cancel_download(CheckpointId::HyMt2_7b);
            }
        }
        if self
            .ui
            .button(cx, ids!(translation_choose_model))
            .clicked(actions)
        {
            self.open_translation_folder_picker(cx);
        }

        if self
            .ui
            .button(cx, ids!(image_download_model))
            .clicked(actions)
        {
            self.trigger_checkpoint_primary_action(cx, CheckpointId::QwenImage21Turbo);
        }
        if self
            .ui
            .button(cx, ids!(image_download_pause))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.pause_download(CheckpointId::QwenImage21Turbo);
            }
        }
        if self
            .ui
            .button(cx, ids!(image_download_cancel))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.cancel_download(CheckpointId::QwenImage21Turbo);
            }
        }

        let music_target = self
            .downloads
            .as_ref()
            .and_then(|d| d.music_missing_checkpoint)
            .unwrap_or(CheckpointId::Yue2_3b);
        if self
            .ui
            .button(cx, ids!(music_download_model))
            .clicked(actions)
        {
            self.trigger_checkpoint_primary_action(cx, music_target);
        }
        if self
            .ui
            .button(cx, ids!(music_download_pause))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.pause_download(music_target);
            }
        }
        if self
            .ui
            .button(cx, ids!(music_download_cancel))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.cancel_download(music_target);
            }
        }
        if self
            .ui
            .button(cx, ids!(music_choose_recovery))
            .clicked(actions)
        {
            self.open_music_models_picker(cx);
        }
        if self
            .ui
            .button(cx, ids!(omni_choose_recovery))
            .clicked(actions)
        {
            self.open_omni_model_picker(cx);
        }

        if !self.settings_panel_open() {
            return;
        }

        // Settings sub-tab switch (Models & Download vs External Providers & Slots).
        if self
            .ui
            .button(cx, ids!(settings_tab_models))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.settings_tab = SettingsTab::ModelsAndDownload;
            }
            self.ui
                .view(cx, ids!(pages))
                .set_scroll_pos(cx, Vec2d::default());
            return;
        }
        if self
            .ui
            .button(cx, ids!(settings_tab_providers))
            .clicked(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.settings_tab = SettingsTab::ExternalProviders;
            }
            self.ui
                .view(cx, ids!(pages))
                .set_scroll_pos(cx, Vec2d::default());
            return;
        }

        let active_tab = self
            .downloads
            .as_ref()
            .map(|d| d.ctrl.settings_tab)
            .unwrap_or(SettingsTab::ModelsAndDownload);
        if active_tab == SettingsTab::ExternalProviders {
            self.handle_provider_settings_actions(cx, actions);
            return;
        }

        // Track unsaved edits in mirror / proxy / token inputs.
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(mirror_url_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.route_draft.mirror_url = text;
                d.ctrl.route_draft.dirty = true;
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(proxy_url_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.route_draft.proxy_url = text;
                d.ctrl.route_draft.dirty = true;
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(hf_token_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.hf_token.set_draft(text.clone());
                d.last_hf_rendered = text;
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(ms_token_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.ms_token.set_draft(text.clone());
                d.last_ms_rendered = text;
            }
        }

        // Route & fallback presets.
        if self.ui.button(cx, ids!(route_modelscope)).clicked(actions) {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.route_draft.route_preset = RoutePreset::ModelScope;
            d.ctrl.route_draft.dirty = true;
        }
        if self.ui.button(cx, ids!(route_huggingface)).clicked(actions) {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.route_draft.route_preset = RoutePreset::HuggingFace;
            d.ctrl.route_draft.dirty = true;
        }
        if self
            .ui
            .button(cx, ids!(route_custom_mirror))
            .clicked(actions)
        {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.route_draft.route_preset = RoutePreset::CustomMirror;
            d.ctrl.route_draft.dirty = true;
        }
        if self.ui.button(cx, ids!(fallback_prompt)).clicked(actions) {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.route_draft.fallback_preset = FallbackPreset::Prompt;
            d.ctrl.route_draft.dirty = true;
        }
        if self.ui.button(cx, ids!(fallback_hf)).clicked(actions) {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.route_draft.fallback_preset = FallbackPreset::UseHuggingFace;
            d.ctrl.route_draft.dirty = true;
        }
        if self.ui.button(cx, ids!(fallback_mirror)).clicked(actions) {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.route_draft.fallback_preset = FallbackPreset::UseCustomMirror;
            d.ctrl.route_draft.dirty = true;
        }

        // Save route / Test connection.
        if self
            .ui
            .button(cx, ids!(settings_save_route))
            .clicked(actions)
        {
            let d = self.downloads.as_mut().unwrap();
            if let Err(err) = d.ctrl.save_route_draft() {
                d.ctrl.route_feedback = format!("保存失败：{err}");
            }
        }
        if self
            .ui
            .button(cx, ids!(settings_test_connection))
            .clicked(actions)
        {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.start_connectivity_test();
        }

        // Token toggle / save / clear (HF & ModelScope).
        self.handle_token_buttons(
            cx,
            actions,
            &TokenFieldIds {
                target: TokenTarget::Hf,
                input: ids!(hf_token_input),
                toggle: ids!(hf_token_toggle),
                save: ids!(hf_token_save),
                clear: ids!(hf_token_clear),
            },
        );
        self.handle_token_buttons(
            cx,
            actions,
            &TokenFieldIds {
                target: TokenTarget::ModelScope,
                input: ids!(ms_token_input),
                toggle: ids!(ms_token_toggle),
                save: ids!(ms_token_save),
                clear: ids!(ms_token_clear),
            },
        );

        // Per-checkpoint download/pause/cancel/choose actions in Settings list.
        for row in checkpoint_rows() {
            if self.ui.button(cx, row.action).clicked(actions) {
                self.trigger_checkpoint_primary_action(cx, row.id);
            } else if self.ui.button(cx, row.pause).clicked(actions) {
                if let Some(d) = self.downloads.as_mut() {
                    d.ctrl.pause_download(row.id);
                }
            } else if self.ui.button(cx, row.cancel).clicked(actions) {
                if let Some(d) = self.downloads.as_mut() {
                    d.ctrl.cancel_download(row.id);
                }
            } else if self.ui.button(cx, row.choose).clicked(actions) {
                self.choose_checkpoint_directory(cx, row.id);
            }
        }
    }

    fn handle_provider_settings_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        // Saved profile switcher & deletion.
        if self
            .ui
            .button(cx, ids!(provider_profile_prev))
            .clicked(actions)
        {
            if let Some(mut state) = self.downloads.take() {
                if state.ctrl.step_provider_profile(-1).is_some() {
                    self.seed_provider_inputs_from_state(cx, &mut state);
                }
                self.downloads = Some(state);
            }
            return;
        }
        if self
            .ui
            .button(cx, ids!(provider_profile_next))
            .clicked(actions)
        {
            if let Some(mut state) = self.downloads.take() {
                if state.ctrl.step_provider_profile(1).is_some() {
                    self.seed_provider_inputs_from_state(cx, &mut state);
                }
                self.downloads = Some(state);
            }
            return;
        }
        if self.ui.button(cx, ids!(provider_delete)).clicked(actions) {
            let mut reset_slots = Vec::new();
            if let Some(mut state) = self.downloads.take() {
                match state.ctrl.delete_current_provider_profile() {
                    Ok(slots) => {
                        reset_slots = slots;
                        self.seed_provider_inputs_from_state(cx, &mut state);
                    }
                    Err(err) => {
                        state.ctrl.provider_feedback = err;
                    }
                }
                self.downloads = Some(state);
            }
            for slot in reset_slots {
                self.on_slot_routing_changed(cx, slot);
            }
            return;
        }

        // Provider preset buttons.
        if self
            .ui
            .button(cx, ids!(provider_preset_magpie))
            .clicked(actions)
        {
            if let Some(mut state) = self.downloads.take() {
                state.ctrl.apply_magpie_preset();
                self.seed_provider_inputs_from_state(cx, &mut state);
                self.downloads = Some(state);
            }
            return;
        }
        if self
            .ui
            .button(cx, ids!(provider_preset_custom))
            .clicked(actions)
        {
            if let Some(mut state) = self.downloads.take() {
                state.ctrl.apply_custom_preset();
                self.seed_provider_inputs_from_state(cx, &mut state);
                self.downloads = Some(state);
            }
            return;
        }

        // Provider text inputs.
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(provider_id_input))
            .changed(actions)
        {
            if let Some(mut state) = self.downloads.take() {
                state.ctrl.provider_draft.id = text;
                state.ctrl.provider_draft.dirty = true;
                state.ctrl.sync_provider_api_key_state();
                state.last_provider_key_rendered = self.apply_token_input(
                    cx,
                    ids!(provider_api_key_input),
                    &state.ctrl.provider_api_key,
                    state.ctrl.credentials.as_ref(),
                );
                self.downloads = Some(state);
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(provider_name_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.provider_draft.name = text;
                d.ctrl.provider_draft.dirty = true;
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(provider_base_url_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.provider_draft.base_url = text;
                d.ctrl.provider_draft.dirty = true;
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(provider_timeout_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.provider_draft.timeout_secs = text;
                d.ctrl.provider_draft.dirty = true;
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(provider_api_key_env_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.provider_draft.api_key_env = text;
                d.ctrl.provider_draft.dirty = true;
                d.ctrl.sync_provider_api_key_state();
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(provider_custom_headers_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.provider_draft.custom_headers = text;
                d.ctrl.provider_draft.dirty = true;
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(provider_api_key_input))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.provider_api_key.set_draft(text.clone());
                d.last_provider_key_rendered = text;
            }
        }

        // Provider API key reveal / save / clear (reusing 0600 CredentialStore & handle_token_buttons).
        self.handle_token_buttons(
            cx,
            actions,
            &TokenFieldIds {
                target: TokenTarget::ProviderApiKey,
                input: ids!(provider_api_key_input),
                toggle: ids!(provider_api_key_toggle),
                save: ids!(provider_api_key_save),
                clear: ids!(provider_api_key_clear),
            },
        );

        // Save provider profile / fetch /v1/models.
        if self.ui.button(cx, ids!(provider_save)).clicked(actions) {
            let d = self.downloads.as_mut().unwrap();
            if let Err(err) = d.ctrl.save_provider_draft() {
                d.ctrl.provider_feedback = format!("保存失败：{err}");
            }
        }
        if self
            .ui
            .button(cx, ids!(provider_fetch_models))
            .clicked(actions)
        {
            let d = self.downloads.as_mut().unwrap();
            d.ctrl.start_provider_probe();
        }

        // Per-slot model & voice text inputs.
        for row in slot_rows() {
            if let Some(text) = self.ui.text_input(cx, row.model).changed(actions) {
                if let Some(d) = self.downloads.as_mut() {
                    d.ctrl.set_slot_model(row.slot, text);
                }
            }
        }
        if let Some(text) = self
            .ui
            .text_input(cx, ids!(slot_tts_voice))
            .changed(actions)
        {
            if let Some(d) = self.downloads.as_mut() {
                d.ctrl.set_tts_slot_voice(text);
            }
        }

        // Per-slot Local / External toggles, provider profile selector, and quick-fill buttons.
        for row in slot_rows() {
            if self.ui.button(cx, row.local).clicked(actions) {
                if let Some(d) = self.downloads.as_mut() {
                    let _ = d.ctrl.set_slot_mode(row.slot, false);
                }
                self.on_slot_routing_changed(cx, row.slot);
            } else if self.ui.button(cx, row.external).clicked(actions) {
                let filled_model = {
                    let d = self.downloads.as_mut().unwrap();
                    if let Err(err) = d.ctrl.set_slot_mode(row.slot, true) {
                        d.ctrl.slot_feedback = format!("切换失败：{err}");
                        None
                    } else {
                        Some(d.ctrl.slot_drafts[slot_index(row.slot)].model.clone())
                    }
                };
                if let Some(model) = filled_model {
                    self.ui.text_input(cx, row.model).set_text(cx, &model);
                    self.on_slot_routing_changed(cx, row.slot);
                }
            } else if self.ui.button(cx, row.provider).clicked(actions) {
                let changed_external = {
                    let d = self.downloads.as_mut().unwrap();
                    match d.ctrl.cycle_slot_provider(row.slot) {
                        Ok(_) => d.ctrl.is_slot_external(row.slot),
                        Err(err) => {
                            d.ctrl.slot_feedback = err;
                            false
                        }
                    }
                };
                if changed_external {
                    self.on_slot_routing_changed(cx, row.slot);
                }
            } else if self.ui.button(cx, row.use_first).clicked(actions) {
                let picked = self
                    .downloads
                    .as_mut()
                    .and_then(|d| d.ctrl.use_first_discovered_model(row.slot));
                if let Some(model) = picked {
                    self.ui.text_input(cx, row.model).set_text(cx, &model);
                } else if let Some(d) = self.downloads.as_mut() {
                    d.ctrl.slot_feedback =
                        "尚未拉取到可用模型，请先点击「拉取模型 / 测试连接」".to_owned();
                }
            }
        }

        if self.ui.button(cx, ids!(slot_save_all)).clicked(actions) {
            let res = self.downloads.as_mut().unwrap().ctrl.save_all_slots();
            if let Err(err) = res {
                self.downloads.as_mut().unwrap().ctrl.slot_feedback = format!("保存失败：{err}");
            } else {
                for slot in CapabilitySlot::ALL {
                    self.on_slot_routing_changed(cx, slot);
                }
            }
        }
    }

    fn handle_token_buttons(&mut self, cx: &mut Cx, actions: &Actions, ids: &TokenFieldIds) {
        let toggle_clicked = self.ui.button(cx, ids.toggle).clicked(actions);
        let save_clicked = self.ui.button(cx, ids.save).clicked(actions);
        let clear_clicked = self.ui.button(cx, ids.clear).clicked(actions);
        if !toggle_clicked && !save_clicked && !clear_clicked {
            return;
        }
        let Some(mut state) = self.downloads.take() else {
            return;
        };
        let store = state.ctrl.credentials.clone();
        let (field, feedback, last_rendered) = match ids.target {
            TokenTarget::Hf => (
                &mut state.ctrl.hf_token,
                &mut state.ctrl.token_feedback,
                &mut state.last_hf_rendered,
            ),
            TokenTarget::ModelScope => (
                &mut state.ctrl.ms_token,
                &mut state.ctrl.token_feedback,
                &mut state.last_ms_rendered,
            ),
            TokenTarget::ProviderApiKey => (
                &mut state.ctrl.provider_api_key,
                &mut state.ctrl.provider_feedback,
                &mut state.last_provider_key_rendered,
            ),
        };
        if toggle_clicked {
            field.toggle_reveal(store.as_ref());
        } else if save_clicked {
            match field.save(store.as_ref()) {
                Ok(msg) => *feedback = msg,
                Err(err) => *feedback = err,
            }
        } else if clear_clicked {
            match field.clear(store.as_ref()) {
                Ok(msg) => *feedback = msg,
                Err(err) => *feedback = err,
            }
        }
        *last_rendered = self.apply_token_input(cx, ids.input, field, store.as_ref());
        self.downloads = Some(state);
    }

    pub fn refresh_download_controls(&mut self, cx: &mut Cx) {
        let Some(d) = self.downloads.as_ref() else {
            return;
        };

        // Top-bar settings button label + non-blocking background download banner.
        self.ui.button(cx, ids!(settings_toggle)).set_text(
            cx,
            if d.ctrl.settings_open {
                "● 模型与设置"
            } else {
                "模型与设置"
            },
        );
        let active_summary = d.ctrl.active_download_summary();
        let show_banner = active_summary.is_some() && !d.ctrl.settings_open;
        self.ui
            .view(cx, ids!(download_banner))
            .set_visible(cx, show_banner);
        if let Some(summary) = active_summary {
            self.ui
                .label(cx, ids!(download_banner_status))
                .set_text(cx, &summary);
        }

        // Paint active-provider badges on every workbench header.
        let asr_badge = format!("当前提供方：{}", d.ctrl.slot_badge(CapabilitySlot::Asr));
        self.ui
            .label(cx, ids!(asr_provider_badge))
            .set_text(cx, &asr_badge);
        let tts_badge = format!("当前提供方：{}", d.ctrl.slot_badge(CapabilitySlot::Tts));
        self.ui
            .label(cx, ids!(tts_provider_badge))
            .set_text(cx, &tts_badge);
        let enhance_badge = format!(
            "提示词增强提供方：{}",
            d.ctrl.slot_badge(CapabilitySlot::PromptEnhancement)
        );
        self.ui
            .label(cx, ids!(tts_enhance_provider_badge))
            .set_text(cx, &enhance_badge);
        let image_enhance_badge = format!(
            "提示词改写：{}",
            d.ctrl.slot_badge(CapabilitySlot::PromptEnhancement)
        );
        self.ui
            .label(cx, ids!(image_rewrite_badge))
            .set_text(cx, &image_enhance_badge);
        let tr_badge = format!(
            "翻译提供方：{}",
            d.ctrl.slot_badge(CapabilitySlot::Translation)
        );
        self.ui
            .label(cx, ids!(translation_provider_badge))
            .set_text(cx, &tr_badge);
        self.ui
            .label(cx, ids!(tts_translation_provider_badge))
            .set_text(cx, &tr_badge);

        // Refresh all 6 inline workbench recovery cards (suppressed when slot is External).
        let asr_external = d.ctrl.is_slot_external(CapabilitySlot::Asr);
        let mut asr_card = d.ctrl.recovery_card_with_error(
            CheckpointId::Confucius4R2t2,
            d.asr_missing_shown && !asr_external,
            (!asr_external)
                .then_some(d.asr_load_error.as_deref())
                .flatten(),
        );
        if asr_external {
            asr_card.visible = false;
        }
        self.paint_inline_recovery(
            cx,
            &asr_card,
            &RecoveryCardIds {
                container: ids!(model_recovery),
                info: ids!(asr_recovery_info),
                hint: ids!(asr_recovery_hint),
                download: ids!(asr_download_model),
                pause: Some(ids!(asr_download_pause)),
                cancel: Some(ids!(asr_download_cancel)),
            },
            !self.transcription.busy(),
        );

        let tts_external = d.ctrl.is_slot_external(CapabilitySlot::Tts);
        let mut tts_card = d.ctrl.recovery_card_with_error(
            CheckpointId::BreezeTts2,
            d.tts_missing_shown && !tts_external,
            (!tts_external)
                .then_some(d.tts_load_error.as_deref())
                .flatten(),
        );
        if tts_external {
            tts_card.visible = false;
        }
        self.paint_inline_recovery(
            cx,
            &tts_card,
            &RecoveryCardIds {
                container: ids!(tts_recovery),
                info: ids!(tts_recovery_info),
                hint: ids!(tts_recovery_hint),
                download: ids!(tts_download_model),
                pause: Some(ids!(tts_download_pause)),
                cancel: Some(ids!(tts_download_cancel)),
            },
            true,
        );

        let tr_external = d.ctrl.is_slot_external(CapabilitySlot::Translation);
        let mut tr_card = d.ctrl.recovery_card(
            CheckpointId::HyMt2_7b,
            d.translation_missing_shown && !tr_external,
        );
        if tr_external {
            tr_card.visible = false;
        }
        self.paint_inline_recovery(
            cx,
            &tr_card,
            &RecoveryCardIds {
                container: ids!(translation_recovery),
                info: ids!(translation_recovery_info),
                hint: ids!(translation_recovery_hint),
                download: ids!(translation_download_model),
                pause: Some(ids!(translation_download_pause)),
                cancel: Some(ids!(translation_download_cancel)),
            },
            true,
        );

        let image_card = d.ctrl.recovery_card_with_error(
            CheckpointId::QwenImage21Turbo,
            d.image_missing_shown,
            d.image_load_error.as_deref(),
        );
        self.paint_inline_recovery(
            cx,
            &image_card,
            &RecoveryCardIds {
                container: ids!(image_recovery),
                info: ids!(image_recovery_info),
                hint: ids!(image_recovery_hint),
                download: ids!(image_download_model),
                pause: Some(ids!(image_download_pause)),
                cancel: Some(ids!(image_download_cancel)),
            },
            true,
        );

        let music_id = d.music_missing_checkpoint.unwrap_or(CheckpointId::Yue2_3b);
        let music_card = d
            .ctrl
            .recovery_card(music_id, d.music_missing_checkpoint.is_some());
        self.paint_inline_recovery(
            cx,
            &music_card,
            &RecoveryCardIds {
                container: ids!(music_recovery),
                info: ids!(music_recovery_info),
                hint: ids!(music_recovery_hint),
                download: ids!(music_download_model),
                pause: Some(ids!(music_download_pause)),
                cancel: Some(ids!(music_download_cancel)),
            },
            true,
        );

        let omni_show = d.omni_missing_shown || self.omni.as_ref().is_some_and(|o| o.visible());
        let omni_card = d.ctrl.recovery_card(CheckpointId::Lfm3Omni2b, omni_show);
        self.paint_inline_recovery(
            cx,
            &omni_card,
            &RecoveryCardIds {
                container: ids!(omni_recovery),
                info: ids!(omni_recovery_info),
                hint: ids!(omni_recovery_hint),
                download: ids!(omni_download_model),
                pause: None,
                cancel: None,
            },
            true,
        );

        if !d.ctrl.settings_open {
            return;
        }

        // Settings sub-tab visibility & labels.
        let on_models = d.ctrl.settings_tab == SettingsTab::ModelsAndDownload;
        self.ui.button(cx, ids!(settings_tab_models)).set_text(
            cx,
            if on_models {
                "● 模型与下载"
            } else {
                "模型与下载"
            },
        );
        self.ui.button(cx, ids!(settings_tab_providers)).set_text(
            cx,
            if on_models {
                "外部提供方与槽位"
            } else {
                "● 外部提供方与槽位"
            },
        );
        self.ui
            .view(cx, ids!(settings_models_section))
            .set_visible(cx, on_models);
        self.ui
            .view(cx, ids!(settings_providers_section))
            .set_visible(cx, !on_models);

        if !on_models {
            self.paint_provider_settings(cx, d);
            return;
        }

        // Paint Settings panel route & proxy controls.
        let draft = &d.ctrl.route_draft;
        self.ui.button(cx, ids!(route_modelscope)).set_text(
            cx,
            if draft.route_preset == RoutePreset::ModelScope {
                "● 魔搭 ModelScope（默认）"
            } else {
                "魔搭 ModelScope"
            },
        );
        self.ui.button(cx, ids!(route_huggingface)).set_text(
            cx,
            if draft.route_preset == RoutePreset::HuggingFace {
                "● Hugging Face 官方"
            } else {
                "Hugging Face 官方"
            },
        );
        self.ui.button(cx, ids!(route_custom_mirror)).set_text(
            cx,
            if draft.route_preset == RoutePreset::CustomMirror {
                "● 自定义 HF 镜像"
            } else {
                "自定义 HF 镜像"
            },
        );

        self.ui.button(cx, ids!(fallback_prompt)).set_text(
            cx,
            if draft.fallback_preset == FallbackPreset::Prompt {
                "● 仅 HF 模型时提示切换"
            } else {
                "仅 HF 模型时提示切换"
            },
        );
        self.ui.button(cx, ids!(fallback_hf)).set_text(
            cx,
            if draft.fallback_preset == FallbackPreset::UseHuggingFace {
                "● 自动走 HF 官方"
            } else {
                "自动走 HF 官方"
            },
        );
        self.ui.button(cx, ids!(fallback_mirror)).set_text(
            cx,
            if draft.fallback_preset == FallbackPreset::UseCustomMirror {
                "● 自动走自定义镜像"
            } else {
                "自动走自定义镜像"
            },
        );

        let mirror_hint = draft.mirror_validation_hint().unwrap_or_else(|| {
            "例如 https://hf-mirror.com 或自建反向代理；切换线路时输入会被保留。".to_owned()
        });
        self.ui
            .label(cx, ids!(mirror_url_hint))
            .set_text(cx, &mirror_hint);
        let proxy_hint = draft.proxy_validation_hint().unwrap_or_else(|| {
            "留空则沿用系统 HTTPS_PROXY / HTTP_PROXY；支持 http:// 或 https:// 代理。".to_owned()
        });
        self.ui
            .label(cx, ids!(proxy_url_hint))
            .set_text(cx, &proxy_hint);

        let save_check = draft.can_save();
        self.ui
            .button(cx, ids!(settings_save_route))
            .set_action_enabled(cx, save_check.is_ok());
        self.ui
            .button(cx, ids!(settings_test_connection))
            .set_action_enabled(cx, !d.ctrl.connectivity_testing && save_check.is_ok());
        self.ui.button(cx, ids!(settings_test_connection)).set_text(
            cx,
            if d.ctrl.connectivity_testing {
                "测试中…"
            } else {
                "测试连接"
            },
        );

        let route_status = if let Err(reason) = save_check {
            format!("无法保存：{reason}")
        } else if draft.dirty {
            "有未保存的线路修改 · 可先「测试连接」或「保存线路配置」".to_owned()
        } else if !d.ctrl.route_feedback.is_empty() {
            d.ctrl.route_feedback.clone()
        } else {
            format!(
                "当前生效线路：{}",
                d.ctrl.settings.download.route.display_label()
            )
        };
        self.ui
            .label(cx, ids!(settings_save_status))
            .set_text(cx, &route_status);
        self.ui
            .label(cx, ids!(connectivity_status))
            .set_text(cx, &d.ctrl.connectivity_message);

        // Token status & button labels.
        self.ui
            .label(cx, ids!(hf_token_source))
            .set_text(cx, &d.ctrl.hf_token.source_badge());
        self.ui.button(cx, ids!(hf_token_toggle)).set_text(
            cx,
            if d.ctrl.hf_token.revealed {
                "隐藏"
            } else {
                "显示"
            },
        );
        self.ui.button(cx, ids!(hf_token_clear)).set_action_enabled(
            cx,
            d.ctrl.hf_token.has_saved_file || !d.ctrl.hf_token.draft_input.is_empty(),
        );

        self.ui
            .label(cx, ids!(ms_token_source))
            .set_text(cx, &d.ctrl.ms_token.source_badge());
        self.ui.button(cx, ids!(ms_token_toggle)).set_text(
            cx,
            if d.ctrl.ms_token.revealed {
                "隐藏"
            } else {
                "显示"
            },
        );
        self.ui.button(cx, ids!(ms_token_clear)).set_action_enabled(
            cx,
            d.ctrl.ms_token.has_saved_file || !d.ctrl.ms_token.draft_input.is_empty(),
        );

        let token_note = if d.ctrl.token_feedback.is_empty() {
            "Token 仅保存在独立 0600 凭据文件；环境变量 HF_TOKEN / MODELSCOPE_API_TOKEN 优先生效，不写入日志或下载清单。"
        } else {
            d.ctrl.token_feedback.as_str()
        };
        self.ui
            .label(cx, ids!(token_status))
            .set_text(cx, token_note);

        // Paint all 11 managed checkpoint rows in Settings.
        for row in checkpoint_rows() {
            let desc = row.id.descriptor();
            let lc = d.ctrl.lifecycle(row.id);
            let card = d.ctrl.recovery_card(row.id, true);
            let status_line = format!(
                "{} · {} · {} · {}",
                desc.panel_label,
                desc.formatted_size(),
                desc.source_badge(),
                lc.status_badge()
            );
            self.ui.label(cx, row.status).set_text(cx, &status_line);

            let show_progress = lc.is_active() || matches!(lc, CheckpointLifecycle::Paused { .. });
            self.ui
                .view(cx, row.progress)
                .set_visible(cx, show_progress);
            if show_progress {
                crate::engine_ui::set_progress(
                    &self.ui,
                    cx,
                    row.progress,
                    row.progress_fill,
                    lc.progress_fraction(),
                );
            }

            let hint = match lc {
                CheckpointLifecycle::Ready => {
                    format!("目录：{}", d.ctrl.directory(row.id).display())
                }
                _ => card.hint_text.clone(),
            };
            self.ui.label(cx, row.hint).set_text(cx, &hint);

            self.ui
                .button(cx, row.action)
                .set_text(cx, &card.download_button_text);
            self.ui
                .button(cx, row.action)
                .set_action_enabled(cx, card.download_enabled);

            self.ui
                .button(cx, row.pause)
                .set_visible(cx, card.can_pause);
            self.ui
                .button(cx, row.pause)
                .set_action_enabled(cx, card.can_pause);
            self.ui
                .button(cx, row.cancel)
                .set_visible(cx, card.can_cancel);
            self.ui
                .button(cx, row.cancel)
                .set_action_enabled(cx, card.can_cancel);
            self.ui
                .button(cx, row.choose)
                .set_action_enabled(cx, !lc.is_active() && !self.runtime.dialog_busy());
        }
    }

    fn paint_inline_recovery(
        &self,
        cx: &mut Cx,
        card: &RecoveryCardModel,
        ids: &RecoveryCardIds,
        extra_allow_download: bool,
    ) {
        self.ui
            .view(cx, ids.container)
            .set_visible(cx, card.visible);
        if !card.visible {
            return;
        }
        self.ui.label(cx, ids.info).set_text(cx, &card.info_text);
        self.ui.label(cx, ids.hint).set_text(cx, &card.hint_text);
        self.ui
            .button(cx, ids.download)
            .set_text(cx, &card.download_button_text);
        self.ui
            .button(cx, ids.download)
            .set_action_enabled(cx, card.download_enabled && extra_allow_download);
        if let Some(pause) = ids.pause {
            self.ui.button(cx, pause).set_visible(cx, card.can_pause);
            self.ui
                .button(cx, pause)
                .set_action_enabled(cx, card.can_pause);
        }
        if let Some(cancel) = ids.cancel {
            self.ui.button(cx, cancel).set_visible(cx, card.can_cancel);
            self.ui
                .button(cx, cancel)
                .set_action_enabled(cx, card.can_cancel);
        }
    }

    fn paint_provider_settings(&self, cx: &mut Cx, d: &UiState) {
        let draft = &d.ctrl.provider_draft;
        let saved_count = d.ctrl.settings.providers.profiles.len();
        let current_is_saved = d.ctrl.settings.providers.profile(draft.id.trim()).is_some();
        let can_switch_profile = saved_count > 1 || (saved_count > 0 && !current_is_saved);
        self.ui
            .label(cx, ids!(provider_profile_summary))
            .set_text(cx, &d.ctrl.provider_profile_summary());
        self.ui
            .button(cx, ids!(provider_profile_prev))
            .set_action_enabled(cx, can_switch_profile);
        self.ui
            .button(cx, ids!(provider_profile_next))
            .set_action_enabled(cx, can_switch_profile);
        self.ui
            .button(cx, ids!(provider_delete))
            .set_action_enabled(cx, current_is_saved);

        let url_hint = draft.url_validation_hint().unwrap_or_else(|| {
            "默认 Magpie 本地网关为 http://127.0.0.1:3425/v1（允许空 API Key），也可填写任意 OpenAI 兼容网关。".to_owned()
        });
        self.ui
            .label(cx, ids!(provider_url_hint))
            .set_text(cx, &url_hint);

        let key_source = if d.ctrl.provider_api_key.env_override.is_none()
            && !d.ctrl.provider_api_key.has_saved_file
        {
            "未配置（本地 Magpie 网关可留空）".to_owned()
        } else {
            d.ctrl.provider_api_key.source_badge()
        };
        self.ui
            .label(cx, ids!(provider_api_key_source))
            .set_text(cx, &key_source);
        self.ui.button(cx, ids!(provider_api_key_toggle)).set_text(
            cx,
            if d.ctrl.provider_api_key.revealed {
                "隐藏"
            } else {
                "显示"
            },
        );
        self.ui
            .button(cx, ids!(provider_api_key_clear))
            .set_action_enabled(
                cx,
                d.ctrl.provider_api_key.has_saved_file
                    || !d.ctrl.provider_api_key.draft_input.is_empty(),
            );

        let save_check = draft.can_save();
        self.ui
            .button(cx, ids!(provider_save))
            .set_action_enabled(cx, save_check.is_ok());
        self.ui
            .button(cx, ids!(provider_fetch_models))
            .set_action_enabled(cx, !d.ctrl.provider_probing && save_check.is_ok());
        self.ui.button(cx, ids!(provider_fetch_models)).set_text(
            cx,
            if d.ctrl.provider_probing {
                "查询中…"
            } else {
                "拉取模型 / 测试连接"
            },
        );

        let save_status = if let Err(reason) = save_check {
            format!("无法保存：{reason}")
        } else if !d.ctrl.provider_feedback.is_empty() {
            d.ctrl.provider_feedback.clone()
        } else if draft.dirty {
            "有未保存的网关配置 · 可先「拉取模型 / 测试连接」或「保存提供方」".to_owned()
        } else {
            format!(
                "已配置 {} 个外部提供方",
                d.ctrl.settings.providers.profiles.len()
            )
        };
        self.ui
            .label(cx, ids!(provider_save_status))
            .set_text(cx, &save_status);
        self.ui
            .label(cx, ids!(provider_probe_status))
            .set_text(cx, &d.ctrl.provider_probe_message);

        let models_summary = if draft.discovered_models.is_empty() {
            "尚未拉取模型列表 · 点击「拉取模型 / 测试连接」从 GET /v1/models 自动发现".to_owned()
        } else {
            let preview: Vec<&str> = draft
                .discovered_models
                .iter()
                .take(8)
                .map(String::as_str)
                .collect();
            format!(
                "已发现 {} 个模型：{}{}",
                draft.discovered_models.len(),
                preview.join("、"),
                if draft.discovered_models.len() > 8 {
                    "…"
                } else {
                    ""
                }
            )
        };
        self.ui
            .label(cx, ids!(provider_models_summary))
            .set_text(cx, &models_summary);

        let any_models = !draft.discovered_models.is_empty()
            || d.ctrl
                .settings
                .providers
                .profiles
                .iter()
                .any(|p| !p.discovered_models.is_empty());
        for row in slot_rows() {
            let ext = d.ctrl.is_slot_external(row.slot);
            self.ui
                .label(cx, row.badge)
                .set_text(cx, &d.ctrl.slot_badge(row.slot));
            self.ui.button(cx, row.local).set_text(
                cx,
                if ext {
                    "本地模型"
                } else {
                    "● 本地模型"
                },
            );
            self.ui.button(cx, row.external).set_text(
                cx,
                if ext {
                    "● 外部网关"
                } else {
                    "外部网关"
                },
            );
            self.ui
                .button(cx, row.provider)
                .set_text(cx, &d.ctrl.slot_provider_label(row.slot));
            self.ui
                .button(cx, row.provider)
                .set_action_enabled(cx, saved_count > 1);
            self.ui
                .button(cx, row.use_first)
                .set_action_enabled(cx, any_models);
        }

        let slot_status = if d.ctrl.slot_feedback.is_empty() {
            "4 个能力槽位互不干扰，修改后立即生效并保存到 app-settings.json。".to_owned()
        } else {
            d.ctrl.slot_feedback.clone()
        };
        self.ui
            .label(cx, ids!(slot_save_status))
            .set_text(cx, &slot_status);
    }
}
