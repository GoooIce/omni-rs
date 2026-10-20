use crate::ActionButtonState;
use crate::App;
use makepad_widgets::*;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Page {
    #[default]
    Language,
    Creative,
}

#[derive(Default)]
pub struct State {
    page: Page,
    pending: Option<Page>,
    narrow: Option<bool>,
    asr_details: bool,
    tts_details: bool,
    music_details: bool,
    transcript_height: f64,
    clip_passes: u8,
}

impl State {
    pub fn language(&self) -> bool {
        self.page == Page::Language
    }

    pub fn creative(&self) -> bool {
        self.page == Page::Creative
    }

    pub fn indicator_language(&self) -> bool {
        match self.pending {
            Some(Page::Language) => true,
            Some(Page::Creative) => false,
            None => self.page == Page::Language,
        }
    }
}

impl App {
    pub fn refresh_workspace_layout(&mut self, cx: &mut Cx, width: f64) {
        let narrow = width < 900.0;
        if self.workspace.narrow == Some(narrow) {
            return;
        }
        self.workspace.narrow = Some(narrow);
        for id in [ids!(language_columns), ids!(creative_columns)] {
            if let Some(mut view) = self.ui.view(cx, id).borrow_mut() {
                view.layout.flow = if narrow { Flow::Down } else { Flow::right() };
                view.redraw(cx);
            }
        }
        for id in [ids!(language_stack_gap), ids!(creative_stack_gap)] {
            self.ui.view(cx, id).set_visible(cx, false);
        }
        if !self.workspace.asr_details {
            let height = if narrow { 128.0 } else { 188.0 };
            self.workspace.transcript_height = height;
            self.ui
                .text_input(cx, ids!(transcript))
                .set_height(cx, Size::Fixed(height));
        }
        self.ui
            .text_input(cx, ids!(music_lyrics))
            .set_height(cx, Size::Fixed(if narrow { 76.0 } else { 112.0 }));
        self.ui
            .text_input(cx, ids!(music_style))
            .set_height(cx, Size::Fixed(if narrow { 48.0 } else { 76.0 }));
    }

    pub fn reveal_tts_details(&mut self, cx: &mut Cx) {
        self.workspace.tts_details = true;
        self.ui.view(cx, ids!(tts_details)).set_visible(cx, true);
        self.ui
            .button(cx, ids!(tts_details_toggle))
            .set_text(cx, "收起声音设置");
    }

    pub fn refresh_workspace_controls(&self, cx: &mut Cx) {
        self.ui
            .button(cx, ids!(asr_to_tts))
            .set_action_enabled(cx, !self.transcript.is_empty());
        self.ui
            .button(cx, ids!(asr_translation_to_tts))
            .set_action_enabled(
                cx,
                !self
                    .ui
                    .text_input(cx, ids!(asr_translation))
                    .text()
                    .is_empty(),
            );
    }

    fn show_page(&mut self, cx: &mut Cx, page: Page) {
        self.workspace.page = page;
        self.ui
            .view(cx, ids!(language_page))
            .set_visible(cx, self.workspace.language());
        self.ui
            .view(cx, ids!(creative_page))
            .set_visible(cx, self.workspace.creative());
        self.ui
            .view(cx, ids!(pages))
            .set_scroll_pos(cx, Vec2d::default());
        self.ui.button(cx, ids!(language_tab)).set_text(
            cx,
            if self.workspace.language() {
                "● 语言"
            } else {
                "语言"
            },
        );
        self.ui.button(cx, ids!(creative_tab)).set_text(
            cx,
            if self.workspace.creative() {
                "● 创作"
            } else {
                "创作"
            },
        );
    }

    /// Move the dot onto the destination while both pages are still on the track.
    pub fn reveal_pending_page(&mut self, cx: &mut Cx) {
        let Some(page) = self.workspace.pending else {
            return;
        };
        if self.workspace.page == page {
            return;
        }
        self.workspace.page = page;
        self.ui.button(cx, ids!(language_tab)).set_text(
            cx,
            if self.workspace.language() {
                "● 语言"
            } else {
                "语言"
            },
        );
        self.ui.button(cx, ids!(creative_tab)).set_text(
            cx,
            if self.workspace.creative() {
                "● 创作"
            } else {
                "创作"
            },
        );
    }

    pub fn apply_pending_page(&mut self, cx: &mut Cx) {
        let Some(page) = self.workspace.pending.take() else {
            return;
        };
        self.show_page(cx, page);
        self.rest_page_track(cx);
        self.follow_tab_indicator(cx, false);
    }

    pub fn handle_workspace_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        let page = if self.ui.button(cx, ids!(language_tab)).clicked(actions) {
            Some(Page::Language)
        } else if self.ui.button(cx, ids!(creative_tab)).clicked(actions) {
            Some(Page::Creative)
        } else {
            None
        };
        if let Some(page) = page {
            if self.workspace.page != page && self.workspace.pending != Some(page) {
                self.workspace.pending = Some(page);
                if !self.follow_tab_indicator(cx, true) {
                    self.apply_pending_page(cx);
                }
            }
        }

        for (button, panel, open, closed, expanded) in [
            (
                ids!(asr_details_toggle),
                ids!(asr_details),
                &mut self.workspace.asr_details,
                "识别设置",
                "收起识别设置",
            ),
            (
                ids!(tts_details_toggle),
                ids!(tts_details),
                &mut self.workspace.tts_details,
                "声音设置",
                "收起声音设置",
            ),
            (
                ids!(music_details_toggle),
                ids!(music_details),
                &mut self.workspace.music_details,
                "从录音转谱",
                "收起转谱",
            ),
        ] {
            if self.ui.button(cx, button).clicked(actions) {
                *open = !*open;
                self.ui.view(cx, panel).set_visible(cx, *open);
                self.ui
                    .button(cx, button)
                    .set_text(cx, if *open { expanded } else { closed });
            }
        }
        if self
            .ui
            .button(cx, ids!(asr_details_toggle))
            .clicked(actions)
        {
            self.sync_asr_details_room(cx);
        }

        if self
            .ui
            .button(cx, ids!(tts_translate_toggle))
            .clicked(actions)
        {
            let open = !self.ui.view(cx, ids!(tts_translation_panel)).visible();
            self.ui
                .view(cx, ids!(tts_translation_panel))
                .set_visible(cx, open);
            self.ui.button(cx, ids!(tts_translate_toggle)).set_text(
                cx,
                if open {
                    "收起这段翻译"
                } else {
                    "翻译这段朗读"
                },
            );
        }

        if self.workspace.language() {
            let transfer = if self.ui.button(cx, ids!(asr_to_tts)).clicked(actions) {
                Some((self.transcript.clone(), "已填入转写文字 · 可编辑后朗读"))
            } else if self
                .ui
                .button(cx, ids!(asr_translation_to_tts))
                .clicked(actions)
            {
                Some((
                    self.ui.text_input(cx, ids!(asr_translation)).text(),
                    "已填入译文 · 可编辑后朗读",
                ))
            } else {
                None
            };
            if let Some((text, message)) = transfer.filter(|(text, _)| !text.is_empty()) {
                self.ui.text_input(cx, ids!(tts_text)).set_text(cx, &text);
                self.ui.label(cx, ids!(tts_status)).set_text(cx, message);
            }
            if self.ui.button(cx, ids!(tts_load)).clicked(actions)
                || (self.ui.button(cx, ids!(tts_synthesize)).clicked(actions)
                    && self.tts.as_ref().is_some_and(|state| state.needs_model()))
            {
                self.activate_tts(cx);
            }
        } else if self.ui.button(cx, ids!(image_load)).clicked(actions)
            || (self.ui.button(cx, ids!(image_generate)).clicked(actions)
                && self.image.as_ref().is_some_and(|state| state.needs_model()))
        {
            self.activate_image(cx);
        }
    }

    pub fn asr_clip_pending(&self) -> bool {
        self.workspace.clip_passes > 0
    }

    pub fn sync_asr_details_room(&mut self, cx: &mut Cx) {
        let open = self.workspace.asr_details;
        let closed = if self.workspace.narrow == Some(true) {
            128.0
        } else {
            188.0
        };
        self.workspace.transcript_height = if open { 44.0 } else { closed };
        self.ui
            .text_input(cx, ids!(transcript))
            .set_height(cx, Size::Fixed(self.workspace.transcript_height));
        self.ui
            .text_input(cx, ids!(asr_translation))
            .set_height(cx, Size::Fixed(56.0));
        if let Some(mut pad) = self.ui.view(cx, ids!(asr_details_pad)).borrow_mut() {
            pad.walk.height = Size::Fixed(0.0);
            pad.redraw(cx);
        }
        self.workspace.clip_passes = 0;
        if open {
            self.ui
                .view(cx, ids!(pages))
                .set_scroll_pos(cx, Vec2d::default());
            cx.stop_timer(self.timer);
            self.timer = cx.start_interval(1.0 / 30.0);
        }
    }

    /// Pull the open recognition settings up until the viewport edge falls
    /// between widgets. A leftover sliver is pushed below the fold.
    pub fn settle_asr_clip(&mut self, cx: &mut Cx) {
        if self.workspace.clip_passes == 0 || !self.workspace.asr_details {
            return;
        }
        let transcript = self.ui.widget(cx, ids!(transcript)).area().rect(cx);
        if transcript.size.y < 8.0
            || (transcript.size.y - self.workspace.transcript_height).abs() > 8.0
        {
            return;
        }
        let pages = self.ui.widget(cx, ids!(pages)).area().rect(cx);
        if pages.size.y < 40.0 {
            return;
        }
        self.workspace.clip_passes -= 1;
        let bottom = pages.pos.y + pages.size.y - 2.0;
        let watched = [
            ids!(asr_translate),
            ids!(asr_translation),
            ids!(asr_details_title),
            ids!(audio_path_name),
            ids!(audio_path),
            ids!(asr_other_language),
        ];
        let mut straddler: Option<(f64, f64)> = None;
        for id in watched {
            let rect = self.ui.widget(cx, id).area().rect(cx);
            if rect.size.y < 2.0 {
                continue;
            }
            let top = rect.pos.y;
            let end = top + rect.size.y;
            let touches = end >= bottom - 6.0 && top < bottom - 10.0;
            if top < bottom && (end > bottom + 1.0 || touches) {
                let closer = straddler.map(|(prev, _)| top < prev).unwrap_or(true);
                if closer {
                    straddler = Some((top, rect.size.y));
                }
            }
        }
        let Some((top, height)) = straddler else {
            self.workspace.clip_passes = 0;
            return;
        };
        let end = top + height;
        let visible = (bottom - top).max(0.0);
        let hidden = if end > bottom + 1.0 {
            (end - bottom).max(22.0)
        } else {
            22.0
        };
        let floor = 32.0;
        let next = (self.workspace.transcript_height - hidden - 6.0).max(floor);
        if next < self.workspace.transcript_height - 0.5 {
            self.workspace.transcript_height = next;
            self.ui
                .text_input(cx, ids!(transcript))
                .set_height(cx, Size::Fixed(next));
            self.workspace.clip_passes = self.workspace.clip_passes.max(1);
            return;
        }
        if visible < 18.0 {
            if let Some(mut pad) = self.ui.view(cx, ids!(asr_details_pad)).borrow_mut() {
                let current = match pad.walk.height {
                    Size::Fixed(value) => value,
                    _ => 0.0,
                };
                pad.walk.height = Size::Fixed(current + visible + 8.0);
                pad.redraw(cx);
            }
            self.workspace.clip_passes = self.workspace.clip_passes.max(1);
        }
    }
}
