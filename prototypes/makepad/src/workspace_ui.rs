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
    narrow: Option<bool>,
    asr_details: bool,
    tts_details: bool,
    music_details: bool,
}

impl State {
    pub fn language(&self) -> bool {
        self.page == Page::Language
    }

    pub fn creative(&self) -> bool {
        self.page == Page::Creative
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

    pub fn handle_workspace_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        let page = if self.ui.button(cx, ids!(language_tab)).clicked(actions) {
            Some(Page::Language)
        } else if self.ui.button(cx, ids!(creative_tab)).clicked(actions) {
            Some(Page::Creative)
        } else {
            None
        };
        if let Some(page) = page {
            if self.workspace.page != page {
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
                        "语言工作台 · 当前"
                    } else {
                        "语言工作台"
                    },
                );
                self.ui.button(cx, ids!(creative_tab)).set_text(
                    cx,
                    if self.workspace.creative() {
                        "创作工作台 · 当前"
                    } else {
                        "创作工作台"
                    },
                );
            }
        }

        for (button, panel, open, closed, expanded) in [
            (
                ids!(asr_details_toggle),
                ids!(asr_details),
                &mut self.workspace.asr_details,
                "转写设置",
                "收起转写设置",
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
                "音乐分析 / 转谱 / ABC",
                "收起音乐工具",
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
            if self.ui.button(cx, ids!(tts_load)).clicked(actions) {
                self.activate_tts(cx);
            }
        } else if self.ui.button(cx, ids!(image_load)).clicked(actions) {
            self.activate_image(cx);
        }
    }
}
