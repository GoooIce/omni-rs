use crate::ActionButtonState;
use crate::App;
use makepad_widgets::*;

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    #[default]
    Language,
    Creative,
}

struct Slide {
    from_x: f64,
    from_w: f64,
    started: std::time::Instant,
    to_creative: bool,
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
    nav_timer: Timer,
    slide: Option<Slide>,
    nav_placed: bool,
    dot: Page,
}

impl State {
    pub fn page(&self) -> Page {
        self.page
    }

    pub fn pending(&self) -> Option<Page> {
        self.pending
    }

    /// The tab the indicator is traveling toward. A pending page wins over the
    /// page still on screen, so the dot can leave before the transition lands.
    pub fn indicator(&self) -> Page {
        self.pending.unwrap_or(self.page)
    }

    /// Ask for another page. The current page stays until `land`. Repeating the
    /// page the indicator already names does not start a second transition.
    pub fn request(&mut self, page: Page) -> bool {
        if self.indicator() == page {
            return false;
        }
        self.pending = Some(page);
        true
    }

    pub fn land(&mut self) {
        if let Some(page) = self.pending.take() {
            self.page = page;
        }
    }

    pub fn language(&self) -> bool {
        self.page == Page::Language
    }

    pub fn creative(&self) -> bool {
        self.page == Page::Creative
    }

    pub fn indicator_language(&self) -> bool {
        self.indicator() == Page::Language
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
            .set_action_enabled(cx, !self.transcription.transcript().is_empty());
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

    fn paint_tabs(&mut self, cx: &mut Cx) {
        let language = self.workspace.indicator_language();
        self.workspace.dot = self.workspace.indicator();
        self.ui
            .button(cx, ids!(language_tab))
            .set_text(cx, if language { "● 语言" } else { "语言" });
        self.ui
            .button(cx, ids!(creative_tab))
            .set_text(cx, if language { "创作" } else { "● 创作" });
    }

    fn show_settled_page(&mut self, cx: &mut Cx) {
        self.ui
            .view(cx, ids!(language_page))
            .set_visible(cx, self.workspace.language());
        self.ui
            .view(cx, ids!(creative_page))
            .set_visible(cx, self.workspace.creative());
        self.ui
            .view(cx, ids!(pages))
            .set_scroll_pos(cx, Vec2d::default());
        self.paint_tabs(cx);
    }

    /// Move the dot onto the destination while both pages are still on the track.
    pub fn reveal_pending_page(&mut self, cx: &mut Cx) {
        if self.workspace.pending().is_none() || self.workspace.dot == self.workspace.indicator() {
            return;
        }
        self.paint_tabs(cx);
    }

    pub fn apply_pending_page(&mut self, cx: &mut Cx) {
        if self.workspace.pending().is_none() {
            return;
        }
        self.workspace.land();
        self.show_settled_page(cx);
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
            if self.workspace.request(page) && !self.follow_tab_indicator(cx, true) {
                self.apply_pending_page(cx);
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
                Some((
                    self.transcription.transcript().to_owned(),
                    "已填入转写文字 · 可编辑后朗读",
                ))
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

    pub fn step_workspace_slide(&mut self, cx: &mut Cx, event: &Event) {
        if self.workspace.nav_timer.is_event(event).is_some() {
            self.step_tab_indicator(cx);
        }
    }

    pub fn place_workspace_indicator(&mut self, cx: &mut Cx, event: &Event) {
        if let Event::WindowGeomChange(geometry) = event {
            self.refresh_workspace_layout(cx, geometry.new_geom.inner_size.x);
            self.follow_tab_indicator(cx, false);
        }
        if self.workspace.slide.is_none() {
            self.follow_tab_indicator(cx, false);
        }
    }

    fn tab_indicator_target(&self, cx: &mut Cx) -> Option<(f64, f64)> {
        let nav = self.ui.widget(cx, ids!(navigation)).area().rect(cx);
        let id = if self.workspace.indicator_language() {
            ids!(language_tab)
        } else {
            ids!(creative_tab)
        };
        let tab = self.ui.widget(cx, id).area().rect(cx);
        if nav.size.x < 1.0 || tab.size.x < 1.0 {
            return None;
        }
        Some(((tab.pos.x - nav.pos.x).max(0.0), tab.size.x))
    }

    fn place_tab_indicator(&self, cx: &mut Cx, x: f64, width: f64) {
        if let Some(mut view) = self.ui.view(cx, ids!(tab_indicator)).borrow_mut() {
            view.walk.margin.left = x;
            view.walk.width = Size::Fixed(width.max(28.0));
            view.redraw(cx);
        }
    }

    fn follow_tab_indicator(&mut self, cx: &mut Cx, animate: bool) -> bool {
        let Some((x, width)) = self.tab_indicator_target(cx) else {
            return false;
        };
        let current = self
            .ui
            .view(cx, ids!(tab_indicator))
            .borrow()
            .map(|view| {
                let width = match view.walk.width {
                    Size::Fixed(width) => width,
                    _ => 48.0,
                };
                (view.walk.margin.left, width)
            })
            .unwrap_or((0.0, 48.0));
        if animate && self.workspace.nav_placed {
            self.workspace.slide = Some(Slide {
                from_x: current.0,
                from_w: current.1,
                started: std::time::Instant::now(),
                to_creative: !self.workspace.indicator_language(),
            });
            self.workspace.nav_timer = cx.start_interval(1.0 / 60.0);
            return true;
        }
        if self.workspace.nav_placed
            && (current.0 - x).abs() < 0.5
            && (current.1 - width).abs() < 0.5
        {
            return false;
        }
        self.place_tab_indicator(cx, x, width);
        self.workspace.nav_placed = true;
        false
    }

    fn step_tab_indicator(&mut self, cx: &mut Cx) {
        let Some(slide) = &self.workspace.slide else {
            cx.stop_timer(self.workspace.nav_timer);
            return;
        };
        let started = slide.started;
        let from_x = slide.from_x;
        let from_w = slide.from_w;
        let to_creative = slide.to_creative;
        let Some((x, width)) = self.tab_indicator_target(cx) else {
            return;
        };
        let t = (started.elapsed().as_secs_f64() / 0.28).clamp(0.0, 1.0);
        let page_eased = 1.0 - (1.0 - t) * (1.0 - t);
        let line_eased = (page_eased * 1.7).min(1.0);
        self.place_tab_indicator(
            cx,
            from_x + (x - from_x) * line_eased,
            from_w + (width - from_w) * line_eased,
        );
        self.slide_pages(cx, page_eased, to_creative);
        if t > 0.0 {
            self.reveal_pending_page(cx);
        }
        if t >= 1.0 {
            self.workspace.slide = None;
            cx.stop_timer(self.workspace.nav_timer);
            self.apply_pending_page(cx);
        }
    }

    fn slide_pages(&mut self, cx: &mut Cx, eased: f64, to_creative: bool) {
        let pages = self.ui.widget(cx, ids!(pages)).area().rect(cx);
        let width = pages.size.x;
        if width < 40.0 {
            return;
        }
        let shift = if to_creative {
            -width * eased
        } else {
            -width * (1.0 - eased)
        };
        self.ui.view(cx, ids!(language_page)).set_visible(cx, true);
        self.ui.view(cx, ids!(creative_page)).set_visible(cx, true);
        for id in [ids!(language_page), ids!(creative_page)] {
            if let Some(mut view) = self.ui.view(cx, id).borrow_mut() {
                view.walk.width = Size::Fixed(width);
                view.redraw(cx);
            }
        }
        if let Some(mut track) = self.ui.view(cx, ids!(page_track)).borrow_mut() {
            track.walk.width = Size::Fixed(width * 2.0);
            track.walk.margin.left = shift;
            track.redraw(cx);
        }
        self.place_slide_lips(
            cx,
            true,
            pages.pos.x,
            pages.pos.y,
            width,
            pages.size.y,
            shift,
        );
    }

    fn place_slide_lips(
        &mut self,
        cx: &mut Cx,
        show: bool,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        shift: f64,
    ) {
        let mut slivers: Vec<(f64, f64)> = Vec::new();
        if show {
            let gap = 14.0;
            let wide = (width - gap) * 170.0 / 270.0;
            let narrow = width - gap - wide;
            let view_left = x;
            let view_right = x + width;
            let lang_x = x + shift;
            let crea_x = lang_x + width;
            for (card_x, card_w) in [
                (lang_x, wide),
                (lang_x + wide + gap, narrow),
                (crea_x, narrow),
                (crea_x + narrow + gap, wide),
            ] {
                let card_right = card_x + card_w;
                let overlaps = card_right > view_left + 1.0 && card_x < view_right - 1.0;
                let full = card_x >= view_left - 1.0 && card_right <= view_right + 1.0;
                if overlaps && !full {
                    let left = card_x.max(view_left);
                    let right = card_right.min(view_right);
                    if right - left > 1.0 {
                        slivers.push((left, right));
                    }
                }
            }
        }
        slivers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let slots = [ids!(slide_lip_left), ids!(slide_lip_right)];
        for (index, id) in slots.into_iter().enumerate() {
            let sliver = if index == 0 {
                slivers.first()
            } else {
                slivers.get(slivers.len().saturating_sub(1))
            };
            let distinct = index == 0 || slivers.len() > 1;
            self.ui
                .view(cx, id)
                .set_visible(cx, sliver.is_some() && distinct);
            if let Some((left, right)) = sliver.filter(|_| distinct) {
                if let Some(mut view) = self.ui.view(cx, id).borrow_mut() {
                    view.walk.abs_pos = Some(dvec2(*left, y));
                    view.walk.width = Size::Fixed((*right - *left).max(1.0));
                    view.walk.height = Size::Fixed(height.max(200.0));
                    view.redraw(cx);
                }
            }
        }
    }

    fn rest_page_track(&mut self, cx: &mut Cx) {
        for id in [ids!(language_page), ids!(creative_page)] {
            if let Some(mut view) = self.ui.view(cx, id).borrow_mut() {
                view.walk.width = Size::fill();
                view.redraw(cx);
            }
        }
        if let Some(mut track) = self.ui.view(cx, ids!(page_track)).borrow_mut() {
            track.walk.width = Size::fill();
            track.walk.margin.left = 0.0;
            track.redraw(cx);
        }
        let pages = self.ui.widget(cx, ids!(pages)).area().rect(cx);
        self.place_slide_lips(
            cx,
            false,
            pages.pos.x,
            pages.pos.y,
            pages.size.x,
            pages.size.y,
            0.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indicator_points_at_the_requested_page_while_that_page_is_still_pending() {
        let mut workspace = State::default();
        assert_eq!(workspace.page(), Page::Language);
        assert_eq!(workspace.pending(), None);
        assert_eq!(workspace.indicator(), Page::Language);

        assert!(workspace.request(Page::Creative));
        assert_eq!(workspace.page(), Page::Language);
        assert_eq!(workspace.pending(), Some(Page::Creative));
        assert_eq!(workspace.indicator(), Page::Creative);

        workspace.land();
        assert_eq!(workspace.page(), Page::Creative);
        assert_eq!(workspace.pending(), None);
        assert_eq!(workspace.indicator(), Page::Creative);

        assert!(workspace.request(Page::Language));
        assert_eq!(workspace.page(), Page::Creative);
        assert_eq!(workspace.pending(), Some(Page::Language));
        assert_eq!(workspace.indicator(), Page::Language);
    }
}
