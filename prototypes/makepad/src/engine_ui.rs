//! Shared presentation for native engine loading; no model state lives here.
use makepad_widgets::*;

pub fn set_progress(ui: &WidgetRef, cx: &mut Cx, track: &[LiveId], fill: &[LiveId], progress: f64) {
    if let Some(mut fill) = ui.view(cx, fill).borrow_mut() {
        fill.walk.width = Size::Rel {
            base: Base::Parent,
            factor: progress.clamp(0.0, 1.0),
        };
    }
    ui.view(cx, track).redraw(cx);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn choose_directory(cx: &mut Cx, current: &std::path::Path, title: &str) {
    let mut dialog = FileDialog::new().set_title(title.into());
    if let Some(parent) = current.ancestors().find(|p| p.is_dir()) {
        dialog = dialog.set_location(parent.to_owned());
    }
    cx.open_select_folder_dialog(dialog);
}
