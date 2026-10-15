//! One model operation at a time, independent of pages, playback and exports.
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
#[derive(Default, Clone)]
pub struct Runtime(Arc<Mutex<State>>);
#[derive(Default)]
struct State {
    next: u64,
    active: Option<Active>,
    dialog: Option<&'static str>,
}
struct Active {
    id: u64,
    name: &'static str,
    stop: Arc<AtomicBool>,
}
pub struct Lease {
    runtime: Runtime,
    id: u64,
}
pub struct Dialog {
    runtime: Runtime,
    owner: &'static str,
}
impl Runtime {
    pub fn busy(&self) -> bool {
        self.0.lock().unwrap().active.is_some()
    }
    pub fn snapshot(&self) -> Option<(&'static str, bool)> {
        self.0
            .lock()
            .unwrap()
            .active
            .as_ref()
            .map(|a| (a.name, a.stop.load(Ordering::Acquire)))
    }
    pub fn stop(&self) {
        if let Some(active) = &self.0.lock().unwrap().active {
            active.stop.store(true, Ordering::Release);
        }
    }
    pub fn acquire(&self, name: &'static str, stop: Arc<AtomicBool>) -> Option<Lease> {
        let mut state = self.0.lock().unwrap();
        if state.active.is_some() {
            return None;
        }
        state.next += 1;
        let id = state.next;
        stop.store(false, Ordering::Release);
        state.active = Some(Active { id, name, stop });
        Some(Lease {
            runtime: self.clone(),
            id,
        })
    }
    pub fn dialog_busy(&self) -> bool {
        self.0.lock().unwrap().dialog.is_some()
    }
    pub fn dialog(&self, owner: &'static str) -> Option<Dialog> {
        let mut state = self.0.lock().unwrap();
        if state.dialog.is_some() {
            return None;
        }
        state.dialog = Some(owner);
        Some(Dialog {
            runtime: self.clone(),
            owner,
        })
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.runtime.0.lock().unwrap();
        if state.active.as_ref().is_some_and(|a| a.id == self.id) {
            state.active = None;
        }
    }
}
impl Drop for Dialog {
    fn drop(&mut self) {
        let mut state = self.runtime.0.lock().unwrap();
        if state.dialog == Some(self.owner) {
            state.dialog = None;
        }
    }
}
