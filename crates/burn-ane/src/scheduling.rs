use burn_backend::{Device, DeviceId};
use burn_ir::BackendIr;
use std::{any::TypeId, cell::RefCell, marker::PhantomData, rc::Rc};

type Key = (TypeId, DeviceId);
thread_local! {
    static SCOPES: RefCell<Vec<Key>> = const { RefCell::new(Vec::new()) };
}

/// RAII routing scope. It cannot cross threads because the routing preference
/// belongs to the submitting thread, not the shared inference session.
pub struct FallbackScope {
    key: Key,
    _not_send: PhantomData<Rc<()>>,
}

pub(crate) fn active<B: BackendIr>(device: &crate::AneDevice<B>) -> bool {
    SCOPES.with(|scopes| {
        scopes
            .borrow()
            .contains(&(TypeId::of::<B>(), device.to_id()))
    })
}

pub(crate) fn enter<B: BackendIr>(device: &crate::AneDevice<B>) -> FallbackScope {
    let key = (TypeId::of::<B>(), device.to_id());
    SCOPES.with(|scopes| scopes.borrow_mut().push(key));
    FallbackScope {
        key,
        _not_send: PhantomData,
    }
}

impl Drop for FallbackScope {
    fn drop(&mut self) {
        SCOPES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            let index = scopes
                .iter()
                .rposition(|key| *key == self.key)
                .expect("live fallback scope");
            scopes.remove(index);
        });
    }
}
