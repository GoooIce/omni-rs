use makepad_widgets::{makepad_platform::makepad_wasm_bridge::*, *};

#[derive(FromWasm)]
pub struct FromWasmCommand {
    pub json: String,
}

#[derive(ToWasm)]
pub struct ToWasmUpdate {
    pub json: String,
}

// Called by our JS subclass before Makepad creates its typed message bridge.
// The pointer is the Cx owned by wasm_create_app; it never crosses into Burn.
#[no_mangle]
pub unsafe extern "C" fn r2t2_register_bridge(cx_ptr: u32) {
    let cx = &mut *(cx_ptr as *mut Cx);
    cx.os.append_to_wasm_js(&[ToWasmUpdate::to_js_code()]);
    cx.os.append_from_wasm_js(&[FromWasmCommand::to_js_code()]);
}
