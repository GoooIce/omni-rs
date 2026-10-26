//! Platform probe: which runtime class implements the private selector the
//! ML-encoder pipeline expects (`_MTLMLLibrary`, the Core ML-model library).
use objc2::runtime::AnyClass;
use std::ffi::CStr;

unsafe extern "C" {
    fn objc_copyClassList(out_count: *mut u32) -> *mut *mut AnyClass;
    fn class_copyMethodList(cls: &AnyClass, out_count: *mut u32) -> *mut *mut std::ffi::c_void;
    fn method_getName(m: *mut std::ffi::c_void) -> *const i8;
    fn sel_getName(sel: *const i8) -> *const i8;
}

#[test]
fn hunt_executable_selector() {
    unsafe {
        let mut count: u32 = 0;
        let list = objc_copyClassList(&mut count);
        let target = b"executableWithDeviceSelection:\0";
        let mut hits: Vec<String> = Vec::new();
        for i in 0..count as usize {
            let class = &*(*list.add(i));
            let mut mcount: u32 = 0;
            let methods = class_copyMethodList(class, &mut mcount);
            if methods.is_null() {
                continue;
            }
            for m in 0..mcount as usize {
                let method = *methods.add(m);
                let name_ptr = sel_getName(method_getName(method));
                let name = CStr::from_ptr(name_ptr);
                if name.to_bytes() == &target[..target.len() - 1] {
                    hits.push(class.name().to_str().unwrap_or("?").to_string());
                }
            }
        }
        eprintln!("classes implementing executableWithDeviceSelection:: {hits:?}");
    }
}
