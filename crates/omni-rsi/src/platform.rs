//! Platform signals for conservative idle admission; unknown signals block work.
use crate::idle::IdleSnapshot;
use std::{collections::HashMap, path::Path};

/// Parse process snapshots while ignoring this scheduler's own worker tree.
/// Missing or malformed information fails closed rather than declaring idle.
pub fn competing_process_from_ps(output: &str, current: u32) -> Option<bool> {
    let mut rows = Vec::new();
    let mut parents = HashMap::new();
    for line in output.lines().filter(|line| !line.trim().is_empty()) {
        let mut words = line.split_whitespace();
        let pid: u32 = words.next()?.parse().ok()?;
        let parent: u32 = words.next()?.parse().ok()?;
        let executable = words.next()?;
        if parents.insert(pid, parent).is_some() {
            return None;
        }
        rows.push((pid, executable, line));
    }
    if !parents.contains_key(&current) {
        return None;
    }
    for (pid, executable, line) in rows {
        let mut ancestor = pid;
        let mut own = false;
        let mut resolved = false;
        for _ in 0..=parents.len() {
            if ancestor == current {
                own = true;
                resolved = true;
                break;
            }
            match parents.get(&ancestor).copied() {
                Some(0) | None => {
                    resolved = true;
                    break;
                }
                Some(parent) => ancestor = parent,
            }
        }
        if !resolved {
            return None;
        }
        if own {
            continue;
        }
        let name = Path::new(executable).file_name()?.to_str()?;
        if matches!(
            name,
            "omni-api" | "r2t2-makepad" | "r2t2" | "breeze-tts" | "lfm3-omni" | "qwen-image"
        ) || (name.starts_with("python") && line.contains("ddust"))
        {
            return Some(true);
        }
    }
    Some(false)
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{IdleSnapshot, competing_process_from_ps};
    use std::{
        ffi::{CStr, CString, c_char, c_void},
        os::unix::ffi::OsStrExt,
        path::Path,
        process::Command,
    };

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(source: i32, event: u32) -> f64;
    }
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOPSCopyPowerSourcesInfo() -> *const c_void;
        fn IOPSGetProvidingPowerSourceType(info: *const c_void) -> *const c_void;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringGetCString(
            value: *const c_void,
            buffer: *mut c_char,
            size: isize,
            encoding: u32,
        ) -> bool;
        fn CFRelease(value: *const c_void);
    }
    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}
    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }

    fn power() -> Option<bool> {
        unsafe {
            let info = IOPSCopyPowerSourcesInfo();
            if info.is_null() {
                return None;
            }
            let kind = IOPSGetProvidingPowerSourceType(info);
            let mut buffer = [0 as c_char; 64];
            let valid = !kind.is_null()
                && CFStringGetCString(
                    kind,
                    buffer.as_mut_ptr(),
                    buffer.len() as isize,
                    0x0800_0100,
                );
            let result = valid.then(|| CStr::from_ptr(buffer.as_ptr()).to_bytes() == b"AC Power");
            CFRelease(info);
            result
        }
    }

    fn process_info() -> (Option<bool>, Option<bool>) {
        unsafe {
            let class = objc_getClass(c"NSProcessInfo".as_ptr());
            if class.is_null() {
                return (None, None);
            }
            let send_object: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(objc_msgSend as *const ());
            let send_integer: unsafe extern "C" fn(*mut c_void, *mut c_void) -> isize =
                std::mem::transmute(objc_msgSend as *const ());
            let send_bool: unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool =
                std::mem::transmute(objc_msgSend as *const ());
            let responds: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> bool =
                std::mem::transmute(objc_msgSend as *const ());
            let info = send_object(class, sel_registerName(c"processInfo".as_ptr()));
            if info.is_null() {
                return (None, None);
            }
            let supports = sel_registerName(c"respondsToSelector:".as_ptr());
            let thermal = sel_registerName(c"thermalState".as_ptr());
            let low_power = sel_registerName(c"isLowPowerModeEnabled".as_ptr());
            let nominal =
                responds(info, supports, thermal).then(|| send_integer(info, thermal) == 0);
            let low = responds(info, supports, low_power).then(|| send_bool(info, low_power));
            (nominal, low)
        }
    }

    fn memory_normal() -> Option<bool> {
        let mut value = 0_i32;
        let mut length = std::mem::size_of_val(&value);
        let status = unsafe {
            libc::sysctlbyname(
                c"kern.memorystatus_vm_pressure_level".as_ptr(),
                &mut value as *mut _ as *mut c_void,
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && length == std::mem::size_of_val(&value)).then_some(value == 1)
    }

    fn free_bytes(path: &Path) -> Option<u64> {
        let path = CString::new(path.as_os_str().as_bytes()).ok()?;
        let mut value = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        if unsafe { libc::statvfs(path.as_ptr(), value.as_mut_ptr()) } != 0 {
            return None;
        }
        let value = unsafe { value.assume_init() };
        Some((value.f_bavail as u64).saturating_mul(value.f_frsize))
    }

    fn competing_process() -> Option<bool> {
        let output = Command::new("/bin/ps")
            .args(["-axo", "pid=,ppid=,comm=,args="])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        competing_process_from_ps(
            std::str::from_utf8(&output.stdout).ok()?,
            std::process::id(),
        )
    }

    pub fn snapshot(root: &Path) -> IdleSnapshot {
        let idle = unsafe { CGEventSourceSecondsSinceLastEventType(1, u32::MAX) };
        let (thermal_nominal, low_power) = process_info();
        IdleSnapshot {
            idle_seconds: (idle.is_finite() && idle >= 0.0).then_some(idle),
            ac_power: power(),
            low_power,
            thermal_nominal,
            memory_normal: memory_normal(),
            free_bytes: free_bytes(root),
            competing_process: competing_process(),
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos::snapshot;

#[cfg(not(target_os = "macos"))]
pub fn snapshot(_root: &Path) -> IdleSnapshot {
    IdleSnapshot::default()
}
