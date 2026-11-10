//! Cargo-home discovery and build-script change tracking.
use std::{ffi::OsString, path::PathBuf};

pub fn cargo_home(
    cargo: Option<OsString>,
    home: Option<OsString>,
    profile: Option<OsString>,
    windows: bool,
) -> Option<PathBuf> {
    cargo
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let base = if windows { profile } else { home };
            base.filter(|value| !value.is_empty())
                .map(|value| PathBuf::from(value).join(".cargo"))
        })
}

pub fn cargo_home_from_env() -> Option<PathBuf> {
    cargo_home(
        std::env::var_os("CARGO_HOME"),
        std::env::var_os("HOME"),
        std::env::var_os("USERPROFILE"),
        cfg!(windows),
    )
    .or_else(|| {
        // Match Cargo's OS home-directory fallback when the usual environment
        // variable is unset. On Windows HOME does not override USERPROFILE.
        #[allow(deprecated)]
        std::env::home_dir().map(|home| home.join(".cargo"))
    })
}

pub fn watch_configuration_environment() {
    for name in ["CARGO_HOME", "HOME", "USERPROFILE", "PWD"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
}
