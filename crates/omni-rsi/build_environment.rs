//! Cargo-home discovery and build-script change tracking.
use std::path::PathBuf;

pub fn cargo_home_from_env() -> Option<PathBuf> {
    // Cargo 1.69 and later reject CARGO_HOME in their [env] configuration;
    // the project requires a newer Cargo for its 2024 edition. Unlike HOME,
    // this value cannot be replaced by Cargo when it launches the script.
    // A direct Cargo invocation may leave it unset; that context is unverified.
    std::env::var_os("CARGO_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub fn watch_configuration_environment() {
    for name in ["CARGO_HOME", "HOME", "USERPROFILE", "PWD"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
}
