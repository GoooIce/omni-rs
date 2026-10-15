fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // rustc-link-arg from a dependency does not propagate to this executable.
    // The statically linked Core ML bridge uses Apple's system Swift runtime.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
