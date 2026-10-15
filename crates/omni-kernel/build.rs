fn main() {
    // ANE's Swift bridge also needs an rpath in this crate's own test/example
    // executables; dependency link arguments do not propagate to final binaries.
    if std::env::var("CARGO_FEATURE_ANE").is_ok()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
