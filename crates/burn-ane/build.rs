use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=native/Graph.swift");
    println!("cargo:rerun-if-changed=native/Direct.m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-O",
            "-parse-as-library",
            "-emit-library",
            "-static",
            "-module-name",
            "BurnAne",
            "-target",
            "arm64-apple-macos15.0",
            "native/Graph.swift",
            "-o",
        ])
        .arg(out.join("libBurnAne.a"))
        .status()
        .expect("burn-ane requires Xcode Command Line Tools (swiftc)");
    assert!(status.success(), "failed to compile Burn ANE runtime");
    let status = Command::new("xcrun")
        .args([
            "clang",
            "-O2",
            "-fobjc-arc",
            "-fobjc-exceptions",
            "-fobjc-arc-exceptions",
            "-fmodules",
            "-target",
            "arm64-apple-macos15.0",
            "-c",
            "native/Direct.m",
            "-o",
        ])
        .arg(out.join("Direct.o"))
        .status()
        .expect("clang unavailable");
    assert!(status.success(), "failed to compile direct ANE runtime");
    let status = Command::new("xcrun")
        .args(["ar", "rcs"])
        .arg(out.join("libBurnDirectAne.a"))
        .arg(out.join("Direct.o"))
        .status()
        .unwrap();
    assert!(status.success());
    let sdk = Command::new("xcrun")
        .arg("--show-sdk-path")
        .output()
        .unwrap();
    assert!(sdk.status.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!(
        "cargo:rustc-link-search=native={}/usr/lib/swift",
        String::from_utf8(sdk.stdout).unwrap().trim()
    );
    println!("cargo:rustc-link-lib=static=BurnAne");
    println!("cargo:rustc-link-lib=static=BurnDirectAne");
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CryptoKit");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}
