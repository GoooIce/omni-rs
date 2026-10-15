use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_ANE").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
    println!("cargo:rerun-if-changed=native/CoreMLEncoder.swift");
    if env::var_os("CARGO_FEATURE_COREML").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
    {
        return;
    }
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        other => panic!("unsupported Core ML target: {other:?}"),
    };
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-O",
            "-parse-as-library",
            "-emit-library",
            "-static",
            "-module-name",
            "R2T2CoreML",
            "-target",
            &format!("{arch}-apple-macos15.0"),
            "native/CoreMLEncoder.swift",
            "-o",
        ])
        .arg(out.join("libR2T2CoreML.a"))
        .status()
        .expect("Core ML requires Xcode Command Line Tools (swiftc)");
    assert!(status.success(), "failed to compile Core ML bridge");
    let sdk = Command::new("xcrun")
        .arg("--show-sdk-path")
        .output()
        .unwrap();
    assert!(sdk.status.success());
    let sdk = String::from_utf8(sdk.stdout).unwrap();
    println!("cargo:rustc-link-search=native={}", out.display());
    println!(
        "cargo:rustc-link-search=native={}/usr/lib/swift",
        sdk.trim()
    );
    println!("cargo:rustc-link-lib=static=R2T2CoreML");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}
