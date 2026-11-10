//! Exercise the complete production build script in a synthetic Cargo graph.
#![cfg(unix)]

use std::{
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Fixture {
    _root: tempfile::TempDir,
    repo: PathBuf,
    package: PathBuf,
    home: PathBuf,
    fake_home: PathBuf,
    cargo_home: PathBuf,
    target: PathBuf,
    replacement: PathBuf,
    cargo: PathBuf,
    rustc: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let artifacts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
        fs::create_dir_all(&artifacts).unwrap();
        let root = tempfile::Builder::new()
            .prefix("backend-context-")
            .tempdir_in(artifacts)
            .unwrap();
        let base = fs::canonicalize(root.path()).unwrap();
        let repo = base.join("repo");
        let package = repo.join("crates/omni-rsi");
        let home = base.join("home");
        let fake_home = base.join("fake-home");
        let cargo_home = home.join(".cargo");
        let target = base.join("target");
        let old = repo.join("vendor/dep");
        let replacement = base.join("replacement");
        for directory in [
            package.join("src"),
            cargo_home.clone(),
            fake_home.clone(),
            old.join("src"),
            replacement.join("src"),
        ] {
            fs::create_dir_all(directory).unwrap();
        }
        let current_home = std::env::var_os("CARGO_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cargo"));
        symlink(current_home.join("registry"), cargo_home.join("registry")).unwrap();
        for (directory, marker) in [(&old, "old_dependency"), (&replacement, "new_dependency")] {
            fs::write(
                directory.join("Cargo.toml"),
                "[package]\nname='probe-dep'\nversion='0.1.0'\nedition='2024'\n",
            )
            .unwrap();
            fs::write(
                directory.join("src/lib.rs"),
                format!("pub fn marker() -> &'static str {{ {marker:?} }}\n"),
            )
            .unwrap();
        }
        fs::write(
            package.join("Cargo.toml"),
            "[package]\nname='backend-context-probe'\nversion='0.1.0'\nedition='2024'\n\
             [workspace]\n[features]\ncpu=[]\nmetal=[]\n\
             [dependencies]\nprobe-dep='=0.1.0'\n\
             [build-dependencies]\nsha2='0.10'\nlibc='0.2'\n\
             [patch.crates-io]\nprobe-dep={path=\"../../vendor/dep\"}\n",
        )
        .unwrap();
        fs::write(
            package.join("src/main.rs"),
            "fn main() { println!(\"identity={} dependency={}\", env!(\"OMNI_RSI_BACKEND_CPU\"), probe_dep::marker()); }\n",
        )
        .unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR"));
        for name in [
            "build.rs",
            "build_backend.rs",
            "build_environment.rs",
            "build_invocation.rs",
        ] {
            fs::copy(source.join(name), package.join(name)).unwrap();
        }
        // The graph checker needs all anchor files even when these synthetic
        // contents deliberately make the graph unknown.
        for name in [
            "Cargo.toml",
            "Cargo.lock",
            "crates/lfm3-omni/Cargo.toml",
            "crates/lfm3-omni/Cargo.lock",
            "vendor/burn/Cargo.toml",
            "vendor/cubecl-common/Cargo.toml",
            "crates/burn-ane/Cargo.toml",
        ] {
            let path = repo.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"synthetic graph anchor").unwrap();
        }
        let sysroot = Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .unwrap();
        assert!(sysroot.status.success());
        let sysroot = String::from_utf8(sysroot.stdout).unwrap();
        let cargo = PathBuf::from(sysroot.trim()).join("bin/cargo");
        let rustc = PathBuf::from(sysroot.trim()).join("bin/rustc");
        assert!(
            cargo.is_file() && rustc.is_file(),
            "direct toolchain is unavailable"
        );
        let fixture = Self {
            _root: root,
            repo,
            package,
            home,
            fake_home,
            cargo_home,
            target,
            replacement,
            cargo,
            rustc,
        };
        fixture.write_config(false);
        let lock = fixture.cargo(false, &["generate-lockfile", "--offline"]);
        assert!(
            lock.status.success(),
            "{}",
            String::from_utf8_lossy(&lock.stderr)
        );
        fixture
    }

    fn write_config(&self, replace: bool) {
        let paths = if replace {
            format!("paths = [{:?}]\n", self.replacement.to_str().unwrap())
        } else {
            String::new()
        };
        fs::write(
            self.cargo_home.join("config.toml"),
            format!(
                "{paths}[env]\nHOME = {{ value = {:?}, force = true }}\n",
                self.fake_home.to_str().unwrap()
            ),
        )
        .unwrap();
    }

    fn cargo(&self, explicit_home: bool, args: &[&str]) -> Output {
        let mut command = Command::new(&self.cargo);
        command
            .current_dir(&self.repo)
            .args(args)
            .arg("--manifest-path")
            .arg(self.package.join("Cargo.toml"))
            .env("HOME", &self.home)
            .env("RUSTC", &self.rustc)
            .env("CARGO_TARGET_DIR", &self.target)
            .env("CARGO_BUILD_JOBS", "1")
            .env("CARGO_INCREMENTAL", "0");
        if explicit_home {
            command.env("CARGO_HOME", &self.cargo_home);
        } else {
            command.env_remove("CARGO_HOME");
        }
        command.output().unwrap()
    }

    fn run(&self, explicit_home: bool) -> Output {
        self.cargo(
            explicit_home,
            &[
                "run",
                "--offline",
                "--locked",
                "--quiet",
                "--features",
                "cpu",
            ],
        )
    }

    fn run_empty_home(&self) -> Output {
        Command::new(&self.cargo)
            .current_dir(&self.repo)
            .args([
                "run",
                "--offline",
                "--locked",
                "--quiet",
                "--features",
                "cpu",
                "--manifest-path",
            ])
            .arg(self.package.join("Cargo.toml"))
            .env("HOME", &self.home)
            .env("CARGO_HOME", "")
            .env("RUSTC", &self.rustc)
            .env("CARGO_TARGET_DIR", &self.target)
            .env("CARGO_BUILD_JOBS", "1")
            .env("CARGO_INCREMENTAL", "0")
            .output()
            .unwrap()
    }

    fn rustup_run(&self, proxy: &Path, rustup_home: &Path) -> Output {
        Command::new(proxy)
            .current_dir(&self.repo)
            .args([
                "run",
                "--offline",
                "--locked",
                "--quiet",
                "--features",
                "cpu",
                "--manifest-path",
            ])
            .arg(self.package.join("Cargo.toml"))
            .env("HOME", &self.home)
            .env_remove("CARGO_HOME")
            .env("RUSTUP_HOME", rustup_home)
            .env("RUSTC", &self.rustc)
            .env("CARGO_TARGET_DIR", &self.target)
            .env("CARGO_BUILD_JOBS", "1")
            .env("CARGO_INCREMENTAL", "0")
            .output()
            .unwrap()
    }

    fn build_script(&self) -> PathBuf {
        let paths: Vec<_> = fs::read_dir(self.target.join("debug/build"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("backend-context-probe-")
            })
            .map(|entry| entry.path().join("build-script-build"))
            .filter(|path| path.is_file())
            .collect();
        assert_eq!(
            paths.len(),
            1,
            "expected one synthetic build-script executable"
        );
        paths[0].clone()
    }
}

#[test]
fn cargo_rejects_configuration_that_sets_cargo_home() {
    let fixture = Fixture::new();
    fs::write(
        fixture.cargo_home.join("config.toml"),
        format!(
            "[env]\nCARGO_HOME = {{ value = {:?}, force = true }}\n",
            fixture.fake_home.to_str().unwrap()
        ),
    )
    .unwrap();
    let result = fixture.run(false);
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("setting the `CARGO_HOME` environment variable is not supported"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn rustup_injected_cargo_home_keeps_valid_identity_and_detects_override() {
    let original_home = PathBuf::from(std::env::var_os("HOME").unwrap());
    let original_cargo_home = std::env::var_os("CARGO_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| original_home.join(".cargo"));
    let proxy = original_cargo_home.join("bin/cargo");
    let rustup_home = std::env::var_os("RUSTUP_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| original_home.join(".rustup"));
    if !proxy.is_file() || !rustup_home.is_dir() {
        eprintln!("rustup proxy unavailable; no rustup claim for this environment");
        return;
    }
    let fixture = Fixture::new();
    let before = fixture.rustup_run(&proxy, &rustup_home);
    assert!(
        before.status.success(),
        "{}",
        String::from_utf8_lossy(&before.stderr)
    );
    let before = String::from_utf8_lossy(&before.stdout);
    eprintln!("rustup-before={before}");
    assert!(before.contains("dependency=old_dependency"));
    let verified = !before.contains("identity=backend-unverified-context");
    if verified {
        let identity = before
            .split_whitespace()
            .find_map(|part| part.strip_prefix("identity="))
            .unwrap();
        assert_eq!(identity.len(), 64);
        assert!(identity.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
    fixture.write_config(true);
    let after = fixture.rustup_run(&proxy, &rustup_home);
    let stdout = String::from_utf8_lossy(&after.stdout);
    let stderr = String::from_utf8_lossy(&after.stderr);
    eprintln!(
        "rustup-after status={} stdout={stdout} stderr={stderr}",
        after.status
    );
    if verified {
        assert!(!after.status.success());
        assert!(stderr.contains("path overrides are unsupported"));
    } else {
        assert!(
            (!after.status.success() && stderr.contains("path overrides are unsupported"))
                || (after.status.success()
                    && stdout.contains("identity=backend-unverified-context"))
        );
    }
}

#[test]
fn direct_cargo_without_trusted_home_fails_closed_despite_dependency_override() {
    let fixture = Fixture::new();
    let before = fixture.run(false);
    fixture.write_config(true);
    let after = fixture.run(false);
    assert!(
        before.status.success(),
        "{}",
        String::from_utf8_lossy(&before.stderr)
    );
    assert!(
        after.status.success(),
        "{}",
        String::from_utf8_lossy(&after.stderr)
    );
    let before = String::from_utf8_lossy(&before.stdout);
    let after = String::from_utf8_lossy(&after.stdout);
    eprintln!("direct-before={before}direct-after={after}");
    assert!(before.contains("dependency=old_dependency"));
    assert!(after.contains("dependency=new_dependency"));
    assert!(before.contains("identity=backend-unverified-context"));
    assert!(after.contains("identity=backend-unverified-context"));
    let empty = fixture.run_empty_home();
    assert!(
        empty.status.success(),
        "{}",
        String::from_utf8_lossy(&empty.stderr)
    );
    assert!(String::from_utf8_lossy(&empty.stdout).contains("identity=backend-unverified-context"));

    // Reusing the same target with an explicitly supplied protected home must
    // recheck configuration rather than retaining the unverified result.
    fixture.write_config(false);
    let explicit = fixture.run(true);
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    let stdout = String::from_utf8_lossy(&explicit.stdout);
    eprintln!("explicit-home={stdout}");
    assert!(stdout.contains("dependency=old_dependency"));
    let verified = !stdout.contains("identity=backend-unverified-context");
    fixture.write_config(true);
    let guarded = fixture.run(true);
    let stdout = String::from_utf8_lossy(&guarded.stdout);
    let stderr = String::from_utf8_lossy(&guarded.stderr);
    if verified {
        assert!(!guarded.status.success());
        assert!(stderr.contains("path overrides are unsupported"));
    } else {
        assert!(
            (!guarded.status.success() && stderr.contains("path overrides are unsupported"))
                || (guarded.status.success()
                    && stdout.contains("identity=backend-unverified-context"))
        );
    }
}

#[test]
fn production_build_script_marks_an_unverified_parent_as_unverified() {
    let fixture = Fixture::new();
    let build = fixture.run(true);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    // Executing the actual build-script binary from this test process makes
    // the direct parent verifiably different from Cargo.
    let out = fixture.repo.join("synthetic-build-out");
    fs::create_dir(&out).unwrap();
    let result = Command::new(fixture.build_script())
        .env("CARGO", &fixture.cargo)
        .env("CARGO_HOME", &fixture.cargo_home)
        .env("CARGO_MANIFEST_DIR", &fixture.package)
        .env("OUT_DIR", &out)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let directives = String::from_utf8_lossy(&result.stdout);
    assert!(directives.contains("cargo:rustc-env=OMNI_RSI_BACKEND_CPU=backend-unverified-context"));
    assert!(directives.contains("cargo:rustc-env=OMNI_RSI_BACKEND_GPU=backend-unverified-context"));
}
