//! Cargo integration coverage for configuration changes under one target.
#![cfg(all(unix, any(target_os = "linux", target_os = "macos")))]

use std::{
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Fixture {
    _root: tempfile::TempDir,
    repo: PathBuf,
    manifest: PathBuf,
    home: PathBuf,
    target: PathBuf,
    replacement: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let artifacts =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/review-ee83042");
        fs::create_dir_all(&artifacts).unwrap();
        let root = tempfile::Builder::new()
            .prefix("config-fixture-")
            .tempdir_in(artifacts)
            .unwrap();
        let base = fs::canonicalize(root.path()).unwrap();
        let repo = base.join("repo");
        let package = repo.join("crates/omni-rsi");
        let home = base.join("cargo-home");
        let target = base.join("shared-target");
        let old = base.join("old-dep");
        let replacement = base.join("new-dep");
        for path in [
            &package.join("src"),
            &home,
            &old.join("src"),
            &replacement.join("src"),
        ] {
            fs::create_dir_all(path).unwrap();
        }
        let current_home = std::env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cargo"));
        symlink(current_home.join("registry"), home.join("registry")).unwrap();
        for (dir, marker) in [(&old, "old_dependency"), (&replacement, "new_dependency")] {
            fs::write(
                dir.join("Cargo.toml"),
                "[package]\nname='probe-dep'\nversion='0.1.0'\nedition='2024'\n",
            )
            .unwrap();
            fs::write(
                dir.join("src/lib.rs"),
                format!("pub fn marker() -> &'static str {{ {marker:?} }}\n"),
            )
            .unwrap();
        }
        let manifest = package.join("Cargo.toml");
        fs::write(&manifest, format!("[package]\nname='config-freshness-probe'\nversion='0.1.0'\nedition='2024'\n[workspace]\n[dependencies]\nprobe-dep='=0.1.0'\n[build-dependencies]\nsha2='0.10'\nlibc='0.2'\n[patch.crates-io]\nprobe-dep={{path={:?}}}\n", old.to_str().unwrap())).unwrap();
        let production = Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs");
        fs::write(
            package.join("build.rs"),
            format!(
                r#"
#[allow(dead_code)]
#[path = {:?}] mod actual;
use sha2::{{Digest, Sha256}};
fn main() {{
    let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    if let Some(cwd) = actual::build_invocation::cargo_invocation_cwd() {{
        let mut hash = Sha256::new();
        actual::config_identity(&mut hash, &cwd, &manifest).unwrap();
        hash.update(std::fs::read({:?}).unwrap());
        println!("cargo:rustc-env=PROBE_ID={{:x}}", hash.finalize());
    }} else {{
        println!("cargo:rustc-env=PROBE_ID=unverified");
    }}
}}
"#,
                production.to_str().unwrap(),
                old.join("src/lib.rs").to_str().unwrap()
            ),
        )
        .unwrap();
        fs::write(
            package.join("src/main.rs"),
            "fn main() { println!(\"{} {}\", env!(\"PROBE_ID\"), probe_dep::marker()); }\n",
        )
        .unwrap();
        let fixture = Self {
            _root: root,
            repo,
            manifest,
            home,
            target,
            replacement,
        };
        let lock = fixture.cargo(&["generate-lockfile", "--offline"]);
        assert!(
            lock.status.success(),
            "{}",
            String::from_utf8_lossy(&lock.stderr)
        );
        fixture
    }

    fn cargo(&self, args: &[&str]) -> Output {
        self.cargo_at(args, &self.repo, Some(&self.repo))
    }

    fn cargo_at(&self, args: &[&str], cwd: &Path, pwd: Option<&Path>) -> Output {
        let mut command = Command::new(env!("CARGO"));
        command
            .current_dir(cwd)
            .args(args)
            .arg("--manifest-path")
            .arg(&self.manifest)
            .env("CARGO_HOME", &self.home)
            .env("CARGO_TARGET_DIR", &self.target)
            .env("CARGO_BUILD_JOBS", "1")
            .env("CARGO_INCREMENTAL", "0");
        match pwd {
            Some(pwd) => {
                command.env("PWD", pwd);
            }
            None => {
                command.env_remove("PWD");
            }
        }
        command.output().unwrap()
    }

    fn run(&self) -> Output {
        self.cargo(&["run", "--offline", "--locked", "--quiet"])
    }

    fn replacement_config(&self) -> String {
        format!("paths = [{:?}]\n", self.replacement.to_str().unwrap())
    }

    fn assert_cargo_selected_replacement(&self) {
        let tree = self.cargo(&["tree", "--offline", "--locked"]);
        let stdout = String::from_utf8_lossy(&tree.stdout);
        assert!(
            tree.status.success(),
            "{}",
            String::from_utf8_lossy(&tree.stderr)
        );
        assert!(
            stdout.contains(self.replacement.to_str().unwrap()),
            "Cargo did not select replacement: {stdout}"
        );
        eprintln!("Cargo selected replacement dependency: {stdout}");
    }

    fn assert_override_rejected_or_unverified(&self, result: &Output) {
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            (!result.status.success() && stderr.contains("path overrides are unsupported"))
                || (result.status.success() && stdout.trim() == "unverified new_dependency"),
            "replacement must be rejected or carry no trusted identity: {stdout}\n{stderr}"
        );
    }
}

#[test]
fn newly_created_cargo_home_config_cannot_reuse_the_old_identity() {
    let fixture = Fixture::new();
    let first = fixture.run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("old_dependency"));
    fs::write(
        fixture.home.join("config.toml"),
        fixture.replacement_config(),
    )
    .unwrap();
    fixture.assert_cargo_selected_replacement();
    let second = fixture.run();
    fixture.assert_override_rejected_or_unverified(&second);
}

#[test]
fn newly_created_ancestor_config_cannot_reuse_the_old_identity() {
    let fixture = Fixture::new();
    let first = fixture.run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("old_dependency"));
    let directory = fixture.repo.join(".cargo");
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("config.toml"), fixture.replacement_config()).unwrap();
    fixture.assert_cargo_selected_replacement();
    let second = fixture.run();
    fixture.assert_override_rejected_or_unverified(&second);
}

#[test]
fn retargeting_config_symlink_to_a_path_override_is_rejected() {
    let fixture = Fixture::new();
    let configs = fixture.repo.join("configs");
    fs::create_dir(&configs).unwrap();
    let safe = configs.join("safe.toml");
    let override_file = configs.join("override.toml");
    fs::write(&safe, "[alias]\nmarker = 'check'\n").unwrap();
    fs::write(&override_file, fixture.replacement_config()).unwrap();
    let link = fixture.home.join("config.toml");
    symlink(&safe, &link).unwrap();
    let first = fixture.run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("old_dependency"));
    fs::remove_file(&link).unwrap();
    symlink(&override_file, &link).unwrap();
    fixture.assert_cargo_selected_replacement();
    let second = fixture.run();
    fixture.assert_override_rejected_or_unverified(&second);
}

#[test]
fn retargeting_safe_config_symlink_changes_the_identity() {
    let fixture = Fixture::new();
    let configs = fixture.repo.join("configs");
    fs::create_dir(&configs).unwrap();
    let first_config = configs.join("first.toml");
    let second_config = configs.join("second.toml");
    fs::write(&first_config, "[alias]\nmarker = 'check'\n").unwrap();
    fs::write(&second_config, "[alias]\nmarker = 'fmt'\n").unwrap();
    let link = fixture.home.join("config.toml");
    symlink(&first_config, &link).unwrap();
    let first = fixture.run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    fs::remove_file(&link).unwrap();
    symlink(&second_config, &link).unwrap();
    let second = fixture.run();
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("old_dependency"));
    assert!(String::from_utf8_lossy(&second.stdout).contains("old_dependency"));
    if String::from_utf8_lossy(&first.stdout).starts_with("unverified ") {
        assert_eq!(first.stdout, second.stdout);
    } else {
        assert_ne!(
            first.stdout, second.stdout,
            "the selected config changed but the identity did not"
        );
    }
}

#[test]
fn deleting_legacy_config_rechecks_the_newly_effective_toml_file() {
    let fixture = Fixture::new();
    fs::write(fixture.home.join("config"), "[alias]\nmarker = 'check'\n").unwrap();
    fs::write(
        fixture.home.join("config.toml"),
        fixture.replacement_config(),
    )
    .unwrap();
    let first = fixture.run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("old_dependency"));
    fs::remove_file(fixture.home.join("config")).unwrap();
    fixture.assert_cargo_selected_replacement();
    let second = fixture.run();
    fixture.assert_override_rejected_or_unverified(&second);
}

#[test]
fn newly_created_legacy_config_takes_precedence_and_is_rechecked() {
    let fixture = Fixture::new();
    fs::write(
        fixture.home.join("config.toml"),
        "[alias]\nmarker = 'check'\n",
    )
    .unwrap();
    let first = fixture.run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("old_dependency"));
    fs::write(fixture.home.join("config"), fixture.replacement_config()).unwrap();
    fixture.assert_cargo_selected_replacement();
    let second = fixture.run();
    fixture.assert_override_rejected_or_unverified(&second);
}

#[test]
fn missing_or_stale_pwd_cannot_hide_a_cargo_path_override() {
    for (other_cwd, stale) in [(false, false), (false, true), (true, false), (true, true)] {
        let fixture = Fixture::new();
        let first = fixture.run();
        assert!(
            first.status.success(),
            "{}",
            String::from_utf8_lossy(&first.stderr)
        );
        let cwd = if other_cwd {
            let path = fixture.repo.join("other-invocation");
            fs::create_dir(&path).unwrap();
            path
        } else {
            fixture.manifest.parent().unwrap().to_path_buf()
        };
        fs::create_dir(cwd.join(".cargo")).unwrap();
        fs::write(cwd.join(".cargo/config.toml"), fixture.replacement_config()).unwrap();
        let pwd = stale.then_some(fixture.repo.as_path());
        let tree = fixture.cargo_at(&["tree", "--offline", "--locked"], &cwd, pwd);
        assert!(
            tree.status.success(),
            "{}",
            String::from_utf8_lossy(&tree.stderr)
        );
        assert!(
            String::from_utf8_lossy(&tree.stdout).contains(fixture.replacement.to_str().unwrap())
        );
        let result = fixture.cargo_at(&["run", "--offline", "--locked", "--quiet"], &cwd, pwd);
        let output = String::from_utf8_lossy(&result.stdout);
        let errors = String::from_utf8_lossy(&result.stderr);
        eprintln!(
            "other_cwd={other_cwd} stale_pwd={stale} status={} output={output} error={errors}",
            result.status
        );
        assert!(
            (!result.status.success() && errors.contains("path overrides are unsupported"))
                || (result.status.success() && output.contains("unverified new_dependency")),
            "other_cwd={other_cwd} stale={stale}: Cargo selected a replacement with a supposedly trusted identity: {output}\n{errors}"
        );
    }
}

#[test]
fn valid_comments_and_escaped_alias_keys_do_not_block_cargo() {
    let fixture = Fixture::new();
    fs::write(
        fixture.home.join("config.toml"),
        "# legal comment containing \"\"\"\n[alias]\n\"a\\u002db\" = 'check'\n",
    )
    .unwrap();
    let metadata = fixture.cargo(&[
        "metadata",
        "--offline",
        "--locked",
        "--format-version",
        "1",
        "--no-deps",
    ]);
    assert!(
        metadata.status.success(),
        "Cargo rejected fixture: {}",
        String::from_utf8_lossy(&metadata.stderr)
    );
    let run = fixture.run();
    assert!(
        run.status.success(),
        "valid Cargo configuration was rejected: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("old_dependency"));
}

#[test]
fn escaped_root_path_override_cannot_bypass_the_guard() {
    let fixture = Fixture::new();
    fs::write(
        fixture.home.join("config.toml"),
        format!(
            "\"\\u0070aths\" = [{:?}]\n",
            fixture.replacement.to_str().unwrap()
        ),
    )
    .unwrap();
    fixture.assert_cargo_selected_replacement();
    let run = fixture.run();
    fixture.assert_override_rejected_or_unverified(&run);
}
