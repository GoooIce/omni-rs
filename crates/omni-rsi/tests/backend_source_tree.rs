#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;
use build_script::build_backend;

use std::{fs, path::Path};

fn fixture(root: &Path) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='fixture'\nversion='1.0.0'\n",
    )
    .unwrap();
    fs::write(root.join("src/kernel.rs"), "pub fn kernel() -> u32 { 1 }").unwrap();
}

#[test]
fn source_identity_binds_changes_additions_deletions_and_native_build_files() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    fixture(first.path());
    fixture(second.path());
    let digest = |root: &Path| build_backend::source_tree_digest(root).unwrap().0;
    let original = digest(first.path());
    assert_eq!(
        original,
        digest(second.path()),
        "checkout location must not affect identity"
    );
    fs::write(
        first.path().join("src/kernel.rs"),
        "pub fn kernel() -> u32 { 2 }",
    )
    .unwrap();
    assert_ne!(original, digest(first.path()));
    fixture(first.path());
    fs::write(
        first.path().join("src/untracked.metal"),
        "kernel void added() {}",
    )
    .unwrap();
    let added = digest(first.path());
    assert_ne!(original, added);
    fs::remove_file(first.path().join("src/untracked.metal")).unwrap();
    assert_eq!(original, digest(first.path()));
    fs::write(first.path().join("build.rs"), "fn main() {}").unwrap();
    assert_ne!(original, digest(first.path()));
}

#[test]
fn independent_self_reading_implementation_identity_is_unchanged() {
    // Pinned after the idle-runner changes, local numerical patch routing and
    // aligned dependency locks (ADR-0005). Independently audited input changes
    // are limited to the three included lockfiles; the identity algorithm and
    // checkpoint rejection guards are unchanged.
    // Repinned on 2026-11-24 after the native-only manifests: since 1373a67 the
    // hashed inputs changed only in those lockfiles (wasm-only dependencies
    // left, one crate moved from 1.0.151 to 1.0.154). The earlier pin matched
    // neither that revision nor its own.
    // Repinned on 2026-11-26 after ADR-0006 unified the 16 detached manifests
    // into one root Cargo workspace (`Cargo.lock`), removing the redundant
    // per-crate `crates/omni-rsi/Cargo.lock` and `crates/lfm3-omni/Cargo.lock`
    // inputs without changing any training/loss/checkpoint implementation source.
    // Repinned on 2026-11-26 after deepening idle supervision into
    // `idle_supervisor.rs` and unifying cooperative file leases across
    // `omni-lfm3::lease` and `omni-rsi::lease` (`O_NONBLOCK` FIFO guard in
    // `ForegroundLease`); no training, loss, or checkpoint algorithm changed.
    assert_eq!(
        omni_rsi::speech_checkpoint::implementation_hash(),
        "82beea6d3df1e990b35f576a0498e99ec19d169d30f1c2c86970da24fb9d345a"
    );
}

#[test]
fn build_outputs_are_ignored_but_symlinked_sources_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    fixture(root.path());
    let before = build_backend::source_tree_digest(root.path()).unwrap().0;
    for name in ["target/debug/object", ".git/index", "artifacts/output"] {
        let path = root.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "noise").unwrap();
    }
    assert_eq!(
        before,
        build_backend::source_tree_digest(root.path()).unwrap().0
    );
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("kernel.rs", root.path().join("src/linked.rs")).unwrap();
        assert!(build_backend::source_tree_digest(root.path()).is_err());
    }
}

#[cfg(unix)]
#[test]
fn backend_graph_rejects_symlinked_dependency_roots() {
    let temporary = tempfile::tempdir().unwrap();
    let repo = fs::canonicalize(temporary.path()).unwrap();
    let source = repo.join("source");
    fs::create_dir(&source).unwrap();
    fixture(&source);
    let link = repo.join("link");
    std::os::unix::fs::symlink(&source, &link).unwrap();
    assert!(build_script::graph_identity(&repo, vec![link.clone()], b"fixture", "cpu").is_err());
    assert!(
        build_script::graph_identity(&repo, vec![link.join("src")], b"fixture", "cpu").is_err()
    );
}

#[test]
fn patch_manifest_discovers_new_local_crates_and_rejects_unsupported_forms() {
    let manifest = "[package]\nname='fixture'\n[patch.crates-io]\nburn = { path = \"../../burn\" }\nnew-kernel = { path = \"../../kernel\" }\n";
    assert_eq!(
        build_backend::patch_paths(manifest).unwrap(),
        vec![
            ("burn".into(), "../../burn".into()),
            ("new-kernel".into(), "../../kernel".into())
        ]
    );
    for invalid in [
        "[patch.crates-io]\nburn = { git = \"https://example.invalid/burn\" }\n",
        "[patch.crates-io.burn]\npath = \"../../burn\"\n",
        "[patch.crates-io]\nburn = { path = \"../../burn\", version = \"1\" }\n",
    ] {
        assert!(build_backend::patch_paths(invalid).is_err());
    }
}

#[test]
fn cargo_dependency_paths_cannot_be_silently_missed_in_valid_toml() {
    for manifest in [
        "[dependencies]\nfoo.path = \"../foo\"\n",
        "[dependencies]\nfoo = { \"path\" = \"../foo\" }\n",
        "[dependencies]\nfoo.'path' = '../foo'\n",
        "[build-dependencies.foo]\n\"path\" = \"../foo\"\n",
    ] {
        if let Ok(paths) = build_script::local_dependencies(manifest) {
            assert_eq!(paths, vec!["../foo"], "silently omitted {manifest}");
        }
    }
}

#[test]
fn visible_cargo_path_overrides_are_rejected() {
    for config in [
        "paths = [\"../replacement\"]\n",
        "\"paths\" = [\"../replacement\"]\n",
        "[patch.crates-io]\nburn = { path = \"../replacement\" }\n",
        "[\"patch\".crates-io]\nburn.path = \"../replacement\"\n",
        "patch.crates-io.burn.path = \"../replacement\"\n",
        "\"\\u0070aths\" = [\"../replacement\"]\n",
        "[\"\\u0070atch\".crates-io]\nburn.path = \"../replacement\"\n",
    ] {
        assert!(
            build_backend::validate_config(config).is_err(),
            "accepted {config}"
        );
    }
    assert!(build_backend::validate_config("[build]\njobs = 2\n").is_ok());
    assert!(
        build_backend::validate_config("[alias]\npatch = \"fmt --check\"\npaths = \"metadata\"\n")
            .is_ok()
    );
    assert!(
        build_backend::validate_config(
            "# comment \"\"\"\n[\"ali\\u0061s\"]\n\"a\\u002db\" = 'check'\n"
        )
        .is_ok()
    );
}

#[test]
fn cargo_parent_requires_the_expected_executable_and_an_existing_directory() {
    let root = tempfile::tempdir().unwrap();
    let expected = root.path().join("cargo");
    let other = root.path().join("other");
    fs::write(&expected, b"cargo fixture").unwrap();
    fs::write(&other, b"other fixture").unwrap();
    assert_eq!(
        build_script::build_invocation::verified_parent_cwd(&expected, &expected, root.path()),
        Some(fs::canonicalize(root.path()).unwrap())
    );
    assert!(
        build_script::build_invocation::verified_parent_cwd(&expected, &other, root.path())
            .is_none()
    );
    assert!(
        build_script::build_invocation::verified_parent_cwd(
            &expected,
            &expected,
            &root.path().join("missing")
        )
        .is_none()
    );
    assert!(
        build_script::build_invocation::verified_parent_cwd(
            Path::new("cargo"),
            &expected,
            root.path()
        )
        .is_none()
    );
    assert!(
        build_script::build_invocation::verified_parent_cwd(root.path(), root.path(), root.path())
            .is_none()
    );
}

#[test]
fn foreign_proc_namespace_is_not_accepted_as_the_current_process() {
    use build_script::build_invocation::proc_self_matches_pid;
    assert!(proc_self_matches_pid(Path::new("731"), 731));
    assert!(!proc_self_matches_pid(Path::new("732"), 731));
    assert!(!proc_self_matches_pid(Path::new("/proc/731"), 731));
    assert!(!proc_self_matches_pid(Path::new("self"), 731));
}

#[test]
#[cfg(unix)]
fn scanner_manifest_keeps_the_real_cpu_roots_without_scanning_gpu_only_paths() {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::symlink;

    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let repo = fs::canonicalize(temp.path()).unwrap();
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "crates/lfm3-omni/Cargo.toml",
        "crates/omni-rsi/Cargo.toml",
        "vendor/burn/Cargo.toml",
        "vendor/cubecl-common/Cargo.toml",
        "crates/burn-ane/Cargo.toml",
    ] {
        let destination = repo.join(name);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(source.join(name), destination).unwrap();
    }
    for name in [
        "burn-cubecl",
        "burn-fusion",
        "burn-router",
        "cubecl-cpp",
        "cubecl-runtime",
        "cubecl-wgpu",
    ] {
        symlink(
            "inaccessible-synthetic-gpu-only",
            repo.join("vendor").join(name),
        )
        .unwrap();
    }
    let manifest = repo.join("crates/omni-rsi");
    let current = fs::read_to_string(manifest.join("Cargo.toml")).unwrap();
    let original_hash = "1f19d2ed09a4704028cada1ee7ae3a510ac728a5f43ed67d74b5a5864cd864fc";
    let scanner_hash = "f1502a1d91d591305f1675d2f382b902aa80e6bc05e07031716129205d1c22e2";
    let original = include_str!("fixtures/backend-cpu-original.toml");
    let scanner = include_str!("fixtures/backend-cpu-scanner.toml");
    assert_eq!(
        format!("{:x}", Sha256::digest(original.as_bytes())),
        original_hash
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(scanner.as_bytes())),
        scanner_hash
    );
    let other_anchors = [
        (
            "Cargo.toml",
            "abae3e721ce3b59e511dbb30e3e7e491646193c50dd64cf7bd9b878085a01418",
        ),
        (
            "Cargo.lock",
            "88851b9ed39e38c47483ee8c0bb1f9d51ca65beaa44e73fe8bf25334f958bc3e",
        ),
        (
            "crates/lfm3-omni/Cargo.toml",
            "cbb8099b360eff538b8d76585c1cd7691e0d47d6a85b7ca2dcc016b90c1d8de0",
        ),
        (
            "vendor/burn/Cargo.toml",
            "ae43f28673449477f9e34cde9e9f3289ce55b4a352ea059aa357649d9b872701",
        ),
        (
            "vendor/cubecl-common/Cargo.toml",
            "68c03a500f4eafe809cf3b1beb5a6d348d510bb082bc8ab1c7c6fc316b688150",
        ),
        (
            "crates/burn-ane/Cargo.toml",
            "0ff1aaa0f6ac6257257fd8af012675042938f348da068ebc24e36f3cde301dc5",
        ),
    ];
    let other_anchors_match = other_anchors.iter().all(|(path, expected)| {
        format!("{:x}", Sha256::digest(fs::read(repo.join(path)).unwrap())) == *expected
    });
    let mut known_cpu_hash = None;
    for frozen in [original, scanner] {
        fs::write(manifest.join("Cargo.toml"), frozen).unwrap();
        assert_eq!(
            build_script::known_cpu_graph(&repo).unwrap(),
            other_anchors_match
        );
        let (cpu, gpu) = build_script::backend_roots(&repo, &manifest).unwrap();
        if other_anchors_match {
            let identity = build_script::graph_identity(&repo, cpu, b"fixture", "cpu").unwrap();
            if let Some(previous) = &known_cpu_hash {
                assert_eq!(&identity, previous);
            } else {
                known_cpu_hash = Some(identity);
            }
        } else {
            assert!(build_script::graph_identity(&repo, cpu, b"fixture", "cpu").is_err());
        }
        assert!(build_script::graph_identity(&repo, gpu, b"fixture", "gpu").is_err());
    }

    let third = format!("{original}\n# another legal manifest\n");
    for candidate in [&current, &third] {
        fs::write(manifest.join("Cargo.toml"), candidate).unwrap();
        let hash = format!("{:x}", Sha256::digest(candidate.as_bytes()));
        let known = other_anchors_match && (hash == original_hash || hash == scanner_hash);
        assert_eq!(build_script::known_cpu_graph(&repo).unwrap(), known);
        let (cpu, _) = build_script::backend_roots(&repo, &manifest).unwrap();
        if known {
            assert_eq!(
                build_script::graph_identity(&repo, cpu, b"fixture", "cpu").unwrap(),
                *known_cpu_hash.as_ref().unwrap()
            );
        } else {
            let root_patches = fs::read_to_string(repo.join("Cargo.toml")).unwrap();
            let (patch_base, parsed) = match build_backend::patch_paths(candidate) {
                Ok(patches) => (manifest.as_path(), patches),
                Err(_) => (
                    repo.as_path(),
                    build_backend::patch_paths(&root_patches).unwrap(),
                ),
            };
            let expected: std::collections::BTreeSet<_> = parsed
                .into_iter()
                .map(|(_, path)| patch_base.join(path))
                .collect();
            assert_eq!(
                cpu.iter()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>(),
                expected
            );
            if candidate == &third {
                assert!(build_script::graph_identity(&repo, cpu, b"fixture", "cpu").is_err());
            }
        }
    }
}

#[test]
fn cpu_selection_does_not_scan_the_disabled_gpu_graph() {
    let mut calls = Vec::new();
    let result = build_script::backend_digests(true, false, |backend| {
        calls.push(backend.to_owned());
        if backend == "gpu" {
            return Err("GPU tree is deliberately inaccessible".into());
        }
        Ok("cpu-source-identity".into())
    });
    let (cpu, _) = result.unwrap();
    assert_eq!(cpu, "cpu-source-identity");
    assert_eq!(calls, ["cpu"]);
    calls.clear();
    let _ = build_script::backend_digests(false, false, |backend| {
        calls.push(backend.to_owned());
        Ok("unused".into())
    })
    .unwrap();
    assert!(calls.is_empty());
}

#[test]
fn cargo_config_scope_uses_invocation_directory_not_manifest_subdirectories() {
    let root = tempfile::tempdir().unwrap();
    let repo = root.path().join("repo");
    let manifest = repo.join("crates/omni-rsi");
    fs::create_dir_all(&manifest).unwrap();
    let dirs = build_script::config_dirs(&repo, &manifest, None);
    assert!(dirs.contains(&repo.join(".cargo")));
    assert!(!dirs.contains(&manifest.join(".cargo")));
}

#[cfg(unix)]
#[test]
fn cargo_config_can_be_a_symlink_to_a_regular_file() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("actual.toml");
    fs::write(&target, b"[alias]\npatch = 'fmt --check'\n").unwrap();
    let path = root.path().join("config.toml");
    std::os::unix::fs::symlink(&target, &path).unwrap();
    assert_eq!(
        build_script::read_config(&path).unwrap().unwrap(),
        fs::read(target).unwrap()
    );
}

#[cfg(unix)]
#[test]
fn aliased_manifest_and_repo_share_one_normalized_path() {
    let root = tempfile::tempdir().unwrap();
    let real = root.path().join("real");
    let manifest = real.join("crates/omni-rsi");
    fs::create_dir_all(&manifest).unwrap();
    let alias = root.path().join("alias");
    std::os::unix::fs::symlink(&real, &alias).unwrap();
    let (manifest, repo) = build_script::manifest_and_repo(&alias.join("crates/omni-rsi")).unwrap();
    assert!(
        manifest.starts_with(repo),
        "manifest and repository must use the same normalization"
    );
}

#[test]
fn cargo_rebuilds_when_the_cargo_home_changes() {
    use std::process::Command;
    let root = tempfile::tempdir().unwrap();
    let project = root.path().join("project");
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        "[package]\nname='cargo-home-switch-fixture'\nversion='0.1.0'\nedition='2024'\n",
    )
    .unwrap();
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("build_environment.rs");
    fs::write(
        project.join("build.rs"),
        format!(
            r#"
#[path = {:?}] mod config;
fn main() {{
    config::watch_configuration_environment();
    let value = match config::cargo_home_from_env() {{
        Some(home) => {{
            let input = std::fs::read_to_string(home.join("config.toml")).unwrap();
            if input.contains("value-A") {{ "A" }} else {{ "B" }}
        }},
        None => "UNVERIFIED",
    }};
    println!("cargo:rustc-env=OBSERVED_CONFIG={{value}}");
}}
"#,
            helper
        ),
    )
    .unwrap();
    fs::write(
        project.join("src/main.rs"),
        "fn main() { println!(\"{}\", env!(\"OBSERVED_CONFIG\")); }\n",
    )
    .unwrap();
    let home_a = root.path().join("cargo-a");
    let home_b = root.path().join("cargo-b");
    for (home, value) in [(&home_a, "value-A"), (&home_b, "value-B")] {
        fs::create_dir(home).unwrap();
        fs::write(
            home.join("config.toml"),
            format!("[alias]\nmarker = \"{value}\"\n"),
        )
        .unwrap();
    }
    let run = |home: &Path| {
        let result = Command::new(env!("CARGO"))
            .current_dir(&project)
            .args(["run", "--offline", "--quiet"])
            .env("CARGO_HOME", home)
            .env("CARGO_TARGET_DIR", root.path().join("target"))
            .env("CARGO_BUILD_JOBS", "1")
            .env("CARGO_INCREMENTAL", "0")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap().trim().to_owned()
    };
    assert_eq!(run(&home_a), "A");
    assert_eq!(run(&home_b), "B", "Cargo home changed with the same target");

    #[cfg(unix)]
    {
        let default_a = root.path().join("default-a");
        let default_b = root.path().join("default-b");
        for (home, value) in [(&default_a, "value-A"), (&default_b, "value-B")] {
            fs::create_dir_all(home.join(".cargo")).unwrap();
            fs::write(
                home.join(".cargo/config.toml"),
                format!("[alias]\nmarker = \"{value}\"\n"),
            )
            .unwrap();
        }
        let rustup_home = std::env::var_os("RUSTUP_HOME").or_else(|| {
            std::env::var_os("HOME").map(|h| Path::new(&h).join(".rustup").into_os_string())
        });
        let run_default = |home: &Path| {
            let mut command = Command::new(env!("CARGO"));
            command
                .current_dir(&project)
                .args(["run", "--offline", "--quiet"])
                .env_remove("CARGO_HOME")
                .env("HOME", home)
                .env("CARGO_TARGET_DIR", root.path().join("target"))
                .env("CARGO_BUILD_JOBS", "1")
                .env("CARGO_INCREMENTAL", "0");
            if let Some(rustup_home) = &rustup_home {
                command.env("RUSTUP_HOME", rustup_home);
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            String::from_utf8(result.stdout).unwrap().trim().to_owned()
        };
        let first = run_default(&default_a);
        let second = run_default(&default_b);
        assert!(
            (first == "A" && second == "B") || (first == "UNVERIFIED" && second == "UNVERIFIED"),
            "default-home contexts must either use protected CARGO_HOME or fail closed"
        );
    }
}

#[test]
#[ignore = "Cargo recursively watches excluded output below a source directory; retained as a diagnostic"]
fn ignored_output_changes_do_not_rerun_source_fingerprinting() {
    use std::process::Command;
    let root = tempfile::tempdir().unwrap();
    let project = root.path();
    let dependency = project.join("dependency");
    fs::create_dir(&dependency).unwrap();
    fixture(&dependency);
    fs::create_dir_all(dependency.join("target/debug")).unwrap();
    fs::create_dir(project.join("src")).unwrap();
    fs::write(project.join("Cargo.toml"), "[package]\nname='ignored-output-fixture'\nversion='0.1.0'\nedition='2024'\n[build-dependencies]\nsha2='0.10'\nlibc='0.2'\n").unwrap();
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("build_backend.rs");
    fs::write(project.join("build.rs"), format!(r#"
#[path = {:?}] mod backend;
fn main() {{
    let (digest, paths) = backend::source_tree_digest(std::path::Path::new("dependency")).unwrap();
    for path in paths {{ println!("cargo:rerun-if-changed={{}}", path.display()); }}
    let count = std::fs::read_to_string("runs").ok().and_then(|x| x.parse::<u32>().ok()).unwrap_or(0) + 1;
    std::fs::write("runs", count.to_string()).unwrap();
    println!("cargo:rustc-env=BACKEND_FIXTURE={{digest}}");
}}
"#, helper)).unwrap();
    fs::write(
        project.join("src/main.rs"),
        "fn main() { println!(\"{}\", env!(\"BACKEND_FIXTURE\")); }\n",
    )
    .unwrap();
    let run = || {
        let result = Command::new(env!("CARGO"))
            .current_dir(project)
            .args(["run", "--offline", "--quiet"])
            .env("CARGO_TARGET_DIR", project.join("outside-target"))
            .env("CARGO_BUILD_JOBS", "1")
            .env("CARGO_INCREMENTAL", "0")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap()
    };
    let before = run();
    assert_eq!(fs::read_to_string(project.join("runs")).unwrap(), "1");
    fs::write(
        dependency.join("target/debug/generated.o"),
        "ignored build output",
    )
    .unwrap();
    assert_eq!(before, run());
    assert_eq!(
        fs::read_to_string(project.join("runs")).unwrap(),
        "1",
        "ignored output needlessly reran the build script"
    );
}

#[test]
fn cargo_rebuild_refreshes_identity_for_source_edits_additions_and_removals() {
    use std::process::Command;
    let root = tempfile::tempdir().unwrap();
    let package = root.path();
    let dependency = package.join("dependency");
    fs::create_dir(&dependency).unwrap();
    fixture(&dependency);
    fs::create_dir(package.join("src")).unwrap();
    fs::write(package.join("Cargo.toml"), "[package]\nname='backend-refresh-fixture'\nversion='0.1.0'\nedition='2024'\n[build-dependencies]\nsha2='0.10'\nlibc='0.2'\n").unwrap();
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("build_backend.rs");
    fs::write(
        package.join("build.rs"),
        format!(
            r#"
#[path = {:?}] mod backend;
fn main() {{
    let (digest, paths) = backend::source_tree_digest(std::path::Path::new("dependency")).unwrap();
    for path in paths {{ println!("cargo:rerun-if-changed={{}}", path.display()); }}
    println!("cargo:rustc-env=BACKEND_FIXTURE={{digest}}");
}}
"#,
            helper
        ),
    )
    .unwrap();
    fs::write(
        package.join("src/main.rs"),
        "fn main() { println!(\"{}\", env!(\"BACKEND_FIXTURE\")); }\n",
    )
    .unwrap();
    let build = || {
        let result = Command::new(env!("CARGO"))
            .current_dir(package)
            .args(["run", "--quiet", "--offline"])
            .env("CARGO_TARGET_DIR", package.join("isolated-target"))
            .env("CARGO_BUILD_JOBS", "1")
            .env("CARGO_INCREMENTAL", "0")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap()
    };
    let first = build();
    assert_eq!(first, build());
    fs::write(dependency.join("src/kernel.rs"), "changed").unwrap();
    let changed = build();
    assert_ne!(first, changed);
    fs::write(dependency.join("src/new.metal"), "new").unwrap();
    let added = build();
    assert_ne!(changed, added);
    fs::remove_file(dependency.join("src/new.metal")).unwrap();
    assert_eq!(changed, build());
    fs::write(dependency.join("new-build.rs"), "new root source").unwrap();
    assert_ne!(changed, build());
    fs::remove_file(dependency.join("new-build.rs")).unwrap();
    assert_eq!(changed, build());
}
