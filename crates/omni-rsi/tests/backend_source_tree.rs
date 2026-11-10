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
    ] {
        assert!(
            build_backend::validate_config(config).is_err(),
            "accepted {config}"
        );
    }
    assert!(build_backend::validate_config("[build]\njobs = 2\n").is_ok());
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
}
