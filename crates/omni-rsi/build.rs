//! Compile-time identities for this repository's supported local Cargo patches.
pub mod build_backend;
pub mod build_environment;
pub mod build_invocation;

use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

fn field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

pub fn config_dirs(cwd: &Path, _manifest: &Path, home: Option<&Path>) -> BTreeSet<PathBuf> {
    let mut dirs = BTreeSet::new();
    for parent in cwd.ancestors() {
        dirs.insert(parent.join(".cargo"));
    }
    if let Some(home) = home {
        dirs.insert(home.to_path_buf());
    }
    dirs
}

pub fn read_config(path: &Path) -> Result<Option<Vec<u8>>, String> {
    const MAX_CONFIG: u64 = 4 * 1024 * 1024;
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Cargo permits symlinked configs; do not block on a substituted FIFO.
        options.custom_flags(libc::O_NONBLOCK);
    }
    match options.open(path) {
        Ok(file) => {
            let meta = file.metadata().map_err(|e| e.to_string())?;
            if !meta.is_file() || meta.len() > MAX_CONFIG {
                return Err(format!("invalid Cargo configuration: {}", path.display()));
            }
            let mut bytes = Vec::new();
            file.take(MAX_CONFIG + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            if bytes.len() as u64 > MAX_CONFIG {
                return Err(format!(
                    "Cargo configuration exceeds size limit: {}",
                    path.display()
                ));
            }
            Ok(Some(bytes))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

pub fn manifest_and_repo(manifest: &Path) -> Result<(PathBuf, PathBuf), String> {
    let manifest = fs::canonicalize(manifest).map_err(|e| e.to_string())?;
    let repo = manifest
        .parent()
        .and_then(Path::parent)
        .ok_or("missing repo")?
        .to_path_buf();
    Ok((manifest, repo))
}

fn watch_config_changes() -> Result<(), String> {
    build_environment::watch_configuration_environment();
    // Cargo follows symlinks when watching files and does not track files that
    // did not exist when a build script last ran. Force a recheck on each Cargo
    // invocation without recursively watching Cargo home or filesystem roots.
    let guard = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing build output directory")?)
        .join("__omni_rsi_recheck_cargo_configuration__");
    match fs::symlink_metadata(&guard) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Ok(_) => return Err("reserved Cargo configuration guard already exists".into()),
        Err(error) => return Err(error.to_string()),
    }
    println!("cargo:rerun-if-changed={}", guard.display());
    Ok(())
}

pub fn config_identity(hash: &mut Sha256, cwd: &Path, manifest: &Path) -> Result<(), String> {
    watch_config_changes()?;
    let home = build_environment::cargo_home_from_env().ok_or("cannot determine Cargo home")?;
    let home = if home.is_absolute() {
        home
    } else {
        cwd.join(home)
    };
    let dirs = config_dirs(cwd, manifest, Some(&home));
    for dir in dirs {
        // Watching Cargo home itself would rebuild on unrelated registry/cache changes.
        // Watch existing config files, and an existing repository-local .cargo directory.
        if dir.starts_with(
            manifest
                .parent()
                .and_then(Path::parent)
                .ok_or("missing repo")?,
        ) && dir.is_dir()
        {
            println!("cargo:rerun-if-changed={}", dir.display());
        }
        // Cargo gives the legacy `config` precedence when both names exist.
        for name in ["config", "config.toml"] {
            let path = dir.join(name);
            if let Some(bytes) = read_config(&path)? {
                let config = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
                build_backend::validate_config(config)?;
                field(hash, name.as_bytes());
                field(hash, &bytes);
                println!("cargo:rerun-if-changed={}", path.display());
                if let Ok(target) = fs::canonicalize(&path)
                    && target != path
                {
                    println!("cargo:rerun-if-changed={}", target.display());
                }
                break;
            }
        }
    }
    for name in [
        "TARGET",
        "HOST",
        "PROFILE",
        "OPT_LEVEL",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CC",
        "CXX",
        "AR",
        "CFLAGS",
        "CXXFLAGS",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
    ] {
        field(hash, name.as_bytes());
        field(hash, env::var(name).unwrap_or_default().as_bytes());
        if !matches!(
            name,
            "TARGET" | "HOST" | "PROFILE" | "OPT_LEVEL" | "CARGO_ENCODED_RUSTFLAGS"
        ) {
            println!("cargo:rerun-if-env-changed={name}");
        }
    }
    for name in ["CARGO_FEATURE_CPU", "CARGO_FEATURE_METAL"] {
        field(hash, name.as_bytes());
        field(hash, env::var(name).unwrap_or_default().as_bytes());
    }
    Ok(())
}

// Follow local dependency/build-dependency paths even if optional. This is
// deliberately conservative; nested [patch] tables are ignored by Cargo.
pub fn local_dependencies(manifest: &str) -> Result<Vec<String>, String> {
    let mut dependencies = false;
    let mut result = Vec::new();
    for original in manifest.lines() {
        let line = build_backend::uncomment(original)?.trim();
        if line.starts_with('[') {
            let section = if line.starts_with("[[") {
                line.strip_prefix("[[")
                    .and_then(|s| s.strip_suffix("]]"))
                    .ok_or("invalid Cargo table")?
            } else {
                line.strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or("invalid Cargo table")?
            };
            let section = build_backend::keys(section)?;
            dependencies = section.first().is_none_or(|s| s != "patch")
                && section.iter().any(|s| {
                    s == "dependencies" || s == "build-dependencies" || s == "dev-dependencies"
                });
            continue;
        }
        if line.is_empty() {
            continue;
        }
        if let Ok((lhs, _)) = build_backend::assignment(line) {
            let key = build_backend::keys(lhs)?;
            if dependencies
                || key.first().is_some_and(|k| {
                    k == "dependencies" || k == "build-dependencies" || k == "dev-dependencies"
                })
            {
                build_backend::dependency_assignment(line, &mut result)?;
            }
        } else if dependencies && (line.contains('{') || line.contains('=')) {
            return Err("unsupported dependency syntax".into());
        }
    }
    Ok(result)
}

pub fn graph_identity(
    repo: &Path,
    initial: Vec<PathBuf>,
    common: &[u8],
    label: &str,
) -> Result<String, String> {
    let mut pending = initial;
    let mut visited = BTreeSet::new();
    let mut entries = Vec::new();
    while let Some(path) = pending.pop() {
        let mut checked = PathBuf::new();
        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    if !checked.pop() {
                        return Err("invalid backend source path".into());
                    }
                }
                std::path::Component::CurDir => (),
                _ => {
                    checked.push(component.as_os_str());
                    if checked.starts_with(repo) {
                        let meta = fs::symlink_metadata(&checked)
                            .map_err(|e| format!("{}: {e}", checked.display()))?;
                        if meta.file_type().is_symlink() {
                            return Err(format!(
                                "backend source contains symlink: {}",
                                checked.display()
                            ));
                        }
                    }
                }
            }
        }
        if !checked.starts_with(repo) {
            return Err("local backend path outside repository is unsupported".into());
        }
        let path = fs::canonicalize(&checked).map_err(|e| format!("{}: {e}", checked.display()))?;
        if !path.starts_with(repo) {
            return Err("local backend path outside repository is unsupported".into());
        }
        if !visited.insert(path.clone()) {
            continue;
        }
        let (digest, watches) = build_backend::source_tree_digest(&path)?;
        for watched in &watches {
            println!("cargo:rerun-if-changed={}", watched.display());
            if watched.file_name().is_some_and(|name| name == "Cargo.toml") {
                let contents = fs::read_to_string(watched).map_err(|e| e.to_string())?;
                for dependency in local_dependencies(&contents)? {
                    pending.push(
                        watched
                            .parent()
                            .ok_or("invalid dependency manifest")?
                            .join(dependency),
                    );
                }
            }
        }
        let relative = path
            .strip_prefix(repo)
            .map_err(|e| e.to_string())?
            .to_str()
            .ok_or("invalid source path")?;
        entries.push((relative.to_owned(), digest));
    }
    entries.sort();
    let mut hash = Sha256::new();
    field(&mut hash, b"omni-local-backend-graph-v1");
    field(&mut hash, label.as_bytes());
    field(&mut hash, common);
    for (path, digest) in entries {
        field(&mut hash, path.as_bytes());
        field(&mut hash, digest.as_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

// These files describe the CPU dependency graph whose local packages were
// inspected. Any graph change falls back to including every local patch.
pub fn known_cpu_graph(repo: &Path) -> Result<bool, String> {
    let inputs = [
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
            "crates/omni-rsi/Cargo.toml",
            "1f19d2ed09a4704028cada1ee7ae3a510ac728a5f43ed67d74b5a5864cd864fc",
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
    let mut known = true;
    for (name, expected) in inputs {
        let path = repo.join(name);
        let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        // The scanner manifest adds only a gated binary target; its dependency
        // graph and every other anchor are identical to the original manifest.
        known &= actual == expected
            || (name == "crates/omni-rsi/Cargo.toml"
                && actual == "f1502a1d91d591305f1675d2f382b902aa80e6bc05e07031716129205d1c22e2");
        println!("cargo:rerun-if-changed={}", path.display());
    }
    Ok(known)
}

/// Derive both backend root sets from the actual manifest (or workspace root) and graph anchors.
pub fn backend_roots(repo: &Path, manifest: &Path) -> Result<(Vec<PathBuf>, Vec<PathBuf>), String> {
    let local_text = fs::read_to_string(manifest.join("Cargo.toml")).map_err(|e| e.to_string())?;
    let (patch_base, patches) = match build_backend::patch_paths(&local_text) {
        Ok(patches) => (manifest, patches),
        Err(err) if err == "missing local Cargo patches" && manifest != repo => {
            let root_text =
                fs::read_to_string(repo.join("Cargo.toml")).map_err(|e| e.to_string())?;
            (repo, build_backend::patch_paths(&root_text)?)
        }
        Err(err) => return Err(err),
    };
    let known_cpu_graph = known_cpu_graph(repo)?;
    let gpu_only = [
        "burn-cubecl",
        "burn-fusion",
        "burn-router",
        "cubecl-cpp",
        "cubecl-runtime",
        "cubecl-wgpu",
    ];
    let mut cpu = Vec::new();
    let mut gpu = vec![repo.join("crates/omni-kernel")];
    for (name, path) in patches {
        let path = patch_base.join(path);
        if !known_cpu_graph || !gpu_only.contains(&name.as_str()) {
            cpu.push(path.clone());
        }
        gpu.push(path);
    }
    Ok((cpu, gpu))
}

pub fn backend_digests(
    cpu_enabled: bool,
    metal_enabled: bool,
    mut digest: impl FnMut(&str) -> Result<String, String>,
) -> Result<(String, String), String> {
    let cpu = if cpu_enabled {
        digest("cpu")?
    } else {
        "backend-not-enabled".into()
    };
    let gpu = if metal_enabled {
        digest("gpu")?
    } else {
        "backend-not-enabled".into()
    };
    Ok((cpu, gpu))
}

fn run() -> Result<(), String> {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing manifest")?);
    let (manifest, repo) = manifest_and_repo(&manifest)?;
    let unverified = |reason: &str| -> Result<(), String> {
        watch_config_changes()?;
        println!("cargo:warning={reason}; audio training identity is unavailable");
        println!("cargo:rustc-env=OMNI_RSI_BACKEND_CPU=backend-unverified-context");
        println!("cargo:rustc-env=OMNI_RSI_BACKEND_GPU=backend-unverified-context");
        Ok(())
    };
    let Some(cwd) = build_invocation::cargo_invocation_cwd() else {
        return unverified("Cannot verify Cargo's invocation directory");
    };
    if build_environment::cargo_home_from_env().is_none() {
        return unverified("Cannot determine Cargo home");
    }
    let manifest_file = manifest.join("Cargo.toml");
    let (cpu, gpu) = backend_roots(&repo, &manifest)?;
    let mut common = Sha256::new();
    field(&mut common, b"omni-backend-build-config-v1");
    config_identity(&mut common, &cwd, &manifest)?;
    let common = common.finalize();
    println!("cargo:rerun-if-changed={}", manifest_file.display());
    println!("cargo:rerun-if-changed=build_backend.rs");
    println!("cargo:rerun-if-changed=build_environment.rs");
    println!("cargo:rerun-if-changed=build_invocation.rs");
    let (cpu_digest, gpu_digest) = backend_digests(
        env::var_os("CARGO_FEATURE_CPU").is_some(),
        env::var_os("CARGO_FEATURE_METAL").is_some(),
        |backend| match backend {
            "cpu" => graph_identity(&repo, cpu.clone(), &common, "cpu"),
            _ => graph_identity(&repo, gpu.clone(), &common, "gpu"),
        },
    )?;
    println!("cargo:rustc-env=OMNI_RSI_BACKEND_CPU={cpu_digest}");
    println!("cargo:rustc-env=OMNI_RSI_BACKEND_GPU={gpu_digest}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        panic!("cannot fingerprint local numerical backend: {error}");
    }
}
