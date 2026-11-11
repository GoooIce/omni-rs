//! Verify Chinese inputs against the existing frozen protocol without changing
//! the v3 training identity. File operations reuse the voice recheck's bounded,
//! descriptor-relative no-follow readers and its strict protocol hash mirror.

use crate::{
    number_development::RegisteredDevelopment,
    number_regression::ChineseManifest,
    number_v3_voice_recheck::{open_child, open_directory, read_child, strict_registered_protocol},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    ffi::OsStr,
    path::{Component, Path},
    sync::atomic::AtomicBool,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    version: u32,
    criterion: String,
    groups: Vec<Group>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    name: String,
    manifest_file: String,
}

struct Current {
    plan_sha256: String,
    manifest_file: String,
    manifest: ChineseManifest,
}

fn valid_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && name.len() <= 255
        && name.ends_with(".json")
        && !name.contains('\\')
        && !name.chars().any(char::is_control)
}

fn load(data: &Path, stop: &AtomicBool) -> Result<Current, String> {
    let root = open_directory(data)?;
    let mut remaining = 512 * 1024;
    let bytes = read_child(
        &root,
        OsStr::new("regression-plan.json"),
        256 * 1024,
        &mut remaining,
        stop,
    )?;
    let plan: Plan =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid regression plan: {e}"))?;
    let required: HashSet<_> = ["chinese", "english", "other_omni", "reading", "dialogue"]
        .into_iter()
        .collect();
    let mut seen = HashSet::new();
    if plan.version != 1
        || plan.criterion != "no_degradation"
        || plan.groups.len() != required.len()
    {
        return Err("invalid regression plan".into());
    }
    let mut chinese = None;
    for group in plan.groups {
        if !required.contains(group.name.as_str())
            || !seen.insert(group.name.clone())
            || !valid_name(&group.manifest_file)
        {
            return Err("invalid regression group".into());
        }
        if group.name == "chinese" {
            chinese = Some(group.manifest_file);
        }
    }
    let manifest_file = chinese.ok_or("missing Chinese regression")?;
    let directory = open_child(&root, OsStr::new("regression-inputs"), true)?;
    let manifest_bytes = read_child(
        &directory,
        OsStr::new(&manifest_file),
        256 * 1024,
        &mut remaining,
        stop,
    )?;
    Ok(Current {
        plan_sha256: format!("{:x}", Sha256::digest(&bytes)),
        manifest_file,
        manifest: ChineseManifest::from_bytes(&manifest_bytes)?,
    })
}

pub(super) fn current(data: &Path, stop: &AtomicBool) -> Result<ChineseManifest, String> {
    Ok(load(data, stop)?.manifest)
}

pub(super) fn frozen(
    data: &Path,
    registration: &Path,
    registered: &RegisteredDevelopment,
    stop: &AtomicBool,
) -> Result<ChineseManifest, String> {
    let directory = open_directory(registration)?;
    let mut remaining = 32 * 1024 * 1024;
    let bytes = read_child(
        &directory,
        OsStr::new("protocol.json"),
        32 * 1024 * 1024,
        &mut remaining,
        stop,
    )?;
    let protocol = strict_registered_protocol(&bytes, registered.protocol_sha256(), stop)?;
    let current = load(data, stop)?;
    let inputs = protocol["regression_inputs"]
        .as_array()
        .ok_or("missing frozen regression inputs")?;
    let mut chinese = inputs.iter().filter(|item| item["name"] == "chinese");
    let identity = chinese.next().ok_or("missing frozen Chinese regression")?;
    if chinese.next().is_some()
        || protocol["regression_plan_sha256"] != current.plan_sha256
        || identity["manifest_file"] != current.manifest_file
        || identity["sha256"] != current.manifest.sha256()
    {
        return Err("frozen Chinese regression plan or manifest changed".into());
    }
    Ok(current.manifest)
}
