#!/usr/bin/env python3
"""Check unified workspace dev/test profiles, macOS split-debuginfo, and explicit debug-full opt-in."""
import json
import os
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]

_PROFILE_KEYS = ("debug", "split-debuginfo", "incremental", "opt-level")


def profile_failures(root, config):
    failures = []
    policies = config.get("profiles")
    if not policies or set(policies) != {"dev", "debug-full"}:
        return ["scripts/checks.json: declare dev and debug-full profile policies"]
    root_manifest = root / "Cargo.toml"
    if not root_manifest.is_file():
        return ["Cargo.toml: missing workspace root manifest"]
    root_value = tomllib.loads(root_manifest.read_text())
    root_profiles = root_value.get("profile", {})

    for name, expected in policies.items():
        actual = root_profiles.get(name, {})
        for key, value in expected.items():
            if key == "package_debug":
                pkg_wild = actual.get("package", {}).get("*", {})
                if pkg_wild.get("debug") != value:
                    failures.append(
                        f'Cargo.toml: profile.{name}.package."*".debug must be {value!r}'
                    )
                continue
            if actual.get(key) != value:
                failures.append(f"Cargo.toml: profile.{name}.{key} must be {value!r}")

    dev_policy = policies["dev"]
    pkg_debug_expected = dev_policy.get("package_debug")
    for name in ("dev", "test"):
        actual = root_profiles.get(name, {})
        for key in ("debug", "split-debuginfo", "incremental"):
            if key not in dev_policy:
                continue
            expected = dev_policy[key]
            if key in actual and actual[key] != expected:
                failures.append(f"Cargo.toml: profile.{name}.{key} overrides the routine-check policy")
            for package, overrides in actual.get("package", {}).items():
                if key not in overrides:
                    continue
                if key == "debug" and package == "*" and pkg_debug_expected is not None:
                    if overrides[key] != pkg_debug_expected:
                        failures.append(
                            f'Cargo.toml: profile.{name}.package."*".debug must be {pkg_debug_expected!r}'
                        )
                elif overrides[key] != expected:
                    failures.append(
                        f"Cargo.toml: profile.{name}.package.{package}.{key} "
                        "overrides the routine-check policy"
                    )
            overrides = actual.get("build-override", {})
            if key in overrides and overrides[key] != expected:
                failures.append(
                    f"Cargo.toml: profile.{name}.build-override.{key} "
                    "overrides the routine-check policy"
                )

    for row in config["manifests"]:
        if row["path"] == "Cargo.toml":
            continue
        path = root / row["path"]
        if not path.is_file():
            continue
        member = tomllib.loads(path.read_text())
        if member.get("profile"):
            failures.append(
                f"{row['path']}: workspace member must not declare [profile.*]; "
                "profiles are defined once in root Cargo.toml"
            )
        if member.get("patch"):
            failures.append(
                f"{row['path']}: workspace member must not declare [patch.*]; "
                "vendor patches are defined once in root Cargo.toml"
            )
        if "workspace" in member:
            failures.append(
                f"{row['path']}: workspace member must not declare standalone [workspace]"
            )
    return failures


def environment_failures(root, profile="dev", environ=None):
    """Reject hidden overrides instead of silently certifying different settings."""
    env = os.environ if environ is None else environ
    config = json.loads((root / "scripts/checks.json").read_text())
    expected = config["profiles"].get("dev" if profile == "test" else profile)
    names = ("dev", "test") if profile in ("dev", "test") else ((profile,) if expected else ())
    failures = []

    def matches(key, value):
        if key == "incremental":
            return str(value).lower() in (("true", "1") if expected[key] else ("false", "0"))
        return value == expected[key] or str(value) == str(expected[key])

    checked_keys = [k for k in ("debug", "split-debuginfo", "incremental") if expected and k in expected]

    for name in names:
        for key in checked_keys:
            env_key = key.upper().replace("-", "_")
            for suffix in (env_key, "BUILD_OVERRIDE_" + env_key):
                variable = f"CARGO_PROFILE_{name.upper().replace('-', '_')}_{suffix}"
                if variable in env and not matches(key, env[variable]):
                    failures.append(f"{variable} overrides profile {profile}; "
                                    "use --profile debug-full for full debugging")
    if expected and "CARGO_INCREMENTAL" in env and not matches("incremental", env["CARGO_INCREMENTAL"]):
        failures.append(f"CARGO_INCREMENTAL overrides profile {profile}")

    def flag_override(value):
        text = " ".join(value) if isinstance(value, list) else str(value)
        return re.search(r"(?:debuginfo|split-debuginfo|incremental)(?:=|\s)", text)

    for key in ("RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
        if expected and flag_override(env.get(key, "")):
            failures.append(f"{key} contains a debug/incremental override; use a registered profile")
    if env.get("CARGO_BUILD_BUILD_DIR"):
        failures.append("CARGO_BUILD_BUILD_DIR separates intermediates from the registered target")

    starts = [root, *[(root / row["path"]).parent for row in config["manifests"]]]
    cargo_home = Path(env.get("CARGO_HOME", Path.home() / ".cargo"))
    files = {cargo_home / ("config" if (cargo_home / "config").is_file() else "config.toml")}
    for start in starts:
        for directory in (start, *start.parents):
            legacy = directory / ".cargo/config"
            files.add(legacy if legacy.is_file() else directory / ".cargo/config.toml")
    pkg_debug_expected = expected.get("package_debug") if expected else None
    for path in sorted(files):
        if not path.is_file():
            continue
        value = tomllib.loads(path.read_text())
        for name in names:
            actual = value.get("profile", {}).get(name, {})
            scopes = [(f"profile.{name}", actual, None),
                      (f"profile.{name}.build-override", actual.get("build-override", {}), None)]
            scopes += [(f"profile.{name}.package.{package}", overrides, package)
                       for package, overrides in actual.get("package", {}).items()]
            for label, overrides, package in scopes:
                for key in checked_keys:
                    if key not in overrides:
                        continue
                    if key == "debug" and package == "*" and pkg_debug_expected is not None:
                        if overrides[key] != pkg_debug_expected:
                            failures.append(f"{path}: {label}.{key} overrides profile {profile}")
                    elif not matches(key, overrides[key]):
                        failures.append(f"{path}: {label}.{key} overrides profile {profile}")
        if expected and "incremental" in value.get("build", {}) and \
                not matches("incremental", value["build"]["incremental"]):
            failures.append(f"{path}: build.incremental overrides profile {profile}")
        if "build-dir" in value.get("build", {}):
            failures.append(f"{path}: build.build-dir separates intermediates from the registered target")
        flag_scopes = [("build", value.get("build", {})),
                       *[(f"target.{name}", table) for name, table in value.get("target", {}).items()]]
        for label, table in flag_scopes:
            if expected and flag_override(table.get("rustflags", "")):
                failures.append(f"{path}: {label}.rustflags overrides the registered profile")
        for variable, definition in value.get("env", {}).items():
            configured = definition.get("value", "") if isinstance(definition, dict) else definition
            if variable == "CARGO_BUILD_BUILD_DIR":
                failures.append(f"{path}: env.{variable} separates Cargo intermediates")
            if expected and variable == "CARGO_INCREMENTAL" and not matches("incremental", configured):
                failures.append(f"{path}: env.{variable} overrides profile {profile}")
            if expected and variable in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS") \
                    and flag_override(configured):
                failures.append(f"{path}: env.{variable} overrides the registered profile")
    return failures


def main():
    config = json.loads((ROOT / "scripts/checks.json").read_text())
    failures = profile_failures(ROOT, config) + environment_failures(ROOT)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Unified workspace profile verified: {len(config['manifests'])} manifests; "
          "routine dev uses split-debuginfo=off; full debugging: --profile debug-full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
