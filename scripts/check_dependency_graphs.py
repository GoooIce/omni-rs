#!/usr/bin/env python3
"""Inspect locked normal/build Cargo graphs without compiling or running a device."""
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VULKAN_PACKAGES = {
    "ash", "gpu-allocator", "gpu-descriptor", "gpu-descriptor-types",
    "presser", "renderdoc-sys", "spirv",
}


def parse_tree(text):
    packages = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        package, separator, features = line.partition("|")
        match = re.fullmatch(r"([\w-]+) v(\S+?)(?: \((.+)\))?", package)
        if not separator or not match:
            raise ValueError(f"unrecognized Cargo tree row: {line}")
        name, version, source = match.groups()
        key = (name, version, source)
        packages.setdefault(key, set()).update(
            value for value in features.removesuffix(" (*)").split(",") if value
        )
    return [dict(name=name, version=version, source=source, features=sorted(features))
            for (name, version, source), features in sorted(
                packages.items(), key=lambda item: tuple(value or "" for value in item[0]))]


def graph_failures(root, packages, policy, *, training=False, macos=True):
    failures = []
    names = {package["name"] for package in packages}
    # NdArray itself depends on burn-autodiff. The supported ANE fallback can
    # therefore legitimately include both, even in an inference graph.
    root = root.resolve()
    fallbacks = policy.get("ndarray_fallbacks", {})
    cpu_fallback = any(
        package["name"] == "burn-ane"
        and package["source"] == str((root / manifest).parent.resolve())
        and isinstance(reason, str) and bool(reason.strip())
        for package in packages for manifest, reason in fallbacks.items()
    )
    if "burn-ndarray" in names and not cpu_fallback:
        failures.append("normal GPU graph includes burn-ndarray without the ANE fallback")
    if "burn-autodiff" in names and not (training or cpu_fallback):
        failures.append("normal inference graph includes burn-autodiff")
    for package in packages:
        name = package["name"]
        source = Path(package["source"]) if package["source"] else None
        # Cargo also prints git URLs in the source field. Resolving a URL as a
        # relative path would incorrectly classify it as part of this checkout.
        source = source.resolve() if source is not None and source.is_absolute() else None
        if source and source.is_relative_to(root) and not source.is_relative_to(root / "vendor"):
            features = set(package["features"])
            for feature in sorted(features & set(policy.get("unsupported_features", []))):
                failures.append(f"first-party consumer {name} enables unsupported {feature}")
            if "cpu" in features:
                failures.append(f"first-party GPU consumer {name} enables cpu")
            manifest = (source / "Cargo.toml").relative_to(root).as_posix()
            if "training" in features and not (
                training and manifest in policy.get("training_manifests", [])
            ):
                failures.append(f"first-party inference consumer {name} enables training")
        expected = policy.get("vendored_paths", {}).get(name)
        if expected and package["source"] != str((root / expected).resolve()):
            failures.append(f"{name} resolves to {package['source'] or 'the registry'}, expected {expected}")
        expected_version = policy.get("aligned_versions", {}).get(name)
        if expected_version and package["version"] != expected_version:
            failures.append(f"{name} resolves to {package['version']}, expected {expected_version}")
        if macos and name in VULKAN_PACKAGES:
            failures.append(f"normal macOS GPU graph includes unused Vulkan package {name}")
        if macos and name == "wgpu":
            required = {"metal", "wgsl"}
            if not required <= set(package["features"]):
                failures.append(f"macOS WGPU graph is missing {sorted(required - set(package['features']))}")
            unsupported = {"webgpu", "webgl", "vulkan", "dx12", "gles"} & set(package["features"])
            if unsupported:
                failures.append(f"native Metal WGPU graph enables unsupported {sorted(unsupported)}")
    return failures


def cases(root, config):
    training = set(config["dependency_policy"].get("training_manifests", []))
    for row in config["manifests"]:
        manifest = row["path"]
        value = tomllib.loads((root / manifest).read_text())
        features = value.get("features", {})
        yield dict(manifest=manifest, mode="default", flags=[], training=manifest in training)
        for backend in ("metal", "ane"):
            if backend in features:
                selected = (["native"] if "native" in features else []) + [backend]
                yield dict(manifest=manifest, mode=backend,
                           flags=["--no-default-features", "--features", ",".join(selected)],
                           training=False)
        if value.get("package", {}).get("name") == "burn-ane":
            yield dict(manifest=manifest, mode="metal-shared",
                       flags=["--features", "metal-shared"], training=False)


def check(root, report_path):
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        raise ValueError("Native dependency graphs require Apple Silicon macOS")
    config = json.loads((root / "scripts/checks.json").read_text())
    policy = config["dependency_policy"]
    host = subprocess.check_output(["rustc", "-vV"], text=True, timeout=15)
    target = next(line.removeprefix("host: ") for line in host.splitlines() if line.startswith("host: "))
    report = {"target": target, "status": "running", "graphs": []}
    failures = []
    for case in cases(root, config):
        command = ["cargo", "tree", "--offline", "--locked", "--manifest-path", case["manifest"],
                   "--target", target, "--edges", "normal,build", "--prefix", "none",
                   "--format", "{p}|{f}", *case["flags"]]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=90)
        row = dict(case, command=command, exit_code=result.returncode, stderr=result.stderr)
        if result.returncode:
            errors = [result.stderr[-2000:]]
        else:
            row["packages"] = parse_tree(result.stdout)
            row["package_count"] = len(row["packages"])
            errors = graph_failures(
                root, row["packages"], policy, training=case["training"],
                macos=True)
        row["failures"] = errors
        report["graphs"].append(row)
        failures += [f"{case['manifest']} [{case['mode']}]: {error}" for error in errors]
        print(f"{'FAIL' if errors else 'PASS'} {case['manifest']} [{case['mode']}]", flush=True)
    report["status"] = "failed" if failures else "passed"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if failures:
        raise ValueError("\n".join(failures))
    return len(report["graphs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(f"Checked {check(args.root.resolve(), args.report.resolve())} normal/build graphs.")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
