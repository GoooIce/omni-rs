"""Opt-in checks for direct Cargo dependencies and local feature wiring.

API: dependency_failures(root, config) -> list[str]. No Cargo subprocesses.
The existing config["manifests"] list supplies {"path": "…/Cargo.toml"} rows.
Omit dependency_policy to disable these checks; {} enables the base rules:

    "dependency_policy": {
        "unsupported_features": ["wasm", "webgpu", "vulkan"],
        "training_manifests": ["crates/omni-rsi/Cargo.toml"],
        "ndarray_fallbacks": {
            "crates/burn-ane/Cargo.toml": "ANE host operations need CPU fallback"
        }
    }

Direct burn/burn-* and cubecl/cubecl-* dependencies require =0.21.0 and
=0.10.0 respectively; wgpu requires =29.0.4. All require an explicit
default-features = false, including development and path dependencies.
Development dependencies may enable reference CPU/autodiff implementations.
Default inference rejects CPU effects except the explicit ANE fallback.
training_manifests only permits training in the default feature closure.
ndarray_fallbacks only permits direct production burn-ndarray in a package
actually named burn-ane, with a nonempty reason.

Repository-local path targets outside vendor/ are first-party. Their declared
default features, even an empty list, require default-features = false.
Target cfg expressions are not evaluated. Weak dependency feature requests
are conservatively inspected even when their optional dependency is inactive.
Workspace inheritance is reported as unsupported, rather than certified.
This is not a Cargo resolver: use cargo tree to verify cross-package feature
propagation, feature unification, external defaults and the resolved graph.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


_KINDS = ("dependencies", "build-dependencies", "dev-dependencies")
_BACKENDS = {"burn-ndarray": "cpu", "burn-autodiff": "training"}
_BURN_FEATURES = {"ndarray": "cpu", "cpu": "cpu", "autodiff": "training", "train": "training"}


@dataclass
class _Dependency:
    scope: str
    kind: str
    alias: str
    spec: dict

    @property
    def package(self):
        return self.spec.get("package", self.alias)


@dataclass
class _Closure:
    local: set
    requests: set
    activated: set


def _dependencies(manifest):
    scopes = [("", manifest)]
    scopes += [(f"target[{selector!r}].", table)
               for selector, table in manifest.get("target", {}).items()]
    for prefix, table in scopes:
        for kind in _KINDS:
            for alias, value in table.get(kind, {}).items():
                spec = {"version": value} if isinstance(value, str) else value
                yield _Dependency(prefix + kind, kind, alias, spec)


def _pin(package):
    if package == "burn" or package.startswith("burn-"):
        return "=0.21.0"
    if package == "cubecl" or package.startswith("cubecl-"):
        return "=0.10.0"
    if package == "wgpu":
        return "=29.0.4"
    return None


def _closure(features, dependencies, start, include_base=False):
    """Follow local edges, aliases, dep: activation and dependency feature lists."""
    local, requests, activated, seen = set(), set(), set(), set()
    pending = [start]

    def activate(alias):
        if alias in activated or alias not in dependencies:
            return
        activated.add(alias)
        for dep in dependencies[alias]:
            pending.extend(f"{alias}/{name}" for name in dep.spec.get("features", []))

    if include_base:
        for alias, declarations in dependencies.items():
            if any(dep.spec.get("optional") is not True for dep in declarations):
                activate(alias)

    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        if item in features:
            local.add(item)
            pending.extend(features[item])
        elif item.startswith("dep:"):
            activate(item[4:])
        elif "/" in item:
            alias, name = item.split("/", 1)
            requests.add((alias.removesuffix("?"), name))
            if not alias.endswith("?"):
                activate(alias)
        else:
            # Cargo's implicit feature for an optional dependency.
            activate(item)
    return _Closure(local, requests, activated)


def _forwarded_effects(closure, dependencies):
    effects = set()
    for alias, name in closure.requests:
        for dep in dependencies.get(alias, []):
            category = name if name in ("cpu", "training") else None
            if dep.package == "burn":
                category = _BURN_FEATURES.get(name, category)
            if category:
                effects.add((category, f"{alias}/{name}"))
    return effects


def _effects(closure, dependencies):
    effects = _forwarded_effects(closure, dependencies)
    effects.update((name, f"local feature {name}")
                   for name in closure.local & {"cpu", "training"})
    for alias in closure.activated:
        for dep in dependencies[alias]:
            if dep.package in _BACKENDS:
                effects.add((_BACKENDS[dep.package], f"dep:{alias} ({dep.package})"))
    return effects


def dependency_failures(root, config) -> list[str]:
    """Return deterministic, manifest/scope-qualified policy diagnostics."""
    if "dependency_policy" not in config:
        return []
    policy = config["dependency_policy"]
    if not isinstance(policy, dict):
        return ["dependency_policy: expected an object (omit it to disable checks)"]
    root = Path(root).resolve()
    training = set(policy.get("training_manifests", []))
    unsupported = set(policy.get("unsupported_features", []))
    fallbacks = policy.get("ndarray_fallbacks", {})
    failures, cache = [], {}

    def label(path):
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)

    def read(path):
        if path not in cache:
            try:
                cache[path] = tomllib.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
                failures.append(f"{label(path)}: cannot read Cargo manifest: {error}")
                cache[path] = None
        return cache[path]

    for row in config["manifests"]:
        path = (root / row["path"]).resolve()
        manifest = read(path)
        if manifest is None:
            continue
        relative = label(path)
        dependencies = list(_dependencies(manifest))
        production = {}
        for dep in dependencies:
            if dep.kind != "dev-dependencies":
                production.setdefault(dep.alias, []).append(dep)
        features = manifest.get("features", {})
        for name, members in features.items():
            if name in unsupported:
                failures.append(f"{relative}: features.{name} is no longer supported")
            for member in members:
                requested = member.split("/", 1)[-1]
                if requested in unsupported:
                    failures.append(f"{relative}: features.{name} requests unsupported feature {member}")
        closures = {name: _closure(features, production, name)
                    for name in features.keys() | {"cpu", "ane", "training"}}
        cpu_effects = (_forwarded_effects(closures["cpu"], production)
                       | _forwarded_effects(closures["ane"], production))
        training_effects = _forwarded_effects(closures["training"], production)

        reason = fallbacks.get(relative)
        fallback = (manifest.get("package", {}).get("name") == "burn-ane"
                    and isinstance(reason, str) and bool(reason.strip()))
        for dep in dependencies:
            context = f"{relative}: {dep.scope}.{dep.alias} ({dep.package})"
            for name in sorted(set(dep.spec.get("features", [])) & unsupported):
                failures.append(f"{context}: requests unsupported feature {name}")
            if dep.spec.get("workspace"):
                failures.append(f"{context}: workspace inheritance cannot be checked statically; "
                                "use an explicit declaration")
                continue
            pin = _pin(dep.package)
            if pin:
                if dep.spec.get("version") != pin:
                    failures.append(f"{context}: version must be {pin!r}")
                if dep.spec.get("default-features") is not False:
                    failures.append(f"{context}: default-features must be false")

            if "path" in dep.spec:
                target = (path.parent / dep.spec["path"] / "Cargo.toml").resolve()
                if target.is_relative_to(root) and target.relative_to(root).parts[0] != "vendor":
                    target_manifest = read(target)
                    if (target_manifest is not None
                            and "default" in target_manifest.get("features", {})
                            and dep.spec.get("default-features") is not False):
                        failures.append(f"{context}: first-party path {label(target)} declares "
                                        "default features; default-features must be false")

            if dep.kind == "dev-dependencies":
                continue
            if dep.package == "burn":
                for name in dep.spec.get("features", []):
                    if name not in _BURN_FEATURES:
                        continue
                    gates = ("cpu", "ane") if _BURN_FEATURES[name] == "cpu" else ("training",)
                    if (dep.spec.get("optional") is not True
                            or not any(dep.alias in closures[gate].activated for gate in gates)):
                        failures.append(f"{context}: {name} must be enabled through local "
                                        f"feature {' or '.join(gates)}, not unconditionally")
            if dep.package == "burn-ndarray" and not fallback:
                failures.append(f"{context}: production burn-ndarray is only allowed for a "
                                "burn-ane fallback with a nonempty ndarray_fallbacks reason")
            if dep.package == "burn-autodiff":
                if dep.spec.get("optional") is not True:
                    failures.append(f"{context}: production burn-autodiff must be optional")
                if dep.alias not in closures["training"].activated:
                    failures.append(f"{context}: burn-autodiff must be activated by local "
                                    "feature training")

        forwarded = set().union(*(_forwarded_effects(value, production)
                                  for value in closures.values()))
        for category, request in sorted(forwarded):
            allowed = cpu_effects if category == "cpu" else training_effects
            if (category, request) not in allowed:
                gates = "cpu or ane" if category == "cpu" else "training"
                failures.append(f"{relative}: {request} must be enabled through local feature {gates}")

        for name in ("metal", "webgpu", "default"):
            if name != "default" and name not in features:
                continue
            effects = _effects(_closure(features, production, name, include_base=True), production)
            if fallback:
                effects -= {("cpu", f"dep:{dep.alias} (burn-ndarray)")
                            for dep in dependencies
                            if dep.kind != "dev-dependencies" and dep.package == "burn-ndarray"}
            if name == "default":
                # The explicit ANE gate may request its host fallback. Its
                # presence does not exempt another model's normal CPU feature.
                effects -= {effect for effect in _forwarded_effects(closures["ane"], production)
                            if effect[0] == "cpu"}
                if relative in training:
                    effects = {effect for effect in effects if effect[0] != "training"}
            if effects:
                details = ", ".join(f"{category} via {source}" for category, source in sorted(effects))
                failures.append(f"{relative}: features.{name} reaches {details}")
    return failures
