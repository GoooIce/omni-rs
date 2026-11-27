from pathlib import Path
import tempfile
import unittest

from dependency_policy import dependency_failures


_PACKAGE = '[package]\nname = "demo"\nversion = "0.1.0"\n'
_BURN = '{ version = "=0.21.0", default-features = false }'
_CUBECL = '{ version = "=0.10.0", default-features = false }'
_WGPU = '{ version = "=29.0.4", default-features = false }'


class DependencyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config = {"manifests": [{"path": "Cargo.toml"}], "dependency_policy": {}}

    def write(self, content, path="Cargo.toml", package="demo"):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_PACKAGE.replace('"demo"', f'"{package}"') + content, encoding="utf-8")

    def failures(self):
        return dependency_failures(self.root, self.config)

    def errors(self):
        return "\n".join(self.failures())

    def test_retired_platform_features_cannot_return_through_aliases_or_scopes(self):
        self.config["dependency_policy"]["unsupported_features"] = ["wasm", "webgpu", "vulkan"]
        for name in ("wasm", "webgpu", "vulkan"):
            with self.subTest(name=name):
                self.write(f'[features]\n{name} = []\n')
                self.assertIn(f"features.{name} is no longer supported", self.errors())
                self.write(f"""
[features]
metal = ["hidden"]
hidden = ["engine?/{name}"]
[dependencies]
engine = {{ package = "burn", version = "=0.21.0", default-features = false, optional = true }}
""")
                self.assertIn(f"unsupported feature engine?/{name}", self.errors())
                for scope in ("dependencies", "build-dependencies", "dev-dependencies",
                              'target.\'cfg(unix)\'.dependencies'):
                    self.write(f'[{scope}]\nengine = {{ package = "burn", version = "=0.21.0", '
                               f'default-features = false, features = ["{name}"] }}')
                    self.assertIn(f"unsupported feature {name}", self.errors())

    def test_retiring_webgpu_preserves_the_native_wgpu_runtime(self):
        self.config["dependency_policy"]["unsupported_features"] = ["wasm", "webgpu", "vulkan"]
        self.write(f"""
[features]
metal = ["burn/metal"]
[dependencies]
burn = {_BURN}
wgpu = {{ version = "=29.0.4", default-features = false, features = ["metal", "wgsl"] }}
""")
        self.assertEqual(self.failures(), [])

    def test_policy_is_opt_in_and_does_not_read_manifests_when_absent(self):
        self.assertEqual(dependency_failures(self.root, {}), [])
        self.assertEqual(dependency_failures(self.root, {"manifests": [{"path": "missing"}]}), [])
        self.config["dependency_policy"] = None
        self.assertIn("expected an object", self.errors())

    def test_valid_inference_training_and_dev_references(self):
        self.write("""
[features]
default = ["metal"]
cpu = ["engine/ndarray"]
ane = ["engine/ndarray"]
training = ["engine/autodiff"]
metal = ["gpu"]
webgpu = ["gpu"]
gpu = ["engine/metal"]
[dependencies]
engine = { package = "burn", version = "=0.21.0", default-features = false, features = ["std"] }
[dev-dependencies]
engine = { package = "burn", version = "=0.21.0", default-features = false, features = ["std", "ndarray", "autodiff"] }
""")
        self.assertEqual(self.failures(), [])

    def test_aliases_and_all_target_dependency_scopes(self):
        for scope in ("dependencies", "build-dependencies", "dev-dependencies"):
            for prefix in ("", 'target.\'cfg(target_os = "macos")\'.'):
                with self.subTest(scope=scope, prefix=prefix):
                    self.write(f"""
[{prefix}{scope}]
engine = {{ package = "burn-core", version = "=0.20.0", default-features = false }}
compute = {{ package = "cubecl-runtime", version = "=0.10.0", default-features = true }}
graphics = {{ package = "wgpu", version = "29.0.4" }}
""")
                    errors = self.errors()
                    self.assertIn(f"{scope}.engine (burn-core): version must be '=0.21.0'", errors)
                    self.assertIn(f"{scope}.compute (cubecl-runtime): default-features", errors)
                    self.assertIn(f"{scope}.graphics (wgpu): version must be '=29.0.4'", errors)
                    self.assertIn(f"{scope}.graphics (wgpu): default-features", errors)
                    if prefix:
                        self.assertIn('target[\'cfg(target_os = "macos")\']', errors)

    def test_exact_pins_and_explicit_default_features_are_required(self):
        for package, pin in (("burn", "=0.21.0"), ("burn-tensor", "=0.21.0"),
                             ("cubecl", "=0.10.0"), ("cubecl-common", "=0.10.0"),
                             ("wgpu", "=29.0.4")):
            for value in (f'"{pin}"', f'{{ version = "{pin}", default-features = true }}',
                          f'{{ version = "{pin[1:]}", default-features = false }}',
                          '{ default-features = false }'):
                with self.subTest(package=package, value=value):
                    self.write(f"[dependencies]\n{package} = {value}\n")
                    self.assertTrue(self.failures())
            self.write(f'[dependencies]\n{package} = {{ version = "{pin}", default-features = false }}')
            self.assertEqual(self.failures(), [])

    def test_multiple_configured_manifests_are_checked(self):
        self.write(f"[dependencies]\nburn = {_BURN}\n")
        path = "crates/other/Cargo.toml"
        self.write('[build-dependencies]\ngpu = { package = "wgpu", version = "28" }', path)
        self.config["manifests"].append({"path": path})
        self.assertTrue(all(error.startswith(path + ":") for error in self.failures()))
        self.assertIn("build-dependencies.gpu", self.errors())

    def test_unconditional_burn_backends_fail_in_production_and_build_scopes(self):
        for scope in ("dependencies", "build-dependencies",
                      'target.\'cfg(unix)\'.dependencies'):
            for name, gate in (("ndarray", "cpu or ane"), ("autodiff", "training"),
                               ("train", "training")):
                with self.subTest(scope=scope, name=name):
                    self.write(f"""
[features]
cpu = ["engine/ndarray"]
training = ["engine/autodiff"]
[{scope}]
engine = {{ package = "burn", version = "=0.21.0", default-features = false, features = ["{name}"] }}
""")
                    self.assertIn(f"{name} must be enabled through local feature {gate}", self.errors())
                    self.assertIn("not unconditionally", self.errors())

    def test_optional_burn_feature_lists_require_activation_by_the_right_gate(self):
        self.write("""
[features]
cpu = ["dep:engine"]
metal = []
[dependencies]
engine = { package = "burn", version = "=0.21.0", default-features = false, optional = true, features = ["ndarray"] }
""")
        self.assertEqual(self.failures(), [])
        content = (self.root / "Cargo.toml").read_text()
        (self.root / "Cargo.toml").write_text(content.replace('cpu = ["dep:engine"]', "cpu = []"))
        self.assertIn("local feature cpu or ane", self.errors())

    def test_sensitive_forwarding_requires_local_cpu_ane_or_training(self):
        for feature, gate in (("ndarray", "cpu or ane"), ("autodiff", "training"),
                              ("train", "training")):
            with self.subTest(feature=feature):
                self.write(f"""
[features]
helper = ["engine/{feature}"]
[dependencies]
engine = {{ package = "burn", version = "=0.21.0", default-features = false }}
""")
                self.assertIn(f"engine/{feature} must be enabled through local feature {gate}",
                              self.errors())

    def test_cpu_and_training_gates_may_use_intermediate_features_and_cycles(self):
        self.write(f"""
[features]
cpu = ["host"]
host = ["cycle", "burn/ndarray"]
cycle = ["host"]
training = ["gradients"]
gradients = ["burn/autodiff"]
metal = ["burn/metal"]
[dependencies]
burn = {_BURN}
""")
        self.assertEqual(self.failures(), [])

    def test_gpu_closure_catches_cpu_and_training_through_intermediate_features(self):
        for gpu in ("metal", "webgpu"):
            for gate, backend in (("cpu", "ndarray"), ("training", "autodiff")):
                with self.subTest(gpu=gpu, gate=gate):
                    self.write(f"""
[features]
{gpu} = ["middle"]
middle = ["cycle", "{gate}"]
cycle = ["middle"]
{gate} = ["engine/{backend}"]
[dependencies]
engine = {{ package = "burn", version = "=0.21.0", default-features = false }}
""")
                    self.assertIn(f"features.{gpu} reaches", self.errors())
                    self.assertIn(f"engine/{backend}", self.errors())

    def test_default_inference_rejects_a_target_specific_model_cpu_dependency(self):
        self.write("""
[target.'cfg(target_os = "macos")'.dependencies]
model = { path = "crates/model", default-features = false, features = ["cpu"] }
""")
        self.write('[features]\ndefault = ["metal"]\nmetal = []\ncpu = []\n',
                   "crates/model/Cargo.toml", package="omni-minicpm-v")
        self.assertIn("features.default reaches cpu via model/cpu", self.errors())

    def test_default_ane_fallback_does_not_exempt_another_model_cpu_feature(self):
        self.write(f"""
[features]
default = ["ane"]
ane = ["burn/ndarray"]
[dependencies]
burn = {_BURN}
model = {{ version = "1", default-features = false, features = ["cpu"] }}
""")
        self.assertIn("cpu via model/cpu", self.errors())
        self.assertNotIn("cpu via burn/ndarray", self.errors())
        path = self.root / "Cargo.toml"
        path.write_text(path.read_text().replace('features = ["cpu"]', 'features = []'))
        self.assertEqual(self.failures(), [])

    def test_default_training_exception_does_not_allow_cpu_training_by_default(self):
        self.config["dependency_policy"]["training_manifests"] = ["Cargo.toml"]
        self.write(f"""
[features]
default = ["training", "cpu"]
training = ["burn/autodiff"]
cpu = ["burn/ndarray"]
[dependencies]
burn = {_BURN}
""")
        self.assertIn("features.default reaches cpu", self.errors())
        self.assertNotIn("training via burn/autodiff", self.errors())

    def test_gpu_closure_catches_bypassing_the_named_cpu_gate(self):
        self.write(f"""
[features]
metal = ["middle"]
middle = ["burn/ndarray"]
cpu = ["burn/ndarray"]
[dependencies]
burn = {_BURN}
""")
        self.assertIn("features.metal reaches cpu via burn/ndarray", self.errors())

    def test_weak_feature_requests_are_inspected_conservatively(self):
        self.write("""
[features]
cpu = ["engine?/ndarray"]
metal = ["middle"]
middle = ["engine?/ndarray"]
[dependencies]
engine = { package = "burn", version = "=0.21.0", default-features = false, optional = true }
""")
        self.assertIn("features.metal reaches cpu via engine/ndarray", self.errors())

    def test_forwarded_dependency_cpu_and_training_features_are_checked(self):
        for name in ("cpu", "training"):
            with self.subTest(name=name):
                self.write(f"""
[features]
{name} = ["model/{name}"]
metal = ["middle"]
middle = ["model/{name}"]
[dependencies]
model = {{ version = "1", default-features = false }}
""")
                self.assertIn(f"features.metal reaches {name} via model/{name}", self.errors())

    def test_dev_reference_backends_are_allowed_including_target_scopes(self):
        for scope in ("dev-dependencies", 'target.\'cfg(unix)\'.dev-dependencies'):
            with self.subTest(scope=scope):
                self.write(f"""
[features]
default = ["metal"]
metal = ["burn/metal"]
[dependencies]
burn = {_BURN}
[{scope}]
burn = {{ version = "=0.21.0", default-features = false, features = ["ndarray", "autodiff"] }}
reference = {{ package = "burn-ndarray", version = "=0.21.0", default-features = false }}
gradient = {{ package = "burn-autodiff", version = "=0.21.0", default-features = false }}
""")
                self.assertEqual(self.failures(), [])

    def test_direct_autodiff_requires_optional_and_training_activation(self):
        for activation in ("dep:gradient", "gradient", "gradient/std"):
            with self.subTest(activation=activation):
                self.write(f"""
[features]
training = ["helper"]
helper = ["{activation}"]
[dependencies]
gradient = {{ package = "burn-autodiff", version = "=0.21.0", default-features = false, optional = true }}
""")
                self.assertEqual(self.failures(), [])
        content = (self.root / "Cargo.toml").read_text()
        (self.root / "Cargo.toml").write_text(content.replace("optional = true", "optional = false"))
        self.assertIn("burn-autodiff must be optional", self.errors())
        (self.root / "Cargo.toml").write_text(content.replace("gradient/std", "gradient?/std"))
        self.assertIn("must be activated by local feature training", self.errors())

    def test_gpu_cannot_activate_optional_autodiff_even_with_a_training_gate(self):
        self.write("""
[features]
training = ["dep:gradient"]
metal = ["helper"]
helper = ["dep:gradient"]
[dependencies]
gradient = { package = "burn-autodiff", version = "=0.21.0", default-features = false, optional = true }
""")
        self.assertIn("features.metal reaches training via dep:gradient (burn-autodiff)",
                      self.errors())

    def test_training_manifest_exception_only_applies_to_default_training(self):
        path = "crates/omni-rsi/Cargo.toml"
        self.config["manifests"] = [{"path": path}]
        self.write(f"""
[features]
default = ["metal", "helper"]
helper = ["training"]
training = ["burn/autodiff"]
metal = ["burn/metal"]
[dependencies]
burn = {_BURN}
""", path)
        self.assertIn("features.default reaches training", self.errors())
        self.config["dependency_policy"]["training_manifests"] = [path]
        self.assertEqual(self.failures(), [])
        content = (self.root / path).read_text()
        (self.root / path).write_text(content.replace('metal = ["burn/metal"]',
                                                     'metal = ["burn/metal", "helper"]'))
        self.assertIn("features.metal reaches training", self.errors())

    def test_default_inference_cannot_enable_cpu_or_bypass_training_name(self):
        self.write(f"""
[features]
default = ["cpu"]
cpu = ["burn/ndarray"]
training = ["burn/autodiff"]
[dependencies]
burn = {_BURN}
""")
        self.assertIn("features.default reaches cpu", self.errors())
        content = (self.root / "Cargo.toml").read_text()
        (self.root / "Cargo.toml").write_text(content.replace('default = ["cpu"]',
                                                            'default = ["burn/autodiff"]'))
        self.assertIn("features.default reaches training via burn/autodiff", self.errors())

    def test_cpu_fallback_needs_burn_ane_manifest_and_nonempty_reason(self):
        path = "crates/burn-ane/Cargo.toml"
        self.config["manifests"] = [{"path": path}]
        self.write(f"[dependencies]\nreference = {{ package = \"burn-ndarray\", "
                   'version = "=0.21.0", default-features = false }', path, package="burn-ane")
        self.assertIn("ndarray_fallbacks reason", self.errors())
        for reason in ("", "   ", None):
            with self.subTest(reason=reason):
                self.config["dependency_policy"]["ndarray_fallbacks"] = {path: reason}
                self.assertTrue(self.failures())
        self.config["dependency_policy"]["ndarray_fallbacks"] = {path: "ANE host fallback"}
        self.assertEqual(self.failures(), [])
        content = (self.root / path).read_text()
        (self.root / path).write_text(content.replace('name = "burn-ane"', 'name = "other"'))
        self.assertIn("production burn-ndarray", self.errors())

    def test_optional_ndarray_still_requires_the_cpu_fallback_exception(self):
        self.write("""
[features]
cpu = ["dep:reference"]
[dependencies]
reference = { package = "burn-ndarray", version = "=0.21.0", default-features = false, optional = true }
""")
        self.assertIn("production burn-ndarray", self.errors())

    def test_first_party_paths_disable_declared_defaults_in_every_scope(self):
        target = "crates/model/Cargo.toml"
        for default in ('["training"]', "[]"):
            self.write(f"[features]\ndefault = {default}\ntraining = []\n", target)
            for scope in ("dependencies", "build-dependencies", "dev-dependencies",
                          'target.\'cfg(unix)\'.dependencies',
                          'target.\'cfg(unix)\'.build-dependencies',
                          'target.\'cfg(unix)\'.dev-dependencies'):
                with self.subTest(default=default, scope=scope):
                    self.write(f'[{scope}]\nalias = {{ package = "demo", path = "crates/model" }}')
                    self.assertIn(f"{scope.split('.')[-1]}.alias", self.errors())
                    self.assertIn("first-party path crates/model/Cargo.toml declares default",
                                  self.errors())
                    self.write(f'[{scope}]\nalias = {{ package = "demo", path = "crates/model", '
                               'default-features = false }')
                    self.assertEqual(self.failures(), [])

    def test_path_resolution_is_relative_to_each_manifest(self):
        self.write("[features]\ndefault = []", "crates/model/Cargo.toml")
        path = "crates/consumer/Cargo.toml"
        self.config["manifests"] = [{"path": path}]
        self.write('[dependencies]\nmodel = { path = "../model" }', path)
        self.assertIn("first-party path crates/model/Cargo.toml", self.errors())

    def test_unconditional_dependency_features_reach_gpu_and_implicit_default(self):
        for feature in ("cpu", "training"):
            with self.subTest(feature=feature):
                self.write(f"""
[features]
metal = []
[dependencies]
model = {{ version = "1", default-features = false, features = ["{feature}"] }}
""")
                self.assertIn(f"features.metal reaches {feature} via model/{feature}", self.errors())
                if feature == "training":
                    self.assertIn("features.default reaches training via model/training", self.errors())

    def test_paths_without_defaults_and_vendor_paths_are_not_first_party_default_leaks(self):
        self.write("", "crates/model/Cargo.toml")
        self.write("[features]\ndefault = []", "vendor/external/Cargo.toml")
        self.write("""
[dependencies]
model = { path = "crates/model" }
external = { path = "vendor/external" }
""")
        self.assertEqual(self.failures(), [])

    def test_path_dependency_ecosystem_pins_and_defaults_are_still_checked(self):
        self.write("", "crates/burn-ane/Cargo.toml", package="burn-ane")
        self.write('[dependencies]\nbackend = { package = "burn-ane", path = "crates/burn-ane" }')
        self.assertIn("version must be '=0.21.0'", self.errors())
        self.assertIn("default-features must be false", self.errors())

    def test_patch_tables_are_not_direct_dependencies(self):
        self.write(f"""
[dependencies]
burn = {_BURN}
cubecl = {_CUBECL}
wgpu = {_WGPU}
[patch.crates-io]
burn = {{ path = "vendor/burn" }}
""")
        self.assertEqual(self.failures(), [])

    def test_missing_and_invalid_manifests_are_reported(self):
        self.assertIn("Cargo.toml: cannot read Cargo manifest", self.errors())
        (self.root / "Cargo.toml").write_text("[invalid")
        self.assertIn("cannot read Cargo manifest", self.errors())
        self.write('[dependencies]\nmodel = { path = "crates/missing" }')
        self.assertIn("crates/missing/Cargo.toml: cannot read Cargo manifest", self.errors())

    def test_workspace_inheritance_resolves_against_root_workspace_dependencies(self):
        self.write('[dependencies]\nengine = { workspace = true }')
        self.assertIn("not found in root [workspace.dependencies]", self.errors())
        self.write(f'[workspace.dependencies]\nburn = {_BURN}\n')
        self.config["manifests"] = [{"path": "Cargo.toml"}, {"path": "crates/member/Cargo.toml"}]
        self.write('[dependencies]\nburn = { workspace = true }\n', "crates/member/Cargo.toml", "member")
        self.assertEqual(self.failures(), [])

    def test_diagnostics_are_deterministic(self):
        self.write("""
[features]
metal = ["cpu", "training"]
cpu = ["burn/ndarray"]
training = ["burn/autodiff"]
[dependencies]
burn = { version = "0.21.0" }
""")
        first = self.failures()
        self.assertEqual(first, self.failures())
        self.assertTrue(first)


if __name__ == "__main__":
    unittest.main()
