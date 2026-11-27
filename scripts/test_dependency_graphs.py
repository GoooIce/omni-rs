from pathlib import Path
import unittest

from check_dependency_graphs import graph_failures, parse_tree, portable_graph_failures


class DependencyGraphTests(unittest.TestCase):
    def test_tree_keeps_source_and_unifies_repeated_features(self):
        rows = parse_tree("burn v0.21.0 (/repo/vendor/burn)|std\n"
                          "burn v0.21.0 (/repo/vendor/burn)|metal,std (*)\n"
                          "serde_json v1.0.154|default,std\n")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], "/repo/vendor/burn")
        self.assertEqual(rows[0]["features"], ["metal", "std"])
        with self.assertRaises(ValueError):
            parse_tree("unexpected output")

    def test_registry_patch_and_version_drift_are_rejected(self):
        policy = {"vendored_paths": {"burn": "vendor/burn"},
                  "aligned_versions": {"serde_json": "1.0.154"}}
        rows = parse_tree("burn v0.21.0|std\nserde_json v1.0.151|std\n")
        errors = graph_failures(Path("/repo"), rows, policy)
        self.assertTrue(any("the registry" in error for error in errors))
        self.assertTrue(any("1.0.151" in error for error in errors))
        rows[0]["source"] = "/repo/vendor/burn"
        rows[1]["version"] = "1.0.154"
        self.assertEqual(graph_failures(Path("/repo"), rows, policy), [])

    def test_inference_training_and_ane_fallback_are_distinct(self):
        root = Path("/repo")
        rows = parse_tree("burn-autodiff v0.21.0|std\n")
        self.assertTrue(graph_failures(root, rows, {}))
        self.assertEqual(graph_failures(root, rows, {}, training=True), [])
        rows += parse_tree("burn-ndarray v0.21.0|std\n")
        self.assertTrue(graph_failures(root, rows, {}, training=True))
        rows += parse_tree("burn-ane v0.21.0 (/repo/crates/burn-ane)|\n")
        policy = {"ndarray_fallbacks": {"crates/burn-ane/Cargo.toml": "supported host fallback"}}
        self.assertEqual(graph_failures(root, rows, policy), [])

    def test_macos_rejects_vulkan_and_retains_metal_runtime_wgsl(self):
        rows = parse_tree("ash v0.38.0+1.3.281|std\nwgpu v29.0.4|std,metal,wgsl\n")
        errors = graph_failures(Path("/repo"), rows, {})
        self.assertTrue(any("Vulkan" in error for error in errors))
        self.assertEqual(graph_failures(Path("/repo"), rows, {}, macos=False), [])
        self.assertEqual(graph_failures(Path("/repo"), rows[1:], {}), [])
        rows[1]["features"].remove("metal")
        self.assertTrue(any("missing" in error
                            for error in graph_failures(Path("/repo"), rows, {})))

    def test_native_graph_rejects_browser_and_other_backend_features(self):
        for feature in ("webgpu", "webgl", "vulkan", "dx12", "gles"):
            with self.subTest(feature=feature):
                rows = parse_tree(f"wgpu v29.0.4|std,metal,wgsl,{feature}\n")
                self.assertTrue(any("unsupported" in error
                                    for error in graph_failures(Path("/repo"), rows, {})))
        rows = parse_tree("omni-r2t2 v0.1.0 (/repo)|metal,webgpu\n")
        self.assertTrue(any("unsupported webgpu" in error for error in graph_failures(
            Path("/repo"), rows, {"unsupported_features": ["webgpu"]})))

    def test_git_sources_are_not_misclassified_as_first_party_paths(self):
        rows = parse_tree(
            "makepad-platform v2.0.0 (https://github.com/makepad/makepad#711521d)|vulkan\n")
        self.assertEqual(graph_failures(
            Path.cwd(), rows, {"unsupported_features": ["vulkan"]}), [])

    def test_supported_ane_cannot_exempt_another_models_cpu_feature(self):
        rows = parse_tree("burn-autodiff v0.21.0|std\nburn-ndarray v0.21.0|std\n"
                          "burn-ane v0.21.0 (/repo/crates/burn-ane)|metal-shared\n"
                          "omni-minicpm-v v0.1.0 (/repo/crates/minicpm-v)|metal,cpu\n")
        policy = {"ndarray_fallbacks": {"crates/burn-ane/Cargo.toml": "supported host fallback"}}
        for feature in ("cpu", "training"):
            next(row for row in rows if row["name"] == "omni-minicpm-v")["features"] = ["metal", feature]
            self.assertTrue(any("omni-minicpm-v" in error and feature in error
                                for error in graph_failures(Path("/repo"), rows, policy, training=True)))

    def test_ane_fallback_requires_a_configured_local_source(self):
        rows = parse_tree("burn-autodiff v0.21.0|std\nburn-ndarray v0.21.0|std\n"
                          "burn-ane v0.21.0 (/repo/crates/burn-ane)|\n")
        policy = {"ndarray_fallbacks": {"crates/burn-ane/Cargo.toml": "supported host fallback"}}
        self.assertTrue(graph_failures(Path("/repo"), rows, {}))
        self.assertEqual(graph_failures(Path("/repo"), rows, policy), [])
        next(row for row in rows if row["name"] == "burn-ane")["source"] = None
        self.assertTrue(graph_failures(Path("/repo"), rows, policy))

    def test_portable_cpu_graph_rejects_apple_and_gpu_packages_and_features(self):
        root = Path("/repo")
        policy = {
            "apple_only_packages": ["burn-ane", "objc2-metal", "mtl4", "cubecl-cpp"],
            "apple_only_features": ["metal", "ane", "coreml", "metal-shared"],
        }
        clean = parse_tree("omni-qwen-image v0.1.0 (/repo/crates/qwen-image)|cpu\n"
                           "burn-ndarray v0.21.0|std\n")
        self.assertEqual(portable_graph_failures(
            root, clean, policy, mode="cpu", target="aarch64-apple-darwin"), [])
        for bad in ("burn-ane v0.21.0 (/repo/crates/burn-ane)|\n",
                    "objc2-metal v0.3.2|std\n",
                    "mtl4 v0.1.0 (/repo/crates/mtl4)|\n",
                    "wgpu v29.0.4|std,wgsl\n"):
            with self.subTest(bad=bad):
                errors = portable_graph_failures(
                    root, clean + parse_tree(bad), policy, mode="cpu", target="aarch64-apple-darwin")
                self.assertTrue(errors)
        leaked_feature = parse_tree("omni-qwen-image v0.1.0 (/repo/crates/qwen-image)|cpu,metal\n")
        self.assertTrue(any("Apple feature metal" in err for err in portable_graph_failures(
            root, leaked_feature, policy, mode="cpu", target="aarch64-apple-darwin")))

    def test_portable_wgpu_graph_enforces_non_apple_and_host_boundaries(self):
        root = Path("/repo")
        policy = {
            "apple_only_packages": ["burn-ane", "objc2-metal", "mtl4", "cubecl-cpp"],
            "apple_only_features": ["metal", "ane", "coreml", "metal-shared"],
        }
        clean = parse_tree("omni-kernel v0.1.0 (/repo/crates/omni-kernel)|wgpu\n"
                           "wgpu v29.0.4|std,wgsl\n")
        self.assertEqual(portable_graph_failures(
            root, clean, policy, mode="wgpu", target="x86_64-unknown-linux-gnu"), [])
        # Non-Apple targets must reject objc2-metal completely.
        with_objc2 = clean + parse_tree("objc2-metal v0.3.2|std\n")
        self.assertTrue(any("objc2-metal" in err for err in portable_graph_failures(
            root, with_objc2, policy, mode="wgpu", target="x86_64-unknown-linux-gnu")))
        # On macOS hosts wgpu-hal pulls objc2-metal, but burn-ane / mtl4 / cubecl-cpp remain forbidden.
        self.assertEqual(portable_graph_failures(
            root, with_objc2, policy, mode="wgpu", target="aarch64-apple-darwin"), [])
        for bad in ("burn-ane v0.21.0 (/repo/crates/burn-ane)|\n",
                    "cubecl-cpp v0.10.0 (/repo/vendor/cubecl-cpp)|metal\n",
                    "burn-ndarray v0.21.0|std\n"):
            with self.subTest(bad=bad):
                errors = portable_graph_failures(
                    root, clean + parse_tree(bad), policy, mode="wgpu", target="aarch64-apple-darwin")
                self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
