import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


BRIDGE_PATH = Path(__file__).resolve().parents[1] / "agh" / "bridge.py"
SPEC = importlib.util.spec_from_file_location("agh_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class BridgeTests(unittest.TestCase):
    def test_profile_catalog_contains_allowlisted_profile(self):
        payload = bridge.list_profiles()
        self.assertTrue(payload["ok"])
        self.assertIn("aneurism-v4", [item["id"] for item in payload["profiles"]])

    def test_profile_id_rejects_path_traversal(self):
        with self.assertRaises(bridge.BridgeError):
            bridge.load_profile("../secret")

    def test_run_id_rejects_path_traversal(self):
        with self.assertRaises(bridge.BridgeError):
            bridge.run_dir_for("../../escape")

    def test_failure_classifier(self):
        self.assertEqual(bridge.classify_failure("CUDA out of memory"), "CUDA_OUT_OF_MEMORY")
        self.assertEqual(bridge.classify_failure("ModuleNotFoundError: skimage"), "MISSING_DEPENDENCY")
        self.assertEqual(bridge.classify_failure("FileNotFoundError"), "MISSING_FILE")

    def test_verification_rejects_incomplete_run(self):
        original = bridge.RUNS_DIR
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            bridge.RUNS_DIR = Path(temporary)
            run_id = "20261012T000000Z-deadbeef"
            run_dir = bridge.RUNS_DIR / run_id
            run_dir.mkdir()
            (run_dir / "status.json").write_text(
                json.dumps({"run_id": run_id, "state": "failed"}), encoding="utf-8"
            )
            result = bridge.build_verification(run_dir)
            self.assertFalse(result["verified"])
            self.assertFalse(result["checks"]["pipeline_completed"])
        bridge.RUNS_DIR = original


if __name__ == "__main__":
    unittest.main()
