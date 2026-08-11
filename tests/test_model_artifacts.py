import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.verify_paddle_models import REQUIRED_FILES, build_manifest


class ModelArtifactTest(unittest.TestCase):
    def test_builds_hash_manifest_for_complete_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "official_models" / "example"
            model_dir.mkdir(parents=True)
            for name in REQUIRED_FILES:
                (model_dir / name).write_bytes(name.encode("utf-8"))

            manifest = build_manifest(root, ["official_models/example"])

        files = manifest["models"]["official_models/example"]
        expected = hashlib.sha256(b"inference.json").hexdigest()
        self.assertEqual(
            files["official_models/example/inference.json"]["sha256"],
            expected,
        )

    def test_rejects_incomplete_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model").mkdir()

            with self.assertRaisesRegex(ValueError, "is missing"):
                build_manifest(root, ["model"])

    def test_rejects_missing_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Missing Paddle model directory"):
                build_manifest(Path(temp_dir), ["missing"])


if __name__ == "__main__":
    unittest.main()
