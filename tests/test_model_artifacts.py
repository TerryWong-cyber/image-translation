import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prefetch_paddle_models import download_models, main
from scripts.verify_paddle_models import REQUIRED_FILES, build_manifest


class ModelArtifactTest(unittest.TestCase):
    def test_prefetch_downloads_exact_runtime_models_into_flat_directories(self):
        downloads = []

        def snapshot(**kwargs):
            downloads.append(kwargs)
            destination = Path(kwargs["local_dir"])
            for name in REQUIRED_FILES:
                (destination / name).write_bytes(b"model artifact")
            return str(destination)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            download_models(root, True, snapshot)
            names = ["PP-LCNet_x1_0_textline_ori", "PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec"]
            self.assertEqual([call["model_id"] for call in downloads], [f"PaddlePaddle/{name}" for name in names])
            self.assertEqual([call["local_dir"] for call in downloads], [str(root.resolve() / "official_models" / name) for name in names])
            build_manifest(root, [f"official_models/{name}" for name in names])

    def test_prefetch_rejects_missing_and_empty_downloads(self):
        for empty in (False, True):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as temp_dir:
                def snapshot(**kwargs):
                    if empty:
                        for name in REQUIRED_FILES:
                            (Path(kwargs["local_dir"]) / name).write_bytes(b"")
                    return kwargs["local_dir"]
                with self.assertRaisesRegex(RuntimeError, "Incomplete model PP-OCRv4_mobile_det"):
                    download_models(Path(temp_dir), False, snapshot)

    def test_prefetch_rejects_unsupported_configuration_before_download(self):
        with patch("sys.argv", ["prefetch", "--cache-home", "/tmp/models", "--language", "zh"]):
            with self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, 2)

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
