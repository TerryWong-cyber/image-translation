import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from image_translation.components.ocr.device import (
    _local_model_name,
    paddleocr_device,
    paddleocr_init_kwargs,
)
from image_translation.config import OcrDevice, OcrSettings


def _settings(device: OcrDevice, gpu_id: int = 0) -> OcrSettings:
    return OcrSettings(
        device=device,
        gpu_id=gpu_id,
        language="en",
        version="PP-OCRv4",
        use_angle_classifier=True,
        unclip_ratio=1.5,
        task_timeout_seconds=180,
        worker_shutdown_timeout_seconds=5,
        box_height_tolerance_ratio=0.7,
        require_local_models=False,
        textline_orientation_model_dir=None,
        detection_model_dir=None,
        recognition_model_dir=None,
    )


def _paddle(*, cuda: bool, gpu_count: int):
    return SimpleNamespace(
        device=SimpleNamespace(
            is_compiled_with_cuda=lambda: cuda,
            cuda=SimpleNamespace(device_count=lambda: gpu_count),
        )
    )


class PaddleOcrDeviceTest(unittest.TestCase):
    def test_cpu_selects_cpu_without_cuda_probe(self):
        paddle = SimpleNamespace(device=None)
        self.assertEqual(paddleocr_device(_settings("cpu"), paddle), "cpu")

    def test_gpu_selects_configured_device(self):
        self.assertEqual(
            paddleocr_device(_settings("gpu", 1), _paddle(cuda=True, gpu_count=2)),
            "gpu:1",
        )

    def test_init_kwargs_use_paddleocr_3_names(self):
        kwargs = paddleocr_init_kwargs(
            _settings("gpu", 1),
            _paddle(cuda=True, gpu_count=2),
        )

        self.assertEqual(kwargs["device"], "gpu:1")
        self.assertEqual(kwargs["ocr_version"], "PP-OCRv4")
        self.assertEqual(kwargs["text_det_unclip_ratio"], 1.5)
        self.assertTrue(kwargs["use_textline_orientation"])
        self.assertNotIn("use_gpu", kwargs)
        self.assertNotIn("gpu_id", kwargs)
        self.assertNotIn("det_db_unclip_ratio", kwargs)

    def test_init_kwargs_use_explicit_local_model_directories(self):
        settings = _settings("cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = {}
            for role, name in (
                ("textline_orientation", "PP-LCNet_x1_0_textline_ori"),
                ("detection", "PP-OCRv4_mobile_det"),
                ("recognition", "en_PP-OCRv4_mobile_rec"),
            ):
                model_dir = Path(temp_dir) / role
                model_dir.mkdir()
                (model_dir / "inference.yml").write_text(
                    f"Global:\n  model_name: {name}\n", encoding="utf-8"
                )
                paths[f"{role}_model_dir"] = model_dir
            settings = OcrSettings(**{**settings.__dict__, **paths})
            kwargs = paddleocr_init_kwargs(settings, SimpleNamespace(device=None))

        self.assertEqual(kwargs["text_detection_model_dir"], str(paths["detection_model_dir"]))
        self.assertEqual(kwargs["text_recognition_model_dir"], str(paths["recognition_model_dir"]))
        self.assertEqual(
            kwargs["textline_orientation_model_dir"],
            str(paths["textline_orientation_model_dir"]),
        )
        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv4_mobile_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "en_PP-OCRv4_mobile_rec")
        self.assertEqual(kwargs["textline_orientation_model_name"], "PP-LCNet_x1_0_textline_ori")
        self.assertNotIn("lang", kwargs)
        self.assertNotIn("ocr_version", kwargs)

    def test_local_model_name_rejects_missing_or_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            with self.assertRaisesRegex(RuntimeError, "Cannot read Paddle model name"):
                _local_model_name(model_dir)
            for metadata in (
                "{}",
                "Global: null",
                "Global: [",
                "Global:\n  model_name: null",
                "Global:\n  model_name: ''",
            ):
                with self.subTest(metadata=metadata):
                    (model_dir / "inference.yml").write_text(metadata, encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        _local_model_name(model_dir)

    def test_gpu_requires_cuda_runtime(self):
        with self.assertRaisesRegex(RuntimeError, "not compiled with CUDA"):
            paddleocr_device(_settings("gpu"), _paddle(cuda=False, gpu_count=0))

    def test_gpu_id_must_exist(self):
        with self.assertRaisesRegex(RuntimeError, "OCR_GPU_ID=2 is unavailable"):
            paddleocr_device(_settings("gpu", 2), _paddle(cuda=True, gpu_count=2))


if __name__ == "__main__":
    unittest.main()
