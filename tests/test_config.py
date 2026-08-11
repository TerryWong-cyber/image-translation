import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_translation.config import ConfigurationError, get_settings


EXAMPLE_ENV = Path(__file__).resolve().parents[1] / ".env.example"


class SettingsTest(unittest.TestCase):
    def tearDown(self):
        get_settings.cache_clear()

    def test_loads_typed_settings_and_builds_urls(self):
        with patch.dict(os.environ, {"IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV)}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.server.port, 8000)
        self.assertTrue(settings.server.reload)
        self.assertTrue(settings.mcp.enabled)
        self.assertEqual(settings.mcp.path, "/mcp")
        self.assertTrue(settings.mcp.stateless_http)
        self.assertEqual(settings.mcp.max_concurrency, 2)
        self.assertIn("localhost", settings.mcp.allowed_hosts)
        self.assertIn("localhost:*", settings.mcp.allowed_hosts)
        self.assertEqual(settings.ocr.device, "cpu")
        self.assertEqual(settings.ocr.gpu_id, 0)
        self.assertEqual(settings.ocr.version, "PP-OCRv4")
        self.assertEqual(
            settings.llm.translation_url,
            "http://127.0.0.1:5051/api/inference/qwen_translate",
        )
        self.assertEqual(
            settings.oss.file_url("source", "nested/image.jpg"),
            "http://127.0.0.1:5000/file/source/nested/image.jpg",
        )
        self.assertTrue(settings.prompts.translate_zh_en.endswith("\n"))
        self.assertEqual(settings.prompts.vision_file.name, "vision.json")
        self.assertEqual(settings.prompts.translations_file.name, "translations.json")
        self.assertIn("mm", settings.text_translation.no_translate_terms)
        self.assertIn("pcb", settings.text_translation.no_translate_terms)

    def test_process_environment_overrides_dotenv(self):
        env = {
            "IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV),
            "SERVER_PORT": "9123",
            "SERVER_RELOAD": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.server.port, 9123)
        self.assertFalse(settings.server.reload)

    def test_ocr_device_can_be_selected_from_environment(self):
        env = {
            "IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV),
            "OCR_DEVICE": "GPU",
            "OCR_GPU_ID": "2",
        }
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.ocr.device, "gpu")
        self.assertEqual(settings.ocr.gpu_id, 2)

    def test_invalid_ocr_device_fails_fast(self):
        env = {
            "IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV),
            "OCR_DEVICE": "auto",
        }
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ConfigurationError, "OCR_DEVICE"):
                get_settings()

    def test_invalid_ocr_version_fails_fast(self):
        env = {
            "IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV),
            "OCR_VERSION": "PP-OCRv2",
        }
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ConfigurationError, "OCR_VERSION"):
                get_settings()

    def test_optional_runtime_defaults_keep_existing_env_files_compatible(self):
        legacy_content = "\n".join(
            line
            for line in EXAMPLE_ENV.read_text(encoding="utf-8").splitlines()
            if not line.startswith("MCP_")
            and not line.startswith(("OCR_DEVICE=", "OCR_GPU_ID=", "OCR_VERSION="))
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as env_file:
            env_file.write(legacy_content)
            env_file.flush()
            with patch.dict(
                os.environ,
                {"IMAGE_TRANSLATION_ENV_FILE": env_file.name},
                clear=True,
            ):
                get_settings.cache_clear()
                settings = get_settings()

        self.assertTrue(settings.mcp.enabled)
        self.assertEqual(settings.mcp.path, "/mcp")
        self.assertEqual(settings.mcp.max_request_body_size, 1024 * 1024)
        self.assertEqual(settings.ocr.device, "cpu")
        self.assertEqual(settings.ocr.gpu_id, 0)
        self.assertEqual(settings.ocr.version, "PP-OCRv4")

    def test_missing_required_setting_fails_fast(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as env_file:
            env_file.write("SERVER_HOST=127.0.0.1\n")
            env_file.write("NO_TRANSLATE_TERMS_FILE=configs/no_translate_terms.json\n")
            env_file.flush()
            with patch.dict(
                os.environ,
                {"IMAGE_TRANSLATION_ENV_FILE": env_file.name},
                clear=True,
            ):
                get_settings.cache_clear()
                with self.assertRaisesRegex(ConfigurationError, "SERVER_APP"):
                    get_settings()


if __name__ == "__main__":
    unittest.main()
