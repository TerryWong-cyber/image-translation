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
        self.assertFalse(settings.ocr.require_local_models)
        self.assertIsNone(settings.ocr.detection_model_dir)
        self.assertEqual(
            settings.llm.translation_url,
            "http://127.0.0.1:5051/api/inference/qwen_translate",
        )
        self.assertEqual(
            settings.oss.file_url("source", "nested/image.jpg"),
            "http://127.0.0.1:5000/file/source/nested/image.jpg",
        )
        self.assertTrue(settings.prompts.translate_zh_en.endswith("\n"))
        self.assertIn("English", settings.prompts.translate_any_en)
        self.assertIn("简体中文", settings.prompts.translate_any_zh_cn)
        self.assertIn("繁體中文", settings.prompts.translate_any_zh_tw)
        self.assertIn("한국어", settings.prompts.translate_any_ko)
        self.assertIn("日本語", settings.prompts.translate_any_ja)
        self.assertIn("ภาษาไทย", settings.prompts.translate_any_th)
        self.assertIn("French", settings.prompts.translate_any_fr)
        self.assertIn("Arabic", settings.prompts.translate_any_ar)
        self.assertIn("German", settings.prompts.translate_any_de)
        self.assertIn("Russian", settings.prompts.translate_any_ru)
        self.assertIn("Dutch", settings.prompts.translate_any_nl)
        self.assertIn("Portuguese", settings.prompts.translate_any_pt)
        self.assertIn("Spanish", settings.prompts.translate_any_es)
        self.assertIn("Italian", settings.prompts.translate_any_it)
        self.assertIn("Vietnamese", settings.prompts.translate_any_vi)
        self.assertIn("Indonesian", settings.prompts.translate_any_id)
        self.assertEqual(settings.paths.font_zh_file.name, "NotoSansSC-Regular.ttf")
        self.assertEqual(settings.paths.font_zh_cn_file.name, "NotoSansSC-Regular.ttf")
        self.assertEqual(settings.paths.font_zh_tw_file.name, "NotoSansTC-Regular.ttf")
        self.assertEqual(settings.paths.font_latin_file.name, "NotoSans-Regular.ttf")
        self.assertEqual(settings.paths.font_ja_file.name, "NotoSansJP-Regular.ttf")
        self.assertEqual(settings.paths.font_ko_file.name, "NotoSansKR-Regular.ttf")
        self.assertEqual(settings.paths.font_arabic_file.name, "NotoSansArabic-Regular.ttf")
        self.assertEqual(settings.paths.font_cyrillic_file.name, "NotoSansCyrillic-Regular.ttf")
        self.assertEqual(settings.paths.font_thai_file.name, "NotoSansThai-Regular.ttf")
        self.assertEqual(
            settings.paths.font_vietnamese_file.name,
            "NotoSansVietnamese-Regular.ttf",
        )
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

    def test_local_ocr_model_directories_are_validated(self):
        with tempfile.TemporaryDirectory() as model_root:
            root = Path(model_root)
            orientation = root / "orientation"
            detection = root / "detection"
            recognition = root / "recognition"
            for path in (orientation, detection, recognition):
                path.mkdir()
            env = {
                "IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV),
                "OCR_REQUIRE_LOCAL_MODELS": "true",
                "OCR_TEXTLINE_MODEL_DIR": str(orientation),
                "OCR_DETECTION_MODEL_DIR": str(detection),
                "OCR_RECOGNITION_MODEL_DIR": str(recognition),
            }
            with patch.dict(os.environ, env, clear=True):
                get_settings.cache_clear()
                settings = get_settings()

        self.assertTrue(settings.ocr.require_local_models)
        self.assertEqual(settings.ocr.detection_model_dir, detection)
        self.assertEqual(settings.ocr.recognition_model_dir, recognition)

    def test_required_local_ocr_models_fail_fast_when_missing(self):
        env = {
            "IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV),
            "OCR_REQUIRE_LOCAL_MODELS": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ConfigurationError, "OCR_DETECTION_MODEL_DIR"):
                get_settings()

    def test_detection_and_recognition_model_dirs_are_atomic(self):
        with tempfile.TemporaryDirectory() as model_root:
            env = {
                "IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV),
                "OCR_DETECTION_MODEL_DIR": model_root,
            }
            with patch.dict(os.environ, env, clear=True):
                get_settings.cache_clear()
                with self.assertRaisesRegex(ConfigurationError, "configured together"):
                    get_settings()

    def test_optional_runtime_defaults_keep_existing_env_files_compatible(self):
        legacy_content = "\n".join(
            line
            for line in EXAMPLE_ENV.read_text(encoding="utf-8").splitlines()
            if not line.startswith("MCP_")
            and not line.startswith(
                (
                    "OCR_DEVICE=",
                    "OCR_GPU_ID=",
                    "OCR_VERSION=",
                    "OCR_REQUIRE_LOCAL_MODELS=",
                    "OCR_TEXTLINE_MODEL_DIR=",
                    "OCR_DETECTION_MODEL_DIR=",
                    "OCR_RECOGNITION_MODEL_DIR=",
                    "FONT_ZH_FILE=",
                    "FONT_ZH_CN_FILE=",
                    "FONT_ZH_TW_FILE=",
                    "FONT_LATIN_FILE=",
                    "FONT_JA_FILE=",
                    "FONT_KO_FILE=",
                    "FONT_ARABIC_FILE=",
                    "FONT_CYRILLIC_FILE=",
                    "FONT_THAI_FILE=",
                    "FONT_VIETNAMESE_FILE=",
                )
            )
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
        self.assertEqual(settings.paths.font_latin_file, settings.paths.font_file)
        self.assertEqual(settings.paths.font_arabic_file, settings.paths.font_file)
        self.assertEqual(settings.paths.font_zh_cn_file, settings.paths.font_file)
        self.assertEqual(settings.paths.font_zh_tw_file, settings.paths.font_file)

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
