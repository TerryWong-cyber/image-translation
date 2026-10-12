import importlib.util
import unittest
from typing import get_args


PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None

if PYDANTIC_AVAILABLE:
    from pydantic import ValidationError

    from image_translation.contracts import (
        UrlTranslationRequest,
        TranslationItem,
        TranslationLanguage,
        TranslationResult,
    )
    from image_translation.utils.translation_prompts import AUTO_TRANSLATION_TARGETS


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic is not installed")
class TranslationContractsTest(unittest.TestCase):
    def test_contract_contains_every_configured_auto_detect_target(self):
        expected = {"en_zh", "zh_en"} | {
            f"any_{target_code}" for target_code in AUTO_TRANSLATION_TARGETS
        }
        self.assertEqual(set(get_args(TranslationLanguage)), expected)

    def test_request_accepts_complete_url_and_supported_language(self):
        command = UrlTranslationRequest(
            image_url="https://images.example.com/manuals/page-1.png",
            language="zh_en",
        )

        self.assertEqual(str(command.image_url), "https://images.example.com/manuals/page-1.png")

    def test_command_accepts_auto_detect_target_languages(self):
        for target_code in AUTO_TRANSLATION_TARGETS:
            language = f"any_{target_code}"
            with self.subTest(language=language):
                command = UrlTranslationRequest(
                    image_url="https://images.example.com/manuals/page-1.png",
                    language=language,
                )
                self.assertEqual(command.language, language)

    def test_command_rejects_unknown_auto_detect_target_language(self):
        with self.assertRaises(ValidationError):
            UrlTranslationRequest(
                image_url="https://images.example.com/manuals/page-1.png",
                language="any_xx",
            )

    def test_request_rejects_non_url_and_storage_fields(self):
        for value in ("../secret.png", "images/a.png", "file:///tmp/a.png", "http://user:pass@example.com/a.png", "https://example.com/a.png#fragment"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                UrlTranslationRequest(image_url=value)
        with self.assertRaises(ValidationError):
            UrlTranslationRequest(image_url="https://example.com/a.png", bucket="images")

    def test_result_contains_urls_request_id_and_duration(self):
        result = TranslationResult(
            request_id="request-1",
            source_url="https://images.example.com/input.png",
            translated_url="https://files.example.com/translations/result.png",
            data=[
                TranslationItem(
                    coordinate=[[0, 0], [10, 0], [10, 10], [0, 10]],
                    source_text="Hello",
                    translated_text="你好",
                )
            ],
            duration_ms=10,
        )

        payload = result.model_dump(mode="json")
        self.assertEqual(
            set(payload),
            {"status", "request_id", "source_url", "translated_url", "data", "duration_ms"},
        )
        self.assertEqual(payload["request_id"], "request-1")


if __name__ == "__main__":
    unittest.main()
