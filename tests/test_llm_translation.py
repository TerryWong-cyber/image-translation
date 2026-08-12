import unittest
from types import SimpleNamespace

from image_translation.utils.translation_prompts import (
    AUTO_TRANSLATION_TARGETS,
    translation_prompt,
    translation_target,
)


class TranslationPromptTest(unittest.TestCase):
    def test_resolves_every_supported_auto_detect_target(self):
        prompts = SimpleNamespace(
            **{
                f"translate_any_{code}": f"prompt-{code}"
                for code in AUTO_TRANSLATION_TARGETS
            }
        )

        for target_code in AUTO_TRANSLATION_TARGETS:
            with self.subTest(target_code=target_code):
                self.assertEqual(
                    translation_prompt(prompts, f"any_{target_code}"),
                    f"prompt-{target_code}",
                )

    def test_maps_each_target_to_the_expected_font_setting(self):
        expected_font_settings = {
            "zh": "font_zh_cn_file",
            "zh_cn": "font_zh_cn_file",
            "zh_tw": "font_zh_tw_file",
            "en": "font_latin_file",
            "ja": "font_ja_file",
            "ko": "font_ko_file",
            "fr": "font_latin_file",
            "ar": "font_arabic_file",
            "de": "font_latin_file",
            "ru": "font_cyrillic_file",
            "nl": "font_latin_file",
            "pt": "font_latin_file",
            "th": "font_thai_file",
            "es": "font_latin_file",
            "it": "font_latin_file",
            "vi": "font_vietnamese_file",
            "id": "font_latin_file",
        }

        self.assertEqual(set(AUTO_TRANSLATION_TARGETS), set(expected_font_settings))
        for target_code, font_setting in expected_font_settings.items():
            with self.subTest(target_code=target_code):
                self.assertEqual(
                    translation_target(f"any_{target_code}").font_setting,
                    font_setting,
                )

    def test_rejects_an_unknown_direction(self):
        with self.assertRaisesRegex(ValueError, "Unsupported translation language"):
            translation_prompt(SimpleNamespace(), "any_xx")


if __name__ == "__main__":
    unittest.main()
