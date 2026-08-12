import os
import unittest
from pathlib import Path
from unittest.mock import patch

from image_translation.components.text_filter.translation_filter import should_translate
from image_translation.config import get_settings


EXAMPLE_ENV = Path(__file__).resolve().parents[1] / ".env.example"


class TranslationFilterTest(unittest.TestCase):
    def tearDown(self):
        get_settings.cache_clear()

    def test_should_translate_uses_configured_terms_and_language_rules(self):
        with patch.dict(os.environ, {"IMAGE_TRANSLATION_ENV_FILE": str(EXAMPLE_ENV)}, clear=True):
            get_settings.cache_clear()
            self.assertFalse(should_translate("12 mm"))
            self.assertFalse(should_translate("2.5 GHz"))
            self.assertFalse(should_translate("PCB"))
            self.assertTrue(should_translate("Hello world"))
            self.assertTrue(should_translate("中文", language="zh_en"))
            self.assertFalse(should_translate("中文", language="any_zh"))
            self.assertTrue(should_translate("繁體中文", language="any_zh_cn"))
            self.assertTrue(should_translate("简体中文", language="any_zh_tw"))
            self.assertTrue(should_translate("中文", language="any_en"))
            self.assertTrue(should_translate("English", language="any_ko"))
            self.assertTrue(should_translate("日本語", language="any_th"))


if __name__ == "__main__":
    unittest.main()
