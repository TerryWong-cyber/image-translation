import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


FONT_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "PIL")
)

if FONT_DEPS_AVAILABLE:
    from image_translation.components.font import font_process


@unittest.skipUnless(FONT_DEPS_AVAILABLE, "font rendering dependencies are not installed")
class FontProcessTest(unittest.TestCase):
    def setUp(self):
        font_process.font_cache.clear()

    def test_variable_font_uses_medium_and_preserves_other_axes(self):
        font = Mock()
        font.get_variation_axes.return_value = [
            {"name": b"Width", "minimum": 50, "default": 100, "maximum": 200},
            {"name": b"Weight", "minimum": 100, "default": 100, "maximum": 900},
        ]
        with patch.object(font_process.ImageFont, "truetype", return_value=font):
            self.assertIs(font_process.get_font(24, "variable.ttf"), font)
        font.set_variation_by_axes.assert_called_once_with([100, 500])

    def test_static_font_keeps_its_embedded_weight(self):
        font = Mock()
        font.get_variation_axes.side_effect = OSError("invalid argument")
        with patch.object(font_process.ImageFont, "truetype", return_value=font):
            self.assertIs(font_process.get_font(24, "medium.ttf"), font)
        font.set_variation_by_axes.assert_not_called()

    def test_font_cache_separates_weights_and_reuses_same_weight(self):
        def make_font(*args):
            font = Mock()
            font.get_variation_axes.return_value = [
                {"name": b"Weight", "minimum": 100, "default": 100, "maximum": 900},
            ]
            return font

        with patch.object(font_process.ImageFont, "truetype", side_effect=make_font):
            medium = font_process.get_font(24, "variable.ttf")
            semibold = font_process.get_font(24, "variable.ttf", weight=600)
            self.assertIs(font_process.get_font(24, "variable.ttf"), medium)
        self.assertIsNot(medium, semibold)
        semibold.set_variation_by_axes.assert_called_once_with([600])

    def test_resolves_language_specific_font_file(self):
        paths = SimpleNamespace(
            font_zh_file=Path("zh.ttf"),
            font_zh_cn_file=Path("zh-cn.ttf"),
            font_zh_tw_file=Path("zh-tw.ttf"),
            font_latin_file=Path("latin.ttf"),
            font_ja_file=Path("ja.ttf"),
            font_ko_file=Path("ko.ttf"),
            font_arabic_file=Path("arabic.ttf"),
            font_cyrillic_file=Path("cyrillic.ttf"),
            font_thai_file=Path("thai.ttf"),
            font_vietnamese_file=Path("vietnamese.ttf"),
        )
        expected = {
            "any_zh": "zh-cn.ttf",
            "any_zh_cn": "zh-cn.ttf",
            "any_zh_tw": "zh-tw.ttf",
            "any_en": "latin.ttf",
            "any_ja": "ja.ttf",
            "any_ko": "ko.ttf",
            "any_fr": "latin.ttf",
            "any_ar": "arabic.ttf",
            "any_de": "latin.ttf",
            "any_ru": "cyrillic.ttf",
            "any_nl": "latin.ttf",
            "any_pt": "latin.ttf",
            "any_th": "thai.ttf",
            "any_es": "latin.ttf",
            "any_it": "latin.ttf",
            "any_vi": "vietnamese.ttf",
            "any_id": "latin.ttf",
            "en_zh": "zh-cn.ttf",
            "zh_en": "latin.ttf",
        }

        for language, filename in expected.items():
            with self.subTest(language=language):
                self.assertEqual(
                    font_process.font_file_for_language(language, paths),
                    Path(filename),
                )

    def test_arabic_uses_rtl_layout_when_raqm_is_available(self):
        with patch.object(font_process.features, "check", return_value=True):
            self.assertEqual(
                font_process.text_layout_options("any_ar"),
                {"direction": "rtl", "language": "ar"},
            )

    def test_non_arabic_does_not_force_rtl_layout(self):
        with patch.object(font_process.features, "check", return_value=True):
            self.assertEqual(font_process.text_layout_options("any_fr"), {})


if __name__ == "__main__":
    unittest.main()
