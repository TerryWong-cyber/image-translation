import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from image_translation.utils.image_encoding import (
    encode_image,
    replace_image_extension,
    resolve_image_encoding,
)


class ImageEncodingTest(unittest.TestCase):
    def test_detects_png_from_bytes_before_mismatched_suffix(self):
        spec = resolve_image_encoding(
            b"\x89PNG\r\n\x1a\ncontent",
            "drawings/example.jpg",
            "image/jpeg",
        )

        self.assertEqual(spec.format_name, "png")
        self.assertEqual(spec.output_extension, ".png")
        self.assertEqual(spec.content_type, "image/png")
        self.assertEqual(
            spec.parameters,
            (("IMWRITE_PNG_COMPRESSION", 3),),
        )

    def test_preserves_jpeg_extension_alias(self):
        spec = resolve_image_encoding(b"\xff\xd8\xffcontent", "example.jpeg")

        self.assertEqual(spec.codec_extension, ".jpg")
        self.assertEqual(spec.output_extension, ".jpeg")
        self.assertEqual(spec.content_type, "image/jpeg")

    def test_uses_suffix_then_defaults_to_jpeg(self):
        webp_spec = resolve_image_encoding(b"unknown", "example.webp")
        default_spec = resolve_image_encoding(b"unknown", "example")

        self.assertEqual(webp_spec.format_name, "webp")
        self.assertEqual(default_spec.format_name, "jpeg")
        self.assertEqual(default_spec.output_extension, ".jpg")

    def test_encodes_png_without_lossy_jpeg_conversion(self):
        image = object()
        spec = resolve_image_encoding(b"\x89PNG\r\n\x1a\n", "example.png")
        buffer = Mock()
        buffer.tobytes.return_value = b"\x89PNG\r\n\x1a\nencoded"
        cv2_module = SimpleNamespace(
            IMWRITE_PNG_COMPRESSION=16,
            imencode=Mock(return_value=(True, buffer)),
        )

        with patch.dict(sys.modules, {"cv2": cv2_module}):
            encoded = encode_image(image, spec)

        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
        cv2_module.imencode.assert_called_once_with(".png", image, [16, 3])

    def test_replaces_or_adds_output_extension(self):
        self.assertEqual(
            replace_image_extension("folder/example.jpg", ".png"),
            "folder/example.png",
        )
        self.assertEqual(
            replace_image_extension("folder/example", ".jpg"),
            "folder/example.jpg",
        )


if __name__ == "__main__":
    unittest.main()
