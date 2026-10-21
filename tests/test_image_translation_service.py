import unittest
from io import BytesIO
from unittest.mock import AsyncMock, patch

import image_translation.services.image_translation_service as service_module
from image_translation.services import ImageTranslationService
from PIL import Image


class ImageTranslationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_url_is_copied_before_ocr_and_result_publication(self):
        stream = BytesIO()
        Image.new("RGB", (8, 8)).save(stream, "PNG")
        content = stream.getvalue()
        ocr = AsyncMock()
        ocr.recognize.return_value = []
        service = ImageTranslationService(ocr)
        service._image_source.fetch = AsyncMock(return_value=(content, "image/png"))
        items = [
            {
                "coordinate": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "source_text": "Hello",
                "translated_text": "你好",
            }
        ]
        with patch.object(
            service_module,
            "translate_image",
            side_effect=lambda image, result, language: (image, items),
        ):
            result = await service.translate_url(
                "https://oss.toolup.cn/file/old/a.png", "any_zh", True
            )
        service._image_source.fetch.assert_awaited_once_with("https://oss.toolup.cn/file/old/a.png")
        ocr.recognize.assert_awaited_once_with(content, True)
        self.assertNotEqual(result.source_files[0]["file_id"], result.files[0]["file_id"])
        self.assertTrue(result.source_url.startswith("https://files.example/"))
        self.assertTrue(result.translated_url.startswith("https://files.example/"))
        self.assertEqual(result.data[0].translated_text, "你好")
