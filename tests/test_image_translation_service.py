import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import AsyncMock, patch

from PIL import Image
import image_translation.services.image_translation_service as service_module
from image_translation.services import ImageTranslationService


class ImageTranslationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_url_orchestrates_download_ocr_translation_and_default_storage(self):
        stream = BytesIO()
        Image.new("RGB", (8, 8)).save(stream, "PNG")
        content = stream.getvalue()
        oss = AsyncMock()
        ocr = AsyncMock()
        ocr.recognize.return_value = []
        service = ImageTranslationService(oss, ocr)
        service._settings = replace(service._settings, oss=replace(service._settings.oss, default_bucket="results", public_base_url="https://files.example.com"))
        service._image_source.fetch = AsyncMock(return_value=(content, "image/png"))
        items = [{"coordinate": [[0, 0], [1, 0], [1, 1], [0, 1]], "source_text": "Hello", "translated_text": "你好"}]
        with patch.object(service_module, "translate_image", side_effect=lambda image, result, language: (image, items)):
            result = await service.translate_url("https://images.example.com/a.png", "any_zh", True)
        service._image_source.fetch.assert_awaited_once_with("https://images.example.com/a.png")
        ocr.recognize.assert_awaited_once_with(content, True)
        args = oss.upload_image.call_args.args
        self.assertEqual(args[0], "results")
        self.assertEqual(args[1], f"translations/{result.request_id}.png")
        self.assertEqual(args[3], "image/png")
        self.assertEqual(result.source_url, "https://images.example.com/a.png")
        self.assertEqual(result.translated_url, f"https://files.example.com/file/results/translations/{result.request_id}.png")
        self.assertEqual(result.data[0].translated_text, "你好")
