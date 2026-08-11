import importlib.util
import unittest
from unittest.mock import patch


SERVICE_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("cv2", "numpy", "pydantic", "PIL", "sklearn")
)

if SERVICE_DEPS_AVAILABLE:
    import image_translation.services.image_translation_service as service_module
    from image_translation.contracts import TranslationCommand
    from image_translation.services import ImageTranslationService


@unittest.skipUnless(SERVICE_DEPS_AVAILABLE, "image service dependencies are not installed")
class ImageTranslationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrates_oss_ocr_translation_and_upload(self):
        class FakeOssClient:
            def __init__(self):
                self.upload = None

            async def get_oss_file(self, bucket_name, file_key, as_string=True):
                return {
                    "content": b"\x89PNG\r\n\x1a\nsource",
                    "content_type": "image/png",
                }

            async def upload_image(self, bucket_name, key, image_bytes, content_type="image/jpeg"):
                self.upload = (bucket_name, key, image_bytes, content_type)
                return {"file_key": key}

        class FakeOcrService:
            async def recognize(self, image_content, segment_enabled=False):
                return [[[[[0, 0], [10, 0], [10, 10], [0, 10]], ("Hello", 0.99)]]]

        oss_client = FakeOssClient()
        service = ImageTranslationService(oss_client, FakeOcrService())
        raw_items = [
            {
                "coordinate": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "source_text": "Hello",
                "translated_text": "你好",
            }
        ]

        with (
            patch.object(service_module.np, "frombuffer", return_value=object()),
            patch.object(service_module.cv2, "imdecode", return_value=object()),
            patch.object(service_module, "translate_image", return_value=(object(), raw_items)),
            patch.object(service_module, "encode_image", return_value=b"translated-png"),
        ):
            result = await service.translate(
                TranslationCommand(
                    bucket="source",
                    image_key="folder/input.png",
                    save_bucket="translated",
                )
            )

        self.assertEqual(result.source_url, "source/folder/input.png")
        self.assertEqual(
            result.translated_url,
            "translated/en_zh_translated_folder/input.png",
        )
        self.assertEqual(result.data[0].translated_text, "你好")
        self.assertEqual(
            oss_client.upload,
            (
                "translated",
                "en_zh_translated_folder/input.png",
                b"translated-png",
                "image/png",
            ),
        )


if __name__ == "__main__":
    unittest.main()
