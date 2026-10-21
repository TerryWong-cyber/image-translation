import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import AsyncMock, patch

import httpx
import image_translation.api.main as api
import image_translation.services.image_translation_service as service_module
from fastapi import FastAPI
from image_translation.services import ImageTranslationService
from PIL import Image


def png():
    stream = BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, "PNG")
    return stream.getvalue()


class TranslationApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ocr = AsyncMock()
        self.ocr.recognize.return_value = []
        self.service = ImageTranslationService(self.ocr)
        self.service._image_source.fetch = AsyncMock(return_value=(png(), "image/png"))
        self.app = FastAPI()
        self.app.state.translation_service = self.service
        self.app.add_api_route("/translate", api.process_image_translate, methods=["POST"])
        self.app.add_api_route("/binary", api.process_binary_translate, methods=["POST"])
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )
        self.render = patch.object(
            service_module, "translate_image", side_effect=lambda image, ocr, language: (image, [])
        )
        self.render.start()
        self.config = patch.object(
            api,
            "settings",
            replace(
                api.settings,
                api=replace(api.settings.api, binary_enabled=True, binary_token="secret"),
            ),
        )
        self.config.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.render.stop()
        self.config.stop()

    async def test_url_copies_source_and_publishes_separate_result(self):
        results = []
        for _ in range(2):
            response = await self.client.post(
                "/translate",
                json={"image_url": "https://oss.toolup.cn/file/old/a.png", "language": "any_en"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertNotEqual(result["source_files"][0]["file_id"], result["files"][0]["file_id"])
            self.assertTrue(result["source_url"].startswith("https://files.example/"))
            self.assertTrue(result["translated_url"].startswith("https://files.example/"))
            results.append(result["files"][0]["file_id"])
        self.assertNotEqual(*results)

    async def test_legacy_storage_fields_invalid_url_and_invalid_language_are_rejected(self):
        for body in (
            {"image_url": "images/a.png", "bucket": "images"},
            {"image_url": "https://example.com/a.png", "save_bucket": "other"},
            {"image_url": "file:///tmp/a.png"},
            {"image_url": "https://example.com/a.png", "language": "invalid"},
        ):
            response = await self.client.post("/translate", json=body)
            self.assertEqual(response.status_code, 422, response.text)
        self.service._image_source.fetch.assert_not_called()

    async def test_file_service_upload_returns_reference_and_propagates_failure(self):
        import os

        from image_translation.clients.file_service_client import (
            FileServiceClient,
            FileServiceError,
        )

        artifact = {
            "file_id": "00000000-0000-0000-0000-000000000010",
            "file_url": "https://files.example/content?token=short",
            "filename": "translated.png",
            "mime_type": "image/png",
            "size_bytes": 10,
            "expires_at": "2099-01-01T00:00:00Z",
        }
        with (
            patch.dict(os.environ, {"FILE_SERVICE_BASE_URL": "https://files.example"}),
            patch.object(FileServiceClient, "upload", new_callable=AsyncMock) as upload,
        ):
            upload.return_value = artifact
            response = await self.client.post(
                "/translate", json={"image_url": "https://example.com/a.png"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["translated_url"], artifact["file_url"])
            self.assertEqual(response.json()["files"][0]["file_id"], artifact["file_id"])
            upload.side_effect = FileServiceError(404, "file_not_found")
            response = await self.client.post(
                "/translate", json={"image_url": "https://example.com/a.png"}
            )
            self.assertEqual(response.status_code, 404)

    async def test_binary_preserves_format_and_publishes_files(self):
        for format_name, mime in (
            ("PNG", "image/png"),
            ("JPEG", "image/jpeg"),
            ("WEBP", "image/webp"),
        ):
            content = BytesIO()
            Image.new("RGB", (8, 8)).save(content, format_name)
            response = await self.client.post(
                "/binary?language=any_en",
                content=content.getvalue(),
                headers={"authorization": "Bearer secret", "content-type": mime},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["content-type"], mime)
            self.assertEqual(Image.open(BytesIO(response.content)).format, format_name)
            from uuid import UUID

            UUID(response.headers["x-request-id"])
        self.service._image_source.fetch.assert_not_called()

    async def test_binary_auth_media_type_and_size_limits(self):
        response = await self.client.post(
            "/binary", content=png(), headers={"content-type": "image/png"}
        )
        self.assertEqual(response.status_code, 401)
        response = await self.client.post(
            "/binary",
            content=b"{}",
            headers={"authorization": "Bearer secret", "content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 415)
        with patch.object(
            api, "settings", replace(api.settings, api=replace(api.settings.api, max_image_bytes=2))
        ):
            response = await self.client.post(
                "/binary",
                content=png(),
                headers={"authorization": "Bearer secret", "content-type": "image/png"},
            )
        self.assertEqual(response.status_code, 413)
        self.assertIn("x-request-id", response.headers)
        self.ocr.recognize.assert_not_called()

    async def test_bad_image_and_pixel_limit_fail_before_ocr(self):
        for content, limit, status in ((b"bad", 100, 400), (png(), 10, 413)):
            self.service._settings = replace(
                self.service._settings,
                api=replace(self.service._settings.api, max_image_pixels=limit),
            )
            response = await self.client.post(
                "/binary",
                content=content,
                headers={"authorization": "Bearer secret", "content-type": "image/png"},
            )
            self.assertEqual(response.status_code, status, response.text)
        self.ocr.recognize.assert_not_called()

    async def test_upload_and_download_failure_are_not_success(self):
        from image_translation.clients.file_service_client import (
            FileServiceClient,
            FileServiceError,
        )
        from image_translation.errors import ImageDownloadError

        self.service._image_source.fetch.side_effect = ImageDownloadError(
            "Failed to download source image"
        )
        response = await self.client.post(
            "/translate", json={"image_url": "https://example.com/a.png"}
        )
        self.assertEqual(response.status_code, 502)
        self.service._image_source.fetch.side_effect = None
        with patch.object(
            FileServiceClient,
            "upload",
            AsyncMock(side_effect=FileServiceError(502, "storage_upload_failed")),
        ):
            response = await self.client.post(
                "/translate", json={"image_url": "https://example.com/a.png"}
            )
            self.assertEqual(response.status_code, 502)

    async def test_model_failure_aborts_both_routes_without_output_success(self):
        from image_translation.errors import ModelTimeoutError, ModelUpstreamError

        for error, status in (
            (ModelUpstreamError("Model returned empty text"), 502),
            (ModelTimeoutError("Model request timed out"), 504),
        ):
            with patch.object(service_module, "translate_image", side_effect=error):
                response = await self.client.post(
                    "/translate", json={"image_url": "https://example.com/a.png"}
                )
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(response.json(), {"error": str(error)})
                self.assertIn("x-request-id", response.headers)
                response = await self.client.post(
                    "/binary",
                    content=png(),
                    headers={"authorization": "Bearer secret", "content-type": "image/png"},
                )
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(response.json(), {"error": str(error)})

    async def test_recognition_model_failure_is_not_an_empty_ocr_success(self):
        from image_translation.errors import ModelUpstreamError

        self.ocr.recognize.side_effect = ModelUpstreamError("Recognition model unavailable")
        response = await self.client.post(
            "/translate", json={"image_url": "https://example.com/a.png"}
        )
        self.assertEqual(response.status_code, 502)


class DefaultBinaryRouteTest(unittest.TestCase):
    def test_binary_endpoint_is_not_registered_or_documented_by_default(self):
        self.assertFalse(api.settings.api.binary_enabled)
        self.assertNotIn(
            "/internal/image-translate", [getattr(route, "path", None) for route in api.app.routes]
        )
        self.assertNotIn("/internal/image-translate", api.app.openapi()["paths"])
