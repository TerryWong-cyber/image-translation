import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from PIL import Image

import image_translation.api.main as api
import image_translation.services.image_translation_service as service_module
from image_translation.services import ImageTranslationService


def png():
    stream = BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, "PNG")
    return stream.getvalue()


class TranslationApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.oss = AsyncMock()
        self.ocr = AsyncMock()
        self.ocr.recognize.return_value = []
        self.service = ImageTranslationService(self.oss, self.ocr)
        self.service._settings = replace(self.service._settings, oss=replace(self.service._settings.oss, default_bucket="results", public_base_url="https://files.example.com/storage"))
        self.service._image_source.fetch = AsyncMock(return_value=(png(), "image/png"))
        self.app = FastAPI()
        self.app.state.translation_service = self.service
        self.app.add_api_route("/translate", api.process_image_translate, methods=["POST"])
        self.app.add_api_route("/binary", api.process_binary_translate, methods=["POST"])
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.render = patch.object(service_module, "translate_image", side_effect=lambda image, ocr, language: (image, []))
        self.render.start()
        self.config = patch.object(api, "settings", replace(api.settings, api=replace(api.settings.api, binary_enabled=True, binary_token="secret")))
        self.config.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.render.stop()
        self.config.stop()

    async def test_url_returns_full_link_uploads_default_bucket_and_uses_unique_keys(self):
        results = []
        for _ in range(2):
            response = await self.client.post("/translate", json={"image_url": "https://example.com/input.png", "language": "any_en"})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result["translated_url"].startswith("https://files.example.com/storage/file/results/translations/"))
            self.assertEqual(result["source_url"], "https://example.com/input.png")
            self.assertIn("request_id", result)
            self.assertEqual(response.headers["x-request-id"], result["request_id"])
            results.append(result["translated_url"])
        self.assertNotEqual(*results)
        args = self.oss.upload_image.call_args.args
        self.assertEqual(args[0], "results")
        self.assertEqual(args[3], "image/png")
        self.assertEqual(Image.open(BytesIO(args[2])).format, "PNG")

    async def test_legacy_storage_fields_invalid_url_and_invalid_language_are_rejected(self):
        for body in ({"image_url": "images/a.png", "bucket": "images"}, {"image_url": "https://example.com/a.png", "save_bucket": "other"}, {"image_url": "file:///tmp/a.png"}, {"image_url": "https://example.com/a.png", "language": "invalid"}):
            response = await self.client.post("/translate", json=body)
            self.assertEqual(response.status_code, 422, response.text)
        self.service._image_source.fetch.assert_not_called()

    async def test_binary_preserves_format_and_never_downloads_or_uploads(self):
        for format_name, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
            content = BytesIO()
            Image.new("RGB", (8, 8)).save(content, format_name)
            response = await self.client.post("/binary?language=any_en", content=content.getvalue(), headers={"authorization": "Bearer secret", "content-type": mime})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["content-type"], mime)
            self.assertEqual(Image.open(BytesIO(response.content)).format, format_name)
            from uuid import UUID
            UUID(response.headers["x-request-id"])
        self.oss.upload_image.assert_not_called()
        self.service._image_source.fetch.assert_not_called()

    async def test_binary_auth_media_type_and_size_limits(self):
        response = await self.client.post("/binary", content=png(), headers={"content-type": "image/png"})
        self.assertEqual(response.status_code, 401)
        response = await self.client.post("/binary", content=b"{}", headers={"authorization": "Bearer secret", "content-type": "application/json"})
        self.assertEqual(response.status_code, 415)
        with patch.object(api, "settings", replace(api.settings, api=replace(api.settings.api, max_image_bytes=2))):
            response = await self.client.post("/binary", content=png(), headers={"authorization": "Bearer secret", "content-type": "image/png"})
        self.assertEqual(response.status_code, 413)
        self.assertIn("x-request-id", response.headers)
        self.ocr.recognize.assert_not_called()

    async def test_bad_image_and_pixel_limit_fail_before_ocr(self):
        for content, limit, status in ((b"bad", 100, 400), (png(), 10, 413)):
            self.service._settings = replace(self.service._settings, api=replace(self.service._settings.api, max_image_pixels=limit))
            response = await self.client.post("/binary", content=content, headers={"authorization": "Bearer secret", "content-type": "image/png"})
            self.assertEqual(response.status_code, status, response.text)
        self.ocr.recognize.assert_not_called()

    async def test_upload_and_download_failure_are_not_success(self):
        from image_translation.errors import ImageDownloadError
        self.service._image_source.fetch.side_effect = ImageDownloadError("Failed to download source image")
        response = await self.client.post("/translate", json={"image_url": "https://example.com/a.png"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("x-request-id", response.headers)
        self.service._image_source.fetch.side_effect = None
        self.oss.upload_image.side_effect = RuntimeError("upload failed")
        response = await self.client.post("/translate", json={"image_url": "https://example.com/a.png"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("x-request-id", response.headers)


class LocalOssHttpTest(unittest.IsolatedAsyncioTestCase):
    async def test_url_download_upload_and_returned_url_over_real_http(self):
        import json
        import threading
        from email.parser import BytesParser
        from email.policy import default
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from image_translation.clients.oss_client import OSSClient
        from image_translation.clients.image_source import ImageSourceClient
        from image_translation.config import OssSettings

        objects = {"/file/source/input.png": png()}
        uploads = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = objects.get(self.path)
                self.send_response(200 if body else 404)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(body or b"missing")

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                message = BytesParser(policy=default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body
                )
                fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True) for part in message.iter_parts()}
                uploads.append(fields)
                objects[f"/file/{fields['bucket_name'].decode()}/{fields['key'].decode()}"] = fields["file"]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"file_key": fields["key"].decode()}).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            oss_settings = OssSettings(base, "/upload", "/file/{bucket}/{key}", "results", base)
            oss = OSSClient(oss_settings, 5)
            ocr = AsyncMock()
            ocr.recognize.return_value = []
            service = ImageTranslationService(oss, ocr)
            service._settings = replace(service._settings, oss=oss_settings)
            service._image_source = ImageSourceClient(service._settings.api, oss_settings)
            app = FastAPI()
            app.state.translation_service = service
            app.add_api_route("/translate", api.process_image_translate, methods=["POST"])
            with patch.object(service_module, "translate_image", side_effect=lambda image, ocr, language: (image, [])):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post("/translate", json={"image_url": base + "/file/source/input.png"})
            self.assertEqual(response.status_code, 200, response.text)
            async with httpx.AsyncClient(trust_env=False) as client:
                downloaded = await client.get(response.json()["translated_url"])
            self.assertEqual(downloaded.status_code, 200)
            self.assertEqual(Image.open(BytesIO(downloaded.content)).format, "PNG")
            self.assertEqual(uploads[0]["bucket_name"], b"results")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class DefaultBinaryRouteTest(unittest.TestCase):
    def test_binary_endpoint_is_not_registered_or_documented_by_default(self):
        self.assertFalse(api.settings.api.binary_enabled)
        self.assertNotIn("/internal/image-translate", [getattr(route, "path", None) for route in api.app.routes])
        self.assertNotIn("/internal/image-translate", api.app.openapi()["paths"])
