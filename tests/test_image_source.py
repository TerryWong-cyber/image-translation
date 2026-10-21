import socket
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx
from image_translation.clients.image_source import ImageSourceClient
from image_translation.config import ApiSettings
from image_translation.errors import ImageDownloadError, ImageTooLargeError, InvalidImageUrlError


class ImageSourceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api = ApiSettings("/translate", max_image_bytes=8)
        self.source = ImageSourceClient(self.api)

    async def test_private_hosts_credentials_and_non_http_are_rejected(self):
        for url in (
            "file:///tmp/a.png",
            "http://user:pass@example.com/a.png",
            "http://example.com/a#fragment",
        ):
            with self.assertRaises(InvalidImageUrlError):
                self.source._url(url)
        for address in ("127.0.0.1", "169.254.169.254", "10.0.0.1", "::1", "::ffff:127.0.0.1"):
            with patch(
                "asyncio.BaseEventLoop.getaddrinfo",
                AsyncMock(
                    return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]
                ),
            ):
                with self.assertRaises(InvalidImageUrlError):
                    await self.source._destination(httpx.URL("http://example.com/a.png"))

    async def test_dns_is_pinned_but_host_and_tls_name_are_preserved(self):
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
        with patch("asyncio.BaseEventLoop.getaddrinfo", AsyncMock(return_value=addresses)):
            url, headers, extensions = await self.source._destination(
                httpx.URL("https://example.com/a.png?token=abc")
            )
        self.assertEqual(str(url), "https://8.8.8.8/a.png?token=abc")
        self.assertEqual(headers, {"Host": "example.com"})
        self.assertEqual(extensions, {"sni_hostname": "example.com"})

    async def test_old_gateway_url_has_no_network_privilege(self):
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 5000))]
        with patch("asyncio.BaseEventLoop.getaddrinfo", AsyncMock(return_value=addresses)):
            with self.assertRaises(InvalidImageUrlError):
                await self.source._destination(httpx.URL("http://127.0.0.1:5000/file/images/a.png"))

    async def fetch_with_transport(self, handler, url="https://source.example/image.png"):
        self.source.api = replace(self.source.api, url_import_allowed_hosts=("source.example",))
        client_class = httpx.AsyncClient
        with patch(
            "image_translation.clients.image_source.httpx.AsyncClient",
            side_effect=lambda **kwargs: client_class(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        ):
            return await self.source.fetch(url)

    async def test_reads_bounded_body_and_checks_redirect_destination(self):
        content, mime = await self.fetch_with_transport(
            lambda request: httpx.Response(
                200, content=b"png", headers={"content-type": "image/png"}
            )
        )
        self.assertEqual((content, mime), (b"png", "image/png"))
        with patch(
            "asyncio.BaseEventLoop.getaddrinfo",
            AsyncMock(
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]
            ),
        ):
            with self.assertRaises(InvalidImageUrlError):
                await self.fetch_with_transport(
                    lambda request: httpx.Response(
                        302, headers={"location": "http://169.254.169.254/latest/meta-data"}
                    )
                )

    async def test_stream_and_content_length_limits(self):
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"01234"
                yield b"56789"

        for response in (
            httpx.Response(200, headers={"content-length": "100"}),
            httpx.Response(200, stream=Body()),
        ):
            with self.assertRaises(ImageTooLargeError):
                await self.fetch_with_transport(lambda request: response)

    async def test_redirect_loop_and_upstream_error(self):
        for response in (
            httpx.Response(302, headers={"location": "/file/images/a.png"}),
            httpx.Response(404),
        ):
            with self.assertRaises(ImageDownloadError):
                await self.fetch_with_transport(lambda request: response)

    async def test_public_redirect_keeps_host_tls_name_and_signed_query(self):
        requests = []

        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(
                    302, headers={"location": "https://other.example.com/b.png?signature=abc"}
                )
            return httpx.Response(200, content=b"png")

        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
        with patch("asyncio.BaseEventLoop.getaddrinfo", AsyncMock(return_value=addresses)):
            content, _ = await self.fetch_with_transport(handler, "https://example.com/a.png")
        self.assertEqual(content, b"png")
        self.assertEqual(
            [request.headers["host"] for request in requests], ["example.com", "other.example.com"]
        )
        self.assertEqual(requests[1].extensions["sni_hostname"], "other.example.com")
        self.assertEqual(str(requests[1].url), "https://8.8.8.8/b.png?signature=abc")

    async def test_compressed_http_response_is_rejected_before_decoding(self):
        with self.assertRaises(ImageDownloadError):
            await self.fetch_with_transport(
                lambda request: httpx.Response(200, headers={"content-encoding": "gzip"})
            )

    async def test_total_download_deadline(self):
        import asyncio

        self.source.api = replace(self.api, download_timeout_seconds=0.01)

        async def stalled(value):
            await asyncio.sleep(1)

        with patch.object(self.source, "_fetch", side_effect=stalled):
            with self.assertRaises(TimeoutError):
                await self.source.fetch("https://example.com/a.png")
