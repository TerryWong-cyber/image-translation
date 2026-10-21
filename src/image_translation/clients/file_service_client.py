"""File Service v1 client. Vendored by consumers; canonical source: file-service/integrations.

Only configured backend credentials are sent to the configured API origin. Signed
URLs are opaque bearer credentials; keep file_id for renewal, never OSS locations.
Uploads are not retried: the service has no idempotency key support.
"""

import asyncio
import json
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FileServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FILE_SERVICE_", env_file=".env", extra="ignore")
    base_url: str = ""
    token: SecretStr = SecretStr("")
    storage_namespace: str = "default"
    timeout_seconds: float = 120

    @field_validator("base_url")
    @classmethod
    def origin(cls, value):
        if value:
            url = urlsplit(value)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.path not in {"", "/"}
            ):
                raise ValueError(
                    "FILE_SERVICE_BASE_URL must be an HTTP(S) origin without path or credentials"
                )
        return value.rstrip("/")

    @field_validator("storage_namespace")
    @classmethod
    def namespace(cls, value):
        import re

        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
            raise ValueError("invalid FILE_SERVICE_STORAGE_NAMESPACE")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def timeout(cls, value):
        if not 0 < value <= 3600:
            raise ValueError("FILE_SERVICE_TIMEOUT_SECONDS must be in (0, 3600]")
        return value


class FileServiceError(Exception):
    def __init__(self, status=502, code="file_service_unavailable"):
        self.status = status
        self.code = code
        super().__init__(code)


class FileServiceClient:
    def __init__(self, settings=None):
        self.settings = settings or FileServiceSettings()

    @property
    def configured(self):
        return bool(self.settings.base_url)

    def url(self, file_id=None, suffix=""):
        if not self.configured:
            raise FileServiceError(503, "file_service_not_configured")
        base = self.settings.base_url + "/api/file-service/v1/files"
        return base + ("/" + str(UUID(str(file_id))) + suffix if file_id else "")

    def headers(self):
        token = self.settings.token.get_secret_value()
        return {"Authorization": "Bearer " + token} if token else {}

    @staticmethod
    def decode(response):
        if response.is_error:
            # Do not include upstream body, request URL, credentials or signed queries in errors.
            code = "file_service_request_failed"
            try:
                candidate = response.json()["error"]["code"]
                if isinstance(candidate, str) and candidate.replace("_", "").isalnum():
                    code = candidate[:100]
            except (ValueError, KeyError, TypeError):
                pass
            raise FileServiceError(
                502 if response.status_code >= 500 else response.status_code, code
            )
        try:
            return response.json()
        except ValueError:
            raise FileServiceError(502, "file_service_invalid_response") from None

    async def request(self, client, method, file_id=None, suffix="", **kwargs):
        try:
            response = await client.request(
                method,
                self.url(file_id, suffix),
                headers=self.headers(),
                timeout=self.settings.timeout_seconds,
                follow_redirects=False,
                **kwargs,
            )
            if response.is_redirect:
                raise FileServiceError(502, "file_service_redirect_rejected")
            return self.decode(response)
        except httpx.HTTPError:
            raise FileServiceError() from None

    async def resolve(self, client, file_id):
        item = await self.request(client, "GET", file_id)
        link = await self.request(client, "POST", file_id, "/download-url")
        try:
            url = urlsplit(link["url"])
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.fragment
            ):
                raise ValueError()
            return {
                "file_id": str(UUID(item["id"])),
                "file_url": link["url"],
                "filename": item["file_name"],
                "mime_type": item["mime_type"] or "application/octet-stream",
                "size_bytes": item["size_bytes"],
                "sha256": item.get("sha256"),
                "expires_at": link["expires_at"],
            }
        except (KeyError, TypeError, ValueError):
            raise FileServiceError(502, "file_service_invalid_response") from None

    async def upload(self, client, filename, content, mime=None, metadata=None):
        # content may be bytes or a seekable spool/file; httpx streams file objects.
        item = await self.request(
            client,
            "POST",
            data={
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                "storage_namespace": self.settings.storage_namespace,
            },
            files={"file": (filename, content, mime or "application/octet-stream")},
        )
        try:
            file_id = str(UUID(item["id"]))
        except (KeyError, TypeError, ValueError):
            raise FileServiceError(502, "file_service_invalid_response") from None
        try:
            return await self.resolve(client, file_id)
        except FileServiceError as exc:
            # Upload succeeded: preserve the resource identity for recovery; do not retry upload.
            exc.file_id = file_id
            raise

    async def publish(self, path, filename=None, mime=None, metadata=None):
        path = Path(path)
        filename = filename or path.name
        async with httpx.AsyncClient() as client:
            content = await asyncio.to_thread(path.open, "rb")
            try:
                return await self.upload(
                    client, filename, content, mime or mimetypes.guess_type(filename)[0], metadata
                )
            finally:
                await asyncio.to_thread(content.close)

    def publish_sync(self, path, filename=None, mime=None, metadata=None):
        # PDF composition already runs in a worker thread; keep its public API synchronous.
        return asyncio.run(self.publish(path, filename, mime, metadata))

    async def metadata(self, file_id):
        async with httpx.AsyncClient() as client:
            return await self.request(client, "GET", file_id)

    async def download_bytes(self, file_id, max_bytes=100 * 1024 * 1024):
        from contextlib import aclosing

        response = await self.content_response(file_id)
        content = bytearray()
        async with aclosing(response.body_iterator):
            async for chunk in response.body_iterator:
                if len(content) + len(chunk) > max_bytes:
                    raise FileServiceError(413, "file_too_large")
                content.extend(chunk)
        return bytes(content)

    async def content_response(self, file_id, download=False):
        """Stream authenticated File Service content; no local-file fallback."""
        import anyio
        from starlette.responses import StreamingResponse

        url = self.url(file_id, "/content")
        client = httpx.AsyncClient(timeout=self.settings.timeout_seconds, follow_redirects=False)
        context = client.stream(
            "GET", url, params={"download": str(download).lower()}, headers=self.headers()
        )
        entered = False
        try:
            response = await context.__aenter__()
            entered = True
            if response.is_redirect:
                raise FileServiceError(502, "file_service_redirect_rejected")
            if response.is_error:
                await response.aread()
                self.decode(response)
        except BaseException as error:
            with anyio.CancelScope(shield=True):
                if entered:
                    await context.__aexit__(None, None, None)
                await client.aclose()
            if isinstance(error, httpx.HTTPError):
                raise FileServiceError() from None
            raise
        headers = {
            "Cache-Control": "private, no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        }
        for name in ("Content-Type", "Content-Length", "Content-Disposition"):
            if name in response.headers:
                headers[name] = response.headers[name]

        closed = False

        async def close():
            nonlocal closed
            if closed:
                return
            closed = True
            with anyio.CancelScope(shield=True):
                await context.__aexit__(None, None, None)
                await client.aclose()

        async def chunks():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await close()

        class ContentResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    await close()

        return ContentResponse(chunks(), headers=headers)
