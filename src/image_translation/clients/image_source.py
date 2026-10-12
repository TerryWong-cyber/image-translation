"""Bounded image downloads with per-hop URL checks and pinned public DNS."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import unquote

import httpx

from image_translation.config import ApiSettings, OssSettings
from image_translation.errors import ImageDownloadError, ImageTooLargeError, InvalidImageUrlError


class ImageSourceClient:
    def __init__(self, api: ApiSettings, oss: OssSettings):
        self.api = api
        self.oss = oss

    def _url(self, value: str) -> httpx.URL:
        try:
            url = httpx.URL(value)
            if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.fragment:
                raise ValueError()
            # Force port parsing before dispatch; invalid ports must be client errors.
            _ = url.port
            return url
        except (ValueError, httpx.InvalidURL) as exc:
            raise InvalidImageUrlError("Expected an HTTP(S) image URL without credentials or fragment") from exc

    def _trusted_oss(self, url: httpx.URL) -> bool:
        base = httpx.URL(self.oss.base_url)
        origin = lambda u: (u.scheme, u.host, u.port)
        # Only the configured OSS file route can use a private address, never /upload or arbitrary endpoints.
        file_prefix = httpx.URL(self.oss.file_url("PLACEHOLDER", "PLACEHOLDER")).path.split("PLACEHOLDER", 1)[0]
        path = unquote(unquote(url.path))
        return (
            origin(url) == origin(base)
            and path.startswith(file_prefix)
            and ".." not in path.split("/")
            and "\\" not in path
            and "\x00" not in path
        )

    async def _destination(self, url: httpx.URL) -> tuple[httpx.URL, dict, dict]:
        if self._trusted_oss(url):
            return url, {}, {}
        try:
            addresses = await asyncio.get_running_loop().getaddrinfo(
                url.host, url.port or (443 if url.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise ImageDownloadError("Cannot resolve image host") from exc
        ips = [ipaddress.ip_address(address[4][0]) for address in addresses]
        if not ips or any(not ip.is_global or (getattr(ip, "ipv4_mapped", None) is not None and not ip.ipv4_mapped.is_global) for ip in ips):
            raise InvalidImageUrlError("Image URL must resolve to public addresses or the configured OSS file service")
        # Pin the connection to the checked IP; retain HTTP Host and TLS server name.
        return url.copy_with(host=str(ips[0])), {"Host": url.netloc.decode("ascii")}, {"sni_hostname": url.host}

    async def fetch(self, value: str) -> tuple[bytes, str | None]:
        try:
            return await asyncio.wait_for(self._fetch(value), timeout=self.api.download_timeout_seconds)
        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            raise TimeoutError("Image download timed out") from exc
        except httpx.HTTPError as exc:
            raise ImageDownloadError("Failed to download source image") from exc

    async def _fetch(self, value: str) -> tuple[bytes, str | None]:
        url = self._url(value)
        for hop in range(self.api.max_redirects + 1):
            target, headers, extensions = await self._destination(url)
            headers["Accept-Encoding"] = "identity"
            # A fresh client per hop prevents a TLS connection pinned to one IP
            # from being reused for a different hostname on that same IP.
            async with httpx.AsyncClient(
                trust_env=False, follow_redirects=False,
                timeout=self.api.download_timeout_seconds,
            ) as client:
                async with client.stream("GET", target, headers=headers, extensions=extensions) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        if hop == self.api.max_redirects or not response.headers.get("location"):
                            raise ImageDownloadError("Image redirect limit exceeded or missing Location")
                        url = self._url(str(url.join(response.headers["location"])))
                        continue
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ImageDownloadError("Compressed HTTP image responses are not supported")
                    length = response.headers.get("content-length")
                    if length and length.isdigit() and int(length) > self.api.max_image_bytes:
                        raise ImageTooLargeError("Image exceeds the configured byte limit")
                    content = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(content) + len(chunk) > self.api.max_image_bytes:
                            raise ImageTooLargeError("Image exceeds the configured byte limit")
                        content.extend(chunk)
                    return bytes(content), response.headers.get("content-type")
        raise ImageDownloadError("Failed to download source image")
