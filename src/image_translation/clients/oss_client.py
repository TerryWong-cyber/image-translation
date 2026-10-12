"""Internal result storage; callers submit source URLs to ImageSourceClient."""

import httpx
from typing import Optional
from contextlib import asynccontextmanager

from image_translation.config import OssSettings, get_settings


class OSSClient:
    def __init__(self, settings: OssSettings, request_timeout_seconds: float):
        self.settings = settings
        self.request_timeout_seconds = request_timeout_seconds
        self.client: Optional[httpx.AsyncClient] = None

    def set_shared_client(self, client: httpx.AsyncClient):
        """设置由 FastAPI 生命周期管理的共享客户端"""
        self.client = client

    async def close_shared_client(self):
        """Close the lifecycle-owned HTTP client."""
        if self.client:
            await self.client.aclose()

    @asynccontextmanager
    async def _get_client(self):
        """内部工具：优先使用共享客户端，否则创建临时客户端"""
        if self.client and not self.client.is_closed:
            yield self.client
        else:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                yield client

    async def upload_image(self, bucket_name: str, key: str, image_bytes: bytes, content_type: str = "image/jpeg"):
        """Persist translated bytes; any storage failure must fail the request."""
        async with self._get_client() as client:
            data = {'bucket_name': bucket_name, 'key': key, 'content_type': content_type}
            files = {'file': (key, image_bytes, content_type)}

            response = await client.post(self.settings.upload_url(), data=data, files=files)
            response.raise_for_status()
            return response.json()


def get_oss_client():
    settings = get_settings()
    return OSSClient(settings.oss, settings.http.request_timeout_seconds)


oss_client_instance = get_oss_client()
