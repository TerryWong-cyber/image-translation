import httpx
import mimetypes
from typing import Optional
from fastapi import UploadFile
from io import BytesIO
import io
from werkzeug.datastructures import FileStorage
import os
from contextlib import asynccontextmanager

from src.image_translation.config import OssSettings, get_settings


class OSSClient:
    def __init__(self, settings: OssSettings, request_timeout_seconds: float):
        self.settings = settings
        self.request_timeout_seconds = request_timeout_seconds
        self.client: Optional[httpx.AsyncClient] = None

    def set_shared_client(self, client: httpx.AsyncClient):
        """设置由 FastAPI 生命周期管理的共享客户端"""
        self.client = client

    async def close_shared_client(self):
        """关闭共享客户端 (修复 f_api.py 退出时的潜在崩溃)"""
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
        """
        新增方法：直接上传图片字节流 (修复 AttributeError)
        """
        async with self._get_client() as client:
            data = {'bucket_name': bucket_name, 'key': key, 'content_type': content_type}
            files = {'file': (key, image_bytes, content_type)}

            try:
                response = await client.post(self.settings.upload_url(), data=data, files=files)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400 and "Key already exists" in e.response.text:
                    return {"file_key": key, "message": "Key already exists"}
                raise

    async def get_oss_file(self, bucket_name: str, file_key: str, as_string: bool = True):
        async with self._get_client() as client:
            url = self.settings.file_url(bucket_name, file_key)
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.RequestError as e:
                print(f"请求失败: {url}, 错误: {e}")
                raise
            content_type = response.headers.get('Content-Type', 'application/octet-stream')
            content = response.content
            if as_string: content = content.decode('utf-8')
            return {"content_type": content_type, "content": content}

    async def get_oss_file_as_storage(self, bucket_name: str, file_key: str):
        async with self._get_client() as client:
            url = self.settings.file_url(bucket_name, file_key)
            response = await client.get(url)
            response.raise_for_status()
            file_stream = io.BytesIO(response.content)
            return FileStorage(stream=file_stream, filename=file_key, content_type=response.headers.get('Content-Type'))

    async def upload_file(self, bucket_name: str, file_path: str, key: Optional[str] = None,
                          content_type: Optional[str] = None):
        async with self._get_client() as client:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            data = {'bucket_name': bucket_name, 'key': key or os.path.basename(file_path), 'content_type': content_type}
            files = {'file': (key or os.path.basename(file_path), file_data, content_type)}
            response = await client.post(self.settings.upload_url(), data=data, files=files)
            response.raise_for_status()
            return response.json()

    async def get_oss_file_as_uploadfile(self, bucket_name: str, file_key: str) -> UploadFile:
        async with self._get_client() as client:
            url = self.settings.file_url(bucket_name, file_key)
            response = await client.get(url)
            response.raise_for_status()
            return UploadFile(file=BytesIO(response.content), filename=file_key)

    async def upload_content(self, bucket_name: str, key: str, content: str, content_type: Optional[str] = None):
        async with self._get_client() as client:
            content_bytes = content.encode('utf-8')
            if not content_type: content_type, _ = mimetypes.guess_type(key or "")
            data = {'bucket_name': bucket_name, 'key': key, 'content_type': content_type}
            files = {'file': (key, content_bytes, content_type)}
            try:
                response = await client.post(self.settings.upload_url(), data=data, files=files)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400 and "Key already exists" in e.response.text:
                    return {"file_key": key, "content_type": content_type, "message": "Key already exists"}
                raise


def get_oss_client():
    settings = get_settings()
    return OSSClient(settings.oss, settings.http.request_timeout_seconds)


oss_client_instance = get_oss_client()
