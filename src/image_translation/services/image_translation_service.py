"""Shared orchestration used by REST and MCP transports."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError
from typing import Protocol
from uuid import uuid4

import cv2
import numpy as np

from image_translation.contracts import (
    TranslationItem,
    TranslationResult,
)
from image_translation.errors import InvalidImageError, ImageTooLargeError, OcrTimeoutError
from image_translation.config import get_settings
from image_translation.clients.image_source import ImageSourceClient
from image_translation.image_translation import translate_image
from image_translation.utils.image_encoding import (
    encode_image,
    resolve_image_encoding,
)


class OssClient(Protocol):
    async def upload_image(
        self,
        bucket_name: str,
        key: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> dict: ...


class OcrService(Protocol):
    async def recognize(self, image_content: bytes, segment_enabled: bool = False) -> list: ...


@dataclass(frozen=True)
class TranslatedImage:
    content: bytes
    content_type: str
    extension: str
    data: list[TranslationItem]


class ImageTranslationService:
    """Translate image bytes; protocol adapters choose download and persistence."""

    def __init__(self, oss_client: OssClient, ocr_service: OcrService):
        self._oss_client = oss_client
        self._ocr_service = ocr_service
        self._settings = get_settings()
        self._image_source = ImageSourceClient(self._settings.api, self._settings.oss)

    def _decode(self, content: bytes):
        if not content:
            raise InvalidImageError("Image body is empty")
        if len(content) > self._settings.api.max_image_bytes:
            raise ImageTooLargeError("Image exceeds the configured byte limit")
        try:
            with Image.open(BytesIO(content)) as source:
                if source.width * source.height > self._settings.api.max_image_pixels:
                    raise ImageTooLargeError("Image exceeds the configured pixel limit")
                source.verify()
        except Image.DecompressionBombError as exc:
            raise ImageTooLargeError("Image exceeds the safe pixel limit") from exc
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise InvalidImageError("Input is not a decodable image") from exc
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Input is not a decodable image")
        return image

    async def translate_bytes(
        self, content: bytes, language: str = "en_zh", segment: bool = False,
        file_name: str = "", content_type: str | None = None,
    ) -> TranslatedImage:
        image = await asyncio.to_thread(self._decode, content)
        encoding = resolve_image_encoding(content, file_name, content_type)
        try:
            ocr_result = await self._ocr_service.recognize(content, segment)
        except asyncio.TimeoutError as exc:
            raise OcrTimeoutError("OCR processing timed out") from exc
        result_image, raw_items = await asyncio.to_thread(translate_image, image, ocr_result, language)
        result_bytes = await asyncio.to_thread(encode_image, result_image, encoding)
        return TranslatedImage(
            result_bytes, encoding.content_type, encoding.output_extension,
            [TranslationItem.model_validate(item) for item in raw_items],
        )

    async def translate_url(self, image_url: str, language: str = "en_zh", segment: bool = False) -> TranslationResult:
        started_at = time.monotonic()
        request_id = str(uuid4())
        content, content_type = await self._image_source.fetch(image_url)
        translated = await self.translate_bytes(content, language, segment, urlsplit(image_url).path, content_type)
        bucket = self._settings.oss.default_bucket
        key = f"translations/{request_id}{translated.extension}"
        await self._oss_client.upload_image(bucket, key, translated.content, translated.content_type)
        return TranslationResult(
            request_id=request_id, source_url=image_url,
            translated_url=self._settings.oss.public_file_url(bucket, key),
            data=translated.data,
            duration_ms=max(0, round((time.monotonic() - started_at) * 1000)),
        )
