"""Shared orchestration used by REST and MCP transports."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from image_translation.clients.file_service_client import FileServiceClient, FileServiceSettings
from image_translation.clients.image_source import ImageSourceClient
from image_translation.config import get_settings
from image_translation.contracts import (
    TranslationItem,
    TranslationResult,
)
from image_translation.errors import ImageTooLargeError, InvalidImageError, OcrTimeoutError
from image_translation.image_translation import translate_image
from image_translation.utils.image_encoding import (
    encode_image,
    resolve_image_encoding,
)


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

    def __init__(self, ocr_service: OcrService):
        self._ocr_service = ocr_service
        self._settings = get_settings()
        self._image_source = ImageSourceClient(self._settings.api)

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
        self,
        content: bytes,
        language: str = "en_zh",
        segment: bool = False,
        file_name: str = "",
        content_type: str | None = None,
    ) -> TranslatedImage:
        image = await asyncio.to_thread(self._decode, content)
        encoding = resolve_image_encoding(content, file_name, content_type)
        try:
            ocr_result = await self._ocr_service.recognize(content, segment)
        except asyncio.TimeoutError as exc:
            raise OcrTimeoutError("OCR processing timed out") from exc
        result_image, raw_items = await asyncio.to_thread(
            translate_image, image, ocr_result, language
        )
        result_bytes = await asyncio.to_thread(encode_image, result_image, encoding)
        return TranslatedImage(
            result_bytes,
            encoding.content_type,
            encoding.output_extension,
            [TranslationItem.model_validate(item) for item in raw_items],
        )

    async def translate_url(
        self,
        image_url: str,
        language: str = "en_zh",
        segment: bool = False,
        *,
        request_id: str | None = None,
    ) -> TranslationResult:
        started_at = time.monotonic()
        request_id = request_id or str(uuid4())
        content, content_type = await self._image_source.fetch(image_url)
        result, _ = await self.persist_and_translate(
            content,
            language,
            segment,
            urlsplit(image_url).path,
            content_type,
            request_id=request_id,
        )
        return result.model_copy(
            update={"duration_ms": max(0, round((time.monotonic() - started_at) * 1000))}
        )

    async def persist_and_translate(
        self,
        content,
        language="en_zh",
        segment=False,
        filename="",
        content_type=None,
        *,
        request_id=None,
    ):
        started_at = time.monotonic()
        request_id = request_id or str(uuid4())
        import os

        import httpx

        from image_translation.config import PROJECT_ROOT

        files = FileServiceClient(
            FileServiceSettings(
                _env_file=os.environ.get("IMAGE_TRANSLATION_ENV_FILE", PROJECT_ROOT / ".env")
            )
        )
        files.url()  # configuration is mandatory before source acquisition
        encoding = resolve_image_encoding(content, filename, content_type)
        async with httpx.AsyncClient() as client:
            source = await files.upload(
                client,
                f"source-{request_id}{encoding.output_extension}",
                content,
                encoding.content_type,
                {"source": "image-translation", "kind": "url-import"},
            )
            translated = await self.translate_bytes(
                content, language, segment, source["filename"], content_type
            )
            artifact = await files.upload(
                client,
                f"translated-{request_id}{translated.extension}",
                translated.content,
                translated.content_type,
                {
                    "source": "image-translation",
                    "kind": "result",
                    "source_file_id": source["file_id"],
                },
            )
        result = TranslationResult(
            request_id=request_id,
            source_url=source["file_url"],
            source_files=[source],
            translated_url=artifact["file_url"],
            files=[artifact],
            data=translated.data,
            duration_ms=max(0, round((time.monotonic() - started_at) * 1000)),
        )

        return result, translated
