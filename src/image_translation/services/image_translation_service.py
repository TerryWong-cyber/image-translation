"""Shared orchestration used by REST and MCP transports."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol
from uuid import uuid4

import cv2
import numpy as np

from image_translation.contracts import (
    TranslationCommand,
    TranslationItem,
    TranslationResult,
)
from image_translation.errors import InvalidImageError, OcrTimeoutError
from image_translation.image_translation import translate_image
from image_translation.utils.image_encoding import (
    encode_image,
    replace_image_extension,
    resolve_image_encoding,
)


class OssClient(Protocol):
    async def get_oss_file(
        self,
        bucket_name: str,
        file_key: str,
        as_string: bool = True,
    ) -> dict: ...

    async def upload_image(
        self,
        bucket_name: str,
        key: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> dict: ...


class OcrService(Protocol):
    async def recognize(self, image_content: bytes, segment_enabled: bool = False) -> list: ...


class ImageTranslationService:
    """Download, OCR, translate, render, and upload one image."""

    def __init__(self, oss_client: OssClient, ocr_service: OcrService):
        self._oss_client = oss_client
        self._ocr_service = ocr_service

    async def translate(self, command: TranslationCommand) -> TranslationResult:
        started_at = time.monotonic()
        request_id = str(uuid4())

        image_info = await self._oss_client.get_oss_file(
            command.bucket,
            command.image_key,
            as_string=False,
        )
        image_content = image_info["content"]
        image_encoding = resolve_image_encoding(
            image_content,
            command.image_key,
            image_info.get("content_type"),
        )

        try:
            ocr_result = await self._ocr_service.recognize(
                image_content,
                command.segment,
            )
        except asyncio.TimeoutError as exc:
            raise OcrTimeoutError("OCR processing timed out") from exc

        image_array = np.frombuffer(image_content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("The object does not contain a decodable image")

        result_image, raw_items = await asyncio.to_thread(
            translate_image,
            image,
            ocr_result,
            command.language,
        )
        result_bytes = await asyncio.to_thread(
            encode_image,
            result_image,
            image_encoding,
        )

        save_bucket = command.save_bucket or command.bucket
        default_key = f"{command.language}_translated_{command.image_key}"
        output_key = replace_image_extension(
            command.output_key or default_key,
            image_encoding.output_extension,
        )
        await self._oss_client.upload_image(
            save_bucket,
            output_key,
            result_bytes,
            content_type=image_encoding.content_type,
        )

        return TranslationResult(
            request_id=request_id,
            source_url=f"{command.bucket}/{command.image_key}",
            translated_url=f"{save_bucket}/{output_key}",
            data=[TranslationItem.model_validate(item) for item in raw_items],
            duration_ms=max(0, round((time.monotonic() - started_at) * 1000)),
        )
