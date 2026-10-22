"""Browser adapters for the static translation workbench; File Service secrets stay here."""

from io import BytesIO
from pathlib import PurePosixPath
from typing import Annotated
from uuid import UUID

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from image_translation.clients.file_service_client import (
    FileServiceClient,
    FileServiceError,
)
from image_translation.contracts import TranslationResult, UrlTranslationRequest

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
IMAGE_TYPES = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}


class WebTranslationRequest(UrlTranslationRequest):
    file_id: UUID | None = None


def file_error(exc: FileServiceError):
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": exc.code,
            **({"file_id": exc.file_id} if hasattr(exc, "file_id") else {}),
        },
        headers={"Cache-Control": "no-store"},
    )


def create_web_router(translate_handler):
    router = APIRouter(prefix="/api/web/images", tags=["translation-workbench"])

    @router.post("/upload")
    async def upload_image(file: Annotated[UploadFile, File()]):
        try:
            filename = PurePosixPath((file.filename or "image").replace("\\", "/")).name
            filename = "".join(c for c in filename if ord(c) >= 32 and ord(c) != 127)[
                :255
            ]
            expected = IMAGE_TYPES.get(filename.rsplit(".", 1)[-1].lower())
            if not expected:
                raise HTTPException(400, "仅支持 JPG、PNG、WEBP 图片。")
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if not content or len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "图片大小须介于 1 B 到 20 MB。")
            try:
                with Image.open(BytesIO(content)) as image:
                    if image.format != expected:
                        raise HTTPException(400, "图片内容与文件格式不符。")
                    image.verify()
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as exc:
                raise HTTPException(400, "图片内容无效或无法读取。") from exc
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[
                expected
            ]
            async with httpx.AsyncClient() as client:
                artifact = await FileServiceClient().upload(
                    client,
                    filename,
                    content,
                    mime,
                    {"source": "translation-web", "kind": "image"},
                )
            return JSONResponse(
                {**artifact, "image_url": artifact["file_url"], "files": [artifact]},
                headers={"Cache-Control": "no-store"},
            )
        except FileServiceError as exc:
            return file_error(exc)
        finally:
            await file.close()

    @router.post("/translate", response_model=TranslationResult)
    async def translate_image(body: WebTranslationRequest, request: Request):
        payload = body.model_dump(exclude={"file_id"})
        try:
            if body.file_id is not None:
                async with httpx.AsyncClient() as client:
                    artifact = await FileServiceClient().resolve(
                        client, str(body.file_id)
                    )
                payload["image_url"] = artifact["file_url"]
            # Reuse the shared REST/MCP service and its failure mapping, without loopback HTTP.
            return await translate_handler(
                UrlTranslationRequest.model_validate(payload), request
            )
        except FileServiceError as exc:
            return file_error(exc)

    @router.get("/files/{file_id}/content")
    async def file_content(file_id: UUID, request: Request, download: bool = False):
        if request.headers.get("range"):
            raise HTTPException(400, "range_not_supported")
        try:
            return await FileServiceClient().content_response(str(file_id), download)
        except FileServiceError as exc:
            return file_error(exc)

    return router
