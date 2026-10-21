"""FastAPI host exposing URL and binary translation APIs plus MCP."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings

from image_translation.clients.file_service_client import FileServiceError
from image_translation.config import get_settings
from image_translation.contracts import (
    TranslationLanguage,
    TranslationResult,
    UrlTranslationRequest,
)
from image_translation.errors import (
    ImageDownloadError,
    ImageTooLargeError,
    InvalidImageError,
    ModelTimeoutError,
    ModelUpstreamError,
    OcrTimeoutError,
)
from image_translation.mcp import TranslationServiceRegistry, create_mcp_server
from image_translation.runtime import OcrRuntime
from image_translation.services import ImageTranslationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

service_registry = TranslationServiceRegistry()
mcp_server = create_mcp_server(
    service_registry,
    max_concurrency=settings.mcp.max_concurrency,
)

mcp_app = None
if settings.mcp.enabled:
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(settings.mcp.allowed_hosts),
        allowed_origins=list(settings.mcp.allowed_origins),
    )
    mcp_app = mcp_server.streamable_http_app(
        streamable_http_path=settings.mcp.path,
        json_response=settings.mcp.json_response,
        stateless_http=settings.mcp.stateless_http,
        max_request_body_size=settings.mcp.max_request_body_size,
        transport_security=transport_security,
        host=settings.server.host,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own every resource shared by REST and MCP requests."""

    logger.info("Starting image translation service")
    ocr_runtime = OcrRuntime(settings.ocr)
    try:
        await ocr_runtime.start()
        app.state.ocr_runtime = ocr_runtime
        translation_service = ImageTranslationService(
            ocr_service=ocr_runtime,
        )
        app.state.translation_service = translation_service
        service_registry.set(translation_service)

        if settings.mcp.enabled:
            async with mcp_server.session_manager.run():
                yield
        else:
            yield
    finally:
        service_registry.clear()
        if hasattr(app.state, "translation_service"):
            del app.state.translation_service
        if hasattr(app.state, "ocr_runtime"):
            del app.state.ocr_runtime
        await ocr_runtime.close()
        logger.info("Image translation service stopped")


app = FastAPI(lifespan=lifespan)


@app.get("/health/live", include_in_schema=False)
async def health_live():
    """Process liveness endpoint for container supervision."""
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
async def health_ready(request: Request):
    """Report ready only while the shared service and OCR worker are alive."""
    runtime = getattr(request.app.state, "ocr_runtime", None)
    service = getattr(request.app.state, "translation_service", None)
    if runtime is None or service is None or not runtime.is_started:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {
        "status": "ready",
        "ocr_device": settings.ocr.device,
        "ocr_gpu_id": settings.ocr.gpu_id if settings.ocr.device == "gpu" else None,
        "ocr_version": settings.ocr.version,
    }


def translation_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, FileServiceError):
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": exc.code,
                **({"file_id": exc.file_id} if hasattr(exc, "file_id") else {}),
            },
        )
    if isinstance(exc, ImageTooLargeError):
        code = 413
    elif isinstance(exc, InvalidImageError):
        code = 400
    elif isinstance(
        exc, (ModelTimeoutError, OcrTimeoutError, TimeoutError, httpx.TimeoutException)
    ):
        code = 504
    elif isinstance(exc, (ModelUpstreamError, ImageDownloadError, httpx.HTTPError)):
        code = 502
    else:
        logger.exception("Image translation failed")
        return JSONResponse(status_code=500, content={"error": "Image translation failed"})
    message = "Upstream service request failed" if isinstance(exc, httpx.HTTPError) else str(exc)
    return JSONResponse(status_code=code, content={"error": message})


@app.post(settings.api.image_translate_path, response_model=TranslationResult)
async def process_image_translate(body: UrlTranslationRequest, request: Request):
    """Synchronously translate a complete URL and persist to the default bucket."""
    service: ImageTranslationService = request.app.state.translation_service
    request_id = str(uuid4())
    try:
        result = await service.translate_url(
            str(body.image_url), body.language, body.segment, request_id=request_id
        )
        return JSONResponse(
            status_code=200,
            content=result.model_dump(mode="json"),
            headers={"X-Request-ID": request_id},
        )
    except Exception as exc:
        response = translation_error(exc)
        response.headers["X-Request-ID"] = request_id
        return response


async def process_binary_translate(
    request: Request,
    language: TranslationLanguage = Query(default="en_zh"),
    segment: bool = Query(default=False),
):
    """Internal raw-image endpoint. Authentication precedes reading the body."""
    if not secrets.compare_digest(
        request.headers.get("authorization", "").encode(),
        f"Bearer {settings.api.binary_token}".encode(),
    ):
        raise HTTPException(status_code=401, detail="Invalid internal API token")
    media_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if not media_type.startswith("image/") and media_type != "application/octet-stream":
        raise HTTPException(
            status_code=415, detail="Send raw image bytes with an image Content-Type"
        )
    content = bytearray()
    request_id = str(uuid4())
    try:
        async for chunk in request.stream():
            if len(content) + len(chunk) > settings.api.max_image_bytes:
                raise ImageTooLargeError("Image exceeds the configured byte limit")
            content.extend(chunk)
        service: ImageTranslationService = request.app.state.translation_service
        result, _ = await service.persist_and_translate(
            bytes(content), language, segment, content_type=media_type, request_id=request_id
        )
        from image_translation.clients.file_service_client import FileServiceClient

        response = await FileServiceClient().content_response(result.files[0]["file_id"])
        response.headers["X-Request-ID"] = request_id
        response.headers["X-File-ID"] = result.files[0]["file_id"]
        response.headers["X-Source-File-ID"] = result.source_files[0]["file_id"]
        return response
    except Exception as exc:
        response = translation_error(exc)
        response.headers["X-Request-ID"] = request_id
        return response


if settings.api.binary_enabled:
    app.add_api_route(
        "/internal/image-translate",
        process_binary_translate,
        methods=["POST"],
        include_in_schema=False,
    )


if mcp_app is not None:
    # The catch-all mount must stay after every FastAPI route. The MCP sub-app
    # owns MCP_PATH itself, which avoids a POST redirect from /mcp to /mcp/.
    app.mount("/", mcp_app, name="mcp")


if __name__ == "__main__":
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)
