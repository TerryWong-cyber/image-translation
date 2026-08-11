"""FastAPI host exposing both the legacy REST API and MCP."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ValidationError

from image_translation.clients.oss_client import oss_client_instance
from image_translation.config import get_settings
from image_translation.contracts import TranslationCommand
from image_translation.errors import (
    ImageTranslationError,
    InvalidImageError,
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
    limits = httpx.Limits(
        max_connections=settings.http.max_connections,
        max_keepalive_connections=settings.http.max_keepalive_connections,
        keepalive_expiry=settings.http.keepalive_expiry_seconds,
    )
    shared_client = httpx.AsyncClient(
        timeout=settings.http.request_timeout_seconds,
        limits=limits,
    )
    oss_client_instance.set_shared_client(shared_client)

    try:
        await ocr_runtime.start()
        app.state.ocr_runtime = ocr_runtime
        translation_service = ImageTranslationService(
            oss_client=oss_client_instance,
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
        await oss_client_instance.close_shared_client()
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


@app.post(settings.api.image_translate_path)
async def process_image_translate(request: Request):
    """Legacy REST adapter; the successful response shape remains unchanged."""

    request_data = await request.json()
    image_key = request_data.get("image_url")
    bucket = request_data.get("bucket")
    if not all([image_key, bucket]):
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'image_url' or 'bucket'"},
        )

    try:
        command = TranslationCommand(
            bucket=bucket,
            image_key=image_key,
            language=request_data.get("language", "en_zh"),
            save_bucket=request_data.get("save_bucket") or bucket,
            output_key=request_data.get("output_key"),
            segment=request_data.get("segment", False),
        )
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": exc.errors(include_url=False)[0]["msg"]},
        )

    service: ImageTranslationService = request.app.state.translation_service
    try:
        result = await service.translate(command)
    except OcrTimeoutError:
        logger.error("OCR processing timed out for image_url: %s", image_key)
        return JSONResponse(
            status_code=504,
            content={"error": "OCR processing timed out"},
        )
    except InvalidImageError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except ImageTranslationError as exc:
        logger.exception("Image translation failed for %s", image_key)
        return JSONResponse(status_code=500, content={"error": str(exc)})
    except Exception:
        logger.exception("Unexpected image translation failure for %s", image_key)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during image translation"},
        )

    return JSONResponse(status_code=200, content=result.to_rest_payload())


if mcp_app is not None:
    # The catch-all mount must stay after every FastAPI route. The MCP sub-app
    # owns MCP_PATH itself, which avoids a POST redirect from /mcp to /mcp/.
    app.mount("/", mcp_app, name="mcp")


if __name__ == "__main__":
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)
