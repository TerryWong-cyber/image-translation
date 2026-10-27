"""Browser asynchronous adapter; existing synchronous REST/MCP contracts remain valid."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from image_translation.api.web import WebTranslationRequest
from image_translation.clients.file_service_client import FileServiceClient
from image_translation.clients.tracking import tracking
from image_translation.contracts import TranslationResult, UrlTranslationRequest

logger = logging.getLogger(__name__)


def create_task_router(translate_handler):
    jobs = {}
    running = set()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            # Shutdown explicitly interrupts in-flight work; browser disconnects do not.
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    router = APIRouter(prefix="/api/web/images/tasks", lifespan=lifespan)

    @router.post("", status_code=202)
    async def submit(body: WebTranslationRequest, request: Request):
        if len(running) >= 16:
            raise HTTPException(429, "too many image translation tasks")
        now = time.monotonic()
        for key in list(jobs):
            if jobs[key]["status"] in {"succeeded", "failed", "cancelled"} and (now - jobs[key]["created"] > 3600 or len(jobs) >= 256):
                del jobs[key]
        native_id = str(uuid4())
        job = {"id": native_id, "task_id": tracking.identifier(native_id) if tracking.enabled else None,
               "status": "queued", "result": None, "error": None, "created": now}
        jobs[native_id] = job
        await asyncio.to_thread(tracking.start, native_id, "图片翻译", queued=True)

        async def execute():
            job["status"] = "running"
            await asyncio.to_thread(tracking.write, native_id, "running")
            try:
                payload = body.model_dump(exclude={"file_id"})
                if body.file_id is not None:
                    async with httpx.AsyncClient() as client:
                        artifact = await FileServiceClient().resolve(client, str(body.file_id))
                    payload["image_url"] = artifact["file_url"]
                with tracking.managed(native_id):
                    result = await translate_handler(UrlTranslationRequest.model_validate(payload), request)
                # The real HTTP handler returns JSONResponse for both success and errors.
                if isinstance(result, JSONResponse):
                    if not 200 <= result.status_code < 300:
                        logger.warning(
                            "image_translation_task_handler_failed task_id=%s status=%s",
                            native_id, result.status_code,
                        )
                        raise RuntimeError("image translation handler failed")
                    result = TranslationResult.model_validate_json(result.body)
                if not hasattr(result, "model_dump"):
                    raise RuntimeError("image translation handler returned an invalid result")
                job["result"] = result.model_dump(mode="json")
                job["status"] = "succeeded"
                await asyncio.to_thread(tracking.finish, native_id, job["result"])
            except asyncio.CancelledError:
                job["status"] = "cancelled"
                await asyncio.to_thread(tracking.fail, native_id, cancelled=True)
                raise
            except Exception as exc:  # noqa: BLE001 - persist a terminal state for any failure
                logger.error(
                    "image_translation_task_failed task_id=%s error_type=%s",
                    native_id, type(exc).__name__,
                )
                job["status"] = "failed"
                job["error"] = "图片翻译失败，请检查文件或服务状态后重试。"
                await asyncio.to_thread(tracking.fail, native_id)
        task = asyncio.create_task(execute(), name="image-translation-" + native_id)
        running.add(task)
        task.add_done_callback(running.discard)
        return {key: value for key, value in job.items() if key != "created"}

    @router.get("/{native_id}")
    async def get_task(native_id: str):
        job = jobs.get(native_id)
        if job is None:
            raise HTTPException(404, "image task expired or service restarted; use the central task list")
        return {key: value for key, value in job.items() if key != "created"}

    return router
