"""Lifecycle-managed bridge to the multiprocessing OCR worker."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from queue import Empty
from typing import Any
from uuid import uuid4

from image_translation.config import OcrSettings
from image_translation.errors import OcrProcessingError
from image_translation.utils.image_segmentation import adjust_box_heights, segment_image


logger = logging.getLogger(__name__)


def _run_ocr_worker(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Import PaddleOCR only inside the spawned worker process."""
    from image_translation.components.ocr.ocr_worker import ocr_worker_process

    ocr_worker_process(task_queue, result_queue)


@dataclass
class _PendingBatch:
    event: asyncio.Event
    expected_count: int
    results: dict[int, Any] = field(default_factory=dict)


class OcrRuntime:
    """Own one OCR subprocess and multiplex requests over its queues."""

    def __init__(self, settings: OcrSettings):
        self._settings = settings
        self._task_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._process: mp.Process | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._pending: dict[str, _PendingBatch] = {}
        self._pending_lock = asyncio.Lock()
        self._closing = asyncio.Event()

    @property
    def is_started(self) -> bool:
        return self._process is not None and self._process.is_alive()

    async def start(self) -> None:
        if self.is_started:
            return

        self._closing.clear()
        context = mp.get_context("spawn")
        self._task_queue = context.Queue()
        self._result_queue = context.Queue()
        self._process = context.Process(
            target=_run_ocr_worker,
            args=(self._task_queue, self._result_queue),
        )
        self._process.start()

        try:
            init_status, init_error = await asyncio.wait_for(
                asyncio.to_thread(self._result_queue.get),
                timeout=self._settings.task_timeout_seconds,
            )
        except BaseException:
            await self.close()
            raise

        if init_status != "INIT_OK":
            await self.close()
            raise RuntimeError(f"OCR worker initialization failed: {init_error}")

        self._listener_task = asyncio.create_task(
            self._result_listener(),
            name="image-translation-ocr-result-listener",
        )
        logger.info("OCR worker is ready")

    async def close(self) -> None:
        self._closing.set()

        if self._task_queue is not None and self._process is not None and self._process.is_alive():
            self._task_queue.put((None, None))

        if self._listener_task is not None:
            try:
                await asyncio.wait_for(self._listener_task, timeout=1.5)
            except asyncio.TimeoutError:
                self._listener_task.cancel()
                await asyncio.gather(self._listener_task, return_exceptions=True)
            self._listener_task = None

        if self._process is not None:
            await asyncio.to_thread(
                self._process.join,
                self._settings.worker_shutdown_timeout_seconds,
            )
            if self._process.is_alive():
                logger.warning("OCR worker did not terminate gracefully; terminating it")
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 1)
            self._process = None

        for queue in (self._task_queue, self._result_queue):
            if queue is not None:
                queue.cancel_join_thread()
                queue.close()
        self._task_queue = None
        self._result_queue = None

        async with self._pending_lock:
            for batch in self._pending.values():
                batch.event.set()
            self._pending.clear()

    async def recognize(self, image_content: bytes, segment_enabled: bool = False) -> list:
        if not self.is_started or self._task_queue is None:
            raise RuntimeError("OCR runtime is not started")

        request_id = str(uuid4())
        tasks = self._build_tasks(request_id, image_content, segment_enabled)
        if not tasks:
            logger.warning("Image segmentation produced no OCR tasks for %s", request_id)
            return []

        batch = _PendingBatch(event=asyncio.Event(), expected_count=len(tasks))
        async with self._pending_lock:
            self._pending[request_id] = batch

        for subtask_id, segment_bytes, _ in tasks:
            self._task_queue.put((subtask_id, segment_bytes))

        try:
            await asyncio.wait_for(
                batch.event.wait(),
                timeout=self._settings.task_timeout_seconds,
            )
            processed_results = [
                (batch.results[index], offset)
                for index, (_, _, offset) in enumerate(tasks)
            ]
        finally:
            async with self._pending_lock:
                self._pending.pop(request_id, None)

        for result, _ in processed_results:
            if isinstance(result, Exception):
                raise OcrProcessingError(str(result)) from result

        combined_result = []
        for result, (x_offset, y_offset) in processed_results:
            if not result or not result[0]:
                continue
            for line_result in result[0]:
                if not line_result:
                    continue
                box, (text, score) = line_result
                adjusted_box = [
                    [point[0] + x_offset, point[1] + y_offset]
                    for point in box
                ]
                combined_result.append([adjusted_box, (text, score)])

        return adjust_box_heights(
            [combined_result],
            self._settings.box_height_tolerance_ratio,
        )

    def _build_tasks(
        self,
        request_id: str,
        image_content: bytes,
        segment_enabled: bool,
    ) -> list[tuple[str, bytes, tuple[int, int]]]:
        if not segment_enabled:
            return [(f"{request_id}_0", image_content, (0, 0))]

        return [
            (
                f"{request_id}_{index}",
                segment.image_data,
                (segment.x_offset, segment.y_offset),
            )
            for index, segment in enumerate(segment_image(image_content))
        ]

    async def _result_listener(self) -> None:
        if self._result_queue is None:
            return

        while not self._closing.is_set():
            try:
                task_id, result = await asyncio.to_thread(
                    self._result_queue.get,
                    True,
                    0.5,
                )
            except Empty:
                continue
            except (EOFError, OSError):
                if not self._closing.is_set():
                    logger.exception("OCR result queue closed unexpectedly")
                return

            request_id, separator, index_text = str(task_id).rpartition("_")
            if not separator or not index_text.isdigit():
                logger.warning("Ignoring malformed OCR task id: %r", task_id)
                continue

            async with self._pending_lock:
                batch = self._pending.get(request_id)
                if batch is None:
                    logger.warning("Ignoring result for expired OCR request %s", request_id)
                    continue
                batch.results[int(index_text)] = result
                if len(batch.results) == batch.expected_count:
                    batch.event.set()
