"""Resolve explicit PaddleOCR device settings against the installed runtime."""

from __future__ import annotations

from typing import Any

from image_translation.config import OcrSettings


def paddleocr_device_kwargs(settings: OcrSettings, paddle: Any) -> dict[str, bool | int]:
    """Build PaddleOCR device arguments and fail fast for unusable GPU settings."""
    if settings.device == "cpu":
        return {"use_gpu": False}

    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError(
            "OCR_DEVICE=gpu was requested, but the installed PaddlePaddle runtime "
            "was not compiled with CUDA. Install a CUDA-compatible paddlepaddle-gpu build "
            "or set OCR_DEVICE=cpu."
        )

    gpu_count = paddle.device.cuda.device_count()
    if settings.gpu_id >= gpu_count:
        raise RuntimeError(
            f"OCR_GPU_ID={settings.gpu_id} is unavailable; PaddlePaddle detected "
            f"{gpu_count} CUDA device(s)."
        )

    return {"use_gpu": True, "gpu_id": settings.gpu_id}
