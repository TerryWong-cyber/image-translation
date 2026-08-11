"""Build PaddleOCR 3.x initialization settings for the selected device."""

from __future__ import annotations

from typing import Any

from image_translation.config import OcrSettings


def paddleocr_device(settings: OcrSettings, paddle: Any) -> str:
    """Resolve a PaddleOCR 3.x device string and reject unusable GPU settings."""
    if settings.device == "cpu":
        return "cpu"

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

    return f"gpu:{settings.gpu_id}"


def paddleocr_init_kwargs(settings: OcrSettings, paddle: Any) -> dict[str, Any]:
    """Return the explicit PaddleOCR 3.x pipeline configuration."""
    return {
        "lang": settings.language,
        "ocr_version": settings.version,
        "device": paddleocr_device(settings, paddle),
        "engine": "paddle",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": settings.use_angle_classifier,
        "text_det_unclip_ratio": settings.unclip_ratio,
    }
