"""Convert PaddleOCR 3.x pipeline results to the service's stable OCR contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _result_mapping(result: Any) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(f"Unexpected PaddleOCR result type: {type(result).__name__}")
    return result


def _values(result: Mapping[str, Any], key: str) -> list[Any]:
    value = result.get(key)
    if value is None:
        return []
    return list(value)


def _box_to_list(box: Any) -> list[list[float]]:
    array = np.asarray(box)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 4:
        raise ValueError(f"Invalid PaddleOCR polygon shape: {array.shape}")
    return array.astype(float).tolist()


def paddleocr_results_to_legacy(results: Iterable[Any]) -> list[list[Any]]:
    """Return ``[[box, (text, score)], ...]`` wrapped as one image result."""
    lines: list[Any] = []
    for raw_result in results:
        result = _result_mapping(raw_result)
        boxes = _values(result, "rec_polys")
        texts = _values(result, "rec_texts")
        scores = _values(result, "rec_scores")
        if not (len(boxes) == len(texts) == len(scores)):
            raise ValueError(
                "PaddleOCR result fields rec_polys, rec_texts, and rec_scores "
                "must have the same length"
            )
        lines.extend(
            [_box_to_list(box), (str(text), float(score))]
            for box, text, score in zip(boxes, texts, scores)
        )
    return [lines]


def paddleocr_detection_boxes(results: Iterable[Any]) -> list[list[list[float]]]:
    """Extract all detected polygons, including those not retained by recognition."""
    boxes: list[list[list[float]]] = []
    for raw_result in results:
        result = _result_mapping(raw_result)
        boxes.extend(_box_to_list(box) for box in _values(result, "dt_polys"))
    return boxes
