"""Fail model requests explicitly instead of treating failures as empty text."""

import logging

import requests

from image_translation.errors import ModelTimeoutError, ModelUpstreamError

logger = logging.getLogger(__name__)


def validate_responses(responses, expected_count, *, allow_empty=False):
    if not isinstance(responses, list) or len(responses) != expected_count:
        raise ModelUpstreamError("Model returned an invalid response count")
    for index, value in enumerate(responses):
        if not isinstance(value, str) or (not allow_empty and not value.replace("\\n", " ").strip()):
            raise ModelUpstreamError(f"Model returned invalid or empty text at index {index}")
    return responses


def request_model(url, data, timeout, expected_count, *, response_field="responses", allow_empty=False):
    response = None
    try:
        response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or result.get("status") != "success":
            raise ModelUpstreamError("Model reported an unsuccessful response")
        responses = result.get(response_field)
        if response_field == "response":
            responses = [responses]
        return validate_responses(responses, expected_count, allow_empty=allow_empty)
    except requests.exceptions.Timeout as exc:
        logger.exception("Model request timed out: url=%s timeout=%s", url, timeout)
        raise ModelTimeoutError("Model request timed out") from exc
    except (requests.exceptions.RequestException, ValueError, ModelUpstreamError) as exc:
        logger.exception(
            "Model request failed: url=%s status=%s response=%s",
            url,
            response.status_code if response is not None else None,
            response.text[:2000] if response is not None else None,
        )
        if isinstance(exc, ModelUpstreamError):
            raise
        raise ModelUpstreamError("Model upstream request failed") from exc
