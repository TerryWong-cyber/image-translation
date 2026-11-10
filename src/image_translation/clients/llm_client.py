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


def request_chat_batch(url, contents, timeout, *, model, api_key="", max_tokens=4096, batch_size=1):
    """Send independent chat requests concurrently while preserving input order."""
    from concurrent.futures import ThreadPoolExecutor

    def request_one(content):
        response = None
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = requests.post(url, json={
                "model": model, "messages": [{"role": "user", "content": content}],
                "stream": False, "enable_thinking": False, "max_tokens": max_tokens,
                "temperature": 0, "presence_penalty": 0,
            }, headers=headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices") if isinstance(result, dict) else None
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ModelUpstreamError("Model returned invalid choices")
            choice = choices[0]
            if choice.get("finish_reason") != "stop" or not isinstance(choice.get("message"), dict):
                raise ModelUpstreamError("Model response was incomplete or invalid")
            return validate_responses([choice["message"].get("content")], 1)[0]
        except requests.exceptions.Timeout as exc:
            logger.exception("Model request timed out: url=%s timeout=%s", url, timeout)
            raise ModelTimeoutError("Model request timed out") from exc
        except (requests.exceptions.RequestException, ValueError, ModelUpstreamError) as exc:
            logger.exception("Model request failed: url=%s status=%s", url,
                             response.status_code if response is not None else None)
            if isinstance(exc, ModelUpstreamError):
                raise
            raise ModelUpstreamError("Model upstream request failed") from exc

    if not contents:
        return []
    with ThreadPoolExecutor(max_workers=min(batch_size, len(contents))) as executor:
        return list(executor.map(request_one, contents))
