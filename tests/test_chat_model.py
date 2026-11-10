from unittest.mock import Mock, patch

import pytest
import requests

from image_translation.clients.llm_client import request_chat_batch
from image_translation.errors import ModelTimeoutError, ModelUpstreamError
from image_translation.utils.llm_recognition import call_llm_recognition_api_batch
from image_translation.utils.llm_translation import call_inference_api_v2


def response(text="ok", finish="stop"):
    result = Mock(status_code=200)
    result.json.return_value = {"choices": [{"finish_reason": finish, "message": {"content": text}}]}
    return result


def test_translation_sends_one_non_streaming_request_per_text():
    def send(url, **kwargs):
        return response(kwargs["json"]["messages"][0]["content"])
    with patch("image_translation.clients.llm_client.requests.post", side_effect=send) as post:
        assert call_inference_api_v2("Translate: ", ["one", "two"], batch_size=2) == ["Translate: one", "Translate: two"]
    for call in post.call_args_list:
        payload = call.kwargs["json"]
        assert payload["model"] == "gemma4-26b-a4b"
        assert payload["enable_thinking"] is False and payload["stream"] is False


def test_recognition_sends_individual_images():
    with patch("image_translation.clients.llm_client.requests.post", return_value=response()) as post:
        assert call_llm_recognition_api_batch(["iVBORabc", "/9j/abc"]) == ["ok", "ok"]
    contents = [call.kwargs["json"]["messages"][0]["content"] for call in post.call_args_list]
    assert contents[0][0]["image_url"]["url"] == "data:image/png;base64,iVBORabc"
    assert contents[1][0]["image_url"]["url"] == "data:image/jpeg;base64,/9j/abc"


@pytest.mark.parametrize("payload", [response(""), response("partial", "length"), Mock(status_code=200, **{"json.return_value": {}})])
def test_rejects_bad_chat_responses(payload):
    with patch("image_translation.clients.llm_client.requests.post", return_value=payload):
        with pytest.raises(ModelUpstreamError):
            request_chat_batch("http://model", ["hello"], 1, model="gemma")


def test_timeout_and_authorization():
    with patch("image_translation.clients.llm_client.requests.post", side_effect=requests.Timeout):
        with pytest.raises(ModelTimeoutError):
            request_chat_batch("http://model", ["hello"], 1, model="gemma")
    with patch("image_translation.clients.llm_client.requests.post", return_value=response()) as post:
        request_chat_batch("http://model", ["hello"], 1, model="gemma", api_key="key")
        assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer key"}
