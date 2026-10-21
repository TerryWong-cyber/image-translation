import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from image_translation.clients.llm_client import request_model
from image_translation.errors import ModelTimeoutError, ModelUpstreamError
from image_translation.utils.llm_translation import llm_translate


class ModelFailureTest(unittest.TestCase):
    def response(self, payload):
        response = Mock(status_code=200, text=repr(payload))
        response.json.return_value = payload
        return response

    def test_connection_and_timeout_failures_are_logged_and_raised(self):
        for failure, expected in (
            (requests.ConnectionError("connection refused"), ModelUpstreamError),
            (requests.Timeout("timed out"), ModelTimeoutError),
        ):
            with (
                self.subTest(failure=failure),
                patch("image_translation.clients.llm_client.requests.post", side_effect=failure),
                self.assertLogs("image_translation.clients.llm_client", level="ERROR") as logs,
            ):
                with self.assertRaises(expected):
                    request_model("http://model/translate", {}, 5, 1)
            self.assertIn("http://model/translate", logs.output[0])
            self.assertIn(str(failure), logs.output[0])

    def test_invalid_success_responses_abort_instead_of_returning_empty_text(self):
        for payload in (
            {"status": "error", "error": "engine unavailable"},
            {"status": "success"},
            {"status": "success", "responses": []},
            {"status": "success", "responses": ["ok", "extra"]},
            {"status": "success", "responses": "text"},
            *(
                {"status": "success", "responses": [value]}
                for value in ("", "  \n", "\\n", None, 123)
            ),
            [],
        ):
            with (
                self.subTest(payload=payload),
                patch(
                    "image_translation.clients.llm_client.requests.post",
                    return_value=self.response(payload),
                ),
                self.assertLogs("image_translation.clients.llm_client", level="ERROR"),
            ):
                with self.assertRaises(ModelUpstreamError):
                    request_model("http://model/translate", {}, 5, 1)

    def test_http_error_and_non_json_response_are_logged(self):
        for http_error in (True, False):
            response = self.response(None)
            response.text = "engine crashed"
            if http_error:
                response.status_code = 503
                response.raise_for_status.side_effect = requests.HTTPError("503", response=response)
            else:
                response.json.side_effect = ValueError("invalid JSON")
            with (
                patch("image_translation.clients.llm_client.requests.post", return_value=response),
                self.assertLogs("image_translation.clients.llm_client", level="ERROR") as logs,
            ):
                with self.assertRaises(ModelUpstreamError):
                    request_model("http://model/translate", {}, 5, 1)
            self.assertIn("engine crashed", logs.output[0])

    def test_valid_unchanged_translation_keeps_original_box(self):
        with patch(
            "image_translation.utils.llm_translation.call_inference_api_v2", return_value=["OpenAI"]
        ):
            self.assertEqual(llm_translate([7], ["OpenAI"], SimpleNamespace(), 1), ({}, [7], {}))

    def test_partial_batch_is_rejected_before_distributing_text(self):
        with patch(
            "image_translation.utils.llm_translation.call_inference_api_v2", return_value=["你好"]
        ):
            with self.assertRaises(ModelUpstreamError):
                llm_translate([1, 2], ["Hello", "World"], SimpleNamespace(), 2)

    def test_recognition_and_translation_use_the_same_failure_policy(self):
        from image_translation.utils.llm_recognition import call_llm_recognition_api_batch
        from image_translation.utils.llm_translation import call_inference_api_v2

        for call in (
            lambda: call_llm_recognition_api_batch(["base64"]),
            lambda: call_inference_api_v2("Translate", ["Hello"]),
        ):
            with (
                patch(
                    "image_translation.clients.llm_client.requests.post",
                    return_value=self.response({"status": "success", "responses": [""]}),
                ),
                self.assertLogs("image_translation.clients.llm_client", level="ERROR"),
            ):
                with self.assertRaises(ModelUpstreamError):
                    call()

    def test_model_failure_stops_before_inpainting_and_no_text_still_succeeds(self):
        import image_translation.image_translation as core
        import numpy as np

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            patch.object(core, "inpaint_image") as inpaint,
            patch.object(core, "process_ocr_res", return_value=([], {}, {}, {}, {})),
            patch.object(core, "aggregate_ocr_results", return_value=([], None)),
            patch.object(
                core, "process_and_filter_aggregated_boxes", return_value=({}, {}, {1: "Hello"}, {})
            ),
            patch(
                "image_translation.clients.llm_client.requests.post",
                side_effect=requests.ConnectionError("refused"),
            ),
            self.assertLogs("image_translation.clients.llm_client", level="ERROR"),
        ):
            with self.assertRaises(ModelUpstreamError):
                core.translate_image(image, [["detected text"]], "en_zh")
            inpaint.assert_not_called()
        with patch.object(core, "process_llm_image_translate") as translate:
            output, items = core.translate_image(image, [[]], "en_zh")
            self.assertIs(output, image)
            self.assertEqual(items, [])
            translate.assert_not_called()


class OcrModelErrorTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_model_error_survives_runtime_bridge(self):
        import pickle

        from image_translation.config import get_settings
        from image_translation.runtime.ocr_runtime import OcrRuntime

        for failure in (
            ModelUpstreamError("Model unavailable"),
            ModelTimeoutError("Model timed out"),
        ):
            runtime = OcrRuntime(get_settings().ocr)
            runtime._process = Mock()
            runtime._process.is_alive.return_value = True
            runtime._result_queue = Mock()
            runtime._build_tasks = Mock(return_value=[("task", b"image", (0, 0))])

            def deliver_result(task):
                batch = next(iter(runtime._pending.values()))
                batch.results[0] = pickle.loads(pickle.dumps(failure))
                batch.event.set()

            runtime._task_queue = Mock()
            runtime._task_queue.put.side_effect = deliver_result
            with self.assertRaises(type(failure)):
                await runtime.recognize(b"image")
            self.assertFalse(runtime._pending)


class RecognitionWorkerTest(unittest.TestCase):
    def test_language_check_propagates_model_failure(self):
        import importlib.util
        import sys
        from pathlib import Path

        import numpy as np

        # Load worker logic without installing or initializing Paddle/GPU models.
        source = (
            Path(__file__).resolve().parents[1]
            / "src/image_translation/components/ocr/ocr_worker.py"
        )
        spec = importlib.util.spec_from_file_location("worker_under_test", source)
        worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"paddle": Mock(), "paddleocr": Mock()}):
            spec.loader.exec_module(worker)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        for failure in (
            ModelUpstreamError("Model unavailable"),
            ModelTimeoutError("Model timed out"),
        ):
            with patch.object(worker, "call_llm_recognition_api_batch", side_effect=failure):
                with self.assertRaises(type(failure)):
                    worker.has_non_zh_en_text(image)
                with self.assertRaises(type(failure)):
                    worker.det_to_ocr_result(image, [[[0, 0], [7, 0], [7, 7], [0, 7]]])
