import asyncio
import json
import unittest
from types import SimpleNamespace

from image_translation.api.main import health_live, health_ready, settings


class HealthEndpointTest(unittest.TestCase):
    def test_liveness_is_process_only(self):
        self.assertEqual(asyncio.run(health_live()), {"status": "ok"})

    def test_readiness_rejects_missing_runtime(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

        response = asyncio.run(health_ready(request))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body), {"status": "not_ready"})

    def test_readiness_requires_live_ocr_worker(self):
        state = SimpleNamespace(
            ocr_runtime=SimpleNamespace(is_started=False),
            translation_service=object(),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        response = asyncio.run(health_ready(request))

        self.assertEqual(response.status_code, 503)

    def test_readiness_reports_ready_service(self):
        state = SimpleNamespace(
            ocr_runtime=SimpleNamespace(is_started=True),
            translation_service=object(),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        response = asyncio.run(health_ready(request))

        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["ocr_version"], settings.ocr.version)


if __name__ == "__main__":
    unittest.main()
