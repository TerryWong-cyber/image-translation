import unittest

import httpx

from image_translation.clients.oss_client import OSSClient
from image_translation.config import OssSettings


class OssClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_object_collision_is_not_reported_as_success(self):
        settings = OssSettings("http://storage.test", "/upload", "/file/{bucket}/{key}")
        oss = OSSClient(settings, 5)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(400, text="Key already exists")
        )) as client:
            oss.set_shared_client(client)
            with self.assertRaises(httpx.HTTPStatusError):
                await oss.upload_image("results", "translations/id.png", b"png", "image/png")
