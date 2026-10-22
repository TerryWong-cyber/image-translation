from io import BytesIO
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from image_translation.api.web import create_web_router
from image_translation.clients.file_service_client import (
    FileServiceClient,
    FileServiceError,
)


def png():
    stream = BytesIO()
    Image.new("RGB", (20, 10), "red").save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def web_client():
    translate = AsyncMock(
        return_value={
            "status": "success",
            "request_id": "test",
            "source_url": "https://files.test/a",
            "translated_url": "https://files.test/b",
            "files": [],
            "source_files": [],
            "data": [],
            "duration_ms": 1,
        }
    )
    app = FastAPI()
    app.include_router(create_web_router(translate))
    with TestClient(app) as client:
        yield client, translate


def test_upload_translation_and_preview(web_client, file_service):
    client, translate = web_client
    response = client.post(
        "/api/web/images/upload", files={"file": ("photo.png", png(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_url"] == data["file_url"]
    assert data["files"][0]["file_id"] == data["file_id"]
    file_id = data["file_id"]
    assert file_service.contents[file_id] == png()
    # Use file_id for a fresh signed URL rather than the now-expired UI input URL.
    response = client.post(
        "/api/web/images/translate",
        json={
            "file_id": file_id,
            "image_url": "https://expired.example/a?token=old",
            "language": "any_en",
        },
    )
    assert response.status_code == 200
    body = translate.call_args.args[0]
    assert str(body.image_url) == data["file_url"]
    assert body.language == "any_en"
    assert not hasattr(body, "file_id")
    preview = client.get(f"/api/web/images/files/{file_id}/content")
    assert preview.status_code == 200
    assert preview.content == png()
    assert "no-store" in preview.headers["cache-control"]
    file_service.revoked.add(file_id)
    assert client.get(f"/api/web/images/files/{file_id}/content").status_code == 404


def test_direct_url_and_web_request_validation(web_client):
    client, translate = web_client
    url = "https://files.test/a.png?token=a%2Fb%2Bc"
    assert (
        client.post("/api/web/images/translate", json={"image_url": url}).status_code
        == 200
    )
    assert str(translate.call_args.args[0].image_url) == url
    for payload in (
        {"image_url": url, "bucket": "legacy"},
        {"image_url": url, "file_id": "invalid"},
        {"image_url": "file:///tmp/image.png"},
        {"image_url": url, "language": "invalid"},
    ):
        assert client.post("/api/web/images/translate", json=payload).status_code == 422


def test_bad_uploads_fail_before_file_service(web_client, file_service):
    client, _ = web_client
    for name, data, status in (
        ("empty.png", b"", 413),
        ("fake.png", b"not an image", 400),
        ("wrong.jpg", png(), 400),
        ("a.svg", b"<svg/>", 400),
        ("large.png", b"x" * (20 * 1024 * 1024 + 1), 413),
    ):
        response = client.post("/api/web/images/upload", files={"file": (name, data)})
        assert response.status_code == status, response.text
    assert not file_service.calls
    assert (
        client.get(
            f"/api/web/images/files/{uuid4()}/content", headers={"Range": "bytes=0-1"}
        ).status_code
        == 400
    )


def test_upload_failure_retains_id_and_never_retries(web_client, monkeypatch):
    client, _ = web_client
    file_id = str(uuid4())
    calls = []

    async def fail(self, *args, **kwargs):
        calls.append(1)
        error = FileServiceError(502, "signing_unavailable")
        error.file_id = file_id
        raise error

    monkeypatch.setattr(FileServiceClient, "upload", fail)
    response = client.post(
        "/api/web/images/upload", files={"file": ("photo.png", png())}
    )
    assert response.status_code == 502
    assert response.json() == {"error": "signing_unavailable", "file_id": file_id}
    assert len(calls) == 1


def test_file_service_credentials_remain_on_backend(web_client, monkeypatch):
    monkeypatch.setenv("FILE_SERVICE_TOKEN", "backend-only-secret")
    # Restore the native method beneath the autouse fixture to verify real HTTP headers.
    original = httpx.AsyncClient
    from image_translation.clients import file_service_client as module

    original_request = module.FileServiceClient.request
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer backend-only-secret"
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    import asyncio

    async def check():
        async with httpx.AsyncClient() as http:
            result = await original_request(FileServiceClient(), http, "GET")
            assert result == {"ok": True}

    asyncio.run(check())
    assert len(calls) == 1
