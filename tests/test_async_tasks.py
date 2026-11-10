import asyncio
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from image_translation.api.tasks import create_task_router
from image_translation.contracts import TranslationResult


def test_async_submission_finishes_after_the_submission_response_closes():
    async def translate(body, request):
        await asyncio.sleep(.05)
        return TranslationResult(status="success", request_id="request", source_url="https://files.test/a", translated_url="https://files.test/b",
                                 source_files=[], files=[], data=[], duration_ms=1)
    app = FastAPI()
    app.include_router(create_task_router(translate))
    with TestClient(app) as client:
        accepted = client.post("/api/web/images/tasks", json={"image_url": "https://images.test/a.png", "language": "any_en"})
        assert accepted.status_code == 202
        identifier = accepted.json()["id"]
        assert accepted.json()["status"] in {"queued", "running"}
        for _ in range(100):
            job = client.get(f"/api/web/images/tasks/{identifier}").json()
            if job["status"] == "succeeded": break
            time.sleep(.01)
        assert job["result"]["request_id"] == "request"


def test_async_errors_are_terminal_and_do_not_expose_exception_details():
    async def translate(body, request): raise RuntimeError("secret-upstream-key")
    app = FastAPI()
    app.include_router(create_task_router(translate))
    with TestClient(app) as client:
        identifier = client.post("/api/web/images/tasks", json={"image_url": "https://images.test/a.png"}).json()["id"]
        for _ in range(100):
            job = client.get(f"/api/web/images/tasks/{identifier}").json()
            if job["status"] == "failed": break
            time.sleep(.01)
        assert job["status"] == "failed" and "secret" not in str(job)


def test_async_accepts_the_actual_success_json_response_contract(monkeypatch):
    async def resolve(self, client, file_id):
        assert file_id == "c88c8683-cf20-4757-81a5-014f3ba695aa"
        return {"file_url": "https://files.test/renewed"}

    monkeypatch.setattr("image_translation.api.tasks.FileServiceClient.resolve", resolve)

    async def translate(body, request):
        assert str(body.image_url) == "https://files.test/renewed"
        result = TranslationResult(status="success", request_id="json-response", source_url="https://files.test/a", translated_url="https://files.test/b", source_files=[], files=[], data=[], duration_ms=1)
        return JSONResponse(status_code=200, content=result.model_dump(mode="json"))

    app = FastAPI()
    app.include_router(create_task_router(translate))
    with TestClient(app) as client:
        response = client.post("/api/web/images/tasks", json={
            "image_url": "https://images.test/a.png",
            "file_id": "c88c8683-cf20-4757-81a5-014f3ba695aa",
        })
        assert response.status_code == 202
        identifier = response.json()["id"]
        for _ in range(100):
            job = client.get(f"/api/web/images/tasks/{identifier}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(.01)
        assert job["status"] == "succeeded"
        assert job["result"]["request_id"] == "json-response"


def test_async_rejects_failed_json_response_without_exposing_details():
    async def translate(body, request):
        return JSONResponse(status_code=502, content={"error": "secret-upstream-key"})

    app = FastAPI()
    app.include_router(create_task_router(translate))
    with TestClient(app) as client:
        response = client.post("/api/web/images/tasks", json={"image_url": "https://images.test/a.png"})
        assert response.status_code == 202
        identifier = response.json()["id"]
        for _ in range(100):
            job = client.get(f"/api/web/images/tasks/{identifier}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(.01)
        assert job["status"] == "failed"
        assert "secret-upstream-key" not in str(job)


def test_delegated_async_job_retains_owner_and_blocks_other_users():
    from image_translation.clients.service_identity import UserContextMiddleware, current_user_id
    seen=[]
    async def translate(body, request):
        await asyncio.sleep(.01)
        seen.append(current_user_id())
        return TranslationResult(status="success", request_id="scoped",source_url="https://files.test/a",translated_url="https://files.test/b",source_files=[],files=[],data=[],duration_ms=1)
    app=FastAPI()
    app.add_middleware(UserContextMiddleware,token=lambda scope:"internal-test")
    app.include_router(create_task_router(translate))
    def headers(user): return {"Authorization":"Bearer internal-test","X-User-ID":user}
    with TestClient(app) as client:
        result=client.post("/api/web/images/tasks",headers=headers("alice"),json={"image_url":"https://images.test/a.png"})
        assert result.status_code == 202,result.text
        path="/api/web/images/tasks/"+result.json()["id"]
        assert client.get(path,headers=headers("bob")).status_code == 404
        for _ in range(100):
            response=client.get(path,headers=headers("alice"))
            if response.json()["status"] == "succeeded": break
            time.sleep(.01)
        assert response.json()["status"] == "succeeded"
    assert seen == ["alice"]
