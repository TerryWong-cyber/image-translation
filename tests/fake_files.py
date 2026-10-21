"""UUID-backed File Service test adapter for consumer processing regressions."""

import hashlib
import mimetypes
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from uuid import uuid4

from starlette.responses import Response


def install(monkeypatch, client_class, error_class, project):
    monkeypatch.setenv("FILE_SERVICE_BASE_URL", "https://files.example")
    state = SimpleNamespace(records={}, contents={}, revoked=set(), calls=[])

    def save(filename, data, mime, metadata, path=None):
        file_id = str(uuid4())
        url = f"https://files.example/api/file-service/v1/files/{file_id}/content?token=test"
        if path and (metadata or {}).get("kind") == "export":
            url = f"http://testserver/api/v1/exports/{path.stem}/download"
        if path and project == "converter" and (metadata or {}).get("kind") == "result":
            url = f"http://testserver/api/jobs/{path.parent.name}/download"
        artifact = dict(
            file_id=file_id,
            file_url=url,
            filename=filename,
            mime_type=mime or mimetypes.guess_type(filename)[0] or "application/octet-stream",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            expires_at="2099-01-01T00:00:00Z",
        )
        state.records[file_id], state.contents[file_id] = artifact, data
        state.calls.append((filename, mime, metadata, file_id))
        return artifact.copy()

    async def publish(self, path, filename=None, mime=None, metadata=None):
        path = Path(path)
        return save(filename or path.name, path.read_bytes(), mime, metadata, path)

    async def upload(self, client, filename, content, mime=None, metadata=None):
        data = content if isinstance(content, bytes) else content.read()
        return save(filename, data, mime, metadata)

    async def resolve(self, client, file_id):
        if file_id in state.revoked or file_id not in state.records:
            raise error_class(404, "file_not_found")
        return state.records[file_id].copy()

    async def content_response(self, file_id, download=False):
        artifact = await self.resolve(None, file_id)
        return Response(
            state.contents[file_id],
            media_type=artifact["mime_type"],
            headers={
                "Content-Disposition": ("attachment" if download else "inline")
                + "; filename*=UTF-8''"
                + quote(artifact["filename"]),
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    monkeypatch.setattr(client_class, "publish", publish)
    monkeypatch.setattr(client_class, "upload", upload)
    monkeypatch.setattr(client_class, "resolve", resolve)
    monkeypatch.setattr(client_class, "content_response", content_response)
    return state
