"""Trusted service delegation. Canonical: task-service/task_service/service_identity.py."""

import re
import secrets
from contextlib import contextmanager
from contextvars import ContextVar

_user = ContextVar("service_user_id", default=None)
DEFAULT_USER_ID = "local"


def validate_user_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}", value):
        raise ValueError("invalid user identity")
    return value


def current_user_id():
    return _user.get() or DEFAULT_USER_ID


def delegated_user_id():
    return _user.get()


@contextmanager
def user_scope(user_id):
    mark = _user.set(validate_user_id(user_id) if user_id is not None else None)
    try:
        yield
    finally:
        _user.reset(mark)


def authenticated_user(headers, token):
    values = (
        headers.getlist("x-user-id") if hasattr(headers, "getlist") else [headers.get("x-user-id")]
    )
    user = headers.get("x-user-id")
    if user is None:
        return None
    authorization = headers.get("authorization", "")
    if not token or not secrets.compare_digest(
        authorization.encode(), ("Bearer " + token).encode()
    ):
        raise PermissionError("valid service token required for user delegation")
    if len(values) != 1:
        raise ValueError("ambiguous user identity")
    return validate_user_id(user)


def user_headers(headers):
    result = {key: value for key, value in headers.items() if key.lower() != "x-user-id"}
    user = delegated_user_id()
    if user is not None:
        credential = next((v for k, v in result.items() if k.lower() == "authorization"), "")
        if not credential.startswith("Bearer ") or not credential[7:]:
            raise RuntimeError("service token required to propagate user identity")
        result["X-User-ID"] = user
    return result


class UserContextMiddleware:
    """Pure ASGI scope keeps identity alive for streams and resets after completion."""

    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from starlette.datastructures import Headers
        from starlette.responses import JSONResponse

        try:
            headers = Headers(scope=scope)
            user = (
                authenticated_user(headers, self.token(scope)) if "x-user-id" in headers else None
            )
        except (PermissionError, ValueError) as exc:
            response = JSONResponse(
                {"detail": str(exc)}, status_code=401 if isinstance(exc, PermissionError) else 400
            )
            return await response(scope, receive, send)
        with user_scope(user):
            await self.app(scope, receive, send)
