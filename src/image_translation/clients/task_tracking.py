"""Vendored PostgreSQL task tracker.

Canonical source: task-service/task_service/tracking.py. Schema is managed only
by task-service Alembic migrations. Snapshots and pending delivery commit together.
"""

from __future__ import annotations

import asyncio
from .service_identity import delegated_user_id, user_scope, authenticated_user, UserContextMiddleware
import functools
import hashlib
import inspect
import json
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import NAMESPACE_URL, uuid4, uuid5

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import make_url

SCHEMA = "task_mgmt"
metadata = sa.MetaData(
    schema=SCHEMA,
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    },
)
producer_instances = sa.Table(
    "producer_instances",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("owner", sa.Text, nullable=False),
    sa.Column("provider", sa.Text, nullable=False),
    sa.Column(
        "heartbeat_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("owner", "provider", "id"),
    sa.CheckConstraint(
        "length(trim(owner)) > 0 AND length(trim(provider)) > 0",
        name="identity_nonempty",
    ),
    sa.Index("ix_producer_instances_expiry", "owner", "provider", "expires_at"),
)
producer_snapshots = sa.Table(
    "producer_snapshots",
    metadata,
    sa.Column("owner", sa.Text, primary_key=True),
    sa.Column("provider", sa.Text, primary_key=True),
    sa.Column("native_id", sa.Text, primary_key=True),
    sa.Column("instance_id", sa.Uuid, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("version", sa.BigInteger, nullable=False),
    sa.Column("payload", JSONB, nullable=False),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.ForeignKeyConstraint(
        ["owner", "provider", "instance_id"],
        [
            SCHEMA + ".producer_instances.owner",
            SCHEMA + ".producer_instances.provider",
            SCHEMA + ".producer_instances.id",
        ],
    ),
    sa.CheckConstraint("version > 0", name="version_positive"),
    sa.CheckConstraint(
        "status IN ('queued','running','unknown','succeeded','failed','cancelled')",
        name="status_valid",
    ),
    sa.CheckConstraint("jsonb_typeof(payload) = 'object'", name="payload_object"),
    sa.CheckConstraint(
        "payload->>'native_id' = native_id AND (payload->>'version')::bigint = version AND payload->>'status' = status",
        name="payload_identity",
    ),
    sa.CheckConstraint(
        "length(trim(owner)) > 0 AND length(trim(provider)) > 0 AND length(trim(native_id)) > 0",
        name="identity_nonempty",
    ),
    sa.Index("ix_producer_snapshots_recovery", "owner", "provider", "status", "updated_at"),
)
producer_outbox = sa.Table(
    "producer_outbox",
    metadata,
    sa.Column("owner", sa.Text, primary_key=True),
    sa.Column("provider", sa.Text, primary_key=True),
    sa.Column("native_id", sa.Text, primary_key=True),
    sa.Column("version", sa.BigInteger, nullable=False),
    sa.Column("payload", JSONB, nullable=False),
    sa.Column("claimed_by", sa.Uuid),
    sa.Column("claimed_until", sa.DateTime(timezone=True)),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    sa.Column(
        "next_attempt_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column("last_error", sa.Text),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.ForeignKeyConstraint(
        ["owner", "provider", "native_id"],
        [
            SCHEMA + ".producer_snapshots.owner",
            SCHEMA + ".producer_snapshots.provider",
            SCHEMA + ".producer_snapshots.native_id",
        ],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint("version > 0 AND attempts >= 0", name="counters_valid"),
    sa.CheckConstraint("jsonb_typeof(payload) = 'object'", name="payload_object"),
    sa.CheckConstraint(
        "payload->>'native_id' = native_id AND (payload->>'version')::bigint = version",
        name="payload_identity",
    ),
    sa.CheckConstraint("(claimed_by IS NULL) = (claimed_until IS NULL)", name="claim_pair"),
    sa.Index(
        "ix_producer_outbox_delivery",
        "owner",
        "provider",
        "next_attempt_at",
        "claimed_until",
    ),
)

_managed = ContextVar("managed_tracking_provider", default=None)
_parent = ContextVar("task_parent", default=None)
_conversation = ContextVar("task_conversation", default=None)
logger = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled"}


def postgres_url(value):
    url = make_url(value)
    if url.drivername not in {"postgresql", "postgresql+psycopg"}:
        raise ValueError("Task storage requires postgresql+psycopg")
    return url.set(drivername="postgresql+psycopg")


def build_engine(value, application_name="task-service"):
    return sa.create_engine(
        postgres_url(value),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_timeout=10,
        connect_args={"connect_timeout": 5, "application_name": application_name[:63]},
    )


def references(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, tuple):
        value = value[0]
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return {}
    result = {
        key: value[key][:100_000]
        for key in ("answer", "export_id", "document_id", "file_id")
        if isinstance(value.get(key), str)
    }
    result["files"] = [
        {key: f[key] for key in ("file_id", "filename", "size", "content_type") if key in f}
        for f in value.get("files", [])
        if isinstance(f, dict) and isinstance(f.get("file_id"), str)
    ]
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Tracker:
    def __init__(self, provider):
        try:
            from dotenv import dotenv_values

            for path in [
                os.getenv("TASK_SERVICE_ENV_FILE", ".env.task-service"),
                ".env",
            ]:
                if Path(path).is_file():
                    for key, value in dotenv_values(path).items():
                        if key.startswith("TASK_SERVICE_") and value is not None:
                            os.environ.setdefault(key, value)
        except ImportError:
            pass
        self.provider = provider
        self.base = os.getenv("TASK_SERVICE_BASE_URL", "").rstrip("/")
        self.token = os.getenv("TASK_SERVICE_TOKEN", "")
        self.default_owner = os.getenv("TASK_SERVICE_OWNER_ID", "local")
        self._owners = {self.default_owner}
        self.context_token = os.getenv("TASK_SERVICE_CONTEXT_TOKEN", "")
        self.enabled = bool(self.base)
        self.database_url = os.getenv("TASK_SERVICE_DATABASE_URL", "")
        if self.enabled and (not self.token or not self.database_url):
            raise ValueError(
                "TASK_SERVICE_TOKEN and TASK_SERVICE_DATABASE_URL are required when tracking is enabled"
            )
        if self.enabled:
            postgres_url(self.database_url)
        self.instance_id = uuid4()
        self._engine = None
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def owner(self):
        return delegated_user_id() or self.default_owner

    @property
    def lease_id(self):
        return uuid5(self.instance_id, self.owner)

    @property
    def engine(self):
        with self._lock:
            if self._pid != os.getpid():
                if self._engine is not None:
                    self._engine.dispose(close=False)
                self._engine = None
                self._pid = os.getpid()
                self.instance_id = uuid4()
            if self._engine is None:
                self._engine = build_engine(self.database_url, "task-producer:" + self.provider)
            return self._engine

    def _where(self, table, native_id=None, owner=None):
        clause = (table.c.owner == (owner or self.owner)) & (table.c.provider == self.provider)
        return clause if native_id is None else clause & (table.c.native_id == str(native_id))

    def _provider_lock(self, connection):
        # Serialize snapshot versions and lease recovery across processes, not just threads.
        key = int.from_bytes(
            hashlib.sha256(json.dumps([self.owner, self.provider]).encode()).digest()[:8],
            "big",
            signed=True,
        )
        connection.execute(sa.text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})

    def _heartbeat(self, connection, stopping=False):
        now = sa.func.clock_timestamp()
        expiry = now if stopping else now + sa.text("interval '120 seconds'")
        connection.execute(
            insert(producer_instances)
            .values(
                id=self.lease_id,
                owner=self.owner,
                provider=self.provider,
                heartbeat_at=now,
                expires_at=expiry,
            )
            .on_conflict_do_update(
                index_elements=["id"], set_={"heartbeat_at": now, "expires_at": expiry}
            )
        )

    def identifier(self, native_id):
        return (
            "task_"
            + uuid5(
                NAMESPACE_URL,
                json.dumps([self.owner, self.provider, str(native_id)], separators=(",", ":")),
            ).hex
        )

    def get(self, native_id):
        if not self.enabled:
            return None
        with self.engine.connect() as connection:
            return connection.execute(
                sa.select(producer_snapshots.c.payload).where(
                    self._where(producer_snapshots, native_id)
                )
            ).scalar_one_or_none()

    def _write(self, connection, native_id, status, title=None, result=None, error=None):
        previous = connection.execute(
            sa.select(producer_snapshots.c.payload).where(
                self._where(producer_snapshots, native_id)
            )
        ).scalar_one_or_none()
        if previous and previous["status"] in TERMINAL:
            return
        payload = {
            "native_id": str(native_id),
            "title": (title or (previous or {}).get("title") or self.provider)[:200],
            "status": status,
            "version": (previous or {}).get("version", 0) + 1,
            "parent_id": (previous or {}).get("parent_id") or _parent.get(),
            "conversation_id": (previous or {}).get("conversation_id") or _conversation.get(),
            "result": references(result),
            "error": error[:500] if error else None,
        }
        values = dict(
            owner=self.owner,
            provider=self.provider,
            native_id=str(native_id),
            version=payload["version"],
            payload=payload,
            updated_at=sa.func.clock_timestamp(),
        )
        connection.execute(
            insert(producer_snapshots)
            .values(**values, status=status, instance_id=self.lease_id)
            .on_conflict_do_update(
                index_elements=["owner", "provider", "native_id"],
                set_={**values, "status": status, "instance_id": self.lease_id},
            )
        )
        connection.execute(
            insert(producer_outbox)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["owner", "provider", "native_id"],
                set_={
                    **values,
                    "claimed_by": None,
                    "claimed_until": None,
                    "attempts": 0,
                    "last_error": None,
                    "next_attempt_at": sa.func.clock_timestamp(),
                },
            )
        )

    def write(self, native_id, status, title=None, result=None, error=None):
        if not self.enabled:
            return
        with self._lock:
            self._owners.add(self.owner)
        with self.engine.begin() as connection:
            self._provider_lock(connection)
            self._heartbeat(connection)
            self._write(connection, native_id, status, title, result, error)

    def start(self, native_id, title=None, queued=False, parent_native_id=None):
        if parent_native_id:
            with self.scope(parent_native_id):
                self.write(native_id, "queued" if queued else "running", title)
        else:
            self.write(native_id, "queued" if queued else "running", title)
        return self.identifier(native_id)

    def finish(self, native_id, result=None):
        self.write(native_id, "succeeded", result=result)

    def fail(self, native_id, cancelled=False):
        self.write(
            native_id,
            "cancelled" if cancelled else "failed",
            error=None if cancelled else "processing_failed",
        )

    @contextmanager
    def scope(self, native_id):
        mark = _parent.set(self.identifier(native_id) if self.enabled else _parent.get())
        try:
            yield
        finally:
            _parent.reset(mark)

    @contextmanager
    def managed(self, native_id):
        mark = _managed.set(self.provider)
        try:
            with self.scope(native_id):
                yield
        finally:
            _managed.reset(mark)

    async def mcp_context(self, ctx, call_next):
        headers = dict(ctx.request.headers) if ctx.request is not None else {}
        with self.incoming(headers):
            return await call_next(ctx)

    def propagation_headers(self, provider=None):
        # Explicit allowlist: never send a shared context credential to arbitrary MCPs.
        trusted = {
            v.strip()
            for v in os.getenv("TASK_SERVICE_TRUSTED_PROVIDERS", "").split(",")
            if v.strip()
        }
        if not self.context_token or (provider is not None and provider not in trusted):
            return {}
        headers = {"X-Task-Context-Token": self.context_token}
        if _parent.get():
            headers["X-Task-Parent"] = _parent.get()
        if _conversation.get():
            headers["X-Task-Conversation"] = _conversation.get()
        return headers

    @contextmanager
    def incoming(self, headers):
        supplied = headers.get("x-task-context-token", "")
        trusted = bool(self.context_token) and secrets.compare_digest(supplied, self.context_token)
        parent = headers.get("x-task-parent", "") if trusted else ""
        conversation = headers.get("x-task-conversation", "") if trusted else ""
        owner = delegated_user_id()
        if headers.get("x-user-id") is not None and owner != headers.get("x-user-id"):
            owner = authenticated_user(headers, os.getenv("INTERNAL_SERVICE_TOKEN", "") or self.token)
        a = _parent.set(parent if parent.startswith("task_") and len(parent) <= 80 else None)
        b = _conversation.set(conversation[:200] or None)
        try:
            with user_scope(owner):
                yield
        finally:
            _parent.reset(a)
            _conversation.reset(b)

    def _claim(self):
        claim = uuid4()
        with self.engine.begin() as connection:
            rows = (
                connection.execute(
                    sa.select(producer_outbox)
                    .where(
                        producer_outbox.c.provider == self.provider,
                        producer_outbox.c.next_attempt_at <= sa.func.clock_timestamp(),
                        sa.or_(
                            producer_outbox.c.claimed_until.is_(None),
                            producer_outbox.c.claimed_until <= sa.func.clock_timestamp(),
                        ),
                    )
                    .order_by(producer_outbox.c.updated_at)
                    .limit(50)
                    .with_for_update(skip_locked=True)
                )
                .mappings()
                .all()
            )
            for row in rows:
                connection.execute(
                    sa.update(producer_outbox)
                    .where(self._where(producer_outbox, row["native_id"], row["owner"]))
                    .values(
                        claimed_by=claim,
                        claimed_until=sa.func.clock_timestamp() + sa.text("interval '300 seconds'"),
                    )
                )
        return claim, rows

    def flush_once(self):
        if not self.enabled:
            return 0
        claim, rows = self._claim()
        sent = 0
        try:
            for row in rows:
                if self._stop.is_set():
                    break
                # Network delivery never holds a database row lock.
                request = Request(
                    self.base
                    + "/internal/v1/providers/"
                    + quote(self.provider, safe="")
                    + "/tasks",
                    data=json.dumps(row["payload"], ensure_ascii=False).encode(),
                    headers={
                        "Authorization": "Bearer " + self.token,
                        "Content-Type": "application/json",
                        "X-User-ID": row["owner"],
                    },
                    method="POST",
                )
                delivered = False
                try:
                    with build_opener(NoRedirect).open(request, timeout=5) as response:
                        delivered = response.status == 202
                except (HTTPError, URLError, TimeoutError, OSError):
                    pass
                with self.engine.begin() as connection:
                    condition = sa.and_(
                        self._where(producer_outbox, row["native_id"], row["owner"]),
                        producer_outbox.c.version == row["version"],
                        producer_outbox.c.claimed_by == claim,
                    )
                    if delivered:
                        # A newer snapshot may have replaced this in-flight version.
                        connection.execute(sa.delete(producer_outbox).where(condition))
                        sent += 1
                    else:
                        connection.execute(
                            sa.update(producer_outbox)
                            .where(condition)
                            .values(
                                claimed_by=None,
                                claimed_until=None,
                                attempts=producer_outbox.c.attempts + 1,
                                last_error="delivery_deferred",
                                next_attempt_at=sa.func.clock_timestamp()
                                + sa.text("interval '2 seconds'"),
                            )
                        )
                if not delivered:
                    logger.warning("task_tracking_delivery_deferred provider=%s", self.provider)
                    break
        finally:
            # Release unattempted claims promptly; crashes instead rely on claim expiry.
            with self.engine.begin() as connection:
                connection.execute(
                    sa.update(producer_outbox)
                    .where(
                        producer_outbox.c.provider == self.provider,
                        producer_outbox.c.claimed_by == claim,
                    )
                    .values(claimed_by=None, claimed_until=None)
                )
        return sent

    def renew_and_recover(self):
        with self.engine.connect() as connection:
            owners = set(connection.execute(sa.select(producer_snapshots.c.owner).where(
                producer_snapshots.c.provider == self.provider,
                producer_snapshots.c.status.in_(["queued", "running"]),
            ).distinct()).scalars())
        with self._lock:
            owners.update(self._owners)
        for owner in sorted(owners):
            with user_scope(owner):
                self._renew_owner()

    def _renew_owner(self):
        with self.engine.begin() as connection:
            self._provider_lock(connection)
            self._heartbeat(connection)
            if os.getenv("TASK_SERVICE_RECOVERY", "unknown") == "external":
                return
            snapshot = producer_snapshots
            rows = (
                connection.execute(
                    sa.select(snapshot.c.native_id)
                    .select_from(
                        snapshot.join(
                            producer_instances,
                            snapshot.c.instance_id == producer_instances.c.id,
                        )
                    )
                    .where(
                        self._where(snapshot),
                        snapshot.c.status.in_(["queued", "running"]),
                        producer_instances.c.expires_at <= sa.func.clock_timestamp(),
                    )
                    .order_by(snapshot.c.updated_at)
                    .limit(100)
                )
                .scalars()
                .all()
            )
            for native_id in rows:
                self._write(
                    connection,
                    native_id,
                    "unknown",
                    error="execution_state_unknown_after_restart",
                )

    def start_delivery(self):
        if not self.enabled or self._thread is not None:
            return
        with self.engine.connect() as connection:
            connection.execute(sa.select(producer_outbox.c.version).limit(0))
        self._stop.clear()
        # A dedicated lease thread keeps executing jobs alive during slow HTTP delivery.
        self.renew_and_recover()

        def maintain():
            while not self._stop.wait(10):
                try:
                    self.renew_and_recover()
                except Exception:
                    logger.error("task_tracking_lease_error provider=%s", self.provider)

        self._lease_thread = threading.Thread(
            target=maintain, name="task-lease-" + self.provider, daemon=True
        )
        self._lease_thread.start()

        def deliver():
            while not self._stop.is_set():
                try:
                    self.flush_once()
                except Exception:
                    logger.error("task_tracking_outbox_error provider=%s", self.provider)
                self._stop.wait(2)

        self._thread = threading.Thread(
            target=deliver, name="task-outbox-" + self.provider, daemon=True
        )
        self._thread.start()

    def stop_delivery(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)
            self._lease_thread.join(timeout=6)
            self._thread = None
        if self.enabled and self._engine is not None:
            with self.engine.begin() as connection:
                with self._lock:
                    owners = tuple(self._owners)
                for owner in sorted(owners):
                    with user_scope(owner):
                        self._provider_lock(connection)
                        self._heartbeat(connection, stopping=True)
            self._engine.dispose()
        # Pending PostgreSQL rows survive process exit; execution recovery uses leases.

    def install(self, app, *, user_auth=True):
        if user_auth:
            app.add_middleware(UserContextMiddleware, token=lambda scope: self.token if scope["path"].startswith("/internal/task-tracking/") else os.getenv("INTERNAL_SERVICE_TOKEN", "") or self.token)
        if not self.enabled:
            return
        from fastapi import HTTPException
        from fastapi import Request as WebRequest

        # Resolve local annotation when annotations are postponed.
        globals()["WebRequest"] = WebRequest

        @app.get("/internal/task-tracking/{native_id}", include_in_schema=False)
        def snapshot(native_id: str, request: WebRequest):
            if not secrets.compare_digest(
                request.headers.get("authorization", ""), "Bearer " + self.token
            ):
                raise HTTPException(401, "invalid tracking credential")
            try:
                owner = authenticated_user(request.headers, self.token)
            except (PermissionError, ValueError) as exc:
                raise HTTPException(401, "invalid service identity") from exc
            with user_scope(owner):
                result = self.get(native_id)
            if result is None:
                raise HTTPException(404, "tracking snapshot unavailable")
            return result

        tracker = self

        class ContextMiddleware:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
                with tracker.incoming(headers):
                    await self.app(scope, receive, send)

        app.add_middleware(ContextMiddleware)
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application):
            await asyncio.to_thread(self.start_delivery)
            try:
                async with original(application) as state:
                    yield state
            finally:
                await asyncio.to_thread(self.stop_delivery)

        app.router.lifespan_context = lifespan

    def tracked(self, title):
        def decorate(function):
            if inspect.iscoroutinefunction(function):

                @functools.wraps(function)
                async def async_call(*args, **kwargs):
                    if _managed.get() == self.provider:
                        return await function(*args, **kwargs)
                    identifier = str(uuid4())
                    await asyncio.to_thread(self.start, identifier, title)
                    try:
                        with self.managed(identifier):
                            result = await function(*args, **kwargs)
                    except BaseException as exc:
                        await asyncio.to_thread(
                            self.fail,
                            identifier,
                            cancelled=isinstance(exc, asyncio.CancelledError),
                        )
                        raise
                    await asyncio.to_thread(self.finish, identifier, result)
                    return result

                return async_call

            @functools.wraps(function)
            def sync_call(*args, **kwargs):
                if _managed.get() == self.provider:
                    return function(*args, **kwargs)
                identifier = str(uuid4())
                self.start(identifier, title)
                try:
                    with self.managed(identifier):
                        result = function(*args, **kwargs)
                except BaseException:
                    self.fail(identifier)
                    raise
                self.finish(identifier, result)
                return result

            return sync_call

        return decorate
