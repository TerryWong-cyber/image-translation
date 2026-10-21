"""MCP tools backed by the shared image translation application service."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal, Protocol

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from image_translation.clients.file_service_client import FileServiceError
from image_translation.contracts import (
    TranslationLanguage,
    TranslationResult,
    UrlTranslationRequest,
)
from image_translation.errors import ImageTranslationError


class TranslationService(Protocol):
    async def translate_url(
        self, image_url: str, language: str = "en_zh", segment: bool = False
    ) -> TranslationResult: ...


class TranslationDryRunResult(BaseModel):
    """Validated MCP execution plan that performs no external work."""

    status: Literal["dry_run"] = "dry_run"
    dry_run: Literal[True] = True
    provider: Literal["image_translation"] = "image_translation"
    tool_name: Literal["translate_image_from_url"] = "translate_image_from_url"
    arguments: UrlTranslationRequest


TranslationToolOutcome = Annotated[
    TranslationResult | TranslationDryRunResult,
    Field(discriminator="status"),
]


class TranslationToolResult(RootModel[TranslationToolOutcome]):
    """Structured MCP result for either execution or a dry-run plan."""

    # Both union members are objects; MCP requires this at the schema root.
    model_config = ConfigDict(json_schema_extra={"type": "object"})


class TranslationServiceRegistry:
    """Mutable lifecycle slot used by MCP handlers created at import time."""

    def __init__(self) -> None:
        self._service: TranslationService | None = None

    def set(self, service: TranslationService) -> None:
        self._service = service

    def clear(self) -> None:
        self._service = None

    def get(self) -> TranslationService:
        if self._service is None:
            raise ToolError("Image translation service is not ready")
        return self._service


def create_mcp_server(
    registry: TranslationServiceRegistry,
    *,
    max_concurrency: int = 2,
) -> MCPServer:
    """Create the protocol adapter without starting any runtime resources."""

    server = MCPServer(
        name="image-translation",
        title="Image Translation",
        description="Translate text embedded in an image from a complete HTTP(S) URL.",
        instructions=(
            "Call translate_image_from_url with a complete public or configured OSS image URL. "
            "Use an any_<target> language mode to detect the source language automatically. "
            "Supported modes are any_zh_cn, any_zh_tw, any_zh, any_en, any_ja, any_ko, any_fr, "
            "any_ar, any_de, any_ru, "
            "any_nl, any_pt, any_th, any_es, any_it, any_vi, and any_id. The explicit legacy "
            "directions en_zh and zh_en remain supported. Set "
            "dry_run=true only when the user "
            "asks to validate or preview the call without downloading or storing an image. The normal "
            "tool call stores the translated image in the default bucket and returns a complete download URL and structured text regions."
        ),
        version="0.1.0",
    )
    concurrency = asyncio.Semaphore(max_concurrency)

    @server.tool(
        title="Translate an image from URL",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    async def translate_image_from_url(
        image_url: Annotated[
            str,
            Field(
                min_length=1,
                max_length=4096,
                description="Complete HTTP(S) image URL, including signed query parameters when needed. Accepts public images or the configured OSS file service; no bucket/key or local paths.",
                json_schema_extra={"format": "uri"},
            ),
        ],
        language: Annotated[
            TranslationLanguage,
            Field(
                description=(
                    "Translation direction. Use any_zh_cn for Simplified Chinese, any_zh_tw for "
                    "Traditional Chinese (Taiwan), or legacy alias any_zh for Simplified Chinese. "
                    "Other auto-detect modes: any_en, any_ja, any_ko, "
                    "any_fr, any_ar, any_de, any_ru, any_nl, any_pt, any_th, any_es, any_it, "
                    "any_vi, and any_id. Legacy en_zh and zh_en are also supported."
                ),
                examples=["any_en"],
            ),
        ],
        segment: Annotated[
            bool,
            Field(description="Split a large image into regions before OCR."),
        ] = False,
        dry_run: Annotated[
            bool,
            Field(
                description=(
                    "Validate the arguments and return the execution plan without downloading, OCR, "
                    "translation, rendering, or upload. Defaults to false, so normal calls execute."
                ),
            ),
        ] = False,
    ) -> TranslationToolResult:
        """Translate a URL image using an explicit or auto-detected source language."""

        try:
            command = UrlTranslationRequest(
                image_url=image_url,
                language=language,
                segment=segment,
            )
        except ValidationError as exc:
            raise ToolError(exc.errors(include_url=False)[0]["msg"]) from exc
        if dry_run:
            return TranslationToolResult(
                root=TranslationDryRunResult(arguments=command),
            )

        try:
            async with concurrency:
                result = await registry.get().translate_url(
                    str(command.image_url), command.language, command.segment
                )
        except (ImageTranslationError, TimeoutError, FileServiceError) as exc:
            raise ToolError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ToolError("Upstream service request failed") from exc
        return TranslationToolResult(root=result)

    return server
