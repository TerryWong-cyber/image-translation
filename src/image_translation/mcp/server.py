"""MCP tools backed by the shared image translation application service."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal, Protocol

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, RootModel

from image_translation.contracts import (
    TranslationCommand,
    TranslationLanguage,
    TranslationResult,
)


class TranslationService(Protocol):
    async def translate(self, command: TranslationCommand) -> TranslationResult: ...


class TranslationDryRunResult(BaseModel):
    """Validated MCP execution plan that performs no external work."""

    status: Literal["dry_run"] = "dry_run"
    dry_run: Literal[True] = True
    provider: Literal["image_translation"] = "image_translation"
    tool_name: Literal["translate_image_from_oss"] = "translate_image_from_oss"
    arguments: TranslationCommand


TranslationToolOutcome = Annotated[
    TranslationResult | TranslationDryRunResult,
    Field(discriminator="status"),
]


class TranslationToolResult(RootModel[TranslationToolOutcome]):
    """Structured MCP result for either execution or a dry-run plan."""


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
            raise RuntimeError("Image translation service is not ready")
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
        description="Translate text embedded in images stored in the configured object storage.",
        instructions=(
            "Call translate_image_from_oss with an object-storage bucket and relative image key. "
            "Use an any_<target> language mode to detect the source language automatically. "
            "Supported modes are any_zh_cn, any_zh_tw, any_zh, any_en, any_ja, any_ko, any_fr, "
            "any_ar, any_de, any_ru, "
            "any_nl, any_pt, any_th, any_es, any_it, any_vi, and any_id. The explicit legacy "
            "directions en_zh and zh_en remain supported. Set "
            "dry_run=true only when the user "
            "asks to validate or preview the call without reading or writing object storage. The normal "
            "tool call writes a translated image back to object storage and returns structured text regions."
        ),
        version="0.1.0",
    )
    concurrency = asyncio.Semaphore(max_concurrency)

    @server.tool(
        title="Translate an image from object storage",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def translate_image_from_oss(
        bucket: Annotated[
            str,
            Field(min_length=1, max_length=255, description="Source object-storage bucket."),
        ],
        image_key: Annotated[
            str,
            Field(
                min_length=1,
                max_length=2048,
                description="Relative object key of the source image; do not pass image bytes or a URL.",
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
        save_bucket: Annotated[
            str | None,
            Field(description="Destination bucket; defaults to the source bucket."),
        ] = None,
        output_key: Annotated[
            str | None,
            Field(description="Destination object key; a deterministic key is generated when omitted."),
        ] = None,
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
        """Translate an OSS image using an explicit or auto-detected source language."""

        command = TranslationCommand(
            bucket=bucket,
            image_key=image_key,
            language=language,
            save_bucket=save_bucket,
            output_key=output_key,
            segment=segment,
        )
        if dry_run:
            return TranslationToolResult(
                root=TranslationDryRunResult(arguments=command),
            )

        async with concurrency:
            result = await registry.get().translate(command)
        return TranslationToolResult(root=result)

    return server
