"""MCP tools backed by the shared image translation application service."""

from __future__ import annotations

import asyncio
from typing import Annotated, Protocol

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from image_translation.contracts import (
    TranslationCommand,
    TranslationLanguage,
    TranslationResult,
)


class TranslationService(Protocol):
    async def translate(self, command: TranslationCommand) -> TranslationResult: ...


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
            "The tool writes a translated image back to object storage and returns structured text regions."
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
            Field(description="Translation direction."),
        ] = "en_zh",
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
    ) -> TranslationResult:
        """Translate text in an OSS image and write the rendered result back to OSS."""

        command = TranslationCommand(
            bucket=bucket,
            image_key=image_key,
            language=language,
            save_bucket=save_bucket,
            output_key=output_key,
            segment=segment,
        )
        async with concurrency:
            return await registry.get().translate(command)

    return server
