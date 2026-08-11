"""Typed contracts shared by HTTP and MCP adapters."""

from image_translation.contracts.translation import (
    TranslationCommand,
    TranslationItem,
    TranslationLanguage,
    TranslationResult,
)

__all__ = [
    "TranslationCommand",
    "TranslationItem",
    "TranslationLanguage",
    "TranslationResult",
]
