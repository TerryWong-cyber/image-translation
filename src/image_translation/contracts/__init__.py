"""Typed contracts shared by HTTP and MCP adapters."""

from image_translation.contracts.translation import (
    UrlTranslationRequest,
    TranslationItem,
    TranslationLanguage,
    TranslationResult,
)

__all__ = [
    "UrlTranslationRequest",
    "TranslationItem",
    "TranslationLanguage",
    "TranslationResult",
]
