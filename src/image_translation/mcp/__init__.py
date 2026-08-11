"""MCP adapter for the image translation service."""

from image_translation.mcp.server import TranslationServiceRegistry, create_mcp_server

__all__ = ["TranslationServiceRegistry", "create_mcp_server"]
