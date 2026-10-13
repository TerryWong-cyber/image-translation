import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock


MCP_AVAILABLE = (
    importlib.util.find_spec("mcp") is not None
    and importlib.util.find_spec("pydantic") is not None
)

if MCP_AVAILABLE:
    import httpx
    from mcp import Client
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from image_translation.contracts import TranslationResult
    from image_translation.mcp import TranslationServiceRegistry, create_mcp_server


@unittest.skipUnless(MCP_AVAILABLE, "mcp and pydantic are not installed")
class McpServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_language_schema_explains_every_translation_direction(self):
        server = create_mcp_server(TranslationServiceRegistry())

        async with Client(server, raise_exceptions=True) as client:
            result = await client.list_tools()

        tool = next(
            item for item in result.tools if item.name == "translate_image_from_url"
        )
        description = tool.input_schema["properties"]["language"]["description"]
        for language in (
            "any_zh",
            "any_zh_cn",
            "any_zh_tw",
            "any_en",
            "any_ja",
            "any_ko",
            "any_fr",
            "any_ar",
            "any_de",
            "any_ru",
            "any_nl",
            "any_pt",
            "any_th",
            "any_es",
            "any_it",
            "any_vi",
            "any_id",
        ):
            self.assertIn(language, description)
        self.assertIn("language", tool.input_schema["required"])
        self.assertNotIn("default", tool.input_schema["properties"]["language"])

    async def test_tool_schema_exposes_only_url_and_translation_options(self):
        server = create_mcp_server(TranslationServiceRegistry())
        async with Client(server, raise_exceptions=True) as client:
            result = await client.list_tools()
        self.assertEqual([tool.name for tool in result.tools], ["translate_image_from_url"])
        tool = result.tools[0]
        self.assertEqual(set(tool.input_schema["properties"]), {"image_url", "language", "segment", "dry_run"})
        self.assertIn("image_url", tool.input_schema["required"])
        self.assertFalse(tool.annotations.idempotent_hint)
        self.assertTrue(tool.annotations.open_world_hint)

    async def test_dry_run_is_explicit_and_defaults_to_false(self):
        server = create_mcp_server(TranslationServiceRegistry())

        async with Client(server, raise_exceptions=True) as client:
            result = await client.list_tools()

        tool = next(
            item for item in result.tools if item.name == "translate_image_from_url"
        )
        dry_run_schema = tool.input_schema["properties"]["dry_run"]
        self.assertEqual(dry_run_schema["default"], False)
        self.assertIn("without downloading", dry_run_schema["description"])

    async def test_tool_calls_shared_service_and_returns_structured_output(self):
        class FakeService:
            def __init__(self):
                self.command = None

            async def translate_url(self, image_url, language="en_zh", segment=False):
                self.command = (image_url, language, segment)
                return TranslationResult(
                    request_id="request-1",
                    source_url=image_url,
                    translated_url="https://files.example.com/translations/result.png",
                    data=[],
                    duration_ms=25,
                )

        service = FakeService()
        registry = TranslationServiceRegistry()
        registry.set(service)
        server = create_mcp_server(registry)

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "translate_image_from_url",
                {
                    "image_url": "https://images.example.com/input.png",
                    "language": "en_zh",
                },
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["request_id"], "request-1")
        self.assertEqual(result.structured_content["translated_url"], "https://files.example.com/translations/result.png")
        self.assertEqual(service.command, ("https://images.example.com/input.png", "en_zh", False))

    async def test_dry_run_validates_and_returns_plan_without_service(self):
        server = create_mcp_server(TranslationServiceRegistry())

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "translate_image_from_url",
                {
                    "image_url": "https://images.example.com/posters/launch.jpg",
                    "language": "zh_en",
                    "segment": True,
                    "dry_run": True,
                },
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "dry_run")
        self.assertIs(result.structured_content["dry_run"], True)
        self.assertEqual(result.structured_content["provider"], "image_translation")
        self.assertEqual(
            result.structured_content["tool_name"],
            "translate_image_from_url",
        )
        self.assertEqual(
            result.structured_content["arguments"],
            {
                "image_url": "https://images.example.com/posters/launch.jpg",
                "language": "zh_en",
                "segment": True,
            },
        )

    async def test_dry_run_still_rejects_a_non_http_url(self):
        server = create_mcp_server(TranslationServiceRegistry())

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "translate_image_from_url",
                {
                    "image_url": "file:///tmp/launch.jpg",
                    "language": "en_zh",
                    "dry_run": True,
                },
            )

        self.assertTrue(result.is_error)
        self.assertIn("URL", result.content[0].text)

    async def test_dry_run_rejects_credentials_fragments_and_storage_only_input(self):
        server = create_mcp_server(TranslationServiceRegistry())
        async with Client(server, raise_exceptions=True) as client:
            for arguments in (
                {"image_url": "http://user:pass@example.com/a.png"},
                {"image_url": "https://example.com/a.png#fragment"},
                {"bucket": "images", "image_key": "a.png"},
            ):
                with self.subTest(arguments=arguments):
                    result = await client.call_tool("translate_image_from_url", {**arguments, "language": "any_en", "dry_run": True})
                    self.assertTrue(result.is_error)

    async def test_unready_service_is_reported_as_tool_error(self):
        server = create_mcp_server(TranslationServiceRegistry())

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "translate_image_from_url",
                {
                    "image_url": "https://images.example.com/input.png",
                    "language": "en_zh",
                },
            )

        self.assertTrue(result.is_error)
        self.assertIn("not ready", result.content[0].text)

    async def test_model_failure_is_reported_as_tool_error(self):
        from image_translation.errors import ModelUpstreamError, ModelTimeoutError

        for failure in (ModelUpstreamError("Model unavailable"), ModelTimeoutError("Model timed out")):
            service = SimpleNamespace(translate_url=AsyncMock(side_effect=failure))
            registry = TranslationServiceRegistry()
            registry.set(service)
            server = create_mcp_server(registry)
            async with Client(server, raise_exceptions=True) as client:
                result = await client.call_tool("translate_image_from_url", {
                    "image_url": "https://images.example.com/a.png", "language": "en_zh"
                })
            self.assertTrue(result.is_error)
            self.assertIn(str(failure), result.content[0].text)

    async def test_streamable_http_is_available_at_exact_mcp_path(self):
        registry = TranslationServiceRegistry()
        service = SimpleNamespace(translate_url=AsyncMock(return_value=TranslationResult(
            request_id="http-request", source_url="https://example.com/a.png",
            translated_url="https://files.example.com/translations/result.png", data=[], duration_ms=12,
        )))
        registry.set(service)
        server = create_mcp_server(registry)
        mcp_app = server.streamable_http_app(
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=["localhost", "localhost:*"],
                allowed_origins=["http://localhost", "http://localhost:*"],
            ),
        )
        host_app = Starlette(routes=[Mount("/", app=mcp_app)])

        async with server.session_manager.run():
            transport = httpx.ASGITransport(app=host_app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-11-25",
                            "capabilities": {},
                            "clientInfo": {"name": "test-client", "version": "1.0"},
                        },
                    },
                    headers={"accept": "application/json, text/event-stream"},
                )
                listed = await client.post(
                    "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                    headers={"accept": "application/json, text/event-stream"},
                )
                planned = await client.post(
                    "/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                        "name": "translate_image_from_url",
                        "arguments": {"image_url": "https://example.com/a.png", "language": "any_en", "dry_run": True},
                    }},
                    headers={"accept": "application/json, text/event-stream"},
                )
                service.translate_url.assert_not_called()
                translated = await client.post(
                    "/mcp", json={"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                        "name": "translate_image_from_url",
                        "arguments": {"image_url": "https://example.com/a.png", "language": "any_en"},
                    }}, headers={"accept": "application/json, text/event-stream"},
                )

        self.assertEqual(translated.status_code, 200)
        self.assertEqual(translated.json()["result"]["structuredContent"]["translated_url"], "https://files.example.com/translations/result.png")
        service.translate_url.assert_awaited_once_with("https://example.com/a.png", "any_en", False)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([tool["name"] for tool in listed.json()["result"]["tools"]], ["translate_image_from_url"])
        self.assertEqual(planned.status_code, 200)
        self.assertEqual(planned.json()["result"]["structuredContent"]["arguments"]["image_url"], "https://example.com/a.png")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("location"))
        self.assertEqual(
            response.json()["result"]["serverInfo"]["name"],
            "image-translation",
        )


if __name__ == "__main__":
    unittest.main()
