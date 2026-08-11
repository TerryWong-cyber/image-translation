import importlib.util
import unittest


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
    async def test_tool_calls_shared_service_and_returns_structured_output(self):
        class FakeService:
            def __init__(self):
                self.command = None

            async def translate(self, command):
                self.command = command
                return TranslationResult(
                    request_id="request-1",
                    source_url=f"{command.bucket}/{command.image_key}",
                    translated_url=f"{command.bucket}/en_zh_translated_input.png",
                    data=[],
                    duration_ms=25,
                )

        service = FakeService()
        registry = TranslationServiceRegistry()
        registry.set(service)
        server = create_mcp_server(registry)

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "translate_image_from_oss",
                {
                    "bucket": "source",
                    "image_key": "input.png",
                    "language": "en_zh",
                },
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["request_id"], "request-1")
        self.assertEqual(result.structured_content["translated_url"], "source/en_zh_translated_input.png")
        self.assertEqual(service.command.image_key, "input.png")

    async def test_unready_service_is_reported_as_tool_error(self):
        server = create_mcp_server(TranslationServiceRegistry())

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "translate_image_from_oss",
                {"bucket": "source", "image_key": "input.png"},
            )

        self.assertTrue(result.is_error)
        self.assertIn("not ready", result.content[0].text)

    async def test_streamable_http_is_available_at_exact_mcp_path(self):
        registry = TranslationServiceRegistry()
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

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("location"))
        self.assertEqual(
            response.json()["result"]["serverInfo"]["name"],
            "image-translation",
        )


if __name__ == "__main__":
    unittest.main()
