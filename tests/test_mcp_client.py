from types import SimpleNamespace

from browser_agent.mcp_client import BrowserMCP


class Session:
    calls = 0

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="navigate",
                    description="Go",
                    inputSchema={
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                )
            ]
        )

    async def call_tool(self, name, arguments, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            isError=False,
            content=[
                SimpleNamespace(type="text", text="page"),
                SimpleNamespace(type="image", mimeType="image/png", data="aW1hZ2U="),
            ],
        )


async def test_discovers_schema_and_rejects_invalid_calls_before_execution():
    session = Session()
    client = BrowserMCP(session, 10)
    await client.discover()
    assert client.tools[0]["parameters"]["required"] == ["url"]
    _, _, error = await client.call("navigate", {"url": 12})
    assert error and session.calls == 0
    _, _, error = await client.call("invented_tool", {})
    assert error and session.calls == 0
    text, images, error = await client.call("navigate", {"url": "https://example.com"})
    assert not error and text == "page"
    assert images[0]["image_url"] == "data:image/png;base64,aW1hZ2U="


async def test_timeout_returns_uncertain_state_without_replaying_action():
    session = Session()

    async def fail(*args, **kwargs):
        session.calls += 1
        raise TimeoutError()

    session.call_tool = fail
    client = BrowserMCP(session, 5)
    await client.discover()
    output, _, error = await client.call("navigate", {"url": "https://example.com"})
    assert error and '"state_unknown": true' in output
    assert session.calls == 1
