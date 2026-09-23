import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from time import perf_counter

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from browser_agent.config import Settings


class BrowserMCP:
    def __init__(self, session: ClientSession, timeout: int):
        self.session = session
        self.timeout = timeout
        self.schemas: dict[str, dict] = {}
        self.tools: list[dict] = []
        self.last_call_seconds = 0.0

    async def discover(self):
        result = await self.session.list_tools()
        for tool in result.tools:
            self.schemas[tool.name] = tool.inputSchema
            self.tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "parameters": tool.inputSchema,
                    # Не удаляем необязательные свойства схем MCP при передаче модели.
                    "strict": False,
                }
            )

    async def call(self, name: str, arguments: dict) -> tuple[str, list[dict], bool]:
        if name not in self.schemas:
            return json.dumps({"error": f"Unknown tool: {name}"}), [], True
        errors = list(Draft202012Validator(self.schemas[name]).iter_errors(arguments))
        if errors:
            return json.dumps({"error": errors[0].message}), [], True
        try:
            started = perf_counter()
            result = await self.session.call_tool(
                name, arguments, read_timeout_seconds=timedelta(seconds=self.timeout)
            )
        except Exception as exc:
            # Таймаут не доказывает неудачу: повторять клик автоматически нельзя.
            return (
                json.dumps(
                    {
                        "error": type(exc).__name__,
                        "state_unknown": True,
                        "next": "Observe current page before deciding whether to retry.",
                    }
                ),
                [],
                True,
            )
        finally:
            self.last_call_seconds = round(perf_counter() - started, 3)
        texts, images = [], []
        for block in result.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "image":
                images.append(
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": f"data:{block.mimeType};base64,{block.data}",
                    }
                )
        text = "\n".join(texts) or "Screenshot attached in the next observation."
        return text, images, result.isError


@asynccontextmanager
async def connect_browser(settings: Settings):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
    }
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "AGENT_HEADLESS": str(settings.headless).lower(),
            "AGENT_RECORD_ACTIONS": str(settings.record_actions).lower(),
            "AGENT_PROFILE_DIR": str(settings.profile_dir.resolve()),
            "AGENT_ARTIFACTS_DIR": str(settings.artifacts_dir.resolve()),
        }
    )
    if settings.browser_executable:
        env["AGENT_BROWSER_EXECUTABLE"] = settings.browser_executable
    if settings.cdp_url:
        env["AGENT_CDP_URL"] = settings.cdp_url
    if settings.run_dir:
        env["AGENT_RUN_DIR"] = str(settings.run_dir.resolve())
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "browser_agent.server"], env=env
    )
    log_dir = settings.run_dir or settings.artifacts_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "mcp-server.log").open("a", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=120)
            ) as session:
                await session.initialize()
                bridge = BrowserMCP(session, settings.tool_timeout_seconds)
                await bridge.discover()
                yield bridge
