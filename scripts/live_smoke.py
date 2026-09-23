"""Короткая проверка OpenAI -> MCP -> Chrome с лимитами на локальной странице."""

import asyncio
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import AsyncOpenAI

from browser_agent.agent import Agent
from browser_agent.config import Settings
from browser_agent.mcp_client import connect_browser
from browser_agent.state import RunLog


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"""<!doctype html><html><meta charset='utf-8'><title>Browser check</title>
        <h1>Browser check</h1><p>Verification code: CEDAR-742</p></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


async def main():
    directory = Path("artifacts") / ("live-smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    settings = Settings(
        max_steps=3,
        max_api_calls=4,
        max_output_tokens=600,
        reasoning_effort="none",
        api_max_retries=0,
        profile_dir=directory / "profile",
        run_dir=directory,
        browser_executable="C:/Program Files/Google/Chrome/Application/chrome.exe",
    )
    if not settings.api_key.get_secret_value():
        raise RuntimeError("OPENAI_API_KEY is missing")
    site = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    worker = threading.Thread(target=site.serve_forever, daemon=True)
    worker.start()
    log = RunLog(directory, settings.api_key.get_secret_value())

    async def ask(question):
        raise RuntimeError("Unexpected user input requested in the small smoke test")

    try:
        async with AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(), max_retries=0, timeout=40
        ) as client:
            async with connect_browser(settings) as browser:
                agent = Agent(settings, client, browser, log, ask, lambda s: print(s, flush=True))
                result = await agent.run(
                    f"Открой http://127.0.0.1:{site.server_port} и прочитай проверочный код "
                    "со страницы. Сообщи код и заголовок страницы. Ответ короткий."
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
                (directory / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"Artifacts: {directory.resolve()}", flush=True)
                if result["status"] != "completed" or "CEDAR-742" not in (
                    result.get("summary", "") + result.get("evidence", "")
                ):
                    raise RuntimeError("Smoke test did not verify the expected page code")
    finally:
        site.shutdown()
        site.server_close()
        worker.join()


if __name__ == "__main__":
    asyncio.run(main())
