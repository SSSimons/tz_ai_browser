"""Проверить сохранённый результат одним API-запросом со свежим снимком браузера."""

import argparse
import asyncio
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from live_smoke import Page
from openai import AsyncOpenAI

from browser_agent.agent import Agent
from browser_agent.config import Settings
from browser_agent.mcp_client import connect_browser
from browser_agent.state import RunLog


async def verify(directory: Path):
    if (directory / "verified-result.json").exists():
        print((directory / "verified-result.json").read_text(encoding="utf-8"))
        return
    saved = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    previous = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    previous_calls = sum(
        json.loads(line).get("event") == "model_response"
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    )
    messages = [
        item
        for batch in saved["rounds"]
        for item in batch
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    answer = " ".join(block.get("text", "") for block in messages[-1]["content"])
    if "CEDAR-742" not in answer:
        raise RuntimeError("The recorded model answer does not contain the expected code")
    settings = Settings(
        max_api_calls=1,
        max_output_tokens=300,
        reasoning_effort="none",
        profile_dir=directory / "verify-profile",
        run_dir=directory / "verification",
        browser_executable="C:/Program Files/Google/Chrome/Application/chrome.exe",
    )
    site = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    worker = threading.Thread(target=site.serve_forever, daemon=True)
    worker.start()
    log = RunLog(settings.run_dir, settings.api_key.get_secret_value())
    try:
        async with AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(), max_retries=0, timeout=40
        ) as client:
            async with connect_browser(settings) as browser:
                text, _, error = await browser.call(
                    "browser_navigate", {"url": f"http://127.0.0.1:{site.server_port}"}
                )
                if error:
                    raise RuntimeError(text)
                agent = Agent(settings, client, browser, log, None, print)
                agent.task = saved["task"]
                agent.rounds = saved["rounds"]
                agent.memory = (
                    "Локальный проверочный сайт был перезапущен на новом порту; его содержимое "
                    "не изменялось. Проверь прежний ответ модели по свежему снимку."
                )
                candidate = {
                    "status": "completed",
                    "summary": answer,
                    "evidence": "Original model answer plus a fresh browser observation",
                }
                verified, feedback = await agent.review(candidate)
                result = {
                    "verified": verified,
                    "feedback": feedback,
                    "summary": answer,
                    "api_calls_total": previous_calls + agent.api_calls,
                    "tokens_total": {
                        key: previous["tokens"][key] + agent.tokens[key] for key in agent.tokens
                    },
                }
                (directory / "verified-result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if not verified:
                    raise RuntimeError("Verification was rejected")
    finally:
        site.shutdown()
        site.server_close()
        worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    asyncio.run(verify(parser.parse_args().run_directory))
