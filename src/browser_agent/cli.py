import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI
from rich.console import Console

from browser_agent.agent import Agent
from browser_agent.browser import BrowserRuntime, find_browser
from browser_agent.config import Settings
from browser_agent.mcp_client import connect_browser
from browser_agent.state import RunLog

console = Console(markup=False)


def error_details(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "\n".join(error_details(child) for child in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


async def ask(question: str) -> str:
    console.print(f"\nАгент: {question}", style="bold yellow")
    return await asyncio.to_thread(input, "Вы: ")


async def run(settings: Settings, args) -> int:
    if args.command == "doctor":
        checks = {
            "Python": sys.version.split()[0],
            "Модель": settings.model,
            "API-ключ задан": bool(settings.api_key.get_secret_value()),
            "Браузер": settings.browser_executable or find_browser() or "Chrome не найден",
            "CDP": settings.cdp_url or "SeleniumBase запустит отдельный браузер",
            "Профиль": str(settings.profile_dir.resolve()),
        }
        console.print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0
    if args.command == "login":
        settings.headless = False
        browser = BrowserRuntime(settings)
        try:
            await browser.start()
            console.print(
                "Браузер открыт. Войдите в нужные аккаунты вручную. "
                "Профиль сохранится для следующих запусков."
            )
            await asyncio.to_thread(input, "После входа нажмите Enter, чтобы закрыть браузер: ")
        finally:
            await browser.close()
        return 0
    if not settings.api_key.get_secret_value():
        console.print(
            "OPENAI_API_KEY не задан. Скопируйте .env.example в .env и задайте ключ.", style="red"
        )
        return 2
    saved = None
    if args.resume:
        saved = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        if saved.get("version") != 1:
            raise ValueError("Unsupported checkpoint version")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    settings.run_dir = settings.artifacts_dir / f"{stamp}-{uuid4().hex[:6]}"
    log = RunLog(settings.run_dir, settings.api_key.get_secret_value())
    console.print(f"Модель: {settings.model}\nАртефакты: {settings.run_dir.resolve()}")
    console.print("Открываю браузер и подключаю MCP…")
    client_options = {
        "api_key": settings.api_key.get_secret_value(),
        "timeout": 90,
        "max_retries": settings.api_max_retries,
    }
    if settings.openai_base_url:
        client_options["base_url"] = settings.openai_base_url
    async with AsyncOpenAI(**client_options) as client:
        async with connect_browser(settings) as browser:
            console.print(f"MCP подключён: {len(browser.tools)} инструментов.")
            task = saved["task"] if saved else args.task
            while True:
                if not task:
                    task = await asyncio.to_thread(input, "\nЗадача (/exit - выход): ")
                if task.strip().lower() in {"/exit", "/quit"}:
                    return 0
                if not task.strip():
                    task = None
                    continue
                agent = Agent(settings, client, browser, log, ask, console.print)
                result = await agent.run(task, saved)
                status = {
                    "completed": "Готово",
                    "blocked": "Нужна помощь",
                    "incomplete": "Остановлено",
                }.get(result["status"], result["status"])
                console.print(f"\n{status}: {result['summary']}", style="bold")
                console.print(
                    f"Токены: вход {result['tokens']['input']}, выход {result['tokens']['output']}"
                )
                if result["status"] != "completed":
                    console.print(
                        f'Продолжить: browser-agent --resume "{log.directory / "checkpoint.json"}"'
                    )
                if args.task or args.resume:
                    return 0 if result["status"] == "completed" else 1
                # Браузер общий, но состояние каждой задачи сохраняется отдельно.
                task, saved = None, None
                log = RunLog(
                    settings.run_dir / f"task-{uuid4().hex[:6]}",
                    settings.api_key.get_secret_value(),
                )


def main():
    parser = argparse.ArgumentParser(description="Автономный браузерный AI-агент через MCP")
    parser.add_argument("command", nargs="?", choices=["chat", "login", "doctor"], default="chat")
    parser.add_argument("--task", help="Выполнить одну задачу и завершиться")
    parser.add_argument("--resume", help="Продолжить задачу из checkpoint.json")
    parser.add_argument("--headless", action="store_true", help="Скрытый браузер для тестов")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.task and args.resume:
        parser.error("--task и --resume взаимоисключающие")
    settings = Settings()
    if args.headless:
        settings.headless = True
    if args.max_steps is not None:
        if not 1 <= args.max_steps <= 1000:
            parser.error("--max-steps должен быть от 1 до 1000")
        settings.max_steps = args.max_steps
    try:
        code = asyncio.run(run(settings, args))
    except (KeyboardInterrupt, EOFError):
        console.print("\nОстановлено. Завершённые шаги сохранены в checkpoint.json.")
        code = 130
    except Exception as exc:
        from browser_agent.state import scrub

        console.print(
            f"Ошибка: {scrub(error_details(exc), settings.api_key.get_secret_value())}\n"
            "Подробности браузера: artifacts/<сеанс>/mcp-server.log",
            style="red",
        )
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
