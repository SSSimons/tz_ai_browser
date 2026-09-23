"""Три ограниченные платные проверки исправлений на вымышленных локальных данных."""

import argparse
import asyncio
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import AsyncOpenAI

from browser_agent.agent import Agent
from browser_agent.cli import error_details
from browser_agent.config import Settings
from browser_agent.mcp_client import connect_browser
from browser_agent.state import RunLog, scrub


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        # Надпись доступна только в изображении canvas, а не в дереве доступности.
        body = """<!doctype html><html lang="ru"><meta charset="utf-8">
        <title>Локальная визуальная проверка</title>
        <h1>Тестовый экран</h1><canvas width="780" height="260"></canvas>
        <script>const c=document.querySelector('canvas').getContext('2d');
        c.fillStyle='#fff4f4';c.fillRect(0,0,780,260);
        c.fillStyle='#b00020';c.font='bold 34px Arial';
        c.fillText('ОТКЛИК НЕ ОТПРАВЛЕН',30,85);
        c.fillStyle='#222';c.font='26px Arial';
        c.fillText('Осталось заполнить анкету',30,145);
        </script></html>""".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


async def main(checks):
    directory = Path("artifacts") / ("live-review-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    settings = Settings(
        model="gpt-5.4-mini",
        max_api_calls=len(checks),
        max_output_tokens=1000,
        reasoning_effort="none",
        api_max_retries=0,
        headless=True,
        record_actions=False,
        profile_dir=directory / "profile",
        knowledge_path=directory / "test-facts.json",
        run_dir=directory,
    )
    if not settings.api_key.get_secret_value():
        raise RuntimeError("В .env не задан OPENAI_API_KEY")
    log = RunLog(directory, settings.api_key.get_secret_value())
    site = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    worker = threading.Thread(target=site.serve_forever, daemon=True)
    worker.start()
    questions = []
    result = {"status": "running", "selected_checks": checks, "checks": {}}
    agent = None

    async def simulated_user(question):
        questions.append(question)
        return "До налогов, gross. Это вымышленные данные тестового профиля."

    try:
        async with AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(), timeout=40, max_retries=0
        ) as client:
            async with connect_browser(settings) as browser:
                agent = Agent(settings, client, browser, log, simulated_user, print)
                agent.task = "Заполни поле зарплатных ожиданий в рублях gross за месяц."
                if "known" in checks:
                    agent.knowledge.save_answer("Зарплата", "180000 рублей gross в месяц")
                    agent.search_knowledge("зарплата")
                    output, answer = await agent.ask_missing(
                        {
                            "question": "Какую зарплату указать?",
                            "reason": "missing_fact",
                            "missing_detail": "Нужна сумма зарплаты, валюта, период и gross/net",
                        }
                    )
                    result["checks"]["known_fact"] = json.loads(output)
                    if (
                        answer is not None
                        or questions
                        or not json.loads(output).get("ask_deferred")
                    ):
                        raise AssertionError("Известная зарплата вызвала лишний вопрос")

                if "missing" in checks:
                    agent.knowledge.save_answer("Зарплата", "180000 рублей в месяц")
                    agent.search_knowledge("зарплата")
                    output, answer = await agent.ask_missing(
                        {
                            "question": "Указанная зарплата до или после налогов?",
                            "reason": "missing_fact",
                            "missing_detail": "Неизвестно gross или net",
                        }
                    )
                    result["checks"]["missing_basis"] = json.loads(output)
                    if answer is None or len(questions) != 1:
                        raise AssertionError("Не запрошена отсутствующая база расчёта")
                    if not any(word in questions[0].lower() for word in ("налог", "gross", "net")):
                        raise AssertionError("Уточнение не относится к недостающей базе расчёта")

                if "visual" in checks:
                    text, _, error = await browser.call(
                        "browser_navigate", {"url": f"http://127.0.0.1:{site.server_port}"}
                    )
                    if error:
                        raise RuntimeError(text)
                    if "ОТКЛИК НЕ ОТПРАВЛЕН" in text:
                        raise AssertionError("Надпись должна проверяться по изображению")
                    agent.task = "Проверь, подтверждена ли отправка отклика на экране."
                    agent.visual_verification = True
                    verified, feedback = await agent.review(
                        {
                            "status": "completed",
                            "summary": "Отклик успешно отправлен",
                            "evidence": "Исполнитель считает отправку успешной. Проверь экран.",
                        }
                    )
                    result["checks"]["reject_false_visual_success"] = {
                        "verified": verified,
                        "feedback": feedback,
                    }
                    if verified:
                        raise AssertionError("Проверяющий принял ложный успех вопреки изображению")

                result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = scrub(error_details(exc), settings.api_key.get_secret_value())
        raise
    finally:
        result["api_calls"] = agent.api_calls if agent else 0
        result["tokens"] = agent.tokens if agent else {"input": 0, "output": 0}
        (directory / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print(f"Отчёт: {directory.resolve() / 'result.json'}", flush=True)
        site.shutdown()
        site.server_close()
        worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Платная проверка: максимум 3 запроса без повторов"
    )
    parser.add_argument("--check", choices=["all", "known", "missing", "visual"], default="all")
    args = parser.parse_args()
    checks = ["known", "missing", "visual"] if args.check == "all" else [args.check]
    asyncio.run(main(checks))
