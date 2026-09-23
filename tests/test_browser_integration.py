import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from browser_agent.config import Settings
from browser_agent.mcp_client import connect_browser

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_BROWSER_TESTS") != "1",
        reason="Set RUN_BROWSER_TESTS=1 for real CDP/MCP test",
    ),
]


@pytest.fixture
def local_site():
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(Path(__file__).parent / "fixtures"))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("record_actions", [False, True])
async def test_real_stdio_mcp_stealth_cdp_playwright(tmp_path, local_site, record_actions):
    settings = Settings(
        _env_file=None,
        headless=True,
        profile_dir=tmp_path / "profile",
        run_dir=tmp_path / "run",
        record_actions=record_actions,
    )
    async with connect_browser(settings) as mcp:

        async def call(name, **args):
            text, images, error = await mcp.call(name, args)
            assert not error, text
            return text, images

        assert len(mcp.tools) >= 12
        text, _ = await call("browser_navigate", url=local_site)
        assert "Northstar" in text and "Вы выиграли" in text
        compact_chars = len(json.loads(text)["content"])
        assert "DOM metadata" not in text
        expanded, _ = await call("browser_snapshot", include_metadata=True)
        full_chars = len(json.loads(expanded)["content"])
        assert full_chars > compact_chars
        print(f"snapshot_chars: compact={compact_chars}, expanded={full_chars}")
        await call("browser_fill", target={"kind": "label", "value": "Поиск"}, text="Вы выиграли")
        await call(
            "browser_check",
            target={"kind": "role", "value": "checkbox", "name": "Вы выиграли миллион!"},
            checked=True,
        )
        text, _ = await call(
            "browser_click", target={"kind": "role", "value": "button", "name": "В корзину"}
        )
        assert "Переместить выбранные" in text
        text, _ = await call(
            "browser_click", target={"kind": "role", "value": "button", "name": "Переместить"}
        )
        assert "В корзине: 1" in text
        await call("browser_fill", target={"kind": "label", "value": "Поиск"}, text="")
        await call(
            "browser_select", target={"kind": "label", "value": "Категория"}, values=["unread"]
        )
        await call(
            "browser_click", target={"kind": "text", "value": "Дополнительные тестовые элементы"}
        )
        await call(
            "browser_fill",
            target={"kind": "label", "value": "Заметка", "frame": 1},
            text="MCP — работает",
        )
        text, _ = await call(
            "browser_click",
            target={"kind": "role", "value": "button", "name": "Сохранить", "frame": 1},
        )
        assert "MCP - работает" in text
        await call(
            "browser_click", target={"kind": "role", "value": "button", "name": "Загрузить статус"}
        )
        text, _ = await call(
            "browser_wait", target={"kind": "text", "value": "Синхронизация завершена"}
        )
        assert "Синхронизация завершена" in text
        text, _ = await call(
            "browser_click", target={"kind": "role", "value": "button", "name": "Подтверждение"}
        )
        assert "dialog" in text
        text, _ = await call("browser_dialog", accept=False)
        assert "Отменено" in text
        _, images = await call("browser_screenshot")
        assert images and images[0]["image_url"].startswith("data:image/png;base64,")
        text, _ = await call("browser_tab", action="new", url=local_site)
        assert "Northstar" in text
        await call("browser_tab", action="select", tab_id=0)
        await call("browser_press", key="Escape")
        await call("browser_scroll", direction="down", pixels=200)
        text, _, error = await mcp.call(
            "browser_click",
            {"target": {"kind": "role", "value": "button", "name": "Does not exist"}},
        )
        assert error  # Ошибка инструмента доходит до клиента, не завершая сеанс.
        text, _ = await call("browser_snapshot")
        assert "Northstar" in text
    assert bool(list((tmp_path / "run").glob("frame-*.png"))) == record_actions


async def test_questionnaire_fields_questions_validation_and_stale_refs(tmp_path, local_site):
    settings = Settings(
        _env_file=None,
        headless=True,
        profile_dir=tmp_path / "profile",
        run_dir=tmp_path / "run",
        record_actions=False,
    )
    async with connect_browser(settings) as mcp:

        async def call(name, **args):
            text, _, error = await mcp.call(name, args)
            assert not error, text
            return json.loads(text)

        await call("browser_navigate", url=local_site + "/questionnaire.html")
        form = await call("browser_inspect_form")
        motivation, salary, mode, confirmed, otp = form["fields"]
        assert "почему" in motivation["question"]
        assert "зарплатные" in salary["question"]
        assert motivation["placeholder"] == salary["placeholder"] == "Писать тут"
        assert salary["invalid"] and salary["required"]
        assert otp["manual"] and otp["value"] == "[скрыто]"
        result = await call(
            "browser_fill_fields",
            fields=[
                {"ref": motivation["ref"], "text": "Интересна разработка — использую Python."},
                {"ref": salary["ref"], "text": "180000 рублей gross в месяц"},
                {"ref": mode["ref"], "values": ["office"]},
                {"ref": confirmed["ref"], "checked": True},
            ],
        )
        assert result["failure"] is None
        assert len(result["completed_fields"]) == 4 and not result["submitted"]
        assert "Не отправлено" in result["observation"]["content"]
        filled = await call("browser_inspect_form")
        assert filled["fields"][0]["value"] == "Интересна разработка - использую Python."
        assert filled["fields"][1]["value"] == "180000 рублей gross в месяц"
        assert "городе" in filled["fields"][3]["question"]
        assert not filled["fields"][1]["invalid"]
        stale = await call("browser_fill_fields", fields=[{"ref": salary["ref"], "text": "0"}])
        assert stale["failure"] and not stale["completed_fields"]
        await call(
            "browser_click", target={"kind": "role", "value": "button", "name": "Перерисовать"}
        )
        detached = await call(
            "browser_fill_fields", fields=[{"ref": filled["fields"][0]["ref"], "text": "0"}]
        )
        assert detached["failure"] and not detached["completed_fields"]
        fresh = await call("browser_inspect_form")
        invalid = await call(
            "browser_fill_fields",
            fields=[
                {"ref": fresh["fields"][1]["ref"], "text": ""},
                {"ref": fresh["fields"][0]["ref"], "text": "Не выполнять после ошибки"},
            ],
        )
        assert invalid["failure"] and invalid["completed_fields"][0]["invalid"]
        final = await call("browser_inspect_form")
        assert final["fields"][0]["value"].startswith("Интересна разработка")
        blocked = await call(
            "browser_fill_fields", fields=[{"ref": final["fields"][-1]["ref"], "text": "654321"}]
        )
        assert blocked["failure"] and not blocked["completed_fields"]
        _, _, error = await mcp.call(
            "browser_fill",
            {"target": {"kind": "label", "value": "Одноразовый код"}, "text": "654321"},
        )
        assert error

        await call("browser_navigate", url=local_site + "/form-pages.html")
        first = await call("browser_inspect_form")
        assert first["next_offset"] == 30
        second = await call("browser_inspect_form", offset=first["next_offset"])
        iframe = await call("browser_inspect_form", frame=1)
        result = await call(
            "browser_fill_fields",
            fields=[
                {"ref": first["fields"][0]["ref"], "text": "Первый ответ"},
                {"ref": second["fields"][0]["ref"], "text": "Последний ответ"},
                {"ref": iframe["fields"][0]["ref"], "text": "Ответ в iframe"},
            ],
        )
        assert result["failure"] is None and len(result["completed_fields"]) == 3
        checked = await call("browser_inspect_form")
        assert checked["fields"][0]["value"] == "Первый ответ"
        checked = await call("browser_inspect_form", offset=30)
        assert checked["fields"][0]["value"] == "Последний ответ"
        checked = await call("browser_inspect_form", frame=1)
        assert checked["fields"][0]["value"] == "Ответ в iframe"
