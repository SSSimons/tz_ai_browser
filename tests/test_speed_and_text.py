import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from browser_agent import server
from browser_agent.browser import BrowserRuntime, Target
from browser_agent.config import Settings
from browser_agent.state import deduplicate_observations
from browser_agent.text import action_status


async def test_outgoing_text_is_normalized_without_changing_locator(monkeypatch):
    locator = SimpleNamespace(get_attribute=AsyncMock(return_value="text"), fill=AsyncMock())
    targets = []

    async def perform(operation):
        await operation

    def locate(target):
        targets.append(target)
        return locator

    runtime = SimpleNamespace(
        locate=locate, perform=perform, after_action=AsyncMock(return_value={"ok": True})
    )
    monkeypatch.setattr(server, "runtime", runtime)
    target = Target(kind="label", value="Имя — получателя")
    await server.browser_fill(target, "Добрый день — ответ готов\nА—Б")
    locator.fill.assert_awaited_once_with("Добрый день - ответ готов\nА-Б")
    assert targets[0].value == "Имя — получателя"


async def test_prompt_text_is_normalized(monkeypatch):
    page = object()
    dialog = SimpleNamespace(accept=AsyncMock())
    runtime = SimpleNamespace(
        dialogs={page: dialog},
        active=lambda: page,
        settle_dialog_action=AsyncMock(),
        after_action=AsyncMock(return_value={}),
    )
    monkeypatch.setattr(server, "runtime", runtime)
    await server.browser_dialog(True, "Текст — ответа")
    dialog.accept.assert_awaited_once_with("Текст - ответа")


async def test_default_actions_do_not_take_screenshots(tmp_path, monkeypatch):
    runtime = BrowserRuntime(Settings(_env_file=None, artifacts_dir=tmp_path))
    monkeypatch.setattr(runtime, "active", lambda: "page")
    runtime.capture = AsyncMock()
    runtime.snapshot = AsyncMock(return_value={"content": "result"})
    assert await runtime.after_action() == {"content": "result"}
    runtime.capture.assert_not_called()


async def test_recording_remains_available_when_enabled(tmp_path, monkeypatch):
    runtime = BrowserRuntime(Settings(_env_file=None, run_dir=tmp_path, record_actions=True))
    monkeypatch.setattr(runtime, "active", lambda: "page")
    runtime.capture = AsyncMock(return_value=b"png")
    runtime.snapshot = AsyncMock(return_value={})
    await runtime.after_action()
    assert (tmp_path / "frame-00001.png").read_bytes() == b"png"


async def test_frame_reads_overlap_and_metadata_is_on_demand():
    in_flight, peak = 0, 0

    async def tree(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return '- button "Продолжить"'

    frames = [
        SimpleNamespace(
            url=f"https://site.test/{i}",
            locator=lambda _: SimpleNamespace(aria_snapshot=tree),
            evaluate=AsyncMock(return_value=[{"id": "next"}]),
        )
        for i in range(6)
    ]
    page = Mock(
        url="https://site.test",
        frames=frames,
        is_closed=lambda: False,
        title=AsyncMock(return_value="Page"),
    )
    runtime = BrowserRuntime(Settings(_env_file=None))
    runtime.page, runtime.pages = page, {0: page}
    result = await runtime.snapshot()
    assert 1 < peak <= 4
    assert result["content"].index("FRAME 0") < result["content"].index("FRAME 5")
    for frame in frames:
        frame.evaluate.assert_not_awaited()
    expanded = await runtime.snapshot(include_metadata=True)
    assert '"id": "next"' in expanded["content"]
    assert len(expanded["content"]) > len(result["content"])


def test_duplicate_observations_are_lossless_and_originals_are_unchanged():
    text = json.dumps({"untrusted_page_content": True, "content": "Важный факт " * 200})
    items = [{"type": "function_call_output", "call_id": str(i), "output": text} for i in range(5)]
    compact = deduplicate_observations(items)
    assert compact[0] == items[0]
    for i in range(1, 5):
        assert json.loads(compact[i]["output"])["same_observation_as"] == "0"
        assert items[i]["output"] == text
    assert len(json.dumps(compact)) < len(json.dumps(items)) / 3


def test_status_is_short_and_does_not_print_field_content_or_url_secrets():
    status = action_status(
        "browser_fill", {"target": {"value": "Сообщение"}, "text": "Личный текст " * 200}
    )
    assert status == "Заполняю поле: Сообщение"
    assert action_status(
        "browser_navigate", {"url": "https://user:pass@site.test/?key=secret"}
    ) == ("Открываю сайт: site.test")
    assert len(action_status("browser_click", {"target": {"name": "x" * 500}})) < 90
