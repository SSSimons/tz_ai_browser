import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_agent import make_agent, response

from browser_agent import server
from browser_agent.browser import Target
from browser_agent.input_policy import requires_manual_input
from browser_agent.state import bounded_observations


def observation(index):
    return [
        {
            "type": "function_call",
            "name": "browser_snapshot",
            "call_id": str(index),
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": str(index),
            "output": json.dumps(
                {"untrusted_page_content": True, "content": str(index) + "x" * 16000}
            ),
        },
    ]


async def test_large_observations_do_not_trigger_summary_every_step(tmp_path):
    agent, model, _ = make_agent(tmp_path, [response(text="Факты " + "x" * 5000)] * 10)
    sizes = []
    for index in range(12):
        agent.rounds.append(observation(index))
        await agent.compact()
        sizes.append(len(json.dumps(agent.history(), ensure_ascii=False)))
    assert 1 <= len(model.calls) <= 2
    assert max(sizes) < agent.settings.context_chars
    assert json.loads(agent.read_observation("0", 15000, 2000))["content"].endswith('"}')


def test_shortened_history_keeps_original_and_supports_reading_tail(tmp_path):
    agent, _, _ = make_agent(tmp_path, [])
    original = observation(1)
    agent.rounds.append(original)
    view = json.loads(agent.history()[-1]["output"])
    assert view["truncated"]
    restored = ""
    offset = 0
    while True:
        chunk = json.loads(agent.read_observation(view["read_observation_call_id"], offset, 5000))
        restored += chunk["content"]
        if chunk["next_offset"] is None:
            break
        offset = chunk["next_offset"]
    assert restored == original[-1]["output"]
    assert "truncated" not in json.loads(original[-1]["output"])


async def test_does_not_pay_for_summary_with_small_expected_reduction(tmp_path):
    agent, model, _ = make_agent(tmp_path, [])
    agent.task = "x" * 30000
    agent.rounds = [observation(index) for index in range(3)]
    await agent.compact()
    assert not model.calls
    assert len(agent.rounds) == 3


def test_shortening_preserves_failure_and_partial_actions():
    data = {
        "failure": {"ref": "b", "error": "Ошибка"},
        "completed_fields": [{"ref": "a"}],
        "observation": {"content": "x" * 20000},
    }
    result = bounded_observations(
        [{"type": "function_call_output", "call_id": "x", "output": json.dumps(data)}], 8000
    )
    shortened = json.loads(result[0]["output"])
    assert shortened["failure"] == data["failure"]
    assert shortened["completed_fields"] == data["completed_fields"]


async def test_visual_review_uses_fresh_image_and_fails_if_unavailable(tmp_path):
    agent, model, browser = make_agent(
        tmp_path, [response(text='{"verified":true,"feedback":"ok"}')]
    )
    agent.visual_verification = True
    browser.call = AsyncMock(
        side_effect=[
            ('{"content":"canvas"}', [], False),
            ("", [{"type": "input_image", "image_url": "data:image/png;base64,FRESH"}], False),
        ]
    )
    assert (await agent.review({"summary": "График построен"}))[0]
    content = model.calls[0]["input"][0]["content"]
    assert content[-1]["image_url"].endswith("FRESH")
    browser.call = AsyncMock(side_effect=[('{"content":"canvas"}', [], False), ("", [], True)])
    assert not (await agent.review({"summary": "График построен"}))[0]
    assert len(model.calls) == 1


async def test_visual_requirement_survives_checkpoint_and_resume(tmp_path):
    agent, _, _ = make_agent(tmp_path, [])
    agent.visual_verification = True
    agent.checkpoint("running")
    saved = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    resumed, _, _ = make_agent(tmp_path, [response("browser_snapshot")], max_steps=1)
    await resumed.run("Задача с canvas", saved)
    assert resumed.visual_verification


@pytest.mark.parametrize(
    "field_type,autocomplete",
    [
        ("password", None),
        ("text", "one-time-code"),
        ("text", "section-login ONE-TIME-CODE"),
    ],
)
async def test_ordinary_fill_blocks_manual_fields(monkeypatch, field_type, autocomplete):
    async def get_attribute(name):
        return {"type": field_type, "autocomplete": autocomplete}.get(name)

    element = SimpleNamespace(get_attribute=get_attribute, fill=AsyncMock())
    runtime = SimpleNamespace(
        locate=lambda _: element, perform=AsyncMock(), after_action=AsyncMock()
    )
    monkeypatch.setattr(server, "runtime", runtime)
    assert requires_manual_input(field_type, autocomplete)
    with pytest.raises(ValueError, match="вручную"):
        await server.browser_fill(Target(kind="label", value="Код"), "123456")
    element.fill.assert_not_called()
    runtime.perform.assert_not_called()
