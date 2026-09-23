import json
from types import SimpleNamespace

import pytest

from browser_agent.agent import Agent
from browser_agent.config import Settings
from browser_agent.state import RunLog


class Item:
    def __init__(self, **data):
        self.__dict__.update(data)

    def model_dump(self, **kwargs):
        return vars(self)


def response(name=None, arguments=None, text="", extra=None):
    output = extra or []
    if name:
        output = [
            *output,
            Item(
                type="function_call",
                name=name,
                call_id=f"call_{name}",
                arguments=json.dumps(arguments or {}),
            ),
        ]
    return SimpleNamespace(output=output, output_text=text, status="completed", usage=None)


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.responses = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.replies)


class Browser:
    tools = []

    def __init__(self):
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return '{"title":"Completed","content":"3 items processed"}', [], False


def make_agent(tmp_path, replies, max_steps=10, **settings):
    model, browser = Model(replies), Browser()

    async def ask(question):
        return "Пользователь уточнил: только три элемента"

    agent = Agent(
        Settings(
            _env_file=None, max_steps=max_steps, knowledge_path=tmp_path / "facts.json", **settings
        ),
        model,
        browser,
        RunLog(tmp_path),
        ask,
        lambda text: None,
    )
    return agent, model, browser


async def test_full_loop_preserves_reasoning_and_verifies_result(tmp_path):
    reasoning = Item(type="reasoning", id="rs_1", summary=[], encrypted_content="opaque")
    agent, model, browser = make_agent(
        tmp_path,
        [
            response("browser_snapshot", extra=[reasoning]),
            response(
                "ask_user",
                {
                    "question": "Сколько элементов?",
                    "reason": "missing_fact",
                    "missing_detail": "Количество не указано",
                },
            ),
            response("finish", {"status": "completed", "summary": "Готово", "evidence": "3 items"}),
            response(text='{"verified":true,"feedback":"Confirmed"}'),
        ],
    )
    result = await agent.run("Обработай элементы")
    assert result["status"] == "completed"
    assert any(item.get("encrypted_content") == "opaque" for item in model.calls[1]["input"])
    assert len(browser.calls) == 2  # Начальный снимок и независимая финальная проверка.
    assert all(call["model"] == "gpt-5.4-mini" for call in model.calls)
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"


async def test_rejected_success_continues_and_does_not_claim_done(tmp_path):
    candidate = {"status": "completed", "summary": "Готово", "evidence": "clicked"}
    agent, _, browser = make_agent(
        tmp_path,
        [
            response("finish", candidate),
            response(text='{"verified":false,"feedback":"Need visible confirmation"}'),
            response("browser_snapshot"),
        ],
        max_steps=2,
    )
    result = await agent.run("Обработай элементы")
    assert result["status"] == "incomplete"
    assert len(browser.calls) == 2


async def test_malformed_call_is_returned_to_model(tmp_path):
    broken = response("browser_click")
    broken.output[0].arguments = "{bad json"
    agent, model, _ = make_agent(tmp_path, [broken, response("browser_snapshot")], max_steps=2)
    await agent.run("Задача")
    outputs = [i for i in model.calls[1]["input"] if i.get("type") == "function_call_output"]
    assert "error" in outputs[0]["output"]


async def test_compaction_keeps_complete_call_output_pairs(tmp_path):
    agent, _, _ = make_agent(
        tmp_path, [response(text="Facts and completed actions")], context_chars=8000
    )
    agent.task = "original user request"
    for index in range(5):
        agent.rounds.append(
            [
                {
                    "type": "function_call",
                    "call_id": str(index),
                    "name": "browser_snapshot",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": str(index), "output": "x" * 2500},
            ]
        )
    await agent.compact()
    assert len(agent.rounds) == 2
    assert agent.memory == "Facts and completed actions"
    assert agent.history()[0]["content"] == "original user request"
    for round_ in agent.rounds:
        assert round_[0]["call_id"] == round_[1]["call_id"]


async def test_resume_requires_fresh_observation(tmp_path):
    agent, model, _ = make_agent(tmp_path, [response("browser_snapshot")], max_steps=1)
    await agent.run("Original task", {"memory": "Already clicked", "notes": [], "rounds": []})
    assert "Already clicked" in model.calls[0]["input"][1]["content"]
    assert "Сеанс возобновлен" in model.calls[0]["input"][2]["content"]


async def test_repeated_actions_are_not_executed_forever(tmp_path):
    agent, _, browser = make_agent(tmp_path, [response("browser_click", {"x": 1})] * 5, max_steps=5)
    result = await agent.run("Task")
    assert len(browser.calls) == 3
    assert result["status"] == "incomplete"


async def test_api_failure_preserves_checkpoint(tmp_path):
    agent, _, _ = make_agent(tmp_path, [])
    with pytest.raises((RuntimeError, StopIteration)):
        await agent.run("Unfinished task")
    saved = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert saved["status"] == "interrupted"


async def test_request_limit_includes_verification_and_preserves_result_status(tmp_path):
    agent, model, _ = make_agent(
        tmp_path,
        [
            response("browser_snapshot"),
            response("finish", {"status": "completed", "summary": "Done", "evidence": "Seen"}),
        ],
        max_api_calls=2,
        max_output_tokens=400,
    )
    result = await agent.run("Task")
    assert result["status"] == "incomplete"
    assert len(model.calls) == 2
    assert all(call["max_output_tokens"] == 400 for call in model.calls)
    saved = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert saved["status"] == "incomplete"


async def test_compaction_also_counts_against_request_limit(tmp_path):
    from browser_agent.agent import ApiCallLimitReached

    agent, model, _ = make_agent(tmp_path, [response("browser_snapshot")], max_api_calls=1)
    await agent.request(input="test", max_output_tokens=1000)
    with pytest.raises(ApiCallLimitReached):
        await agent.request(input="summary", max_output_tokens=1000)
    assert len(model.calls) == 1


async def test_plain_text_completion_requires_verification(tmp_path):
    agent, model, browser = make_agent(
        tmp_path,
        [
            response(text="Три элемента обработаны"),
            response(text='{"verified":true,"feedback":"Confirmed on the page"}'),
        ],
        max_api_calls=2,
    )
    result = await agent.run("Обработай три элемента")
    assert result["status"] == "completed"
    assert len(model.calls) == 2
    assert browser.calls == [("browser_snapshot", {})]


async def test_plain_text_claim_is_not_success_when_reviewer_disagrees(tmp_path):
    agent, _, _ = make_agent(
        tmp_path,
        [
            response(text="Я выполню задачу"),
            response(text='{"verified":false,"feedback":"Only an intention"}'),
        ],
        max_steps=1,
    )
    result = await agent.run("Task")
    assert result["status"] == "incomplete"


async def test_progress_reports_current_activity_without_model_monologue(tmp_path):
    agent, _, _ = make_agent(
        tmp_path,
        [
            response(
                "browser_fill",
                {"target": {"kind": "label", "value": "Сообщение"}, "text": "Личный текст"},
                text="Длинное объяснение модели",
            ),
        ],
        max_steps=1,
    )
    output = []
    agent.emit = output.append
    await agent.run("Заполни поле")
    assert output == ["[1] Выбираю следующее действие", "Заполняю поле: Сообщение"]
