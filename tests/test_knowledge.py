import json

import pytest
from test_agent import make_agent, response

from browser_agent.knowledge import Knowledge


def test_persistent_search_and_update(tmp_path):
    path = tmp_path / "facts.json"
    Knowledge(path).save_answer("Зарплата", "180000 RUB gross в месяц")
    found = Knowledge(path).search("зарплатные ожидания")
    assert found["facts"][0]["answer"] == "180000 RUB gross в месяц"
    assert Knowledge(path).search("salary")["facts"] == found["facts"]
    Knowledge(path).save_answer("Зарплата", "200000 RUB gross в месяц")
    assert len(Knowledge(path).read()["facts"]) == 1


def test_invalid_database_is_not_overwritten(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text('{"facts": "ошибка"}', encoding="utf-8")
    with pytest.raises(ValueError):
        Knowledge(path).save_answer("Вопрос", "Ответ")
    assert json.loads(path.read_text(encoding="utf-8"))["facts"] == "ошибка"


async def test_known_fact_defers_question_without_calling_user(tmp_path):
    agent, _, _ = make_agent(tmp_path, [])
    agent.knowledge.save_answer("Зарплата", "180000 рублей gross ежемесячно")

    async def unexpected(question):
        pytest.fail("Известные сведения нельзя повторно запрашивать")

    agent.ask = unexpected
    output, answer = await agent.ask_missing(
        {
            "question": "Ваши зарплатные ожидания?",
            "reason": "missing_fact",
            "missing_detail": "Нужна зарплата",
        }
    )
    assert answer is None
    assert json.loads(output)["facts"][0]["answer"].startswith("180000")
    assert json.loads(output)["ask_deferred"]


async def test_unknown_fact_saved_but_manual_input_not_saved(tmp_path):
    agent, _, _ = make_agent(tmp_path, [])
    output, answer = await agent.ask_missing(
        {
            "question": "Количество?",
            "reason": "missing_fact",
            "missing_detail": "Нет в базе или контексте",
        }
    )
    assert answer and json.loads(output)["user_answer"] == answer
    assert Knowledge(agent.settings.knowledge_path).search("Количество")["facts"]
    await agent.ask_missing(
        {"question": "Войдите в браузере", "reason": "manual_intervention", "missing_detail": ""}
    )
    assert len(agent.knowledge.read()["facts"]) == 1


async def test_salary_basis_can_be_clarified_after_reading_known_amount(tmp_path):
    agent, _, _ = make_agent(
        tmp_path,
        [
            response(
                text=json.dumps(
                    {
                        "needs_user": True,
                        "answer": "180000 рублей в месяц",
                        "missing_detail": "Не указано, до или после налогов",
                        "question": "180000 рублей - до или после налогов?",
                    }
                )
            )
        ],
    )
    agent.knowledge.save_answer("Зарплата", "180000 рублей в месяц")
    agent.search_knowledge("зарплата")
    _, answer = await agent.ask_missing(
        {
            "question": "180000 рублей до или после налогов?",
            "reason": "missing_fact",
            "missing_detail": "Для зарплаты неизвестно gross или net",
        }
    )
    assert answer is not None


async def test_repeated_known_question_is_blocked_and_verdict_cached(tmp_path):
    agent, model, _ = make_agent(
        tmp_path,
        [
            response(
                text=json.dumps(
                    {
                        "needs_user": False,
                        "answer": "180000 RUB gross в месяц",
                        "missing_detail": "",
                        "question": "",
                    }
                )
            )
        ],
    )
    agent.knowledge.save_answer("Зарплата", "180000 RUB gross в месяц")

    async def unexpected(question):
        pytest.fail("Ответ уже известен; повторный вопрос недопустим")

    agent.ask = unexpected
    arguments = {
        "question": "Зарплата?",
        "reason": "missing_fact",
        "missing_detail": "Нужна зарплата",
    }
    for _ in range(3):
        output, answer = await agent.ask_missing(arguments)
        assert json.loads(output)["ask_deferred"] and answer is None
        agent.rounds.append([{"type": "function_call_output", "call_id": "ask", "output": output}])
    assert len(model.calls) == 1
    assert agent.api_calls == 1


async def test_changed_fact_invalidates_question_verdict(tmp_path):
    agent, model, _ = make_agent(
        tmp_path,
        [
            response(
                text=json.dumps(
                    {
                        "needs_user": False,
                        "answer": "180000 gross",
                        "missing_detail": "",
                        "question": "",
                    }
                )
            ),
            response(
                text=json.dumps(
                    {
                        "needs_user": True,
                        "answer": "200000",
                        "missing_detail": "Не указана база расчёта",
                        "question": "200000 - gross или net?",
                    }
                )
            ),
        ],
    )
    arguments = {
        "question": "Зарплата?",
        "reason": "missing_fact",
        "missing_detail": "Нужна сумма и база расчёта",
    }
    agent.knowledge.save_answer("Зарплата", "180000 gross")
    agent.search_knowledge("зарплата")
    _, answer = await agent.ask_missing(arguments)
    assert answer is None
    agent.knowledge.save_answer("Зарплата", "200000")
    agent.search_knowledge("зарплата")
    _, answer = await agent.ask_missing(arguments)
    assert answer is not None and len(model.calls) == 2


async def test_question_verification_respects_api_limit(tmp_path):
    from browser_agent.agent import ApiCallLimitReached

    agent, _, _ = make_agent(tmp_path, [], max_api_calls=1)
    agent.api_calls = 1
    agent.knowledge.save_answer("Зарплата", "180000 gross")
    agent.search_knowledge("зарплата")
    with pytest.raises(ApiCallLimitReached):
        await agent.ask_missing(
            {"question": "Зарплата?", "reason": "missing_fact", "missing_detail": "Сумма"}
        )


async def test_knowledge_tool_is_available_in_model_loop(tmp_path):
    agent, model, _ = make_agent(
        tmp_path, [response("knowledge_search", {"query": "стек"})], max_steps=1
    )
    agent.knowledge.save_answer("Навыки и стек", "Python, SQL")
    await agent.run("Заполни анкету")
    assert "Навыки и стек" in model.calls[0]["input"][1]["content"]
    assert "Python, SQL" in agent.rounds[0][-1]["output"]
