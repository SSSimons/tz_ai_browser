"""Проверить сериализацию OpenAI SDK без настоящего ключа и сетевых запросов."""

import json

import httpx
from openai import AsyncOpenAI

from browser_agent.agent import Agent
from browser_agent.config import Settings
from browser_agent.state import RunLog


async def test_responses_sdk_roundtrip_with_image_and_verifier(tmp_path):
    requests = []

    async def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        step = len(requests)
        if step == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "browser_screenshot",
                    "arguments": "{}",
                    "status": "completed",
                }
            ]
        elif step == 2:
            assert body["input"][-1]["content"][-1]["type"] == "input_image"
            assert body["input"][-2]["type"] == "function_call_output"
            output = [
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call_2",
                    "name": "finish",
                    "arguments": json.dumps(
                        {"status": "completed", "summary": "Ready", "evidence": "Observed"}
                    ),
                    "status": "completed",
                }
            ]
        else:
            assert body["text"]["format"]["type"] == "json_schema"
            assert body["input"][0]["content"][-1]["type"] == "input_image"
            output = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"verified":true,"feedback":"ok"}',
                            "annotations": [],
                        }
                    ],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": f"resp_{step}",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.4-mini",
                "status": "completed",
                "output": output,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "total_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    class Browser:
        tools = [
            {
                "type": "function",
                "name": "browser_screenshot",
                "description": "See page",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ]

        async def call(self, name, arguments):
            if name == "browser_screenshot":
                return (
                    "Image",
                    [{"type": "input_image", "image_url": "data:image/png;base64,YQ=="}],
                    False,
                )
            return "Current result confirmed", [], False

    async def ask(question):
        raise AssertionError("Unexpected question")

    async with AsyncOpenAI(
        api_key="test-not-a-real-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    ) as client:
        agent = Agent(
            Settings(_env_file=None), client, Browser(), RunLog(tmp_path), ask, lambda _: None
        )
        result = await agent.run("Read current page")
    assert result["status"] == "completed"
    assert result["tokens"] == {"input": 30, "output": 30}
    assert all(r["model"] == "gpt-5.4-mini" and r["store"] is False for r in requests)
