import json
import re
from datetime import datetime, timezone
from pathlib import Path


def scrub(value, secret: str = ""):
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED]", value)
        return value
    if isinstance(value, list):
        return [scrub(item, secret) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower() in {"password", "api_key", "authorization", "cookie"}
                else scrub(item, secret)
            )
            for key, item in value.items()
        }
    return value


def without_images(items):
    """Убрать изображения из запроса сжатия и сохраняемого состояния."""
    result = []
    for item in items:
        item = dict(item)
        if isinstance(item.get("content"), list):
            item["content"] = [
                {"type": "input_text", "text": "[Earlier screenshot omitted; request a fresh one]"}
                if block.get("type") == "input_image"
                else block
                for block in item["content"]
            ]
        result.append(item)
    return result


def deduplicate_observations(items):
    """Заменить полностью одинаковые снимки ссылкой на первый, не теряя содержимое."""
    seen = {}
    compact = []
    for item in items:
        output = item.get("output")
        if item.get("type") == "function_call_output" and isinstance(output, str):
            try:
                data = json.loads(output)
            except (ValueError, TypeError):
                data = None
            if isinstance(data, dict) and data.get("untrusted_page_content") is True:
                if output in seen:
                    item = {
                        **item,
                        "output": json.dumps(
                            {
                                "same_observation_as": seen[output],
                                "untrusted_page_content": True,
                            }
                        ),
                    }
                else:
                    seen[output] = item["call_id"]
        compact.append(item)
    return compact


def bounded_observations(items, context_chars: int):
    """Сократить только представление наблюдений; оригиналы остаются в истории."""
    candidates = []
    for index, item in enumerate(items):
        if item.get("type") != "function_call_output":
            continue
        try:
            data = json.loads(item.get("output", ""))
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or data.get("observation_chunk"):
            continue
        if data.get("untrusted_page_content") or isinstance(data.get("observation"), dict):
            candidates.append((index, data))
    result = list(items)
    for position, (index, data) in enumerate(candidates):
        limit = max(500, context_chars // (4 if position == len(candidates) - 1 else 16))
        original = items[index]["output"]
        if len(original) <= limit:
            continue
        shortened = {
            "untrusted_page_content": True,
            "truncated": True,
            "original_chars": len(original),
            "read_observation_call_id": items[index]["call_id"],
            "preview": original[:limit],
        }
        # Ошибки и частично выполненные действия нельзя скрывать за обрезкой текста.
        for key in ("error", "failure", "state_unknown", "completed_fields"):
            if key in data:
                shortened[key] = data[key]
        result[index] = {**items[index], "output": json.dumps(shortened, ensure_ascii=False)}
    return result


class RunLog:
    def __init__(self, directory: Path, secret: str = ""):
        self.directory = directory
        self.secret = secret
        directory.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **data):
        row = scrub(
            {"time": datetime.now(timezone.utc).isoformat(), "event": kind, **data}, self.secret
        )
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def checkpoint(self, task, memory, rounds, notes, status, *, visual_verification=False):
        payload = scrub(
            {
                "version": 1,
                "task": task,
                "memory": memory,
                "rounds": [without_images(r) for r in rounds],
                "notes": notes,
                "status": status,
                "visual_verification": visual_verification,
            },
            self.secret,
        )
        temporary = self.directory / "checkpoint.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "checkpoint.json")
