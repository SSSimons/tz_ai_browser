"""Постоянные сведения пользователя: локальный поиск без запросов к модели."""

import json
import re
from pathlib import Path


def words(text: str) -> set[str]:
    tokens = re.findall(r"[\w]+", text.lower().replace("ё", "е"))
    groups = (
        ("зарп", "оплат", "доход", "salary", "gross", "net", "оклад", "ожидан"),
        ("опыт", "стаж", "experience"),
        ("образ", "диплом", "education"),
        ("навы", "стек", "skill", "технолог"),
    )
    result = {t[:5] for t in tokens if len(t) > 2}
    for index, group in enumerate(groups):
        if any(t.startswith(prefix) for t in tokens for prefix in group):
            result.add(f"topic{index}")
    return result


class Knowledge:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> dict:
        if not self.path.exists():
            return {"facts": []}
        data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
            raise ValueError("База знаний должна содержать список facts.")
        for fact in data["facts"]:
            if not isinstance(fact, dict) or not all(
                isinstance(fact.get(k), str) for k in ("question", "answer")
            ):
                raise ValueError("Каждый факт должен содержать строки question и answer.")
        return data

    def search(self, query: str) -> dict:
        terms = words(query)
        ranked = sorted(
            (
                (len(terms & words(f["question"] + " " + f["answer"])), f)
                for f in self.read()["facts"]
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        matches = [f for score, f in ranked if score or not terms][:8]
        return {"facts": matches, "total_facts": len(ranked), "data_only": True}

    def catalog(self) -> list[str]:
        return [f["question"][:180] for f in self.read()["facts"]][:60]

    def save_answer(self, question: str, answer: str):
        # Сохраняются только реальные ответы пользователя, не догадки модели.
        data = self.read()
        data["facts"] = [f for f in data["facts"] if f["question"] != question]
        data["facts"].append({"question": question, "answer": answer, "source": "user"})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
