from urllib.parse import urlsplit


def outbound_text(text: str) -> str:
    """Заменить длинное тире только в тексте, вводимом в браузер."""
    return text.replace("—", "-")


def short_label(value: str, limit: int = 64) -> str:
    """Убрать переносы и управляющие символы из краткого статуса."""
    value = " ".join("".join(c for c in value if c.isprintable() or c.isspace()).split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def action_status(name: str, arguments: dict) -> str:
    """Показать действие и цель, не выводя отправляемый текст и данные формы."""
    target = arguments.get("target") or {}
    label = short_label(str(target.get("name") or target.get("value") or ""))
    actions = {
        "browser_snapshot": "Читаю страницу",
        "browser_navigate": "Открываю сайт",
        "browser_click": "Нажимаю",
        "browser_fill": "Заполняю поле",
        "browser_inspect_form": "Читаю вопросы анкеты",
        "browser_fill_fields": "Заполняю анкету",
        "knowledge_search": "Ищу сведения в базе знаний",
        "read_observation": "Читаю сохранённое наблюдение",
        "browser_select": "Выбираю значение",
        "browser_check": "Меняю отметку",
        "browser_press": "Нажимаю клавишу",
        "browser_scroll": "Прокручиваю страницу",
        "browser_wait": "Жду загрузки",
        "browser_screenshot": "Смотрю скриншот",
        "browser_pointer": "Работаю с элементом на экране",
        "browser_tab": "Управляю вкладками",
        "browser_back": "Возвращаюсь назад",
        "browser_dialog": "Обрабатываю диалог",
        "ask_user": "Нужно уточнение",
        "remember": "Сохраняю заметку",
        "finish": "Завершаю задачу",
    }
    if name == "browser_navigate":
        try:
            label = short_label(urlsplit(arguments.get("url", "")).hostname or "")
        except ValueError:
            label = ""
    elif name == "browser_press":
        label = short_label(str(arguments.get("key", "")))
    action = actions.get(name, "Выполняю действие")
    return f"{action}: {label}" if label else action
