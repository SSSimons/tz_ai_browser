"""Единая проверка полей, которые пользователь заполняет вручную."""


def requires_manual_input(field_type: str | None, autocomplete: str | None) -> bool:
    tokens = (autocomplete or "").lower().split()
    return (field_type or "").lower() == "password" or "one-time-code" in tokens


async def ensure_automatic_input(element):
    if requires_manual_input(
        await element.get_attribute("type"), await element.get_attribute("autocomplete")
    ):
        raise ValueError("Пароль или одноразовый код пользователь вводит вручную в браузере.")
