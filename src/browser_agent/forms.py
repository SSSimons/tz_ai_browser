"""Распознавание полей по живому DOM без правил для отдельных сайтов."""

from contextlib import suppress
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from browser_agent.input_policy import ensure_automatic_input, requires_manual_input
from browser_agent.text import outbound_text

CONTROL_SELECTOR = "input:not([type=hidden]),textarea,select,[contenteditable=true]"
DESCRIBE = """e => {
    const text = n => (n?.innerText || n?.textContent || '').trim().slice(0, 700);
    const ids = name => (e.getAttribute(name) || '').split(/\\s+/)
        .map(id => text(e.ownerDocument.getElementById(id))).filter(Boolean).join(' ');
    const label = [...(e.labels || [])].map(text).join(' ') ||
        ids('aria-labelledby') || e.getAttribute('aria-label') || '';
    let question = '';
    for (let node = e, depth = 0; node && depth < 5; node = node.parentElement, depth++) {
        const parts = [];
        for (let prev = node.previousElementSibling, n = 0; prev && n < 3;
             prev = prev.previousElementSibling, n++) {
            if (prev.matches('input,textarea,select') ||
                prev.querySelector('input,textarea,select')) break;
            if (text(prev)) parts.unshift(text(prev));
        }
        if (parts.length) { question = parts.join(' ').slice(-900); break; }
    }
    const sensitive = e.type === 'password' ||
        (e.autocomplete || '').toLowerCase().split(/\\s+/).includes('one-time-code');
    return {
        tag: e.tagName.toLowerCase(), type: e.type || '', autocomplete: e.autocomplete || '',
        label, question,
        name: e.name || '', placeholder: e.getAttribute('placeholder') || '',
        description: ids('aria-describedby'),
        required: !!e.required || e.getAttribute('aria-required') === 'true',
        disabled: !!e.disabled || e.getAttribute('aria-disabled') === 'true',
        readonly: !!e.readOnly, manual: sensitive,
        value: sensitive ? '[скрыто]' : (e.value ?? e.innerText ?? '').slice(0, 2000),
        checked: ['checkbox','radio'].includes(e.type) ? e.checked : null,
        invalid: e.getAttribute('aria-invalid') === 'true' || e.validity?.valid === false,
        validation: e.validationMessage || '',
        options: e.options ? [...e.options].map(o => ({value:o.value, text:text(o),
            disabled:o.disabled, selected:o.selected})) : undefined
    };
}"""


class FieldAnswer(BaseModel):
    ref: str
    text: str | None = Field(default=None, max_length=12000)
    checked: bool | None = None
    values: list[str] | None = None

    @model_validator(mode="after")
    def one_action(self):
        if sum(v is not None for v in (self.text, self.checked, self.values)) != 1:
            raise ValueError("Укажите ровно одно действие: text, checked или values.")
        return self


class Forms:
    def __init__(self, runtime):
        self.runtime = runtime
        self.refs = {}
        self.scopes = {}
        self.page = None

    async def clear(self):
        for handle, _, _ in self.refs.values():
            with suppress(Exception):
                await handle.dispose()
        self.refs.clear()
        self.scopes.clear()
        self.page = None

    async def inspect(self, frame: int = 0, offset: int = 0, limit: int = 30) -> dict:
        page = self.runtime.active()
        if frame < 0 or frame >= len(page.frames) or offset < 0 or not 1 <= limit <= 50:
            raise ValueError("Проверьте frame, offset и limit (1-50).")
        if self.runtime.dialogs.get(page):
            return await self.runtime.snapshot()
        # Части одной анкеты и разные iframe сохраняют независимые ссылки.
        if page != self.page:
            await self.clear()
            self.page = page
        scope = page.frames[frame]
        modal = await self.runtime.action_scope(scope)
        control_scope = modal if modal is not None else scope
        handles = await control_scope.locator(CONTROL_SELECTOR).element_handles()
        visible = []
        for handle in handles:
            if await handle.is_visible():
                visible.append(handle)
            else:
                await handle.dispose()
        fields = []
        # Обновляем только перечитанный диапазон текущего iframe.
        for ref, (ref_scope, index) in list(self.scopes.items()):
            if ref_scope == scope and (offset <= index < offset + limit or index >= len(visible)):
                with suppress(Exception):
                    await self.refs[ref][0].dispose()
                del self.refs[ref]
                del self.scopes[ref]
        for index, handle in enumerate(visible):
            if not offset <= index < offset + limit:
                await handle.dispose()
                continue
            data = await handle.evaluate(DESCRIBE)
            data["manual"] = requires_manual_input(data["type"], data["autocomplete"])
            ref = uuid4().hex[:12]
            self.refs[ref] = (handle, page, data)
            self.scopes[ref] = (scope, index)
            fields.append({"ref": ref, **data})
        return {
            "url": page.url,
            "frame": frame,
            "fields": fields,
            "next_offset": offset + limit if len(visible) > offset + limit else None,
            "total_fields": len(visible),
            "untrusted_page_content": True,
            "note": "Следующие части и iframe сохраняют прежние ref. Повторное чтение диапазона "
            "обновляет только его ref. required отражает разметку, учитывайте текст сайта.",
        }

    async def fill(self, fields: list[FieldAnswer]) -> dict:
        if not 1 <= len(fields) <= 30 or len({f.ref for f in fields}) != len(fields):
            raise ValueError("Нужно 1-30 разных полей из осмотренных частей анкеты.")
        completed = []
        failure = None
        for answer in fields:
            try:
                if answer.ref not in self.refs:
                    raise ValueError("Ссылка на поле устарела. Снова осмотрите форму.")
                handle, page, before = self.refs[answer.ref]
                if page != self.runtime.active() or not await handle.evaluate("e => e.isConnected"):
                    raise ValueError("Поле устарело. Снова осмотрите форму.")
                ref_scope = self.scopes[answer.ref][0]
                await self.runtime.ensure_active_element(ref_scope, handle)
                current = await handle.evaluate(DESCRIBE)
                if any(current[k] != before[k] for k in ("label", "question", "name", "type")):
                    raise ValueError("Вопрос изменился. Снова осмотрите форму.")
                if current["manual"] or current["disabled"] or current["readonly"]:
                    raise ValueError("Поле требует ручного ввода или недоступно.")
                await ensure_automatic_input(handle)
                if answer.text is not None:
                    expected = outbound_text(answer.text)
                    operation = handle.fill(expected, timeout=5000)
                elif answer.checked is not None:
                    operation = handle.set_checked(answer.checked, timeout=5000)
                else:
                    operation = handle.select_option(answer.values, timeout=5000)
                await self.runtime.perform(operation)
                if self.runtime.dialogs.get(page):
                    raise ValueError(
                        "Открыт диалог. Проверьте результат текущего поля после его закрытия."
                    )
                # Смена фокуса запускает проверку и сохранение у динамических форм.
                await self.runtime.perform(handle.evaluate("e => e.blur()"))
                if self.runtime.dialogs.get(page):
                    raise ValueError("Открыт диалог при проверке поля.")
                after = await handle.evaluate(DESCRIBE)
                actual = await handle.evaluate("e => e.value ?? e.innerText ?? ''")
                if answer.text is not None and actual != expected:
                    raise ValueError("Значение изменилось после ввода. Проверьте форму.")
                if answer.checked is not None and after["checked"] != answer.checked:
                    raise ValueError("Состояние отметки не сохранилось.")
                if answer.values is not None and sorted(answer.values) != sorted(
                    o["value"] for o in after.get("options", []) if o["selected"]
                ):
                    raise ValueError("Выбор не сохранился.")
                completed.append(
                    {
                        "ref": answer.ref,
                        "invalid": after["invalid"],
                        "validation": after["validation"],
                    }
                )
                if after["invalid"]:
                    raise ValueError("Сайт сообщает об ошибке поля; исправьте её перед отправкой.")
            except Exception as exc:
                failure = {
                    "ref": answer.ref,
                    "error": str(exc),
                    "next": "Осмотрите форму; не повторяйте уже выполненные поля вслепую.",
                }
                break
        return {
            "completed_fields": completed,
            "failure": failure,
            "submitted": False,
            "observation": await self.runtime.after_action(),
        }
