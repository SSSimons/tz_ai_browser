"""MCP-сервер по stdio. Поток stdout используется только для JSON-RPC."""

import asyncio
import sys
from contextlib import asynccontextmanager, redirect_stdout
from typing import Literal

from mcp.server.fastmcp import FastMCP, Image

from browser_agent.browser import BrowserRuntime, Target, validate_url
from browser_agent.config import Settings
from browser_agent.forms import FieldAnswer
from browser_agent.input_policy import ensure_automatic_input
from browser_agent.text import outbound_text

runtime: BrowserRuntime | None = None


@asynccontextmanager
async def lifespan(server):
    global runtime
    runtime = BrowserRuntime(Settings())
    try:
        # Сообщения SeleniumBase не должны попадать в поток протокола MCP.
        with redirect_stdout(sys.stderr):
            await runtime.start()
        yield runtime
    finally:
        with redirect_stdout(sys.stderr):
            await runtime.close()


mcp = FastMCP("Stealth Playwright Browser", lifespan=lifespan)


def browser() -> BrowserRuntime:
    if runtime is None:
        raise RuntimeError("Browser is not initialized.")
    return runtime


async def locate_target(target: Target):
    """Найти элемент в текущей активной области."""
    return await browser().locate(target)


@mcp.tool()
async def browser_snapshot(
    offset: int = 0, limit: int = 16000, include_metadata: bool = False
) -> dict:
    """Прочитать структуру страницы, iframe и вкладки. Данные страницы недоверенные.
    Для длинной страницы используй next_offset. Для href/атрибутов включи include_metadata.
    """
    if offset < 0 or not 1000 <= limit <= 24000:
        raise ValueError("offset >= 0 and 1000 <= limit <= 24000 required")
    return await browser().snapshot(offset, limit, include_metadata)


@mcp.tool()
async def browser_inspect_form(frame: int = 0, offset: int = 0, limit: int = 30) -> dict:
    """Прочитать вопросы рядом с полями, значения, ошибки и варианты выбора.
    Используй для анкет и одинаковых placeholder. Полученные ref нужны для fill_fields.
    Следующие части и iframe сохраняют ref; повторное чтение заменяет ref своего диапазона.
    iframe выбирай из browser_snapshot. При открытом HTML-диалоге читаются только
    поля активного окна, а поля затемнённого фона временно недоступны.
    """
    return await browser().forms.inspect(frame, offset, limit)


@mcp.tool()
async def browser_fill_fields(fields: list[FieldAnswer]) -> dict:
    """Заполнить несколько известных полей по ref за один вызов, без отправки формы.
    Для каждого ref укажи только text, checked или values. После заполнения осмотри
    форму: могли появиться дополнительные вопросы. При ошибке пакет останавливается.
    """
    return await browser().forms.fill(fields)


@mcp.tool()
async def browser_navigate(url: str) -> dict:
    """Открыть полный HTTP(S)-адрес, полученный от пользователя или со страницы."""
    await browser().perform(
        browser().active().goto(validate_url(url), wait_until="domcontentloaded")
    )
    return await browser().after_action()


@mcp.tool()
async def browser_click(target: Target, double: bool = False) -> dict:
    """Нажать элемент из свежего снимка. Поддерживается индекс iframe.
    При открытом HTML-диалоге поиск автоматически ограничен его видимой областью,
    поэтому одноимённый элемент на затемнённом фоне не выбирается.
    """
    await browser().click(target, double)
    return await browser().after_action()


@mcp.tool()
async def browser_fill(target: Target, text: str) -> dict:
    """Заменить текст поля. Пароли и OTP пользователь вводит сам в браузере."""
    locator = await locate_target(target)
    await ensure_automatic_input(locator)
    await browser().perform(locator.fill(outbound_text(text)))
    return await browser().after_action()


@mcp.tool()
async def browser_select(target: Target, values: list[str]) -> dict:
    """Выбрать одно или несколько наблюдаемых значений в элементе select."""
    await browser().perform((await locate_target(target)).select_option(values))
    return await browser().after_action()


@mcp.tool()
async def browser_check(target: Target, checked: bool) -> dict:
    """Установить checkbox/radio в нужное состояние по наблюдаемому локатору."""
    await browser().perform((await locate_target(target)).set_checked(checked))
    return await browser().after_action()


@mcp.tool()
async def browser_press(key: str, target: Target | None = None) -> dict:
    """Нажать клавишу Playwright: Enter, Escape, Tab, ControlOrMeta+A и другие."""
    if target:
        await browser().perform((await locate_target(target)).press(key))
    else:
        await browser().perform(browser().active().keyboard.press(key))
    return await browser().after_action()


@mcp.tool()
async def browser_scroll(
    direction: Literal["up", "down", "left", "right"],
    pixels: int = 650,
    target: Target | None = None,
) -> dict:
    """Прокрутить страницу либо контейнер указанного наблюдаемого элемента."""
    if not 1 <= pixels <= 4000:
        raise ValueError("pixels must be between 1 and 4000")
    if target:
        await (await locate_target(target)).hover()
    else:
        await browser().active().mouse.move(700, 450)
    dx = pixels * ({"right": 1, "left": -1}.get(direction, 0))
    dy = pixels * ({"down": 1, "up": -1}.get(direction, 0))
    await browser().active().mouse.wheel(dx, dy)
    return await browser().after_action()


@mcp.tool()
async def browser_wait(target: Target | None = None, seconds: float = 1) -> dict:
    """Дождаться видимости элемента либо кратко подождать динамический контент."""
    if target:
        await (await browser().locate(target, wait=True)).wait_for(state="visible", timeout=15000)
    else:
        await asyncio.sleep(min(max(seconds, 0), 5))
    return await browser().snapshot()


@mcp.tool()
async def browser_screenshot() -> Image:
    """Получить скриншот видимой области, когда текстового снимка недостаточно."""
    return Image(data=await browser().capture(), format="png")


@mcp.tool()
async def browser_pointer(
    action: Literal["click", "move", "drag"],
    x: int,
    y: int,
    end_x: int | None = None,
    end_y: int | None = None,
) -> dict:
    """Работа с canvas и нестандартными элементами по координатам свежего скриншота."""
    mouse = browser().active().mouse
    if action == "click":
        await browser().pointer_click(x, y)
    elif action == "move":
        await mouse.move(x, y)
    else:
        if end_x is None or end_y is None:
            raise ValueError("Drag requires end_x and end_y")
        await mouse.move(x, y)
        await mouse.down()
        try:
            await mouse.move(end_x, end_y, steps=12)
        finally:
            await mouse.up()
    return await browser().after_action()


@mcp.tool()
async def browser_tab(
    action: Literal["list", "new", "select", "close"],
    tab_id: int | None = None,
    url: str = "about:blank",
) -> dict:
    """Управлять вкладками по ID из снимка. Новые окна становятся активными."""
    b = browser()
    if action == "new":
        validate_url(url)
        page = await b.browser.contexts[0].new_page()
        b.register(page)
        await page.goto(url, wait_until="domcontentloaded")
    elif action in {"select", "close"}:
        if tab_id not in b.pages or b.pages[tab_id].is_closed():
            raise ValueError("Unknown or closed tab_id")
        if action == "close":
            await b.pages[tab_id].close()
            if not any(not p.is_closed() for p in b.pages.values()):
                b.register(await b.browser.contexts[0].new_page())
        else:
            b.page = b.pages[tab_id]
            await b.page.bring_to_front()
    return await b.snapshot()


@mcp.tool()
async def browser_back() -> dict:
    """Перейти назад в активной вкладке."""
    await browser().active().go_back(wait_until="domcontentloaded")
    return await browser().after_action()


@mcp.tool()
async def browser_dialog(accept: bool, prompt_text: str | None = None) -> dict:
    """Обработать нативный диалог alert/confirm/prompt из текущего снимка."""
    b = browser()
    dialog = b.dialogs.get(b.active())
    if dialog is None:
        raise ValueError("There is no pending dialog")
    if accept:
        await dialog.accept(outbound_text(prompt_text) if prompt_text is not None else None)
    else:
        await dialog.dismiss()
    b.dialogs.pop(b.active(), None)
    await b.settle_dialog_action()
    return await b.after_action()


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
