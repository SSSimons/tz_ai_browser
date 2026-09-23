import asyncio
import inspect
import json
import os
import shutil
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

from playwright.async_api import Browser, Dialog, Locator, Page, async_playwright
from pydantic import BaseModel, Field

from browser_agent.config import Settings
from browser_agent.dom import CLICK_STATE, FIND_MODAL
from browser_agent.forms import Forms


class Target(BaseModel):
    """Локатор из текущей страницы без заранее заданных правил для сайтов."""

    kind: Literal["role", "text", "label", "placeholder", "css"]
    value: str = Field(min_length=1)
    name: str | None = None
    exact: bool = True
    index: int | None = Field(default=None, ge=0)
    frame: int = Field(default=0, ge=0, description="Индекс iframe из browser_snapshot")


def find_browser() -> str | None:
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chrome"),
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    ]
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if url != "about:blank" and (parsed.scheme not in {"https", "http"} or not parsed.hostname):
        raise ValueError("Use an absolute http(s) URL or about:blank.")
    if parsed.username or parsed.password:
        raise ValueError("Do not put credentials in navigation URLs.")
    return url


class BrowserRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = None
        self.playwright = None
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.pages: dict[int, Page] = {}
        self.dialogs: dict[Page, Dialog] = {}
        self.dialog_opened = asyncio.Event()
        self.pending_actions: set[asyncio.Task] = set()
        self.downloads: list[dict] = []
        self.artifacts = settings.run_dir or settings.artifacts_dir / "browser"
        self.frame_number = 0
        self.endpoint = settings.cdp_url
        self.forms = Forms(self)
        self.last_click = None

    async def start(self):
        if self.browser:
            return
        self.artifacts.mkdir(parents=True, exist_ok=True)
        try:
            if not self.endpoint:
                from seleniumbase import cdp_driver

                executable = self.settings.browser_executable or find_browser()
                if not executable:
                    raise RuntimeError(
                        "Chrome/Edge not found. Install Chrome or set AGENT_BROWSER_EXECUTABLE."
                    )
                self.settings.profile_dir.mkdir(parents=True, exist_ok=True)
                self.driver = await cdp_driver.start_async(
                    browser_executable_path=executable,
                    user_data_dir=str(self.settings.profile_dir.resolve()),
                    headless=self.settings.headless,
                    host="127.0.0.1",
                )
                self.endpoint = self.driver.get_endpoint_url()
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.connect_over_cdp(self.endpoint)
            if not self.browser.contexts:
                raise RuntimeError("CDP browser has no context.")
            for context in self.browser.contexts:
                context.set_default_timeout(10000)
                context.set_default_navigation_timeout(30000)
                context.on("page", self.register)
                for page in context.pages:
                    self.register(page)
            if not self.page:
                self.register(await self.browser.contexts[0].new_page())
        except BaseException:
            await self.close()
            raise

    def register(self, page: Page):
        if page in self.pages.values():
            return
        self.pages[max(self.pages, default=-1) + 1] = page
        self.page = page  # Новое окно становится активной вкладкой.

        def on_dialog(dialog):
            self.dialogs[page] = dialog
            self.dialog_opened.set()

        page.on("dialog", on_dialog)
        page.on(
            "download",
            lambda download: self.downloads.append(
                {
                    "url": download.url,
                    "filename": download.suggested_filename,
                    "note": "Download observed; file is not automatically exported.",
                }
            ),
        )

    def active(self) -> Page:
        if self.page is None or self.page.is_closed():
            self.page = next(
                (p for p in reversed(list(self.pages.values())) if not p.is_closed()), None
            )
        if self.page is None:
            raise RuntimeError("No open tabs. Use browser_tab(action='new').")
        return self.page

    async def active_modal(self, frame):
        """Выбрать верхнее окно по видимости и попаданию указателя, без обхода ошибок."""
        token = await frame.evaluate(FIND_MODAL, uuid4().hex)
        return frame.locator(f'[data-browser-agent-scope="{token}"]') if token else None

    async def action_scope(self, frame):
        """Проверить также диалоги в родительских документах iframe."""
        child = frame
        while child.parent_frame is not None:
            parent = child.parent_frame
            modal = await self.active_modal(parent)
            if modal is not None:
                element = await child.frame_element()
                try:
                    if not await modal.evaluate("(m, e) => m.contains(e)", element):
                        raise ValueError("Этот iframe закрыт диалогом. Осмотрите активное окно.")
                finally:
                    await element.dispose()
            child = parent
        return await self.active_modal(frame)

    async def ensure_active_element(self, frame, element):
        modal = await self.action_scope(frame)
        if modal is not None and not await modal.evaluate("(m, e) => m.contains(e)", element):
            raise ValueError("Элемент находится на затемнённом фоне. Осмотрите активное окно.")
        return modal

    async def locate(self, target: Target, *, wait: bool = False) -> Locator:
        page = self.active()
        if target.frame >= len(page.frames):
            raise ValueError("Frame no longer exists. Take a new snapshot.")
        scope = page.frames[target.frame]
        modal = await self.action_scope(scope)
        if modal is not None:
            # Пока окно открыто, фон вообще не входит в область поиска.
            scope = modal
        if target.kind == "role":
            kwargs = {"exact": target.exact}
            if target.name is not None:
                kwargs["name"] = target.name
            locator = scope.get_by_role(target.value, **kwargs)
        elif target.kind == "text":
            locator = scope.get_by_text(target.value, exact=target.exact)
        elif target.kind == "label":
            locator = scope.get_by_label(target.value, exact=target.exact)
        elif target.kind == "placeholder":
            locator = scope.get_by_placeholder(target.value, exact=target.exact)
        else:
            locator = scope.locator(target.value)
        locator = locator.filter(visible=True)
        locator = locator.nth(target.index) if target.index is not None else locator
        if not wait:
            count = await locator.count()
            if count != 1:
                raise ValueError(
                    f"В активной области найдено элементов: {count}. "
                    "Осмотрите её заново и уточните локатор; действие не выполнялось."
                )
        return locator

    async def click(self, target: Target, double: bool = False):
        locator = await self.locate(target)
        # Закрепляем конкретный DOM-элемент: перерисовка не должна перенаправить клик.
        element = await locator.element_handle(timeout=1500)
        if element is None:
            raise ValueError("Элемент исчез. Осмотрите страницу заново.")
        await self.click_element(element, double=double)

    async def click_element(self, element, *, double=False, position=None):
        """Общая проверка обычного и координатного клика по конкретному элементу."""
        keep = False
        try:
            frame = await element.owner_frame()
            await self.ensure_active_element(frame, element)
            digest = sha256((await element.evaluate(CLICK_STATE)).encode()).hexdigest()
            if self.last_click:
                previous, previous_digest, previous_frame = self.last_click
                same = False
                # После навигации старый JS-контекст уже может быть уничтожен.
                if previous_frame == frame:
                    with suppress(Exception):
                        same = await element.evaluate("(e, previous) => e === previous", previous)
                if same and digest == previous_digest:
                    raise ValueError(
                        "Повторный клик заблокирован: эта кнопка уже нажата, а состояние "
                        "не изменилось. Дождитесь результата или осмотрите форму; "
                        "не повторяйте отправку."
                    )
            # Проверка попадания и анимации не посылает событие click.
            await element.click(trial=True, position=position, timeout=2000)
            await self.ensure_active_element(frame, element)
            if self.last_click:
                await self.last_click[0].dispose()
            self.last_click = (element, digest, frame)
            keep = True
            operation = (
                element.dblclick(position=position, timeout=3000)
                if double
                else element.click(position=position, timeout=3000)
            )
            await self.perform(operation)
        finally:
            if not keep:
                await element.dispose()

    async def point_element(self, x, y):
        """Разрешить координаты скриншота в реальный элемент, включая iframe."""
        frame = self.active().main_frame
        local_x, local_y = x, y
        while True:
            handle = await frame.evaluate_handle(
                """([x, y]) => {
                    const e = document.elementFromPoint(x, y);
                    return e?.closest('button, a, input, select, textarea, [role=button]') || e;
                }""",
                [local_x, local_y],
            )
            element = handle.as_element()
            if element is None:
                await handle.dispose()
                raise ValueError("Координаты вне страницы. Получите свежий скриншот.")
            try:
                await self.ensure_active_element(frame, element)
                child = await element.content_frame()
                if child is None:
                    return element
                box = await element.bounding_box()
                border = await element.evaluate("e => [e.clientLeft, e.clientTop]")
                local_x, local_y = x - box["x"] - border[0], y - box["y"] - border[1]
                frame = child
            except BaseException:
                await element.dispose()
                raise
            await element.dispose()

    async def pointer_click(self, x, y):
        element = await self.point_element(x, y)
        box = await element.bounding_box()
        if box is None:
            await element.dispose()
            raise ValueError("Элемент исчез. Получите свежий скриншот.")
        await self.click_element(element, position={"x": x - box["x"], "y": y - box["y"]})

    async def perform(self, operation):
        """Вернуть управление агенту, если действие заблокировано нативным диалогом."""
        self.dialog_opened.clear()
        action = asyncio.create_task(operation)
        opened = asyncio.create_task(self.dialog_opened.wait())
        try:
            done, _ = await asyncio.wait({action, opened}, return_when=asyncio.FIRST_COMPLETED)
            if action in done:
                return action.result()
            self.pending_actions.add(action)
        except BaseException:
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
            raise
        finally:
            opened.cancel()
            await asyncio.gather(opened, return_exceptions=True)

    async def settle_dialog_action(self):
        tasks = list(self.pending_actions)
        self.pending_actions.clear()
        if tasks:
            # Сохраняем ошибку действия, даже если диалог уже закрыт.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    raise result

    async def snapshot(
        self, offset: int = 0, limit: int = 16000, include_metadata: bool = False
    ) -> dict:
        page = self.active()
        tabs = [
            {"id": i, "url": p.url, "active": p == page}
            for i, p in self.pages.items()
            if not p.is_closed()
        ]
        dialog = self.dialogs.get(page)
        if dialog:
            return {"tabs": tabs, "dialog": {"type": dialog.type, "message": dialog.message}}
        semaphore = asyncio.Semaphore(4)

        async def read_frame(i, frame):
            async with semaphore:
                return await self._frame_snapshot(i, frame, include_metadata)

        parts = await asyncio.gather(*(read_frame(i, f) for i, f in enumerate(page.frames)))
        document = "\n\n".join(parts)
        end = min(offset + limit, len(document))
        return {
            "url": page.url,
            "title": await page.title(),
            "tabs": tabs,
            "content": document[offset:end],
            "total_chars": len(document),
            "next_offset": end if end < len(document) else None,
            "downloads": self.downloads[-5:],
            "untrusted_page_content": True,
        }

    async def _frame_snapshot(self, i, frame, include_metadata):
        try:
            modal = await self.action_scope(frame)
            scope = modal if modal is not None else frame.locator("body")
            tree = await scope.aria_snapshot(timeout=3000)
            modal_note = "\nACTIVE MODAL (фон исключён):" if modal is not None else ""
            if not include_metadata:
                return f"FRAME {i} {frame.url}{modal_note}\n{tree}"
            # Общие метаданные DOM, без правил и маршрутов конкретных сайтов.
            metadata = await scope.evaluate("""root => [...root.querySelectorAll(
                    'a[href],input,textarea,select,button,[role],[contenteditable=true]'
                )].filter(e => e.getClientRects().length).slice(0,250).map(e => ({
                    tag:e.tagName.toLowerCase(), id:e.id || undefined,
                    name:e.getAttribute('name') || undefined,
                    type:e.getAttribute('type') || undefined,
                    role:e.getAttribute('role') || undefined,
                    label:e.getAttribute('aria-label') || undefined,
                    placeholder:e.getAttribute('placeholder') || undefined,
                    href:e.tagName==='A' ? e.href : undefined,
                    text:(e.innerText || '').slice(0,160)
                }))""")
            return f"FRAME {i} {frame.url}{modal_note}\n{tree}\nDOM metadata:\n" + json.dumps(
                metadata, ensure_ascii=False
            )
        except ValueError as exc:
            return f"FRAME {i} недоступен: {exc}"
        except Exception as exc:
            return f"FRAME {i} unavailable: {type(exc).__name__}"

    async def capture(self) -> bytes:
        return await self.active().screenshot(type="png", timeout=10000)

    async def after_action(self) -> dict:
        if self.dialogs.get(self.active()):
            return await self.snapshot()
        # Кадры нужны для видео, но не должны задерживать обычную работу.
        if self.settings.record_actions:
            with suppress(Exception):
                data = await self.capture()
                self.frame_number += 1
                (self.artifacts / f"frame-{self.frame_number:05d}.png").write_bytes(data)
        try:
            return await self.snapshot()
        except Exception as exc:
            return {
                "action_executed": True,
                "observation_error": type(exc).__name__,
                "next": "Take a fresh browser_snapshot; do not blindly repeat the action.",
            }

    async def close(self):
        for task in self.pending_actions:
            task.cancel()
        await asyncio.gather(*self.pending_actions, return_exceptions=True)
        self.pending_actions.clear()
        # Останавливаем связь Playwright; внешний браузер не закрываем.
        await self.forms.clear()
        if self.last_click:
            with suppress(Exception):
                await self.last_click[0].dispose()
            self.last_click = None
        if self.playwright:
            with suppress(Exception):
                await self.playwright.stop()
        if self.driver:
            with suppress(Exception):
                result = self.driver.stop()
                if inspect.isawaitable(result):
                    await result
        self.browser = None
        self.playwright = None
        self.driver = None
