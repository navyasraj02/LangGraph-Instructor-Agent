"""Minimal Playwright browser wrapper with accessibility tree support."""
import asyncio
import os
import re
from playwright.async_api import async_playwright
from logging_config import logger

# Actionable elements — included in minimal role:name output.
_CORE_ROLES = {
    "link", "button", "textbox", "combobox", "checkbox", "radio", "tab",
    "menuitem", "menuitemcheckbox", "menuitemradio", "searchbox", "switch",
    "option", "slider", "spinbutton", "treeitem",
    "dialog", "alertdialog",
}
# Page-orienting — included in minimal role:name output.
_ORIENT_ROLES = {"heading"}

_ROLE_LINE_RE = re.compile(r"^\s*-\s+([a-zA-Z]+)\b")


def _filter_snapshot_to_core_roles(snapshot: str) -> str:
    """Keep original aria_snapshot() lines, filtered to `_CORE_ROLES` only."""
    if not snapshot:
        return ""
    out: list[str] = []
    for line in snapshot.split("\n"):
        m = _ROLE_LINE_RE.match(line)
        if not m:
            continue
        role = m.group(1)
        if role in _CORE_ROLES:
            out.append(line)
    return "\n".join(out)


class BrowserController:
    def __init__(self, headless=False, storage_state_path: str | None = None):
        self.headless = headless
        self.storage_state_path = storage_state_path
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)

        # Restore cookies/localStorage from disk if available — enables session persistence
        # across runs so the login phase can be skipped when the session is still valid.
        context_kwargs = {
            "extra_http_headers": {"ngrok-skip-browser-warning": "true"},
        }
        if self.storage_state_path and os.path.exists(self.storage_state_path):
            context_kwargs["storage_state"] = self.storage_state_path
            logger.info(f"Browser started (restored session from {self.storage_state_path})")
        else:
            logger.info("Browser started (no prior session)")

        self.context = await self.browser.new_context(**context_kwargs)
        self.page = await self.context.new_page()
        self.page.set_default_timeout(15000)

    async def save_storage_state(self) -> bool:
        """Persist cookies + localStorage to `storage_state_path`. Returns True on success."""
        if not self.storage_state_path or not self.context:
            return False
        try:
            os.makedirs(os.path.dirname(self.storage_state_path) or ".", exist_ok=True)
            await self.context.storage_state(path=self.storage_state_path)
            logger.info(f"Saved session to {self.storage_state_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")
            return False

    def clear_storage_state(self) -> None:
        """Delete the persisted session file (e.g. after an auth failure)."""
        if self.storage_state_path and os.path.exists(self.storage_state_path):
            try:
                os.remove(self.storage_state_path)
                logger.info(f"Cleared stale session at {self.storage_state_path}")
            except Exception as e:
                logger.warning(f"Failed to clear session: {e}")

    async def close(self):
        """Close page → context → browser → Playwright so transports shut down before the asyncio loop ends (reduces Windows Proactor noise)."""
        if self.page:
            try:
                await self.page.close()
            except Exception:
                pass
            self.page = None
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
            self.context = None
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
            self.browser = None
        # Let pipe handles drain before stopping Playwright (avoids "Event loop is closed" in __del__).
        await asyncio.sleep(0.25)
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None

    async def navigate(self, url):
        await self.page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(1)

    async def wait_for_load(self):
        try:
            await self.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        await asyncio.sleep(1)

    async def get_current_url(self):
        return self.page.url

    async def get_page_text(self):
        """Get visible text from the full page body"""
        try:
            return await self.page.inner_text("body")
        except Exception:
            return ""

    async def get_accessibility_tree(self) -> str:
        """Return aria_snapshot() filtered to `_CORE_ROLES` lines only."""
        try:
            raw = await self.page.locator("body").aria_snapshot() or ""
            filtered = _filter_snapshot_to_core_roles(raw)
            if not filtered:
                return ""
            return filtered
        except Exception as e:
            logger.error(f"ARIA snapshot error: {e}")
            return ""
