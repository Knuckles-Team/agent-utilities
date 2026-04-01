import logging
from typing import List, Optional
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages the lifecycle of a Playwright browser instance."""

    def __init__(self):
        self.playwright: Optional[async_playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.pages: List[Page] = []
        self.headless: bool = True
        self.browser_type: str = "chromium"
        self.homepage: str = "https://www.google.com"
        self._initialized: bool = False

    async def async_initialize(self):
        if self._initialized:
            return

        self.playwright = await async_playwright().start()

        if self.browser_type == "chromium":
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
        elif self.browser_type == "firefox":
            self.browser = await self.playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == "webkit":
            self.browser = await self.playwright.webkit.launch(headless=self.headless)
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")

        self.context = await self.browser.new_context()
        page = await self.context.new_page()
        await page.goto(self.homepage)
        self.pages.append(page)
        self._initialized = True

    async def get_current_page(self) -> Optional[Page]:
        return self.pages[-1] if self.pages else None

    async def new_page(self, url: Optional[str] = None) -> Page:
        if not self._initialized:
            await self.async_initialize()
        page = await self.context.new_page()
        if url:
            await page.goto(url)
        self.pages.append(page)
        return page

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._initialized = False
        self.pages = []


_BROWSER_MANAGER: Optional[BrowserManager] = None


def get_browser_manager() -> BrowserManager:
    global _BROWSER_MANAGER
    if _BROWSER_MANAGER is None:
        _BROWSER_MANAGER = BrowserManager()
    return _BROWSER_MANAGER
