#!/usr/bin/python
"""Browser Management Module.

CONCEPT:AU-ECO.messaging.native-backend-abstraction

This module implements a singleton BrowserManager that handles the
asynchronous lifecycle of Playwright-based browser instances, including
context initialization and page tracking.
"""

import logging
from urllib.parse import urlsplit

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger(__name__)

_NAVIGATION_TIMEOUT_MS = 30_000


def browser_fetch_enabled() -> bool:
    """Return whether the larger browser egress surface is explicitly enabled."""
    from agent_utilities.core.config import config

    return bool(config.source_http_allow_browser_fetch)


def require_safe_browser_url(url: str) -> None:
    """Apply the shared source-egress policy before Playwright sees a URL.

    Playwright is deliberately not treated as a generic local browser: ``file:``
    and other non-HTTP schemes are rejected, private destinations require an
    exact operator allowlist entry, and DNS is checked for every request hop.
    """
    if not browser_fetch_enabled():
        raise PermissionError("browser-backed source access is disabled by policy")
    if not isinstance(url, str) or len(url.encode("utf-8")) > 8_192:
        raise ValueError("browser URL is invalid")
    from agent_utilities.core.config import config
    from agent_utilities.protocols.source_connectors.http_safety import (
        require_safe_source_url,
    )

    require_safe_source_url(
        url,
        allowed_private_hosts=config.source_http_allowed_private_hosts,
        resolve_dns=True,
    )


class BrowserManager:
    """Core orchestrator for Playwright browser life-cycle and state.

    This class provides a high-level API for launching browser engines,
    managing isolated contexts, and tracking active pages within a
    session.
    """

    def __init__(self) -> None:
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.pages: list[Page] = []
        self.headless: bool = True
        self.browser_type: str = "chromium"
        self.homepage: str = "https://www.google.com"
        self._initialized: bool = False

    async def async_initialize(self) -> None:
        """Asynchronously launch the browser engine and create a default context.

        Initializes Playwright, launches the specified browser type
        (Chromium, Firefox, or Webkit), and sets up the initial homepage.
        """
        if self._initialized:
            return

        require_safe_browser_url(self.homepage)
        if self.browser_type not in {"chromium", "firefox", "webkit"}:
            raise ValueError("Unsupported browser type")

        try:
            self.playwright = await async_playwright().start()

            if self.browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(
                    headless=self.headless
                )
            elif self.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(
                    headless=self.headless
                )
            else:
                self.browser = await self.playwright.webkit.launch(
                    headless=self.headless
                )

            if self.browser is None:
                raise RuntimeError("Failed to launch browser")
            self.context = await self.browser.new_context(
                accept_downloads=False,
                service_workers="block",
            )

            async def _guard_request(route, request) -> None:
                # Route interception covers redirects, frames, scripts, images and
                # script-initiated fetches.  ``data:``/``blob:`` are local browser
                # objects and carry no network destination; every other non-HTTP
                # scheme (especially ``file:``) is denied.
                scheme = urlsplit(request.url).scheme.lower()
                if scheme in {"data", "blob"}:
                    await route.continue_()
                    return
                try:
                    require_safe_browser_url(request.url)
                except (PermissionError, ValueError):
                    await route.abort("blockedbyclient")
                    return
                await route.continue_()

            await self.context.route("**/*", _guard_request)
            page: Page = await self.context.new_page()
            await page.goto(
                self.homepage,
                timeout=_NAVIGATION_TIMEOUT_MS,
                wait_until="domcontentloaded",
            )
            self.pages.append(page)
            self._initialized = True
        except Exception:
            await self.close()
            raise

    async def get_current_page(self) -> Page | None:
        """Retrieve the last active page in the current context.

        Returns:
            The most recently opened Playwright Page object, if any.

        """
        return self.pages[-1] if self.pages else None

    async def new_page(self, url: str | None = None) -> Page:
        """Open a new tab/page within the active browser context.

        Args:
            url: Optional URL to navigate to immediately.

        Returns:
            The newly created Playwright Page object.

        """
        if not self._initialized:
            await self.async_initialize()
        if self.context is None:
            raise RuntimeError("Browser context not initialized")
        if url:
            require_safe_browser_url(url)
        page = await self.context.new_page()
        if url:
            await self.navigate(page, url)
        self.pages.append(page)
        return page

    async def navigate(self, page: Page, url: str) -> None:
        """Navigate one page through the browser egress policy."""
        require_safe_browser_url(url)
        await page.goto(
            url,
            timeout=_NAVIGATION_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )

    async def close(self) -> None:
        """Shutdown the browser engine and release all system resources.

        Closes all contexts and pages, and stops the Playwright driver.
        """
        try:
            if self.browser:
                await self.browser.close()
        finally:
            if self.playwright:
                await self.playwright.stop()
            self._initialized = False
            self.pages = []
            self.context = None
            self.browser = None
            self.playwright = None


_BROWSER_MANAGER: BrowserManager | None = None


def get_browser_manager() -> BrowserManager:
    """Retrieve the singleton instance of the BrowserManager.

    Returns:
        The global BrowserManager instance.

    """
    global _BROWSER_MANAGER
    if _BROWSER_MANAGER is None:
        _BROWSER_MANAGER = BrowserManager()
    return _BROWSER_MANAGER
