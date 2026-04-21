#!/usr/bin/python
from __future__ import annotations

"""Browser Control Tools Module.

This module provides tools for lifecycle management of the browser
instance, including initialization, status monitoring, and resource cleanup.
"""

from typing import Any

from pydantic_ai import RunContext

from ...models import AgentDeps
from .browser_manager import get_browser_manager


async def initialize_browser(
    ctx: RunContext[AgentDeps],
    headless: bool = True,
    browser_type: str = "chromium",
    homepage: str = "https://www.google.com",
) -> dict[str, Any]:
    """Initialize the global browser instance with custom configuration.

    Args:
        ctx: The agent run context.
        headless: Whether to run in headless mode (no visible UI).
        browser_type: The engine to use ('chromium', 'firefox', or 'webkit').
        homepage: The initial URL to navigate to upon startup.

    Returns:
        A dictionary containing the initialization status and browser metadata.

    """
    manager = get_browser_manager()
    manager.headless = headless
    manager.browser_type = browser_type
    manager.homepage = homepage
    await manager.async_initialize()
    page = await manager.get_current_page()
    return {
        "success": True,
        "browser_type": browser_type,
        "headless": headless,
        "homepage": homepage,
        "current_url": page.url if page else "Unknown",
    }


async def close_browser(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    """Close the active browser instance and release all associated resources.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary indicating the success of the operation.

    """
    manager = get_browser_manager()
    await manager.close()
    return {"success": True, "message": "Browser closed"}


async def browser_status(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    """Retrieve the current initialization state and active URL of the browser.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary containing status flags and the current URL.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    return {
        "success": True,
        "status": "initialized" if manager._initialized else "not_initialized",
        "current_url": page.url if page else None,
    }


async def browser_new_page(
    ctx: RunContext[AgentDeps], url: str | None = None
) -> dict[str, Any]:
    """Open a new tab or page in the active browser instance.

    Args:
        ctx: The agent run context.
        url: Optional URL to navigate to immediately after opening.

    Returns:
        A dictionary containing the URL and page title of the new tab.

    """
    manager = get_browser_manager()
    page = await manager.new_page(url)
    return {"success": True, "url": page.url, "title": await page.title()}
