#!/usr/bin/python
from __future__ import annotations

"""Browser Navigation Tools Module.

This module provides tools for controlling the primary navigation flow
of the active browser instance, including direct URL navigation,
history traversals, and page refreshes.
"""

from typing import Any

from pydantic_ai import RunContext

from ...models import AgentDeps
from ...security.persistence_privacy import persistence_reference
from .browser_manager import browser_fetch_enabled, get_browser_manager


def _page_reference(url: str) -> str:
    return persistence_reference("browser_page", url, namespace="navigation")


def _disabled() -> dict[str, Any]:
    return {
        "success": False,
        "error": "Browser-backed source access is disabled by policy.",
    }


async def navigate_to_url(ctx: RunContext[AgentDeps], url: str) -> dict[str, Any]:
    """Instruct the browser to navigate to a target URL.

    Args:
        ctx: The agent run context.
        url: The destination web address.

    Returns:
        A dictionary containing the actual URL reached and the page title.

    """
    if not browser_fetch_enabled():
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    try:
        await manager.navigate(page, url)
    except (PermissionError, ValueError):
        return {
            "success": False,
            "error": "Browser destination was rejected by policy.",
        }
    title = str(await page.title())
    return {
        "success": True,
        "page_ref": _page_reference(page.url),
        "title_ref": persistence_reference(
            "browser_title", title, namespace="navigation"
        ),
    }


async def browser_go_back(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    """Navigate backwards through the current session history.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary containing the resulting URL.

    """
    if not browser_fetch_enabled():
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.go_back()
    return {"success": True, "page_ref": _page_reference(page.url)}


async def browser_go_forward(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    """Navigate forwards through the current session history.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary containing the resulting URL.

    """
    if not browser_fetch_enabled():
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.go_forward()
    return {"success": True, "page_ref": _page_reference(page.url)}


async def reload_page(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    """Refresh the content of the current browser page.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary containing the resulting URL.

    """
    if not browser_fetch_enabled():
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.reload()
    return {"success": True, "page_ref": _page_reference(page.url)}
