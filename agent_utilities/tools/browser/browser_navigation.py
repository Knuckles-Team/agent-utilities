#!/usr/bin/python
# coding: utf-8
"""Browser Navigation Tools Module.

This module provides tools for controlling the primary navigation flow
of the active browser instance, including direct URL navigation,
history traversals, and page refreshes.
"""

from typing import Any, Dict
from .browser_manager import get_browser_manager
from pydantic_ai import RunContext
from ...models import AgentDeps


async def navigate_to_url(ctx: RunContext[AgentDeps], url: str) -> Dict[str, Any]:
    """Instruct the browser to navigate to a target URL.

    Args:
        ctx: The agent run context.
        url: The destination web address.

    Returns:
        A dictionary containing the actual URL reached and the page title.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.goto(url)
    return {"success": True, "url": page.url, "title": await page.title()}


async def browser_go_back(ctx: RunContext[AgentDeps]) -> Dict[str, Any]:
    """Navigate backwards through the current session history.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary containing the resulting URL.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.go_back()
    return {"success": True, "url": page.url}


async def browser_go_forward(ctx: RunContext[AgentDeps]) -> Dict[str, Any]:
    """Navigate forwards through the current session history.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary containing the resulting URL.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.go_forward()
    return {"success": True, "url": page.url}


async def reload_page(ctx: RunContext[AgentDeps]) -> Dict[str, Any]:
    """Refresh the content of the current browser page.

    Args:
        ctx: The agent run context.

    Returns:
        A dictionary containing the resulting URL.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.reload()
    return {"success": True, "url": page.url}
