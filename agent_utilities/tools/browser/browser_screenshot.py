#!/usr/bin/python
from __future__ import annotations

"""Browser Screenshot Tools Module.

This module provides tools for capturing visual snapshots of the active
browser page or specific web elements, with support for temporary file
storage.
"""

import os
import tempfile
from typing import Any

from pydantic_ai import RunContext

from ...models import AgentDeps
from .browser_manager import get_browser_manager


async def take_screenshot(
    ctx: RunContext[AgentDeps], path: str | None = None
) -> dict[str, Any]:
    """Capture a visual snapshot of the currently active browser page.

    Args:
        ctx: The agent run context.
        path: Optional destination path for the screenshot image. If not
              provided, a temporary file will be created.

    Returns:
        A dictionary containing the saved path and the source URL.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}

    if not path:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"screenshot_{os.getpid()}.png")

    await page.screenshot(path=path)
    return {"success": True, "path": path, "url": page.url}


async def take_element_screenshot(
    ctx: RunContext[AgentDeps], selector: str, path: str | None = None
) -> dict[str, Any]:
    """Capture a visual snapshot of a specific web element.

    Args:
        ctx: The agent run context.
        selector: The CSS or XPath selector for the target element.
        path: Optional destination path for the screenshot image. If not
              provided, a temporary file will be created.

    Returns:
        A dictionary containing the saved path and the selector used.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    element = await page.query_selector(selector)
    if not element:
        return {"success": False, "error": f"Element not found: {selector}"}

    if not path:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"element_{os.getpid()}.png")

    await element.screenshot(path=path)
    return {"success": True, "path": path, "selector": selector}
