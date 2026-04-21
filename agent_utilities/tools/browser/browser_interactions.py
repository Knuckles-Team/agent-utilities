#!/usr/bin/python
from __future__ import annotations

"""Browser Interactions Tools Module.

This module provides tools for direct interaction with web elements,
including clicking, typing, text extraction, and dropdown selection.
"""

from typing import Any

from pydantic_ai import RunContext

from ...models import AgentDeps
from .browser_manager import get_browser_manager


async def click_element(ctx: RunContext[AgentDeps], selector: str) -> dict[str, Any]:
    """Execute a mouse click on an element identified by a CSS or XPath selector.

    Args:
        ctx: The agent run context.
        selector: The selector string for the target element.

    Returns:
        A dictionary indicating the success of the operation.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.click(selector)
    return {"success": True, "message": f"Clicked element: {selector}"}


async def type_text(
    ctx: RunContext[AgentDeps], selector: str, text: str
) -> dict[str, Any]:
    """Input text into a form field or editable element.

    Args:
        ctx: The agent run context.
        selector: The selector string for the target input.
        text: The string to be typed.

    Returns:
        A dictionary indicating the success of the operation.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.type(selector, text)
    return {"success": True, "message": f"Typed text into: {selector}"}


async def get_element_text(ctx: RunContext[AgentDeps], selector: str) -> dict[str, Any]:
    """Extract the inner text content of a specified web element.

    Args:
        ctx: The agent run context.
        selector: The selector string for the target element.

    Returns:
        A dictionary containing the extracted text.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    text = await page.inner_text(selector)
    return {"success": True, "text": text}


async def select_option(
    ctx: RunContext[AgentDeps], selector: str, value: str
) -> dict[str, Any]:
    """Select a specific option from a dropdown (select) element.

    Args:
        ctx: The agent run context.
        selector: The selector string for the target dropdown.
        value: The value string to be selected.

    Returns:
        A dictionary indicating the success of the operation.

    """
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.select_option(selector, value)
    return {"success": True, "message": f"Selected option: {value}"}
