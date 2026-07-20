#!/usr/bin/python
from __future__ import annotations

"""Browser Interactions Tools Module.

This module provides tools for direct interaction with web elements,
including clicking, typing, text extraction, and dropdown selection.
"""

from typing import Any

from pydantic_ai import RunContext

from ...models import AgentDeps
from ...security.persistence_privacy import PersistencePrivacyGuard
from .browser_manager import browser_fetch_enabled, get_browser_manager

_MAX_SELECTOR_BYTES = 4 * 1024
_MAX_INPUT_BYTES = 1024 * 1024
_MAX_TEXT_RESULT_BYTES = 64 * 1024


def _enabled_input(value: str, *, max_bytes: int) -> bool:
    return (
        browser_fetch_enabled()
        and isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= max_bytes
    )


def _disabled() -> dict[str, Any]:
    return {"success": False, "error": "Browser interaction was rejected by policy."}


async def click_element(ctx: RunContext[AgentDeps], selector: str) -> dict[str, Any]:
    """Execute a mouse click on an element identified by a CSS or XPath selector.

    Args:
        ctx: The agent run context.
        selector: The selector string for the target element.

    Returns:
        A dictionary indicating the success of the operation.

    """
    if not _enabled_input(selector, max_bytes=_MAX_SELECTOR_BYTES):
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.click(selector)
    return {"success": True, "message": "Element clicked."}


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
    if not _enabled_input(
        selector, max_bytes=_MAX_SELECTOR_BYTES
    ) or not _enabled_input(text, max_bytes=_MAX_INPUT_BYTES):
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.type(selector, text)
    return {"success": True, "message": "Text entered."}


async def get_element_text(ctx: RunContext[AgentDeps], selector: str) -> dict[str, Any]:
    """Extract the inner text content of a specified web element.

    Args:
        ctx: The agent run context.
        selector: The selector string for the target element.

    Returns:
        A dictionary containing the extracted text.

    """
    if not _enabled_input(selector, max_bytes=_MAX_SELECTOR_BYTES):
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    text = str(await page.inner_text(selector))
    encoded = text.encode("utf-8")
    truncated = len(encoded) > _MAX_TEXT_RESULT_BYTES
    if truncated:
        text = encoded[:_MAX_TEXT_RESULT_BYTES].decode("utf-8", errors="ignore")
    clean, report = PersistencePrivacyGuard().sanitize_text(text)
    return {
        "success": True,
        "text": clean,
        "truncated": truncated,
        "privacy_redactions": report.redactions,
    }


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
    if not _enabled_input(
        selector, max_bytes=_MAX_SELECTOR_BYTES
    ) or not _enabled_input(value, max_bytes=_MAX_INPUT_BYTES):
        return _disabled()
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    await page.select_option(selector, value)
    return {"success": True, "message": "Option selected."}
