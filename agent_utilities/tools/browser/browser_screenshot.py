#!/usr/bin/python
from __future__ import annotations

"""Browser Screenshot Tools Module.

This module provides tools for capturing visual snapshots of the active
browser page or specific web elements, with support for temporary file
storage.
"""

import os
import secrets
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext

from ...models import AgentDeps
from ...security.persistence_privacy import persistence_reference
from .browser_manager import browser_fetch_enabled, get_browser_manager

_MAX_SCREENSHOT_BYTES = 256 * 1024 * 1024
_MAX_PATH_BYTES = 1024
_MAX_SELECTOR_BYTES = 4 * 1024


def _workspace_root(ctx: RunContext[AgentDeps]) -> Path:
    configured = getattr(ctx.deps, "workspace_path", None)
    if not configured:
        raise ValueError("browser workspace is unavailable")
    candidate = Path(str(configured))
    if candidate.is_symlink():
        raise ValueError("browser workspace is unavailable")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("browser workspace is unavailable")
    return root


def _destination(
    ctx: RunContext[AgentDeps], requested: str | None, *, prefix: str
) -> tuple[Path, str]:
    root = _workspace_root(ctx)
    if requested:
        if (
            not isinstance(requested, str)
            or len(requested.encode("utf-8")) > _MAX_PATH_BYTES
        ):
            raise ValueError("browser screenshot path is invalid")
        supplied = Path(requested)
        if supplied.is_absolute() or supplied.suffix.lower() != ".png":
            raise ValueError("browser screenshot path must be a relative PNG path")
        target = (root / supplied).resolve(strict=False)
        try:
            relative = target.relative_to(root)
        except ValueError:
            raise ValueError(
                "browser screenshot path is outside the workspace"
            ) from None
        parent = target.parent.resolve(strict=True)
        parent.relative_to(root)
    else:
        directory = root / ".agents" / "browser"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("browser screenshot directory is unsafe")
        if os.name == "posix":
            directory.chmod(0o700)
        target = directory / f"{prefix}_{secrets.token_hex(12)}.png"
        relative = target.relative_to(root)
    if target.exists() or target.is_symlink():
        raise FileExistsError("browser screenshot destination already exists")
    return target, relative.name


def _write_screenshot(
    ctx: RunContext[AgentDeps], requested: str | None, data: bytes, *, prefix: str
) -> str:
    if not isinstance(data, bytes) or not data or len(data) > _MAX_SCREENSHOT_BYTES:
        raise ValueError("browser screenshot exceeds the supported limit")
    target, relative = _destination(ctx, requested, prefix=prefix)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return f"browser-screenshot://{relative}"


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
    if not browser_fetch_enabled():
        return {
            "success": False,
            "error": "Browser-backed source access is disabled by policy.",
        }
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}

    try:
        image = await page.screenshot()
        asset_ref = _write_screenshot(ctx, path, image, prefix="page")
    except (FileExistsError, OSError, ValueError):
        return {
            "success": False,
            "error": "Browser screenshot destination was rejected.",
        }
    return {
        "success": True,
        "asset_ref": asset_ref,
        "page_ref": persistence_reference(
            "browser_page", page.url, namespace="screenshot"
        ),
        "bytes": len(image),
    }


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
    if (
        not browser_fetch_enabled()
        or not isinstance(selector, str)
        or not selector
        or len(selector.encode("utf-8")) > _MAX_SELECTOR_BYTES
    ):
        return {
            "success": False,
            "error": "Browser interaction was rejected by policy.",
        }
    manager = get_browser_manager()
    page = await manager.get_current_page()
    if not page:
        return {"success": False, "error": "No active page found."}
    element = await page.query_selector(selector)
    if not element:
        return {"success": False, "error": "Element not found."}

    try:
        image = await element.screenshot()
        asset_ref = _write_screenshot(ctx, path, image, prefix="element")
    except (FileExistsError, OSError, ValueError):
        return {
            "success": False,
            "error": "Browser screenshot destination was rejected.",
        }
    return {"success": True, "asset_ref": asset_ref, "bytes": len(image)}
