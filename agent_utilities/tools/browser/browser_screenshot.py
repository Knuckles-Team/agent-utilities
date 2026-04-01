import os
import tempfile
from typing import Any, Dict, Optional
from .browser_manager import get_browser_manager


async def take_screenshot(path: Optional[str] = None) -> Dict[str, Any]:
    """Capture a screenshot of the current browser page."""
    manager = get_browser_manager()
    page = await manager.get_current_page()

    if not path:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"screenshot_{os.getpid()}.png")

    await page.screenshot(path=path)
    return {"success": True, "path": path, "url": page.url}


async def take_element_screenshot(
    selector: str, path: Optional[str] = None
) -> Dict[str, Any]:
    """Capture a screenshot of a specific element."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    element = await page.query_selector(selector)

    if not path:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"element_{os.getpid()}.png")

    await element.screenshot(path=path)
    return {"success": True, "path": path, "selector": selector}
