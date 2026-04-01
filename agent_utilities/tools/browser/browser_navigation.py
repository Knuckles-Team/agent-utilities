from typing import Any, Dict
from .browser_manager import get_browser_manager


async def navigate_to_url(url: str) -> Dict[str, Any]:
    """Navigate current browser page to a new URL."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.goto(url)
    return {"success": True, "url": page.url, "title": await page.title()}


async def browser_go_back() -> Dict[str, Any]:
    """Go back to the previous page in history."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.go_back()
    return {"success": True, "url": page.url}


async def browser_go_forward() -> Dict[str, Any]:
    """Go forward in navigation history."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.go_forward()
    return {"success": True, "url": page.url}


async def reload_page() -> Dict[str, Any]:
    """Reload the current browser page."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.reload()
    return {"success": True, "url": page.url}
