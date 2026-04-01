from typing import Any, Dict
from .browser_manager import get_browser_manager


async def click_element(selector: str) -> Dict[str, Any]:
    """Click an element using a CSS or XPath selector."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.click(selector)
    return {"success": True, "message": f"Clicked element: {selector}"}


async def type_text(selector: str, text: str) -> Dict[str, Any]:
    """Type text into an input element."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.type(selector, text)
    return {"success": True, "message": f"Typed text into: {selector}"}


async def get_element_text(selector: str) -> Dict[str, Any]:
    """Retrieve the text content of an element."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    text = await page.inner_text(selector)
    return {"success": True, "text": text}


async def select_option(selector: str, value: str) -> Dict[str, Any]:
    """Select an option from a dropdown."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    await page.select_option(selector, value)
    return {"success": True, "message": f"Selected option: {value}"}
