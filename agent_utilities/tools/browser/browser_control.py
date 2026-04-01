from typing import Any, Dict, Optional
from .browser_manager import get_browser_manager


from pydantic_ai import RunContext
from ...models import AgentDeps


async def initialize_browser(
    ctx: RunContext[AgentDeps],
    headless: bool = True,
    browser_type: str = "chromium",
    homepage: str = "https://www.google.com",
) -> Dict[str, Any]:
    """Initialize the browser with specified settings."""
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


async def close_browser() -> Dict[str, Any]:
    """Close the browser and clean up resources."""
    manager = get_browser_manager()
    await manager.close()
    return {"success": True, "message": "Browser closed"}


async def browser_status() -> Dict[str, Any]:
    """Get current browser status and information."""
    manager = get_browser_manager()
    page = await manager.get_current_page()
    return {
        "success": True,
        "status": "initialized" if manager._initialized else "not_initialized",
        "current_url": page.url if page else None,
    }


async def browser_new_page(url: Optional[str] = None) -> Dict[str, Any]:
    """Create a new browser page/tab."""
    manager = get_browser_manager()
    page = await manager.new_page(url)
    return {"success": True, "url": page.url, "title": await page.title()}
