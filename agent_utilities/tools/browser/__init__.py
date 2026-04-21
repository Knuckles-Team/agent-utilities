#!/usr/bin/python
"""Browser Tools Package.

This package provides a comprehensive suite of tools for browser automation
and web interaction, sub-divided into manager, control, interaction,
navigation, and screenshot capabilities.
"""

from typing import Any

try:
    from .browser_control import (
        browser_new_page,
        browser_status,
        close_browser,
        initialize_browser,
    )
    from .browser_interactions import (
        click_element,
        get_element_text,
        select_option,
        type_text,
    )
    from .browser_manager import get_browser_manager
    from .browser_navigation import (
        browser_go_back,
        browser_go_forward,
        navigate_to_url,
        reload_page,
    )
    from .browser_screenshot import (
        take_element_screenshot,
        take_screenshot,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    # Placeholders to prevent collection errors
    get_browser_manager: Any = None  # type: ignore
    initialize_browser: Any = None  # type: ignore
    close_browser: Any = None  # type: ignore
    browser_status: Any = None  # type: ignore
    browser_new_page: Any = None  # type: ignore
    click_element: Any = None  # type: ignore
    type_text: Any = None  # type: ignore
    get_element_text: Any = None  # type: ignore
    select_option: Any = None  # type: ignore
    navigate_to_url: Any = None  # type: ignore
    browser_go_back: Any = None  # type: ignore
    browser_go_forward: Any = None  # type: ignore
    reload_page: Any = None  # type: ignore
    take_screenshot: Any = None  # type: ignore
    take_element_screenshot: Any = None  # type: ignore
    PLAYWRIGHT_AVAILABLE = False

__all__ = [
    "get_browser_manager",
    "initialize_browser",
    "close_browser",
    "browser_status",
    "browser_new_page",
    "click_element",
    "type_text",
    "get_element_text",
    "select_option",
    "navigate_to_url",
    "browser_go_back",
    "browser_go_forward",
    "reload_page",
    "take_screenshot",
    "take_element_screenshot",
]

# Tool grouping for registration (filtered if missing)
browser_tools = [
    t
    for t in [
        initialize_browser,
        close_browser,
        browser_status,
        browser_new_page,
        click_element,
        type_text,
        get_element_text,
        select_option,
        navigate_to_url,
        browser_go_back,
        browser_go_forward,
        reload_page,
        take_screenshot,
        take_element_screenshot,
    ]
    if t is not None
]
