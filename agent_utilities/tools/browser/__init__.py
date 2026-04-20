#!/usr/bin/python
# coding: utf-8
"""Browser Tools Package.

This package provides a comprehensive suite of tools for browser automation
and web interaction, sub-divided into manager, control, interaction,
navigation, and screenshot capabilities.
"""

try:
    from .browser_manager import get_browser_manager
    from .browser_control import (
        initialize_browser,
        close_browser,
        browser_status,
        browser_new_page,
    )
    from .browser_interactions import (
        click_element,
        type_text,
        get_element_text,
        select_option,
    )
    from .browser_navigation import (
        navigate_to_url,
        browser_go_back,
        browser_go_forward,
        reload_page,
    )
    from .browser_screenshot import (
        take_screenshot,
        take_element_screenshot,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    # Placeholders to prevent collection errors
    get_browser_manager = None
    initialize_browser = None
    close_browser = None
    browser_status = None
    browser_new_page = None
    click_element = None
    type_text = None
    get_element_text = None
    select_option = None
    navigate_to_url = None
    browser_go_back = None
    browser_go_forward = None
    reload_page = None
    take_screenshot = None
    take_element_screenshot = None
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
