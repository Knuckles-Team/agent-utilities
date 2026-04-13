#!/usr/bin/python
# coding: utf-8
"""Browser Tools Package.

This package provides a comprehensive suite of tools for browser automation
and web interaction, sub-divided into manager, control, interaction,
navigation, and screenshot capabilities.
"""

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

# Tool grouping for registration
browser_tools = [
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
