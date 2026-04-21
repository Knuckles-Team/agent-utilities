#!/usr/bin/python
"""Modular Tool Architecture for Agent OS.

This package provides a unified interface for all specialized tools used by
the agent ecosystem, categorized into workspace management, memory,
scheduling, A2A communication, developer utilities, and browser automation.
"""

from typing import Any

from ..tool_registry import register_agent_tools
from .a2a_tools import (
    delete_a2a_peer,
    list_a2a_peers,
    register_a2a_peer,
)
from .agent_tools import (
    invoke_specialized_agent,
    list_available_agents,
    share_reasoning,
)
from .developer_tools import (
    ShellCommandOutput,
    create_file,
    delete_file,
    project_search,
    replace_in_file,
    run_shell_with_diagnostics,
)
from .git_tools import (
    create_worktree,
    get_git_status,
    list_worktrees,
    remove_worktree,
)
from .scheduler_tools import (
    delete_task,
    list_tasks,
    schedule_task,
    view_cron_log,
)
from .style_tools import list_output_styles, set_output_style
from .team_tools import TEAM_TOOLS
from .workspace_tools import (
    append_note_to_file,
    create_skill,
    delete_skill,
    edit_skill,
    get_skill_content,
    list_files,
    read_workspace_file,
)

try:
    from .browser import (
        browser_go_back,
        browser_go_forward,
        browser_new_page,
        browser_status,
        click_element,
        close_browser,
        get_element_text,
        initialize_browser,
        navigate_to_url,
        reload_page,
        select_option,
        take_element_screenshot,
        take_screenshot,
        type_text,
    )
except ImportError:
    # Define placeholders or just omit from __all__ if missing
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

__all__ = [
    "register_agent_tools",
    # Workspace tools
    "read_workspace_file",
    "append_note_to_file",
    "create_skill",
    "delete_skill",
    "edit_skill",
    "get_skill_content",
    "list_files",
    # Scheduler tools
    "schedule_task",
    "list_tasks",
    "delete_task",
    "view_cron_log",
    # A2A tools
    "list_a2a_peers",
    "register_a2a_peer",
    "delete_a2a_peer",
    # Git tools
    "get_git_status",
    "create_worktree",
    "remove_worktree",
    "list_worktrees",
    # Developer tools
    "project_search",
    "replace_in_file",
    "run_shell_with_diagnostics",
    "create_file",
    "delete_file",
    "ShellCommandOutput",
    # Agent tools
    "invoke_specialized_agent",
    "list_available_agents",
    "share_reasoning",
    # Browser tools
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
    # Team tools
    "TEAM_TOOLS",
    # Style tools
    "set_output_style",
    "list_output_styles",
]
