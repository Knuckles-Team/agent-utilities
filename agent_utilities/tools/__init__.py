#!/usr/bin/python
"""Modular Tool Architecture for Agent OS.

This package provides a unified interface for all specialized tools used by
the agent ecosystem, categorized into workspace management, memory,
scheduling, A2A communication, developer utilities, and browser automation.
"""

import importlib
from typing import Any

_ATTR_MAP = {
    # tool_registry
    "register_agent_tools": ".tool_registry",
    # a2a_tools
    "delete_a2a_peer": ".a2a_tools",
    "list_a2a_peers": ".a2a_tools",
    "register_a2a_peer": ".a2a_tools",
    # agent_tools
    "invoke_specialized_agent": ".agent_tools",
    "list_available_agents": ".agent_tools",
    "share_reasoning": ".agent_tools",
    # developer_tools
    "ShellCommandOutput": ".developer_tools",
    "create_file": ".developer_tools",
    "delete_file": ".developer_tools",
    "project_search": ".developer_tools",
    "replace_in_file": ".developer_tools",
    "run_shell_with_diagnostics": ".developer_tools",
    # git_tools
    "create_worktree": ".git_tools",
    "get_git_status": ".git_tools",
    "list_worktrees": ".git_tools",
    "remove_worktree": ".git_tools",
    # memory_tools
    "init_agents_md": ".memory_tools",
    "read_agents_md": ".memory_tools",
    "update_agents_md": ".memory_tools",
    # scheduler_tools
    "delete_task": ".scheduler_tools",
    "list_tasks": ".scheduler_tools",
    "schedule_task": ".scheduler_tools",
    "view_cron_log": ".scheduler_tools",
    # style_tools
    "list_output_styles": ".style_tools",
    "set_output_style": ".style_tools",
    # team_tools
    "TEAM_TOOLS": ".team_tools",
    # workspace_tools
    "append_note_to_file": ".workspace_tools",
    "create_skill": ".workspace_tools",
    "delete_skill": ".workspace_tools",
    "edit_skill": ".workspace_tools",
    "get_skill_content": ".workspace_tools",
    "list_files": ".workspace_tools",
    "read_workspace_file": ".workspace_tools",
    # x_search_tool
    "x_search": ".x_search_tool",
    "browse_x_post": ".x_search_tool",
    # browser
    "browser_go_back": ".browser",
    "browser_go_forward": ".browser",
    "browser_new_page": ".browser",
    "browser_status": ".browser",
    "click_element": ".browser",
    "close_browser": ".browser",
    "get_element_text": ".browser",
    "initialize_browser": ".browser",
    "navigate_to_url": ".browser",
    "reload_page": ".browser",
    "select_option": ".browser",
    "take_element_screenshot": ".browser",
    "take_screenshot": ".browser",
    "type_text": ".browser",
}

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
    # X (formerly Twitter) tools
    "x_search",
    "browse_x_post",
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
    # Memory tools
    "init_agents_md",
    "read_agents_md",
    "update_agents_md",
]

_cached_modules = {}


def __getattr__(name: str) -> Any:
    if name in _ATTR_MAP:
        submodule_path = _ATTR_MAP[name]
        if submodule_path not in _cached_modules:
            try:
                _cached_modules[submodule_path] = importlib.import_module(
                    submodule_path, __package__
                )
            except ImportError:
                # Browser or other optional tool failure fallback
                if submodule_path == ".browser":
                    return None
                raise
        module = _cached_modules[submodule_path]
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(__all__) + ["__getattr__", "__dir__"])
