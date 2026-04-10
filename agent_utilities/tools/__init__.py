"""
Modular Tool Architecture for Agent OS.
This module re-exports all specialized tools for easier registration.
"""

from ..tool_registry import register_agent_tools
from .workspace_tools import (
    read_workspace_file,
    append_note_to_file,
    create_skill,
    delete_skill,
    edit_skill,
    get_skill_content,
    list_files,
)
from .memory_tools import (
    create_memory,
    search_memory,
    delete_memory_entry,
    compress_memory,
)
from .scheduler_tools import (
    schedule_task,
    list_tasks,
    delete_task,
    view_cron_log,
)
from .a2a_tools import (
    list_a2a_peers,
    register_a2a_peer,
    delete_a2a_peer,
)
from .git_tools import (
    get_git_status,
    create_worktree,
    remove_worktree,
    list_worktrees,
)
from .developer_tools import (
    project_search,
    replace_in_file,
    run_shell_with_diagnostics,
    create_file,
    delete_file,
    ShellCommandOutput,
)
from .agent_tools import (
    invoke_specialized_agent,
    list_available_agents,
    share_reasoning,
)
from .browser import (
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
)

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
    # Memory tools
    "create_memory",
    "search_memory",
    "delete_memory_entry",
    "compress_memory",
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
]
