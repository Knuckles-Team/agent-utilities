#!/usr/bin/python
"""Git Utilities Tools Module.

This module provides tools for inspecting git status, managing isolated
worktrees for parallel development, and auditing version control history.
"""

import logging
import os
import shutil
import subprocess
from typing import Any

from pydantic_ai import RunContext

logger = logging.getLogger(__name__)


async def get_git_status(ctx: RunContext[Any]) -> str:
    """Retrieve a comprehensive snapshot of the current git environment.

    Returns the current branch name, a summarized file status, and the
    last five oneline commit logs.

    Args:
        ctx: The agent run context.

    Returns:
        A formatted summary of the git environment status.

    """
    try:
        branch = subprocess.check_output(  # nosec B607
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(  # nosec B607
            ["git", "status", "--short"], text=True
        ).strip()
        log = subprocess.check_output(  # nosec B607
            ["git", "log", "--oneline", "-n", "5"], text=True
        ).strip()

        return (
            f"Git Status Snapshot:\n"
            f"Current branch: {branch}\n"
            f"Status:\n{status or '(clean)'}\n\n"
            f"Recent commits:\n{log}"
        )
    except Exception as e:
        return f"Error fetching git status: {e}"


async def create_worktree(ctx: RunContext[Any], branch_name: str, path: str) -> str:
    """Create a new git worktree for isolated and parallel feature development.

    This ensures that agents can work on separate branches without
    polluting the primary workspace.

    Args:
        ctx: The agent run context.
        branch_name: The name of the new branch to create/use.
        path: Target absolute or relative directory for the worktree.

    Returns:
        A confirmation message indicating success or an error details.

    """
    try:
        # 1. Create the branch if it doesn't exist
        subprocess.run(["git", "branch", branch_name], check=False)  # nosec B607

        # 2. Add the worktree
        cmd = ["git", "worktree", "add", path, branch_name]
        subprocess.check_call(cmd)

        return f"Successfully created worktree at '{path}' on branch '{branch_name}'."
    except Exception as e:
        return f"Error creating worktree: {e}"


async def remove_worktree(ctx: RunContext[Any], path: str, force: bool = False) -> str:
    """Remove an existing git worktree and clean up the associated directory.

    Args:
        ctx: The agent run context.
        path: The directory path of the worktree to remove.
        force: Whether to force removal even if changes are present.

    Returns:
        A confirmation message indicating success or an error details.

    """
    try:
        cmd = ["git", "worktree", "remove", path]
        if force:
            cmd.append("--force")
        subprocess.check_call(cmd)

        # Also attempt to remove the directory if git left it
        if os.path.exists(path):
            shutil.rmtree(path)

        return f"Successfully removed worktree at '{path}'."
    except Exception as e:
        return f"Error removing worktree: {e}"


async def list_worktrees(ctx: RunContext[Any]) -> str:
    """List all currently active git worktrees in the repository.

    Args:
        ctx: The agent run context.

    Returns:
        A formatted list of active worktrees and their paths.

    """
    try:
        return subprocess.check_output(["git", "worktree", "list"], text=True).strip()  # nosec B607
    except Exception as e:
        return f"Error listing worktrees: {e}"


# Tool grouping for registration
git_tools = [
    get_git_status,
    create_worktree,
    remove_worktree,
    list_worktrees,
]
