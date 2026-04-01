import os
import subprocess
import logging
import shutil
from typing import Any
from pydantic_ai import RunContext

logger = logging.getLogger(__name__)


async def get_git_status(ctx: RunContext[Any]) -> str:
    """
    Get a snapshot of the current git status, branch, and recent commits.
    Mirroring Claude Code's system context logic.
    """
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip()
        log = subprocess.check_output(
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
    """
    Create a new git worktree for isolated task execution.
    """
    try:
        # 1. Create the branch if it doesn't exist
        subprocess.run(["git", "branch", branch_name], check=False)

        # 2. Add the worktree
        cmd = ["git", "worktree", "add", path, branch_name]
        subprocess.check_call(cmd)

        return f"Successfully created worktree at '{path}' on branch '{branch_name}'."
    except Exception as e:
        return f"Error creating worktree: {e}"


async def remove_worktree(ctx: RunContext[Any], path: str, force: bool = False) -> str:
    """
    Remove a git worktree and cleanup.
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
    """List all active git worktrees."""
    try:
        return subprocess.check_output(["git", "worktree", "list"], text=True).strip()
    except Exception as e:
        return f"Error listing worktrees: {e}"


# Tool grouping for registration
git_tools = [
    get_git_status,
    create_worktree,
    remove_worktree,
    list_worktrees,
]
