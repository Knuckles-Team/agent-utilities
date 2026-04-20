#!/usr/bin/python
# coding: utf-8
"""Developer Utilities Tools Module.

This module provides high-level tools for code search, file manipulation,
and shell command execution with diagnostics and error handling.
"""

import os
import subprocess
import logging
import difflib
import asyncio
from pydantic import BaseModel
from pydantic_ai import RunContext
from ..models import AgentDeps

logger = logging.getLogger(__name__)

from .knowledge_tools import (
    search_knowledge_graph,
    add_knowledge_memory,
    get_knowledge_memory,
    update_knowledge_memory,
    delete_knowledge_memory,
    link_knowledge_nodes,
    sync_feature_to_memory,
    get_code_impact,
)


class ShellCommandOutput(BaseModel):
    """Structured output from a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


def _get_diff(old_content: str, new_content: str, filename: str) -> str:
    """Generate a unified diff between two strings.

    Args:
        old_content: The original file content.
        new_content: The modified file content.
        filename: The name of the file being diffed.

    Returns:
        A unified diff string.

    """
    return "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=3,
        )
    )


async def project_search(
    ctx: RunContext[AgentDeps], query: str, path: str = "."
) -> str:
    """Perform an optimized grep/ripgrep search across the codebase.

    Searches for literal strings and returns matching lines prefixed with
    filenames and line numbers.

    Args:
        ctx: The agent run context.
        query: The string to search for.
        path: The directory or file to search within.

    Returns:
        A string containing the search results or a 'no matches' message.

    """
    try:
        # Prefer ripgrep if available
        cmd = [
            "rg",
            "--line-number",
            "--column",
            "--no-heading",
            "--fixed-strings",
            query,
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout or "No matches found."

        # Fallback to grep
        cmd = ["grep", "-rni", query, path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout or "No matches found."
    except Exception as e:
        return f"Error during search: {e}"


async def replace_in_file(
    ctx: RunContext[AgentDeps], path: str, old_str: str, new_str: str
) -> str:
    """Perform a find-and-replace operation on a file with diff validation.

    Replaces the first occurrence of a specified string and provides
    a unified diff of the changes.

    Args:
        ctx: The agent run context.
        path: Relative or absolute path to the target file.
        old_str: The exact character sequence to replace.
        new_str: The replacement character sequence.

    Returns:
        A status message and a unified diff of the changes.

    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return f"Error: File '{path}' not found."

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_str not in content:
            return f"Error: Search string not found in '{path}'."

        new_content = content.replace(old_str, new_str, 1)
        diff = _get_diff(content, new_content, os.path.basename(path))

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Successfully updated {path}.\n\nDiff:\n{diff}"
    except Exception as e:
        return f"Error updating file: {e}"


async def run_shell_with_diagnostics(
    ctx: RunContext[AgentDeps], command: str, cwd: str = ".", timeout: int = 120
) -> ShellCommandOutput:
    """Execute a shell command with real-time diagnostics and timeouts.

    Args:
        ctx: The agent run context.
        command: The shell command string to execute.
        cwd: Current working directory for the command.
        timeout: Maximum execution time in seconds.

    Returns:
        A ShellCommandOutput object containing stdout, stderr, and exit code.

    """
    import time

    start_time = time.time()

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return ShellCommandOutput(
                stdout=stdout.decode(),
                stderr=stderr.decode() + "\n[Command timed out]",
                exit_code=-1,
                duration_ms=(time.time() - start_time) * 1000,
            )

        return ShellCommandOutput(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            exit_code=process.returncode or 0,
            duration_ms=(time.time() - start_time) * 1000,
        )
    except Exception as e:
        return ShellCommandOutput(
            stdout="",
            stderr=str(e),
            exit_code=1,
            duration_ms=(time.time() - start_time) * 1000,
        )


# Ported from code_puppy: create_file, delete_file, delete_snippet
async def create_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """Create a new file in the workspace, creating parent directories if needed.

    Args:
        ctx: The agent run context.
        path: Target path for the new file.
        content: The text content to write to the file.

    Returns:
        A confirmation message or an error description.

    """
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Created file: {path}"
    except Exception as e:
        return f"Error creating file: {e}"


async def delete_file(ctx: RunContext[AgentDeps], path: str) -> str:
    """Permanently delete a file from the local workspace.

    Args:
        ctx: The agent run context.
        path: The path to the file to be deleted.

    Returns:
        A confirmation message or an error description.

    """
    abs_path = os.path.abspath(path)
    try:
        if os.path.exists(abs_path):
            os.remove(abs_path)
            return f"Deleted file: {path}"
        return f"File '{path}' does not exist."
    except Exception as e:
        return f"Error deleting file: {e}"


# Tool grouping for registration
developer_tools = [
    project_search,
    replace_in_file,
    run_shell_with_diagnostics,
    create_file,
    delete_file,
    search_knowledge_graph,
    add_knowledge_memory,
    get_knowledge_memory,
    update_knowledge_memory,
    delete_knowledge_memory,
    link_knowledge_nodes,
    sync_feature_to_memory,
    get_code_impact,
]
