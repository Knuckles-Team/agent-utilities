#!/usr/bin/python
"""Privacy-safe, workspace-confined Git inspection tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace
from agent_utilities.security.persistence_privacy import persistence_reference

from .versioning import tool_version


def _workspace(ctx: RunContext[Any]) -> Path:
    configured = getattr(getattr(ctx, "deps", None), "workspace_path", None)
    if not configured:
        raise ValueError("assigned workspace is unavailable")
    root = Path(configured)
    if root.is_symlink():
        raise ValueError("assigned workspace is unavailable")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or not (resolved / ".git").exists():
        raise ValueError("assigned workspace is not a Git checkout")
    return resolved


def _git_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name
        in {
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "WINDIR",
        }
    }


def _git(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("git")
    if not executable:
        raise RuntimeError("git executable is unavailable")
    command = [executable, *args]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.run(
            command,
            cwd=root,
            env=_git_environment(),
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=30,
        )
        stdout.seek(0)
        output = stdout.read(1024 * 1024 + 1)
    if len(output) > 1024 * 1024:
        raise RuntimeError("git output exceeded the resource limit")
    result = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=output.decode("utf-8", errors="replace"),
        stderr="",
    )
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


@trace(name="get_git_status", trace_type="TOOL")
@tool_version("2.0.0")
async def get_git_status(ctx: RunContext[Any]) -> str:
    """Return bounded checkout state without host paths or commit messages."""
    try:
        root = _workspace(ctx)
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        status_lines = [
            line
            for line in _git(
                root, "status", "--short", "--untracked-files=normal"
            ).stdout.splitlines()
            if line
        ]
        commit_count = _git(root, "rev-list", "--count", "HEAD").stdout.strip()
        staged = sum(line[:1] not in {" ", "?"} for line in status_lines)
        worktree = sum(len(line) > 1 and line[1] != " " for line in status_lines)
        untracked = sum(line.startswith("??") for line in status_lines)
        branch_ref = persistence_reference("git_branch", branch)
        return (
            "Git status: "
            f"branch_ref={branch_ref}; commits={commit_count or '0'}; "
            f"changed={len(status_lines)}; staged={staged}; "
            f"worktree={worktree}; untracked={untracked}."
        )
    except Exception:
        return "Error fetching Git status."


@trace(name="list_worktrees", trace_type="TOOL")
@tool_version("2.0.0")
async def list_worktrees(ctx: RunContext[Any]) -> str:
    """List worktree counts and pseudonymous branches without host paths."""
    try:
        root = _workspace(ctx)
        records = _git(root, "worktree", "list", "--porcelain").stdout.split("\n\n")
        branch_refs: list[str] = []
        count = 0
        for record in records:
            if not record.strip():
                continue
            count += 1
            branch_line = next(
                (line for line in record.splitlines() if line.startswith("branch ")),
                "",
            )
            if branch_line:
                branch_refs.append(
                    persistence_reference(
                        "git_branch", branch_line.removeprefix("branch ")
                    )
                )
        return f"Git worktrees: count={count}; branch_refs={','.join(branch_refs)}"
    except Exception:
        return "Error listing Git worktrees."


git_tools = [get_git_status, list_worktrees]
