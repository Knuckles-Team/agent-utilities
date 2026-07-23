#!/usr/bin/python
"""Workspace-confined read, note, skill, and file-list tools."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

from .versioning import tool_version

_SKILL_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_MAX_TEXT_BYTES = 1024 * 1024
_MAX_SKILL_BYTES = 256 * 1024
_MAX_LIST_ENTRIES = 10_000
_DENIED_FILENAMES = {
    ".env",
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "mcp_config.json",
    "secrets.json",
}


def _workspace_root(ctx: RunContext[Any]) -> Path:
    configured = getattr(getattr(ctx, "deps", None), "workspace_path", None)
    if not configured:
        raise ValueError("assigned workspace is unavailable")
    candidate = Path(configured)
    if candidate.is_symlink():
        raise ValueError("assigned workspace is unavailable")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("assigned workspace is unavailable")
    return root


def _workspace_path(
    ctx: RunContext[Any], value: str, *, must_exist: bool = False
) -> tuple[Path, Path]:
    root = _workspace_root(ctx)
    supplied = Path(str(value or "."))
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("path is outside the assigned workspace") from None
    return root, resolved


def _safe_read(path: Path, *, limit: int = _MAX_TEXT_BYTES) -> str:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.name.lower() in _DENIED_FILENAMES
    ):
        raise ValueError("workspace file is unavailable or sensitive")
    if path.stat().st_size > limit:
        raise ValueError("workspace file exceeds the read limit")
    content = path.read_text(encoding="utf-8")
    clean, _ = PersistencePrivacyGuard().sanitize_text(content)
    return clean


def _skill_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SKILL_RE.fullmatch(normalized):
        raise ValueError("skill name must be a neutral bounded slug")
    return normalized


def _skill_file(ctx: RunContext[Any], name: str) -> tuple[Path, Path]:
    root = _workspace_root(ctx)
    skill_dir = root / ".agents" / "skills" / _skill_name(name)
    return skill_dir, skill_dir / "SKILL.md"


@trace(name="read_workspace_file", trace_type="TOOL")
@tool_version("2.0.0")
async def read_workspace_file(ctx: RunContext[Any], filename: str) -> str:
    """Read one bounded non-sensitive text file from the assigned workspace."""
    try:
        _, path = _workspace_path(ctx, filename, must_exist=True)
        if path.name.lower() == "mcp_config.json":
            # Configuration inspection returns topology metadata only; commands,
            # environment values, endpoints, and credentials remain runtime-only.
            data = json.loads(path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {}) if isinstance(data, dict) else {}
            if not isinstance(servers, dict):
                raise ValueError("invalid MCP configuration")
            return f"MCP configuration: server_count={len(servers)}; redacted=true"
        return _safe_read(path)
    except Exception:
        return "Error reading workspace file."


@trace(name="get_skill_content", trace_type="TOOL")
@tool_version("2.0.0")
async def get_skill_content(ctx: RunContext[Any], name: str) -> str:
    """Read a bounded workspace-local skill definition."""
    try:
        _, skill_file = _skill_file(ctx, name)
        return _safe_read(skill_file, limit=_MAX_SKILL_BYTES)
    except Exception:
        return "Error reading workspace skill."


@trace(name="list_files", trace_type="TOOL")
@tool_version("2.0.0")
async def list_files(
    ctx: RunContext[Any], path: str = ".", recursive: bool = False
) -> str:
    """List bounded repository-relative paths without following symlinks."""
    try:
        root, start = _workspace_path(ctx, path, must_exist=True)
        if start.is_symlink() or not start.is_dir():
            raise ValueError("list target is unavailable")
        files: list[str] = []
        for current, dirs, filenames in os.walk(start, followlinks=False):
            current_path = Path(current)
            dirs[:] = sorted(
                name for name in dirs if not (current_path / name).is_symlink()
            )
            for filename in sorted(filenames):
                candidate = current_path / filename
                if candidate.is_symlink() or filename.lower() in _DENIED_FILENAMES:
                    continue
                files.append(candidate.relative_to(root).as_posix())
                if len(files) >= _MAX_LIST_ENTRIES:
                    return "\n".join(files) + "\n[results truncated]"
            if not recursive:
                break
        return "\n".join(files)
    except Exception:
        return "Error listing workspace files."


workspace_tools = [
    read_workspace_file,
    get_skill_content,
    list_files,
]
