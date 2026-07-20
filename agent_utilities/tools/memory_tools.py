#!/usr/bin/python
"""Privacy-safe, read-only project-memory tools."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

from ..models import AgentDeps
from .versioning import tool_version

_MAX_BYTES = 256 * 1024


def _root(ctx: RunContext[AgentDeps]) -> Path:
    candidate = Path(ctx.deps.workspace_path)
    if candidate.is_symlink():
        raise ValueError("workspace unavailable")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace unavailable")
    return root


def _clean(content: str) -> str:
    sanitized, _ = PersistencePrivacyGuard().sanitize_text(content)
    return sanitized


@trace(name="read_agents_md", trace_type="TOOL")
@tool_version("2.0.0")
async def read_agents_md(ctx: RunContext[AgentDeps]) -> str:
    """Read bounded, sanitized project instructions from the workspace root."""
    try:
        root = _root(ctx)
        content: list[str] = []
        for relative in ("AGENTS.md", ".agents/AGENTS.md", "AGENTS.local.md"):
            path = root / relative
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if resolved.stat().st_size > _MAX_BYTES:
                raise ValueError("project memory exceeds the size limit")
            content.append(_clean(resolved.read_text(encoding="utf-8")))
        return "\n\n".join(content) if content else "No project memory is configured."
    except Exception:
        return "Error reading project memory."


memory_tools = [read_agents_md]
