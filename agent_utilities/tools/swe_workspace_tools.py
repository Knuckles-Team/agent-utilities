"""CONCEPT:AU-ORCH.execution.swe-workspace-tools — SWE workspace tools: the agent's *action* surface over the dev-workspace.

These Pydantic-AI tools translate the model's tool calls into typed runtime actions
(``agent_utilities.runtime.events``) executed inside the agent's :class:`DevWorkspace`
(``ctx.deps.workspace``, OS-5.33), and format the resulting observation back as text. Because
the workspace mirrors every action to the KG (KG-2.64) and gates mutations via ``ActionPolicy``
(OS-5.24), the agent inherits provenance and governance for free.

This is the sole mutation and execution surface. General developer tools remain
read-only. It is distinct from ``workspace_tools.py``, which manages SKILL.md and
project-memory metadata.
"""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace

from ..models import AgentDeps
from ..runtime.events import (
    BrowseAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    TestRunAction,
)
from .versioning import tool_version

logger = logging.getLogger(__name__)

_NO_WS = "No developer workspace is attached to this agent (deps.workspace is None)."


def _ws(ctx: RunContext[AgentDeps]):
    return getattr(ctx.deps, "workspace", None)


def _safe_text(value: object) -> str:
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    clean, _ = PersistencePrivacyGuard().sanitize_text(str(value or ""))
    return clean


@trace(name="run_command", trace_type="TOOL")
@tool_version("1.0.0")
async def run_command(
    ctx: RunContext[AgentDeps], command: str, timeout: float = 120.0
) -> str:
    """Run a shell command in the workspace. Working directory persists across calls."""
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    obs = await ws.act(CmdRunAction(command=command, timeout=timeout))
    parts = [f"exit_code={obs.exit_code}"]
    if obs.stdout:
        parts.append(f"stdout:\n{_safe_text(obs.stdout)}")
    if obs.stderr:
        parts.append(f"stderr:\n{_safe_text(obs.stderr)}")
    return "\n".join(parts)


@trace(name="read_file", trace_type="TOOL")
@tool_version("1.0.0")
async def read_file(
    ctx: RunContext[AgentDeps],
    path: str,
    start: int | None = None,
    end: int | None = None,
) -> str:
    """Read a file (optionally a 1-based inclusive line range) from the workspace."""
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    obs = await ws.act(FileReadAction(path=path, start=start, end=end))
    if obs.kind == "error":
        return "ERROR: governed workspace read failed."
    return _safe_text(obs.content)


@trace(name="write_file", trace_type="TOOL")
@tool_version("1.0.0")
async def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """Create or overwrite a file in the workspace."""
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    obs = await ws.act(FileWriteAction(path=path, content=content))
    if obs.kind == "error":
        return "ERROR: governed workspace write failed."
    return f"Wrote {obs.bytes_written} bytes in the governed workspace."


@trace(name="edit_file", trace_type="TOOL")
@tool_version("1.0.0")
async def edit_file(
    ctx: RunContext[AgentDeps], path: str, old: str, new: str, replace_all: bool = False
) -> str:
    """Replace an exact ``old`` string with ``new`` in a file. ``old`` must be unique unless
    ``replace_all`` is set. Returns the unified diff."""
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    obs = await ws.act(
        FileEditAction(path=path, old=old, new=new, replace_all=replace_all)
    )
    if obs.kind == "error":
        return "ERROR: governed workspace edit failed."
    return f"Applied {obs.replacements} replacement(s):\n{_safe_text(obs.diff)}"


@trace(name="run_tests", trace_type="TOOL")
@tool_version("1.0.0")
async def run_tests(
    ctx: RunContext[AgentDeps], selector: str | None = None, framework: str = "pytest"
) -> str:
    """Run the test suite (or a selector like ``tests/test_x.py::test_y``) in the workspace."""
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    obs = await ws.act(TestRunAction(selector=selector, framework=framework))
    return _safe_text(f"{obs.report}\n{obs.raw[-4000:]}")


@trace(name="browse", trace_type="TOOL")
@tool_version("1.0.0")
async def browse(ctx: RunContext[AgentDeps], url: str, interaction: str = "") -> str:
    """Open a URL in the optional browser tier (ECO-4.44). Returns page text or a not-provisioned
    notice if no browser driver is attached to the workspace."""
    ws = _ws(ctx)
    if ws is None:
        return _NO_WS
    obs = await ws.act(BrowseAction(url=url, interaction=interaction))
    if obs.error:
        return "BROWSER: governed browser action failed."
    return _safe_text(f"[{obs.status}] {obs.title}\n{obs.text[:4000]}")


SWE_WORKSPACE_TOOLS = [run_command, read_file, write_file, edit_file, run_tests, browse]
