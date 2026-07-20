#!/usr/bin/python
"""Read-only developer discovery tools and knowledge-graph utilities.

CONCEPT:AU-ECO.messaging.native-backend-abstraction

Code mutations and command execution belong to the governed DevWorkspace tool
surface. This module deliberately exposes only bounded workspace search plus
the shared knowledge tools used by general-purpose agents.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace

from ..models import AgentDeps
from .knowledge_tools import (
    add_knowledge_memory,
    delete_knowledge_memory,
    get_code_impact,
    get_knowledge_memory,
    link_knowledge_nodes,
    search_knowledge_graph,
    sync_feature_to_memory,
    update_knowledge_memory,
)
from .versioning import tool_version

logger = logging.getLogger(__name__)


class WorkspaceBoundaryError(ValueError):
    """A developer read attempted to leave its assigned workspace."""


def _workspace_root(ctx: RunContext[AgentDeps]) -> Path:
    configured = getattr(ctx.deps, "workspace_path", None)
    if not configured:
        raise WorkspaceBoundaryError("no developer workspace is assigned")
    try:
        root = Path(configured).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkspaceBoundaryError("developer workspace is unavailable") from None
    if not root.is_dir():
        raise WorkspaceBoundaryError("developer workspace is unavailable")
    return root


def _workspace_path(
    ctx: RunContext[AgentDeps], path: str, *, must_exist: bool = False
) -> Path:
    root = _workspace_root(ctx)
    supplied = Path(str(path or "."))
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise WorkspaceBoundaryError("path is outside the assigned workspace") from None
    return resolved


@trace(name="project_search", trace_type="TOOL")
@tool_version("2.0.0")
async def project_search(
    ctx: RunContext[AgentDeps], query: str, path: str = "."
) -> str:
    """Search for a bounded literal string within the assigned workspace."""
    if not isinstance(query, str) or not query or len(query.encode("utf-8")) > 4_096:
        return "Error during search: query is empty or exceeds the input limit."
    try:
        from agent_utilities.core.config import setting

        root = _workspace_root(ctx)
        search_path = _workspace_path(ctx, path, must_exist=True)
        target = search_path.relative_to(root).as_posix() or "."
        max_output = int(setting("DEVELOPER_TOOL_MAX_OUTPUT_BYTES", 65_536) or 65_536)
        max_output = max(1_024, min(max_output, 4 * 1024 * 1024))

        async def _run(command: list[str]) -> tuple[int, str]:
            options: dict[str, object] = {}
            if os.name == "posix":
                options["start_new_session"] = True
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env={
                    name: value
                    for name, value in os.environ.items()
                    if name
                    in {
                        "PATH",
                        "PATHEXT",
                        "SYSTEMROOT",
                        "WINDIR",
                        "LANG",
                        "LC_ALL",
                    }
                },
                **options,
            )
            retained = bytearray()

            async def _drain() -> None:
                if process.stdout is None:
                    return
                while chunk := await process.stdout.read(8_192):
                    if len(retained) < max_output:
                        retained.extend(chunk[: max_output - len(retained)])

            drain = asyncio.create_task(_drain())
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
                await asyncio.wait_for(drain, timeout=5)
            except TimeoutError:
                if os.name == "posix":
                    with __import__("contextlib").suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                await process.wait()
                drain.cancel()
                with __import__("contextlib").suppress(asyncio.CancelledError):
                    await drain
                return -1, ""
            return process.returncode or 0, retained.decode(errors="replace")

        import shutil

        if shutil.which("rg"):
            status, output = await _run(
                [
                    "rg",
                    "--line-number",
                    "--column",
                    "--no-heading",
                    "--fixed-strings",
                    "--",
                    query,
                    target,
                ]
            )
        else:
            status, output = await _run(["grep", "-rni", "--", query, target])
        if status in {0, 1}:
            return output or "No matches found."
        return "Error during search."
    except WorkspaceBoundaryError as exc:
        return f"Error during search: {exc}"
    except Exception:
        return "Error during search."


developer_tools = [
    project_search,
    search_knowledge_graph,
    add_knowledge_memory,
    get_knowledge_memory,
    update_knowledge_memory,
    delete_knowledge_memory,
    link_knowledge_nodes,
    sync_feature_to_memory,
    get_code_impact,
]
