"""The ``claude-code`` harness: headless Claude Code behind :class:`HarnessPort`.

The adapter launches ``claude -p --output-format json`` once per run, with the
prompt on stdin and the working directory set to the provided worktree. Flags
are pinned. ``--strict-mcp-config`` with ``--mcp-config`` makes graph-os the only
MCP server, so every MCP tool call goes through graph-os. ``--setting-sources
project`` keeps user settings, hooks and plugins out of the run. The adapter
reads only the final JSON result object; it never scrapes the TUI.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from agent_utilities.layers.harness_port import (
    DiffStat,
    HarnessRequest,
    OutcomeStatus,
    RunOutcome,
    UsageReport,
    refused,
)
from agent_utilities.layers.harness_process import (
    ProcessResult,
    child_environment,
    read_diff_stat,
    run_process,
)

CLAUDE_CODE_HARNESS = "claude-code"
#: Flags every run carries, in this order.
CLAUDE_PINNED_FLAGS: tuple[str, ...] = (
    "-p",
    "--output-format",
    "json",
    "--strict-mcp-config",
    "--setting-sources",
    "project",
)
#: The MCP server name graph-os takes in a generated configuration.
GRAPHOS_SERVER = "graph-os"

DiffReader = Callable[[Path], Awaitable[DiffStat | None]]


def graphos_mcp_config(url: str, *, token_env: str | None = None) -> dict[str, Any]:
    """An MCP configuration naming graph-os as the only server.

    ``token_env`` names a variable the CLI expands at load time; the token
    itself is never written to the file.
    """
    server: dict[str, Any] = {"type": "http", "url": url}
    if token_env:
        server["headers"] = {"Authorization": f"Bearer ${{{token_env}}}"}
    return {"mcpServers": {GRAPHOS_SERVER: server}}


def write_graphos_mcp_config(
    path: Path, url: str, *, token_env: str | None = None
) -> Path:
    """Write :func:`graphos_mcp_config` to ``path`` and return the path."""
    path.write_text(json.dumps(graphos_mcp_config(url, token_env=token_env)))
    return path


def _result_object(stdout: str) -> dict[str, Any] | None:
    """The last ``{"type": "result"}`` JSON object on stdout, if any."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict) and parsed.get("type") == "result":
            return parsed
    return None


def _usage(report: Mapping[str, Any]) -> UsageReport:
    raw = report.get("usage") or {}
    cost = report.get("total_cost_usd")
    return UsageReport(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_read_input_tokens=int(raw.get("cache_read_input_tokens") or 0),
        cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        cost_usd=None if cost is None else float(cost),
    )


def _model(report: Mapping[str, Any]) -> str:
    models = report.get("modelUsage")
    return next(iter(models), "") if isinstance(models, dict) else ""


def _status(result: ProcessResult, report: Mapping[str, Any] | None) -> OutcomeStatus:
    if result.timed_out:
        return "timeout"
    if report is None or result.exit_code != 0 or report.get("is_error"):
        return "failed"
    return "completed"


def _error(result: ProcessResult, report: Mapping[str, Any] | None) -> str | None:
    if result.timed_out:
        return "claude-code run exceeded its timeout"
    if report is None:
        return f"no JSON result (exit {result.exit_code}): {result.stderr_tail[-500:]}"
    if result.exit_code != 0 or report.get("is_error"):
        return str(report.get("subtype") or f"exit {result.exit_code}")
    return None


def claude_outcome(
    request: HarnessRequest,
    result: ProcessResult,
    diff: DiffStat | None,
    duration_ms: float,
) -> RunOutcome:
    """Read one finished ``claude -p --output-format json`` run."""
    report = _result_object(result.stdout)
    session = (report or {}).get("session_id")
    return RunOutcome(
        run_id=request.run_id,
        harness=CLAUDE_CODE_HARNESS,
        agent_name=request.agent_name,
        status=_status(result, report),
        final_text=str((report or {}).get("result") or ""),
        structured_output=(report or {}).get("structured_output"),
        exit_code=result.exit_code,
        transcript_ref=f"claude-code:session:{session}" if session else None,
        diff_stat=diff,
        usage=None if report is None else _usage(report),
        model=_model(report or {}),
        duration_ms=duration_ms,
        error=_error(result, report),
    )


class ClaudeCodeHarness:
    """L4 adapter over headless Claude Code."""

    def __init__(
        self,
        *,
        mcp_config: Path,
        binary: str = "claude",
        permission_mode: str = "dontAsk",
        max_budget_usd: float | None = None,
        env: Mapping[str, str] | None = None,
        diff_reader: DiffReader = read_diff_stat,
    ) -> None:
        self._mcp_config = mcp_config
        self._binary = binary
        self._permission_mode = permission_mode
        self._max_budget_usd = max_budget_usd
        self._env = dict(env or {})
        self._diff_reader = diff_reader

    @property
    def name(self) -> str:
        return CLAUDE_CODE_HARNESS

    def argv(self, executable: str, request: HarnessRequest) -> list[str]:
        """The pinned command line for one run (prompt goes on stdin)."""
        argv = [executable, *CLAUDE_PINNED_FLAGS]
        argv += ["--permission-mode", self._permission_mode]
        if request.model:
            argv += ["--model", request.model]
        if self._max_budget_usd is not None:
            argv += ["--max-budget-usd", f"{self._max_budget_usd:g}"]
        if request.allowed_tools:
            argv += ["--allowedTools", ",".join(request.allowed_tools)]
        return [*argv, "--mcp-config", str(self._mcp_config)]

    def _refusal(self, request: HarnessRequest) -> str | None:
        if request.workspace is None or not request.workspace.is_dir():
            return "claude-code requires an existing worktree as its workspace"
        if not self._mcp_config.is_file():
            return f"MCP configuration {self._mcp_config} is missing"
        if shutil.which(self._binary) is None:
            return f"{self._binary!r} is not installed on this host"
        return None

    async def run(self, request: HarnessRequest) -> RunOutcome:
        reason = self._refusal(request)
        if reason is not None:
            return refused(self.name, request, reason)
        workspace = Path(str(request.workspace))
        executable = str(shutil.which(self._binary))
        start = time.monotonic()
        result = await run_process(
            self.argv(executable, request),
            cwd=workspace,
            stdin_text=request.task,
            timeout_s=request.timeout_s,
            env=child_environment(self._env),
        )
        duration_ms = (time.monotonic() - start) * 1000
        diff = await self._diff_reader(workspace)
        return claude_outcome(request, result, diff, duration_ms)


__all__ = [
    "CLAUDE_CODE_HARNESS",
    "CLAUDE_PINNED_FLAGS",
    "GRAPHOS_SERVER",
    "ClaudeCodeHarness",
    "claude_outcome",
    "graphos_mcp_config",
    "write_graphos_mcp_config",
]
