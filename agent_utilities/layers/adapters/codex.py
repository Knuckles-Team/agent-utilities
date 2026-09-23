"""Codex adapter: non-interactive ``codex exec --json``.

The run is ephemeral (``--ephemeral``), ignores the operator's user config and
exec-policy rules (``--ignore-user-config --ignore-rules``; login still comes
from ``CODEX_HOME``), is rooted in the leased workspace (``-C``) under Codex's
own ``workspace-write`` sandbox, and receives the run's MCP endpoints as
``-c mcp_servers.<name>.*`` overrides with bearer tokens passed by environment
variable name only.

Honest limits, enforced by negotiation through the descriptor: ``codex exec``
emits no startup tool inventory, so a RunSpec with required tools is refused;
skills delivery is not proven, so a RunSpec carrying skills is refused; the
built-in shell cannot be allowlisted, so an allowlisted RunSpec is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.layers.cli_harness import (
    CLI_DESCRIPTOR_DEFAULTS,
    CliHarness,
    Handler,
    Invocation,
    StreamState,
    as_mapping,
    token_env_name,
)
from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    UsageRecord,
    VendorTerms,
)
from agent_utilities.layers.session import RunContext

HARNESS_NAME = "codex"

DESCRIPTOR = HarnessDescriptor(
    **CLI_DESCRIPTOR_DEFAULTS,
    name=HARNESS_NAME,
    version="codex-exec/jsonl",
    fidelity="tool-calls",
    capabilities=frozenset({"code_edit", "shell", "mcp_client", "cancellation"}),
    usage_quality="measured",
    enforceable_budgets=frozenset({"wall_time"}),
    tool_proof="none",
    skill_proof="none",
    max_skills=0,
    vendor_terms=VendorTerms(
        subscription_automation_allowed=True,
        note="codex exec with a ChatGPT plan login is vendor-documented; "
        "plan usage limits apply",
    ),
)

#: ``item.type`` values that are tool invocations.
_TOOL_ITEMS = frozenset({"command_execution", "mcp_tool_call", "web_search"})


def _item(record: dict) -> dict:
    item = record.get("item")
    return item if isinstance(item, dict) else {}


def _tool_name(item: dict) -> str:
    if item.get("type") == "mcp_tool_call":
        return f"{item.get('server', '')}/{item.get('tool', '')}"
    return str(item.get("type") or "")


def _on_thread_started(run: RunContext, record: dict, state: StreamState) -> None:
    run.provider_session = str(record.get("thread_id") or "") or None
    run.emit("step", "observation", name="thread.started")


def _on_item_started(run: RunContext, record: dict, state: StreamState) -> None:
    item = _item(record)
    if item.get("type") not in _TOOL_ITEMS:
        return
    state.tool_calls += 1
    run.emit(
        "tool_call",
        "observation",
        name=_tool_name(item),
        detail=str(item.get("command") or json.dumps(item.get("arguments")))[:2_000],
        data={"call_id": str(item.get("id") or "")},
    )


def _on_item_completed(run: RunContext, record: dict, state: StreamState) -> None:
    item = _item(record)
    kind = item.get("type")
    if kind == "agent_message":
        state.output = str(item.get("text") or "")
        run.emit("message", "claim", detail=state.output)
    elif kind in _TOOL_ITEMS:
        run.emit(
            "tool_result",
            "observation",
            name=_tool_name(item),
            detail=str(item.get("aggregated_output") or item.get("result") or "")[
                :2_000
            ],
            data={
                "call_id": str(item.get("id") or ""),
                "status": str(item.get("status") or ""),
            },
        )
    elif kind == "file_change":
        run.emit(
            "artifact", "observation", name="file_change", detail=str(item)[:2_000]
        )


def _on_turn_completed(run: RunContext, record: dict, state: StreamState) -> None:
    usage = as_mapping(record.get("usage"))
    state.terminal = True
    state.usage = UsageRecord(
        quality="measured",
        source="codex:turn.completed",
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
    )
    run.emit(
        "usage",
        "observation",
        name="turn.completed",
        data={
            "input_tokens": state.usage.input_tokens,
            "output_tokens": state.usage.output_tokens,
        },
    )


def _on_failure(run: RunContext, record: dict, state: StreamState) -> None:
    error = record.get("error")
    message = error.get("message") if isinstance(error, dict) else record.get("message")
    state.terminal = True
    state.failed = True
    state.error = str(message or record.get("type"))
    run.emit("error", "observation", name=str(record.get("type")), detail=state.error)


_HANDLERS: dict[str, Handler] = {
    "thread.started": _on_thread_started,
    "item.started": _on_item_started,
    "item.completed": _on_item_completed,
    "turn.completed": _on_turn_completed,
    "turn.failed": _on_failure,
    "error": _on_failure,
}


def _mcp_overrides(run: RunContext) -> list[str]:
    overrides: list[str] = []
    for index, endpoint in enumerate(run.spec.toolset.endpoints()):
        prefix = f"mcp_servers.{endpoint.name}"
        overrides += ["-c", f"{prefix}.url={json.dumps(endpoint.url)}"]
        if endpoint.bearer_ref is not None:
            env_name = json.dumps(token_env_name(index))
            overrides += ["-c", f"{prefix}.bearer_token_env_var={env_name}"]
    return overrides


class CodexHarness(CliHarness):
    """:class:`HarnessPort` over the Codex CLI."""

    binary = "codex"
    descriptor = DESCRIPTOR
    record_handlers = _HANDLERS

    def materialize(self, run: RunContext, config_dir: Path) -> None:
        (config_dir / "run.json").write_text(
            json.dumps({"run_id": run.spec.run_id, "spec": run.negotiated.spec_digest}),
            encoding="utf-8",
        )

    def invocation(
        self, run: RunContext, binary_path: str, config_dir: Path
    ) -> Invocation:
        spec = run.spec
        argv = [
            binary_path,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "-C",
            str(config_dir.parent),
            "-s",
            "workspace-write",
            *_mcp_overrides(run),
        ]
        if spec.model:
            argv += ["-m", spec.model]
        argv.append("-")
        env = self.launch_env(run, "CODEX_API_KEY")
        return Invocation(argv=tuple(argv), env=env, stdin_text=spec.task)


__all__ = ["DESCRIPTOR", "HARNESS_NAME", "CodexHarness"]
