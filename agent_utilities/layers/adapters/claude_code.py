"""Claude Code adapter: headless ``claude -p --output-format stream-json``.

Delivery (RF-ADR-010 §6.5): a per-run MCP configuration (``--strict-mcp-config``
so nothing else connects), a per-run project skills directory, and the
governance-derived permission fence from :mod:`agent_utilities.claude_harness`
written as the workspace's project settings (``--setting-sources project``:
no user settings, hooks or plugins load). ``--permission-mode dontAsk`` denies
anything the allowlist/fence does not pre-approve, so an unattended run never
waits on a prompt.

The ``system/init`` record is the startup inventory. Every required tool,
every requested MCP server (connected) and every delivered skill must appear
there or the run fails loudly before the model acts -- the known ``-p``
failure where remote MCP tools are missing (claude-code#43298) cannot pass.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from agent_utilities.layers.cli_harness import (
    CliHarness,
    Handler,
    Invocation,
    StreamState,
    as_mapping,
    mcp_server_entries,
    write_skills,
)
from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    HarnessToolInventoryMismatch,
    UsageRecord,
    VendorTerms,
)
from agent_utilities.layers.credentials import api_key_for
from agent_utilities.layers.session import RunContext

HARNESS_NAME = "claude-code"

DESCRIPTOR = HarnessDescriptor(
    name=HARNESS_NAME,
    version="claude-cli/stream-json",
    fidelity="tool-calls",
    capabilities=frozenset(
        {
            "code_edit",
            "shell",
            "browse",
            "sub_agents",
            "mcp_client",
            "skills",
            "tool_allowlist",
            "cancellation",
        }
    ),
    usage_quality="measured",
    enforceable_budgets=frozenset({"cost_usd", "wall_time"}),
    account_modes=frozenset({"api_key", "subscription"}),
    environment_modes=frozenset({"caller-managed-host"}),
    tool_proof="startup_inventory",
    skill_proof="startup_inventory",
    max_skills=64,
    reconciliation="provider_session",
    vendor_terms=VendorTerms(
        subscription_automation_allowed=True,
        note="headless -p use of a Claude subscription is vendor-documented; "
        "plan rate limits apply",
    ),
)

_MAX_INPUT_PREVIEW = 2_000


def _connected_servers(record: dict) -> set[str]:
    servers = [
        item for item in record.get("mcp_servers") or () if isinstance(item, dict)
    ]
    return {
        str(item.get("name")) for item in servers if item.get("status") == "connected"
    }


def _wanted(run: RunContext) -> list[str]:
    toolset = run.spec.toolset
    return (
        [f"tool:{name}" for name in toolset.required_tools]
        + [f"mcp:{endpoint.name}" for endpoint in toolset.endpoints()]
        + [f"skill:{skill.name}" for skill in toolset.skills]
    )


def _present(record: dict) -> set[str]:
    tools = record.get("tools") or ()
    skills = record.get("skills") or ()
    return (
        {f"tool:{name}" for name in tools}
        | {f"mcp:{name}" for name in _connected_servers(record)}
        | {f"skill:{name}" for name in skills}
    )


def _inventory_gaps(run: RunContext, record: dict) -> list[str]:
    """Required tools, MCP servers and skills the init record does not show."""
    present = _present(record)
    return [item for item in _wanted(run) if item not in present]


def _on_system(run: RunContext, record: dict, state: StreamState) -> None:
    if record.get("subtype") != "init":
        return
    run.provider_session = str(record.get("session_id") or "") or None
    gaps = _inventory_gaps(run, record)
    run.emit(
        "step",
        "observation",
        name="init",
        data={
            "tools": len(record.get("tools") or ()),
            "harness_version": str(record.get("claude_code_version") or ""),
            "model": str(record.get("model") or ""),
            "missing": list(gaps),
        },
    )
    if gaps:
        raise HarnessToolInventoryMismatch(
            f"claude-code startup inventory is missing {gaps}"
        )
    state.inventory_verified = True


def _content_blocks(record: dict) -> list[dict]:
    message = record.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [block for block in content or () if isinstance(block, dict)]


def _preview(value: object) -> str:
    return json.dumps(value, default=str)[:_MAX_INPUT_PREVIEW]


def _on_assistant(run: RunContext, record: dict, state: StreamState) -> None:
    for block in _content_blocks(record):
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            run.emit("message", "claim", detail=str(block["text"]))
        elif kind == "tool_use":
            state.tool_calls += 1
            run.emit(
                "tool_call",
                "observation",
                name=str(block.get("name") or ""),
                detail=_preview(block.get("input")),
                data={"call_id": str(block.get("id") or "")},
            )


def _on_user(run: RunContext, record: dict, state: StreamState) -> None:
    for block in _content_blocks(record):
        if block.get("type") != "tool_result":
            continue
        run.emit(
            "tool_result",
            "observation",
            name=str(block.get("tool_use_id") or ""),
            detail=_preview(block.get("content")),
            data={"is_error": bool(block.get("is_error"))},
        )


def _usage(record: dict) -> UsageRecord:
    usage = as_mapping(record.get("usage"))
    cost = record.get("total_cost_usd")
    return UsageRecord(
        quality="measured",
        source="claude-code:result",
        input_tokens=int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cost_usd=float(cost) if isinstance(cost, int | float) else None,
        wall_s=float(record.get("duration_ms") or 0) / 1000.0,
    )


def _on_result(run: RunContext, record: dict, state: StreamState) -> None:
    state.terminal = True
    state.failed = bool(record.get("is_error")) or record.get("subtype") != "success"
    state.output = str(record.get("result") or "")
    state.usage = _usage(record)
    if state.failed:
        state.error = f"{record.get('subtype')}: {state.output[:2_000]}"
    data: dict[str, JsonValue] = {
        "input_tokens": state.usage.input_tokens,
        "output_tokens": state.usage.output_tokens,
        "cost_usd": state.usage.cost_usd,
    }
    run.emit("usage", "observation", name="result", data=data)


_HANDLERS: dict[str, Handler] = {
    "system": _on_system,
    "assistant": _on_assistant,
    "user": _on_user,
    "result": _on_result,
}


class ClaudeCodeHarness(CliHarness):
    """:class:`HarnessPort` over the Claude Code CLI."""

    binary = "claude"

    def describe(self) -> HarnessDescriptor:
        return DESCRIPTOR

    def handlers(self) -> Mapping[str, Handler]:
        return _HANDLERS

    def record_type(self, record: dict) -> str:
        return str(record.get("type") or "")

    def materialize(self, run: RunContext, config_dir: Path) -> None:
        from agent_utilities.claude_harness import build_settings_dict

        workspace = config_dir.parent
        servers = mcp_server_entries(run.spec.toolset.endpoints())
        (config_dir / "mcp.json").write_text(
            json.dumps({"mcpServers": servers}), encoding="utf-8"
        )
        claude_dir = workspace / ".claude"
        claude_dir.mkdir(exist_ok=True)
        (claude_dir / "settings.json").write_text(
            json.dumps(build_settings_dict()), encoding="utf-8"
        )
        write_skills(run, claude_dir / "skills")

    def invocation(
        self, run: RunContext, binary_path: str, config_dir: Path
    ) -> Invocation:
        spec = run.spec
        argv = [
            binary_path,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--setting-sources",
            "project",
            "--strict-mcp-config",
            "--mcp-config",
            str(config_dir / "mcp.json"),
            "--permission-mode",
            "dontAsk",
        ]
        if spec.toolset.allowed_tools is not None:
            argv += ["--allowedTools", *spec.toolset.allowed_tools]
        if spec.model:
            argv += ["--model", spec.model]
        if spec.budget.max_cost_usd is not None:
            argv += ["--max-budget-usd", f"{spec.budget.max_cost_usd:.4f}"]
        env = self.endpoint_tokens(run)
        if spec.account_mode == "api_key":
            env["ANTHROPIC_API_KEY"] = api_key_for(spec, self.credentials, HARNESS_NAME)
        return Invocation(argv=tuple(argv), env=env, stdin_text=spec.task)


__all__ = ["DESCRIPTOR", "HARNESS_NAME", "ClaudeCodeHarness"]
