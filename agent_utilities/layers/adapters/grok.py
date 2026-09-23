"""Grok Build CLI adapter: headless ``grok --output-format streaming-json``.

Invocation follows docs.x.ai "Headless & scripting" (checked 2026-09-22):
``--prompt-file`` (the task never appears on the process command line),
``--cwd`` (the leased workspace), ``--no-auto-update``, ``--always-approve``
(nothing can answer a prompt in a headless run) and
``--output-format streaming-json``. The stream is Agent Client Protocol
shaped: ``session/update`` notifications (``agent_message_chunk``,
``tool_call``, ``tool_call_update`` ...) followed by the ``session/prompt``
response carrying ``stopReason``.

MCP endpoints are delivered as a project ``.mcp.json`` (Grok reads the Claude
Code configuration format). Grok reports no token usage in this stream, emits
no startup inventory this adapter can verify, and has no documented tool
allowlist, so the descriptor declares ``usage_quality="unavailable"`` and
negotiation refuses required tools, skills, allowlists and strict metered
budgets rather than launching a partial toolset.
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
    mcp_server_entries,
)
from agent_utilities.layers.contracts import HarnessDescriptor, VendorTerms
from agent_utilities.layers.session import RunContext

HARNESS_NAME = "grok"

DESCRIPTOR = HarnessDescriptor(
    **CLI_DESCRIPTOR_DEFAULTS,
    name=HARNESS_NAME,
    version="grok-cli/streaming-json",
    fidelity="tool-calls",
    capabilities=frozenset({"code_edit", "shell", "mcp_client", "cancellation"}),
    usage_quality="unavailable",
    enforceable_budgets=frozenset({"wall_time"}),
    tool_proof="none",
    skill_proof="none",
    max_skills=0,
    vendor_terms=VendorTerms(
        subscription_automation_allowed=True,
        note="headless -p/--single use is vendor-documented for scripts and CI",
    ),
)

#: ACP ``stopReason`` values that mean the turn finished normally.
_SUCCESS_STOPS = frozenset({"end_turn"})


def _update(record: dict) -> dict:
    params = record.get("params")
    update = params.get("update") if isinstance(params, dict) else None
    return update if isinstance(update, dict) else {}


def _message_chunk(run: RunContext, update: dict, state: StreamState) -> None:
    content = update.get("content")
    text = content.get("text") if isinstance(content, dict) else None
    if text:
        state.text_chunks.append(str(text))


def _tool_call(run: RunContext, update: dict, state: StreamState) -> None:
    state.tool_calls += 1
    run.emit(
        "tool_call",
        "observation",
        name=str(update.get("title") or update.get("kind") or ""),
        data={"call_id": str(update.get("toolCallId") or "")},
    )


def _tool_call_update(run: RunContext, update: dict, state: StreamState) -> None:
    status = str(update.get("status") or "")
    if status not in {"completed", "failed"}:
        return
    run.emit(
        "tool_result",
        "observation",
        name=str(update.get("toolCallId") or ""),
        detail=json.dumps(update.get("content"), default=str)[:2_000],
        data={"status": status},
    )


def _plan(run: RunContext, update: dict, state: StreamState) -> None:
    run.emit("step", "claim", name="plan", detail=json.dumps(update)[:2_000])


_UPDATES = {
    "agent_message_chunk": _message_chunk,
    "tool_call": _tool_call,
    "tool_call_update": _tool_call_update,
    "plan": _plan,
}


def _on_session_update(run: RunContext, record: dict, state: StreamState) -> None:
    params = record.get("params")
    if isinstance(params, dict) and params.get("sessionId"):
        run.provider_session = str(params["sessionId"])
    update = _update(record)
    handler = _UPDATES.get(str(update.get("sessionUpdate") or ""))
    if handler is not None:
        handler(run, update, state)


def _on_prompt_result(run: RunContext, record: dict, state: StreamState) -> None:
    result = record.get("result")
    if not isinstance(result, dict) or "stopReason" not in result:
        return
    stop = str(result.get("stopReason") or "")
    state.terminal = True
    state.failed = stop not in _SUCCESS_STOPS
    state.output = "".join(state.text_chunks)
    if state.output:
        run.emit("message", "claim", detail=state.output)
    if state.failed:
        state.error = f"stopReason={stop or 'missing'}"


def _on_rpc_error(run: RunContext, record: dict, state: StreamState) -> None:
    error = record.get("error")
    state.terminal = True
    state.failed = True
    state.error = json.dumps(error, default=str)[:2_000]
    run.emit("error", "observation", name="rpc-error", detail=state.error)


_HANDLERS: dict[str, Handler] = {
    "session/update": _on_session_update,
    "result": _on_prompt_result,
    "error": _on_rpc_error,
}


class GrokHarness(CliHarness):
    """:class:`HarnessPort` over the Grok Build CLI."""

    binary = "grok"
    descriptor = DESCRIPTOR
    record_handlers = _HANDLERS

    def record_type(self, record: dict) -> str:
        if "method" in record:
            return str(record["method"])
        return "error" if "error" in record else "result"

    def materialize(self, run: RunContext, config_dir: Path) -> None:
        servers = mcp_server_entries(run.spec.toolset.endpoints())
        (config_dir.parent / ".mcp.json").write_text(
            json.dumps({"mcpServers": servers}), encoding="utf-8"
        )
        (config_dir / "prompt.txt").write_text(run.spec.task, encoding="utf-8")

    def invocation(
        self, run: RunContext, binary_path: str, config_dir: Path
    ) -> Invocation:
        spec = run.spec
        argv = [
            binary_path,
            "--prompt-file",
            str(config_dir / "prompt.txt"),
            "--cwd",
            str(config_dir.parent),
            "--output-format",
            "streaming-json",
            "--no-auto-update",
            "--always-approve",
        ]
        if spec.model:
            argv += ["--model", spec.model]
        env = self.launch_env(run, "XAI_API_KEY")
        return Invocation(argv=tuple(argv), env=env, stdin_text="")


__all__ = ["DESCRIPTOR", "HARNESS_NAME", "GrokHarness"]
