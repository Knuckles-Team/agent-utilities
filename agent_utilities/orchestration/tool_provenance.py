"""Shared per-tool-call provenance extraction.

CONCEPT:AU-KG.temporal.message-history-read — reads the ``(tool_name, args, result/error)`` of every
tool call from a pydantic-ai run result's message history, so BOTH the direct
single-server loop (:mod:`agent_utilities.orchestration.agent_runner`) and the
multi-agent graph executor (:mod:`agent_utilities.graph.executor`) surface the SAME
``:ToolCall`` provenance to ``run_agent``. Kept in this dependency-free leaf module so
both packages can import it without a circular dependency. Best-effort and
version-tolerant (matches on part class name / ``part_kind``) so a pydantic-ai bump
can never break the run path.
"""

from __future__ import annotations

from typing import Any

_TOOL_ARG_SECRET_KEYS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "private_key",
)


def sanitize_tool_args(args: Any) -> str:
    """Render tool-call args as a compact, secret-redacted JSON string.

    The args are persisted for visibility ("what did the local LLM call, with what"),
    so redact obvious secret-shaped keys and bound the size.
    """
    try:
        import json as _json

        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                return args[:2000]
        if isinstance(args, dict):
            red = {
                k: (
                    "***"
                    if any(s in str(k).lower() for s in _TOOL_ARG_SECRET_KEYS)
                    else v
                )
                for k, v in args.items()
            }
            return _json.dumps(red, default=str)[:2000]
        return _json.dumps(args, default=str)[:2000]
    except Exception:  # noqa: BLE001
        return str(args)[:2000]


def extract_tool_calls(run_result: Any) -> list[dict[str, Any]]:
    """Pull the (tool_name, args, result/error) of every tool call from a run.

    Reads the pydantic-ai message history (``all_messages()``): a ``ToolCallPart``
    opens a call, its paired ``ToolReturnPart`` (matched by ``tool_call_id``) carries
    the result, and a ``RetryPromptPart`` carries a tool error. Returns one ordered
    record per call.
    """
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    try:
        messages = run_result.all_messages()
    except Exception:  # noqa: BLE001 — not every result exposes a history
        return []
    # Only a real materialized history is iterable here; a mock/coroutine is not.
    if not isinstance(messages, list | tuple):
        return []
    for msg in messages or []:
        for part in getattr(msg, "parts", None) or []:
            kind = str(getattr(part, "part_kind", "") or part.__class__.__name__)
            lk = kind.lower()
            if "toolcall" in lk or lk == "tool-call":
                tcid = str(getattr(part, "tool_call_id", "") or f"tc{len(order)}")
                rec = {
                    "tool_call_id": tcid,
                    "tool_name": str(getattr(part, "tool_name", "") or ""),
                    "args": sanitize_tool_args(getattr(part, "args", None)),
                    "result": "",
                    "error": "",
                }
                calls[tcid] = rec
                order.append(tcid)
            elif "toolreturn" in lk or lk == "tool-return":
                tcid = str(getattr(part, "tool_call_id", "") or "")
                if tcid in calls:
                    calls[tcid]["result"] = str(getattr(part, "content", ""))[:2000]
            elif "retryprompt" in lk or lk == "retry-prompt":
                tcid = str(getattr(part, "tool_call_id", "") or "")
                if tcid in calls:
                    calls[tcid]["error"] = str(getattr(part, "content", ""))[:500]
    return [calls[t] for t in order]
