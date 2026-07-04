"""Codex CLI session parser (CONCEPT:AU-ECO.connector.agent-source-ingestion).

Reads ``~/.codex/sessions/**/*.jsonl``. Lines are ``{type, payload}`` records:
``session_meta`` (id/cwd), ``turn_context`` (model), ``response_item``
(message role/content or function_call), ``event_msg`` (token counts). Tolerant
of missing fields.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_utilities.usage.models import (
    ParsedSessionBundle,
    UsageEvent,
    UsageMessage,
    UsageSession,
    UsageToolCall,
)

from .._jsonl import categorize_tool, read_jsonl


def _content_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return ""


def parse(path: Path, source) -> Iterator[ParsedSessionBundle]:
    session_id = path.stem
    project = "unknown"
    model = ""
    messages: list[UsageMessage] = []
    tool_calls: list[UsageToolCall] = []
    usage_events: list[UsageEvent] = []
    started_at = ended_at = None
    first_message = ""
    total_out = 0
    ordinal = 0

    for rec in read_jsonl(path):
        rtype = rec.get("type")
        _payload = rec.get("payload")
        payload = _payload if isinstance(_payload, dict) else rec
        ts = rec.get("timestamp") or payload.get("timestamp")
        if ts:
            started_at = started_at or ts
            ended_at = ts
        if rtype == "session_meta":
            session_id = payload.get("id") or session_id
            cwd = payload.get("cwd")
            if cwd:
                project = cwd
            continue
        if rtype == "turn_context":
            model = payload.get("model") or model
            continue
        if rtype == "response_item":
            ptype = payload.get("type")
            if ptype == "function_call":
                name = payload.get("name", "")
                tool_calls.append(
                    UsageToolCall(
                        session_id=session_id,
                        message_ordinal=ordinal,
                        tool_name=name,
                        category=categorize_tool(name),
                        occurred_at=ts,
                    )
                )
                continue
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _content_text(payload)
            if not text.strip():
                continue
            if role == "user" and not first_message:
                first_message = text[:200]
            messages.append(
                UsageMessage(
                    session_id=session_id,
                    ordinal=ordinal,
                    role=role,
                    content=text,
                    timestamp=ts,
                    model=model,
                    content_length=len(text),
                )
            )
            ordinal += 1
            continue
        if rtype == "event_msg":
            info = payload.get("info") or payload
            tc = info.get("token_count") or info.get("total_token_usage") or {}
            if isinstance(tc, dict) and tc:
                inp = int(tc.get("input_tokens", 0) or 0)
                out = int(tc.get("output_tokens", 0) or 0)
                total_out += out
                usage_events.append(
                    UsageEvent(
                        session_id=session_id,
                        source="agent",
                        model=model,
                        input_tokens=inp,
                        output_tokens=out,
                        cache_read_input_tokens=int(
                            tc.get("cached_input_tokens", 0) or 0
                        ),
                        reasoning_tokens=int(tc.get("reasoning_tokens", 0) or 0),
                        occurred_at=ts,
                        dedup_key=f"evt:{ordinal}",
                    )
                )

    if not messages:
        return
    user_count = sum(1 for m in messages if m.role == "user")
    yield ParsedSessionBundle(
        session=UsageSession(
            id=session_id,
            project=project,
            agent=source.agent_type,
            first_message=first_message,
            started_at=started_at,
            ended_at=ended_at,
            message_count=len(messages),
            user_message_count=user_count,
            total_output_tokens=total_out,
            file_path=str(path),
        ),
        messages=messages,
        tool_calls=tool_calls,
        usage_events=usage_events,
    )
