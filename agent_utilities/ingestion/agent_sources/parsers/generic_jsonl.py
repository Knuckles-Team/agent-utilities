"""Generic JSONL session parser (CONCEPT:ECO-4.38).

A tolerant best-effort parser for the many JSONL-family agents whose records
carry a recognizable role + content (and optionally an Anthropic/OpenAI-style
``usage`` block). Used as the default for agents without a bespoke parser, so
all registered agents are at least detected and ingested. Agents with a
divergent encoding (SQLite, protobuf) ship their own parser module instead.
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

from .._jsonl import (
    categorize_tool,
    read_jsonl,
    text_from_content,
    tool_uses_from_content,
)

_ROLE_KEYS = ("role", "type", "sender", "author")
_CONTENT_KEYS = ("content", "text", "message", "body")


def _extract_role(rec: dict[str, Any]) -> str | None:
    msg = rec.get("message")
    if isinstance(msg, dict):
        r = msg.get("role")
        if r in ("user", "assistant"):
            return r
    for k in _ROLE_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.lower() in ("user", "assistant", "human", "ai"):
            return "user" if v.lower() in ("user", "human") else "assistant"
    return None


def _extract_content(rec: dict[str, Any]) -> Any:
    msg = rec.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"]
    for k in _CONTENT_KEYS:
        if k in rec:
            return rec[k]
    return ""


def _extract_usage(rec: dict[str, Any]) -> dict[str, Any]:
    msg = rec.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("usage"), dict):
        return msg["usage"]
    if isinstance(rec.get("usage"), dict):
        return rec["usage"]
    return {}


def parse(path: Path, source) -> Iterator[ParsedSessionBundle]:
    session_id = path.stem
    messages: list[UsageMessage] = []
    tool_calls: list[UsageToolCall] = []
    usage_events: list[UsageEvent] = []
    started_at = ended_at = None
    first_message = ""
    model = ""
    total_out = 0
    ordinal = 0

    for rec in read_jsonl(path):
        role = _extract_role(rec)
        if role is None:
            continue
        content = _extract_content(rec)
        text, thinking, has_tool = text_from_content(content)
        if not text.strip() and not has_tool:
            continue
        ts = rec.get("timestamp") or rec.get("time") or rec.get("ts")
        if ts:
            started_at = started_at or str(ts)
            ended_at = str(ts)
        msg = rec.get("message") if isinstance(rec.get("message"), dict) else rec
        model = msg.get("model") or model
        usage = _extract_usage(rec)
        out = int(usage.get("output_tokens", 0) or 0)
        inp = int(usage.get("input_tokens", 0) or 0)
        total_out += out
        if role == "user" and not first_message and text:
            first_message = text[:200]
        messages.append(
            UsageMessage(
                session_id=session_id, ordinal=ordinal, role=role, content=text,
                thinking_text=thinking, timestamp=(str(ts) if ts else None),
                model=model or "", output_tokens=out, has_tool_use=has_tool,
                content_length=len(text),
            )
        )
        for tu in tool_uses_from_content(content):
            name = tu["name"]
            tool_calls.append(
                UsageToolCall(
                    session_id=session_id, message_ordinal=ordinal, tool_name=name,
                    category=categorize_tool(name), tool_use_id=tu.get("id"),
                    occurred_at=(str(ts) if ts else None),
                )
            )
        if usage:
            usage_events.append(
                UsageEvent(
                    session_id=session_id, message_ordinal=ordinal, source="agent",
                    model=model or "", input_tokens=inp, output_tokens=out,
                    cache_read_input_tokens=int(
                        usage.get("cache_read_input_tokens", 0) or 0
                    ),
                    cache_creation_input_tokens=int(
                        usage.get("cache_creation_input_tokens", 0) or 0
                    ),
                    occurred_at=(str(ts) if ts else None),
                    dedup_key=f"{ordinal}",
                )
            )
        ordinal += 1

    if not messages:
        return
    user_count = sum(1 for m in messages if m.role == "user")
    yield ParsedSessionBundle(
        session=UsageSession(
            id=session_id, project=source.agent_type, agent=source.agent_type,
            first_message=first_message, started_at=started_at, ended_at=ended_at,
            message_count=len(messages), user_message_count=user_count,
            total_output_tokens=total_out, file_path=str(path),
        ),
        messages=messages, tool_calls=tool_calls, usage_events=usage_events,
    )
