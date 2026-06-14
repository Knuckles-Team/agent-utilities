"""Claude Code session parser (CONCEPT:ECO-4.38).

Reads ``~/.claude/projects/<encoded-cwd>/<session>.jsonl``. Each file is one
session; lines are typed records (user/assistant/system/...). Token usage comes
from assistant ``message.usage``; tool calls from ``tool_use`` content blocks.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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


def _project_from_path(path: Path) -> str:
    """Decode the project dir name (``-home-apps-foo`` -> ``/home/apps/foo``)."""
    name = path.parent.name
    if name.startswith("-"):
        return "/" + name[1:].replace("-", "/")
    return name or "unknown"


def parse(path: Path, source) -> Iterator[ParsedSessionBundle]:
    session_id = path.stem
    project = _project_from_path(path)
    messages: list[UsageMessage] = []
    tool_calls: list[UsageToolCall] = []
    usage_events: list[UsageEvent] = []
    started_at = None
    ended_at = None
    first_message = ""
    agent_total_out = 0
    peak_context = 0
    ordinal = 0

    for rec in read_jsonl(path):
        rtype = rec.get("type")
        if rtype not in ("user", "assistant"):
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or rtype
        ts = rec.get("timestamp")
        if ts:
            started_at = started_at or ts
            ended_at = ts
        content = msg.get("content")
        text, thinking, has_tool = text_from_content(content)
        if role == "user" and not first_message and text:
            first_message = text[:200]
        model = msg.get("model") or ""
        _usage = msg.get("usage")
        usage = _usage if isinstance(_usage, dict) else {}
        out_tokens = int(usage.get("output_tokens", 0) or 0)
        in_tokens = int(usage.get("input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        context = in_tokens + cache_read + cache_creation
        peak_context = max(peak_context, context)
        agent_total_out += out_tokens

        messages.append(
            UsageMessage(
                session_id=session_id,
                ordinal=ordinal,
                role=role,
                content=text,
                thinking_text=thinking,
                timestamp=ts,
                model=model,
                context_tokens=context,
                output_tokens=out_tokens,
                has_tool_use=has_tool,
                content_length=len(text),
            )
        )
        for tu in tool_uses_from_content(content):
            name = tu["name"]
            tool_calls.append(
                UsageToolCall(
                    session_id=session_id,
                    message_ordinal=ordinal,
                    tool_name=name,
                    category=categorize_tool(name),
                    tool_use_id=tu.get("id"),
                    skill_name=(name if categorize_tool(name) == "skill" else None),
                    occurred_at=ts,
                )
            )
        if role == "assistant" and usage:
            usage_events.append(
                UsageEvent(
                    session_id=session_id,
                    message_ordinal=ordinal,
                    source="agent",
                    model=model,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                    occurred_at=ts,
                    dedup_key=f"{ordinal}:{msg.get('id', '')}",
                )
            )
        ordinal += 1

    if not messages:
        return

    user_count = sum(1 for m in messages if m.role == "user")
    session = UsageSession(
        id=session_id,
        project=project,
        agent=source.agent_type,
        first_message=first_message,
        started_at=started_at,
        ended_at=ended_at,
        message_count=len(messages),
        user_message_count=user_count,
        total_output_tokens=agent_total_out,
        peak_context_tokens=peak_context,
        file_path=str(path),
    )
    yield ParsedSessionBundle(
        session=session,
        messages=messages,
        tool_calls=tool_calls,
        usage_events=usage_events,
    )
