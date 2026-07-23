"""Canonical execution-trace and outcome ontology.

This module is the single authority shared by runtime writers, miners,
placement analysis, context assembly, and evaluators.  It deliberately keeps
the durable schema small and privacy-gated:

``RunTrace -[:USED_TOOL]-> ToolCall``
``RunTrace -[:PRODUCED_OUTCOME]-> OutcomeEvaluation``

Every node carries a numeric ``event_sequence``.  Incremental consumers persist
the greatest sequence they have completed and resume with ``> after_sequence``;
timestamps and opaque strings are never used as ordering cursors.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

TRACE_SCHEMA_VERSION = 1
TRACE_NODE_LABEL = "RunTrace"
TOOL_CALL_NODE_LABEL = "ToolCall"
OUTCOME_NODE_LABEL = "OutcomeEvaluation"
TRACE_CURSOR_NODE_LABEL = "TraceConsumerCursor"
TRACE_USED_TOOL_EDGE = "USED_TOOL"
TRACE_PRODUCED_OUTCOME_EDGE = "PRODUCED_OUTCOME"

_sequence_lock = threading.Lock()
_cursor_lock = threading.Lock()
_last_sequence = 0


def next_event_sequence() -> int:
    """Return a process-monotonic, wall-clock-anchored numeric event sequence."""

    global _last_sequence
    with _sequence_lock:
        _last_sequence = max(time.time_ns(), _last_sequence + 1)
        return _last_sequence


@dataclass(frozen=True, order=True)
class TraceCursor:
    """Durable incremental-consumer cursor (numeric; never lexical)."""

    event_sequence: int = 0

    @classmethod
    def from_rows(cls, rows: list[Mapping[str, Any]]) -> TraceCursor:
        greatest = 0
        for row in rows:
            try:
                greatest = max(greatest, int(row.get("event_sequence") or 0))
            except (TypeError, ValueError):
                continue
        return cls(greatest)


def _cursor_node_id(consumer: str) -> str:
    """Return an opaque, stable id for one incremental trace consumer."""

    ref = persistence_reference(
        "trace_cursor", consumer, namespace="execution-trace-consumer"
    )
    return f"trace_cursor:{ref}"


def _cursor_checkpoint_id(consumer: str, event_sequence: int) -> str:
    """Immutable checkpoint id; append-only rows prevent cross-process regression."""

    return f"{_cursor_node_id(consumer)}:{max(0, int(event_sequence))}"


def load_trace_cursor(engine: Any, consumer: str) -> TraceCursor:
    """Load a consumer's durable numeric cursor from the graph.

    Cursor state deliberately lives beside the trace ontology, not in a local
    file. Authority failures are explicit: treating an unavailable read as
    sequence zero can duplicate non-idempotent downstream work.
    """

    if engine is None:
        raise RuntimeError("trace cursor requires graph authority")
    if not str(consumer or "").strip():
        raise ValueError("trace cursor consumer is required")
    consumer_ref = persistence_reference(
        "trace_consumer", consumer, namespace="execution-trace"
    )
    try:
        rows = engine.query_cypher(
            f"MATCH (c:{TRACE_CURSOR_NODE_LABEL} {{consumer_ref: $consumer_ref}}) "
            "RETURN c.event_sequence AS event_sequence "
            "ORDER BY c.event_sequence DESC LIMIT 1",
            {"consumer_ref": consumer_ref},
        )
    except Exception as exc:
        raise RuntimeError(
            f"trace cursor authority read failed (error_type={type(exc).__name__})"
        ) from None
    return TraceCursor.from_rows(
        [row for row in (rows or []) if isinstance(row, Mapping)]
    )


def save_trace_cursor(
    engine: Any, consumer: str, cursor: TraceCursor | int
) -> TraceCursor:
    """Monotonically persist the greatest fully-completed trace sequence.

    The caller decides when a pass is complete; miners use this only after all
    governed downstream work succeeds.  A failed write returns the prior
    cursor. Authority write failures are explicit to the caller.
    """

    with _cursor_lock:
        current = load_trace_cursor(engine, consumer)
        try:
            requested = (
                cursor if isinstance(cursor, TraceCursor) else TraceCursor(int(cursor))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("trace cursor must be an integer sequence") from exc
        completed = max(current, requested)
        if completed == current:
            return current
        consumer_ref = persistence_reference(
            "trace_consumer", consumer, namespace="execution-trace"
        )
        properties = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "consumer_ref": consumer_ref,
            "event_sequence": completed.event_sequence,
            "event_cursor": completed.event_sequence,
        }
        try:
            # The checkpoint is immutable and sequence-addressed, so concurrent
            # writers cannot regress the greatest persisted sequence.
            engine.add_node(
                _cursor_checkpoint_id(consumer, completed.event_sequence),
                TRACE_CURSOR_NODE_LABEL,
                properties={
                    **properties,
                    "cursor_kind": "checkpoint",
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"trace cursor authority write failed (error_type={type(exc).__name__})"
            ) from None
        return completed


def content_digest(value: Any) -> str:
    """Stable keyed reference for content that never crosses persistence."""

    return persistence_reference(
        "trace_content", value, namespace="canonical-execution-trace"
    )


def trace_id(run_id: str) -> str:
    """Canonical trace node id; run identifiers are always opaque refs."""

    value = str(run_id or "")
    if value.startswith("trace:"):
        value = value.removeprefix("trace:")
    return f"trace:{persistence_reference('run', value, namespace='trace')}"


def outcome_id(run_id: str) -> str:
    return "outcome:" + trace_id(run_id).removeprefix("trace:")


def _privacy_safe(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    clean, report = PersistencePrivacyGuard().sanitize(dict(value))
    assert isinstance(clean, dict)
    return clean, {
        "privacy_schema": "persistence-privacy-v1",
        "privacy_redactions": report.redactions,
        "privacy_types": list(report.detected_types),
    }


def trace_properties(
    *,
    run_id: str,
    agent_name: str,
    task: str,
    status: str,
    timestamp: str,
    event_sequence: int | None = None,
    error: str | None = None,
    duration_ms: float | None = None,
    result_preview: str | None = None,
    skill_used: str = "",
    bound_server: str = "",
) -> dict[str, Any]:
    """Build the canonical, persistence-sanitized ``RunTrace`` properties."""

    seq = int(event_sequence or next_event_sequence())
    _clean, privacy = _privacy_safe(
        {
            "task": str(task or "")[:2000],
            "error": str(error or "")[:1000],
            "result_preview": str(result_preview or "")[:2000],
        }
    )
    props: dict[str, Any] = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": trace_id(run_id),
        "attribution_ref": persistence_reference(
            "agent", agent_name, namespace="execution-trace"
        ),
        "task": "",
        "task_digest": content_digest(task),
        "task_character_count": len(str(task or "")),
        "status": str(status or "unknown").lower(),
        "timestamp": timestamp,
        "event_sequence": seq,
        "event_cursor": seq,
        **privacy,
    }
    if error:
        props["error"] = ""
        props["error_digest"] = content_digest(error)
        props["error_character_count"] = len(str(error))
    if duration_ms is not None:
        props["duration_ms"] = max(0.0, round(float(duration_ms), 1))
    if result_preview:
        props["result_preview"] = ""
        props["result_digest"] = content_digest(result_preview)
        props["result_character_count"] = len(str(result_preview))
    if skill_used:
        props["skill_ref"] = persistence_reference(
            "skill", skill_used, namespace="execution-trace"
        )
    if bound_server:
        props["server_ref"] = persistence_reference(
            "server", bound_server, namespace="execution-trace"
        )
    return props


def tool_call_properties(
    *,
    run_id: str,
    tool_name: str,
    args: Any,
    result: Any,
    error: Any,
    status: str,
    sequence: int,
    timestamp: str,
    event_sequence: int | None = None,
) -> dict[str, Any]:
    """Build canonical ToolCall properties without raw identity fields."""

    seq = int(event_sequence or next_event_sequence())
    _clean, privacy = _privacy_safe(
        {
            "args": str(args or "")[:4000],
            "result": str(result or "")[:4000],
            "error": str(error or "")[:1000],
        }
    )
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": trace_id(run_id),
        "tool_name": str(tool_name or ""),
        "args": "",
        "args_digest": content_digest(args),
        "args_character_count": len(str(args or "")),
        "result": "",
        "result_digest": content_digest(result),
        "result_character_count": len(str(result or "")),
        "error": "",
        "error_digest": content_digest(error),
        "error_character_count": len(str(error or "")),
        "status": str(status or "unknown").lower(),
        "sequence": max(0, int(sequence)),
        "timestamp": timestamp,
        "event_sequence": seq,
        "event_cursor": seq,
        **privacy,
    }


def outcome_properties(
    *,
    run_id: str,
    status: str,
    timestamp: str,
    event_sequence: int,
    feedback: str = "",
    reward: float | None = None,
) -> dict[str, Any]:
    """Build the one normalized outcome attached to a ``RunTrace``."""

    normalized = str(status or "unknown").lower()
    succeeded = normalized in {"completed", "success", "succeeded", "ok"}
    if reward is None:
        reward = 1.0 if succeeded else (0.25 if normalized == "degraded" else 0.0)
    _clean, privacy = _privacy_safe({"feedback_text": str(feedback or "")[:1000]})
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": trace_id(run_id),
        "status": normalized,
        "reward": max(0.0, min(1.0, float(reward))),
        "success": succeeded,
        "success_criteria_met": ["execution_completed"] if succeeded else [],
        "feedback_text": "",
        "feedback_digest": content_digest(feedback),
        "feedback_character_count": len(str(feedback or "")),
        "timestamp": timestamp,
        "event_sequence": int(event_sequence),
        "event_cursor": int(event_sequence),
        **privacy,
    }


def trace_candidate_quality(node: Mapping[str, Any]) -> float | None:
    """Return canonical outcome quality when a context candidate is trace-derived."""

    if int(node.get("trace_schema_version") or 0) != TRACE_SCHEMA_VERSION:
        return None
    raw = node.get("reward")
    if raw is None:
        status = str(node.get("status") or "").lower()
        return 1.0 if status in {"completed", "success", "succeeded", "ok"} else 0.25
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return None


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "TRACE_NODE_LABEL",
    "TOOL_CALL_NODE_LABEL",
    "OUTCOME_NODE_LABEL",
    "TRACE_CURSOR_NODE_LABEL",
    "TRACE_USED_TOOL_EDGE",
    "TRACE_PRODUCED_OUTCOME_EDGE",
    "TraceCursor",
    "load_trace_cursor",
    "save_trace_cursor",
    "next_event_sequence",
    "trace_id",
    "outcome_id",
    "trace_properties",
    "tool_call_properties",
    "outcome_properties",
    "trace_candidate_quality",
]
