"""Deterministic object-centric event projection for native process mining.

CONCEPT:EG-KG.mining.process-mining — this module is the lossless-boundary
adapter between provenance-rich event records and the engine's current
case-trace mining kernel.  It performs no LLM calls and never represents the
resulting trace projection as the source event truth.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .semantic_event_model import (
    EventObjectParticipation,
    ObjectCentricGraphSlice,
    ProcessEvent,
)

EventObjectReference = EventObjectParticipation
EventRecord = ProcessEvent


@dataclass(frozen=True, slots=True)
class ProcessTraceProjection:
    """A declared object-type perspective over source events."""

    object_type: str
    traces: tuple[tuple[str, ...], ...]
    object_ids: tuple[str, ...]
    event_count: int
    source_count: int
    lineage_digest: str

    def engine_traces(self) -> list[list[str]]:
        """Return the current engine wire shape without exposing source payloads."""
        return [list(trace) for trace in self.traces]

    def public_metadata(self) -> dict[str, Any]:
        """Bounded projection evidence suitable for an MCP/REST response."""
        return {
            "mode": "object_type_projection",
            "object_type": self.object_type,
            "trace_count": len(self.traces),
            "event_count": self.event_count,
            "source_count": self.source_count,
            "lineage_digest": self.lineage_digest,
        }


def _required_text(value: object, field: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"event field {field!r} is required")
    return rendered


def _parse_timestamp(value: object) -> datetime:
    rendered = _required_text(value, "occurred_at")
    if rendered.endswith("Z"):
        rendered = f"{rendered[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise ValueError("event occurred_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("event occurred_at must include a timezone")
    return parsed.astimezone(UTC)


def _parse_object(value: object) -> EventObjectReference:
    if not isinstance(value, Mapping):
        raise ValueError("each event object reference must be an object")
    return EventObjectReference(
        object_id=_required_text(value.get("id"), "objects[].id"),
        object_type=_required_text(value.get("type"), "objects[].type"),
        qualifier=str(value.get("qualifier") or "").strip(),
    )


def _parse_event(value: object) -> EventRecord:
    if not isinstance(value, Mapping):
        raise ValueError("each event must be an object")
    raw_objects = value.get("objects")
    if not isinstance(raw_objects, Sequence) or isinstance(
        raw_objects, str | bytes | bytearray
    ):
        raise ValueError("event objects must be an array")
    objects = tuple(_parse_object(item) for item in raw_objects)
    if not objects:
        raise ValueError("each event must participate in at least one object")
    tiebreaker = value.get("sequence_tiebreaker", "")
    if not isinstance(tiebreaker, str | int):
        raise ValueError("event sequence_tiebreaker must be a string or integer")
    return EventRecord(
        event_id=_required_text(value.get("event_id"), "event_id"),
        activity=_required_text(value.get("activity"), "activity"),
        occurred_at=_parse_timestamp(value.get("occurred_at")),
        objects=objects,
        source_ref=_required_text(value.get("source_ref"), "source_ref"),
        sequence_tiebreaker=str(tiebreaker),
    )


def project_object_centric_events(
    events: object,
    *,
    object_type: str,
) -> ProcessTraceProjection:
    """Project events into one explicitly named object-type perspective.

    Each object of ``object_type`` becomes one trace. Events participating in
    multiple objects legitimately appear in multiple traces. Ordering is stable
    by event time, source tiebreaker, then immutable event id. The projection
    digest binds the complete event/object/source lineage without returning raw
    source records from the public tool surface.
    """
    perspective = _required_text(object_type, "object_type")
    if not isinstance(events, Sequence) or isinstance(events, str | bytes | bytearray):
        raise ValueError("events must be an array")
    parsed = tuple(_parse_event(value) for value in events)
    if not parsed:
        raise ValueError("events must not be empty")

    event_ids = [event.event_id for event in parsed]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event_id values must be unique within one projection")

    timelines: dict[str, list[EventRecord]] = {}
    lineage_rows: list[dict[str, Any]] = []
    for event in parsed:
        matching = [ref for ref in event.objects if ref.object_type == perspective]
        for ref in matching:
            timelines.setdefault(ref.object_id, []).append(event)
            lineage_rows.append(
                {
                    "event_id": event.event_id,
                    "object_id": ref.object_id,
                    "object_type": ref.object_type,
                    "qualifier": ref.qualifier,
                    "source_ref": event.source_ref,
                }
            )
    if not timelines:
        raise ValueError("no event object references match the requested object_type")

    object_ids = tuple(sorted(timelines))
    traces: list[tuple[str, ...]] = []
    for object_id in object_ids:
        ordered = sorted(
            timelines[object_id],
            key=lambda event: (
                event.occurred_at,
                event.sequence_tiebreaker,
                event.event_id,
            ),
        )
        traces.append(tuple(event.activity for event in ordered))

    canonical_lineage = json.dumps(
        sorted(
            lineage_rows,
            key=lambda row: (
                row["object_type"],
                row["object_id"],
                row["event_id"],
                row["qualifier"],
                row["source_ref"],
            ),
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ProcessTraceProjection(
        object_type=perspective,
        traces=tuple(traces),
        object_ids=object_ids,
        event_count=len(parsed),
        source_count=len({event.source_ref for event in parsed}),
        lineage_digest=hashlib.sha256(canonical_lineage).hexdigest(),
    )


def project_object_centric_slice(
    slice_: ObjectCentricGraphSlice,
    *,
    object_type: str,
) -> ProcessTraceProjection:
    """Project validated OCEL/tEKG source truth without reparsing its records."""
    return project_object_centric_events(
        [
            {
                "event_id": event.event_id,
                "activity": event.activity,
                "occurred_at": event.occurred_at.isoformat(),
                "objects": [
                    {
                        "id": relation.object_id,
                        "type": relation.object_type,
                        "qualifier": relation.qualifier,
                    }
                    for relation in event.objects
                ],
                "source_ref": event.source_ref,
                "sequence_tiebreaker": event.sequence_tiebreaker,
            }
            for event in slice_.events
        ],
        object_type=object_type,
    )
