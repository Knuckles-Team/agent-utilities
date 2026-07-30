"""Deterministic object-centric event projection proofs."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.event_log_adapter import (
    project_object_centric_events,
)


def _event(
    event_id: str,
    activity: str,
    occurred_at: str,
    object_id: str,
    *,
    sequence: str = "",
) -> dict:
    return {
        "event_id": event_id,
        "activity": activity,
        "occurred_at": occurred_at,
        "sequence_tiebreaker": sequence,
        "source_ref": f"source:{event_id}",
        "objects": [
            {
                "id": object_id,
                "type": "Order",
                "qualifier": "subject",
            }
        ],
    }


def test_object_perspective_groups_and_orders_without_llm() -> None:
    projection = project_object_centric_events(
        [
            _event("e3", "ship", "2026-01-02T00:00:00Z", "o1"),
            _event(
                "e2",
                "approve",
                "2026-01-01T00:00:00Z",
                "o1",
                sequence="2",
            ),
            _event(
                "e1",
                "create",
                "2026-01-01T00:00:00Z",
                "o1",
                sequence="1",
            ),
            _event("e4", "create", "2026-01-03T00:00:00Z", "o2"),
        ],
        object_type="Order",
    )

    assert projection.object_ids == ("o1", "o2")
    assert projection.engine_traces() == [
        ["create", "approve", "ship"],
        ["create"],
    ]
    assert projection.event_count == 4
    assert projection.source_count == 4
    assert len(projection.lineage_digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_id", ""),
        ("activity", ""),
        ("occurred_at", "not-a-time"),
        ("source_ref", ""),
    ],
)
def test_projection_rejects_incomplete_event(field: str, value: str) -> None:
    event = _event("e1", "create", "2026-01-01T00:00:00Z", "o1")
    event[field] = value
    with pytest.raises(ValueError):
        project_object_centric_events([event], object_type="Order")


def test_projection_rejects_implicit_or_missing_perspective() -> None:
    event = _event("e1", "create", "2026-01-01T00:00:00Z", "o1")
    with pytest.raises(ValueError, match="object_type"):
        project_object_centric_events([event], object_type="")
    with pytest.raises(ValueError, match="no event object references"):
        project_object_centric_events([event], object_type="Invoice")


def test_projection_digest_is_replay_stable() -> None:
    events = [
        _event("e1", "create", "2026-01-01T00:00:00Z", "o1"),
        _event("e2", "ship", "2026-01-02T00:00:00Z", "o1"),
    ]
    first = project_object_centric_events(events, object_type="Order")
    second = project_object_centric_events(list(reversed(events)), object_type="Order")
    assert first.lineage_digest == second.lineage_digest
    assert first.traces == second.traces
