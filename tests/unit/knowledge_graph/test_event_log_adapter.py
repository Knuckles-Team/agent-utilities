"""Deterministic object-centric event projection proofs.

CONCEPT:AU-KG.mining.governed-perspective-flattening — classical single-case
flattening (grouping events into one trace per object) is only reachable
through an explicit, versioned ``ProcessPerspective``; there is no bare
``object_type``-string entry point left, so undisclosed flattening is a
structural (TypeError/ValueError) impossibility, not a policy choice.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.event_log_adapter import (
    project_object_centric_events,
)
from agent_utilities.knowledge_graph.ingestion.semantic_event_model import (
    ProcessPerspective,
)


def _perspective(
    *object_types: str,
    perspective_id: str = "case:order-view",
    derivation_version: str = "v1",
) -> ProcessPerspective:
    return ProcessPerspective(
        perspective_id=perspective_id,
        object_types=object_types or ("Order",),
        derivation_version=derivation_version,
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
        perspective=_perspective("Order"),
    )

    assert projection.object_ids == ("o1", "o2")
    assert projection.engine_traces() == [
        ["create", "approve", "ship"],
        ["create"],
    ]
    assert projection.event_count == 4
    assert projection.source_count == 4
    assert len(projection.lineage_digest) == 64
    assert projection.public_metadata()["perspective_id"] == "case:order-view"
    assert projection.public_metadata()["derivation_version"] == "v1"


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
        project_object_centric_events([event], perspective=_perspective("Order"))


def test_projection_rejects_a_perspective_with_no_matching_objects() -> None:
    event = _event("e1", "create", "2026-01-01T00:00:00Z", "o1")
    with pytest.raises(ValueError, match="no event object references"):
        project_object_centric_events([event], perspective=_perspective("Invoice"))


def test_undisclosed_flattening_is_structurally_impossible() -> None:
    """There is no code path that flattens without a versioned perspective."""
    event = _event("e1", "create", "2026-01-01T00:00:00Z", "o1")

    # A bare string is not a ProcessPerspective — refused before any grouping.
    with pytest.raises(TypeError, match="ProcessPerspective"):
        project_object_centric_events([event], perspective="Order")  # type: ignore[arg-type]

    # Calling with the legacy keyword name is a TypeError (no such param).
    with pytest.raises(TypeError):
        project_object_centric_events([event], object_type="Order")  # type: ignore[call-arg]

    # A perspective naming more than one case notion is refused too — there
    # is no default "pick one" fallback that would silently choose for you.
    with pytest.raises(ValueError, match="exactly one object type"):
        project_object_centric_events(
            [event], perspective=_perspective("Order", "Invoice")
        )


def test_projection_digest_is_replay_stable() -> None:
    events = [
        _event("e1", "create", "2026-01-01T00:00:00Z", "o1"),
        _event("e2", "ship", "2026-01-02T00:00:00Z", "o1"),
    ]
    perspective = _perspective("Order")
    first = project_object_centric_events(events, perspective=perspective)
    second = project_object_centric_events(
        list(reversed(events)), perspective=perspective
    )
    assert first.lineage_digest == second.lineage_digest
    assert first.traces == second.traces
