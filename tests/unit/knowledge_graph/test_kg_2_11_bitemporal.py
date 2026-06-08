"""CONCEPT:KG-2.11 — Bi-Temporal Memory Layers.

Covers the pure temporal core (stamp / as-of validity / event-time precedence / supersede)
and the additive procedural-layer fields on MemoryNode.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.bitemporal import (
    filter_as_of,
    is_valid_as_of,
    resolve_precedence,
    stamp_bitemporal,
    supersede,
)
from agent_utilities.models.knowledge_graph import MemoryNode


@pytest.mark.concept(id="KG-2.11")
def test_stamp_sets_all_four_time_concepts():
    props = stamp_bitemporal({}, now="2026-06-04T00:00:00+00:00")
    assert props["storage_time"] == "2026-06-04T00:00:00+00:00"
    # No explicit event_time → defaults to storage_time, and valid_from follows event_time.
    assert props["event_time"] == "2026-06-04T00:00:00+00:00"
    assert props["valid_from"] == "2026-06-04T00:00:00+00:00"
    assert props["valid_to"] is None  # open interval


@pytest.mark.concept(id="KG-2.11")
def test_stamp_preserves_explicit_event_time_distinct_from_storage():
    # The hallmark Temporal Truth case: event happened long before it was stored.
    props = stamp_bitemporal(
        {}, event_time="2023-03-04T00:00:00+00:00", now="2026-06-04T00:00:00+00:00"
    )
    assert props["event_time"] == "2023-03-04T00:00:00+00:00"
    assert props["storage_time"] == "2026-06-04T00:00:00+00:00"
    assert props["valid_from"] == "2023-03-04T00:00:00+00:00"


@pytest.mark.concept(id="KG-2.11")
def test_stamp_is_idempotent():
    props = stamp_bitemporal({}, event_time="2023-01-01T00:00:00+00:00")
    again = stamp_bitemporal(dict(props), now="2099-01-01T00:00:00+00:00")
    assert again["event_time"] == "2023-01-01T00:00:00+00:00"
    assert again["storage_time"] == props["storage_time"]


@pytest.mark.concept(id="KG-2.11")
def test_is_valid_as_of_within_and_outside_interval():
    props = {
        "valid_from": "2023-01-01T00:00:00+00:00",
        "valid_to": "2024-01-01T00:00:00+00:00",
    }
    assert is_valid_as_of(props, "2023-06-01T00:00:00+00:00") is True
    assert is_valid_as_of(props, "2022-06-01T00:00:00+00:00") is False  # before
    assert (
        is_valid_as_of(props, "2024-06-01T00:00:00+00:00") is False
    )  # after (>= valid_to)


@pytest.mark.concept(id="KG-2.11")
def test_open_interval_is_valid_forever():
    props = {"valid_from": "2023-01-01T00:00:00+00:00", "valid_to": None}
    assert is_valid_as_of(props, "2099-01-01T00:00:00+00:00") is True


@pytest.mark.concept(id="KG-2.11")
def test_filter_as_of_selects_temporally_correct_row():
    old = {
        "id": "a",
        "content": "lives in Boston",
        "valid_from": "2020-01-01T00:00:00+00:00",
        "valid_to": "2023-01-01T00:00:00+00:00",
    }
    new = {
        "id": "b",
        "content": "lives in Denver",
        "valid_from": "2023-01-01T00:00:00+00:00",
        "valid_to": None,
    }
    rows = [old, new]
    assert [r["id"] for r in filter_as_of(rows, "2021-06-01T00:00:00+00:00")] == ["a"]  # type: ignore[arg-type]
    assert [r["id"] for r in filter_as_of(rows, "2024-06-01T00:00:00+00:00")] == ["b"]  # type: ignore[arg-type]
    # No as_of → unfiltered.
    assert len(filter_as_of(rows, None)) == 2  # type: ignore[arg-type]


@pytest.mark.concept(id="KG-2.11")
def test_precedence_later_event_time_wins():
    a = {"id": "a", "event_time": "2023-01-01T00:00:00+00:00"}
    b = {"id": "b", "event_time": "2024-01-01T00:00:00+00:00"}
    winner, loser = resolve_precedence(a, b)
    assert winner["id"] == "b" and loser["id"] == "a"
    # Order independence.
    winner2, loser2 = resolve_precedence(b, a)
    assert winner2["id"] == "b" and loser2["id"] == "a"


@pytest.mark.concept(id="KG-2.11")
def test_supersede_closes_loser_interval_without_deleting():
    winner = {"id": "b", "event_time": "2024-01-01T00:00:00+00:00"}
    loser = {"id": "a", "event_time": "2023-01-01T00:00:00+00:00", "valid_to": None}
    supersede(winner, loser)
    assert loser["valid_to"] == "2024-01-01T00:00:00+00:00"
    # The superseded fact is still queryable for instants before the boundary.
    assert is_valid_as_of(loser, "2023-06-01T00:00:00+00:00") is True
    assert is_valid_as_of(loser, "2024-06-01T00:00:00+00:00") is False


@pytest.mark.concept(id="KG-2.11")
def test_memory_node_procedural_layer_fields():
    rule = MemoryNode(
        id="m1",
        name="formal-tone",
        memory_type="procedural",
        target_entity="global",
        content="Always use a formal tone.",
    )
    assert rule.memory_type == "procedural"
    assert rule.target_entity == "global"
    # Backward compatible default.
    assert MemoryNode(id="m2", name="x").memory_type == "semantic"
