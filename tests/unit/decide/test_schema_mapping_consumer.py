"""EH-033: unmapped source labels are schema-mapping decisions; proposals only."""

from __future__ import annotations

from agent_utilities.decide.consumers.schema_mapping import apply_decided_mappings
from tests.unit.decide.fakes import FakeTransport, abstained, acted

TARGETS = ["wm:Taxon", "wm:Country"]


def test_eg_maps_what_the_crosswalk_left_unmapped(eg: FakeTransport) -> None:
    eg.answer = acted("wm:Taxon")
    type_map = {"Nation": "wm:Country"}
    methods = {"Nation": ("exact", 1.0)}
    apply_decided_mappings(["Nation", "Species"], TARGETS, {}, type_map, methods)
    assert type_map == {"Nation": "wm:Country", "Species": "wm:Taxon"}
    assert methods["Species"] == ("eg-decision", 0.8)
    assert len(eg.requests) == 1, "the deterministic mapping is never re-asked"
    request = eg.requests[0]
    assert request["question"]["kind"] == "schema_mapping"
    assert request["question"]["safety"] == "policy"
    labels = {o["option_id"]: o["texts"] for o in request["candidates"]["options"]}
    assert labels["wm:Taxon"] == [{"key": "label", "text": "Taxon"}]


def test_an_abstention_falls_back_to_the_suggestion_or_leaves_it_unmapped(
    eg: FakeTransport,
) -> None:
    eg.answer = abstained()
    type_map: dict[str, str] = {}
    methods: dict[str, tuple[str, float]] = {}
    apply_decided_mappings(
        ["Species", "Blob"], TARGETS, {"Species": "wm:Taxon"}, type_map, methods
    )
    assert type_map == {"Species": "wm:Taxon"}
    assert methods["Species"] == ("semantic-proposal", 0.6)
