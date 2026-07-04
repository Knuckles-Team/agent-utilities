"""Tests for relation-direction repair (CONCEPT:AU-KG.enrichment.direction-repair)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.extraction.direction_repair import (
    repair_direction,
    repair_directions,
)
from agent_utilities.knowledge_graph.extraction.extraction_schema import (
    ExtractionSchema,
    Relation,
)
from agent_utilities.knowledge_graph.extraction.fact_extractor import ExtractedFact


def _fact(subject: str, predicate: str, obj: str, conf: int = 80) -> ExtractedFact:
    return ExtractedFact(
        subject=subject, predicate=predicate, object=obj, confidence=conf
    )


_REL_BY_PRED = {
    "works_for": Relation("works_for", "", ("Person",), ("Organization",)),
    "knows": Relation("knows", "", ("Person",), ("Person",), symmetric=True),
}


def test_forward_ok_unchanged():
    f = _fact("Bob", "works_for", "Acme")
    g = {"subject_type": "person", "object_type": "organization"}
    f2, g2, status = repair_direction(f, g, _REL_BY_PRED)
    assert status == "ok"
    assert (f2.subject, f2.object) == ("Bob", "Acme")


def test_reversed_is_swapped():
    # extracted backwards: Acme works_for Bob, but domain=Person range=Organization
    f = _fact("Acme", "works_for", "Bob")
    g = {"subject_type": "organization", "object_type": "person"}
    f2, g2, status = repair_direction(f, g, _REL_BY_PRED)
    assert status == "swapped"
    assert (f2.subject, f2.object) == ("Bob", "Acme")
    # grounding types swapped with the edge
    assert g2["subject_type"] == "person"
    assert g2["object_type"] == "organization"


def test_symmetric_never_flips():
    f = _fact("Bob", "knows", "Alice")
    g = {"subject_type": "person", "object_type": "person"}
    f2, _g, status = repair_direction(f, g, _REL_BY_PRED)
    assert status == "symmetric"
    assert (f2.subject, f2.object) == ("Bob", "Alice")


def test_violation_flagged_not_dropped():
    # neither orientation satisfies Person→Organization
    f = _fact("Paris", "works_for", "London", conf=80)
    g = {"subject_type": "place", "object_type": "place"}
    f2, _g, status = repair_direction(f, g, _REL_BY_PRED)
    assert status == "violation"
    assert "domain_range_violation" in f2.tags
    assert f2.confidence == 40  # damped 0.5×, edge kept


def test_untyped_predicate_passes_through():
    f = _fact("Bob", "vibes_with", "Acme")
    g = {"subject_type": "person", "object_type": "organization"}
    _f, _g, status = repair_direction(f, g, _REL_BY_PRED)
    assert status == "untyped"


def test_untyped_endpoint_passes_through():
    f = _fact("Bob", "works_for", "Acme")
    g = {"subject_type": None, "object_type": "organization"}
    _f, _g, status = repair_direction(f, g, _REL_BY_PRED)
    assert status == "untyped"


def test_repair_directions_batch_and_tally():
    schema = ExtractionSchema(name="t", relations=tuple(_REL_BY_PRED.values()))
    grounded = [
        (
            _fact("Bob", "works_for", "Acme"),
            {"subject_type": "person", "object_type": "organization"},
        ),
        (
            _fact("Acme", "works_for", "Bob"),
            {"subject_type": "organization", "object_type": "person"},
        ),
    ]
    out, tally = repair_directions(grounded, schema)
    assert tally.get("ok") == 1
    assert tally.get("swapped") == 1
    # the swapped one is now Bob works_for Acme
    assert (out[1][0].subject, out[1][0].object) == ("Bob", "Acme")


def test_no_schema_returns_unchanged():
    grounded = [
        (
            _fact("Acme", "works_for", "Bob"),
            {"subject_type": "organization", "object_type": "person"},
        )
    ]
    out, tally = repair_directions(grounded, None)
    assert out == grounded
    assert tally == {}
    # not swapped (no schema)
    assert out[0][0].subject == "Acme"
