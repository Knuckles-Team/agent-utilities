"""Bi-temporal SHACL invariants (CONCEPT:AU-KG.domains.ohlcv-gap-fill / KG-2.251).

Proves the temporal shapes actually FLAG violations (fails-then-passes), which is
the moat over a plain property graph: an unresolved contradiction / malformed
validity window is a detectable SHACL violation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pyshacl = pytest.importorskip("pyshacl")
rdflib = pytest.importorskip("rdflib")

_SHAPES = (
    Path(__file__).resolve().parents[2]
    / "agent_utilities"
    / "knowledge_graph"
    / "shapes"
    / "temporal.shapes.ttl"
)

_PREFIX = """
@prefix : <http://knuckles.team/kg#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _conforms(data_ttl: str) -> bool:
    data = rdflib.Graph().parse(data=_PREFIX + data_ttl, format="turtle")
    shapes = rdflib.Graph().parse(str(_SHAPES), format="turtle")
    conforms, _graph, _text = pyshacl.validate(
        data, shacl_graph=shapes, inference="none", advanced=True
    )
    return conforms


def test_malformed_validity_window_is_flagged():
    bad = ":f a :TemporalFact ; :validFrom 300 ; :validUntil 100 ."
    assert _conforms(bad) is False  # validUntil precedes validFrom


def test_well_formed_validity_window_conforms():
    good = ":f a :TemporalFact ; :validFrom 100 ; :validUntil 300 ."
    assert _conforms(good) is True


def test_superseded_fact_without_closed_belief_is_flagged():
    # f1 was superseded by f2 but its belief window is still open (no :txTo).
    bad = """
    :f1 a :TemporalFact ; :validFrom 100 ; :validUntil 200 .
    :f2 a :TemporalFact ; :validFrom 200 ; :supersedes :f1 .
    """
    assert _conforms(bad) is False


def test_superseded_fact_with_closed_belief_conforms():
    # Same, but f1's belief window is properly closed → contradiction resolved.
    good = """
    :f1 a :TemporalFact ; :validFrom 100 ; :validUntil 200 ; :txTo 200 .
    :f2 a :TemporalFact ; :validFrom 200 ; :supersedes :f1 .
    """
    assert _conforms(good) is True
