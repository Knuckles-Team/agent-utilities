"""Bi-temporal SHACL invariants (CONCEPT:AU-KG.domains.ohlcv-gap-fill / KG-2.251).

Proves the temporal shapes actually FLAG violations (fails-then-passes), which is
the moat over a plain property graph: an unresolved contradiction / malformed
validity window is a detectable SHACL violation.

EH-431: validated through the engine's real ``shacl_validate_ad_hoc`` surface
(the same component-owned-specialist-shape path ``harness_gate.py`` uses in
production) rather than a local ``pyshacl`` call — AU never loads/validates
shapes itself (committed-SHACL-authority rule). Requires a real engine
(``engine_graph``); skips cleanly when none is available in this environment.
"""

from __future__ import annotations

from pathlib import Path

_SHAPES = (
    Path(__file__).resolve().parents[2]
    / "agent_utilities"
    / "knowledge_graph"
    / "shapes"
    / "temporal.shapes.ttl"
)
_SHAPES_TEXT = _SHAPES.read_text(encoding="utf-8")

_PREFIX = """
@prefix : <http://knuckles.team/kg#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _conforms(engine_graph, data_ttl: str) -> bool:
    report = engine_graph.shacl_validate_ad_hoc(_PREFIX + data_ttl, _SHAPES_TEXT)
    return bool(report.conforms)


def test_malformed_validity_window_is_flagged(engine_graph):
    bad = ":f a :TemporalFact ; :validFrom 300 ; :validUntil 100 ."
    assert _conforms(engine_graph, bad) is False  # validUntil precedes validFrom


def test_well_formed_validity_window_conforms(engine_graph):
    good = ":f a :TemporalFact ; :validFrom 100 ; :validUntil 300 ."
    assert _conforms(engine_graph, good) is True


def test_superseded_fact_without_closed_belief_is_flagged(engine_graph):
    # f1 was superseded by f2 but its belief window is still open (no :txTo).
    bad = """
    :f1 a :TemporalFact ; :validFrom 100 ; :validUntil 200 .
    :f2 a :TemporalFact ; :validFrom 200 ; :supersedes :f1 .
    """
    assert _conforms(engine_graph, bad) is False


def test_superseded_fact_with_closed_belief_conforms(engine_graph):
    # Same, but f1's belief window is properly closed → contradiction resolved.
    good = """
    :f1 a :TemporalFact ; :validFrom 100 ; :validUntil 200 ; :txTo 200 .
    :f2 a :TemporalFact ; :validFrom 200 ; :supersedes :f1 .
    """
    assert _conforms(engine_graph, good) is True
