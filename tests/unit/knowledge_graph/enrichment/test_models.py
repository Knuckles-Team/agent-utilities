"""EH-274: EdgeRung, rung_may_overwrite, and dedupe_edges_by_rung.

Proves the monotone-safety guarantee "a higher rung cannot overwrite a lower
rung's fact" against KNOWN-BAD inputs (per the build contract's proof
obligation: a gate/invariant must be shown to catch something that should
fail, not just pass on something that should pass).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.models import (
    EdgeRung,
    EnrichmentEdge,
    dedupe_edges_by_rung,
    rung_may_overwrite,
)


def test_edge_rung_is_ordered_cheapest_to_most_expensive():
    assert (
        EdgeRung.EXTRACTED
        < EdgeRung.INFERRED
        < EdgeRung.DERIVED
        < EdgeRung.MODELED
        < EdgeRung.EMBEDDED
        < EdgeRung.ASSERTED
    )


def test_unknown_sorts_above_every_real_rung():
    """UNKNOWN must fail CLOSED under a naive `<=` comparison, not open."""
    for rung in (
        EdgeRung.EXTRACTED,
        EdgeRung.INFERRED,
        EdgeRung.DERIVED,
        EdgeRung.MODELED,
        EdgeRung.EMBEDDED,
        EdgeRung.ASSERTED,
    ):
        assert EdgeRung.UNKNOWN > rung
        assert not (EdgeRung.UNKNOWN <= rung)


def test_rung_may_overwrite_refuses_a_less_certain_write():
    """KNOWN-BAD input: an ASSERTED (LLM) write must never replace an already
    -classified EXTRACTED (AST) fact."""
    assert rung_may_overwrite(EdgeRung.EXTRACTED, EdgeRung.ASSERTED) is False


def test_rung_may_overwrite_allows_a_more_certain_write():
    assert rung_may_overwrite(EdgeRung.ASSERTED, EdgeRung.EXTRACTED) is True


def test_rung_may_overwrite_allows_same_rung():
    assert rung_may_overwrite(EdgeRung.INFERRED, EdgeRung.INFERRED) is True


def test_rung_may_overwrite_fills_an_unclassified_gap():
    """Pre-EH-274 persisted edges (no recorded rung) accept a real
    classification -- known is always better than none."""
    assert rung_may_overwrite(None, EdgeRung.ASSERTED) is True
    assert rung_may_overwrite(EdgeRung.UNKNOWN, EdgeRung.ASSERTED) is True


def test_rung_may_overwrite_never_regresses_a_classified_edge_to_unknown():
    """KNOWN-BAD input: a write carrying UNKNOWN must never blank out an
    edge that already has a real classification."""
    assert rung_may_overwrite(EdgeRung.EXTRACTED, EdgeRung.UNKNOWN) is False


def test_dedupe_keeps_the_more_certain_edge_regardless_of_order():
    certain = EnrichmentEdge(
        source="a", target="b", rel_type="CALLS", rung=EdgeRung.EXTRACTED
    )
    uncertain = EnrichmentEdge(
        source="a", target="b", rel_type="CALLS", rung=EdgeRung.ASSERTED
    )

    kept_first = dedupe_edges_by_rung([certain, uncertain])
    assert len(kept_first) == 1 and kept_first[0].rung is EdgeRung.EXTRACTED

    kept_second = dedupe_edges_by_rung([uncertain, certain])
    assert len(kept_second) == 1 and kept_second[0].rung is EdgeRung.EXTRACTED


def test_dedupe_leaves_distinct_keys_untouched_and_in_order():
    a = EnrichmentEdge(source="a", target="b", rel_type="CALLS", rung=EdgeRung.INFERRED)
    b = EnrichmentEdge(source="c", target="d", rel_type="CALLS", rung=EdgeRung.INFERRED)
    assert dedupe_edges_by_rung([a, b]) == [a, b]


def test_edge_defaults_to_unknown_rung_and_no_confidence():
    e = EnrichmentEdge(source="a", target="b", rel_type="X")
    assert e.rung is EdgeRung.UNKNOWN
    assert e.confidence is None
