"""Failure-localized, region-preserving repair (Atomic Task Graph paper idea #1,
arXiv:2607.01942 — ``reports/paper-analysis-2607.01942.md`` §4 Rank 1).

``localized_repair_region`` walks ``TRANSITION_TO`` FORWARD from a failed node
to compute exactly the invalidated descendant subgraph, leaving every other
(preserved) node out of it.

@pytest.mark.concept("AU-ORCH.execution.workflow-lifecycle-management")
"""

from __future__ import annotations

import pytest

from agent_utilities.workflows.localized_repair import localized_repair_region

pytestmark = pytest.mark.concept("AU-ORCH.execution.workflow-lifecycle-management")


class _FakeEngine:
    """Serves out_edges from a flat ``(src, rel_type, tgt)`` edge list."""

    def __init__(self, edges: list[tuple[str, str, str]] | None = None) -> None:
        self._edges = edges or []

    def out_edges(self, node_id: str, data: bool = False):
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if s == node_id]


def _diamond_with_sibling_branch() -> list[tuple[str, str, str]]:
    """A -> B -> D
    A -> C -> D
    X -> Y            (a wholly disconnected sibling branch — never touches A)
    """
    return [
        ("A", "TRANSITION_TO", "B"),
        ("A", "TRANSITION_TO", "C"),
        ("B", "TRANSITION_TO", "D"),
        ("C", "TRANSITION_TO", "D"),
        ("X", "TRANSITION_TO", "Y"),
    ]


def test_invalidated_region_is_the_forward_transitive_closure():
    engine = _FakeEngine(_diamond_with_sibling_branch())
    out = localized_repair_region("A", engine=engine)
    assert out["failed"] == "A"
    assert out["invalidated"] == ["A", "B", "C", "D"]


def test_failure_mid_diamond_only_invalidates_its_own_descendants():
    """A failure at B (not A) only invalidates B and D — C is a SIBLING branch of
    B (both fed by A) and must be preserved, not re-run."""
    engine = _FakeEngine(_diamond_with_sibling_branch())
    out = localized_repair_region("B", engine=engine)
    assert out["invalidated"] == ["B", "D"]
    assert "A" not in out["invalidated"]
    assert "C" not in out["invalidated"]


def test_preserved_region_is_the_complement_when_all_nodes_given():
    engine = _FakeEngine(_diamond_with_sibling_branch())
    all_nodes = ["A", "B", "C", "D", "X", "Y"]
    out = localized_repair_region("B", engine=engine, all_nodes=all_nodes)
    assert out["invalidated"] == ["B", "D"]
    # every validated sibling/upstream/unrelated node is explicitly preserved
    assert out["preserved"] == ["A", "C", "X", "Y"]


def test_no_downstream_edges_invalidates_only_the_failed_node():
    engine = _FakeEngine([("A", "TRANSITION_TO", "B")])
    out = localized_repair_region("B", engine=engine)  # B is a terminal node
    assert out["invalidated"] == ["B"]


def test_cycle_does_not_infinite_loop():
    engine = _FakeEngine([("A", "TRANSITION_TO", "B"), ("B", "TRANSITION_TO", "A")])
    out = localized_repair_region("A", engine=engine)
    assert out["invalidated"] == ["A", "B"]


def test_only_the_configured_edge_type_is_walked():
    """A differently-typed edge (not TRANSITION_TO) is not part of the DAG this
    walk cares about."""
    engine = _FakeEngine([("A", "some_other_edge", "B")])
    out = localized_repair_region("A", engine=engine)
    assert out["invalidated"] == ["A"]


# --- guards ---------------------------------------------------------------- #
def test_no_engine_is_a_noop_beyond_the_failed_node():
    out = localized_repair_region("A", engine=None)
    assert out["invalidated"] == ["A"]
    assert out["preserved"] == []
    assert out["degraded"] is False


def test_empty_failed_node_yields_empty_region():
    out = localized_repair_region("", engine=_FakeEngine())
    assert out["invalidated"] == []


# --- CONCEPT:AU-AHE.evaluation.return-none-on-failure: a failed edge read must
# never be silently indistinguishable from "confirmed no downstream edges" ---


class _FlakyEngine:
    """out_edges raises for one specific node, serves normally otherwise."""

    def __init__(self, edges: list[tuple[str, str, str]], *, raises_for: str) -> None:
        self._edges = edges
        self._raises_for = raises_for

    def out_edges(self, node_id: str, data: bool = False):
        if node_id == self._raises_for:
            raise RuntimeError("simulated transient graph read failure")
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if s == node_id]


def test_edge_read_failure_widens_invalidated_to_full_node_set_when_known():
    """Regression: a swallowed out_edges exception used to be indistinguishable
    from 'B has no downstream edges', silently under-invalidating the repair
    region — letting resume_localized preserve (never re-run) C/D even though
    B's TRUE downstream shape was never actually confirmed. With the full node
    set known, a failed read must widen invalidated rather than narrow it."""
    engine = _FlakyEngine(_diamond_with_sibling_branch(), raises_for="B")
    all_nodes = ["A", "B", "C", "D", "X", "Y"]
    out = localized_repair_region("B", engine=engine, all_nodes=all_nodes)
    assert out["degraded"] is True
    # Fails SAFE: invalidates everything rather than guessing B is a leaf.
    assert out["invalidated"] == ["A", "B", "C", "D", "X", "Y"]
    assert out["preserved"] == []


def test_edge_read_failure_without_all_nodes_flags_degraded_not_silently_ok():
    """Without a known universe there is nothing to widen to, but the result
    must still be flagged untrustworthy rather than silently reported clean."""
    engine = _FlakyEngine(_diamond_with_sibling_branch(), raises_for="B")
    out = localized_repair_region("B", engine=engine)
    assert out["degraded"] is True
    assert "B" in out["invalidated"]


def test_clean_read_never_reports_degraded():
    engine = _FakeEngine(_diamond_with_sibling_branch())
    out = localized_repair_region("B", engine=engine, all_nodes=["A", "B", "C", "D"])
    assert out["degraded"] is False
    assert out["invalidated"] == ["B", "D"]
