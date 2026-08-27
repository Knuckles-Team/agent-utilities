"""CA-28: the Loop-engine lakehouse-maintenance propose-only hook.

``state_tools.propose_lakehouse_maintenance_gap`` is what
``core.schedule_engine``'s ``lakehouse-maintenance`` schedule dispatch targets
(``debezium_lag_check``/``opensearch_reindex_staleness_check``/
``lineage_sweep`` — CA-21/24/15/25's real detection logic, stubbed today) will
call on a genuine finding, instead of inventing a second execution path: it
reuses the SAME canonical ``:Gap`` -> SpecProposal -> review lifecycle every
other discovery track files into (``research/gaps.py``'s ``submit_gap``,
exercised the same way ``tests/unit/mcp/test_state_tools_gap_lifecycle.py``
already covers for the operator-facing ``graph_loops`` MCP tool).

Deliberately propose-only: this module proves the hook has NO develop/apply
path of its own, matching ``graph_loops`` ``run``'s existing
``mine_discovery`` default-ON-but-propose-only contract.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.tools.state_tools import propose_lakehouse_maintenance_gap


class _GapStubEngine:
    """Minimal engine double covering exactly what research/gaps.py needs
    (mirrors ``tests/unit/mcp/test_state_tools_gap_lifecycle.py``'s
    ``_GapStubEngine`` — kept local/self-contained rather than shared so this
    file has no cross-test-module coupling)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.backend = object()

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **properties: Any
    ) -> None:
        self.edges.append((source, target, rel_type))

    def query_cypher(
        self, q: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return []


def test_propose_lakehouse_maintenance_gap_files_a_canonical_gap() -> None:
    eng = _GapStubEngine()

    gap = propose_lakehouse_maintenance_gap(
        eng,
        source="lakehouse-maintenance:opensearch_reindex_staleness_check",
        statement="OpenSearch index eg.homelab.Document is 3 CDC generations behind.",
        domain="lakehouse-maintenance",
        severity=0.6,
    )

    assert gap is not None
    assert gap["source"] == "lakehouse-maintenance:opensearch_reindex_staleness_check"
    assert gap["status"] == "open"
    node = eng.nodes[gap["id"]]
    assert node["type"] == "Gap"
    assert node["source"] == "lakehouse-maintenance:opensearch_reindex_staleness_check"


def test_propose_lakehouse_maintenance_gap_is_propose_only_no_apply_path() -> None:
    """The hook's only durable effects are the two nodes ``submit_gap`` itself
    always writes for ANY discovery track: the :Gap and its :WorkItem
    lease (``research/gaps.py``'s D1 "give the gap a WorkItem/lease" —
    the SAME lease every other gap gets, not a second execution path). No
    edge beyond an OPTIONAL concept-provenance link is written, and nothing
    that would apply/execute a lakehouse change lives on this path -- a real
    assertion failure here is NOT swallowed because it is checked on the
    stub's recorded state after the call returns, not from inside it."""
    eng = _GapStubEngine()
    gap = propose_lakehouse_maintenance_gap(
        eng,
        source="lakehouse-maintenance:debezium_lag_check",
        statement="Debezium consumer for source=erp lagging 900s over SLO.",
    )
    assert gap is not None
    # Exactly the gap node + its WorkItem lease (submit_gap's own D1 lease
    # call) -- both are the canonical gap-lifecycle's plumbing, not a second
    # execution path -- and zero edges (no concept_ids given, so no
    # DERIVED_FROM link).
    node_types = {n["type"] for n in eng.nodes.values()}
    assert node_types == {"Gap", "WorkItem"}
    assert gap["id"] in eng.nodes
    assert eng.edges == []


def test_propose_lakehouse_maintenance_gap_idempotent_on_repeated_identical_finding() -> (
    None
):
    """The same finding re-detected on the next tick must not file a second
    :Gap (nor a second WorkItem lease) -- ``signature`` defaults to a stable
    hash of source+statement, and ``submit_gap``'s own
    ``gap:<source>:<signature>`` id (plus its lease's matching id) makes the
    repeat an upsert onto the SAME two node ids, never a growing set."""
    eng = _GapStubEngine()
    statement = "OpenSearch index eg.homelab.Concept is 5 CDC generations behind."

    first = propose_lakehouse_maintenance_gap(
        eng, source="lakehouse-maintenance:opensearch_reindex_staleness_check",
        statement=statement,
    )
    node_ids_after_first = set(eng.nodes)
    second = propose_lakehouse_maintenance_gap(
        eng, source="lakehouse-maintenance:opensearch_reindex_staleness_check",
        statement=statement,
    )

    assert first is not None and second is not None
    assert first["id"] == second["id"]
    # The repeat upserts onto the SAME node ids -- no growth from the first call.
    assert set(eng.nodes) == node_ids_after_first


def test_propose_lakehouse_maintenance_gap_blank_statement_is_a_noop() -> None:
    eng = _GapStubEngine()
    assert (
        propose_lakehouse_maintenance_gap(
            eng, source="lakehouse-maintenance:lineage_sweep", statement="   "
        )
        is None
    )
    assert eng.nodes == {}
