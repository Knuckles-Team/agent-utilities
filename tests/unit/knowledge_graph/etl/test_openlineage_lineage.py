"""Unit tests for ``lineage.record_openlineage_run_event`` (CA-25, DEC-CA-05).

(CONCEPT:AU-KG.ingest.openlineage-consumer)

Covers CA-25's acceptance gates: normal Activity/Entity/used/wasGeneratedBy
creation, quarantine creates nothing, redelivery of the identical RunEvent
does not duplicate the Activity node (idempotency, gate 3), and RunTrace
correlation only links when the run genuinely correlates to an existing
tool-originated trace (gate 4).
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.etl.lineage import record_openlineage_run_event
from agent_utilities.observability.trace_ontology import trace_id


class _FakeEngine:
    """Mirrors tests/unit/knowledge_graph/test_etl_lineage.py's _FakeEngine —
    no query_cypher, so correlate_lineage_run_trace's AttributeError is
    caught and treated as "not tool-originated" (best-effort)."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, str, dict]] = []
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes.append((node_id, str(node_type), dict(properties or {})))

    def link_nodes(self, s, t, rel):
        self.edges.append((s, t, str(rel)))


class _CorrelatingEngine(_FakeEngine):
    """A _FakeEngine that also answers query_cypher for one known run_id,
    so correlate_lineage_run_trace resolves a real :RunTrace hit."""

    def __init__(self, known_run_id: str) -> None:
        super().__init__()
        self._known_trace_id = trace_id(known_run_id)

    def query_cypher(self, query: str, params: dict) -> list[dict]:
        if params.get("run_id") == self._known_trace_id:
            return [{"run_id": self._known_trace_id}]
        return []


def _run_event(
    *,
    run_id: str = "01977c9e-0000-7000-8000-000000000001",
    event_type: str = "COMPLETE",
    job_name: str = "etl-orders",
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    parent_run_id: str = "",
) -> dict[str, Any]:
    run: dict[str, Any] = {"runId": run_id}
    if parent_run_id:
        run["facets"] = {"parent": {"run": {"runId": parent_run_id}}}
    return {
        "eventType": event_type,
        "run": run,
        "job": {"namespace": "spark", "name": job_name},
        "inputs": inputs or [],
        "outputs": outputs or [],
    }


def _dataset(name: str, snapshot: str) -> dict[str, Any]:
    return {
        "namespace": "iceberg://lakehouse/sales",
        "name": name,
        "facets": {"version": {"datasetVersion": snapshot}},
    }


def test_record_openlineage_run_event_creates_activity_and_entities() -> None:
    eng = _FakeEngine()
    event = _run_event(
        inputs=[_dataset("raw_orders", "snap-7")],
        outputs=[_dataset("orders", "snap-42")],
    )
    activity_id = record_openlineage_run_event(eng, event)

    assert activity_id and activity_id.startswith("activity:openlineage_run:")
    activity_node = next(n for n in eng.nodes if n[0] == activity_id)
    assert activity_node[2]["kind"] == "openlineage_run"
    assert activity_node[2]["status"] == "completed"
    assert activity_node[2]["job"] == "etl-orders"

    in_id = "iceberg://lakehouse/sales/raw_orders@snap-7"
    out_id = "iceberg://lakehouse/sales/orders@snap-42"
    marker_ids = {n[0] for n in eng.nodes}
    assert in_id in marker_ids and out_id in marker_ids

    assert (activity_id, in_id, "used") in eng.edges
    assert (out_id, activity_id, "was_generated_by") in eng.edges


def test_record_openlineage_run_event_quarantined_creates_nothing() -> None:
    eng = _FakeEngine()
    event = _run_event()
    event["job"] = {}  # missing job.name -> quarantine
    result = record_openlineage_run_event(eng, event)
    assert result is None
    assert eng.nodes == []
    assert eng.edges == []


def test_record_openlineage_run_event_malformed_dataset_creates_nothing() -> None:
    eng = _FakeEngine()
    event = _run_event(inputs=[{"namespace": "postgres://db/public", "name": "raw"}])
    result = record_openlineage_run_event(eng, event)
    assert result is None
    assert eng.nodes == []


def test_record_openlineage_run_event_idempotent_on_redelivery() -> None:
    """Acceptance gate 3: redelivering the identical RunEvent twice must
    result in ONE Activity node, not two."""
    eng = _FakeEngine()
    event = _run_event(outputs=[_dataset("orders", "snap-42")])

    first = record_openlineage_run_event(eng, event)
    second = record_openlineage_run_event(eng, event)

    assert first == second
    activity_node_ids = {
        n[0] for n in eng.nodes if n[2].get("kind") == "openlineage_run"
    }
    assert len(activity_node_ids) == 1


def test_record_openlineage_run_event_status_transition_on_lifecycle_events() -> None:
    """A redelivered run under a DIFFERENT eventType (START then COMPLETE
    for the SAME run.runId) merge-upserts the same node's status forward,
    never creates a second Activity node."""
    eng = _FakeEngine()
    run_id = "01977c9e-0000-7000-8000-0000000000aa"

    start_id = record_openlineage_run_event(
        eng, _run_event(run_id=run_id, event_type="START")
    )
    complete_id = record_openlineage_run_event(
        eng, _run_event(run_id=run_id, event_type="COMPLETE")
    )

    assert start_id == complete_id
    activity_node_ids = {
        n[0] for n in eng.nodes if n[2].get("kind") == "openlineage_run"
    }
    assert len(activity_node_ids) == 1
    final_props = [n[2] for n in eng.nodes if n[0] == complete_id][-1]
    assert final_props["status"] == "completed"


def test_record_openlineage_run_event_correlates_tool_originated_run() -> None:
    """Acceptance gate 4 (positive): a run.runId matching an existing
    :RunTrace correlates via the additive edge."""
    run_id = "01977c9e-0000-7000-8000-0000000000bb"
    eng = _CorrelatingEngine(known_run_id=run_id)
    activity_id = record_openlineage_run_event(eng, _run_event(run_id=run_id))

    expected_trace_id = trace_id(run_id)
    assert (expected_trace_id, activity_id, "HAS_LINEAGE_ACTIVITY") in eng.edges


def test_record_openlineage_run_event_no_false_run_trace_link() -> None:
    """Acceptance gate 4 (negative): a Spark/Trino-originated run with no
    correlating :RunTrace creates a lineage-only Activity — never a
    fabricated RunTrace link."""
    eng = _FakeEngine()  # no query_cypher -> correlation always misses
    activity_id = record_openlineage_run_event(eng, _run_event())
    assert not any(
        e[1] == activity_id and e[2] == "HAS_LINEAGE_ACTIVITY" for e in eng.edges
    )


def test_record_openlineage_run_event_correlates_via_parent_run_facet() -> None:
    """A Spark/Trino run whose ParentRunFacet points at a tool-originated
    run correlates through the parent, not its own runId."""
    parent_id = "01977c9e-0000-7000-8000-0000000000cc"
    eng = _CorrelatingEngine(known_run_id=parent_id)
    activity_id = record_openlineage_run_event(
        eng,
        _run_event(
            run_id="01977c9e-0000-7000-8000-0000000000dd", parent_run_id=parent_id
        ),
    )
    expected_trace_id = trace_id(parent_id)
    assert (expected_trace_id, activity_id, "HAS_LINEAGE_ACTIVITY") in eng.edges


def test_record_openlineage_run_event_none_engine_returns_none() -> None:
    assert record_openlineage_run_event(None, _run_event()) is None
