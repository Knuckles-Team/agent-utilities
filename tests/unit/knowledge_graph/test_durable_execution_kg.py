"""``:DurableExecutionUnit`` KG mirror (DE1, CONCEPT:AU-KG.storage.durable-execution-unit).

Covers :mod:`agent_utilities.knowledge_graph.durable_execution_kg`: the
``engine=None`` no-op contract (a provenance mirror must never fail the run it
describes), a real upsert writing the exact mirrored properties the DE0
ontology schema names, the reverse ``:produced`` edge to a ``RunTrace``, and
the cross-backend read helper answering "what is durably in flight, waiting on
what" over whichever concrete labels are present.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.durable_execution_kg import (
    DURABLE_RUN_LABEL,
    durable_run_unit_id,
    link_durable_run_produced,
    list_durable_execution_units,
    mirror_durable_run,
)


class FakeEngine:
    """Minimal ``add_node``/``link_nodes``/``query_cypher`` double (mirrors the
    shape ``tests/unit/capabilities/test_kg_audit_sink.py``'s own ``FakeEngine``
    uses)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_types: dict[str, str] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = dict(properties or {})
        self.node_types[node_id] = node_type

    def link_nodes(self, source_id: str, target_id: str, rel_type: str) -> None:
        self.edges.append((source_id, target_id, rel_type))

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        rows = []
        for node_id, props in self.nodes.items():
            label = self.node_types.get(node_id)
            if f"(u:{label})" not in query:
                continue
            status_filter = params.get("status")
            if (
                status_filter is not None
                and props.get("durable_status") != status_filter
            ):
                continue
            awaits = [t for s, t, rel in self.edges if s == node_id and rel == "awaits"]
            coordinated_by = [
                t for s, t, rel in self.edges if s == node_id and rel == "coordinatedBy"
            ]
            rows.append(
                {
                    "u": {**props, "id": node_id},
                    "awaits": awaits,
                    "coordinated_by": coordinated_by,
                }
            )
        return rows


def test_mirror_is_a_noop_without_an_engine():
    """A provenance mirror must never fail (or require) the run it describes."""
    result = mirror_durable_run(
        None, session_id="s1", run_id="r1", durable_status="PENDING"
    )
    assert result is None


def test_mirror_upserts_the_exact_de0_schema_properties():
    engine = FakeEngine()
    node_id = mirror_durable_run(
        engine,
        session_id="s1",
        run_id="r1",
        durable_status="COMPLETED",
        checkpoint_ref="r1:step_a",
        definition_version="v2",
        idempotency_key="r1:step_a",
    )
    assert node_id == durable_run_unit_id("s1", "r1")
    assert engine.node_types[node_id] == DURABLE_RUN_LABEL
    props = engine.nodes[node_id]
    assert props["backend_ref"] == "s1:r1"
    assert props["durable_status"] == "COMPLETED"
    assert props["checkpoint_ref"] == "r1:step_a"
    assert props["definition_version"] == "v2"
    assert props["idempotency_key"] == "r1:step_a"
    assert props["durable_unit_kind"] == "DurableExecutionUnit"


def test_mirror_upsert_is_idempotent_by_node_id():
    """Repeated calls for the same run update ONE node, never accumulate rows."""
    engine = FakeEngine()
    mirror_durable_run(engine, session_id="s1", run_id="r1", durable_status="PENDING")
    mirror_durable_run(engine, session_id="s1", run_id="r1", durable_status="COMPLETED")
    assert len(engine.nodes) == 1
    assert (
        engine.nodes[durable_run_unit_id("s1", "r1")]["durable_status"] == "COMPLETED"
    )


def test_link_produced_is_a_noop_without_an_engine_or_trace_id():
    link_durable_run_produced(
        None, session_id="s1", run_id="r1", run_trace_id="trace:x"
    )
    engine = FakeEngine()
    link_durable_run_produced(engine, session_id="s1", run_id="r1", run_trace_id="")
    assert engine.edges == []


def test_link_produced_writes_the_produced_edge():
    engine = FakeEngine()
    mirror_durable_run(engine, session_id="s1", run_id="r1", durable_status="COMPLETED")
    link_durable_run_produced(
        engine, session_id="s1", run_id="r1", run_trace_id="trace:abc"
    )
    assert engine.edges == [(durable_run_unit_id("s1", "r1"), "trace:abc", "produced")]


def test_list_durable_execution_units_answers_whats_in_flight_and_waiting_on_what():
    engine = FakeEngine()
    mirror_durable_run(engine, session_id="s1", run_id="r1", durable_status="PENDING")
    mirror_durable_run(engine, session_id="s2", run_id="r2", durable_status="COMPLETED")
    engine.link_nodes(
        durable_run_unit_id("s1", "r1"), durable_run_unit_id("s2", "r2"), "awaits"
    )

    all_units = list_durable_execution_units(engine)
    assert {u["run_id"] for u in all_units} == {"r1", "r2"}

    in_flight = list_durable_execution_units(engine, status="PENDING")
    assert len(in_flight) == 1
    assert in_flight[0]["run_id"] == "r1"
    assert in_flight[0]["awaits"] == [durable_run_unit_id("s2", "r2")]


def test_list_durable_execution_units_degrades_on_query_failure():
    class BrokenEngine(FakeEngine):
        def query_cypher(self, query, params=None):
            raise RuntimeError("engine unavailable")

    assert list_durable_execution_units(BrokenEngine()) == []
