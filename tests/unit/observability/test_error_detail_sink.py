"""Durable KG-backed error-detail persistence sink (D-24).

CONCEPT:AU-KG.audit.durable-error-detail — :class:`GraphErrorDetailSink` is the concrete
backend wired through ``error_surface.register_detail_persistence_sink``, and
``error_surface.resolve_error_detail`` falls back to its ``resolve`` method on an
in-process-store miss (e.g. a restart, or a different replica than the one that recorded
the failure).
"""

from __future__ import annotations

from typing import Any

from agent_utilities.observability import correlation
from agent_utilities.observability.error_detail_sink import (
    ERROR_DETAIL_DIAGNOSES_EDGE,
    ERROR_DETAIL_NODE_LABEL,
    GraphErrorDetailSink,
    _error_detail_node_id,
)
from agent_utilities.observability.trace_ontology import (
    TRACE_NODE_LABEL,
)
from agent_utilities.observability.trace_ontology import (
    trace_id as run_trace_id,
)
from agent_utilities.security import error_surface


class FakeEngine:
    """Minimal ``add_node``/``link_nodes``/``query_cypher`` double.

    Mirrors ``tests/unit/capabilities/test_kg_audit_sink.py``'s ``FakeEngine`` shape.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_types: dict[str, str] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> None:
        self.nodes[node_id] = dict(properties or {})
        self.node_types[node_id] = node_type

    def link_nodes(
        self, source_id: str, target_id: str, rel_type: str, **_kw: Any
    ) -> None:
        self.edges.append((source_id, target_id, rel_type))

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        node_id = params.get("id")
        if ERROR_DETAIL_NODE_LABEL not in query or node_id is None:
            return []
        props = self.nodes.get(node_id)
        if props is None or self.node_types.get(node_id) != ERROR_DETAIL_NODE_LABEL:
            return []
        return [
            {
                "error_class": props.get("error_class"),
                "failing_layer": props.get("failing_layer"),
                "traceback": props.get("traceback"),
                "tenant_id": props.get("tenant_id"),
            }
        ]


def _record(**overrides: Any) -> dict[str, Any]:
    base = {
        "error_class": "RuntimeError",
        "failing_layer": "knowledge_graph",
        "traceback": "Traceback (sanitized)",
        "tenant_id": "acme",
    }
    base.update(overrides)
    return base


def test_engine_none_is_a_safe_noop() -> None:
    sink = GraphErrorDetailSink(engine=None)
    sink("correlation:abc", _record())  # must not raise
    assert sink.resolve("correlation:abc") is None


def test_write_then_resolve_round_trips() -> None:
    engine = FakeEngine()
    sink = GraphErrorDetailSink(engine=engine)
    sink("correlation:abc123", _record())

    resolved = sink.resolve("correlation:abc123")
    assert resolved == _record()


def test_resolve_unknown_correlation_id_returns_none() -> None:
    engine = FakeEngine()
    sink = GraphErrorDetailSink(engine=engine)
    assert sink.resolve("correlation:does-not-exist") is None


def test_write_failure_is_best_effort_and_never_raises() -> None:
    class _BrokenEngine:
        def add_node(self, *_a: Any, **_kw: Any) -> None:
            raise RuntimeError("engine is down")

    sink = GraphErrorDetailSink(engine=_BrokenEngine())
    sink("correlation:abc", _record())  # must not raise


def test_links_to_run_trace_when_a_correlation_id_is_active() -> None:
    engine = FakeEngine()
    sink = GraphErrorDetailSink(engine=engine)
    token = correlation._correlation_id.set("run-correlation-xyz")
    try:
        sink("correlation:abc", _record())
    finally:
        correlation._correlation_id.reset(token)

    node_id = _error_detail_node_id("correlation:abc")
    expected_target = run_trace_id("run-correlation-xyz")
    assert (node_id, expected_target, ERROR_DETAIL_DIAGNOSES_EDGE) in engine.edges


def test_no_run_trace_link_when_no_correlation_id_is_active() -> None:
    engine = FakeEngine()
    sink = GraphErrorDetailSink(engine=engine)
    assert correlation.get_correlation_id() is None
    sink("correlation:abc", _record())
    assert engine.edges == []


def test_error_surface_resolve_falls_back_to_durable_sink_on_in_process_miss() -> None:
    """The D-24 live path: error_surface.resolve_error_detail() must reach the
    durable sink when the bounded in-process store doesn't have the ref (as if
    this were a fresh process / a different replica)."""
    engine = FakeEngine()
    sink = GraphErrorDetailSink(engine=engine)
    # Write directly to the durable backend only — never through
    # error_surface's in-process store — to simulate "recorded by another
    # process/replica".
    sink("correlation:durable-only", _record(error_class="ValueError"))

    error_surface.register_detail_persistence_sink(sink)
    try:
        resolved = error_surface.resolve_error_detail("correlation:durable-only")
    finally:
        error_surface.register_detail_persistence_sink(None)

    assert resolved is not None
    assert resolved["error_class"] == "ValueError"


def test_error_surface_resolve_prefers_in_process_store_over_durable_sink() -> None:
    engine = FakeEngine()
    sink = GraphErrorDetailSink(engine=engine)
    error_surface.register_detail_persistence_sink(sink)
    try:
        payload = error_surface.public_error_payload(RuntimeError("boom"))
        detail_ref = payload["error"]["detail_ref"]
        resolved = error_surface.resolve_error_detail(detail_ref)
    finally:
        error_surface.register_detail_persistence_sink(None)

    assert resolved is not None
    assert resolved["error_class"] == "RuntimeError"


def test_error_surface_resolve_survives_a_broken_durable_resolver() -> None:
    class _BrokenResolveSink:
        def __call__(self, _cid: str, _record: dict[str, Any]) -> None:
            return None

        def resolve(self, _cid: str) -> dict[str, Any]:
            raise RuntimeError("durable backend unavailable")

    error_surface.register_detail_persistence_sink(_BrokenResolveSink())
    try:
        assert error_surface.resolve_error_detail("correlation:does-not-exist") is None
    finally:
        error_surface.register_detail_persistence_sink(None)


def test_run_trace_node_label_matches_canonical_trace_ontology() -> None:
    # Sanity: the edge target uses the SAME trace_id() the canonical
    # observability.trace_ontology module (and KgAuditSink) already write
    # RunTrace nodes under, not a second id scheme.
    assert run_trace_id("run-x").startswith("trace:")
    assert TRACE_NODE_LABEL == "RunTrace"
