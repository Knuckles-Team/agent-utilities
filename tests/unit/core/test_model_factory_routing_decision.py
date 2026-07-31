"""Live-path test: role-routed model creation persists a RoutingDecision onto the
current trace (CONCEPT:AU-ORCH.routing.rejected-candidate-provenance).

Exercises the ACTUAL entry point (``model_factory._resolve_role_model``, the same
function ``create_model(role=...)`` calls) rather than only unit-testing
``explain_pick_for_task``/``record_routing_decision`` in isolation — the Wire-First
requirement that a live call path really invokes the new behavior.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.core import model_factory
from agent_utilities.harness import tracing
from agent_utilities.harness.trace_backend import KGTraceBackend

pytestmark = pytest.mark.concept(id="AU-ORCH.routing.rejected-candidate-provenance")


class _FakeEngine:
    """Mirrors the real ``IntelligenceGraphEngine.add_node(node_id, node_type,
    properties)`` contract (D-5.1-5) — no ``**kwargs`` catch-all."""

    def __init__(self):
        self.added: list[tuple[str, dict]] = []
        self.linked: list[tuple[str, str, str]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict | None = None
    ) -> None:
        self.added.append((node_id, {"node_type": node_type, **(properties or {})}))

    def link_nodes(self, src, dst, rel):
        self.linked.append((src, dst, str(rel)))


@pytest.fixture
def registry_path(tmp_path):
    data = {
        "models": [
            {
                "id": "local-light",
                "name": "Local Light",
                "provider": "openai",
                "model_id": "local-light-model",
                "tier": "light",
                "is_default": True,
            },
            {
                "id": "cloud-heavy",
                "name": "Cloud Heavy",
                "provider": "openai",
                "model_id": "cloud-heavy-model",
                "tier": "heavy",
            },
        ]
    }
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_resolve_role_model_persists_routing_decision_on_the_trace(
    monkeypatch, registry_path
):
    monkeypatch.setattr(
        model_factory.config, "model_registry_path", registry_path, raising=False
    )
    engine = _FakeEngine()
    sink = KGTraceBackend(backend=engine)
    prev_sink = tracing.get_kg_trace_sink()
    tracing.set_kg_trace_sink(sink)
    token = tracing._current_trace_id.set("trace:routing-test")
    try:
        model = model_factory._resolve_role_model("planner")
        assert model is not None
    finally:
        tracing._current_trace_id.reset(token)
        tracing.set_kg_trace_sink(prev_sink)

    routing_nodes = [
        props
        for node_id, props in engine.added
        if props.get("node_type") == "routing_decision"
    ]
    assert routing_nodes, "expected a RoutingDecisionNode to be persisted"
    node = routing_nodes[0]
    assert node["trace_id"] == "trace:routing-test"
    assert node["chosen_model_id"] == model.id
    assert node["candidates"]  # bounded candidate set, never empty when models exist
    # Every candidate is explained: chosen has no reason, every rejected one does.
    for c in node["candidates"]:
        if c["model_id"] == model.id:
            assert c["rejected"] is False
        else:
            assert c["rejected"] is True
            assert c["rejection_reason"]
    assert any(
        rel == "has_routing_decision" and src == "trace:routing-test"
        for src, _dst, rel in engine.linked
    )


def test_resolve_role_model_records_nothing_without_a_trace_sink(
    monkeypatch, registry_path
):
    monkeypatch.setattr(
        model_factory.config, "model_registry_path", registry_path, raising=False
    )
    prev_sink = tracing.get_kg_trace_sink()
    tracing.set_kg_trace_sink(None)
    try:
        model = model_factory._resolve_role_model("planner")
        assert model is not None  # unaffected: recording is best-effort/optional
    finally:
        tracing.set_kg_trace_sink(prev_sink)
