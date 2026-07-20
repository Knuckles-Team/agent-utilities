"""Engine-native specialist designation and durable outcome tests."""

from __future__ import annotations

import re
import types
from typing import Any

import pytest

from agent_utilities.graph.routing.enrichers.capability_designation import (
    designate_specialists,
    explain_capability_eligibility,
    record_capability_outcome,
)

NODES = {
    "tool:search": {
        "type": "tool",
        "capabilities": ["web_search"],
        "tenant": "tenant-a",
        "policy_tags": ["cleared"],
    },
    "tool:math": {"type": "tool", "capabilities": ["arithmetic"]},
}


class _Backend:
    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.nodes = nodes

    def execute(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        node = self.nodes.setdefault(str(params.get("id")), {})
        if "SET" in query:
            for prop, param in re.findall(r"n\.(\w+)\s*=\s*\$(\w+)", query):
                node[prop] = params.get(param)
            return []
        return [
            {
                alias: node.get(prop)
                for prop, alias in re.findall(r"n\.(\w+)\s+AS\s+(\w+)", query)
            }
        ]


def _make_engine(nodes: dict[str, dict[str, Any]] | None = None):
    values = {key: dict(value) for key, value in (nodes or NODES).items()}
    graph = types.SimpleNamespace(
        _get_node_properties=lambda node_id: values.get(node_id, {})
    )
    return types.SimpleNamespace(graph=graph, backend=_Backend(values))


def test_designate_uses_engine_native_filtered_search(monkeypatch):
    engine = _make_engine()
    observed: dict[str, Any] = {}

    def search(_engine, embedding, **kwargs):
        observed.update(embedding=embedding, **kwargs)
        return [("tool:math", 0.99)]

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.engine_capability_search."
        "engine_filtered_search",
        search,
    )
    result = designate_specialists(
        engine,
        "calculate",
        k=1,
        required_caps=["arithmetic"],
        tenant="tenant-a",
        policy_tags=["cleared"],
        embed_fn=lambda _query: [0.0, 1.0],
    )
    assert result == ["tool:math"]
    assert observed["required_caps"] == ["arithmetic"]
    assert observed["tenant"] == "tenant-a"
    assert observed["policy_tags"] == ["cleared"]


def test_designate_reports_unavailable_engine_vector_surface(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.engine_capability_search."
        "engine_filtered_search",
        lambda *_args, **_kwargs: None,
    )
    assert (
        designate_specialists(
            _make_engine(), "query", embed_fn=lambda _query: [1.0, 0.0]
        )
        is None
    )


def test_designate_reports_unavailable_embedding():
    assert (
        designate_specialists(
            _make_engine(), "query", embed_fn=lambda _query: None
        )
        is None
    )


def test_record_outcome_persists_to_engine_authority():
    engine = _make_engine()
    updated = record_capability_outcome(engine, "tool:search", success=True)
    assert updated > 0.5
    from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
        read_capability_reward,
    )

    assert read_capability_reward(engine, "tool:search") == pytest.approx(updated)


def test_record_outcome_fails_when_authority_is_unavailable():
    engine = _make_engine()
    engine.backend = None
    with pytest.raises(RuntimeError, match="persistence is unavailable"):
        record_capability_outcome(engine, "tool:search", success=True)


def test_explain_reads_authoritative_node_properties():
    engine = _make_engine()
    report = explain_capability_eligibility(
        engine,
        "tool:search",
        required_caps=["web_search"],
        tenant="tenant-a",
        policy_tags=["cleared"],
    )
    assert report is not None
    assert report["eligible"] is True
    assert report["capabilities_matched"] is True


def test_explain_unknown_entity_returns_none():
    assert explain_capability_eligibility(_make_engine(), "missing") is None
