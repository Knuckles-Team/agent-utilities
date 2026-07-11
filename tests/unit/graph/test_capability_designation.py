"""Plan 08 Synergy 1 -> AU-P1-3: KG-driven specialist designation wired into the router.

Verifies the enricher builds an ANN capability index from the engine's callable
nodes and designates by query embedding, and that it degrades gracefully (returns
None → router falls back) when embeddings/model are unavailable. Also verifies the
AU-P1-3 engine-native-first path, CDC-driven incremental cache maintenance, and
durable outcome persistence.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from agent_utilities.graph.routing.enrichers.capability_designation import (
    CapabilityIndexWatcher,
    build_designation_index,
    designate_specialists,
    explain_capability_eligibility,
    get_designation_index,
    record_capability_outcome,
)


def _make_engine(nodes: dict[str, dict]):
    """Fake engine exposing graph.node_ids() + graph._get_node_properties()."""
    graph = types.SimpleNamespace(
        node_ids=lambda: list(nodes.keys()),
        _get_node_properties=lambda nid: nodes.get(nid, {}),
    )
    return types.SimpleNamespace(graph=graph, backend=None)


# Two callable tools, near-orthogonal embeddings; one non-callable node ignored.
NODES = {
    "tool:search": {
        "type": "tool",
        "embedding": [1.0, 0.0, 0.0],
        "capabilities": ["web_search"],
    },
    "tool:math": {
        "type": "tool",
        "embedding": [0.0, 1.0, 0.0],
        "capabilities": ["arithmetic"],
    },
    "concept:foo": {"type": "concept", "embedding": [0.0, 0.0, 1.0]},  # not callable
}


def test_index_built_only_from_callable_nodes_with_embeddings():
    engine = _make_engine(NODES)
    index = build_designation_index(engine)
    assert index is not None
    assert len(index) == 2  # the concept node is excluded


def test_designate_returns_best_specialist():
    engine = _make_engine(NODES)
    # Query embedding closest to tool:search.
    out = designate_specialists(
        engine, "find me a search", k=1, embed_fn=lambda q: [0.95, 0.05, 0.0]
    )
    assert out == ["tool:search"]


def test_capability_filter_restricts_candidates():
    engine = _make_engine(NODES)
    out = designate_specialists(
        engine,
        "anything",
        k=5,
        required_caps=["arithmetic"],
        embed_fn=lambda q: [0.1, 0.9, 0.0],
    )
    assert out == ["tool:math"]


def test_graceful_fallback_when_no_embeddings():
    engine = _make_engine(
        {"tool:x": {"type": "tool", "capabilities": ["c"]}}  # no embedding
    )
    assert designate_specialists(engine, "q", embed_fn=lambda q: [1.0]) is None


def test_graceful_fallback_when_no_model_and_no_embed_fn():
    engine = _make_engine(NODES)
    # No embed_fn and create_embedding_model unavailable in-test -> None, not raise.
    out = designate_specialists(engine, "q", embed_fn=lambda q: None)
    assert out is None


def test_index_cached_on_engine():
    engine = _make_engine(NODES)
    designate_specialists(engine, "q", embed_fn=lambda q: [1.0, 0.0, 0.0])
    assert getattr(engine, "_designation_index", None) is not None


# ---------------------------------------------------------------------------
# CONCEPT:AU-P1-3 — engine-native-first routing (the in-process cache is a
# fallback, not the authority)
# ---------------------------------------------------------------------------
def test_designate_specialists_prefers_the_engine_native_path(monkeypatch):
    """When the engine's native filtered ANN answers, the in-process bounded
    cache is never even built — the engine IS the authority."""
    engine = _make_engine(NODES)
    calls: dict[str, Any] = {}

    def fake_engine_search(
        _engine,
        _embedding,
        *,
        k,
        required_caps,
        tenant,
        policy_tags,
        capability_hierarchy=None,
    ):
        calls["called"] = True
        calls["k"] = k
        calls["required_caps"] = required_caps
        calls["tenant"] = tenant
        calls["policy_tags"] = policy_tags
        return [("tool:math", 0.99)]

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.engine_capability_search."
        "engine_filtered_search",
        fake_engine_search,
    )

    out = designate_specialists(
        engine,
        "anything",
        k=1,
        required_caps=["arithmetic"],
        tenant="tenant-a",
        policy_tags=["cleared"],
        embed_fn=lambda q: [0.1, 0.9, 0.0],
    )

    assert calls["called"] is True
    assert calls["required_caps"] == ["arithmetic"]
    assert calls["tenant"] == "tenant-a"
    assert calls["policy_tags"] == ["cleared"]
    assert out == ["tool:math"]
    # The bounded in-process cache was never constructed — the engine answered.
    assert getattr(engine, "_capability_index_watcher", None) is None


def test_designate_specialists_falls_back_when_engine_search_returns_none(
    monkeypatch,
):
    """``engine_filtered_search`` returning ``None`` (no engine vector surface at
    all) is the signal to fall back to the bounded in-process cache — the
    existing keyword-scan-degrading behaviour is unchanged."""
    engine = _make_engine(NODES)

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.engine_capability_search."
        "engine_filtered_search",
        lambda *a, **k: None,
    )

    out = designate_specialists(
        engine, "find me a search", k=1, embed_fn=lambda q: [0.95, 0.05, 0.0]
    )
    assert out == ["tool:search"]
    assert getattr(engine, "_capability_index_watcher", None) is not None


# ---------------------------------------------------------------------------
# CONCEPT:AU-P1-3 — CDC-driven incremental indexing (no periodic full rebuild)
# ---------------------------------------------------------------------------
class _FakeStreaming:
    """Minimal stand-in for ``client.streaming`` (``cdc_read`` + ``watch``)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def cdc_read(self, _graph_name, cursor, limit=4096):
        pending = [e for e in self.events if e["seq"] >= cursor]
        return pending[:limit]

    def watch(self, _graph_name, cursor, label="", timeout_ms=0):
        pending = [e for e in self.events if e["seq"] >= cursor]
        next_seq = (pending[-1]["seq"] + 1) if pending else cursor
        return {"events": pending, "next_seq": next_seq}


class _CdcCapableGraph:
    """A callable-node graph that ALSO resolves an engine streaming surface."""

    def __init__(self, nodes: dict[str, dict], streaming: _FakeStreaming) -> None:
        self._nodes = dict(nodes)
        self._client = types.SimpleNamespace(streaming=streaming)
        self.graph_name = "engtest"
        self.node_ids_calls = 0
        self.get_props_calls = 0

    def node_ids(self):
        self.node_ids_calls += 1
        return list(self._nodes.keys())

    def _get_node_properties(self, nid):
        self.get_props_calls += 1
        return self._nodes.get(nid, {})


def test_cdc_incremental_upsert_does_not_trigger_a_full_rescan():
    streaming = _FakeStreaming()
    graph = _CdcCapableGraph(NODES, streaming)
    engine = types.SimpleNamespace(graph=graph, backend=None)

    watcher = CapabilityIndexWatcher(engine)
    assert watcher._subscription is not None
    assert watcher._subscription.available is True
    assert graph.node_ids_calls == 1  # exactly one bootstrap full scan
    assert len(watcher.index) == 2

    # A brand-new capability node arrives purely via the CDC feed.
    streaming.events.append(
        {
            "seq": 1,
            "kind": "upsert",
            "node_id": "tool:new",
            "label": "Tool",
            "after": {
                "type": "tool",
                "embedding": [0.0, 0.0, 1.0],
                "capabilities": ["new_cap"],
            },
        }
    )
    watcher.refresh()

    assert graph.node_ids_calls == 1  # STILL just the one bootstrap scan
    assert "tool:new" in watcher.index
    assert len(watcher.index) == 3


def test_cdc_delete_event_evicts_the_node_incrementally():
    streaming = _FakeStreaming()
    graph = _CdcCapableGraph(NODES, streaming)
    engine = types.SimpleNamespace(graph=graph, backend=None)

    watcher = CapabilityIndexWatcher(engine)
    assert "tool:search" in watcher.index

    streaming.events.append(
        {
            "seq": 1,
            "kind": "delete",
            "node_id": "tool:search",
            "label": "Tool",
            "after": None,
        }
    )
    watcher.refresh()

    assert graph.node_ids_calls == 1  # eviction was incremental, not a rescan
    assert "tool:search" not in watcher.index
    assert len(watcher.index) == 1


def test_cdc_unavailable_degrades_to_full_rebuild_every_refresh():
    """No engine streaming surface (the common fake-engine case in this suite) —
    ``refresh()`` degrades to the pre-AU-P1-3 always-rebuild behaviour."""
    engine = _make_engine(NODES)
    watcher = CapabilityIndexWatcher(engine)
    assert watcher._subscription is None or not watcher._subscription.available
    assert len(watcher.index) == 2

    first = watcher.refresh()
    second = watcher.refresh()
    assert first is not None and second is not None
    assert len(first) == 2
    assert len(second) == 2


# ---------------------------------------------------------------------------
# CONCEPT:AU-P1-3 — durable contextual-bandit outcomes
# ---------------------------------------------------------------------------
class _FakeCypherBackend:
    """A minimal in-memory Cypher executor for the durable-outcome write path."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def execute(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        nid = str(params.get("id"))
        if "SET" in query:
            node = self.nodes.setdefault(nid, {})
            node["capability_reward"] = params.get("r")
            node["capability_reward_count"] = params.get("c")
            return []
        node = self.nodes.get(nid, {})
        return [
            {
                "reward": node.get("capability_reward"),
                "count": node.get("capability_reward_count"),
            }
        ]


def test_record_capability_outcome_updates_inprocess_and_persists_durably():
    engine = _make_engine(NODES)
    # Establish the in-process bounded cache (the fake engine has no ANN surface).
    designate_specialists(engine, "q", embed_fn=lambda q: [1.0, 0.0, 0.0])
    watcher = getattr(engine, "_capability_index_watcher", None)
    assert watcher is not None

    engine.backend = _FakeCypherBackend()

    updated: float = record_capability_outcome(engine, "tool:search", success=True)
    assert updated > 0.5

    # In-process cache reflects it immediately (same-process fast path).
    assert watcher.index.reward_of("tool:search") == updated

    # It ALSO survives independent of any in-process state (durability, not just EMA).
    from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
        read_capability_reward,
    )

    assert read_capability_reward(engine, "tool:search") == pytest.approx(updated)


def test_record_capability_outcome_never_raises_without_a_backend():
    engine = _make_engine(NODES)
    designate_specialists(engine, "q", embed_fn=lambda q: [1.0, 0.0, 0.0])
    # engine.backend is None (from _make_engine) -> durable persistence is a no-op,
    # but the in-process EMA update still applies and nothing raises.
    updated = record_capability_outcome(engine, "tool:search", success=True)
    assert updated > 0.5


# ---------------------------------------------------------------------------
# CONCEPT:AU-P1-3 — explainable routing
# ---------------------------------------------------------------------------
def test_explain_capability_eligibility_reports_matched_and_missing_features():
    engine = _make_engine(NODES)
    designate_specialists(engine, "q", embed_fn=lambda q: [1.0, 0.0, 0.0])

    eligible = explain_capability_eligibility(
        engine, "tool:math", required_caps=["arithmetic"]
    )
    assert eligible is not None
    assert eligible["eligible"] is True
    assert eligible["capabilities_matched"] is True

    ineligible = explain_capability_eligibility(
        engine, "tool:math", required_caps=["web_search"]
    )
    assert ineligible is not None
    assert ineligible["eligible"] is False
    assert ineligible["missing_caps"] == ["web_search"]


def test_explain_capability_eligibility_returns_none_for_unknown_entity():
    engine = _make_engine(NODES)
    designate_specialists(engine, "q", embed_fn=lambda q: [1.0, 0.0, 0.0])
    assert explain_capability_eligibility(engine, "no_such_id") is None


def test_get_designation_index_force_refresh_bypasses_cdc_gating():
    """``refresh=True`` still forces a full rebuild (pre-AU-P1-3 semantics kept)."""
    engine = _make_engine(NODES)
    first = get_designation_index(engine)
    assert first is not None and len(first) == 2

    # Even with nothing "changed", an explicit refresh re-derives the index.
    forced = get_designation_index(engine, refresh=True)
    assert forced is not None and len(forced) == 2
