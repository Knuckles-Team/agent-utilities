"""Batched cross-graph writer facade (CONCEPT:AU-KG.ingest.batched-cross-graph-writer).

``GraphComputeEngine.multi_graph_batch_update`` ships a ``graph → ops`` map to the
engine's ``MultiGraphBatchUpdate`` op in ONE round-trip when the client supports
it, and degrades to per-graph writes against an older client. These tests exercise
the facade logic over a stub client (no live engine) by bypassing ``__init__``.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _engine_with_client(client: Any, graph_name: str = "__commons__") -> GraphComputeEngine:
    eng = object.__new__(GraphComputeEngine)
    eng._client = client
    eng.graph_name = graph_name
    return eng


class _Lifecycle:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def multi_graph_batch_update(self, batches: dict) -> dict:
        self.calls.append(batches)
        return {
            "results": {g: {"added_nodes": len(ops)} for g, ops in batches.items()},
            "errors": {},
        }


class _Client:
    def __init__(self) -> None:
        self.lifecycle = _Lifecycle()


def test_multi_graph_batch_update_uses_engine_op_when_available() -> None:
    client = _Client()
    eng = _engine_with_client(client)
    batches = {
        "src:freshrss#0": [{"op": "add_node", "id": "a"}],
        "src:freshrss#1": [{"op": "add_node", "id": "b"}, {"op": "add_node", "id": "c"}],
    }
    out = eng.multi_graph_batch_update(batches)
    # ONE round-trip carried the whole map.
    assert len(client.lifecycle.calls) == 1
    assert client.lifecycle.calls[0] == batches
    assert out["results"]["src:freshrss#0"]["added_nodes"] == 1
    assert out["results"]["src:freshrss#1"]["added_nodes"] == 2
    assert out["errors"] == {}


def test_multi_graph_batch_update_empty_is_noop() -> None:
    eng = _engine_with_client(_Client())
    assert eng.multi_graph_batch_update({}) == {"results": {}, "errors": {}}


def test_multi_graph_batch_update_degrades_without_engine_op(monkeypatch) -> None:
    """Older client (no ``multi_graph_batch_update``): each sub-batch applies on its
    own graph-bound engine; the current graph uses ``self.batch_update`` directly."""

    class _OldLifecycle:
        pass  # no multi_graph_batch_update

    class _OldClient:
        lifecycle = _OldLifecycle()

    eng = _engine_with_client(_OldClient(), graph_name="src:home")
    applied_here: list[list] = []
    eng.batch_update = lambda ops: applied_here.append(ops) or {"added_nodes": len(ops)}  # type: ignore[method-assign]

    # Stub engine_for_graph so the "other graph" path is exercised without an engine.
    other_applied: dict[str, list] = {}

    class _OtherGC:
        def __init__(self, name: str) -> None:
            self.name = name

        def batch_update(self, ops: list) -> dict:
            other_applied.setdefault(self.name, []).extend(ops)
            return {"added_nodes": len(ops)}

    class _OtherEng:
        def __init__(self, name: str) -> None:
            self.graph_compute = _OtherGC(name)

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.ingest_routing.engine_for_graph",
        lambda name: _OtherEng(name),
    )

    batches = {
        "src:home": [{"op": "add_node", "id": "h"}],
        "src:other": [{"op": "add_node", "id": "o1"}, {"op": "add_node", "id": "o2"}],
    }
    out = eng.multi_graph_batch_update(batches)
    # Current graph went through self.batch_update; the other via engine_for_graph.
    assert applied_here == [[{"op": "add_node", "id": "h"}]]
    assert len(other_applied["src:other"]) == 2
    assert out["results"]["src:home"]["added_nodes"] == 1
    assert out["results"]["src:other"]["added_nodes"] == 2
    assert out["errors"] == {}
