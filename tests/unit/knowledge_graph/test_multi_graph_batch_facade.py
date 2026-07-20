"""Batched cross-graph writer facade (CONCEPT:AU-KG.ingest.batched-cross-graph-writer).

``GraphComputeEngine.multi_graph_batch_update`` ships a ``graph → ops`` map to the
engine's mandatory ``MultiGraphBatchUpdate`` op in ONE round-trip. These tests exercise
the facade logic over a stub client (no live engine) by bypassing ``__init__``.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _engine_with_client(
    client: Any, graph_name: str = "__commons__"
) -> GraphComputeEngine:
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
        "src:freshrss#1": [
            {"op": "add_node", "id": "b"},
            {"op": "add_node", "id": "c"},
        ],
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


def test_multi_graph_batch_update_requires_current_engine_op() -> None:
    class _IncompleteClient:
        lifecycle = object()

    eng = _engine_with_client(_IncompleteClient())
    with pytest.raises(RuntimeError, match="requires MultiGraphBatchUpdate"):
        eng.multi_graph_batch_update(
            {"source-alias": [{"op": "add_node", "id": "opaque"}]}
        )
