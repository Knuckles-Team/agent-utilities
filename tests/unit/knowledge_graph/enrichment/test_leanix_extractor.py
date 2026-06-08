"""Tests for the LeanIX EA source extractor (CONCEPT:KG-2.9)."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.enrichment.extractors.leanix import extract
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)


class FakeLeanIXClient:
    """Duck-typed stand-in for a LeanIX API client (no network)."""

    def __init__(self):
        self._sheets: dict[str, list[dict[str, Any]]] = {
            "Application": [
                {
                    "id": "a1",
                    "name": "Billing",
                    "type": "Application",
                    "relApplicationToBusinessCapability": [{"factSheetId": "c1"}],
                    "relApplicationToITComponent": "ic1",
                },
                {
                    "id": "a2",
                    "name": "CRM",
                    "type": "Application",
                    # tolerant LeanIX edges/node envelope
                    "relApplicationToBusinessCapability": {
                        "edges": [{"node": {"id": "c1"}}]
                    },
                    "relApplicationToITComponent": [{"id": "ic1"}],
                },
            ],
            "ITComponent": [
                {"id": "ic1", "name": "PostgreSQL", "type": "ITComponent"},
            ],
            "BusinessCapability": [
                {
                    "id": "c1",
                    "name": "Revenue Management",
                    "type": "BusinessCapability",
                },
            ],
            "DataObject": [],
        }

    def factsheets(self, type=None):  # noqa: A002 - mirror LeanIX API kwarg
        if type is None:
            out: list[dict[str, Any]] = []
            for items in self._sheets.values():
                out.extend(items)
            return out
        return self._sheets.get(type, [])


class FakeConfig:
    def __init__(self, client):
        self.client = client


class FakeBackend:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, s, t, **props):
        self.edges.append((s, t, props.get("rel_type")))


def _run():
    return extract(FakeConfig(FakeLeanIXClient()))


def test_nodes_are_typed_and_prefixed():
    batch = _run()
    by_id = {n.id: n for n in batch.nodes}

    assert by_id["app:a1"].type == "Application"
    assert by_id["app:a1"].props["name"] == "Billing"
    assert by_id["app:a2"].type == "Application"
    assert by_id["itcomponent:ic1"].type == "ITComponent"
    assert by_id["capability:c1"].type == "BusinessCapability"
    # 2 Applications + 1 ITComponent + 1 BusinessCapability
    assert len(batch.nodes) == 4


def test_supports_and_depends_on_edges():
    batch = _run()
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}

    assert ("app:a1", "capability:c1", "SUPPORTS") in triples
    assert ("app:a1", "itcomponent:ic1", "DEPENDS_ON") in triples
    assert ("app:a2", "capability:c1", "SUPPORTS") in triples
    assert ("app:a2", "itcomponent:ic1", "DEPENDS_ON") in triples
    # No edges originate from non-Application nodes.
    assert all(s.startswith("app:") for s, _, _ in triples)


def test_source_is_registered():
    src = get_source("leanix")
    assert src is not None
    assert src.extract is extract
    assert "LeanIX" in src.description


def test_write_batch_persists_to_backend():
    batch = _run()
    backend = FakeBackend()
    n, e = write_batch(backend, batch)

    assert n == 4
    assert e == 4
    assert backend.nodes["app:a1"]["type"] == "Application"
    assert backend.nodes["app:a1"]["name"] == "Billing"
    assert backend.nodes["capability:c1"]["type"] == "BusinessCapability"
    assert ("app:a1", "capability:c1", "SUPPORTS") in backend.edges
    assert ("app:a2", "itcomponent:ic1", "DEPENDS_ON") in backend.edges


def test_empty_client_yields_empty_batch():
    class Empty:
        client = None

    batch = extract(Empty())
    assert batch.category == "leanix"
    assert batch.nodes == [] and batch.edges == []
