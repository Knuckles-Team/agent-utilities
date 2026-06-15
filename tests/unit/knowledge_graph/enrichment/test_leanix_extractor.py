"""Tests for the LeanIX EA source extractor (CONCEPT:KG-2.9)."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.enrichment.extractors.leanix import extract
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)

# A live-metamodel slice: the core EA types plus a CUSTOM type (DataCenter) with a
# CUSTOM relation, so the extractor must mirror beyond the hardcoded handful.
META_MODEL = {
    "factSheets": {
        "Application": {
            "fields": {"displayName": {"type": "STRING"}},
            "relations": {
                "relApplicationToBusinessCapability": {
                    "targetFactSheetType": "BusinessCapability"
                },
                "relApplicationToITComponent": {"targetFactSheetType": "ITComponent"},
                "relApplicationToDataCenter": {"targetFactSheetType": "DataCenter"},
            },
        },
        "ITComponent": {"fields": {}, "relations": {}},
        "BusinessCapability": {"fields": {}, "relations": {}},
        "DataCenter": {"fields": {"region": {"type": "STRING"}}, "relations": {}},
    }
}


class FakeLeanIXClient:
    """Duck-typed stand-in for a LeanIX client (no network)."""

    def __init__(self):
        self._sheets: dict[str, list[dict[str, Any]]] = {
            "Application": [
                {
                    "id": "a1",
                    "name": "Billing",
                    "type": "Application",
                    "relApplicationToBusinessCapability": [{"factSheetId": "c1"}],
                    "relApplicationToITComponent": "ic1",
                    # custom relation, LeanIX edges/node/factSheet envelope
                    "relApplicationToDataCenter": {
                        "edges": [
                            {"node": {"factSheet": {"id": "dc1", "type": "DataCenter"}}}
                        ]
                    },
                },
                {
                    "id": "a2",
                    "name": "CRM",
                    "type": "Application",
                    "relApplicationToBusinessCapability": {
                        "edges": [{"node": {"id": "c1"}}]
                    },
                    "relApplicationToITComponent": [{"id": "ic1"}],
                },
            ],
            "ITComponent": [{"id": "ic1", "name": "PostgreSQL", "type": "ITComponent"}],
            "BusinessCapability": [
                {"id": "c1", "name": "Revenue Management", "type": "BusinessCapability"}
            ],
            "DataCenter": [{"id": "dc1", "name": "eu-west", "type": "DataCenter"}],
        }

    def meta_model(self):
        return META_MODEL

    def factsheets(self, type=None, since=None, ids=None):  # noqa: A002 - mirror API
        items = (
            [x for v in self._sheets.values() for x in v]
            if type is None
            else self._sheets.get(type, [])
        )
        if since:
            items = [x for x in items if str(x.get("updatedAt") or "z") > since]
        if ids:
            items = [x for x in items if x.get("id") in set(ids)]
        return items


class FakeConfig:
    def __init__(self, client, since=None):
        self.client = client
        self.since = since


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


def test_nodes_are_typed_prefixed_and_federation_stamped():
    batch = _run()
    by_id = {n.id: n for n in batch.nodes}

    assert by_id["app:a1"].type == "Application"
    assert by_id["app:a1"].props["name"] == "Billing"
    assert by_id["itcomponent:ic1"].type == "ITComponent"
    assert by_id["capability:c1"].type == "BusinessCapability"
    # Custom type discovered from the live metamodel (lowercased prefix).
    assert by_id["datacenter:dc1"].type == "DataCenter"
    # Federation key on every node — required for write-back resolution.
    assert by_id["app:a1"].props["externalToolId"] == "a1"
    assert by_id["app:a1"].props["domain"] == "leanix"
    # 2 Applications + ITComponent + BusinessCapability + DataCenter
    assert len(batch.nodes) == 5


def test_core_and_custom_relations_become_edges():
    # Metamodel-driven: edge names match the generated OWL object-property names
    # (UPPER_SNAKE of the LeanIX relation field) — consistent with ontology_leanix.ttl.
    batch = _run()
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}

    assert (
        "app:a1",
        "capability:c1",
        "REL_APPLICATION_TO_BUSINESS_CAPABILITY",
    ) in triples
    assert ("app:a1", "itcomponent:ic1", "REL_APPLICATION_TO_IT_COMPONENT") in triples
    assert (
        "app:a2",
        "capability:c1",
        "REL_APPLICATION_TO_BUSINESS_CAPABILITY",
    ) in triples
    # Custom relation, target resolved via the embedded factSheet type.
    assert ("app:a1", "datacenter:dc1", "REL_APPLICATION_TO_DATA_CENTER") in triples


def test_fallback_without_metamodel_keeps_core_types():
    class NoMetaClient(FakeLeanIXClient):
        meta_model = None  # type: ignore[assignment]

    batch = extract(FakeConfig(NoMetaClient()))
    by_id = {n.id: n for n in batch.nodes}
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    # Fallback maps mirror the 4 core types with friendly names; no custom type.
    assert by_id["app:a1"].type == "Application"
    assert "datacenter:dc1" not in by_id
    assert ("app:a1", "capability:c1", "SUPPORTS") in triples
    assert ("app:a1", "itcomponent:ic1", "DEPENDS_ON") in triples


def test_source_is_registered():
    src = get_source("leanix")
    assert src is not None
    assert src.extract is extract
    assert "LeanIX" in src.description


def test_write_batch_persists_to_backend():
    batch = _run()
    backend = FakeBackend()
    n, e = write_batch(backend, batch)

    assert n == 5
    assert backend.nodes["app:a1"]["type"] == "Application"
    assert backend.nodes["app:a1"]["externalToolId"] == "a1"
    assert ("app:a1", "capability:c1", "REL_APPLICATION_TO_BUSINESS_CAPABILITY") in (
        backend.edges
    )


def test_empty_client_yields_empty_batch():
    class Empty:
        client = None

    batch = extract(Empty())
    assert batch.category == "leanix"
    assert batch.nodes == [] and batch.edges == []
