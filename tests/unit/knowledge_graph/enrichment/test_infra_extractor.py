"""Tests for the infra source extractor (CONCEPT:KG-2.9).

Verifies inventory hosts → Server nodes, Docker services → Service nodes, and
RUNS_ON edges, plus self-registration and FakeBackend persistence (mirroring the
patterns in test_registry.py). Pure/deterministic — no daemon, no network.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.infra import extract
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)


class FakeBackend:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, s, t, **props):
        self.edges.append((s, t, props.get("rel_type")))


SAMPLE_INVENTORY = {
    "all": {
        "hosts": {
            "r820": {"ansible_host": "10.0.0.13", "roles": ["manager"],
                     "groups": ["swarm"]},
            "rw710": {"ip": "10.0.0.14", "role": "worker", "groups": "swarm"},
        }
    }
}

SAMPLE_SERVICES = [
    {"name": "pggraph", "image": "pggraph:latest", "replicas": 1, "node": "r820"},
    {"name": "kafka", "image": "kafka:3.7", "replicas": 3, "host": "rw710"},
    {"name": "floating", "image": "nginx", "replicas": 2},  # no node -> no edge
]


def test_extract_servers_services_and_edges():
    batch = extract({"inventory": SAMPLE_INVENTORY, "services": SAMPLE_SERVICES})
    assert batch.category == "infra"

    by_id = {n.id: n for n in batch.nodes}
    # Servers
    assert by_id["server:r820"].type == "Server"
    assert by_id["server:r820"].props["hostname"] == "r820"
    assert by_id["server:r820"].props["ip"] == "10.0.0.13"
    assert by_id["server:r820"].props["roles"] == ["manager"]
    assert by_id["server:rw710"].props["ip"] == "10.0.0.14"
    assert by_id["server:rw710"].props["roles"] == ["worker"]
    assert by_id["server:rw710"].props["groups"] == ["swarm"]

    # Services
    assert by_id["service:pggraph"].type == "Service"
    assert by_id["service:pggraph"].props["image"] == "pggraph:latest"
    assert by_id["service:pggraph"].props["replicas"] == 1

    # RUNS_ON edges (only services naming a node)
    rels = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("service:pggraph", "server:r820", "RUNS_ON") in rels
    assert ("service:kafka", "server:rw710", "RUNS_ON") in rels
    assert all(e.source != "service:floating" for e in batch.edges)


def test_flat_inventory_shape():
    batch = extract({"inventory": {"node1": {"ip": "192.168.1.5"}}})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["server:node1"].props["ip"] == "192.168.1.5"


def test_extract_from_yaml_file(tmp_path):
    inv = tmp_path / "inventory.yaml"
    inv.write_text(
        "all:\n"
        "  hosts:\n"
        "    r820:\n"
        "      ansible_host: 10.0.0.13\n"
        "      roles: [manager]\n",
        encoding="utf-8",
    )
    batch = extract(
        {
            "inventory": str(inv),
            "services": [{"name": "pggraph", "image": "pggraph", "replicas": 1,
                          "node": "r820"}],
        }
    )
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["server:r820"].props["ip"] == "10.0.0.13"
    assert any(
        e.source == "service:pggraph" and e.target == "server:r820"
        and e.rel_type == "RUNS_ON"
        for e in batch.edges
    )


def test_self_registration():
    src = get_source("infra")
    assert src is not None
    assert src.extract is extract
    assert src.description == "tunnel-manager inventory + Docker services → KG"


def test_write_batch_persists_via_fake_backend():
    batch = extract({"inventory": SAMPLE_INVENTORY, "services": SAMPLE_SERVICES})
    backend = FakeBackend()
    n, e = write_batch(backend, batch)

    assert n == len(batch.nodes)
    assert e == len(batch.edges)
    assert backend.nodes["server:r820"]["type"] == "Server"
    assert backend.nodes["server:r820"]["hostname"] == "r820"
    assert backend.nodes["service:pggraph"]["type"] == "Service"
    assert ("service:pggraph", "server:r820", "RUNS_ON") in backend.edges
