"""Tests for the infra source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

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
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend

SAMPLE_INVENTORY = {
    "all": {
        "hosts": {
            "analysis-node-a": {
                "ansible_host": "192.0.2.13",
                "roles": ["manager"],
                "groups": ["swarm"],
            },
            "worker-node-b": {
                "ip": "192.0.2.14",
                "role": "worker",
                "groups": "swarm",
            },
        }
    }
}

SAMPLE_SERVICES = [
    {
        "name": "graph-store",
        "image": "graph-store:latest",
        "replicas": 1,
        "node": "analysis-node-a",
    },
    {
        "name": "event-broker",
        "image": "event-broker:stable",
        "replicas": 3,
        "host": "worker-node-b",
    },
    {"name": "floating", "image": "nginx", "replicas": 2},  # no node -> no edge
]


def test_extract_servers_services_and_edges():
    batch = extract({"inventory": SAMPLE_INVENTORY, "services": SAMPLE_SERVICES})
    assert batch.category == "infra"

    by_id = {n.id: n for n in batch.nodes}
    # Servers
    assert by_id["server:analysis-node-a"].type == "Server"
    assert by_id["server:analysis-node-a"].props["hostname"] == "analysis-node-a"
    assert by_id["server:analysis-node-a"].props["ip"] == "192.0.2.13"
    assert by_id["server:analysis-node-a"].props["roles"] == ["manager"]
    assert by_id["server:worker-node-b"].props["ip"] == "192.0.2.14"
    assert by_id["server:worker-node-b"].props["roles"] == ["worker"]
    assert by_id["server:worker-node-b"].props["groups"] == ["swarm"]

    # Services
    assert by_id["service:graph-store"].type == "Service"
    assert by_id["service:graph-store"].props["image"] == "graph-store:latest"
    assert by_id["service:graph-store"].props["replicas"] == 1

    # RUNS_ON edges (only services naming a node)
    rels = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("service:graph-store", "server:analysis-node-a", "RUNS_ON") in rels
    assert ("service:event-broker", "server:worker-node-b", "RUNS_ON") in rels
    assert all(e.source != "service:floating" for e in batch.edges)


def test_flat_inventory_shape():
    batch = extract({"inventory": {"node1": {"ip": "198.51.100.5"}}})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["server:node1"].props["ip"] == "198.51.100.5"


def test_extract_from_yaml_file(tmp_path):
    inv = tmp_path / "inventory.yaml"
    inv.write_text(
        "all:\n"
        "  hosts:\n"
        "    analysis-node-a:\n"
        "      ansible_host: 192.0.2.13\n"
        "      roles: [manager]\n",
        encoding="utf-8",
    )
    batch = extract(
        {
            "inventory": str(inv),
            "services": [
                {
                    "name": "graph-store",
                    "image": "graph-store",
                    "replicas": 1,
                    "node": "analysis-node-a",
                }
            ],
        }
    )
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["server:analysis-node-a"].props["ip"] == "192.0.2.13"
    assert any(
        e.source == "service:graph-store"
        and e.target == "server:analysis-node-a"
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
    assert backend.nodes["server:analysis-node-a"]["node_type"] == "Server"
    assert backend.nodes["server:analysis-node-a"]["hostname"] == "analysis-node-a"
    assert backend.nodes["service:graph-store"]["node_type"] == "Service"
    assert (
        "service:graph-store",
        "server:analysis-node-a",
        "RUNS_ON",
    ) in backend.edges
