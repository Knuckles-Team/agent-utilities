"""Tests for the Grafana observability source extractor (CONCEPT:KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.grafana import (
    extract,
)
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)


class FakeGrafanaClient:
    """Duck-typed Grafana client returning fixed observability objects."""

    def dashboards(self):
        return [
            {
                "uid": "dash1",
                "title": "Cluster Overview",
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU service=pggraph",
                        "targets": [{"expr": "rate(cpu[5m])"}],
                    },
                    {
                        "id": 2,
                        "title": "Memory",
                        "targets": [],
                        "labels": {"service": "kafka"},
                    },
                ],
            }
        ]

    def alert_rules(self):
        return [
            {
                "uid": "alertA",
                "title": "High latency service=epistemic-graph",
                "condition": "B > 0.5",
                "labels": {"severity": "page"},
            }
        ]

    def datasources(self):
        return [{"uid": "ds1", "name": "Prometheus", "type": "prometheus"}]


from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


def test_extract_produces_nodes_and_edges():
    batch = extract({"client": FakeGrafanaClient()})
    by_id = {n.id: n for n in batch.nodes}

    # Dashboard
    assert "dashboard:dash1" in by_id
    assert by_id["dashboard:dash1"].type == "Dashboard"
    assert by_id["dashboard:dash1"].props["title"] == "Cluster Overview"

    # Panels
    assert "panel:dash1:1" in by_id
    assert "panel:dash1:2" in by_id
    assert by_id["panel:dash1:1"].type == "Panel"
    assert by_id["panel:dash1:1"].props["targets"] == [{"expr": "rate(cpu[5m])"}]

    # Alert + DataSource
    assert by_id["alert:alertA"].type == "Alert"
    assert by_id["datasource:ds1"].type == "DataSource"
    assert by_id["datasource:ds1"].props["ds_type"] == "prometheus"

    # PART_OF edges (one per panel)
    rels = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("panel:dash1:1", "dashboard:dash1", "PART_OF") in rels
    assert ("panel:dash1:2", "dashboard:dash1", "PART_OF") in rels

    # MONITORS edges: title marker + labels + alert title marker
    assert ("panel:dash1:1", "service:pggraph", "MONITORS") in rels
    assert ("panel:dash1:2", "service:kafka", "MONITORS") in rels
    assert ("alert:alertA", "service:epistemic-graph", "MONITORS") in rels


def test_source_is_registered():
    src = get_source("grafana")
    assert src is not None
    assert src.extract is extract
    assert "Grafana" in src.description


def test_write_batch_persists_to_backend():
    batch = extract({"client": FakeGrafanaClient()})
    backend = FakeBackend()
    n, e = write_batch(backend, batch)

    assert n == len(batch.nodes)
    assert e == len(batch.edges)
    assert backend.nodes["dashboard:dash1"]["type"] == "Dashboard"
    assert ("panel:dash1:1", "dashboard:dash1", "PART_OF") in backend.edges
    assert ("panel:dash1:1", "service:pggraph", "MONITORS") in backend.edges


def test_missing_client_is_tolerated():
    batch = extract({})
    assert batch.category == "grafana"
    assert batch.nodes == []
    assert batch.edges == []
