"""Phase-1 ops/observability connectors: extract + write (CONCEPT:KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import kafka as kafka_ext
from agent_utilities.knowledge_graph.enrichment.extractors import lgtm as lgtm_ext
from agent_utilities.knowledge_graph.enrichment.extractors import portainer as pf_ext
from agent_utilities.knowledge_graph.enrichment.extractors import (
    technitium_dns as dns_ext,
)
from agent_utilities.knowledge_graph.enrichment.writeback import (
    core,
    run_writeback,
)


# ── Extractors ───────────────────────────────────────────────────────────────
class FakeDNS:
    def list_zones(self):
        return {"zones": [{"name": "example.com"}]}

    def get_records(self, zone):
        return {
            "records": [{"name": "www", "type": "A", "rData": {"value": "10.0.0.1"}}]
        }


def test_dns_extract():
    batch = dns_ext.extract({"client": FakeDNS()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["dnszone:example.com"].props["ci_class"] == "dns_zone"
    assert by_id["dnsrecord:example.com:www:A"].props["record_type"] == "A"
    assert by_id["dnsrecord:example.com:www:A"].props["domain"] == "technitium_dns"
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("dnszone:example.com", "dnsrecord:example.com:www:A", "CONTAINS") in triples


def test_kafka_extract():
    class FakeKafka:
        def list_topics(self):
            return ["orders", "events"]

        def list_consumer_groups(self):
            return [{"group_id": "billing"}]

    batch = kafka_ext.extract({"client": FakeKafka()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["kafka_topic:orders"].type == "Topic"
    assert by_id["kafka_group:billing"].type == "Service"


def test_portainer_extract():
    class FakePortainer:
        def get_endpoints(self):
            return [{"Id": 1, "Name": "local"}]

        def list_containers(self, eid):
            return [
                {"Id": "abc", "Names": ["/web"], "Image": "nginx", "State": "running"}
            ]

        def list_stacks(self):
            return [{"Id": 5, "Name": "blog"}]

    batch = pf_ext.extract({"client": FakePortainer()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["portainer_endpoint:1"].type == "Server"
    assert by_id["portainer_stack:5"].type == "Service"
    assert by_id["portainer_container:1:abc"].type == "AssetInstance"
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("portainer_container:1:abc", "portainer_endpoint:1", "RUNS_ON") in triples


def test_lgtm_extract():
    class FakeLgtm:
        def get_dashboards(self):
            return [{"uid": "d1", "title": "Overview"}]

        def get_alerts(self):
            return [
                {
                    "fingerprint": "f1",
                    "labels": {"alertname": "HighCPU"},
                    "status": "firing",
                }
            ]

        def list_datasources(self):
            return [{"uid": "ds1", "name": "Prom", "type": "prometheus"}]

    batch = lgtm_ext.extract({"client": FakeLgtm()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["lgtm_dash:d1"].type == "Dashboard"
    assert by_id["lgtm_alert:f1"].type == "Alert"
    assert by_id["lgtm_ds:ds1"].type == "DataSource"


# ── Sinks ────────────────────────────────────────────────────────────────────
class FakeDNSWrite:
    def __init__(self):
        self.added = []

    def add_record(self, zone, name, rtype, value):
        self.added.append((zone, name, rtype, value))


def test_dns_sink_standard_live(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeDNSWrite()
    out = run_writeback(
        "technitium_dns",
        client=client,
        creations=[
            {"zone": "example.com", "name": "api", "type": "A", "value": "10.0.0.2"}
        ],
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.added == [("example.com", "api", "A", "10.0.0.2")]


def test_kafka_high_stakes_is_queued(monkeypatch, tiny_engine):
    # tiny_engine (KG-2.238): the engine-only approval queue persists on the REAL
    # ephemeral engine — no JSON fallback (CONCEPT:KG-2.247).
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)

    class FakeKafkaW:
        def __init__(self):
            self.produced = []

        def produce(self, topic, value):
            self.produced.append((topic, value))

    client = FakeKafkaW()
    out = run_writeback(
        "kafka",
        client=client,
        creations=[{"topic": "orders", "value": "x"}],
        dry_run=False,
    )
    assert out["status"] == "queued"  # high_stakes never auto-executes
    assert client.produced == []


def test_portainer_high_stakes_is_queued(monkeypatch, tiny_engine):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    out = run_writeback(
        "portainer", client=object(), creations=[{"name": "newstack"}], dry_run=False
    )
    assert out["status"] == "queued"
