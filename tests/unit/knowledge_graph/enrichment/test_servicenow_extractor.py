"""ServiceNow ITSM source extractor tests (CONCEPT:KG-2.9).

Uses an injected fake client (no network / no daemon) plus the FakeBackend +
write_batch contract to assert node/edge mapping and persistence.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.servicenow import (
    extract,
)
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


class FakeServiceNowClient:
    """Duck-typed ServiceNow client returning canned dict records."""

    def incidents(self):
        return [
            {
                "sys_id": "abc111",
                "number": "INC0010001",
                "short_description": "Email down",
                "state": "In Progress",
                "priority": "1 - Critical",
                "cmdb_ci": "ci-mail",
                "assigned_to": "alice",
            },
            {
                # no sys_id -> falls back to number; no CI; no assignee
                "number": "INC0010002",
                "short_description": "Printer offline",
                "state": "New",
                "priority": "3 - Moderate",
            },
        ]

    def changes(self):
        return [
            {
                "sys_id": "chg777",
                "number": "CHG0005001",
                "short_description": "Upgrade DB",
                "state": "Scheduled",
                "priority": "2 - High",
                "cmdb_ci": "ci-db",
                "assigned_to": "bob",
            }
        ]

    def cmdb_cis(self):
        return [
            {
                "sys_id": "ci-mail",
                "name": "mail-server-01",
                "short_description": "Primary mail server",
                "state": "Installed",
            },
            {
                "sys_id": "ci-db",
                "name": "db-server-01",
                "short_description": "Primary database",
                "state": "Installed",
            },
        ]


def test_extract_maps_nodes():
    batch = extract({"client": FakeServiceNowClient()})
    by_id = {n.id: n for n in batch.nodes}

    assert batch.category == "servicenow"
    # 2 incidents + 1 change + 2 CIs
    assert len(batch.nodes) == 5

    inc = by_id["incident:abc111"]
    assert inc.type == "Incident"
    assert inc.props["number"] == "INC0010001"
    assert inc.props["short_description"] == "Email down"
    assert inc.props["state"] == "In Progress"
    assert inc.props["priority"] == "1 - Critical"

    # incident without sys_id keyed by number
    assert "incident:INC0010002" in by_id

    chg = by_id["change:chg777"]
    assert chg.type == "Change"
    assert chg.props["number"] == "CHG0005001"

    ci = by_id["ci:ci-mail"]
    assert ci.type == "ConfigurationItem"
    assert ci.props["name"] == "mail-server-01"


def test_extract_maps_edges():
    batch = extract({"client": FakeServiceNowClient()})
    edges = {(e.source, e.target, e.rel_type) for e in batch.edges}

    assert ("incident:abc111", "ci:ci-mail", "AFFECTS") in edges
    assert ("incident:abc111", "person:alice", "ASSIGNED_TO") in edges
    assert ("change:chg777", "ci:ci-db", "AFFECTS") in edges
    assert ("change:chg777", "person:bob", "ASSIGNED_TO") in edges
    # incident with no CI / assignee produced no extra edges
    assert len(batch.edges) == 4


def test_extract_no_client_is_tolerant():
    batch = extract({})
    assert batch.category == "servicenow"
    assert batch.nodes == []
    assert batch.edges == []


def test_source_is_registered():
    src = get_source("servicenow")
    assert src is not None
    assert src.extract is extract
    assert "ServiceNow" in src.description


def test_write_batch_persists():
    batch = extract({"client": FakeServiceNowClient()})
    backend = FakeBackend()
    n, e = write_batch(backend, batch)

    assert n == 5
    assert e == 4
    assert backend.nodes["incident:abc111"]["type"] == "Incident"
    assert backend.nodes["ci:ci-mail"]["name"] == "mail-server-01"
    assert ("incident:abc111", "ci:ci-mail", "AFFECTS") in backend.edges
    assert ("change:chg777", "person:bob", "ASSIGNED_TO") in backend.edges
