"""Salesforce / Ansible / Home Assistant: extract + write-back (CONCEPT:AU-KG.ingest.enterprise-source-extractor)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import ansible as ans_ext
from agent_utilities.knowledge_graph.enrichment.extractors import (
    home_assistant as ha_ext,
)
from agent_utilities.knowledge_graph.enrichment.extractors import salesforce as sf_ext
from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


# ── Salesforce ───────────────────────────────────────────────────────────────
class FakeSF:
    def __init__(self):
        self.created = []

    def query(self, soql):
        if "FROM Account" in soql:
            return {"records": [{"Id": "001", "Name": "Acme"}]}
        if "FROM Contact" in soql:
            return {"records": [{"Id": "003", "Name": "Ada", "AccountId": "001"}]}
        if "FROM Opportunity" in soql:
            return {"records": [{"Id": "006", "Name": "Big deal", "AccountId": "001"}]}
        return {"records": []}

    def create_record(self, sobject, data):
        self.created.append((sobject, data.get("Name")))


def test_salesforce_extract_and_write(monkeypatch):
    batch = sf_ext.extract({"client": FakeSF()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["sfaccount:001"].type == "Customer"
    assert by_id["sfcontact:003"].type == "Person"
    assert by_id["sfopp:006"].type == "SalesOrder"
    assert by_id["sfaccount:001"].props["domain"] == "salesforce"
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("sfopp:006", "sfaccount:001", "PLACED_BY") in triples

    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeSF()
    out = run_writeback(
        "salesforce",
        client=client,
        creations=[{"type": "Customer", "name": "Globex"}],
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.created == [("Account", "Globex")]


# ── Home Assistant ───────────────────────────────────────────────────────────
class FakeHA:
    def __init__(self):
        self.calls = []

    def get_states(self):
        return [
            {
                "entity_id": "light.kitchen",
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Light"},
            }
        ]

    def call_service(self, domain, service, data=None, return_response=False):
        self.calls.append((domain, service, data))
        return {}


def test_home_assistant_extract_and_control(monkeypatch):
    batch = ha_ext.extract({"client": FakeHA()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["ha:light.kitchen"].type == "ConfigurationItem"
    assert by_id["ha:light.kitchen"].props["ci_class"] == "light"
    assert by_id["ha:light.kitchen"].props["domain"] == "homeassistant"

    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeHA()
    out = run_writeback(
        "homeassistant",
        client=client,
        creations=[
            {
                "domain": "light",
                "service": "turn_off",
                "data": {"entity_id": "light.kitchen"},
            }
        ],
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.calls == [("light", "turn_off", {"entity_id": "light.kitchen"})]


# ── Ansible Tower ────────────────────────────────────────────────────────────
class FakeAnsible:
    def __init__(self):
        self.launched = []

    def list_hosts(self):
        return [{"id": 7, "name": "web01", "enabled": True}]

    def launch_job(self, template, extra_vars):
        self.launched.append((template, extra_vars))


def test_ansible_extract_and_launch(monkeypatch):
    batch = ans_ext.extract({"client": FakeAnsible()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["ansible_host:7"].type == "Server"
    assert by_id["ansible_host:7"].props["domain"] == "ansible"

    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeAnsible()
    out = run_writeback(
        "ansible",
        client=client,
        creations=[{"template_id": 42, "extra_vars": {"host": "web01"}}],
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.launched == [(42, {"host": "web01"})]


def test_batch_d_refused_without_flag(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    for target, creations in (
        ("salesforce", [{"type": "Customer", "name": "x"}]),
        ("homeassistant", [{"domain": "light", "service": "turn_on"}]),
        ("ansible", [{"template_id": 1}]),
    ):
        out = run_writeback(target, client=object(), creations=creations, dry_run=False)
        assert out["status"] == "refused", target
