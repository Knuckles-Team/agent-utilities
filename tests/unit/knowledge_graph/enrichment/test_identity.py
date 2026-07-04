"""Okta + Keycloak identity: extract + write-back (CONCEPT:AU-KG.ingest.enterprise-source-extractor)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import keycloak as kc_ext
from agent_utilities.knowledge_graph.enrichment.extractors import okta as okta_ext
from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


class FakeOkta:
    def list_users(self):
        return [
            {
                "id": "u1",
                "status": "ACTIVE",
                "profile": {"login": "ada@x.io", "email": "ada@x.io"},
            }
        ]

    def list_groups(self):
        return [{"id": "g1", "profile": {"name": "Engineers"}}]

    def list_group_members(self, gid):
        return [{"id": "u1"}]

    def list_apps(self):
        return [{"id": "a1", "label": "Billing", "status": "ACTIVE"}]

    def __init__(self):
        self.assigned = []

    def assign_user_to_app(self, app, user):
        self.assigned.append((app, user))


def test_okta_extract():
    batch = okta_ext.extract({"client": FakeOkta()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["okta_user:u1"].type == "IdentityUser"
    assert by_id["okta_user:u1"].props["domain"] == "okta"
    assert by_id["okta_group:g1"].type == "IdentityGroup"
    assert by_id["okta_app:a1"].type == "Application"
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("okta_user:u1", "okta_group:g1", "MEMBER_OF_GROUP") in triples


class FakeKeycloak:
    def list_users(self, realm):
        return [{"id": "k1", "username": "grace", "email": "g@x.io", "enabled": True}]

    def list_clients(self, realm):
        return [{"id": "c1", "clientId": "portal"}]


def test_keycloak_extract():
    batch = kc_ext.extract({"client": FakeKeycloak(), "realm": "corp"})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["kc_user:k1"].type == "IdentityUser"
    assert by_id["kc_user:k1"].props["realm"] == "corp"
    assert by_id["kc_client:c1"].type == "Application"


class _Backend:
    def execute(self, query, params=None):
        if "externalToolId" in query:
            return [
                {"id": "okta_user:u1", "guid": "u1"},
                {"id": "okta_app:a1", "guid": "a1"},
            ]
        return []


def test_okta_writeback_assign_app(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeOkta()
    out = run_writeback(
        "okta",
        backend=_Backend(),
        client=client,
        inferences=[
            {
                "source": "okta_user:u1",
                "rel_type": "ASSIGNED_APP",
                "target": "okta_app:a1",
            }
        ],
        dry_run=False,
    )
    assert out["relations_written"] == 1
    assert client.assigned == [("a1", "u1")]


def test_okta_refused_without_flag(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "okta",
        client=FakeOkta(),
        creations=[{"type": "IdentityUser", "name": "x"}],
        dry_run=False,
    )
    assert out["status"] == "refused"
