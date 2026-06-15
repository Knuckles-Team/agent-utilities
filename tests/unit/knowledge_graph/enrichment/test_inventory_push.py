"""Cross-source inventory push tests (CONCEPT:KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.writeback import core
from agent_utilities.knowledge_graph.enrichment.writeback.inventory import (
    collect_inventory_creations,
    push_inventory,
)


class FakeBackend:
    """Returns inventory candidates; no ALIGNED_WITH rows."""

    def execute(self, query, params=None):
        if "ALIGNED_WITH" in query:
            return []
        if "n.type AS type" in query:
            return [
                {"type": "Server", "name": "host01", "id": "server:host01"},
                {"type": "ITComponent", "name": "PostgreSQL", "id": "itcomponent:pg"},
                {
                    "type": "Person",
                    "name": "alice",
                    "id": "person:alice",
                },  # not inventory
            ]
        return []


class FakeSnowApi:
    def __init__(self):
        self.created = []

    def create_cmdb_instance(self, className, attributes, source):
        self.created.append((className, attributes.get("name")))


def test_collect_filters_to_inventory_types():
    creations = collect_inventory_creations(FakeBackend(), "servicenow")
    names = {c["name"] for c in creations}
    assert names == {"host01", "PostgreSQL"}  # Person excluded
    types = {c["type"] for c in creations}
    assert types == {"Server", "ITComponent"}


def test_push_inventory_dry_run():
    out = push_inventory("servicenow", backend=FakeBackend(), dry_run=True)
    assert out["status"] == "completed"
    assert out["inventory_candidates"] == 2
    ops = {p["op"] for p in out["proposals"]}
    assert ops == {"create_cmdb_instance"}


def test_push_inventory_live(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    api = FakeSnowApi()
    out = push_inventory(
        "servicenow", backend=FakeBackend(), engine=None, dry_run=False
    )
    # No client passed → sink resolves get_client() which is absent here → skipped,
    # but candidates were still collected. (Live create is covered with an injected
    # client in test_cmdb_bidirectional.)
    assert out["inventory_candidates"] == 2
