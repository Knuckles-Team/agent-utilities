"""Tests for LeanIX backfeed / write-back (CONCEPT:KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment import leanix_writeback as wb
from agent_utilities.knowledge_graph.enrichment.leanix_writeback import (
    run_leanix_writeback,
)

META_MODEL = {
    "factSheets": {
        "Application": {
            "fields": {},
            "relations": {
                "relApplicationToITComponent": {"targetFactSheetType": "ITComponent"}
            },
        },
        "ITComponent": {"fields": {}, "relations": {}},
    }
}


class FakeClient:
    def __init__(self):
        self.relations: list[tuple[str, str, str]] = []
        self.created: list[tuple[str, str]] = []

    def meta_model(self):
        return META_MODEL

    def create_fact_sheet_relation(self, fs_id, rel_field, target_id):
        self.relations.append((fs_id, rel_field, target_id))
        return {"id": fs_id}

    def create_fact_sheet(self, fs_type, name):
        self.created.append((fs_type, name))
        return {"id": "new-1"}

    def update_fact_sheet(self, fs_id, patches):
        return {"id": fs_id}

    def add_tag(self, fs_id, tag_id):
        return {"id": fs_id}


class FakeBackend:
    """Resolves two KG nodes to LeanIX GUIDs via externalToolId."""

    def execute(self, query, params=None):
        return [
            {"id": "app:a1", "guid": "lx-a1"},
            {"id": "itcomponent:ic1", "guid": "lx-ic1"},
        ]


def test_dry_run_proposes_without_writing(monkeypatch):
    client = FakeClient()
    out = run_leanix_writeback(
        backend=FakeBackend(),
        client=client,
        inferences=[
            {
                "source": "app:a1",
                "rel_type": "REL_APPLICATION_TO_IT_COMPONENT",
                "target": "itcomponent:ic1",
            }
        ],
        dry_run=True,
    )
    assert out["status"] == "completed"
    assert out["dry_run"] is True
    assert out["relations_written"] == 0
    assert client.relations == []  # nothing actually written
    prop = out["proposals"][0]
    assert prop["op"] == "create_relation"
    assert prop["factSheet"] == "lx-a1"
    assert prop["relation"] == "relApplicationToITComponent"
    assert prop["target"] == "lx-ic1"
    assert prop["provenance"] == wb.PROVENANCE_TAG


def test_live_write_refused_without_enable_flag(monkeypatch):
    monkeypatch.setattr(wb, "setting", lambda k, d=None, cast=None: False)
    out = run_leanix_writeback(
        backend=FakeBackend(),
        client=FakeClient(),
        inferences=[
            {
                "source": "app:a1",
                "rel_type": "REL_APPLICATION_TO_IT_COMPONENT",
                "target": "itcomponent:ic1",
            }
        ],
        dry_run=False,
    )
    assert out["status"] == "refused"
    assert "LEANIX_ENABLE_WRITE" in out["reason"]


def test_live_write_when_enabled(monkeypatch):
    monkeypatch.setattr(wb, "setting", lambda k, d=None, cast=None: True)
    client = FakeClient()
    out = run_leanix_writeback(
        backend=FakeBackend(),
        client=client,
        inferences=[
            {
                "source": "app:a1",
                "rel_type": "REL_APPLICATION_TO_IT_COMPONENT",
                "target": "itcomponent:ic1",
            }
        ],
        creations=[{"type": "DataObject", "name": "Ledger"}],
        dry_run=False,
    )
    assert out["status"] == "completed"
    assert out["relations_written"] == 1
    assert out["factsheets_created"] == 1
    assert client.relations == [("lx-a1", "relApplicationToITComponent", "lx-ic1")]
    assert client.created == [("DataObject", "Ledger")]


def test_unresolvable_relation_is_skipped(monkeypatch):
    monkeypatch.setattr(wb, "setting", lambda k, d=None, cast=None: True)
    client = FakeClient()
    out = run_leanix_writeback(
        backend=FakeBackend(),
        client=client,
        # target node not in the resolver → skipped, never written
        inferences=[
            {
                "source": "app:a1",
                "rel_type": "REL_APPLICATION_TO_IT_COMPONENT",
                "target": "app:unknown",
            }
        ],
        dry_run=False,
    )
    assert out["relations_skipped"] == 1
    assert out["relations_written"] == 0


def test_no_client_skips():
    out = run_leanix_writeback(client=None, dry_run=True)
    assert out["status"] == "skipped"
