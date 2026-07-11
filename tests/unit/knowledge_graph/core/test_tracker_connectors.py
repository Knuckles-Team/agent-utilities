"""First-class Jira/Confluence/Plane delta connectors (KG-2.123/2.124/2.125).

Live-path: drive the real ``_sync_*`` handlers (the seam ``sync_source`` dispatches in
production) with a fake connector, asserting typed-entity mapping, the multi-instance
loop, watermark advance, and delta skip on a re-run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core import source_sync as ss


class _Doc(SimpleNamespace):
    """A minimal SourceDocument-shaped object."""


def _doc(doc_id: str, record: dict[str, Any], updated: str, text: str = "x") -> _Doc:
    return _Doc(
        id=doc_id,
        metadata={"record": record},
        updated_at=updated,
        text=text,
        title=record.get("name") or doc_id,
        source_uri=f"mcp-tool://test/{doc_id}",
    )


class _FakeConn:
    """A connector whose ``poll`` yields one batch then exhausts; honours the
    since-watermark client-side (like the real connector's updated_field filter)."""

    def __init__(self, docs: list[_Doc]):
        self._docs = docs

    def poll(self, checkpoint: Any = None) -> Any:
        since = getattr(checkpoint, "watermark", None)
        docs = [
            d for d in self._docs if since is None or str(d.updated_at) > str(since)
        ]
        return SimpleNamespace(
            documents=docs, checkpoint=SimpleNamespace(has_more=False, watermark=None)
        )


class _FakeBackend:
    def __init__(self) -> None:
        self.watermarks: dict[str, str] = {}

    def execute(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        if "MATCH (n:SourceSyncState" in query and "RETURN" in query:
            wm = self.watermarks.get(params.get("id"))
            return [{"w": wm}] if wm else []
        if "MERGE (n:SourceSyncState" in query:
            self.watermarks[params.get("id")] = params.get("wm")
        return []


class _FakeEngine:
    def __init__(self) -> None:
        self.backend = _FakeBackend()
        self.batches: list[tuple] = []

    def ingest_external_batch(self, domain, entities, relationships=None):
        self.batches.append((domain, entities, relationships))
        return {"status": "ok"}


def _patch_conn(monkeypatch, docs):
    monkeypatch.setattr(ss, "_build_preset_conn", lambda *a, **k: _FakeConn(docs))


def test_jira_sync_maps_typed_entities_and_advances_watermark(monkeypatch):
    """AU-P1-5: ``_sync_jira`` is envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction)
    — one ``ingest_envelope``/``ingest_external_batch`` call per entity (issue/person/
    epic), so ``engine.batches`` now holds one entry per entity instead of a single
    per-run batch. Aggregate across every call; the underlying intent (typed
    entities + has_role/part_of links + watermark advance) is unchanged.
    """
    issue = {
        "fields": {
            "summary": "Fix payment flow",
            "updated": "2026-06-18T10:00:00.000+0000",
            "status": {"name": "In Progress"},
            "assignee": {"accountId": "u1", "displayName": "Luke"},
            "parent": {"key": "DB-1"},
        }
    }
    docs = [_doc("DB-1234", issue, "2026-06-18T10:00:00.000+0000")]
    _patch_conn(monkeypatch, docs)
    engine = _FakeEngine()

    out = ss.sync_source(engine, "jira", mode="delta")
    assert out["status"] == "ok"
    assert engine.batches and all(domain == "jira" for domain, _, _ in engine.batches)
    entities = [e for _domain, es, _rels in engine.batches for e in es]
    rels = [r for _domain, _es, rs in engine.batches for r in rs]
    types = {e["type"] for e in entities}
    assert {"issue", "person", "goal"} <= types
    assert any(r["type"] == "has_role" for r in rels)
    assert any(r["type"] == "part_of" for r in rels)
    # watermark advanced
    assert engine.backend.watermarks.get("sync:jira:jira")

    # Re-run: the watermark filters the (unchanged) doc → nothing re-ingested.
    engine.batches.clear()
    ss.sync_source(engine, "jira", mode="delta")
    assert engine.batches == []


def test_plane_multi_instance_loops_both_servers(monkeypatch):
    """AU-P1-5: ``_sync_plane`` is envelope-native — one ``ingest_envelope`` call per
    entity (issue/project), so ``engine.batches`` holds one entry per entity per
    instance instead of a single per-instance batch. Aggregate across every call.
    """
    # Two Plane instances → the same logic ingests both.
    monkeypatch.setattr(
        ss,
        "_resolve_tracker_instances",
        lambda *a, **k: [
            {"name": "primary", "server": "plane-mcp", "projects": ["p1"]},
            {"name": "secondary", "server": "plane-mcp-b", "projects": ["p9"]},
        ],
    )
    item = {
        "name": "Task A",
        "updated_at": "2026-06-18T09:00:00Z",
        "state": {"name": "Todo"},
    }
    _patch_conn(monkeypatch, [_doc("wi-1", item, "2026-06-18T09:00:00Z")])
    engine = _FakeEngine()

    out = ss.sync_source(engine, "plane", mode="delta")
    assert {r["instance"] for r in out["instances"]} == {"primary", "secondary"}
    # one ingest_external_batch call per entity (issue + project) per instance
    assert len(engine.batches) == 4
    ids = {e["id"] for _, ents, _ in engine.batches for e in ents}
    assert "plane:primary:issue:wi-1" in ids
    assert "plane:secondary:issue:wi-1" in ids


def test_confluence_full_mirror_ingests_pages(monkeypatch):
    page = {"spaceId": "S1", "version": {"number": 3}}
    _patch_conn(
        monkeypatch, [_doc("100", page, "2026-06-18T08:00:00Z", text="# Wiki body")]
    )
    processed: list[dict] = []

    class _Proc:
        def process(self, text, **kw):
            processed.append({"text": text, **kw})

    monkeypatch.setattr(ss, "_confluence_processor", lambda engine: _Proc())
    engine = _FakeEngine()

    out = ss.sync_source(engine, "confluence", mode="delta")
    assert out["pages_ingested"] == 1
    assert processed[0]["doc_type"] == "wiki"
    assert processed[0]["document_id"].startswith("confluence:")


def test_jira_jql_builds_server_side_delta_clause():
    jql = ss._jira_jql({"project_keys": ["DB"]}, "2026-06-18T10:00:00.000+0000", None)
    assert "project in (DB)" in jql
    assert 'updated >= "2026-06-18 10:00"' in jql
    assert jql.endswith("ORDER BY updated DESC")


def test_resolve_instances_falls_back_to_single_host(monkeypatch):
    monkeypatch.setattr(
        ss, "config", SimpleNamespace(jira_instances=None), raising=False
    )
    monkeypatch.setattr(
        "agent_utilities.core.config.config", SimpleNamespace(jira_instances=None)
    )
    monkeypatch.setattr("agent_utilities.core.config.setting", lambda *a, **k: "DB,OPS")
    rows = ss._resolve_tracker_instances(
        "jira_instances",
        default_name="jira",
        default_server="atlassian-mcp",
        scope_key="project_keys",
        scope_setting="JIRA_PROJECT_KEYS",
    )
    assert rows == [
        {"name": "jira", "server": "atlassian-mcp", "project_keys": ["DB", "OPS"]}
    ]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
