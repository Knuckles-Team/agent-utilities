"""Claude Code file-based memory → KG ingestion (CONCEPT:AU-KG.ingest.claude-memory-connector).

Covers the offline, dependency-free handler: frontmatter parsing, ``[[link]]`` →
RELATED_TO edges, MEMORY.md index skipping, ``ids`` narrowing, and the empty-dir skip.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.core.source_sync import (
    _parse_memory_file,
    _sync_claude_memory,
)


class _FakeEngine:
    backend = object()

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict], list[dict]]] = []

    def ingest_external_batch(
        self, domain: str, entities: list[dict], relationships: list[dict] | None = None
    ) -> dict[str, Any]:
        rels = relationships or []
        self.calls.append((domain, entities, rels))
        return {
            "status": "success",
            "nodes": len(entities),
            "edges": len(rels),
            "skipped_unchanged": 0,
        }


def _write(dir_, slug: str, name: str, mtype: str, body: str) -> None:
    (dir_ / f"{slug}.md").write_text(
        f"---\nname: {name}\ndescription: {name} summary line\nmetadata:\n"
        f"  type: {mtype}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_parse_memory_file(tmp_path):
    _write(tmp_path, "foo", "Foo Memory", "project", "Body about [[bar]] and [[baz]].")
    slug, name, desc, mtype, body, links = _parse_memory_file(tmp_path / "foo.md")
    assert slug == "foo"
    assert name == "Foo Memory"
    assert desc == "Foo Memory summary line"
    assert mtype == "project"
    assert "Body about" in body
    assert links == ["bar", "baz"]


def test_sync_claude_memory_ingests_typed_nodes_and_links(tmp_path, monkeypatch):
    _write(tmp_path, "foo", "Foo", "project", "links to [[bar]].")
    _write(tmp_path, "bar", "Bar", "reference", "no links here.")
    # The MEMORY.md / MEMORY-ARCHIVE.md indexes must NOT be ingested.
    (tmp_path / "MEMORY.md").write_text("- [Foo](foo.md) — hook\n", encoding="utf-8")
    (tmp_path / "MEMORY-ARCHIVE.md").write_text("- [old](old.md) — x\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path))

    engine = _FakeEngine()
    res = _sync_claude_memory(engine, mode="full", ids=None, client=None)

    assert res["status"] == "ok"
    assert res["memories_seen"] == 2  # foo + bar, indexes skipped
    domain, entities, rels = engine.calls[0]
    assert domain == "claude_memory"
    assert {e["id"] for e in entities} == {"claude_memory:foo", "claude_memory:bar"}
    assert all(e["type"] == "AgentMemory" for e in entities)
    foo = next(e for e in entities if e["id"] == "claude_memory:foo")
    assert foo["memory_type"] == "project"
    assert foo["description"] == "Foo summary line"
    # the [[bar]] link → a RELATED_TO edge foo → bar
    assert {"source": "claude_memory:foo", "target": "claude_memory:bar", "type": "RELATED_TO"} in rels


def test_ids_narrows_to_slugs(tmp_path, monkeypatch):
    _write(tmp_path, "foo", "Foo", "project", "x")
    _write(tmp_path, "bar", "Bar", "project", "y")
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path))
    engine = _FakeEngine()
    _sync_claude_memory(engine, mode="delta", ids=["foo"], client=None)
    _, entities, _ = engine.calls[0]
    assert [e["id"] for e in entities] == ["claude_memory:foo"]


def test_empty_dir_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path))
    engine = _FakeEngine()
    res = _sync_claude_memory(engine, mode="full", ids=None, client=None)
    assert res["status"] == "skipped"
    assert not engine.calls
