"""Claude Code file-based memory → KG ingestion (CONCEPT:AU-KG.ingest.claude-memory-connector).

Covers the offline, dependency-free handler: frontmatter parsing, ``[[link]]`` →
RELATED_TO edges, MEMORY.md index skipping, ``ids`` narrowing, and the empty-dir skip.

AU-P1-5: the handler is now envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction)
— each topic file is ONE ``ChangeEnvelope`` routed through
:func:`~agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope`,
which commits ONLY through the real native ``ApplyChangeEnvelope`` authority —
there is no ``engine.ingest_external_batch`` fallback anymore. The typed-node
assertions below therefore run against a real, isolated ``engine_graph`` (the
session's real epistemic-graph engine) and read back what was actually
committed, rather than replaying a call log a fake engine recorded.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.source_sync import (
    _parse_memory_file,
    _sync_claude_memory,
)


class _NoOpBackend:
    """A minimal real backend stand-in: every query is a harmless no-op, so the

    envelope-native lineage/checkpoint/watermark bookkeeping in ``ingest_envelope``
    completes cleanly instead of raising against a bare ``object()``.
    """

    def execute(self, query: str, params: dict[str, Any] | None = None) -> list:
        return []


class _FakeEngine:
    def __init__(self) -> None:
        self.backend = _NoOpBackend()
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


@pytest.mark.engine
def test_sync_claude_memory_ingests_typed_nodes_and_links(
    tmp_path, monkeypatch, engine_graph
) -> None:
    """AU-P1-5 (envelope-native): ``ingest_envelope`` commits ONLY through the
    real native ``ApplyChangeEnvelope`` authority (``_resolve_native_authority``,
    ``envelope_ingest.py``) -- there is no legacy per-node
    ``engine.ingest_external_batch`` fallback anymore (D-TC-3's own root cause,
    matching the analogous fail-closed hardening already documented for
    ``failure_analyzer.file_gap_topic``/``_commit_graph_slice``). A hand-built
    ``_FakeEngine`` recording calls to ``ingest_external_batch`` can never
    satisfy that native authority and always fails with
    ``NativeChangeEnvelopeUnavailable`` -- this now drives the real
    ``engine_graph`` fixture (a genuine isolated tenant graph on the session's
    real epistemic-graph engine, CONCEPT:AU-KG.memory.provides-real-ephemeral-one)
    and asserts against the graph it actually wrote, not a call log.
    """
    _write(tmp_path, "foo", "Foo", "project", "links to [[bar]].")
    _write(tmp_path, "bar", "Bar", "reference", "no links here.")
    # The MEMORY.md / MEMORY-ARCHIVE.md indexes must NOT be ingested.
    (tmp_path / "MEMORY.md").write_text("- [Foo](foo.md) — hook\n", encoding="utf-8")
    (tmp_path / "MEMORY-ARCHIVE.md").write_text(
        "- [old](old.md) — x\n", encoding="utf-8"
    )
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path))

    res = _sync_claude_memory(engine_graph, mode="full", ids=None, client=None)

    assert res["status"] == "ok"
    assert res["memories_seen"] == 2  # foo + bar, indexes skipped
    assert res["failed"] == 0

    foo_props = engine_graph._get_node_properties("claude_memory:foo")
    bar_props = engine_graph._get_node_properties("claude_memory:bar")
    assert foo_props, "claude_memory:foo was not committed to the real graph"
    assert bar_props, "claude_memory:bar was not committed to the real graph"
    # Envelope-native connector records keep their raw `type` key verbatim
    # (ChangeEnvelope.from_connector_record's documented, intentional
    # round-trip shape -- `_native_material`'s upsert branch copies `row` as
    # given, unlike RegistryNode.to_graph_properties()'s node_type rename,
    # since this path writes AddNode/properties_msgpack directly and never
    # passes through the add_node() wrapper that rejects a stray `type` key).
    # Confirmed empirically against the real engine.
    assert foo_props.get("type") == "AgentMemory"
    assert bar_props.get("type") == "AgentMemory"
    assert foo_props.get("memory_type") == "project"
    assert foo_props.get("description") == "Foo summary line"
    # the [[bar]] link → a RELATED_TO edge foo → bar
    assert engine_graph.has_edge("claude_memory:foo", "claude_memory:bar")


@pytest.mark.engine
def test_ids_narrows_to_slugs(tmp_path, monkeypatch, engine_graph) -> None:
    _write(tmp_path, "foo", "Foo", "project", "x")
    _write(tmp_path, "bar", "Bar", "project", "y")
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path))
    _sync_claude_memory(engine_graph, mode="delta", ids=["foo"], client=None)
    assert engine_graph._get_node_properties("claude_memory:foo")
    assert not engine_graph._get_node_properties("claude_memory:bar")


def test_empty_dir_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path))
    engine = _FakeEngine()
    res = _sync_claude_memory(engine, mode="full", ids=None, client=None)
    assert res["status"] == "skipped"
    assert not engine.calls
