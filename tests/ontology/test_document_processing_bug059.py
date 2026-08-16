"""BUG-059 — DocumentProcessor._persist_sections was an unconditional raw
write, unlike its sibling _persist (the chunk-slice path), which already
tries the governed ``_BatchedBackend`` wrapper first (stamps
``stamp_ownership``/``stamp_classification``) and only falls back to the raw
per-item seam when the backend has no bulk RPC path. Per BUG-LEDGER's
disposition, this specific write is dormant today (nothing populates
``result.section_nodes`` on any live ingestion path yet) -- but the shared
``_write_node`` seam it used unconditionally is exactly the same one
``_persist`` falls back to, so made symmetric rather than leaving the two
siblings disagree.

Note: ``_BatchedBackend.add_node`` only stamps when it is actually invoked,
which ``_persist``/``_persist_sections`` both gate on
``batched.bulk_available`` (a ``_graph.batch_update``/``bulk_mutate`` RPC on
the backend) -- a real production backend (EpistemicGraphBackend) always has
this, but the plain ``_FakeWriter`` used by
``tests/ontology/test_document_processing.py`` does not, so it exercises
the SAME ungoverned raw fallback ``_persist`` has always used for a
non-bulk-capable backend (a separate, pre-existing gap in ``_persist``
itself, out of scope for this fix -- see the BUG-059 final report). The
fake writer here adds a bulk RPC so the governed path is actually exercised,
matching what a real engine backend provides.
"""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.knowledge_graph.ontology.document_processing import (
    DocumentProcessor,
)
from agent_utilities.security.brain_context import IdentityRequiredError


class _FakeBulkGraph:
    """Minimal ``_graph`` shape ``_BatchedBackend`` recognizes as bulk-capable."""

    def __init__(self, writer: _FakeBulkWriter) -> None:
        self._writer = writer

    def batch_update(self, ops: list[dict]) -> None:
        for op in ops:
            if op["op"] == "add_node":
                self._writer.nodes[op["id"]] = dict(op["properties"])
            elif op["op"] == "add_edge":
                self._writer.edges.append(
                    (op["source"], op["target"], op["properties"].get("relationship"))
                )


class _FakeBulkWriter:
    """An add_node/add_edge sink that ALSO exposes a bulk RPC (``_graph.
    batch_update``), matching a real engine backend closely enough to
    exercise ``_BatchedBackend``'s governed path rather than its raw
    per-item fallback."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []
        self._graph = _FakeBulkGraph(self)

    def add_node(
        self, node_id, label="", **properties
    ):  # pragma: no cover - unused when bulk_available
        self.nodes[node_id] = {"label": label, **properties}

    def add_edge(self, source, target, rel_type="", **properties):  # pragma: no cover
        self.edges.append((source, target, rel_type, properties))


def test_persist_sections_requires_a_bound_actor_like_its_sibling_persist():
    """Known-bad input: no actor bound anywhere. BEFORE this fix,
    _persist_sections wrote section-tree nodes straight through the raw
    writer regardless of actor state. AFTER, it enters the SAME governed
    _BatchedBackend seam _persist already uses, and refuses."""
    writer = _FakeBulkWriter()
    proc = DocumentProcessor(graph=writer)
    section_nodes = [{"id": "section:1", "node_type": "Section", "title": "Intro"}]
    section_edges: list[dict] = []

    def isolated():
        with pytest.raises(IdentityRequiredError):
            proc._persist_sections(section_nodes, section_edges)

    contextvars.Context().run(isolated)
    assert writer.nodes == {}


def test_persist_sections_stamps_ownership_when_actor_bound():
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    writer = _FakeBulkWriter()
    proc = DocumentProcessor(graph=writer)
    section_nodes = [{"id": "section:1", "node_type": "Section", "title": "Intro"}]
    section_edges = [
        {"source": "section:1", "target": "section:2", "relationship": "HAS_SUBSECTION"}
    ]

    actor = ActorContext(
        actor_id="user:doc-owner",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-sections",
        authenticated=True,
    )
    with use_actor(actor):
        ok = proc._persist_sections(section_nodes, section_edges)

    assert ok is True
    props = writer.nodes["section:1"]
    assert props["_owner_id"] == "user:doc-owner"
    assert props["tenant_id"] == "tenant-sections"
    assert props["classification"] == "confidential"
    assert writer.edges == [("section:1", "section:2", "HAS_SUBSECTION")]
