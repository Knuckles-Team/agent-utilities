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

Uses the same ``_FakeWriter`` shape as ``tests/ontology/test_document_processing.py``.
"""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.knowledge_graph.ontology.document_processing import (
    DocumentProcessor,
)
from agent_utilities.security.brain_context import IdentityRequiredError


class _FakeWriter:
    """In-memory add_node/add_edge sink mirroring the backend write contract."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, label="", **properties):
        self.nodes[node_id] = {"label": label, **properties}

    def add_edge(self, source, target, rel_type="", **properties):
        self.edges.append((source, target, rel_type, properties))


def test_persist_sections_requires_a_bound_actor_like_its_sibling_persist():
    """Known-bad input: no actor bound anywhere. BEFORE this fix,
    _persist_sections wrote section-tree nodes straight through the raw
    writer regardless of actor state. AFTER, it enters the SAME governed
    _BatchedBackend seam _persist already uses, and refuses."""
    writer = _FakeWriter()
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

    writer = _FakeWriter()
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
    assert writer.edges == [
        ("section:1", "section:2", "HAS_SUBSECTION", {})
    ]
