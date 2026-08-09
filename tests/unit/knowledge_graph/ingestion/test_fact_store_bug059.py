"""BUG-059 — IngestionEngine._fact_store's ``_Store.add_node`` writes straight
to the raw backend (``writer`` or ``self.backend``), never through
``IntelligenceGraphEngine._upsert_node``/``GraphComputeEngine.add_node``, so
extracted-fact ``Entity`` nodes from this seam skipped ``stamp_ownership``/
``stamp_classification`` regardless of actor state. Now routed: stamped
locally before delegating to the raw backend, same pattern as
``enrichment.pipeline._BatchedBackend.add_node``.
"""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine
from agent_utilities.security.brain_context import IdentityRequiredError


class _FakeBackend:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, label="", **props):
        self.nodes[node_id] = {"label": label, **props}

    def add_edge(self, source, target, rel_type="", **props):
        self.edges.append((source, target, rel_type, props))


def test_fact_store_add_node_requires_a_bound_actor():
    """Known-bad input: no actor bound anywhere. BEFORE this fix, an
    extracted-fact Entity node landed unowned unconditionally. AFTER, it
    raises."""
    backend = _FakeBackend()
    engine = IngestionEngine(kg_engine=None, backend=backend)
    store = engine._fact_store()

    def isolated():
        with pytest.raises(IdentityRequiredError):
            store.add_node("entity:acme-corp", "Acme Corp")

    contextvars.Context().run(isolated)
    assert backend.nodes == {}


def test_fact_store_add_node_stamps_ownership_when_actor_bound():
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    backend = _FakeBackend()
    engine = IngestionEngine(kg_engine=None, backend=backend)
    store = engine._fact_store()

    actor = ActorContext(
        actor_id="user:ingest-caller",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-facts",
        authenticated=True,
    )
    with use_actor(actor):
        store.add_node("entity:acme-corp", "Acme Corp")

    props = backend.nodes["entity:acme-corp"]
    assert props["label"] == "Entity"
    assert props["name"] == "Acme Corp"
    assert props["_owner_id"] == "user:ingest-caller"
    assert props["tenant_id"] == "tenant-facts"
    assert props["classification"] == "confidential"


def test_fact_store_add_node_respects_an_explicit_writer_override():
    """``_fact_store(writer=...)`` targets an explicit routed backend instead
    of ``self.backend`` — the stamping applies there too."""
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    default_backend = _FakeBackend()
    routed_backend = _FakeBackend()
    engine = IngestionEngine(kg_engine=None, backend=default_backend)
    store = engine._fact_store(writer=routed_backend)

    actor = ActorContext(
        actor_id="user:routed",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-routed",
        authenticated=True,
    )
    with use_actor(actor):
        store.add_node("entity:x", "X")

    assert "entity:x" in routed_backend.nodes
    assert default_backend.nodes == {}
