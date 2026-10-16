"""Shared, engine-independent test doubles for the retrieval/context tests.

Deliberately imports none of the real retrieval/compiler production modules
(``agent_utilities.knowledge_graph.retrieval.context_compiler`` and anything
reaching it) so that test modules needing only these fakes can collect
without the compiled ``epistemic_graph.numeric`` kernel. Modules that also
need the real compiler import it themselves, directly.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.models.company_brain import DataClassification, NodeACL
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


class _FakeMarkingStore:
    """Minimal in-memory durable-store stand-in for the mandatory-marking seam.

    Mirrors the ``execute(query, params) -> list[dict]`` shape the real
    process-owned graph store exposes. ``ContextCompiler.compile`` runs every
    candidate through the policy ``enforce`` gate, which resolves the
    mandatory-marking store on every call regardless of whether a marking
    was ever applied -- so every test here needs one installed, not just the
    ones that call ``apply_marking`` directly.
    """

    @staticmethod
    def execute(_query, _params):
        return []


class FakeRetriever:
    """Engine stand-in whose ``search_hybrid`` returns EG-ranked rows.

    Returns a fixed candidate pool (as EG's served ``Discover`` would rank
    it) regardless of query -- the compiler's assembly is what's under test.
    """

    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def search_hybrid(self, query, top_k=10, session=None):  # noqa: ARG002
        return list(self._nodes)[:top_k]


def _actor(**kw) -> ActorContext:
    kw.setdefault("tenant_id", "tenant-test")
    kw.setdefault("authenticated", True)
    return ActorContext(actor_id="principal:test", actor_type=ActorType.AI_AGENT, **kw)


def _session(**kw) -> GraphSession:
    actor = _actor(**kw)
    return GraphSession(
        actor=actor, tenant=actor.tenant_id, scopes=frozenset({"kg:read"})
    )


def _grant_public(nodes: list[dict]) -> None:
    """Grant a PUBLIC-classification ACL for every ``nodes[i]["id"]``.

    ``ContextCompiler.compile`` runs every candidate through the real
    ``enforce`` gate, whose ACL layer is fail-closed (a node with no ACL is
    denied outright). These synthetic test nodes only exist in the fake
    retriever, never in a real graph, so they need an explicit ACL; PUBLIC
    classification satisfies that layer without interfering with whatever
    marking/role behavior the test is exercising.
    """
    permissions = get_company_brain().permissions
    for node in nodes:
        permissions.set_acl(
            NodeACL(node_id=node["id"], classification=DataClassification.PUBLIC)
        )


__all__ = [
    "FakeRetriever",
    "_FakeMarkingStore",
    "_actor",
    "_grant_public",
    "_session",
]
