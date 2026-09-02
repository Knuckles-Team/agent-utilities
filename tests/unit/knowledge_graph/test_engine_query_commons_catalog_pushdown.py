"""BUG-PE-040 — commons-catalog pushdown on ``QueryMixin.query_cypher``'s
non-aggregate (row-read) path.

``filter_commons_catalog`` fails CLOSED by design (``tenant_sharing.py``): a
row with no ``node_type``, no ``tenant_id``, and not the reader's own is
dropped. That is correct for a whole-node row, but a *projecting* query
(``MATCH (t:Tool) RETURN t.id AS id, t.name AS name``) returns rows with no
``node_type`` column at all, even for a catalog-shareable commons node — the
identical defect already fixed one layer up in
``tenant_sharing.read_union`` (BUG-PE-039, commit 7b8075b8d). ``query_cypher``
sits UNDER ``read_union`` (agent-webui's ``_graph_union_executor`` calls it),
so fixing the upper layer alone cannot rescue rows already dropped here.

Covers both required constraints, proved together (constraint (b) matters
only because constraint (a) exists to prove):
    (a) a projecting query against the commons graph still surfaces a
        catalog-shareable commons row (``test_projecting_query_returns_
        commons_rows``);
    (b) that trust never widens into judging a row that DOES carry a
        classifiable ``node_type`` — a non-shareable row stamped with
        another tenant's id is still denied even in a query shape that
        triggers pushdown (``test_projecting_query_still_denies_non_
        shareable_foreign_row``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


@dataclass
class _Backend:
    rows: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def execute_read(self, query: str, params: dict) -> list[dict]:
        self.calls.append((query, params))
        return list(self.rows)


class _Harness(QueryMixin):
    """A commons-graph-bound engine. ``graph_compute.graph_name`` is what
    ``_is_commons_graph_name`` matches against (the literal ``"__commons__"``
    fallback, so no ``config`` object is needed)."""

    def __init__(self, *, backend: _Backend) -> None:
        self.backend = backend
        self.control_backend = None
        self.graph_compute = SimpleNamespace(graph_name="__commons__")


def _actor(tenant: str = "tenant-a") -> ActorContext:
    """A read-scope, non-privileged actor -- the shape this restriction
    exists to constrain (mirrors test_engine_query_aggregate_governance's
    own ``_actor``)."""
    return ActorContext(
        actor_id="agent:mcp-caller",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id=tenant,
        authenticated=True,
    )


def _session(actor: ActorContext) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        policy_version="policy-test",
        audience="agent-services",
    )


@pytest.fixture(autouse=True)
def _bypass_acl_layer(monkeypatch):
    """Isolate the commons-catalog pushdown/classifier under test from the
    separate per-node ACL gate (``secured_reads.visible``/``filter_rows``) --
    that gate is covered elsewhere (test_engine_query_aggregate_governance's
    ``test_non_aggregate_read_respects_public_vs_restricted_acl``). Mirrors
    test_engine_query_control_routing's own ``_identity_read_policy`` fixture.
    Tenant scoping (``secured_reads.scope``) is likewise a no-op here so the
    commons-catalog predicate this test targets is the only query-text
    mutation in play; it is proven separately in
    test_engine_query_aggregate_governance's tenant-scoping tests.
    """
    from agent_utilities.knowledge_graph.core import secured_reads

    monkeypatch.setattr(secured_reads, "scope", lambda query, _actor: (query, {}))
    monkeypatch.setattr(secured_reads, "filter_rows", lambda rows, _actor: rows)
    monkeypatch.setattr(secured_reads, "visible", lambda rows, _actor: rows)
    monkeypatch.setattr(secured_reads, "audit_read", lambda *_args, **_kwargs: None)


def test_projecting_query_returns_commons_rows():
    """Constraint (a): a projecting query against the commons graph must
    still surface a catalog-shareable commons row, even though the returned
    row carries no ``node_type`` column for ``filter_commons_catalog`` to
    classify. Before the fix, this row was unconditionally dropped."""
    backend = _Backend(rows=[{"id": "t1", "name": "Tool-A"}])
    engine = _Harness(backend=backend)
    actor = _actor()
    session = _session(actor)

    with use_actor(actor), use_session(session):
        rows = engine.query_cypher(
            "MATCH (t:Tool) RETURN t.id AS id, t.name AS name", session=session
        )

    assert [r["id"] for r in rows] == ["t1"]
    # The pushdown actually narrowed the executed query text -- proves the
    # row survived because the query was scoped at the source, not because
    # the classifier silently started trusting every unclassifiable row.
    executed_query = backend.calls[0][0]
    assert executed_query != "MATCH (t:Tool) RETURN t.id AS id, t.name AS name"
    assert "node_type" in executed_query


def test_projecting_query_still_denies_non_shareable_foreign_row():
    """Constraint (b), proved alongside (a): trusting the pushdown for a row
    the classifier cannot read a ``node_type`` from must never widen into
    trusting it for a row that DOES carry one. A non-catalog-shareable
    row stamped with another tenant's id is still denied, even returned
    from the SAME call that triggers pushdown for the sibling projected
    row above."""
    backend = _Backend(
        rows=[
            # Unclassifiable projection of a catalog-shareable node -> kept.
            {"id": "t1", "name": "Tool-A"},
            # Classifiable, NOT catalog-shareable, another tenant's data ->
            # dropped even though this same call triggers pushdown.
            {
                "id": "wi-1",
                "name": "someone else's item",
                "node_type": "WorkItem",
                "tenant_id": "other-tenant",
            },
        ]
    )
    engine = _Harness(backend=backend)
    actor = _actor(tenant="tenant-a")
    session = _session(actor)

    with use_actor(actor), use_session(session):
        rows = engine.query_cypher(
            "MATCH (n) RETURN n.id AS id, n.name AS name, n.node_type AS node_type,"
            " n.tenant_id AS tenant_id",
            session=session,
        )

    assert [r["id"] for r in rows] == ["t1"]
