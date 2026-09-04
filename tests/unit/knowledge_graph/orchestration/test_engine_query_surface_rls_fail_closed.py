"""BUG-CX-103 — the non-Cypher engine surfaces must govern rows FAIL-CLOSED.

``QueryMixin.sql()`` / ``.sparql()`` (and ``.uql()``) submit the caller's query
text to the engine verbatim: unlike ``query_cypher`` nothing pushes the tenant
scope or the KG-2.60 owner/scope predicate into the statement, so the AU-side
``secured_reads.filter_rows()`` / ``visible()`` pass is the *only* row-level
authorization those surfaces have.

Before this fix that pass ran inside ``try/except Exception: logger.debug(...)``
and, on any failure, returned the **pre-filter** rows:

* ``filter_rows()`` raises ``PermissionError`` **by design** for a result whose
  rows carry no governed node id — the ordinary shape of a projecting query
  (``SELECT name FROM nodes``, ``SELECT ?name WHERE {...}``) — so ordinary data
  was returned unfiltered.
* ``_verified_actor()`` raises when no verified actor is ambient at all, so an
  unauthenticated caller got **everything**.

Every test here asserts a DENIAL or a correctly-filtered result. None of them
pins the old behaviour; each of the denial tests fails against the pre-fix
``engine_query.py`` (which returns the rows) and passes after.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.models.company_brain import DataClassification, NodeACL
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

pytestmark = pytest.mark.concept("AU-P0-4")


def _actor(actor_id: str = "reader-1", tenant: str = "acme") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AI_AGENT,
        roles=("reader",),
        tenant_id=tenant,
        authenticated=True,
    )


class _SqlNamespace:
    def __init__(self, rows):
        self._rows = rows

    def sql(self, query):
        return self._rows

    def uql(self, query):
        return self._rows


class _Client:
    def __init__(self, rows):
        self.query = _SqlNamespace(rows)


class _Graph:
    """Mimics ``backend.graph`` for all three surfaces at once."""

    def __init__(self, rows):
        self._rows = rows
        self._client = _Client(rows)

    def sparql(self, query, base_iri="", type_convention=""):
        return self._rows


class _Backend:
    def __init__(self, rows):
        self.graph = _Graph(rows)


class _Engine(QueryMixin):
    def __init__(self, rows) -> None:
        self.backend = _Backend(rows)


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


# A projection with no id column — the normal shape of a ``graph_table``-mirrored
# connector table, and exactly what ``filter_rows()`` refuses to classify.
_UNGOVERNABLE = [{"name": "alice", "salary": 250_000}]


def test_sql_denies_a_row_with_no_governed_node_id(brain):
    eng = _Engine(_UNGOVERNABLE)
    with use_actor(_actor()), pytest.raises(PermissionError):
        eng.sql("SELECT name, salary FROM nodes")


def test_sparql_denies_a_row_with_no_governed_node_id(brain):
    eng = _Engine(_UNGOVERNABLE)
    with use_actor(_actor()), pytest.raises(PermissionError):
        eng.sparql("SELECT ?name WHERE { ?s :name ?name }")


def test_uql_denies_a_row_with_no_governed_node_id(brain):
    eng = _Engine([{"score": 0.91}])
    with use_actor(_actor()), pytest.raises(PermissionError):
        eng.uql("MATCH (:Doc) |> LIMIT 5")


def test_sql_denies_an_unauthenticated_actor(brain):
    """An unverified identity is a denial, not an unfiltered full read.

    ``tests/conftest.py::isolate_graph_compute_engine`` binds an authenticated
    ambient actor for EVERY test, so the unverified case has to be bound
    explicitly — a bare "no ``use_actor``" block is silently still authorized.
    """
    eng = _Engine([{"id": "secret-doc", "body": "classified"}])
    unverified = ActorContext(
        actor_id="anon",
        actor_type=ActorType.AI_AGENT,
        roles=(),
        tenant_id="acme",
        authenticated=False,
    )
    with use_actor(unverified), pytest.raises(PermissionError):
        eng.sql("SELECT * FROM nodes")


def test_sparql_denies_an_unauthenticated_actor(brain):
    eng = _Engine([{"id": "secret-doc", "body": "classified"}])
    unverified = ActorContext(
        actor_id="anon",
        actor_type=ActorType.AI_AGENT,
        roles=(),
        tenant_id="acme",
        authenticated=False,
    )
    with use_actor(unverified), pytest.raises(PermissionError):
        eng.sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")


def test_sql_denies_when_no_identity_is_bound_at_all(brain, monkeypatch):
    """``current_actor()``'s ``IdentityRequiredError`` must deny the read.

    It is a ``PermissionError`` subclass, so it propagates rather than being
    re-wrapped — but before the fix the broad ``except`` swallowed it and the
    caller received every row with no identity whatsoever.
    """
    from agent_utilities.knowledge_graph.core import secured_reads
    from agent_utilities.security.brain_context import IdentityRequiredError

    def _no_identity():
        raise IdentityRequiredError("A verified actor context is required")

    monkeypatch.setattr(secured_reads, "current_actor", _no_identity)
    eng = _Engine([{"id": "secret-doc", "body": "classified"}])
    with pytest.raises(IdentityRequiredError):
        eng.sql("SELECT * FROM nodes")


def test_sql_denies_when_the_enforcement_pipeline_itself_fails(brain, monkeypatch):
    """A defect INSIDE the filter is still a failure to authorize.

    It must surface as a ``PermissionError``, never as the pre-filter rows.
    """
    from agent_utilities.knowledge_graph.core import secured_reads

    def _boom(rows, actor=None, **kwargs):
        raise RuntimeError("ACL backend unreachable")

    monkeypatch.setattr(secured_reads, "filter_rows", _boom)
    eng = _Engine([{"id": "secret-doc"}])
    with use_actor(_actor()), pytest.raises(PermissionError):
        eng.sql("SELECT * FROM nodes")


def test_sql_drops_a_governed_row_the_actor_may_not_read(brain):
    """Enforcement discriminates rather than blanket-denying: an unclassified
    node is dropped, a PUBLIC sibling in the same result set survives."""
    brain.permissions.set_acl(
        NodeACL(node_id="public-doc", classification=DataClassification.PUBLIC)
    )
    eng = _Engine([{"id": "public-doc"}, {"id": "secret-doc"}])
    with use_actor(_actor()):
        out = eng.sql("SELECT id FROM nodes")
    assert [row["id"] for row in out] == ["public-doc"]


def test_sparql_drops_a_governed_row_the_actor_may_not_read(brain):
    brain.permissions.set_acl(
        NodeACL(node_id="public-doc", classification=DataClassification.PUBLIC)
    )
    eng = _Engine([{"id": "public-doc"}, {"id": "secret-doc"}])
    with use_actor(_actor()):
        out = eng.sparql("SELECT ?id WHERE { ?id a ?t }")
    assert [row["id"] for row in out] == ["public-doc"]


def test_read_only_guard_still_precedes_row_policy(brain):
    """The SELECT/WITH/EXPLAIN guard is unchanged and still raises ValueError."""
    eng = _Engine([{"id": "n1"}])
    with use_actor(_actor()), pytest.raises(ValueError, match="read-only"):
        eng.sql("DELETE FROM nodes")
