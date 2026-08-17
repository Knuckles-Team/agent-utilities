"""Regression: `QueryMixin.query_cypher`'s tenant/visibility-scoping try block
(``engine_query.py``, roughly lines 204-320) must not launder every failure
inside it into an identical, generic ``PermissionError("Graph query scoping
failed")``.

Reproduced live (2026-08): ``MATCH ()-[r]->() RETURN count(r)`` failed with
``PermissionError("Graph query scoping failed")`` while the structurally
identical ``MATCH (a)-[r]->(b) RETURN count(r)`` succeeded and returned
11,641 rows. A query-shape problem read, to any caller/log/operator, exactly
like an authorization denial.

True root cause, traced (not guessed) through the full call chain:

    engine_query.py:211  scope(query, session.actor)
      -> secured_reads.scope()
           -> company_brain.TenancyManager.scope_cypher_query()
                -> cypher_scoping.first_bound_node_variable()

``first_bound_node_variable`` scans only NODE patterns (``(var:Label)``) in
the query's first ``MATCH`` clause for a variable a tenant predicate can be
written against; a relationship variable (``[r]``) is never a candidate.
``MATCH ()-[r]->()`` binds zero node variables, so it raises
``UnscopableQueryError`` -- a *deliberate*, correctly fail-closed decision
(there is no safe variable to scope this query by), and already a
``PermissionError`` subclass. Two independent layers then each caught it
with a bare ``except Exception`` and re-wrapped it into a NEW, generic
``PermissionError`` (``secured_reads.scope``'s own "Tenant query scoping
failed", then ``engine_query.py``'s "Graph query scoping failed"),
discarding the specific, actionable message both times.

The other failure mode this task closes: any exception that is NOT already
a deliberate ``PermissionError``-family decision (e.g. an ``AttributeError``
from a caller-supplied actor object missing an expected method -- reproduced
via a probe against ``secured_reads.scope``) was *also* laundered into the
same generic ``PermissionError``, actively misdirecting debugging toward the
authorization layer for what is actually a code defect.

Fix: the try block now distinguishes the two cases --
    - ``except PermissionError`` (a genuine, already-typed fail-closed
      decision) is re-raised AS-IS, so its specific type/message reach the
      caller and the log.
    - ``except Exception`` (anything else) is logged with its full cause
      chain and re-raised as ``cypher_scoping.QueryScopingError`` -- a
      distinct, non-``PermissionError`` type -- rather than mislabeled
      ``PermissionError``. It still fails the read; a scoping failure of
      either kind must never fall through to executing an unscoped query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.cypher_scoping import (
    QueryScopingError,
    UnscopableQueryError,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

_LOGGER_NAME = "agent_utilities.knowledge_graph.orchestration.engine_query"


@dataclass
class _Backend:
    rows: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def execute_read(self, query: str, params: dict) -> list[dict]:
        self.calls.append((query, params))
        return list(self.rows)


class _Harness(QueryMixin):
    """Minimal QueryMixin host, mirroring the aggregate-governance suite's."""

    def __init__(self, *, backend: _Backend) -> None:
        self.backend = backend
        self.control_backend = None


def _actor(
    actor_id: str = "agent:mcp-caller",
    *,
    roles: tuple[str, ...] = ("kg:read",),
    tenant: str = "tenant-a",
) -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=roles,
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


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


# --- the reproduced defect: anonymous-relationship pattern -----------------


def test_anonymous_relationship_pattern_fails_with_the_specific_unscopable_error(
    brain, caplog
):
    """``MATCH ()-[r]->() RETURN count(r)`` binds no node variable at all, so
    it genuinely cannot be safely tenant-scoped -- failing closed is correct.
    What was broken is that the SPECIFIC reason ("every node pattern ... is
    anonymous") was discarded by two layers of generic re-wrapping, making it
    indistinguishable from an actual authorization denial. It must now
    surface as the real, typed, specific ``UnscopableQueryError`` -- not the
    generic ``PermissionError("Graph query scoping failed")``."""
    backend = _Backend(rows=[{"count(r)": 11641}])
    engine = _Harness(backend=backend)
    actor = _actor()
    session = _session(actor)

    with (
        use_actor(actor),
        use_session(session),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
        pytest.raises(UnscopableQueryError) as exc_info,
    ):
        engine.query_cypher("MATCH ()-[r]->() RETURN count(r)", session=session)

    # The specific, actionable message -- not the generic collapse.
    assert "anonymous" in str(exc_info.value)
    assert str(exc_info.value) != "Graph query scoping failed"
    assert str(exc_info.value) != "Tenant query scoping failed"
    # It IS still a PermissionError (fail closed, correctly typed) --
    # UnscopableQueryError is a PermissionError subclass.
    assert isinstance(exc_info.value, PermissionError)
    # The backend must never have been asked to run an unscoped read.
    assert backend.calls == []

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "UnscopableQueryError" in logged
    assert "anonymous" in logged


def test_bound_variable_relationship_pattern_succeeds_and_matches_backend(brain):
    """The structurally identical, correctly-shaped sibling query --
    ``MATCH (a)-[r]->(b) RETURN count(r)`` -- binds node variables and must
    scope and succeed, returning the SAME shape of result the anonymous
    variant should have been able to produce were it scopable."""
    backend = _Backend(rows=[{"count(r)": 11641}])
    engine = _Harness(backend=backend)
    actor = _actor()
    session = _session(actor)

    with use_actor(actor), use_session(session):
        rows = engine.query_cypher(
            "MATCH (a)-[r]->(b) RETURN count(r)", session=session
        )

    assert rows == [{"count(r)": 11641}]
    sent_query, params = backend.calls[-1]
    assert "a.tenant_id = $_tenant_scope_id" in sent_query
    assert params["_tenant_scope_id"] == "tenant-a"


# --- the reproduced defect: unrelated code failures laundered into denial --


def test_non_authorization_failure_in_scoping_is_not_reported_as_permission_error(
    brain, caplog
):
    """Reproduced via probe: an unrelated defect surfacing INSIDE the
    scoping block (e.g. an actor object missing an expected method, an
    ``AttributeError``/``TypeError`` bug in the scoping pipeline itself) must
    NOT be reported as ``PermissionError`` -- that actively misdirects
    debugging toward the authorization layer for what is actually a code
    defect. It must still fail the request (never fall through to executing
    an unscoped read), but as a distinct, honestly-labeled type."""
    import unittest.mock as mock

    def _broken_scope(_query, _actor):
        raise AttributeError(
            "'_BrokenActor' object has no attribute 'ensure_credential_current'"
        )

    backend = _Backend(rows=[{"id": "n1"}])
    engine = _Harness(backend=backend)
    actor = _actor()
    session = _session(actor)

    with (
        mock.patch(
            "agent_utilities.knowledge_graph.core.secured_reads.scope",
            side_effect=_broken_scope,
        ),
        use_actor(actor),
        use_session(session),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
        pytest.raises(QueryScopingError) as exc_info,
    ):
        engine.query_cypher("MATCH (n:Doc) RETURN n", session=session)

    # Never a PermissionError -- this was never an authorization decision.
    assert not isinstance(exc_info.value, PermissionError)
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, AttributeError)
    # The backend must never have been asked to run an unscoped read.
    assert backend.calls == []

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "AttributeError" in logged
    assert "ensure_credential_current" in logged
    assert "code defect" in logged


def test_permission_error_from_scoping_still_fails_closed_and_propagates_as_is(
    brain, caplog
):
    """A genuine, already-typed denial raised anywhere in the scoping block
    (not just UnscopableQueryError) is still a ``PermissionError`` and still
    denies -- this task must not widen what is caught or fail open."""
    import unittest.mock as mock

    def _denied_scope(_query, _actor):
        raise PermissionError("actor is not entitled to this tenant")

    backend = _Backend(rows=[{"id": "n1"}])
    engine = _Harness(backend=backend)
    actor = _actor()
    session = _session(actor)

    with (
        mock.patch(
            "agent_utilities.knowledge_graph.core.secured_reads.scope",
            side_effect=_denied_scope,
        ),
        use_actor(actor),
        use_session(session),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
        pytest.raises(PermissionError, match="actor is not entitled to this tenant"),
    ):
        engine.query_cypher("MATCH (n:Doc) RETURN n", session=session)

    assert backend.calls == []
