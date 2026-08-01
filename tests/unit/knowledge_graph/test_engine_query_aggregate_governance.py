"""Aggregate-query governance in ``QueryMixin.query_cypher`` (CONCEPT:AU-KG.query.query-aggregation).

An aggregate/scalar Cypher projection (``count``/``sum``/``avg``/``collect``/...)
collapses many rows into one with no per-row node id — but the read-path
governance chokepoint (``secured_reads.row_node_ids`` via ``filter_rows``)
unconditionally denied a row that carries no identifiable node id. Before this
fix, EVERY aggregate read through ``graph_query``/``engine.query_cypher`` raised
``PermissionError("Graph row-policy or audit enforcement failed")`` regardless of
caller privilege — a live ``ask``/``graph_query`` MCP call with
``cypher="MATCH (n:WorkflowDefinition) RETURN count(n) AS c"`` failed this way,
even though the SAME nodes were readable via a per-row projection.

Covers:
    - ``is_aggregation_cypher`` relocation (the ``mcp.tools.query_tools``
      re-export stays byte-identical for existing callers/tests).
    - an aggregate read now succeeds, and is still tenant + owner/scope scoped
      query-side (pushed into the Cypher text via ``tenant_sharing.apply_visibility``,
      since there is no row to Python-side post-filter).
    - a privileged actor's aggregate read gets no owner/scope restriction
      (mirrors ``secured_reads.visible``'s own privileged bypass).
    - the general (non-aggregate) case is UNCHANGED: a row with no governable id
      still denies (regression guard against silently broadening the boundary).
    - a read-scope (``kg:read``, non-privileged) MCP-shaped actor sees a PUBLIC
      row and is denied a RESTRICTED row it does not own, on a real per-row read
      — proving the fix did not weaken RLS for non-public rows.
    - the swallowed cause is now logged (true failing step + message, sanitized)
      before being collapsed into the generic caller-facing PermissionError.
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
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import (
    QueryMixin,
    is_aggregation_cypher,
)
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
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
    """Minimal QueryMixin host, mirroring test_engine_query_control_routing's."""

    def __init__(self, *, backend: _Backend) -> None:
        self.backend = backend
        self.control_backend = None


def _actor(
    actor_id: str = "agent:mcp-caller",
    *,
    roles: tuple[str, ...] = ("kg:read",),
    tenant: str = "tenant-a",
) -> ActorContext:
    """A read-scope, non-privileged actor shaped like an external MCP caller."""
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


# --- is_aggregation_cypher relocation ---------------------------------------


def test_is_aggregation_cypher_reexported_from_query_tools():
    """query_tools kept its import path; engine_query is now the sole definition."""
    from agent_utilities.mcp.tools.query_tools import (
        is_aggregation_cypher as reexported,
    )

    assert reexported is is_aggregation_cypher


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n:WorkflowDefinition) RETURN count(n) AS c",
        "MATCH (n) RETURN sum(n.cost) AS total",
        "MATCH (n) RETURN avg(n.score), max(n.score)",
        "MATCH (n) RETURN collect(n.id)",
    ],
)
def test_is_aggregation_cypher_detects_aggregates(cypher):
    assert is_aggregation_cypher(cypher) is True


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n:Record) RETURN n.id AS id",
        "MATCH (f:Function {name:'probe'}) RETURN f",
        "MATCH (n) RETURN n.max_depth AS d",
        "MATCH (n) WHERE n.note = 'count(*)' RETURN n",
    ],
)
def test_is_aggregation_cypher_ignores_non_aggregates(cypher):
    assert is_aggregation_cypher(cypher) is False


# --- the bug: aggregate reads were unconditionally denied -------------------


def test_aggregate_read_succeeds_for_a_readscope_actor(brain):
    backend = _Backend(rows=[{"c": 142}])
    engine = _Harness(backend=backend)
    actor = _actor(roles=("kg:read",))
    session = _session(actor)

    with use_actor(actor), use_session(session):
        rows = engine.query_cypher(
            "MATCH (n:WorkflowDefinition) RETURN count(n) AS c", session=session
        )

    assert rows == [{"c": 142}]
    # Still audited -- with no node ids to name (nothing governable in an
    # aggregate row), not silently unaudited.
    assert brain.provenance.read_count == 1
    assert brain.provenance._read_audits[-1].nodes_accessed == []
    assert brain.provenance._read_audits[-1].query_summary == (
        "native-cypher-read (aggregate)"
    )


def test_aggregate_read_is_tenant_and_owner_scope_filtered_query_side(brain):
    """No Python-side row to post-filter by owner/scope, so that boundary is
    pushed INTO the query text instead (tenant_sharing.apply_visibility)."""
    backend = _Backend(rows=[{"c": 3}])
    engine = _Harness(backend=backend)
    actor = _actor("agent:reader", roles=("kg:read",), tenant="acme")
    session = _session(actor)

    with use_actor(actor), use_session(session):
        engine.query_cypher("MATCH (n:Doc) RETURN count(n) AS c", session=session)

    sent_query, params = backend.calls[-1]
    # D-W2T-2: the tenant id / owner id are bound parameters now, not spliced
    # into the query text as string literals.
    assert "tenant_id = $_tenant_scope_id" in sent_query
    assert "_owner_id = $_visibility_owner_id" in sent_query
    assert "_shared_scope IN ['org', 'commons']" in sent_query
    assert params["_tenant_scope_id"] == "acme"
    assert params["_visibility_owner_id"] == "agent:reader"


def test_aggregate_read_on_a_non_n_variable_is_scoped_by_its_own_variable(brain):
    """D-SH-4 (reports/deferred/lane-skill-harvest.md): a query aliased as
    `w` rather than `n` (mirroring the exact shape used by
    orchestration/agent_activation.py's and agent_dispatch_worker.py's engine
    reachability probes, `MATCH (w:WorkItem) RETURN count(w) AS c` -- a
    content label is used here instead of `WorkItem` only so this test
    exercises the content-backend path, not control-plane routing, which is
    orthogonal to this bug) used to get a visibility/tenant predicate written
    against a hardcoded `n`, a variable never bound in this query. Cypher
    treats a reference to an unbound variable as never matching, so the whole
    read silently collapsed to a zero-row/zero-count result instead of the
    real answer -- the exact divergence this lane found between
    `engine.query_cypher` and `engine.backend.execute` on an identical query.
    """
    backend = _Backend(rows=[{"c": 7}])
    engine = _Harness(backend=backend)
    actor = _actor("agent:reader", roles=("kg:read",), tenant="acme")
    session = _session(actor)

    with use_actor(actor), use_session(session):
        rows = engine.query_cypher(
            "MATCH (w:Widget) RETURN count(w) AS c", session=session
        )

    assert rows == [{"c": 7}]
    sent_query, params = backend.calls[-1]
    assert "w.tenant_id = $_tenant_scope_id" in sent_query
    assert "w._owner_id = $_visibility_owner_id" in sent_query
    assert "n.tenant_id" not in sent_query  # the bug: referenced an undefined var
    assert "n._owner_id" not in sent_query
    assert params["_tenant_scope_id"] == "acme"
    assert params["_visibility_owner_id"] == "agent:reader"


def test_aggregate_read_privileged_actor_gets_no_owner_scope_predicate(brain):
    backend = _Backend(rows=[{"c": 3}])
    engine = _Harness(backend=backend)
    actor = _actor("root", roles=("kg:admin", "kg:read"), tenant="acme")
    session = _session(actor)

    with use_actor(actor), use_session(session):
        engine.query_cypher("MATCH (n:Doc) RETURN count(n) AS c", session=session)

    sent_query, params = backend.calls[-1]
    assert "_owner_id" not in sent_query  # no owner/scope restriction injected
    assert (
        "tenant_id = $_tenant_scope_id" in sent_query
    )  # tenant scoping still mandatory
    assert params["_tenant_scope_id"] == "acme"


def test_aggregate_read_generic_admin_remains_owner_scoped(brain):
    backend = _Backend(rows=[{"c": 1}])
    engine = _Harness(backend=backend)
    actor = _actor("app-admin", roles=("admin", "kg:read"), tenant="acme")
    session = _session(actor)

    with use_actor(actor), use_session(session):
        engine.query_cypher("MATCH (n:Doc) RETURN count(n) AS c", session=session)

    sent_query, params = backend.calls[-1]
    assert "_owner_id = $_visibility_owner_id" in sent_query
    assert params["_visibility_owner_id"] == "app-admin"


# --- regression guard: the general (non-aggregate) case is unchanged --------


def test_non_aggregate_read_without_governed_id_still_denies(brain, caplog):
    """The fix must NOT broaden the general case: a row that strips its id
    still can't dodge governance by simply not being an aggregate."""
    backend = _Backend(rows=[{"name": "no id here"}])
    engine = _Harness(backend=backend)
    actor = _actor(roles=("kg:read",))
    session = _session(actor)

    with (
        use_actor(actor),
        use_session(session),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
        pytest.raises(
            PermissionError, match="Graph row-policy or audit enforcement failed"
        ),
    ):
        engine.query_cypher("MATCH (n:Doc) RETURN n.name AS name", session=session)

    assert any("governed node id" in record.getMessage() for record in caplog.records)


def test_non_aggregate_read_respects_public_vs_restricted_acl(brain):
    """The exact scenario this task verifies end to end: a read-scope
    (non-privileged) MCP-shaped actor sees a PUBLIC row and is denied a
    RESTRICTED row it does not own -- the fix did not weaken RLS."""
    brain.permissions.set_acl(
        NodeACL(node_id="wf:public-1", classification=DataClassification.PUBLIC)
    )
    brain.permissions.set_acl(
        NodeACL(
            node_id="wf:restricted-1",
            classification=DataClassification.RESTRICTED,
            data_owner="someone-else",
        )
    )
    backend = _Backend(
        rows=[
            {"id": "wf:public-1", "name": "public workflow"},
            {"id": "wf:restricted-1", "name": "restricted workflow"},
        ]
    )
    engine = _Harness(backend=backend)
    actor = _actor("agent:mcp-caller", roles=("kg:read",), tenant="tenant-a")
    session = _session(actor)

    with use_actor(actor), use_session(session):
        rows = engine.query_cypher(
            "MATCH (n:WorkflowDefinition) RETURN n.id AS id, n.name AS name",
            session=session,
        )

    assert [r["id"] for r in rows] == ["wf:public-1"]


# --- task 1: the swallowed cause is now surfaced in the log -----------------


def test_secured_read_failure_logs_the_true_cause(brain, caplog, monkeypatch):
    """The caller-facing PermissionError message stays generic (no internals
    leaked to callers), but the SERVER LOG must now name which step actually
    failed and why, sanitized, instead of collapsing every distinct cause into
    one indistinguishable message server-side too (the swallowed-cause
    anti-pattern this task's Task 1 exists to close)."""
    from agent_utilities.knowledge_graph.core import secured_reads

    monkeypatch.setattr(secured_reads, "filter_rows", lambda rows, _actor: rows)

    def _boom(_rows, _actor):
        raise PermissionError("Row visibility evaluation failed") from ValueError(
            "https://internal.example/should-be-redacted boom"
        )

    monkeypatch.setattr(secured_reads, "visible", _boom)

    backend = _Backend(rows=[{"id": "n1"}])
    engine = _Harness(backend=backend)
    actor = _actor(roles=("kg:read",))
    session = _session(actor)

    with (
        use_actor(actor),
        use_session(session),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
        pytest.raises(
            PermissionError, match="Graph row-policy or audit enforcement failed"
        ),
    ):
        engine.query_cypher("MATCH (n:Doc) RETURN n.id AS id", session=session)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "Row visibility evaluation failed" in logged  # the true failing step
    assert "ValueError" in logged  # its own chained root cause
    assert "boom" in logged
    assert "https://internal.example" not in logged  # sanitized before logging
    assert "<endpoint>" in logged


# --- task 4: blast radius -- the internal delegation path shares this bug ---


def test_delegation_context_carry_read_used_ambient_session():
    """CONCEPT:AU-ORCH.session.carry-invoker — the internal delegation path
    (agent_runner.run_agent's context_ref carry-over, agent_execution_tools's
    swarm context_ref, graph_context(action='get')) all call
    ``engine.query_cypher(cypher, params)`` with NO explicit ``session=``. That
    means ``resolve_session(None, ...)`` falls back to the AMBIENT session --
    the exact same resolution path (and therefore the exact same secured_reads
    governance) any other MCP-tool-triggered read uses. Delegation is not a
    parallel, differently-shaped session -- it is the SAME chokepoint."""
    import inspect

    from agent_utilities.knowledge_graph.orchestration.engine_query import (
        QueryMixin,
    )

    sig = inspect.signature(QueryMixin.query_cypher)
    assert sig.parameters["session"].default is None  # ambient fallback, not opt-out


def test_delegation_context_carry_query_shape_was_the_bug_now_fixed(brain):
    """The EXACT (pre-fix) query shape used by agent_runner.py's context_ref
    carry-over and agent_execution_tools.py's swarm context_ref --
    ``RETURN c.content AS content`` with no id -- is a real, non-aggregate
    instance of this task's bug: it denies via the SAME row_node_ids path an
    aggregate does, for ANY actor, because the row it returns carries no
    governable node id. This is why those two call sites (plus the direct MCP
    tool ``graph_context(action='get')``) were fixed to project ``c.id AS id``
    too -- proven here end to end against the REAL (non-fake) governed read
    path, not just the fake-engine unit test in test_invoker_context_handoff.py."""
    actor = _actor("agent:delegated-run", roles=("kg:read",), tenant="tenant-a")
    session = _session(actor)

    # Pre-fix shape (still reachable if any OTHER call site regresses to it):
    # denied, same as any other id-less non-aggregate projection.
    old_shape_backend = _Backend(rows=[{"content": "curated invoker context"}])
    old_shape_engine = _Harness(backend=old_shape_backend)
    with (
        use_actor(actor),
        use_session(session),
        pytest.raises(
            PermissionError, match="Graph row-policy or audit enforcement failed"
        ),
    ):
        old_shape_engine.query_cypher(
            "MATCH (c:ContextBlob) WHERE c.id = $id RETURN c.content AS content",
            {"id": "ctx:abc:1"},
            session=session,
        )

    # Fixed shape (what agent_runner.py / agent_execution_tools.py / the
    # graph_context MCP tool now send): the row now carries a governable id,
    # so it is actually EVALUATED by the ACL layer -- properly denied with no
    # ACL registered (fail-closed default-deny, same as any other governed
    # node), and properly permitted once classified PUBLIC. Governance now
    # RUNS, rather than being unconditionally denied by row_node_ids before
    # the ACL layer is ever reached.
    new_shape_backend = _Backend(
        rows=[{"id": "ctx:abc:1", "content": "curated invoker context"}]
    )
    new_shape_engine = _Harness(backend=new_shape_backend)
    with use_actor(actor), use_session(session):
        denied_rows = new_shape_engine.query_cypher(
            "MATCH (c:ContextBlob) WHERE c.id = $id "
            "RETURN c.id AS id, c.content AS content",
            {"id": "ctx:abc:1"},
            session=session,
        )
    assert denied_rows == []  # no ACL registered yet -- fail-closed, not an error

    brain.permissions.set_acl(
        NodeACL(node_id="ctx:abc:1", classification=DataClassification.PUBLIC)
    )
    with use_actor(actor), use_session(session):
        rows = new_shape_engine.query_cypher(
            "MATCH (c:ContextBlob) WHERE c.id = $id "
            "RETURN c.id AS id, c.content AS content",
            {"id": "ctx:abc:1"},
            session=session,
        )
    assert rows == [{"id": "ctx:abc:1", "content": "curated invoker context"}]
