"""Regression: `TenancyManager.scope_cypher_query` must scope by the query's
OWN bound variable, not a hardcoded `n` (D-SH-4).

``reports/deferred/lane-skill-harvest.md`` D-SH-4: against the live pod, a
Cypher `WHERE` read through `engine.query_cypher` returned nothing where
`engine.backend.execute` with the identical query and params returned the
right rows. Root cause: `scope_cypher_query` injected `n.tenant_id = '...'`
into every query regardless of what its node variable was actually called.
`MATCH (s:Skill) WHERE s.name = 'x' RETURN s` got `n.tenant_id = '...'`
injected — `n` is never bound anywhere in that query, so the predicate can
never be true and the read silently returned zero rows.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain import TenancyManager
from agent_utilities.knowledge_graph.core.cypher_scoping import UnscopableQueryError


def test_scopes_by_the_querys_own_variable_not_a_hardcoded_n():
    tm = TenancyManager()

    scoped = tm.scope_cypher_query(
        "MATCH (s:Skill) WHERE s.name = 'foo' RETURN s", tenant_id="acme"
    )

    assert "s.tenant_id = 'acme'" in scoped
    assert "n.tenant_id" not in scoped  # the D-SH-4 bug: referenced an undefined var


def test_scopes_a_where_less_aggregate_query_by_its_own_variable():
    tm = TenancyManager()

    scoped = tm.scope_cypher_query(
        "MATCH (w:WorkItem) RETURN count(w) AS c", tenant_id="acme"
    )

    assert "w.tenant_id = 'acme'" in scoped
    assert "n.tenant_id" not in scoped
    # Injected before RETURN, same discipline as the n-variable case. D-ACL-3:
    # the predicate also admits untagged/commons rows (IS NULL / '').
    assert scoped == (
        "MATCH (w:WorkItem) WHERE (w.tenant_id = 'acme' OR w.tenant_id IS NULL "
        "OR w.tenant_id = '') RETURN count(w) AS c"
    )


def test_still_scopes_the_conventional_n_variable_case_unchanged():
    """Backward-compatible: the common `MATCH (n:...)` convention this
    function's own docstring example uses keeps working byte-for-byte
    (module the D-ACL-3 commons fallback added to the predicate itself)."""
    tm = TenancyManager()

    scoped = tm.scope_cypher_query("MATCH (n:Entity) RETURN n", tenant_id="acme")

    assert scoped == (
        "MATCH (n:Entity) WHERE (n.tenant_id = 'acme' OR n.tenant_id IS NULL "
        "OR n.tenant_id = '') RETURN n"
    )


def test_cypher_scope_and_sql_rls_commons_fallback_agree():
    """D-ACL-3: the Cypher-level tenant scope and the SQL RLS layer must
    agree on whether an untagged ('commons') row is visible.

    ``PostgreSQLBackend.rls_statements`` has always documented and enforced a
    three-way admit condition: ``tenant_id = <session tenant> OR tenant_id IS
    NULL OR tenant_id = ''`` (its own docstring: "that org's rows + commons
    (tenant_id '' / NULL)"). Before this fix, ``TenancyManager
    .scope_cypher_query`` injected a strict equality with no fallback, so a
    read running through the Cypher-scoping chokepoint silently denied
    commons rows that the exact same actor would see via a raw RLS-enforced
    SQL read — two different security mechanisms disagreeing about who can
    see what. This test fails if either side drifts from the other again.
    """
    from agent_utilities.knowledge_graph.backends.postgresql_backend import (
        PostgreSQLBackend,
    )

    rls_cond = " ".join(PostgreSQLBackend.rls_statements("Agent"))
    # Pin the SQL contract this test converges against so a change to RLS's
    # own commons fallback is caught here too, not just on the Cypher side.
    assert "tenant_id IS NULL OR tenant_id = ''" in rls_cond

    tm = TenancyManager()
    scoped = tm.scope_cypher_query("MATCH (n:Entity) RETURN n", tenant_id="acme")

    assert "n.tenant_id = 'acme'" in scoped
    assert "n.tenant_id IS NULL" in scoped
    assert "n.tenant_id = ''" in scoped


def test_multi_variable_join_scopes_by_the_first_matchs_variable():
    tm = TenancyManager()

    scoped = tm.scope_cypher_query(
        "MATCH (a:A)-[:REL]->(b:B) RETURN a, b", tenant_id="acme"
    )

    assert "a.tenant_id = 'acme'" in scoped


def test_fully_anonymous_query_fails_closed_instead_of_running_unscoped():
    """No bound variable exists to scope by. Two unsafe options were on the
    table — inject a reference to a variable that was never in the query (the
    original D-SH-4 bug, just with a different fabricated name), or silently
    run the query with NO tenant predicate at all (which returns matching rows
    from EVERY tenant — a cross-tenant leak, strictly worse than the original
    bug's silent-empty result). This module's contract takes neither: it
    refuses to execute an unscoped read (CONCEPT:AU-KG.backend.company-brain-write-guard)."""
    tm = TenancyManager()

    with pytest.raises(UnscopableQueryError):
        tm.scope_cypher_query("MATCH ()-[r]->() RETURN count(r) AS c", tenant_id="acme")


def test_unsafe_tenant_id_still_fails_closed_with_the_detected_variable():
    tm = TenancyManager()

    scoped = tm.scope_cypher_query(
        "MATCH (s:Skill) RETURN s", tenant_id="acme'; DROP EVERYTHING"
    )

    assert "s.tenant_id = '__no_such_tenant__'" in scoped
