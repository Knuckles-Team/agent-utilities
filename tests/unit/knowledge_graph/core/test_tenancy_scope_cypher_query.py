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

D-W2T-2: `scope_cypher_query` now returns ``(scoped_query, extra_params)`` —
the tenant id is injected as a bound ``$_tenant_scope_id`` Cypher parameter,
not spliced into the query text as a string literal.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain import TenancyManager
from agent_utilities.knowledge_graph.core.cypher_scoping import UnscopableQueryError


def test_scopes_by_the_querys_own_variable_not_a_hardcoded_n():
    tm = TenancyManager()

    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (s:Skill) WHERE s.name = 'foo' RETURN s", tenant_id="acme"
    )

    assert "s.tenant_id = $_tenant_scope_id" in scoped
    assert "n.tenant_id" not in scoped  # the D-SH-4 bug: referenced an undefined var
    assert extra_params == {"_tenant_scope_id": "acme"}
    assert "acme" not in scoped


def test_scopes_a_where_less_aggregate_query_by_its_own_variable():
    tm = TenancyManager()

    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (w:WorkItem) RETURN count(w) AS c", tenant_id="acme"
    )

    assert "w.tenant_id = $_tenant_scope_id" in scoped
    assert "n.tenant_id" not in scoped
    # Injected before RETURN, same discipline as the n-variable case.
    assert scoped == (
        "MATCH (w:WorkItem) WHERE (w.tenant_id = $_tenant_scope_id OR w.tenant_id IS NULL OR w.tenant_id = '') RETURN count(w) AS c"
    )
    assert extra_params == {"_tenant_scope_id": "acme"}


def test_still_scopes_the_conventional_n_variable_case_unchanged():
    """Backward-compatible: the common `MATCH (n:...)` convention this
    function's own docstring example uses keeps working byte-for-byte."""
    tm = TenancyManager()

    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (n:Entity) RETURN n", tenant_id="acme"
    )

    assert (
        scoped
        == "MATCH (n:Entity) WHERE (n.tenant_id = $_tenant_scope_id OR n.tenant_id IS NULL OR n.tenant_id = '') RETURN n"
    )
    assert extra_params == {"_tenant_scope_id": "acme"}


def test_multi_variable_join_scopes_by_the_first_matchs_variable():
    tm = TenancyManager()

    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (a:A)-[:REL]->(b:B) RETURN a, b", tenant_id="acme"
    )

    assert "a.tenant_id = $_tenant_scope_id" in scoped
    assert extra_params == {"_tenant_scope_id": "acme"}


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

    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (s:Skill) RETURN s", tenant_id="acme'; DROP EVERYTHING"
    )

    assert "s.tenant_id = $_tenant_scope_id" in scoped
    assert extra_params == {"_tenant_scope_id": "__no_such_tenant__"}
