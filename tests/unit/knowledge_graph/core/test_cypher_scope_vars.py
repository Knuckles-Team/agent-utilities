"""Tests for the shared Cypher primary-bound-variable detector (D-SH-4).

``reports/deferred/lane-skill-harvest.md`` D-SH-4: `TenancyManager
.scope_cypher_query` and `tenant_sharing.apply_visibility` both hardcoded a
`n` variable when injecting a tenant/visibility predicate into a caller's
Cypher query, even though real callers alias their node differently (e.g.
`MATCH (s:Skill) ...`, `MATCH (w:WorkItem) RETURN count(w) AS c`). Injecting
`n.tenant_id = ...` into a query where `n` is never bound makes the predicate
reference an undefined variable, which Cypher never matches -- the read
silently returns nothing (or a zero aggregate) instead of raising, exactly the
divergence this lane found between `engine.query_cypher` (goes through the
injection) and `engine.backend.execute` (the identical, un-scoped query).

This module tests the detector in isolation; the injection sites themselves
are covered by `test_company_brain_scope_cypher_query.py` (tenant scoping) and
`test_engine_query_aggregate_governance.py` (the aggregate visibility path).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.cypher_scope_vars import (
    primary_bound_variable,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("MATCH (n:Entity) RETURN n", "n"),
        ("MATCH (s:Skill) WHERE s.name = 'foo' RETURN s", "s"),
        ("MATCH (w:WorkItem) RETURN count(w) AS c", "w"),
        ("match (x:Doc) return x", "x"),  # case-insensitive MATCH keyword
        ("MATCH   (   spaced_var  :Label) RETURN spaced_var", "spaced_var"),
        (
            "OPTIONAL MATCH (o:Opt) WHERE o.flag = true RETURN o",
            "o",
        ),
        # First MATCH wins even with a later, differently-named join variable.
        ("MATCH (a:A)-[:REL]->(b:B) RETURN a, b", "a"),
    ],
)
def test_primary_bound_variable_detects_the_first_matchs_own_alias(query, expected):
    assert primary_bound_variable(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "MATCH () RETURN count(*) AS c",
        "MATCH (:Skill) RETURN count(*) AS c",
        "MATCH ()-[r]->() RETURN count(r) AS c",
        "",
        "RETURN 1",  # no MATCH clause at all
    ],
)
def test_primary_bound_variable_returns_none_for_anonymous_or_matchless_queries(
    query,
):
    assert primary_bound_variable(query) is None


def test_primary_bound_variable_is_not_confused_by_a_function_call():
    """A variable name that only appears inside a later function call (not the
    first MATCH's own node pattern) must not be picked up as the primary
    variable -- the anonymous first pattern here has no real bound variable."""
    assert (
        primary_bound_variable("MATCH ()-[r]->() RETURN count(r) AS c, sum(r.w)")
        is None
    )
