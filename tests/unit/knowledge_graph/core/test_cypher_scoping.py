"""Regression tests for the tenant-predicate hardcoded-variable defect.

Production bug: ``TenancyManager.scope_cypher_query`` (company_brain.py) and
``tenant_sharing.apply_visibility`` both hardcoded the injected tenant/owner
predicate to the literal Cypher variable ``n`` (``n.tenant_id = '<tenant>'``),
regardless of what variable name a caller's ``MATCH`` pattern actually bound.
Real callers do not all use ``n`` -- ``find_relevant_policies``
(``MATCH (p:Policy) ...``) and ``find_relevant_processes``
(``MATCH (f:ProcessFlow) ...``) bind ``p``/``f``. A predicate written against
an unbound ``n`` is either a hard binder error, or -- the dangerous case this
module guards against -- silently evaluates to false/null on a lenient
backend, in which case whether the read is scoped AT ALL depends entirely on
how the rest of the query's own ``WHERE`` clause is shaped.

This file:
    1. Proves the fix derives the query's OWN bound variable
       (``cypher_scoping.first_bound_node_variable``), not ``n``.
    2. Proves fail-closed behaviour: a query with no derivable bound variable
       raises ``UnscopableQueryError`` rather than emitting an unscoped read.
    3. Proves a second, independent hole in the same injection is also
       closed: appending the tenant predicate as a bare ``<cond> AND`` before
       an *existing* ``WHERE`` body mis-groups against any top-level ``OR``
       already in that body (Cypher's ``AND`` binds tighter than ``OR``), so
       ``WHERE <cond> AND <a> OR <b>`` parses as ``(<cond> AND <a>) OR <b>``
       -- letting any row matching ``<b>`` alone bypass the tenant predicate
       entirely, regardless of variable correctness. This is not
       hypothetical: ``find_relevant_policies``/``find_relevant_processes``
       both use exactly this ``WHERE ... OR ...`` shape.
    4. Evaluates the actual predicate text the fix emits against synthetic
       two-tenant data (a tiny closed-form evaluator for this module's own
       predicate subset) to prove -- not just assert on substrings -- that a
       ``p``/``f``-bound, ``OR``-shaped read scoped for tenant A does not
       admit tenant B's matching rows.
"""

from __future__ import annotations

import re

import pytest

from agent_utilities.knowledge_graph.core.company_brain import TenancyManager
from agent_utilities.knowledge_graph.core.cypher_scoping import (
    UnscopableQueryError,
    first_bound_node_variable,
    inject_and_predicate,
)

# ---------------------------------------------------------------------------
# 1. Variable derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("MATCH (n:Entity) RETURN n", "n"),
        ("MATCH (p:Policy) WHERE p.name CONTAINS $q RETURN p", "p"),
        ("MATCH (f:ProcessFlow) RETURN f", "f"),
        ("MATCH (w:WorkItem) RETURN count(w) AS c", "w"),
        ("OPTIONAL MATCH (o:Opt) WHERE o.flag = true RETURN o", "o"),
        ("MATCH (a:A)-[:REL]->(b:B) RETURN a, b", "a"),
        # Real production shape (capabilities/kg_audit_sink.py
        # tool_call_rows_by_trace_node_id): the first node in the MATCH
        # clause is anonymous, but a later hop in the SAME clause binds `t`
        # -- that is still a safe, legitimate scoping variable, not a case
        # that should fail closed.
        (
            "MATCH (:RunTrace {id: $tid})-[:USED_TOOL]->(t:ToolCall) "
            "RETURN t.tool_call_id AS tool_call_id ORDER BY t.sequence ASC",
            "t",
        ),
    ],
)
def test_first_bound_node_variable_derives_the_querys_own_variable(query, expected):
    assert first_bound_node_variable(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "MATCH () RETURN count(*) AS c",
        "MATCH (:Skill) RETURN count(*) AS c",
        "MATCH ()-[r]->() RETURN count(r) AS c",
        "RETURN 1",
        "",
    ],
)
def test_first_bound_node_variable_fails_closed_when_unresolvable(query):
    with pytest.raises(UnscopableQueryError):
        first_bound_node_variable(query)


# ---------------------------------------------------------------------------
# 2. scope_cypher_query: real-shaped queries, not a hardcoded n
# ---------------------------------------------------------------------------


def test_scope_cypher_query_scopes_find_relevant_policies_by_its_own_variable():
    """find_relevant_policies: MATCH (p:Policy) ... -- must scope by `p`."""
    tm = TenancyManager()
    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (p:Policy) WHERE p.name CONTAINS $q OR p.description CONTAINS $q RETURN p",
        tenant_id="tenant-a",
    )
    assert "p.tenant_id = $_tenant_scope_id" in scoped
    assert "n.tenant_id" not in scoped  # the historical hardcoded-n bug
    # D-W2T-2: the tenant id is a bound parameter, never spliced into the text.
    assert extra_params == {"_tenant_scope_id": "tenant-a"}
    assert "tenant-a" not in scoped


def test_scope_cypher_query_scopes_find_relevant_processes_by_its_own_variable():
    """find_relevant_processes: MATCH (f:ProcessFlow) ... -- must scope by `f`."""
    tm = TenancyManager()
    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (f:ProcessFlow) WHERE f.goal CONTAINS $q OR f.name CONTAINS $q RETURN f",
        tenant_id="tenant-a",
    )
    assert "f.tenant_id = $_tenant_scope_id" in scoped
    assert "n.tenant_id" not in scoped
    assert extra_params == {"_tenant_scope_id": "tenant-a"}


def test_scope_cypher_query_conventional_n_variable_unchanged():
    """Backward-compatible: the common `MATCH (n:...)` shape keeps working."""
    tm = TenancyManager()
    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (n:Entity) RETURN n", tenant_id="tenant-a"
    )
    assert (
        scoped
        == "MATCH (n:Entity) WHERE (n.tenant_id = $_tenant_scope_id OR n.tenant_id IS NULL OR n.tenant_id = '') RETURN n"
    )
    assert extra_params == {"_tenant_scope_id": "tenant-a"}


def test_scope_cypher_query_fails_closed_for_an_unscopable_query():
    """No bound node variable at all -- refuse the read, do not emit it unscoped."""
    tm = TenancyManager()
    with pytest.raises(UnscopableQueryError):
        tm.scope_cypher_query(
            "MATCH ()-[r]->() RETURN count(r) AS c", tenant_id="tenant-a"
        )


def test_scope_cypher_query_scopes_the_bound_edge_count_shape():
    """D-W2T-1: routers/enhanced.py + routers/commands.py's global-edge-count
    query used to be the exact anonymous `MATCH ()-[r]->()` shape the previous
    test proves is unscopable -- so /graph/stats and 'kg stats' always failed
    with UnscopableQueryError. Both call sites now bind the source node as `a`
    (``MATCH (a)-[r]->() RETURN count(r) AS c``), which this proves IS scopable.
    """
    tm = TenancyManager()
    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (a)-[r]->() RETURN count(r) AS c", tenant_id="tenant-a"
    )
    assert "a.tenant_id = $_tenant_scope_id" in scoped
    assert "n.tenant_id" not in scoped
    assert extra_params == {"_tenant_scope_id": "tenant-a"}


# ---------------------------------------------------------------------------
# 3. inject_and_predicate: the OR-precedence hole
# ---------------------------------------------------------------------------


def test_inject_and_predicate_parenthesizes_existing_or_body():
    """A bare `<cond> AND <existing>` splice would mis-group against a
    top-level OR in `<existing>` and let its second disjunct bypass `<cond>`
    entirely. The fix must parenthesize the existing body as a unit."""
    out = inject_and_predicate(
        "MATCH (p:Policy) WHERE p.name CONTAINS 'x' OR p.description CONTAINS 'x' RETURN p",
        "p.tenant_id = 'tenant-a'",
    )
    assert (
        "MATCH (p:Policy) WHERE (p.tenant_id = 'tenant-a') AND (p.name CONTAINS 'x' OR p.description CONTAINS 'x') RETURN p"
        == out
    )


def test_scope_cypher_query_closes_the_or_bypass_for_find_relevant_policies():
    tm = TenancyManager()
    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (p:Policy) WHERE p.name CONTAINS $q OR p.description CONTAINS $q RETURN p",
        tenant_id="tenant-a",
    )
    # The injected tenant predicate and the pre-existing OR body must each be
    # their own parenthesized unit -- NOT `p.tenant_id = $_tenant_scope_id AND
    # p.name CONTAINS $q OR p.description CONTAINS $q`, which would let any
    # row matching the description clause alone bypass tenant scoping.
    assert (
        "(p.tenant_id = $_tenant_scope_id OR p.tenant_id IS NULL OR p.tenant_id = '') "
        "AND (p.name CONTAINS $q OR p.description CONTAINS $q)" in scoped
    )
    assert extra_params == {"_tenant_scope_id": "tenant-a"}


# ---------------------------------------------------------------------------
# 4. Evaluate the emitted predicate against synthetic two-tenant data
# ---------------------------------------------------------------------------


def _eval_predicate(predicate: str, row: dict, params: dict | None = None) -> bool:
    """Closed-form evaluator for the tiny WHERE-predicate subset this module
    emits (`var.prop = 'val'`, `var.prop CONTAINS 'val'`, composed with
    AND/OR/parens) -- proves whether a GIVEN ROW is actually matched by the
    predicate text this module generated, not just that the text looks right.
    """
    params = params or {}
    expr = predicate
    # `IS NULL` must be translated BEFORE the `=` rule, and bound parameters
    # ($_tenant_scope_id) resolved from `params`, or the D-ACL-3 commons
    # fallback predicate reaches eval() as raw Cypher and SyntaxErrors.
    expr = re.sub(r"\w+\.(\w+)\s+IS\s+NULL", r"(row.get('\1') is None)", expr)
    expr = re.sub(
        r"\w+\.(\w+)\s*=\s*\$(\w+)",
        lambda m: f"(row.get({m.group(1)!r}) == {params.get(m.group(2))!r})",
        expr,
    )
    expr = re.sub(r"\w+\.(\w+)\s*=\s*'([^']*)'", r"(row.get('\1') == '\2')", expr)
    expr = re.sub(
        r"\w+\.(\w+)\s+CONTAINS\s+'([^']*)'",
        r"('\2' in (row.get('\1') or ''))",
        expr,
    )
    expr = re.sub(r"\bAND\b", "and", expr)
    expr = re.sub(r"\bOR\b", "or", expr)
    return bool(eval(expr, {"__builtins__": {}}, {"row": row}))  # noqa: S307 -- closed vocabulary, test-only


def test_tenant_a_read_does_not_match_tenant_bs_row_via_the_or_clause():
    """The core regression: a tenant-scoped read binding a non-`n` variable
    (`p`), with a top-level OR in its own predicate (the real
    find_relevant_policies shape), must NOT match another tenant's row --
    neither via the historical hardcoded-`n` bug nor via the OR-precedence
    bypass that survives even a correct variable fix without parenthesization.
    """
    tm = TenancyManager()
    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (p:Policy) WHERE p.name CONTAINS $q OR p.description CONTAINS $q RETURN p",
        tenant_id="tenant-a",
    )
    # Extract the WHERE...RETURN predicate text and bind the search term the
    # way the real (parameterized) $q/$_tenant_scope_id placeholders would be
    # at execution time.
    predicate = re.search(r"WHERE (.*) RETURN", scoped).group(1)
    predicate = predicate.replace("$q", "'refund'")
    predicate = predicate.replace(
        "$_tenant_scope_id", f"'{extra_params['_tenant_scope_id']}'"
    )

    tenant_a_row = {"tenant_id": "tenant-a", "name": "refund policy", "description": ""}
    tenant_b_row = {"tenant_id": "tenant-b", "name": "refund policy", "description": ""}

    assert _eval_predicate(predicate, tenant_a_row) is True
    assert _eval_predicate(predicate, tenant_b_row) is False


def test_tenant_bs_row_does_not_leak_through_the_second_disjunct():
    """Specifically exercises the OR-precedence hole: a tenant-B row that
    matches ONLY the second disjunct (`description`) must still be excluded
    from a tenant-A-scoped read, not silently admitted because the injected
    tenant predicate only bound to the first disjunct."""
    tm = TenancyManager()
    scoped, extra_params = tm.scope_cypher_query(
        "MATCH (p:Policy) WHERE p.name CONTAINS $q OR p.description CONTAINS $q RETURN p",
        tenant_id="tenant-a",
    )
    predicate = re.search(r"WHERE (.*) RETURN", scoped).group(1)
    predicate = predicate.replace("$q", "'refund'")
    predicate = predicate.replace(
        "$_tenant_scope_id", f"'{extra_params['_tenant_scope_id']}'"
    )

    tenant_b_row_matching_only_description = {
        "tenant_id": "tenant-b",
        "name": "Unrelated",
        "description": "our refund process",
    }
    assert _eval_predicate(predicate, tenant_b_row_matching_only_description) is False
