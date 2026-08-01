#!/usr/bin/python
from __future__ import annotations

"""Shared Cypher tenant/visibility-scoping variable resolution.

CONCEPT:AU-KG.backend.company-brain-write-guard — fail-closed query scoping.

Both :meth:`TenancyManager.scope_cypher_query`
(``agent_utilities/knowledge_graph/core/company_brain.py``) and
:func:`~agent_utilities.knowledge_graph.core.tenant_sharing.apply_visibility`
inject an ``AND``ed predicate that references a specific bound Cypher node
variable (``<var>.tenant_id = '...'``, ``<var>._owner_id = '...'``). Both
previously hardcoded the literal ``n`` — silently wrong for any query that
binds its node under a different variable name (``MATCH (x:Entity) RETURN
x``): the injected ``n.tenant_id = '...'`` references a variable the query
never bound. Depending on the read backend that is either a hard parse error
or — the dangerous case actually observed — a condition that evaluates to
false/null and **silently returns zero rows**, indistinguishable from "no
matching data" to the caller and impossible to distinguish from a correctly
enforced deny.

This module derives the query's actual first bound node variable instead of
guessing, and raises :class:`UnscopableQueryError` (fail closed) when it
cannot find one to scope against — refusing to execute an unscoped read is
always safer than silently injecting a predicate against an unbound name.

It also centralizes the actual ``WHERE``/``RETURN`` injection
(:func:`inject_and_predicate`) so both call sites share one fix for a second,
independent hole in the same injection: appending the scoping predicate as a
bare ``<cond> AND`` right after an *existing* ``WHERE`` is a textual
concatenation, not a parse — Cypher's ``AND`` binds tighter than ``OR``, so
``WHERE <cond> AND <a> OR <b>`` parses as ``(<cond> AND <a>) OR <b>``, and any
row matching ``<b>`` alone bypasses ``<cond>`` entirely regardless of tenant.
This is not hypothetical: the real callers this fix targets
(``find_relevant_policies``: ``WHERE p.name CONTAINS $q OR p.description
CONTAINS $q``; ``find_relevant_processes``: the analogous ``f.goal ... OR
f.name ...``) both have exactly this shape, so a query correctly bound to the
right variable would still leak cross-tenant rows through the ``OR`` branch
without this. :func:`inject_and_predicate` parenthesizes both the injected
predicate and the pre-existing ``WHERE`` body before ANDing them, so the
existing predicate is scoped as a whole, not just its first disjunct.

Convergence note (wave-2 tenant-predicate lane): this same hardcoded-``n``
defect was found and independently patched on multiple other branches —
``fix/green-slice-1`` (0ec7538a), ``fix/green-slice-3`` (77300375),
``fix/green-slice-7`` (a4fc1ef7, which additionally admits unstamped/NULL
``tenant_id`` rows as a "commons" partition — a broader policy change, not
adopted here), and ``fix/sweep-orch-skills``/``feat/retrieval-eval-policy``
(8faf9540 + 4e19249a, already reconciled on ``feat/retrieval-eval-policy`` at
merge commit 211c1f62 into a ``cypher_scope_vars.py`` detector +
``cypher_scoping.py`` fail-closed wrapper — the closest prior art to this
module, and where this module's names/API were deliberately kept identical
to minimize merge friction). This implementation supersedes ALL of the above
because none of them close two further holes real callers actually hit:

1. Every one of them still injects the scoping predicate as a bare
   ``<cond> AND <existing>`` splice with no parentheses around the
   pre-existing ``WHERE`` body — the ``OR``-precedence hole described above,
   which lets ``find_relevant_policies``/``find_relevant_processes`` leak
   cross-tenant rows through their second disjunct even once bound to the
   right variable. :func:`inject_and_predicate` is the fix.
2. Every one of them detects only the variable immediately after the first
   ``MATCH (`` — none walk a multi-hop first-``MATCH`` clause to a *later*
   bound node when the first node is anonymous. A real production query
   (``capabilities/kg_audit_sink.py::tool_call_rows_by_trace_node_id``:
   ``MATCH (:RunTrace {id: $tid})-[:USED_TOOL]->(t:ToolCall) RETURN ...``)
   has exactly this shape and would start raising ``UnscopableQueryError``
   (or silently degrade to an empty audit trail, since its caller
   ``retrieval/context_plane.py::read_rows`` swallows all exceptions) on
   every prior variant, including the reconciled one.
   :func:`first_bound_node_variable` fixes this by scanning every node
   pattern in the first ``MATCH`` clause, in order, for the first bound one.
"""

import re

__all__ = [
    "UnscopableQueryError",
    "first_bound_node_variable",
    "inject_and_predicate",
]

# The clause keywords that can trail a WHERE predicate in this codebase's
# query shapes (RETURN/WITH/ORDER BY/SKIP/LIMIT) — used to find where an
# existing WHERE clause's predicate text ends, so the whole thing can be
# parenthesized rather than just textually followed by "AND <new cond>"
# (which silently mis-groups against any top-level OR in that predicate).
_TRAILING_CLAUSE_RE = re.compile(
    r"\b(?:RETURN|WITH|ORDER\s+BY|SKIP|LIMIT)\b", re.IGNORECASE
)

# Boundary keywords that end the query's FIRST MATCH clause — either a
# following clause keyword, or a second MATCH/OPTIONAL MATCH starting a new
# one. Scanning is deliberately confined to this span: a later MATCH's own
# variables are that clause's concern, not this predicate's.
_MATCH_CLAUSE_BOUNDARY_RE = re.compile(
    r"\b(?:OPTIONAL\s+MATCH|MATCH|WHERE|RETURN|WITH|ORDER\s+BY|SKIP|LIMIT)\b",
    re.IGNORECASE,
)

# Matches EACH node pattern `(var)`, `(var:Label)`, `(:Label)`, `(var {..})`,
# `(:Label {..})` within a MATCH clause — capturing the bound variable, if
# any (``None``/empty for an anonymous node). A multi-hop pattern like
# `MATCH (:RunTrace {id: $tid})-[:USED_TOOL]->(t:ToolCall)` has its first node
# anonymous but a *later* node in the SAME clause bound to `t` — the real
# shape of e.g. capabilities/kg_audit_sink.py's tool-call-by-trace query.
# Deliberately not a general Cypher parser: relationship variables (`[r]`)
# are never node-scoping candidates and are not matched (no `(...)` around
# them); nested `{...}` property maps are matched non-greedily to the first
# `}`, sufficient for this codebase's query shapes without a real parser.
_NODE_PATTERN_RE = re.compile(
    r"\(\s*([A-Za-z_][A-Za-z0-9_]*)?\s*(?::[A-Za-z_][A-Za-z0-9_]*)*"
    r"\s*(?:\{[^{}]*\})?\s*\)"
)


class UnscopableQueryError(PermissionError):
    """A Cypher read query has no node variable this module can safely scope.

    Raised instead of injecting a predicate against a guessed/hardcoded
    variable name — the historical bug this replaces. A ``PermissionError``
    subclass so existing ``except PermissionError`` boundaries (e.g.
    :func:`~agent_utilities.knowledge_graph.core.secured_reads.scope`) still
    catch it and surface a typed denial rather than an unhandled exception.
    """


def first_bound_node_variable(query: str) -> str:
    """Return the first bound node variable in the query's first ``MATCH`` clause.

    Scans every node pattern in the first ``MATCH``/``OPTIONAL MATCH`` clause
    (in order) and returns the first one that actually binds a variable — not
    just the node immediately after the ``MATCH (``. A multi-hop pattern can
    have an anonymous first node (``MATCH (:RunTrace {id: $tid})-[:REL]->
    (t:ToolCall)``); the variable a later hop in the *same* clause binds is
    still a legitimate, safe scoping target.

    Args:
        query: The Cypher read query text.

    Returns:
        The bound node variable name to scope against.

    Raises:
        UnscopableQueryError: no ``MATCH``/``OPTIONAL MATCH`` clause is found,
            or every node pattern in its first clause is anonymous — there is
            no variable this module can safely inject a predicate against.
    """
    clause_match = re.search(r"\bMATCH\b", query, flags=re.IGNORECASE)
    if clause_match is None:
        raise UnscopableQueryError(
            "Cannot derive a bound node variable to scope this query safely "
            "(no `MATCH` clause found); refusing to scope with a guessed "
            "variable name rather than silently under- or mis-scoping the "
            "read."
        )
    clause_start = clause_match.end()
    boundary_match = _MATCH_CLAUSE_BOUNDARY_RE.search(query, clause_start)
    clause_end = boundary_match.start() if boundary_match else len(query)
    clause_text = query[clause_start:clause_end]

    for node_match in _NODE_PATTERN_RE.finditer(clause_text):
        var = node_match.group(1)
        if var:
            return var

    raise UnscopableQueryError(
        "Cannot derive a bound node variable to scope this query safely "
        "(every node pattern in the first `MATCH` clause is anonymous); "
        "refusing to scope with a guessed variable name rather than "
        "silently under- or mis-scoping the read."
    )


def inject_and_predicate(query: str, cond: str) -> str:
    """AND ``cond`` into ``query``'s ``WHERE`` clause, or add one before ``RETURN``.

    Shared by :meth:`~agent_utilities.knowledge_graph.core.company_brain
    .TenancyManager.scope_cypher_query` and
    :func:`~agent_utilities.knowledge_graph.core.tenant_sharing
    .apply_visibility`. When ``query`` already has a ``WHERE`` clause, its
    existing predicate text (up to the next ``RETURN``/``WITH``/``ORDER
    BY``/``SKIP``/``LIMIT``, or end of string) is parenthesized as a unit
    before ANDing in ``cond`` — ``(cond) AND (existing)`` — instead of a bare
    ``cond AND existing`` textual splice, which would silently mis-group
    against any top-level ``OR`` already in ``existing`` (see module
    docstring). Queries with neither ``WHERE`` nor ``RETURN`` (writes/DDL)
    are returned unchanged — there is no predicate site to inject into.
    """
    # `cond` may already arrive as a single fully-parenthesized atom (e.g.
    # `tenant_sharing.visibility_predicate`'s own compound OR-predicate) —
    # avoid a redundant extra wrap in that case, purely cosmetic (an extra
    # pair is semantically harmless either way).
    cond_wrapped = cond if cond.startswith("(") and cond.endswith(")") else f"({cond})"

    where_match = re.search(r"\bWHERE\b", query, flags=re.IGNORECASE)
    if where_match:
        boundary_match = _TRAILING_CLAUSE_RE.search(query, where_match.end())
        boundary = boundary_match.start() if boundary_match else len(query)
        existing = query[where_match.end() : boundary].strip()
        rest = query[boundary:]
        if rest and not rest[:1].isspace():
            rest = " " + rest
        if existing:
            replacement = f" {cond_wrapped} AND ({existing})"
        else:
            replacement = f" {cond}"
        return query[: where_match.end()] + replacement + rest

    return_match = re.search(r"\bRETURN\b", query, flags=re.IGNORECASE)
    if return_match:
        return (
            query[: return_match.start()]
            + f"WHERE {cond} "
            + query[return_match.start() :]
        )
    return query
