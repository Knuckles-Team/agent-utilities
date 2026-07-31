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
"""

import re

__all__ = ["UnscopableQueryError", "first_bound_node_variable"]

# Matches the variable of the FIRST `MATCH (var ...` node pattern in the query
# — `MATCH (n:Entity)`, `MATCH (n)`, `MATCH (n {prop: 1})`, and the first hop
# of a multi-hop pattern (`MATCH (n:Entity)-[:REL]->(m:Entity)`) all resolve to
# the same variable the historical hardcoded-`n` scoping assumed was always
# present. Anchoring on the FIRST bound variable, rather than assuming a name,
# is a strict improvement over the prior contract (which had no fallback at
# all) without claiming to solve multi-variable/multi-MATCH semantic scoping.
_MATCH_VAR_RE = re.compile(
    r"\bMATCH\b\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:)\{]", re.IGNORECASE
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
    """Return the variable bound by the query's first ``MATCH (var...`` clause.

    Args:
        query: The Cypher read query text.

    Returns:
        The bound node variable name to scope against.

    Raises:
        UnscopableQueryError: no ``MATCH (<var>...`` pattern is found — there
            is no variable this module can safely inject a predicate against.
    """
    match = _MATCH_VAR_RE.search(query)
    if not match:
        raise UnscopableQueryError(
            "Cannot derive a bound node variable to scope this query safely "
            "(no `MATCH (<var>...` pattern found); refusing to scope with a "
            "guessed variable name rather than silently under- or "
            "mis-scoping the read."
        )
    return match.group(1)
