#!/usr/bin/python
from __future__ import annotations

"""Fail-closed wrapper over the shared Cypher scoping-variable detector.

CONCEPT:AU-KG.backend.company-brain-write-guard — fail-closed query scoping.

D-SH-4 / this lane's own finding were the SAME bug reached from two sessions:
:meth:`TenancyManager.scope_cypher_query`
(``agent_utilities/knowledge_graph/core/company_brain.py``) and
:func:`~agent_utilities.knowledge_graph.core.tenant_sharing.apply_visibility`
both injected an ``AND``ed predicate (``<var>.tenant_id = '...'``,
``<var>._owner_id = '...'``) against a hardcoded ``n`` — silently wrong for any
query binding its node under a different variable name (``MATCH (x:Entity)
RETURN x``, ``MATCH (s:Skill) ... RETURN s``). Depending on the read backend
that is either a hard parse error or — the case actually observed — a
condition that evaluates to false/null and **silently returns zero rows**,
indistinguishable from "no matching data" and impossible to distinguish from a
correctly enforced deny.

The detection itself lives in :mod:`.cypher_scope_vars`
(:func:`~.cypher_scope_vars.primary_bound_variable` — the canonical, single
detector for both call sites, handling ``OPTIONAL MATCH`` and avoiding
false-positives from a variable name appearing only inside a later function
call). This module adds the FAIL-CLOSED contract on top: when the detector
returns ``None`` (a fully anonymous first pattern, or no ``MATCH`` at all),
:func:`first_bound_node_variable` **raises** :class:`UnscopableQueryError`
rather than silently leaving a ``WHERE``/``RETURN``-bearing query unscoped.
Leaving such a query unscoped is not a safe degrade — it means the read
returns matching rows from EVERY tenant, not zero: strictly worse than the
original silent-empty bug for cross-tenant isolation. Refusing to execute an
unscoped read is always safer than either guessing a variable or running the
query with no tenant/visibility predicate at all.
"""

from .cypher_scope_vars import primary_bound_variable

__all__ = ["UnscopableQueryError", "first_bound_node_variable"]


class UnscopableQueryError(PermissionError):
    """A Cypher read query has no node variable this module can safely scope.

    Raised instead of injecting a predicate against a guessed/hardcoded
    variable name, AND instead of silently leaving the query unscoped (which
    would return rows from every tenant, not zero). A ``PermissionError``
    subclass so existing ``except PermissionError`` boundaries (e.g.
    :func:`~agent_utilities.knowledge_graph.core.secured_reads.scope`) still
    catch it and surface a typed denial rather than an unhandled exception.
    """


def first_bound_node_variable(query: str) -> str:
    """Return the variable bound by the query's first ``MATCH (var...`` clause.

    Args:
        query: The Cypher read query text.

    Returns:
        The bound node variable name to scope against
        (:func:`~.cypher_scope_vars.primary_bound_variable`).

    Raises:
        UnscopableQueryError: the query's first ``MATCH``/``OPTIONAL MATCH``
            pattern is fully anonymous, or there is no ``MATCH`` clause at
            all — there is no variable this module can safely inject a
            predicate against, and leaving the query unscoped would be a
            cross-tenant/visibility leak, not a safe default.
    """
    var = primary_bound_variable(query)
    if var is None:
        raise UnscopableQueryError(
            "Cannot derive a bound node variable to scope this query safely "
            "(no `MATCH (<var>...` pattern found, or its first pattern is "
            "anonymous); refusing to scope with a guessed variable name OR "
            "to run the query unscoped."
        )
    return var
