"""Detect the primary bound node variable in a Cypher read query.

CONCEPT:AU-KG.query.cypher-scope-variable-detection

D-SH-4 (``reports/deferred/lane-skill-harvest.md``): both
:meth:`~agent_utilities.knowledge_graph.core.company_brain.TenancyManager.scope_cypher_query`
and :func:`~agent_utilities.knowledge_graph.core.tenant_sharing.apply_visibility`
injected a tenant/visibility predicate written against a hardcoded ``n``
variable (``n.tenant_id = '...'``), on the assumption every read query aliases
its primary node ``n`` (as the module docstrings' own examples do:
``MATCH (n:Entity) RETURN n``). Real callers do not all follow that
convention — e.g. ``MATCH (s:Skill) ... RETURN s`` or
``MATCH (w:WorkItem) RETURN count(w) AS c`` — so the injected predicate
referenced a variable that does not exist anywhere in the query. Cypher
engines evaluate a reference to an unbound variable as never matching, so the
query silently returned zero rows (or, for an aggregate projection, a single
row with a zero count) instead of raising — exactly the divergence this lane
found between ``engine.query_cypher`` (goes through this injection) and
``engine.backend.execute`` with the identical, un-scoped query text.

This module is a small, dependency-free (no other project imports, to avoid
any import-cycle risk between ``company_brain.py`` and ``tenant_sharing.py``)
regex-based detector for "the variable this query's own tenant/visibility
predicate should be written against" — the first node-pattern variable bound
by the query's own (first) ``MATCH``/``OPTIONAL MATCH`` clause, which is
exactly the variable a caller who *did* follow the ``n`` convention would have
used. It replaces two independent hardcoded ``"n"`` defaults with one shared,
correct detector.

Deliberately NOT a general Cypher parser: it only looks at the node pattern
immediately following the first ``MATCH``/``OPTIONAL MATCH`` keyword, so it
cannot be confused by a variable name that happens to appear inside a later
function call (``count(w)``, ``sum(a.value)``) or property map. A fully
anonymous first pattern (``MATCH () ...``, ``MATCH (:Label) ...``) has no
variable to scope by; callers must treat ``None`` as "cannot inject a
variable-qualified predicate for this query" rather than guessing.
"""

from __future__ import annotations

import re

# Matches the (OPTIONAL) MATCH keyword immediately followed by a node pattern's
# opening paren, optional whitespace, and an optional bound variable name.
# The variable name — if present — is always the FIRST thing inside the node
# pattern (before any ``:Label`` or ``{prop: ...}``), so capturing greedily
# from `\(` is unambiguous.
_FIRST_MATCH_NODE_RE = re.compile(
    r"\bMATCH\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)?",
    re.IGNORECASE,
)


def primary_bound_variable(query: str) -> str | None:
    """Return the first bound node-pattern variable in ``query``'s first MATCH.

    Returns ``None`` when the query has no ``MATCH``/``OPTIONAL MATCH`` clause,
    or when that clause's first node pattern is anonymous (``()``/``(:Label)``)
    — there is then no variable name a textual predicate can be written
    against, and callers must not inject one (fail-closed callers should skip
    scoping rather than reference a fabricated name).
    """
    match = _FIRST_MATCH_NODE_RE.search(query or "")
    if match is None:
        return None
    return match.group(1)
