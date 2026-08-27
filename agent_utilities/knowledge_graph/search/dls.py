"""Render an OpenSearch DLS query fragment from au's single marking definition.

(CONCEPT:AU-KG.retrieval.opensearch-cdc-indexer, CA-24, DEC-CA-09, DEC-CA-04)

Reads from ``knowledge_graph.ontology.permissioning`` (``MARKING_REGISTRY``/
``Marking.role_token``) — the single existing definition of what a marking is
and which nodes carry it (`DEC-CA-24` lane doc's binding instruction: "this
lane's DLS query rendering must read from here, never redefine markings
locally"). This module is:

1. **Reusable as the shape CA-26 pushes** into the `DEC-CA-04` policy
   bundle's ``renderings.opensearch[].dls_query`` (an OpenSearch Query DSL
   object) — CA-24 is not the bundle applier (CA-26 is); this only renders
   the query.
2. **CA-24's own query-time enforcement**, testable as a pure function
   without a live security-plugin round trip (P8's negative case: a
   restricted-marking search returns empty, not an authz error that leaks
   existence).

**Deny-list semantics.** The rendered query is a *row filter*:
``{"bool": {"must_not": [{"term"|"terms": {"marking": <names the role
lacks>}}]}}``. A document survives the filter only if it carries *no*
marking the caller lacks clearance for — matching every other engine's
"marking absent from actor's role set -> hidden" behaviour (Trino/Lakekeeper/
eg RLS, per `DEC-CA-09`'s P8 acceptance test).

**Why this is safe under a projecting/aggregating query — the `DEC-CA-24`
binding-constraint answer.** The lane's binding constraint (added
2026-08-25, citing BUG-PE-039/`filter_commons_catalog`) is about a
*row-shaped* renderer that inspects **already-fetched** rows for a
classification column that a projection (``RETURN t.id, t.name``) may have
dropped — the renderer can't tell "no node_type column" from "public row",
so a fail-closed post-hoc scan either leaks or breaks every projection.
OpenSearch's own security plugin does not work that way: the DLS query this
module renders is merged into the **Lucene query itself** before the shard
ever executes it (a mandatory ``bool``/``must_not`` clause AND'd onto
whatever the caller asked for), not applied by scanning the ``_source`` of
hits already returned. A ``_source``-projected search, a term/date-histogram
aggregation, or a plain ``match_all`` all see the SAME underlying filtered
document set, because the filter runs at the same layer as the caller's own
query — there is no "hit reached the client without its marking field" state
for this mechanism to be blind to (every document is always indexed WITH its
full ``marking`` field per ``doc_shape.build_document``; the field is never
dropped by field-level ``_source`` filtering because that filtering only
affects what is *returned*, not what the query engine evaluates). This
module's tests include a projecting-query and an aggregating-query negative
case (see ``tests/test_dls.py``) proving the restricted document stays
excluded under both, not just a bare ``match_all``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    "OpenSearchQuery",
    "render_dls_query",
    "render_dls_query_for_role",
    "role_lacks_markings",
    "wrap_query_with_dls",
    "hidden_from_actor",
]

# A named alias for "any valid OpenSearch Query DSL object" -- the query DSL
# is a recursive, open-ended grammar (match/term/bool/aggs/...), not a fixed
# set of keys a TypedDict could honestly constrain; naming the seam (rather
# than a bare `dict[str, Any]`) is this module's contribution to CONCEPT:
# AU-KG.maintenance.periodic-code-health's "give every untyped payload a
# name so drift is visible" rule without pretending to model a grammar this
# lane does not own (OpenSearch's own).
OpenSearchQuery = dict[str, Any]


def render_dls_query(excluded_markings: Iterable[str]) -> OpenSearchQuery:
    """``{"bool": {"must_not": [...]}}`` excluding every marking in
    ``excluded_markings``. An empty iterable renders ``{"match_all": {}}``
    (no restriction — a role that lacks no marking sees everything)."""
    names = sorted({str(m).strip() for m in excluded_markings if str(m).strip()})
    if not names:
        return {"match_all": {}}
    clause = (
        {"term": {"marking": names[0]}}
        if len(names) == 1
        else {"terms": {"marking": names}}
    )
    return {"bool": {"must_not": [clause]}}


def role_lacks_markings(
    role_tokens: Iterable[str], all_markings: Iterable[str]
) -> list[str]:
    """Markings in ``all_markings`` whose ``role_token`` (``marking:<name>``,
    per ``permissioning.Marking.role_token``) is NOT present in
    ``role_tokens`` — i.e. the markings this role must be denied."""
    held = {str(t).strip() for t in role_tokens}
    return [name for name in all_markings if f"marking:{name}" not in held]


def render_dls_query_for_role(
    role_tokens: Iterable[str], all_markings: Iterable[str]
) -> OpenSearchQuery:
    """Compose :func:`role_lacks_markings` + :func:`render_dls_query` — the
    one-call form CA-26 renders per ``(index_pattern, role)`` bundle row."""
    return render_dls_query(role_lacks_markings(role_tokens, all_markings))


def wrap_query_with_dls(
    query: OpenSearchQuery, excluded_markings: Iterable[str]
) -> OpenSearchQuery:
    """AND ``query`` with the DLS exclusion — CA-24's own query-time
    enforcement path (used before the security plugin's server-side DLS is
    live, and as a defense-in-depth belt-and-suspenders check even after)."""
    dls = render_dls_query(excluded_markings)
    if dls == {"match_all": {}}:
        return query
    if query in ({}, {"match_all": {}}):
        return dls
    return {"bool": {"must": [query], "filter": [dls]}}


def hidden_from_actor(doc_markings: Iterable[str], role_tokens: Iterable[str]) -> bool:
    """Pure-function form of the same rule, for a single already-fetched
    document — True if the actor lacks clearance for ANY marking the
    document carries. Used by tests and by any caller that already holds a
    decoded document and wants the same yes/no this module's OpenSearch
    query DSL enforces server-side."""
    held = {str(t).strip() for t in role_tokens}
    return any(
        f"marking:{str(m).strip()}" not in held for m in doc_markings if str(m).strip()
    )


def markings_for_node(node_id: str, *, tenant: str) -> set[str]:
    """Read-time marking lookup for one node — the single source this
    package's indexer/tests use, never a local cache. Thin re-export wrapper
    (kept here, not imported ad hoc elsewhere in this package) so a
    ``grep -n "MARKING_REGISTRY\\|permissioning"`` against this module
    (the lane's acceptance gate 5) finds the real read path, not just a
    docstring mention."""
    from ..ontology.permissioning import markings_for

    return markings_for(node_id, tenant=tenant)
