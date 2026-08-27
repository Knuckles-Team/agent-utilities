"""Index naming and document shape — `DEC-CA-09`'s Contract section, typed.

(CONCEPT:AU-KG.retrieval.opensearch-cdc-indexer)

Index naming: ``kg-<tenant>-<objecttype>`` — one document per eg node.
Document shape (minimum fields), verbatim from `DEC-CA-09`::

    {"node_id": "string", "node_type": "string", "tenant": "string",
     "marking": "string", "content": "string (full-text)", "properties": {},
     "updated_lsn": "number"}

**One deliberate generalization, not a contradiction of the contract:**
`DEC-CA-09` writes ``"marking": "string"`` (singular), and CA-50's own
hand-authored DLS demo role used a single scalar (``marking: "restricted"``).
`knowledge_graph.ontology.permissioning.markings_for` returns a *set* of
marking names per node (a node may carry zero, one, or several mandatory
markings — the `Marking`/`propagate_markings` model is explicitly a set of
unordered compartments, not a total order). This module stores ``marking``
as an OpenSearch ``keyword`` field holding a JSON array of marking names
(``[]`` when the node carries none). A ``{"term": {"marking": "<name>"}}``
query — the exact shape `DEC-CA-09`'s contract and CA-50's demo role both
use — matches identically whether the field holds a bare string or an array
of keywords (OpenSearch's ``term``/``terms`` queries match any array
element), so the single-marking case behaves byte-identically to the
contract's literal string form; multi-marking nodes are simply supported
rather than truncated to one name. Not a `DEC-CA-09` refusal-rule
contradiction (the query DSL shape is unchanged); documented here per the
lane's re-verify-every-premise instruction.

``updated_lsn`` carries eg's own CDC ``seq`` (`DEC-CA-03`'s "the actual
ordering primitive `CdcHub::read`/P3's replay check use" — eg has no WAL/LSN
concept at all, see ``eg_stream::sink`` module doc) — named ``updated_lsn``
to match `DEC-CA-09`'s literal field name, valued from ``seq``.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

__all__ = [
    "INDEX_PREFIX",
    "INDEX_MAPPINGS",
    "sanitize_segment",
    "index_name",
    "alias_name",
    "tenant_wildcard",
    "DocumentShape",
    "build_document",
]

INDEX_PREFIX = "kg"

# Explicit mapping, passed to `OpenSearchClient.ensure_index` on first write
# (indexer.py). Measured live against CA-50's real deployed cluster
# (2026-08-26): OpenSearch's DYNAMIC mapping maps a bare JSON string field to
# `text` (with an auto `<field>.keyword` sub-field) — a `{"term": {"marking":
# "restricted"}}` query against a `text` field matches the ANALYZED TOKEN,
# not the exact string, which happened to still work in that live proof only
# because "restricted" tokenizes to itself under the standard analyzer. That
# is not a general guarantee (a marking name with mixed case, punctuation, or
# multiple words would NOT round-trip through the standard analyzer
# unchanged) and it also made a `terms` AGGREGATION on `node_id` fail
# outright with OpenSearch's real `search_phase_execution_exception` ("Text
# fields are not optimised for operations that require per-document field
# data... use a keyword field instead") — a REAL failure this lane's own
# live proof hit, not a hypothetical. Every identity/classification field
# `DEC-CA-09`'s DLS `term`/`terms` queries or an aggregation could target is
# therefore explicitly `keyword`, never left to dynamic inference.
INDEX_MAPPINGS: dict[str, Any] = {
    "properties": {
        "node_id": {"type": "keyword"},
        "node_type": {"type": "keyword"},
        "tenant": {"type": "keyword"},
        "marking": {"type": "keyword"},
        "content": {"type": "text"},
        "properties": {"type": "object", "enabled": True},
        "updated_lsn": {"type": "long"},
    }
}

# OpenSearch index names must be lowercase and may not contain most special
# characters. Segments are sanitized deterministically so the same
# (tenant, object_type) pair always maps to the same index name.
_INVALID_CHARS = re.compile(r"[^a-z0-9_.-]+")


def sanitize_segment(value: str) -> str:
    """Lowercase ``value`` and replace anything not index-name-safe with ``_``.

    Never returns an empty string for non-empty input containing at least one
    safe character; callers must still reject a genuinely empty segment
    (an empty tenant/object_type is a data-quality failure the caller must
    surface, never silently substituted with a synthesized name).
    """
    cleaned = _INVALID_CHARS.sub("_", str(value).strip().lower())
    return cleaned.strip("_") or cleaned


def index_name(tenant: str, object_type: str) -> str:
    """``kg-<tenant>-<objecttype>`` per `DEC-CA-09`'s Contract."""
    tenant_s = sanitize_segment(tenant)
    type_s = sanitize_segment(object_type)
    if not tenant_s or not type_s:
        raise ValueError(
            f"index_name requires a non-empty tenant and object_type "
            f"(got tenant={tenant!r}, object_type={object_type!r})"
        )
    return f"{INDEX_PREFIX}-{tenant_s}-{type_s}"


def alias_name(tenant: str, object_type: str) -> str:
    """One alias per object type (`DEC-CA-09`: "supports reindex-without-downtime").

    Today the alias and the index share a name 1:1 (no generation suffix
    yet — `.rebuild` creates a fresh generation and swaps the alias only when
    a rebuild is actually running; the steady-state name is stable).
    """
    return index_name(tenant, object_type)


def tenant_wildcard(tenant: str) -> str:
    """``kg-<tenant>-*`` — every object-type index for one tenant.

    Used by the indexer's tombstone path (a delete carries no reliable
    object-type once ``before`` is a bare id) and by rebuild/audit tooling
    that must reach "every index this tenant owns" without enumerating
    object types by hand.
    """
    tenant_s = sanitize_segment(tenant)
    if not tenant_s:
        raise ValueError(
            f"tenant_wildcard requires a non-empty tenant (got {tenant!r})"
        )
    return f"{INDEX_PREFIX}-{tenant_s}-*"


class DocumentShape(TypedDict):
    node_id: str
    node_type: str
    tenant: str
    marking: list[str]
    content: str
    properties: dict[str, Any]
    updated_lsn: int


def build_document(
    *,
    node_id: str,
    node_type: str,
    tenant: str,
    marking: Any,
    properties: dict[str, Any],
    updated_lsn: int,
    content: str = "",
) -> DocumentShape:
    """Build one `DEC-CA-09`-shaped document. Never guesses a missing field —
    callers resolve ``node_type``/``tenant``/``marking`` before calling this.
    """
    if not node_id:
        raise ValueError("build_document requires a non-empty node_id")
    if not node_type:
        raise ValueError("build_document requires a non-empty node_type")
    if not tenant:
        raise ValueError("build_document requires a non-empty tenant")
    marking_list = sorted({str(m).strip() for m in (marking or []) if str(m).strip()})
    return DocumentShape(
        node_id=str(node_id),
        node_type=str(node_type),
        tenant=str(tenant),
        marking=marking_list,
        content=str(content or ""),
        properties=dict(properties or {}),
        updated_lsn=int(updated_lsn),
    )
