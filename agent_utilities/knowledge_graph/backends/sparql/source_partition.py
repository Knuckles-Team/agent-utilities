#!/usr/bin/python
from __future__ import annotations

"""Source-partitioned named-graph routing for SPARQL triple stores.

CONCEPT:AU-KG.query.vendor-agnostic-traversal — Vendor-Agnostic Graph Backend Abstraction.

A SPARQL store can hold many *named graphs*. When KG instance data lands in a
triplestore (Stardog), partitioning it by the **source system** it came from
(LeanIX, ServiceNow, …) keeps provenance explicit and lets an operator push,
query, or clear one source's slice without touching the rest.

This module is the single source of truth for that routing decision. It is used
by BOTH the live write path (the Stardog backend's Cypher→SPARQL translation) and
the explicit per-source push serializer, so the two never disagree about which
named graph a node/edge belongs in.

The source is read from the ``source_system`` property, which
:meth:`IntelligenceGraphEngine.ingest_external_batch` stamps on every externally
ingested node/edge with the connector domain (``"leanix"`` / ``"servicenow"`` …).
Anything lacking a real source — including the engine's generic ``"system"``
provenance tag on internal edges — lands in the **default graph**.
"""

import re
from typing import Any

# The named-graph prefix for source-partitioned data. ``urn:`` (not http) so a
# named-graph IRI is never mistaken for a dereferenceable resource.
SOURCE_GRAPH_PREFIX = "urn:source:"

# Property keys that carry a source-system identity, in priority order. The first
# present, non-generic value wins.
_SOURCE_KEYS = ("source_system", "ingested_from", "source")

# Generic provenance values that are NOT a real external source. The engine stamps
# ``source="system"`` on internal edges (link_nodes); that must not create a
# ``urn:source:system`` named graph.
_GENERIC_SOURCES = frozenset({"", "system", "internal", "kg"})

# ── Canonical source-id naming schema (CONCEPT:AU-KG.ingest.source-id-naming-schema) ──
#
# EVERY connector's source id follows ONE hierarchical, colon-delimited grammar so a
# named graph is predictable and an operator can query/clear a whole system, a single
# instance, or a sub-kind without guessing per-connector spelling:
#
#     <system>[:<instance>][:<kind>]        →  urn:source:<system>[:<instance>][:<kind>]
#
#   * ``system``   — the connector family (``leanix``/``servicenow``/``gitlab``/``code`` …).
#                    ONE of :data:`KNOWN_SOURCE_SYSTEMS` for a built-in connector.
#   * ``instance`` — OPTIONAL, disambiguates one deployment of a multi-instance system
#                    (a GitLab host, a Confluence site, a Plane workspace, a DockerHub
#                    namespace, a repo for ``code``).
#   * ``kind``     — OPTIONAL sub-partition within a (system, instance) — e.g. GitLab
#                    ``code`` vs ``issues``.
#
# Every part is slugged (lowercase; ``[a-z0-9._-]`` kept, everything else → ``-``) so the
# colon delimiter is unambiguous. Build ids with :func:`make_source_id` — NEVER hand-format
# an f-string — so the whole fleet routes into named graphs consistently.
KNOWN_SOURCE_SYSTEMS: frozenset[str] = frozenset(
    {
        # code / repos
        "code",  # local-workspace codebase (instance = repo slug)
        "gitlab",  # GitLab-hosted code + issues/MRs (instance = host)
        "github",
        # enterprise architecture / ITSM / process
        "leanix",
        "servicenow",
        "erpnext",
        "camunda",
        "aris",
        "egeria",
        # issues / docs / wiki / crm
        "jira",
        "plane",
        "confluence",
        "twenty",
        "crm",
        "document",
        "paperless_ngx",
        "gramps",
        # feeds / research / media
        "rss",
        "freshrss",
        "archivebox",
        "audiobookshelf",
        "ard",
        # infra / ops / registry
        "fleet",  # an agents/* fleet package (instance = package)
        "dockerhub",  # a DockerHub namespace (instance = namespace)
        "technitium",
        "tunnel_manager",
        "uptime_kuma",
        "home_assistant",
        "firefly_iii",
        "langfuse",
        # agent-native
        "claude_memory",
        "human_review",
        "inspiration",
    }
)

# One slug character-class for every part of a source id. Colon is the reserved
# delimiter, so it is NOT allowed inside a part (it collapses to ``-``).
_SLUG_STRIP = re.compile(r"[^a-z0-9._-]+")


def slug_part(part: str) -> str:
    """Normalize ONE source-id segment: lowercase; keep ``[a-z0-9._-]``; other runs → ``-``.

    Colons collapse to ``-`` so a segment can never smuggle in the reserved delimiter and
    split a two-part id into three. Leading/trailing separators are trimmed.
    """
    s = _SLUG_STRIP.sub("-", str(part).strip().lower())
    return s.strip("-._") or ""


def make_source_id(
    system: str, instance: str | None = None, kind: str | None = None
) -> str:
    """Build a canonical ``<system>[:<instance>][:<kind>]`` source id.

    The ONE formatter for a connector's ``source_system`` value — every part is slugged
    and joined by ``:`` so the whole fleet partitions into predictable named graphs. Empty
    optional parts are dropped. See the schema note above (CONCEPT:AU-KG.ingest.source-id-naming-schema).

    >>> make_source_id("gitlab", "gl.corp", "code")
    'gitlab:gl.corp:code'
    >>> make_source_id("leanix")
    'leanix'
    """
    parts = [slug_part(system)]
    for extra in (instance, kind):
        if extra is None:
            continue
        seg = slug_part(extra)
        if seg:
            parts.append(seg)
    return ":".join(p for p in parts if p)


def source_of(props: dict[str, Any] | None) -> str | None:
    """Return the source-system identity for ``props``, or ``None`` if untagged.

    Reads the provenance keys in priority order and skips generic values so only a
    genuine external connector (leanix/servicenow/…) yields a source.
    """
    if not props:
        return None
    for key in _SOURCE_KEYS:
        raw = props.get(key)
        if raw is None:
            continue
        val = str(raw).strip().lower()
        if val and val not in _GENERIC_SOURCES:
            return val
    return None


def graph_uri_for(props: dict[str, Any] | None) -> str | None:
    """Named-graph IRI for ``props`` (``urn:source:<system>``), or ``None`` for the
    default graph when the data carries no real external source."""
    src = source_of(props)
    return f"{SOURCE_GRAPH_PREFIX}{src}" if src else None


def graph_uri_for_source(source: str) -> str:
    """The named-graph IRI for an explicitly named source (e.g. ``"leanix"``)."""
    return f"{SOURCE_GRAPH_PREFIX}{source.strip().lower()}"


# ── Default-graph guard + coverage (CONCEPT:AU-KG.ingest.default-graph-leak-guard) ──
#
# The safety net that makes source-partitioning provable for EVERY ingestion type, not just
# the ones wired by hand: a node routed to the DEFAULT graph (no real source) is either an
# internal/derived node that BELONGS there, or an external artifact that LEAKED because its
# writer forgot to stamp a source. This seam records every default-graph landing by label so
# a doctor can report leakage, and — under an opt-in strict flag — refuses a non-internal
# label with no source so a new leak can never be silent.

import logging
import threading

logger = logging.getLogger(__name__)

# Node labels that legitimately live in the DEFAULT graph — internal/derived nodes that are
# NOT an externally-ingested source (reasoning artifacts, run provenance, communities, agent
# memory, relations). These never count as a leak, and strict mode never rejects them.
INTERNAL_DEFAULT_LABELS: frozenset[str] = frozenset(
    {
        "",
        "Claim",
        "Evidence",
        "EvidenceSpan",
        "BeliefState",
        "Belief",
        "Community",
        "CommunityReport",
        "Memory",
        "AgentMemory",
        "RunTrace",
        "ToolCall",
        "GenerationNode",
        "Feedback",
        "Lineage",
        "Capability",
        "Concept",  # canonical-by-id cross-source concepts (merged, not per-source)
    }
)

_default_graph_writes: dict[str, int] = {}
_dg_lock = threading.Lock()


def _strict_partition_enabled() -> bool:
    """Whether ``KG_STRICT_SOURCE_PARTITION`` opts into hard rejection of un-sourced,
    non-internal nodes (default OFF — observability only)."""
    try:
        from agent_utilities.core.config import setting

        return str(setting("KG_STRICT_SOURCE_PARTITION", "false")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    except Exception:  # noqa: BLE001 — never let the flag lookup break a write
        return False


def route_graph_uri(props: dict[str, Any] | None, label: str = "") -> str | None:
    """Guarded named-graph routing: like :func:`graph_uri_for`, but records + polices the
    default-graph fallback.

    Returns the ``urn:source:<system>`` IRI, or ``None`` for the default graph. When a node
    would fall into the default graph, its ``label`` is counted (see
    :func:`default_graph_write_report`); a non-internal label under strict mode raises so a
    forgotten source stamp fails loudly instead of leaking silently.
    """
    g = graph_uri_for(props)
    if g is None:
        with _dg_lock:
            key = label or "?"
            _default_graph_writes[key] = _default_graph_writes.get(key, 0) + 1
        if (
            label
            and label not in INTERNAL_DEFAULT_LABELS
            and _strict_partition_enabled()
        ):
            raise ValueError(
                f"source-partition: node label {label!r} has no source_system and would land "
                "in the SPARQL default graph. Stamp a source with "
                "backends.sparql.source_partition.make_source_id(...), or add the label to "
                "INTERNAL_DEFAULT_LABELS if it is intentionally internal."
            )
    return g


def default_graph_write_report() -> dict[str, int]:
    """Per-label counts of nodes routed to the DEFAULT graph since process start.

    The coverage signal for a doctor/CI check: a non-internal label with a nonzero count is a
    source-stamping leak (that ingestion type is not partitioned). Internal labels
    (:data:`INTERNAL_DEFAULT_LABELS`) here are expected.
    """
    with _dg_lock:
        return dict(_default_graph_writes)


def default_graph_leak_labels() -> dict[str, int]:
    """Just the LEAKED labels — default-graph landings whose label is NOT internal."""
    return {
        lbl: n
        for lbl, n in default_graph_write_report().items()
        if lbl and lbl not in INTERNAL_DEFAULT_LABELS
    }


def reset_default_graph_report() -> None:
    """Clear the default-graph counters (tests / a fresh coverage window)."""
    with _dg_lock:
        _default_graph_writes.clear()


def source_partition_coverage(backend: Any) -> dict[str, Any]:
    """Backend-agnostic source-partition coverage — the universal ``is it solved for ALL
    ingestion types?`` check (CONCEPT:AU-KG.ingest.default-graph-leak-guard).

    Runs ONE aggregation over any Cypher-capable graph backend (the epistemic-graph engine
    authority and every property-graph mirror): per node label, how many nodes carry a
    ``source_system`` vs not. A NON-internal label with unsourced nodes is a partition
    **leak** — that ingestion type is not landing in a ``urn:source:*`` partition. For the
    live SPARQL write path, :func:`default_graph_write_report` is the runtime equivalent.

    Returns ``{"supported": False}`` when the backend can't be queried, so a doctor degrades
    gracefully rather than failing.
    """
    execute = getattr(backend, "execute", None)
    if not callable(execute):
        return {"supported": False, "reason": "backend has no execute()"}
    try:
        rows = execute(
            "MATCH (n) RETURN coalesce(n.type, n.label, '?') AS label, "
            "count(n) AS total, count(n.source_system) AS sourced"
        )
    except Exception as exc:  # noqa: BLE001 — coverage is best-effort observability
        return {"supported": False, "reason": str(exc)}

    by_label: dict[str, dict[str, int]] = {}
    leaks: dict[str, int] = {}
    for r in rows or []:
        label = str(r.get("label") or "?")
        total = int(r.get("total") or 0)
        sourced = int(r.get("sourced") or 0)
        unsourced = max(0, total - sourced)
        by_label[label] = {"total": total, "sourced": sourced, "unsourced": unsourced}
        if unsourced > 0 and label and label not in INTERNAL_DEFAULT_LABELS:
            leaks[label] = unsourced
    return {
        "supported": True,
        "by_label": by_label,
        "leaks": leaks,
        "leaking": bool(leaks),
    }
