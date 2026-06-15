#!/usr/bin/python
from __future__ import annotations

"""Source-partitioned named-graph routing for SPARQL triple stores.

CONCEPT:KG-2.7 — Vendor-Agnostic Graph Backend Abstraction.

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
