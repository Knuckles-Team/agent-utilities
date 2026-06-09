#!/usr/bin/python
from __future__ import annotations

"""Object Index Lifecycle — the Object Data Funnel (CONCEPT:KG-2.44).

Palantir provenance: *object-indexing/overview* (batch + streaming pipelines,
data restrictions, metadata sync, staleness/reindex) and *object-backend*.

This package keeps the live search index (the existing
:class:`~agent_utilities.knowledge_graph.retrieval.capability_index.CapabilityIndex`
the router ranks against) consistent with the source-of-truth graph:

* :mod:`.funnel` — :class:`ObjectIndexFunnel` with **batch** (full rebuild) and
  **incremental/streaming** (live upsert/delete delta) sync, gated by a
  composable :class:`DataRestriction` index-eligibility predicate.
* :mod:`.staleness` — :class:`StalenessLedger` tracking per-object index
  freshness (content-hash + watermark) and computing real drift / reindex need.

These are import-populated and wire straight into the retrieval plane: the
funnel produces/maintains the same ``CapabilityIndex`` that
``KnowledgeGraph.designate`` consumes, and :meth:`ObjectIndexFunnel.search`
delegates to that designate path.
"""

from .funnel import (
    DataRestriction,
    FunnelDelta,
    ObjectIndexFunnel,
    SyncResult,
    id_of,
    index_payload_of,
)
from .staleness import (
    ObjectVersion,
    StalenessLedger,
    StalenessReport,
    content_hash,
)

__all__ = [
    # Funnel (KG-2.44)
    "ObjectIndexFunnel",
    "DataRestriction",
    "FunnelDelta",
    "SyncResult",
    "index_payload_of",
    "id_of",
    # Staleness (KG-2.44)
    "StalenessLedger",
    "StalenessReport",
    "ObjectVersion",
    "content_hash",
]
