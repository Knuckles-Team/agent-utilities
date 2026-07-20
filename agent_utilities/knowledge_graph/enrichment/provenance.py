#!/usr/bin/python
from __future__ import annotations

"""The one source-provenance contract for external ingestion (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Both ingestion paths — ``core/engine_ingestion.ingest_external_batch`` (dict
entities; the leanix-delta + hydration family) and ``enrichment/registry.write_batch``
(typed ``ExtractionBatch``; the materialize/extractor family) — converge here so
every externally-ingested node/edge carries the SAME provenance metadata, regardless
of which writer persisted it:

* ``source_system`` — the canonical source id, used for provenance and for
  partitioning into ``urn:source:<system>`` named graphs when mirrored/pushed to a
  SPARQL store (see ``backends/sparql/source_partition``).
* ``domain`` — the federation key the write-back resolver
  (``writeback/core.resolve_external_id``) queries (``MATCH (n) WHERE n.domain = $d``)
  to map a KG node back to its upstream record.

Keeping both on every external node is what lets the unified ETL surface, the
Stardog data backend, and ETL lineage treat every connector identically — no
per-source code. Internal (non-source) writes simply never call this, so they are
unaffected.
"""

from typing import Any


def stamp_source(props: dict[str, Any], source: str | None) -> dict[str, Any]:
    """Stamp ``source_system`` + ``domain`` on ``props`` (in place) when ``source``
    is a real external source. Caller-supplied values always win (``setdefault``).
    A falsy ``source`` is a no-op, so internal-fact writes stay untouched.
    """
    src = (source or "").strip().lower()
    if not src:
        return props
    props.setdefault("source_system", src)
    props.setdefault("domain", src)
    return props
