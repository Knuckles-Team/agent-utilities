"""OpenSearch search tier: client, CDC-fed indexer, and DLS rendering.

(CONCEPT:AU-KG.retrieval.opensearch-cdc-indexer, CA-24, DEC-CA-09, DEC-CA-03)

This package is the au-side half of the search tier `DEC-CA-09` decides:
- :mod:`.client` — a thin ``opensearch-py`` wrapper (no bespoke HTTP client).
- :mod:`.doc_shape` — the `DEC-CA-09` index-naming/document-shape contract.
- :mod:`.indexer` — a Kafka consumer on ``eg.cdc.<graph>`` (`DEC-CA-03`) that
  writes/deletes OpenSearch documents, fail-closed on ordering and on write
  failure.
- :mod:`.dls` — renders a document-level-security query fragment from au's
  single marking definition (``knowledge_graph.ontology.permissioning``),
  reused both by this lane's own query-time enforcement and as the shape
  CA-26 pushes into the `DEC-CA-04` policy bundle's ``renderings.opensearch``.
- :mod:`.rebuild` — drops and reindexes a tenant/object-type from a given
  Kafka offset (default 0), the mechanism behind P3's "replay from offset 0
  rebuilds an identical index".

Non-goals (owned elsewhere, see the lane doc): the `services/opensearch`
deployment (CA-50), the ``opensearch-mcp`` package (CA-43), the
`DEC-CA-04` policy-bundle fetch/apply loop (CA-26), eg's ``federation-search``
adapter (CA-14).
"""

from __future__ import annotations

__all__: list[str] = []
