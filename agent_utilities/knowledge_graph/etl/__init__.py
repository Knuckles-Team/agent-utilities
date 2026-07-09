#!/usr/bin/python
"""Unified ETL surface for the Knowledge Graph (CONCEPT:AU-KG.ontology.one-source / KG-2.99).

One ``source → (ontological transform) → sink`` interface over the existing
ingestion (extractors/hydration), write-back (sinks), and graph-store (mirror/push)
machinery, plus system-to-system data lineage. The KG is the canonical hub.
"""

from .lineage import query_lineage, record_etl_run
from .pipeline import run_etl
from .result import EtlResult
from .transforms import cast, coalesce, dig, flatten, rename, stable_id

__all__ = [
    "run_etl",
    "record_etl_run",
    "query_lineage",
    "EtlResult",
    "dig",
    "coalesce",
    "stable_id",
    "cast",
    "rename",
    "flatten",
]
