"""Self-registering source-extractor registry (CONCEPT:KG-2.9).

Enables **conflict-free parallel development** of enterprise/source extractors:
each source (infra, servicenow, leanix, erpnext, grafana, …) lives in its own
module under ``extractors/`` and calls :func:`register_extractor` at import time, so
adding a source touches NO shared hub file (no edits to ``__init__``/``pipeline``/
``models``). The package auto-discovers and imports all extractor modules, and a
single generic writer persists every ``ExtractionBatch`` through the one
``GraphBackend`` interface.

Contract for a source extractor::

    from ..registry import register_extractor
    from ..models import ExtractionBatch, GraphNode, EnrichmentEdge

    def extract(config) -> ExtractionBatch:
        ...

    register_extractor("servicenow", extract, description="ServiceNow ITSM → KG")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .models import ExtractionBatch

logger = logging.getLogger(__name__)

# config -> ExtractionBatch
ExtractFn = Callable[[Any], ExtractionBatch]


@dataclass(frozen=True)
class SourceExtractor:
    category: str
    extract: ExtractFn
    description: str = ""


_REGISTRY: dict[str, SourceExtractor] = {}


def register_extractor(
    category: str, extract: ExtractFn, description: str = ""
) -> None:
    """Register a source extractor under a unique category key (idempotent)."""
    if category in _REGISTRY and _REGISTRY[category].extract is not extract:
        logger.debug("Overriding source extractor for category %s", category)
    _REGISTRY[category] = SourceExtractor(category, extract, description)


def get_source(category: str) -> SourceExtractor | None:
    return _REGISTRY.get(category)


def list_sources() -> list[SourceExtractor]:
    return sorted(_REGISTRY.values(), key=lambda s: s.category)


def write_batch(
    backend: Any, batch: ExtractionBatch, source: str | None = None
) -> tuple[int, int]:
    """Persist an ExtractionBatch via the single GraphBackend interface.

    Generic — works for every source, so new extractors never touch writer code.
    Returns (nodes_written, edges_written).

    ``source`` (the connector category, e.g. ``"egeria"``) stamps the shared
    provenance contract (``source_system`` + ``domain``) on every node/edge via
    :func:`.provenance.stamp_source`, identical to the ``ingest_external_batch``
    path — so materialized sources route into their ``urn:source:<system>`` named
    graph on a SPARQL mirror just like hydration sources. Omit ``source`` for
    internal-fact batches (finance/synthesize) to leave them untagged.
    """
    from .provenance import stamp_source

    n = e = 0
    for node in batch.nodes:
        props = stamp_source(
            {k: v for k, v in node.props.items() if v is not None}, source
        )
        try:
            backend.add_node(node.id, type=node.type, **props)
            n += 1
        except Exception as exc:  # pragma: no cover - backend transport
            logger.debug("write_batch node %s failed: %s", node.id, exc)
    add_edge = getattr(backend, "add_edge", None)
    if callable(add_edge):
        for edge in batch.edges:
            edge_props = stamp_source(
                {k: v for k, v in edge.props.items() if v is not None}, source
            )
            try:
                add_edge(edge.source, edge.target, rel_type=edge.rel_type, **edge_props)
                e += 1
            except Exception as exc:  # pragma: no cover
                logger.debug("write_batch edge failed: %s", exc)
    return n, e


def discover_extractors() -> int:
    """Import every module under ``extractors/`` so they self-register.

    Called once; new source modules are picked up with no shared-file edits.
    """
    import importlib
    import pkgutil

    from . import extractors as _pkg

    count = 0
    for mod in pkgutil.iter_modules(_pkg.__path__):
        try:
            importlib.import_module(f"{_pkg.__name__}.{mod.name}")
            count += 1
        except Exception as exc:  # pragma: no cover - optional source deps
            logger.debug("extractor %s not loaded: %s", mod.name, exc)
    return count
