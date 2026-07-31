#!/usr/bin/python
from __future__ import annotations

"""KG-backed durable sink for the public error-detail store (D-24).

``agent_utilities.security.error_surface`` keeps a bounded, in-process
``_detail_store`` behind every failed operation's opaque ``error.detail_ref``
(``resolve_error_detail`` / ``resolve_error_detail_for_actor``, D-23). That
store does not survive a process restart and is not shared across replicas —
a ``detail_ref`` handed to a caller could silently become unresolvable the
moment the recording process cycled or a different replica served the
resolve call. ``error_surface.register_detail_persistence_sink()`` is a
de-opinionated seam for exactly this gap; this module is the concrete
backend wired through it.

Follows the same pattern :class:`agent_utilities.capabilities.kg_audit_sink.
KgAuditSink` (PA-R1.7) already established for durable KG-native
observability: a plain object wrapping an engine handle, ``engine=None``
degrades every method to a no-op / empty read (safe to construct
unconditionally), and every write is best-effort — a KG write failure here
must never turn the CALLER's original exception into a second failure.

Persisted shape (CONCEPT:AU-KG.audit.durable-error-detail, `:ErrorDetail` in
``ontology_orchestration.ttl``): one node per correlation id, holding the
SAME sanitized ``error_class`` / ``failing_layer`` / ``traceback`` /
``tenant_id`` bundle already sanitized by
``error_surface._record_error_detail`` through the persistence-privacy gate
— this module never re-derives or re-sanitizes that content, only persists
and reads it back verbatim.

Provenance linkage: when the failure occurred inside a correlated run (the
ambient run-wide correlation id from
:mod:`agent_utilities.observability.correlation` is active), the node also
gets a best-effort ``:DIAGNOSES`` edge to that run's canonical ``:RunTrace``
node (:func:`agent_utilities.observability.trace_ontology.trace_id`). This is
deliberately opportunistic, not guaranteed: most failures raised outside an
orchestrated agent run (a bare request handler, a CLI invocation, a unit
test) have no active run-wide correlation id, and even when one is active the
matching ``:RunTrace`` node may not exist yet (or ever) — a missing edge is
therefore normal, not a defect, and is never logged as a failure. The
``:ErrorDetail`` node itself is the durable value; the edge is a queryability
bonus when it lands.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

ERROR_DETAIL_NODE_LABEL = "ErrorDetail"
ERROR_DETAIL_DIAGNOSES_EDGE = "DIAGNOSES"

#: The record fields persisted/read back verbatim — the SAME keys
#: ``error_surface._record_error_detail`` already stores in-process.
_RECORD_FIELDS = ("error_class", "failing_layer", "traceback", "tenant_id")


def _error_detail_node_id(correlation_id: str) -> str:
    """Canonical node id for a correlation id (opaque refs stay opaque)."""
    ref = str(correlation_id or "").removeprefix("correlation:")
    return f"error_detail:{ref}"


class GraphErrorDetailSink:
    """Durable, KG-backed ``error_surface`` detail-persistence sink.

    Satisfies the existing write-callable contract
    ``register_detail_persistence_sink`` already expects
    (``sink(correlation_id, record)``) via :meth:`__call__` — no change to
    that seam or its tests. Additionally exposes :meth:`resolve`, an OPTIONAL
    method :func:`agent_utilities.security.error_surface.resolve_error_detail`
    detects (``getattr(sink, "resolve", None)``) and uses as a durability
    fallback when the in-process store misses (a restart, or a resolve call
    served by a different replica than the one that recorded the failure).
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    def __call__(self, correlation_id: str, record: dict[str, Any]) -> None:
        """Persist one durable ``:ErrorDetail`` node for ``correlation_id``."""
        if self._engine is None:
            return
        node_id = _error_detail_node_id(correlation_id)
        props = {"correlation_id": correlation_id}
        props.update({field: record.get(field, "") for field in _RECORD_FIELDS})
        try:
            self._engine.add_node(node_id, ERROR_DETAIL_NODE_LABEL, properties=props)
        except Exception:  # noqa: BLE001 — a provenance write must never fail the caller
            logger.warning(
                "GraphErrorDetailSink failed to persist detail for correlation_id=%s",
                correlation_id,
                exc_info=True,
            )
            return
        self._link_to_run_trace(node_id)

    def _link_to_run_trace(self, node_id: str) -> None:
        """Best-effort ``:DIAGNOSES`` edge to the active run's ``:RunTrace``.

        Silent (not logged) on a miss: no active run-wide correlation id, or
        no matching ``:RunTrace`` node, is the common case (see module
        docstring), not a fault.
        """
        try:
            from agent_utilities.observability.correlation import get_correlation_id
            from agent_utilities.observability.trace_ontology import (
                trace_id as run_trace_id,
            )

            run_correlation_id = get_correlation_id()
            if not run_correlation_id:
                return
            self._engine.link_nodes(
                node_id, run_trace_id(run_correlation_id), ERROR_DETAIL_DIAGNOSES_EDGE
            )
        except Exception:  # noqa: BLE001 — opportunistic linkage only
            pass

    def resolve(self, correlation_id: str) -> dict[str, Any] | None:
        """Read a previously persisted detail bundle back, or ``None`` if absent.

        Never raises: a broken/unavailable engine degrades to "no durable
        detail available", matching :func:`agent_utilities.security.
        error_surface.resolve_error_detail`'s own never-raise contract for an
        unknown ref.
        """
        if self._engine is None or not correlation_id:
            return None
        from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

        node_id = _error_detail_node_id(correlation_id)
        rows = read_rows(
            self._engine,
            f"MATCH (e:{ERROR_DETAIL_NODE_LABEL} {{id: $id}}) "
            "RETURN e.error_class AS error_class, e.failing_layer AS failing_layer, "
            "e.traceback AS traceback, e.tenant_id AS tenant_id LIMIT 1",
            {"id": node_id},
        )
        if not rows or not isinstance(rows[0], dict):
            return None
        row = rows[0]
        if not any(row.get(field) for field in _RECORD_FIELDS):
            return None
        return {field: str(row.get(field) or "") for field in _RECORD_FIELDS}


__all__ = [
    "ERROR_DETAIL_DIAGNOSES_EDGE",
    "ERROR_DETAIL_NODE_LABEL",
    "GraphErrorDetailSink",
]
