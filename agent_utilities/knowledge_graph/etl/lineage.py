#!/usr/bin/python
from __future__ import annotations

"""ETL data lineage — record + query system-to-system data flows (CONCEPT:KG-2.99).

Every ``graph_etl`` run records a lineage trail in the KG itself so an operator can
answer impact-analysis questions ("what flows from ServiceNow to LeanIX?", "where did
this Stardog graph's data originate?"). Reuses the existing provenance ontology — NO
new node/edge types:

* a run is a :class:`RegistryNodeType.PROVENANCE_AGENT` node (``kind="etl_run"``) with
  ``source`` / ``sink`` / ``direction`` / ``nodes`` / ``edges`` / ``status`` / ``at`` props;
* ``source`` and ``sink`` systems are PROVENANCE_AGENT marker nodes
  (``urn:source:<s>`` / ``urn:sink:<s>``, ``kind="system"``) — the same ``urn:source:``
  scheme the Stardog named-graph partitioning and ``sparql_ingestor`` already use;
* :class:`RegistryEdgeType.WAS_DERIVED_FROM` edges chain ``sink → run → source`` so a
  graph walk reconstructs the flow.

Lineage is best-effort: a failure to record never fails the ETL run itself.
"""

import logging
import time
from typing import Any

from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType

logger = logging.getLogger(__name__)

_RUN_KIND = "etl_run"
_SYSTEM_KIND = "system"


def _system_marker(engine: Any, system: str, *, role: str) -> str:
    """Ensure a system marker node exists; return its id (``urn:<role>:<system>``)."""
    node_id = f"urn:{role}:{system}"
    try:
        engine.add_node(
            node_id,
            RegistryNodeType.PROVENANCE_AGENT,
            {"kind": _SYSTEM_KIND, "name": system, "role": role},
        )
    except Exception:  # noqa: BLE001 - marker creation is best-effort
        logger.debug("lineage: marker %s failed", node_id, exc_info=True)
    return node_id


def record_etl_run(
    engine: Any,
    *,
    source: str | None,
    sink: str | None,
    direction: str,
    counts: dict[str, Any] | None = None,
    status: str = "ok",
    at: float | None = None,
) -> str | None:
    """Record one ETL run + its source→sink lineage edges. Returns the run id.

    ``direction`` is ``inbound`` (source→KG), ``outbound`` (KG→sink), or ``through``
    (source→KG→sink). Best-effort: returns ``None`` and logs on failure rather than
    raising into the ETL run.
    """
    if engine is None:
        return None
    ts = at if at is not None else time.time()
    counts = counts or {}
    run_id = f"etl-run:{(source or '_')}:{(sink or '_')}:{int(ts * 1000)}"
    try:
        engine.add_node(
            run_id,
            RegistryNodeType.PROVENANCE_AGENT,
            {
                "kind": _RUN_KIND,
                "source": source or "",
                "sink": sink or "",
                "direction": direction,
                "nodes": int(counts.get("nodes", 0) or 0),
                "edges": int(counts.get("edges", 0) or 0),
                "status": status,
                "at": ts,
            },
        )
        if source:
            src_marker = _system_marker(engine, source, role="source")
            engine.link_nodes(run_id, src_marker, RegistryEdgeType.WAS_DERIVED_FROM)
        if sink:
            sink_marker = _system_marker(engine, sink, role="sink")
            engine.link_nodes(sink_marker, run_id, RegistryEdgeType.WAS_DERIVED_FROM)
    except Exception:  # noqa: BLE001 - lineage must never break the ETL run
        logger.debug("lineage: record_etl_run failed", exc_info=True)
        return None
    return run_id


def query_lineage(
    engine: Any,
    *,
    source: str | None = None,
    sink: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return recorded ETL runs (most-recent first), optionally filtered by source
    and/or sink. Property-based query (label-agnostic) so it works across backends.
    """
    backend = getattr(engine, "backend", None)
    if backend is None or not hasattr(backend, "execute"):
        return []
    where = ["n.kind = $kind"]
    params: dict[str, Any] = {"kind": _RUN_KIND}
    if source:
        where.append("n.source = $source")
        params["source"] = source.strip().lower()
    if sink:
        where.append("n.sink = $sink")
        params["sink"] = sink.strip().lower()
    query = (
        f"MATCH (n) WHERE {' AND '.join(where)} "
        f"RETURN n.id AS id, n.source AS source, n.sink AS sink, "
        f"n.direction AS direction, n.nodes AS nodes, n.edges AS edges, "
        f"n.status AS status, n.at AS at "
        f"ORDER BY n.at DESC LIMIT {int(limit)}"
    )
    try:
        rows = backend.execute(query, params)
    except Exception:  # noqa: BLE001 - read is best-effort
        logger.debug("lineage: query failed", exc_info=True)
        return []
    return [dict(r) for r in (rows or []) if isinstance(r, dict)]
