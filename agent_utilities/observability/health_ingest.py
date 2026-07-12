#!/usr/bin/python
from __future__ import annotations

"""Typed KG I/O for the shared health-intelligence kernels (``.health``).

CONCEPT:AU-KG.ingest.enterprise-source-extractor — writers/readers for the unified
``:HealthTrend``/``:HealthBaseline``/``:HealthAnomaly``/``:Incident`` nodes any
telemetry-producing agent (fan-manager, systems-manager, container-manager-mcp,
tunnel-manager, ...) plugs into, over the shared unified infrastructure ontology
(``knowledge_graph/ontology_infrastructure.ttl``). Mirrors ``fan_manager.kg_ingest``'s
thermal write/read path exactly, generalized from °C to any entity/signal pair.

All I/O rides the shared fleet primitive
(:mod:`agent_utilities.knowledge_graph.memory.native_ingest`) — the same lightweight
engine client (``GraphComputeEngine()._client`` + ``txn``) every native connector uses.
Entirely best-effort and engine-guarded: with no reachable engine every entry point
no-ops (returns ``None``/``[]``), so a producer keeps running with zero KG
infrastructure. Node ids follow ``health:<class>:<entityId>:<signal>[:<at>]``.
"""

import logging
import time
from typing import Any

logger = logging.getLogger("agent_utilities.observability.health")

_SOURCE = "agent-utilities-health"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _entity_scaffold(entity_id: str, entity_type: str) -> dict[str, Any]:
    """Minimal MERGE-safe stub for the linked entity — robust even when the owning
    producer (systems-manager/container-manager-mcp/...) hasn't ingested it yet."""
    return {"id": entity_id, "type": entity_type, "name": entity_id}


def _engine() -> Any | None:
    """Return a live :class:`GraphComputeEngine` (for reads) or ``None`` when unavailable."""
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        return GraphComputeEngine()
    except Exception as e:  # noqa: BLE001 — KG stack absent / engine unreachable
        logger.debug("health KG read unavailable: %s", e)
        return None


def _parse_ts(value: Any) -> float | None:
    """Parse an ``observedAt`` ISO-8601 (``...Z``) timestamp into epoch seconds."""
    if not value:
        return None
    try:
        return time.mktime(time.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def ingest_health_trend(
    entity_id: str,
    entity_type: str,
    layer: str,
    signal: str,
    trend: dict[str, Any],
    *,
    host: str | None = None,
) -> dict[str, int] | None:
    """Write ONE distilled trend window as a clean, numeric ``:HealthTrend`` node.

    ``trend`` uses :class:`~agent_utilities.observability.health.HealthTrendBuffer`'s
    flush keys: ``min``/``max``/``avg``/``avg_control``/``samples``/``window_s``
    (+ optional ``observed_at``). Linked to the ``entity_id``/``entity_type`` node via
    the unified ``:affectsEntity`` edge so the derivation loop can read trends straight
    back with :func:`read_health_trends`.
    """
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    at = trend.get("observed_at") or _now()
    tid = f"health:trend:{entity_id}:{signal}:{at}"
    entities = [
        _entity_scaffold(entity_id, entity_type),
        {
            "id": tid,
            "type": "HealthTrend",
            "entity": entity_id,
            "layer": layer,
            "signal": signal,
            "avg": trend.get("avg"),
            "min": trend.get("min"),
            "max": trend.get("max"),
            "avgControl": trend.get("avg_control"),
            "samples": trend.get("samples"),
            "windowS": trend.get("window_s"),
            "host": host,
            "observedAt": at,
        },
    ]
    relationships = [{"source": tid, "target": entity_id, "type": "affectsEntity"}]
    return ingest_entities(entities, relationships, source=_SOURCE, domain=layer)


def read_health_trends(
    entity_id: str,
    signal: str,
    *,
    days: int = 14,
    limit: int = 0,
    engine: Any | None = None,
) -> list[dict[str, Any]]:
    """Read an entity's recent ``:HealthTrend`` rows for ``signal`` (props dicts),
    oldest→newest.

    Best-effort: returns ``[]`` with no reachable engine. Filters to ``entity_id`` +
    ``signal`` and the last ``days``; callers reason over these purely in Python
    (mirrors ``fan_manager.kg_ingest.read_thermal_trends``).
    """
    eng = engine or _engine()
    if eng is None:
        return []
    try:
        rows = eng.get_nodes_by_label("HealthTrend", limit) or []
    except Exception as e:  # noqa: BLE001 — read is best-effort
        logger.debug("health KG read: get_nodes_by_label failed: %s", e)
        return []
    cutoff = time.time() - days * 86400
    out: list[dict[str, Any]] = []
    for _id, props in rows:
        if not isinstance(props, dict):
            continue
        if props.get("entity") != entity_id or props.get("signal") != signal:
            continue
        ts = _parse_ts(props.get("observedAt"))
        if ts is not None and ts < cutoff:
            continue
        out.append(props)
    out.sort(key=lambda p: str(p.get("observedAt") or ""))
    return out


def ingest_health_baseline(
    entity_id: str,
    signal: str,
    baseline: dict[str, Any],
    *,
    entity_type: str = "Entity",
) -> dict[str, int] | None:
    """Write a learned ``:HealthBaseline`` node (one per entity+signal, overwritten
    each derivation pass — same id every time, per :func:`compute_baseline`'s shape)."""
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    bid = f"health:baseline:{entity_id}:{signal}"
    entities = [
        _entity_scaffold(entity_id, entity_type),
        {
            "id": bid,
            "type": "HealthBaseline",
            "entity": entity_id,
            "signal": signal,
            "p50": baseline.get("p50"),
            "p95": baseline.get("p95"),
            "minEnv": baseline.get("min_env"),
            "maxEnv": baseline.get("max_env"),
            "avgControl": baseline.get("avg_control"),
            "inertia": baseline.get("inertia"),
            "windows": baseline.get("windows"),
            "observedAt": _now(),
        },
    ]
    relationships = [{"source": bid, "target": entity_id, "type": "affectsEntity"}]
    return ingest_entities(entities, relationships, source=_SOURCE, domain="health")


def ingest_health_anomaly(
    entity_id: str,
    signal: str,
    anomaly: dict[str, Any],
    *,
    entity_type: str = "Entity",
) -> dict[str, int] | None:
    """Write a ``:HealthAnomaly`` node (an entity off its baseline) linked
    ``affectsEntity`` to the entity it was observed on."""
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    at = _now()
    aid = f"health:anomaly:{entity_id}:{signal}:{at}"
    entities = [
        _entity_scaffold(entity_id, entity_type),
        {
            "id": aid,
            "type": "HealthAnomaly",
            "entity": entity_id,
            "signal": signal,
            "kind": anomaly.get("kind"),
            "zscore": anomaly.get("zscore"),
            "observed": anomaly.get("observed"),
            "expected": anomaly.get("expected"),
            "observedAt": at,
        },
    ]
    relationships = [{"source": aid, "target": entity_id, "type": "affectsEntity"}]
    return ingest_entities(entities, relationships, source=_SOURCE, domain="health")


def ingest_incident(incident: dict[str, Any]) -> dict[str, int] | None:
    """Write a minimal ``:Incident`` node for the future cross-layer correlation loop
    (``reports/unified-infra-intelligence-plan.md`` Phase D).

    ``incident``: ``{"id"?, "kind", "summary"?, "entities": [<entity_id>, ...]}``.
    Each listed entity is linked ``affectsEntity``. Best-effort/minimal by design —
    the correlation loop that populates this richly is a later phase.
    """
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    entity_ids = [e for e in (incident.get("entities") or []) if e]
    iid = incident.get("id") or f"health:incident:{_now()}"
    entities = [
        {
            "id": iid,
            "type": "Incident",
            "kind": incident.get("kind"),
            "summary": incident.get("summary"),
            "observedAt": _now(),
        }
    ]
    relationships = [
        {"source": iid, "target": eid, "type": "affectsEntity"} for eid in entity_ids
    ]
    return ingest_entities(entities, relationships, source=_SOURCE, domain="health")
