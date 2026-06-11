#!/usr/bin/python
from __future__ import annotations

"""PerformanceAnomaly consumer (CONCEPT:AHE-3.19 — Performance Anomaly Consumer).

``PerformanceAnomaly`` nodes are written from several paths — the
kg-report-persister skill, ``ExecutionSummary`` flows, the failure analyzer —
but until now they had no consumer beyond ad-hoc maintainer queries: observed
degradation accumulated in the graph and nothing ever acted on it.

This module is that consumer. A daemon tick (``anomaly_consumer`` in the
engine's maintenance scheduler, flag ``KG_ANOMALY_CONSUMER``, default ON — it
is LLM-free, bounded, and propose-only) periodically:

1. scans unconsumed ``PerformanceAnomaly`` nodes (no ``consumed`` stamp);
2. skips the ones that already evidence a gap Concept (the failure analyzer's
   own anomalies are born with an ``EVIDENCES`` edge);
3. clusters the rest by ``(target, anomaly_type)`` and files one
   ``failure_gap`` ``Concept`` topic per cluster through the failure
   analyzer's shared gap-topic path, with ``EVIDENCES`` provenance from every
   anomaly in the cluster — so the golden loop's existing intake remediates
   them;
4. stamps every scanned anomaly ``consumed`` so the work never repeats.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: Per-tick scan budget — bounded so a backlog can never wedge the scheduler.
DEFAULT_SCAN_LIMIT = 200


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _unconsumed_anomalies(engine: Any, limit: int) -> list[dict[str, Any]]:
    try:
        rows = engine.query_cypher(
            "MATCH (a:PerformanceAnomaly) WHERE a.consumed IS NULL "
            f"RETURN a LIMIT {int(limit)}"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("anomaly scan failed: %s", e)
        return []
    out = []
    for row in rows or []:
        props = row.get("a") if isinstance(row, dict) else None
        if isinstance(props, dict) and props.get("id"):
            out.append(props)
    return out


def _already_evidencing(engine: Any) -> set[str]:
    """Ids of anomalies that already EVIDENCE a gap Concept (skip refiling)."""
    try:
        rows = engine.query_cypher(
            "MATCH (a:PerformanceAnomaly)-[:EVIDENCES]->(c:Concept) "
            "RETURN a.id AS id"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("anomaly evidence lookup failed: %s", e)
        return set()
    return {r["id"] for r in rows or [] if isinstance(r, dict) and r.get("id")}


def _mark_consumed(engine: Any, anomaly_id: str) -> bool:
    try:
        engine.backend.execute(
            "MATCH (a:PerformanceAnomaly {id: $id}) "
            "SET a.consumed = $ts, a.consumed_by = 'anomaly_consumer'",
            {"id": anomaly_id, "ts": _now_iso()},
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("anomaly consume stamp failed: %s", e)
        return False


def consume_anomalies(
    engine: Any, *, limit: int = DEFAULT_SCAN_LIMIT
) -> dict[str, Any]:
    """One consumer pass: scan → cluster → file failure_gap topics → stamp.

    Returns a JSON-able report (``scanned`` / ``already_evidenced`` /
    ``gaps_filed`` / ``consumed`` / ``gap_ids``). Propose-only: the only
    writes are gap ``Concept`` topics, ``EVIDENCES`` edges, and the
    ``consumed`` stamps.
    """
    from .failure_analyzer import FailurePattern, _sig, file_gap_topic

    anomalies = _unconsumed_anomalies(engine, limit)
    evidencing = _already_evidencing(engine) if anomalies else set()

    # Cluster fresh anomalies by (target, anomaly_type) so a noisy target
    # files ONE remediation topic with all its anomalies as evidence.
    clusters: dict[str, dict[str, Any]] = {}
    already = 0
    for a in anomalies:
        if a["id"] in evidencing:
            already += 1
            continue
        target = str(a.get("target_node_id") or "unknown")
        anomaly_type = str(a.get("anomaly_type") or "ANOMALY")
        sig = _sig(target, "performance_anomaly", anomaly_type.lower())
        cluster = clusters.setdefault(
            sig,
            {
                "pattern": FailurePattern(
                    signature=sig,
                    name=target,
                    kind="performance_anomaly",
                    anomaly_type=anomaly_type,
                    count=0,
                    sample_detail=str(a.get("metadata") or ""),
                    value=a.get("threshold_exceeded"),
                    baseline=a.get("baseline"),
                ),
                "anomaly_ids": [],
            },
        )
        cluster["pattern"].count += 1
        cluster["anomaly_ids"].append(a["id"])

    gaps: list[str] = []
    for cluster in clusters.values():
        anomaly_ids = cluster["anomaly_ids"]
        gap = file_gap_topic(
            engine,
            cluster["pattern"],
            anomaly_id=anomaly_ids[0],
            source="anomaly_consumer",
        )
        if gap is None:
            continue
        gaps.append(gap["id"])
        # Provenance from every other anomaly in the cluster.
        for aid in anomaly_ids[1:]:
            try:
                engine.link_nodes(
                    source_id=aid,
                    target_id=gap["id"],
                    rel_type="EVIDENCES",
                    properties={"source": "anomaly_consumer"},
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("EVIDENCES edge failed: %s", e)

    consumed = sum(1 for a in anomalies if _mark_consumed(engine, a["id"]))

    report = {
        "scanned": len(anomalies),
        "already_evidenced": already,
        "gaps_filed": len(gaps),
        "gap_ids": gaps,
        "consumed": consumed,
    }
    if anomalies:
        logger.info(
            "[AHE-3.19] anomaly consumer: scanned=%d gaps=%d consumed=%d",
            report["scanned"],
            report["gaps_filed"],
            report["consumed"],
        )
    return report
