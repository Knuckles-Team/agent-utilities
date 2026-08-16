#!/usr/bin/python
from __future__ import annotations

"""PerformanceAnomaly consumer (CONCEPT:AU-AHE.optimization.performance-anomaly-consumer — Performance Anomaly Consumer).

``PerformanceAnomaly`` nodes are written from several paths — the
graph-runtime-and-governance skill, ``ExecutionSummary`` flows, the failure analyzer —
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
    except Exception as e:  # noqa: BLE001 — scan failure yields an empty batch this tick; nothing gets marked consumed, so the next daemon tick naturally retries the same anomalies (no data lost)
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
            "MATCH (a:PerformanceAnomaly)-[:EVIDENCES]->(c:Concept) RETURN a.id AS id"
        )
    except Exception as e:  # noqa: BLE001 — a failed lookup just yields an empty skip-set; the gap-topic id is deterministic (`failure_gap:<signature>`), so a spuriously-refiled topic MERGEs onto the same node rather than duplicating it
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
    except Exception as e:  # noqa: BLE001 — the stamp write failing safely leaves `consumed` NULL, which is the "unconsumed" state the WHERE clause in _unconsumed_anomalies scans for, so the anomaly is naturally retried next tick rather than silently dropped
        logger.debug("anomaly consume stamp failed: %s", e)
        return False


def consume_anomalies(
    engine: Any, *, limit: int = DEFAULT_SCAN_LIMIT, graph_writer: Any = None
) -> dict[str, Any]:
    """One consumer pass: scan → cluster → file failure_gap topics → stamp.

    Returns a JSON-able report (``scanned`` / ``already_evidenced`` /
    ``gaps_filed`` / ``consumed`` / ``gap_ids``). Propose-only: the only
    writes are gap ``Concept`` topics, ``EVIDENCES`` edges, and the
    ``consumed`` stamps.

    ``graph_writer`` is threaded straight through to :func:`file_gap_topic`'s
    own explicit in-memory test adapter (see ``_commit_graph_slice`` /
    :class:`FailureAnalyzer`, which accepts the same parameter for the
    identical seam) — the daemon tick never passes it, so production always
    takes the native ChangeEnvelope path unchanged.
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
    # CONCEPT:AU-AHE.evaluation.debug-swallow-justification (D-DST-1): anomalies whose cluster
    # failed to file (file_gap_topic returned None — the Concept persist itself raised) must
    # NOT be stamped `consumed` below. Marking them consumed regardless of write success was
    # the exact write-then-mark-seen defect this triage was scoped to find (mirrors the
    # sdd/watcher.py content-hash bug the gate lane fixed): a transient persist failure would
    # otherwise permanently foreclose retry, since `_unconsumed_anomalies` only rescans rows
    # where `a.consumed IS NULL`.
    unfileable_ids: set[str] = set()
    for cluster in clusters.values():
        anomaly_ids = cluster["anomaly_ids"]
        gap = file_gap_topic(
            engine,
            cluster["pattern"],
            anomaly_id=anomaly_ids[0],
            source="anomaly_consumer",
            graph_writer=graph_writer,
        )
        if gap is None:
            unfileable_ids.update(anomaly_ids)
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
            except Exception as e:  # noqa: BLE001 — the gap Concept itself already persisted (this anomaly's cluster is remediated); a missing secondary EVIDENCES edge only weakens provenance completeness, it does not lose the remediation, so it stays consumed rather than being retried forever for a link that keeps failing
                logger.debug("EVIDENCES edge failed: %s", e)

    consumed = sum(
        1
        for a in anomalies
        if a["id"] not in unfileable_ids and _mark_consumed(engine, a["id"])
    )

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
