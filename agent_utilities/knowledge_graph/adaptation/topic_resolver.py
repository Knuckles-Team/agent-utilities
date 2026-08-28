"""Topic resolution — mark KG topics as ADDRESSED_BY acquired sources.

CONCEPT:AU-KG.research.self-evolution-convergence — Research assimilation / self-evolution convergence.

The evolution/golden loop pulls *unresolved* topics (``Concept`` nodes with no
``ADDRESSED_BY`` edge). After research acquisition ingests sources that mention a
topic's concept, this module links source→topic with ``ADDRESSES`` (and the
inverse ``ADDRESSED_BY``) so the loop converges instead of re-surfacing the same
topics forever.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _addressed_concept_ids(engine: Any) -> set[str]:
    """Concept ids that already have an ADDRESSED_BY edge (positive traversal — supported)."""
    try:
        rows = engine.query_cypher(
            "MATCH (c:Concept)-[:ADDRESSED_BY]->(s) RETURN c.id AS id"
        )
        return {r["id"] for r in (rows or []) if isinstance(r, dict) and r.get("id")}
    except Exception as e:  # noqa: BLE001 — a failed "already addressed" lookup just falls through with an empty skip-set, over-including topics as unresolved (re-surfacing at worst) rather than losing any; the caller's own retry cadence corrects it next pass
        logger.debug("unresolved_topics: addressed query failed: %s", e)
        return set()


def _all_concept_rows(
    engine: Any, limit: int, addressed: set[str]
) -> list[dict[str, Any]]:
    """All Concept id/name rows (plain node query — supported), to subtract ``addressed`` from."""
    try:
        return (
            engine.query_cypher(
                "MATCH (c:Concept) RETURN c.id AS id, c.name AS name LIMIT $limit",
                {"limit": int(limit) * 10 if addressed else int(limit)},
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 — no concept rows this pass just returns an empty unresolved-list; nothing is marked/consumed here, so it costs one skipped scan, never a lost topic
        logger.debug("unresolved_topics: concept query failed: %s", e)
        return []


def unresolved_topics(engine: Any, limit: int = 10) -> list[dict[str, Any]]:
    """Return ``Concept`` topics that have no ``ADDRESSED_BY`` source yet.

    A topic is "unresolved" when nothing addresses it — these are the open
    questions the research loop should acquire sources for.

    Computed with SUPPORTED query shapes only: ``WHERE NOT (c)-[:R]->()``
    negation isn't transpiled, so we take all Concepts and subtract the set that
    already has an ``ADDRESSED_BY`` edge (a positive single-hop traversal).
    """
    addressed = _addressed_concept_ids(engine)
    rows = _all_concept_rows(engine, limit, addressed)
    out: list[dict[str, Any]] = []
    for r in rows:
        if not (isinstance(r, dict) and r.get("id")):
            continue
        if r["id"] in addressed:
            continue
        out.append({"id": r["id"], "name": r.get("name") or r["id"]})
        if len(out) >= limit:
            break
    return out


def mark_addressed(
    engine: Any, topic_id: str, source_ids: list[str], *, source: str = "research"
) -> int:
    """Create ``source -[:ADDRESSES]-> topic`` (+ inverse ``ADDRESSED_BY``).

    Returns the number of ADDRESSES edges written. Best-effort and idempotent
    (the backend MERGE/add_edge dedupes).
    """
    written = 0
    for sid in source_ids:
        if not sid or sid == topic_id:
            continue
        try:
            engine.link_nodes(
                source_id=sid,
                target_id=topic_id,
                rel_type="ADDRESSES",
                properties={"source": source},
            )
            engine.link_nodes(
                source_id=topic_id,
                target_id=sid,
                rel_type="ADDRESSED_BY",
                properties={"source": source},
            )
            written += 1
        except Exception as e:  # noqa: BLE001 — this source's edges just don't count toward `written`, which every caller now checks (loop_controller._advance_research, D-DST-2) before treating the topic as addressed, so a failed link here correctly stays retryable rather than being silently marked done
            logger.debug("mark_addressed %s->%s failed: %s", sid, topic_id, e)
    return written
