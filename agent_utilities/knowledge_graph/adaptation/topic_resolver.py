"""Topic resolution — mark KG topics as ADDRESSED_BY acquired sources.

CONCEPT:KG-2.7 — Research assimilation / self-evolution convergence.

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


def unresolved_topics(engine: Any, limit: int = 10) -> list[dict[str, Any]]:
    """Return ``Concept`` topics that have no ``ADDRESSED_BY`` source yet.

    A topic is "unresolved" when nothing addresses it — these are the open
    questions the research loop should acquire sources for.

    Computed with SUPPORTED query shapes only: ``WHERE NOT (c)-[:R]->()``
    negation isn't transpiled, so we take all Concepts and subtract the set that
    already has an ``ADDRESSED_BY`` edge (a positive single-hop traversal).
    """
    # Concepts that are already addressed (positive traversal — supported).
    addressed: set[str] = set()
    try:
        rows = engine.query_cypher(
            "MATCH (c:Concept)-[:ADDRESSED_BY]->(s) RETURN c.id AS id"
        )
        addressed = {
            r["id"] for r in (rows or []) if isinstance(r, dict) and r.get("id")
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("unresolved_topics: addressed query failed: %s", e)

    # All concept topics (plain node query — supported), then subtract.
    try:
        rows = engine.query_cypher(
            "MATCH (c:Concept) RETURN c.id AS id, c.name AS name LIMIT $limit",
            {"limit": int(limit) * 10 if addressed else int(limit)},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("unresolved_topics: concept query failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for r in rows or []:
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
        except Exception as e:  # noqa: BLE001
            logger.debug("mark_addressed %s->%s failed: %s", sid, topic_id, e)
    return written
