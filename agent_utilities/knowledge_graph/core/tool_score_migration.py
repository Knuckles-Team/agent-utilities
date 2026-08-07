#!/usr/bin/python
from __future__ import annotations

"""D-CDX-53 data migration: resync persisted ``Tool.relevance_score`` values
onto ONE canonical domain.

``agent_utilities/models/tool_score.py`` and the ``ToolNode``/``MCPToolInfo``
validators fix *reads* (any value passing through those models is already
normalized). This module is the complementary *write-side* fix: it
transactionally resyncs the values actually persisted on ``:Tool`` nodes so
routing queries that order directly on ``t.relevance_score`` in the database
(``agent_utilities/graph/_router_impl.py``) can eventually trust database-side
ordering again, rather than depending forever on every caller re-normalizing
at read time.

Split into a pure planning function (:func:`plan_tool_relevance_resync`,
trivially unit-testable with plain dict rows) and an engine-facing driver
(:func:`resync_tool_relevance_scores`, dry-run by default) that queries and
writes through whatever object exposes ``query_cypher``/``execute`` — mirrors
the safety model of ``scripts/backfill_embeddings.py`` (dry-run by default,
bounded ``limit``, one write per row, never a silent guess on an ambiguous
value).
"""

import logging
from typing import Any, Protocol

from ...models.tool_score import (
    is_canonical_relevance_score,
    normalize_legacy_relevance_score,
)

logger = logging.getLogger(__name__)


class _CypherEngine(Protocol):
    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...


def plan_tool_relevance_resync(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Classify persisted Tool rows into already-canonical / needs-migration
    / quarantined, WITHOUT touching the graph.

    Args:
        rows: ``[{"id": ..., "relevance_score": ...}, ...]`` as returned by
            ``MATCH (t:Tool) RETURN t.id AS id, t.relevance_score AS relevance_score``.

    Returns:
        ``{"already_canonical": [...], "to_migrate": [...], "quarantined": [...]}``.
        ``to_migrate`` entries carry ``{"id", "old", "new"}``; ``quarantined``
        entries carry ``{"id", "value"}`` for a value that is neither already
        canonical nor a recognized legacy float — these are NEVER silently
        coerced, only reported, so a corrupt row surfaces for manual
        attention exactly like a strict-model construction failure would.
    """
    already_canonical: list[dict[str, Any]] = []
    to_migrate: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []

    for row in rows:
        node_id = row.get("id")
        raw = row.get("relevance_score")
        if is_canonical_relevance_score(raw):
            already_canonical.append({"id": node_id, "value": raw})
            continue
        normalized = normalize_legacy_relevance_score(raw)
        if is_canonical_relevance_score(normalized) and normalized != raw:
            to_migrate.append({"id": node_id, "old": raw, "new": normalized})
            continue
        quarantined.append({"id": node_id, "value": raw})

    return {
        "already_canonical": already_canonical,
        "to_migrate": to_migrate,
        "quarantined": quarantined,
    }


def resync_tool_relevance_scores(
    engine: _CypherEngine,
    *,
    execute: bool = False,
    limit: int = 500,
) -> dict[str, Any]:
    """Query every persisted ``:Tool`` row (bounded by ``limit``) and, only
    when ``execute=True``, write back the canonical score for every row the
    plan classifies as ``to_migrate``.

    Dry-run by default — mirrors ``scripts/backfill_embeddings.py``'s safety
    model. Idempotent: re-running after a successful migration finds every
    row already canonical and writes nothing (``to_migrate`` empty).

    Returns a report with the classification counts plus, when executed, how
    many writes actually succeeded (a per-row write failure is recorded in
    ``write_errors`` rather than aborting the whole run or being silently
    swallowed).
    """
    rows = (
        engine.query_cypher(
            f"MATCH (t:Tool) RETURN t.id AS id, t.relevance_score AS relevance_score LIMIT {int(limit)}"
        )
        or []
    )
    plan = plan_tool_relevance_resync(rows)

    report: dict[str, Any] = {
        "scanned": len(rows),
        "already_canonical": len(plan["already_canonical"]),
        "to_migrate": len(plan["to_migrate"]),
        "quarantined": plan["quarantined"],
        "executed": execute,
        "migrated": 0,
        "write_errors": [],
    }

    if plan["quarantined"]:
        logger.warning(
            "Tool relevance-score resync found quarantined rows that are "
            "neither canonical nor a recognized legacy value; count=%d",
            len(plan["quarantined"]),
        )

    if not execute:
        return report

    for entry in plan["to_migrate"]:
        try:
            engine.query_cypher(
                "MATCH (t:Tool {id: $id}) SET t.relevance_score = $new",
                {"id": entry["id"], "new": entry["new"]},
            )
            report["migrated"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad row must not abort the run
            logger.warning(
                "Tool relevance-score resync failed to write one row: error_type=%s",
                type(exc).__name__,
            )
            report["write_errors"].append({"id": entry["id"], "error": str(exc)})

    return report
