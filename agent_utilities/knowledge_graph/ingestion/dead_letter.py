#!/usr/bin/python
from __future__ import annotations

"""Dead-letter visibility + drain (CONCEPT:AU-KG.ingest.dead-letter-drain).

The universal-ingestion program's Track B "dead-letter must be loud and
drainable, never a silent drop" requirement.

**What already existed.** A ``WorkItem`` (:mod:`~agent_utilities.orchestration.work_item`)
already reaches a durable ``dead_letter`` terminal status once its native
retry/backoff budget is exhausted (``commit_result``), and
``knowledge_graph.retrieval.ops_context.diagnose_ops`` already surfaces a
dead-letter *count* for operational health answers. What did NOT exist: an
actual ``list``/``drain`` API — a dead-lettered item was queryable only as an
aggregate count, with no way to see WHICH items and no way to requeue one.
This module is that gap, built directly over the existing ``WorkItem``
label/read pattern (:func:`~agent_utilities.orchestration.work_item.get_work_item`) —
not a second queue or a parallel DLQ store.

**Draining is a deliberate, operator-initiated action — never an automatic
retry.** A lane elsewhere in this program declined to wrap a stuck KG write in
a retry because that "converts a loud incident into a slow one"; this module
honors the same property from the other direction — a dead-lettered item's
retry budget is ALREADY exhausted (this system already tried, repeatedly, and
gave up loudly), so nothing here re-attempts it silently or on a schedule.
:func:`drain_dead_letter_item` only runs when explicitly called (by an
operator through the ``graph_jobs``/REST surface, or a script), and it always
leaves the ORIGINAL dead-lettered item intact (never deletes/mutates it out
from under an auditor) — it submits a genuinely NEW ``WorkItem`` with a fresh
attempt budget and links the two for lineage.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["list_dead_letter_items", "drain_dead_letter_item"]

_NODE_LABEL = "WorkItem"

_LIST_FIELDS = (
    "kind",
    "queue",
    "tenant",
    "payload_ref",
    "error_ref",
    "resource_class",
    "max_attempts",
    "updated_at",
    "correlation_id",
    "dag_id",
)


def list_dead_letter_items(
    engine: Any, *, kind: str = "", limit: int = 50
) -> list[dict[str, Any]]:
    """Enumerate dead-lettered WorkItems — the "loud" half of this module.

    Best-effort read (never raises): an engine that cannot run the query
    returns an empty list rather than crashing a health/ops surface, exactly
    like every other read in this program's ``ops_context``/``ClaimFlywheel``
    style. ``kind`` optionally narrows to one WorkItem kind (e.g.
    ``"ingest_task"``); omit to see every dead-lettered item.
    """
    limit = max(1, min(200, int(limit)))
    return_clause = ", ".join(f"w.{f} AS {f}" for f in _LIST_FIELDS)
    try:
        rows = (
            engine.query_cypher(
                f"MATCH (w:{_NODE_LABEL}) WHERE w.status = 'dead_letter' "
                + ("AND w.kind = $kind " if kind else "")
                + f"RETURN w.id AS id, {return_clause} "
                f"ORDER BY w.updated_at DESC LIMIT {limit}",
                {"kind": kind} if kind else {},
            )
            or []
        )
    except Exception as exc:  # noqa: BLE001 — a broken read must not hide the backlog as empty
        logger.warning("dead_letter: list query failed: %s", exc)
        return []
    return [dict(row) for row in rows if isinstance(row, dict) and row.get("id")]


def drain_dead_letter_item(
    engine: Any,
    item_id: str,
    *,
    actor: str,
    reason: str = "",
) -> dict[str, Any]:
    """Explicitly requeue one dead-lettered WorkItem as a FRESH attempt.

    Never mutates or removes the original item (it stays visible, exactly as
    it dead-lettered, for audit) — submits a new ``WorkItem`` cloned from its
    ``kind``/``queue``/``payload_ref``/``tenant``/``resource_class`` with a
    full fresh retry budget, and links original -> replacement for lineage.
    Returns ``{"status": "not_found"}`` for an unknown or non-dead-lettered
    id (draining is only ever defined for a genuinely dead-lettered item —
    it is not a generic resubmit).
    """
    from ...orchestration.work_item import get_work_item, submit_work_item

    item = get_work_item(engine, item_id)
    if item is None or item.get("status") != "dead_letter":
        return {"status": "not_found", "item_id": item_id}

    replacement_id = submit_work_item(
        engine,
        kind=str(item.get("kind") or ""),
        queue=item.get("queue"),
        payload_ref=str(item.get("payload_ref") or ""),
        tenant=str(item.get("tenant") or ""),
        resource_class=str(item.get("resource_class") or "default"),
        correlation_id=str(item.get("correlation_id") or "") or None,
        description=f"drained from {item_id}: {reason}".strip(": "),
        created_by=actor,
        metadata={"drained_from": item_id, "drain_reason": reason, "drained_by": actor},
    )
    try:
        from ...models.knowledge_graph import RegistryEdgeType

        engine.link_nodes(
            item_id,
            replacement_id,
            RegistryEdgeType.DRAINED_AS,
            {"_rel": "DRAINED_AS"},
        )
    except Exception as exc:  # noqa: BLE001 — the lineage edge is best-effort provenance
        logger.debug(
            "dead_letter: could not link %s -> %s: %s",
            item_id,
            replacement_id,
            exc,
            exc_info=True,
        )
    return {
        "status": "drained",
        "item_id": item_id,
        "replacement_item_id": replacement_id,
    }
