#!/usr/bin/python
from __future__ import annotations

"""Durable contextual-bandit outcome persistence (CONCEPT:AU-P1-3).

``CapabilityIndex.record_outcome`` only ever mutated an in-process EMA — every
learned routing preference was lost on restart. This module closes that gap by
persisting the reward EMA onto the engine's own node properties (mirrors
``retrieval_quality.UsageTelemetry.flush_to_engine``'s "persist a learned scalar
onto the node it was learned about" pattern), so routing learning survives a
process restart: the EMA is read back from the node the next time it is seen
(via :meth:`.capability_index.CapabilityIndex.add`'s ``reward`` parameter), not
reset to the neutral prior.

Deliberately NOT a new node type / table — the reward lives as three plain
properties (``capability_reward``, ``capability_reward_count``,
``capability_reward_updated_at``) on the SAME capability node the engine already
carries, exactly like ``trust_score`` on a memory node. No engine dependency is
introduced beyond the ``backend.execute`` Cypher surface every other reactive
writer in this codebase already uses.
"""

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["persist_capability_reward", "read_capability_reward"]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_capability_reward(engine: Any, entity_id: str) -> float | None:
    """Read the durably persisted reward EMA for ``entity_id``, or ``None``.

    ``None`` means "no durable record yet" (new entity, or no engine backend
    reachable) — the caller should treat this as the neutral 0.5 prior, exactly
    like the in-process :meth:`CapabilityIndex.reward_of` default.
    """
    backend = getattr(engine, "backend", None)
    if backend is None:
        return None
    try:
        rows = backend.execute(
            "MATCH (n) WHERE n.id = $id RETURN n.capability_reward AS reward",
            {"id": entity_id},
        )
    except Exception as e:  # noqa: BLE001 — durable read is best-effort
        logger.debug(
            "durable_outcome_store: reward read failed for %r: %s", entity_id, e
        )
        return None
    for row in rows or ():
        if isinstance(row, dict) and row.get("reward") is not None:
            try:
                return float(row["reward"])
            except (TypeError, ValueError):
                return None
    return None


def persist_capability_reward(
    engine: Any,
    entity_id: str,
    *,
    success: bool | None = None,
    reward: float | None = None,
    alpha: float = 0.3,
) -> float | None:
    """Durably update the reward EMA for ``entity_id`` on the engine (CONCEPT:AU-P1-3).

    Reads the current durable EMA (0.5 if none recorded yet), applies the same
    EMA update :meth:`CapabilityIndex.record_outcome` uses, and writes it back
    onto the node's ``capability_reward``/``capability_reward_count``/
    ``capability_reward_updated_at`` properties. Returns the updated reward, or
    ``None`` when no engine backend is reachable (the caller should keep relying
    on the in-process EMA only — durability is a best-effort augmentation, never
    load-bearing for the routing decision itself).
    """
    if reward is None:
        if success is None:
            raise ValueError("persist_capability_reward requires success or reward")
        reward = 1.0 if success else 0.0
    reward = min(1.0, max(0.0, float(reward)))

    backend = getattr(engine, "backend", None)
    if backend is None:
        return None

    try:
        rows = backend.execute(
            "MATCH (n) WHERE n.id = $id RETURN n.capability_reward AS reward, "
            "n.capability_reward_count AS count",
            {"id": entity_id},
        )
    except Exception as e:  # noqa: BLE001 — durable read is best-effort
        logger.debug(
            "durable_outcome_store: reward read failed for %r: %s", entity_id, e
        )
        rows = []

    prev, count = 0.5, 0
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        if row.get("reward") is not None:
            try:
                prev = float(row["reward"])
            except (TypeError, ValueError):
                prev = 0.5
        count = int(row.get("count") or 0)
        break

    updated = (1.0 - alpha) * prev + alpha * reward

    try:
        backend.execute(
            "MATCH (n) WHERE n.id = $id SET n.capability_reward = $r, "
            "n.capability_reward_count = $c, n.capability_reward_updated_at = $ts",
            {
                "id": entity_id,
                "r": updated,
                "c": count + 1,
                "ts": _now_iso(),
            },
        )
    except Exception as e:  # noqa: BLE001 — the caller still gets the computed value
        logger.debug(
            "durable_outcome_store: reward write failed for %r: %s", entity_id, e
        )
        return updated
    return updated
