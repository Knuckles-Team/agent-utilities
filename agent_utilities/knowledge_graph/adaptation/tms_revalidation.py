#!/usr/bin/python
from __future__ import annotations

"""TMS staleness consumer — Seam 3 follow-up (CONCEPT:EG-KG.epistemic.truth-maintenance, W3.2).

The engine's durable reasoning projection (``reasoning_projection::spawn``)
already tails the mutation outbox and auto-registers a materialization the
moment a derived node/edge carries recognized provenance (``invalidation_deps``
or a ``:DerivedFrom``/``:GeneratedBy`` edge — see
``candidate_insight.register_claim_materialization``,
``capability_designation._register_capability_reward_materialization``, and
``ContextCompiler._register_bundle_materialization``, the three AU-side
producers this closes the loop for). Until now nothing on the AU side ever
CONSUMED the resulting staleness signal — a materialization could go
``Stale`` and simply sit there forever.

This module is that consumer. A daemon tick (``tms_revalidation`` in the
engine's maintenance scheduler, default ON, background priority) periodically:

1. Reads :meth:`~agent_utilities.knowledge_graph.core.engine.KnowledgeGraphEngine.
   stale_materializations` as a cheap "is anything stale at all" gate — its ids
   are the engine's own privacy-hashed projection identities, not real graph
   ids, so they cannot be reverse-mapped; an empty result skips the rest of
   the tick for free.
2. Otherwise enumerates a BOUNDED candidate set of known derived artifacts per
   owner kind (mined ``Claim``s, capability-index entries carrying a durable
   ``capability_reward``, cached ``ContextBundleMaterialization`` markers) and
   probes each one's REAL status via
   :meth:`~agent_utilities.knowledge_graph.core.engine.KnowledgeGraphEngine.
   materialization_status` (which accepts the true id and hashes it internally).
3. Routes every id actually ``"Stale"`` to its owner's re-validation action —
   never a generic one-size mutation:

   * mined claim → propose a NEW ``:BeliefRevisionProposal`` node (propose-only,
     the SAME advisory-node convention ``loop_controller._run_belief_revision``
     already uses) — the ``Claim`` node itself is NEVER mutated here;
   * capability-index entry → evict it from the process-local
     ``CapabilityIndexWatcher`` cache (a pure performance-cache invalidation,
     safe to do unconditionally — the next ``designate_specialists`` call
     re-scores it fresh from the engine's live properties);
   * cached context bundle → drop it from the shared KV cache and retire its
     tracking marker node (``delete_node``, which the engine's projection
     records as ``Retracted`` — a terminal, no-longer-stale state, so it never
     re-surfaces on a later tick).

Stateless by design: every tick re-reads the engine's durable projection from
scratch (nothing here is cached/remembered across ticks), so the durable redb
store already backing that projection is the ONLY persistence this needs —
this module never grows one of its own, matching the "prove restart survives"
requirement of the audit item it closes.

Best-effort throughout, matching every other maintenance tick in this
codebase: a denied/absent engine surface (no ``stale_materializations``, no
``materialization_status``, no ``query_cypher``) degrades to a skipped/partial
report and is logged, never raised.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["revalidate_stale_materializations"]

#: Per-tick, per-owner-kind scan budget — bounded so a large backlog can never
#: wedge the scheduler (mirrors ``anomaly_consumer.DEFAULT_SCAN_LIMIT``).
DEFAULT_CANDIDATE_LIMIT = 50

#: One Cypher probe per owner kind — the id column is always ``id``; the
#: context-bundle probe also needs ``cache_key`` back to drop the right KV entry.
_CANDIDATE_QUERIES: dict[str, str] = {
    "claim": "MATCH (n:Claim) RETURN n.id AS id ORDER BY n.id LIMIT $limit",
    "capability_index": (
        "MATCH (n) WHERE n.capability_reward IS NOT NULL "
        "RETURN n.id AS id ORDER BY n.id LIMIT $limit"
    ),
    "context_bundle": (
        "MATCH (n:ContextBundleMaterialization) "
        "RETURN n.id AS id, n.cache_key AS cache_key ORDER BY n.id LIMIT $limit"
    ),
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _candidates(
    engine: Any, kind: str, limit: int, errors: list[str]
) -> list[dict[str, Any]]:
    query_cypher = getattr(engine, "query_cypher", None)
    if not callable(query_cypher):
        errors.append(f"{kind}:candidates engine has no query_cypher")
        return []
    try:
        rows = query_cypher(_CANDIDATE_QUERIES[kind], {"limit": int(limit)})
    except Exception as e:  # noqa: BLE001 — a scan failure degrades, never raises
        errors.append(f"{kind}:candidates {e}")
        return []
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("id"):
            out.append(row)
    return out


def _revalidate_claim(engine: Any, claim_id: str, errors: list[str]) -> bool:
    """Propose a NEW ``:BeliefRevisionProposal`` for a stale claim.

    Propose-only: this NEVER writes to ``claim_id`` itself (no
    ``compare_and_set_node_fields``, no repeat ``add_node`` on the same id, no
    ``delete_node``) — only a distinct advisory node a human/agent reviewer
    later acts on, mirroring ``loop_controller._run_belief_revision``'s own
    ``:BeliefRevisionProposal`` convention.
    """
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        errors.append(f"claim:{claim_id} engine has no add_node")
        return False
    now = _now_iso()
    proposal_id = f"BeliefRevisionProposal:{claim_id}:{now}"
    try:
        add_node(
            proposal_id,
            "BeliefRevisionProposal",
            properties={
                "status": "proposal",
                "belief_id": claim_id,
                "reason": "tms_stale_materialization",
                "proposed_at": now,
            },
        )
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        errors.append(f"claim:{claim_id} propose failed: {e}")
        return False
    return True


def _revalidate_capability_index_entry(
    engine: Any, entity_id: str, errors: list[str]
) -> bool:
    """Evict a stale entry from the process-local capability-index cache.

    A pure performance-cache invalidation (never a graph write): the durable
    ``capability_reward`` property on the engine node is untouched; the next
    ``designate_specialists``/``record_capability_outcome`` call re-admits and
    re-scores ``entity_id`` fresh.
    """
    watcher = getattr(engine, "_capability_index_watcher", None)
    if watcher is None:
        return False
    index = getattr(watcher, "index", None)
    if index is None:
        return False
    try:
        return bool(index.remove(entity_id))
    except Exception as e:  # noqa: BLE001 — cache eviction is best-effort
        errors.append(f"capability_index:{entity_id} evict failed: {e}")
        return False


def _revalidate_context_bundle(
    engine: Any, marker_id: str, cache_key: str, errors: list[str]
) -> bool:
    """Drop a stale bundle from the shared KV cache and retire its marker.

    ``kv_backend`` is resolved from the SAME process-wide accessor
    :meth:`ContextCompiler` stores compiled bundles through
    (:func:`~agent_utilities.core.contextual_model.get_context_compiler_cache`).
    The duck-typed KV contract (``get``/``put``) has no mandatory delete
    primitive today, so an optional ``delete`` is called only when present —
    the marker node's removal alone still bounds this artifact: the engine
    records it ``Retracted`` (terminal), so it never re-surfaces on a later
    tick even when the underlying cache entry could not be actively evicted.
    """
    acted = False
    if cache_key:
        try:
            from agent_utilities.core.contextual_model import (
                get_context_compiler_cache,
            )

            kv_backend = get_context_compiler_cache()
        except Exception as e:  # noqa: BLE001 — resolving the cache is best-effort
            errors.append(f"context_bundle:{marker_id} cache lookup failed: {e}")
            kv_backend = None
        if kv_backend is not None:
            delete = getattr(kv_backend, "delete", None)
            if callable(delete):
                try:
                    delete(cache_key)
                    acted = True
                except Exception as e:  # noqa: BLE001 — eviction is best-effort
                    errors.append(f"context_bundle:{marker_id} kv delete failed: {e}")
    delete_node = getattr(engine, "delete_node", None)
    if callable(delete_node):
        try:
            delete_node(marker_id)
            acted = True
        except Exception as e:  # noqa: BLE001 — marker retirement is best-effort
            errors.append(f"context_bundle:{marker_id} delete_node failed: {e}")
    return acted


_OWNER_ACTIONS = {
    "claim": _revalidate_claim,
    "capability_index": _revalidate_capability_index_entry,
}


def revalidate_stale_materializations(
    engine: Any, *, limit: int = DEFAULT_CANDIDATE_LIMIT
) -> dict[str, Any]:
    """One bounded revalidation pass over every stale TMS materialization.

    Stateless: re-reads :meth:`stale_materializations`/:meth:`materialization_status`
    from scratch every call — nothing here persists across ticks, so a restart
    loses nothing (the durable state is entirely the engine's own redb-backed
    reasoning projection). Returns a JSON-able report (``scanned``/``stale``/
    ``revalidated`` per owner kind/``errors``); never raises — an absent or
    denied engine surface degrades to a report with ``errors`` populated and
    every count at zero.
    """
    errors: list[str] = []
    stale_probe = getattr(engine, "stale_materializations", None)
    if not callable(stale_probe):
        errors.append("engine has no stale_materializations")
        return {"scanned": 0, "stale": 0, "revalidated": {}, "errors": errors}
    try:
        stale_refs = stale_probe()
    except Exception as e:  # noqa: BLE001 — the cheap gate degrades, never raises
        errors.append(f"stale_materializations failed: {e}")
        return {"scanned": 0, "stale": 0, "revalidated": {}, "errors": errors}
    if not stale_refs:
        # Nothing stale anywhere in this graph — skip every per-candidate probe.
        return {"scanned": 0, "stale": 0, "revalidated": {}, "errors": errors}

    status_probe = getattr(engine, "materialization_status", None)
    if not callable(status_probe):
        errors.append("engine has no materialization_status")
        return {"scanned": 0, "stale": 0, "revalidated": {}, "errors": errors}

    scanned = 0
    stale = 0
    revalidated: dict[str, int] = {
        "claim": 0,
        "capability_index": 0,
        "context_bundle": 0,
    }
    for kind in ("claim", "capability_index", "context_bundle"):
        for row in _candidates(engine, kind, limit, errors):
            candidate_id = str(row["id"])
            scanned += 1
            try:
                status = status_probe(candidate_id)
            except Exception as e:  # noqa: BLE001 — a probe failure degrades, never raises
                errors.append(f"{kind}:{candidate_id} status probe failed: {e}")
                continue
            if status != "Stale":
                continue
            stale += 1
            if kind == "context_bundle":
                acted = _revalidate_context_bundle(
                    engine, candidate_id, str(row.get("cache_key") or ""), errors
                )
            else:
                acted = _OWNER_ACTIONS[kind](engine, candidate_id, errors)
            if acted:
                revalidated[kind] += 1

    report = {
        "scanned": scanned,
        "stale": stale,
        "revalidated": revalidated,
        "errors": errors,
    }
    if stale:
        logger.info(
            "[EG-KG.epistemic.truth-maintenance] tms_revalidation: scanned=%d stale=%d revalidated=%s",
            scanned,
            stale,
            revalidated,
        )
    return report
