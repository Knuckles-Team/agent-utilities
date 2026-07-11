#!/usr/bin/python
from __future__ import annotations

"""``:AgentTask`` claim backend switch — engine-native ``ClaimNext``/lease/CAS, D13.

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3b)

:func:`agent_utilities.orchestration.agent_dispatch_worker.claim_agent_task` (C3/
Phase 3a) claims a durable ``:AgentTask`` by writing a dedicated ``:AgentLease``
KG node through Cypher — correct, but every claim pays a KG read + a KG write
round-trip. The epistemic-graph engine is growing a native atomic
``ClaimNext``/lease/compare-and-swap primitive purpose-built for exactly this
contention (claim the next ready unit of work, transactionally, engine-side);
once available it is strictly cheaper and races-free at the engine layer
instead of the KG-node-MERGE layer this module's KG path relies on.

This module is the SAME-SIGNATURE alternate backend for
:func:`~agent_utilities.orchestration.agent_dispatch_worker.claim_agent_task`:
selected via ``AGENT_CLAIM_BACKEND`` (``kg`` default | ``engine``), feature-
detected exactly like every other engine surface in this codebase (the
``_invoke`` helper :func:`agent_utilities.mcp.tools.engine_surface_tools._invoke`
degrades to a clean ``degraded``/``error`` JSON payload — never raises — when the
connected engine build has no matching client method).

**Never run both claim backends on one task** — this is the load-bearing
safety property, not a style preference: a task claimed via the engine's own
lease/CAS and ALSO claimed via the KG ``:AgentLease`` node would be visible as
"unclaimed" to whichever backend didn't do the claiming, and a second worker
polling that path could execute the same task concurrently. So the dispatch
here is a strict EITHER/OR, never a "try engine, also do KG for belt-and-
suspenders":

* ``AGENT_CLAIM_BACKEND=kg`` (default) — delegates straight to
  :func:`agent_dispatch_worker.claim_agent_task`; the engine path is never
  even attempted, so today's KG-only deployments are byte-identical to before
  this module existed.
* ``AGENT_CLAIM_BACKEND=engine`` — probes the engine's native claim surface
  ONCE. A LIVE result (non-degraded, non-error, with a resolvable lease) is
  used EXCLUSIVELY — the KG path is never consulted for that claim. A
  DEGRADED/unreachable result means this engine build doesn't ship the
  surface (yet); this call falls back to the KG path for this claim (a
  process-lifetime engine outage still needs *a* claim mechanism), never a
  concurrent attempt of both.
* ``AGENT_CLAIM_BACKEND=workitem`` (AU-P1-1) — routes the claim through the
  unified :mod:`~agent_utilities.orchestration.work_item` state machine
  (:func:`~agent_utilities.orchestration.work_item.claim_agent_task_via_work_item`):
  the SAME CAS/lease/fencing primitives as the ``engine``/``kg`` backends,
  now with atomic dependency release, bounded-retry-then-DLQ, and idempotent
  result commit layered on top. Still opt-in (not the default) while this
  backend proves out; see ``work_item.py``'s module docstring for the full
  migrated-vs-shimmed accounting.
"""

import logging
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.orchestration.agent_dispatch_worker import (
    CLAIM_TTL_S,
    worker_token,
)
from agent_utilities.orchestration.agent_dispatch_worker import (
    claim_agent_task as _claim_agent_task_kg,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_CLAIM_BACKEND_ENGINE",
    "AGENT_CLAIM_BACKEND_KG",
    "AGENT_CLAIM_BACKEND_WORKITEM",
    "resolve_claim_backend",
    "claim_agent_task",
]

AGENT_CLAIM_BACKEND_KG = "kg"
AGENT_CLAIM_BACKEND_ENGINE = "engine"
#: AU-P1-1: the unified WorkItem state machine backend (opt-in).
AGENT_CLAIM_BACKEND_WORKITEM = "workitem"
_BACKENDS = {
    AGENT_CLAIM_BACKEND_KG,
    AGENT_CLAIM_BACKEND_ENGINE,
    AGENT_CLAIM_BACKEND_WORKITEM,
}

#: Candidate ``(sub_client_attr, method_attr)`` probes for the engine's native
#: claim-next/lease/CAS verb — several plausible namings, first callable wins
#: (mirrors every other engine-surface probe list in this codebase, e.g.
#: ``engine_surface_tools._FEDERATED_SEARCH_CANDIDATES``).
_CLAIM_NEXT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("tasks", "claim_next"),
    ("scheduler", "claim_next"),
    ("lease", "claim_next"),
    ("tasks", "claim"),
)


def resolve_claim_backend(explicit: str | None = None) -> str:
    """Resolve the active claim backend: ``explicit`` wins, else ``AGENT_CLAIM_BACKEND``.

    Unrecognized values fall back to the safe default (``kg``) rather than
    raising — a typo'd env var must never silently disable claiming.
    """
    backend = explicit or setting("AGENT_CLAIM_BACKEND", default=AGENT_CLAIM_BACKEND_KG)
    backend = str(backend or AGENT_CLAIM_BACKEND_KG).strip().lower()
    if backend not in _BACKENDS:
        logger.warning(
            "engine_claim: unknown AGENT_CLAIM_BACKEND %r — defaulting to %r",
            backend,
            AGENT_CLAIM_BACKEND_KG,
        )
        return AGENT_CLAIM_BACKEND_KG
    return backend


def _try_engine_claim(
    task_id: str, *, token: str, now: float, claim_ttl_s: float
) -> dict[str, Any] | None:
    """Probe the engine's native claim surface ONCE; ``None`` ⇒ not live (caller falls back).

    A live engine claim returns the SAME payload shape
    :func:`agent_dispatch_worker.claim_agent_task` returns (``task_id``/
    ``lease_id``/``dag_id``/``checkpoint_id``/``depends_on_task_ids``) so
    every downstream consumer (``_execute_orchestrator_turn`` et al.) is
    backend-agnostic. Any of: unreachable engine, no matching client method,
    or a malformed/empty result ⇒ ``None`` (never raises) — the caller then
    falls back to the KG path for this claim, EXCLUSIVELY (see module
    docstring: never both).
    """
    import json as _json

    try:
        from agent_utilities.mcp.tools.engine_surface_tools import _invoke
    except Exception as e:  # noqa: BLE001 — optional import, never fatal
        logger.debug("engine_claim: engine_surface_tools unavailable: %s", e)
        return None

    try:
        raw = _invoke(
            surface="tasks",
            action="claim_next",
            graph="",
            candidates=_CLAIM_NEXT_CANDIDATES,
            params={
                "task_id": task_id,
                "owner_token": token,
                "now": now,
                "lease_ttl_s": claim_ttl_s,
            },
        )
        payload = _json.loads(raw)
    except Exception as e:  # noqa: BLE001 — engine claim is best-effort, KG is the fallback
        logger.debug("engine_claim: engine claim_next invoke failed: %s", e)
        return None

    if not (isinstance(payload, dict) and "error" not in payload):
        return None
    result = payload.get("result")
    if not isinstance(result, dict) or not result.get("claimed", True):
        # Either a malformed result, or the engine explicitly says "not
        # claimed" (e.g. someone else's live lease, or the task is terminal)
        # — either way there is nothing for the KG fallback to do either.
        return None
    lease_id = result.get("lease_id")
    if not lease_id:
        return None
    # Fencing token (AU-P0-3): use the engine's own if it already returns one
    # (``fence_token``/``lease_epoch`` — whichever name a future build ships);
    # otherwise synthesize a minimal MONOTONIC token from the claim time.
    # ``now`` is the caller-supplied claim timestamp, which only ever
    # increases across successive real claims, so it is a valid (if coarse)
    # fencing token until the engine surfaces a true lease-epoch/attempt
    # counter — a real CAS token from the engine should replace this the
    # moment the surface exposes one.
    fence_token = result.get("fence_token")
    if fence_token is None:
        fence_token = result.get("lease_epoch")
    if fence_token is None:
        fence_token = now
    return {
        "task_id": task_id,
        "lease_id": str(lease_id),
        "dag_id": result.get("dag_id") or "",
        "checkpoint_id": result.get("checkpoint_id"),
        "depends_on_task_ids": list(result.get("depends_on_task_ids") or []),
        "fence_token": fence_token,
    }


def claim_agent_task(
    engine: Any,
    task_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = CLAIM_TTL_S,
    backend: str | None = None,
) -> dict[str, Any] | None:
    """Claim one durable ``:AgentTask`` via the active backend; same signature/contract
    as :func:`agent_dispatch_worker.claim_agent_task`.

    ``backend`` overrides ``AGENT_CLAIM_BACKEND`` for this call (mainly for
    tests); production callers leave it unset and let the env/default decide.
    See the module docstring for the "never run both backends on one task"
    safety property this function enforces by construction (an EITHER/OR
    dispatch, never a fan-out to both).
    """
    import time

    token = token or worker_token()
    now = now if now is not None else time.time()
    resolved = resolve_claim_backend(backend)

    if resolved == AGENT_CLAIM_BACKEND_KG:
        return _claim_agent_task_kg(
            engine, task_id, token=token, now=now, claim_ttl_s=claim_ttl_s
        )

    if resolved == AGENT_CLAIM_BACKEND_WORKITEM:
        from agent_utilities.orchestration.work_item import (
            claim_agent_task_via_work_item,
        )

        return claim_agent_task_via_work_item(
            engine, task_id, token=token, now=now, claim_ttl_s=claim_ttl_s
        )

    # AGENT_CLAIM_BACKEND=engine — probe the engine's native claim ONCE;
    # on anything but a live result, fall back to the KG path EXCLUSIVELY
    # (never both) for this claim.
    engine_claim = _try_engine_claim(
        task_id, token=token, now=now, claim_ttl_s=claim_ttl_s
    )
    if engine_claim is not None:
        return engine_claim
    logger.debug(
        "engine_claim: engine backend unavailable for task %s — falling back to kg",
        task_id,
    )
    return _claim_agent_task_kg(
        engine, task_id, token=token, now=now, claim_ttl_s=claim_ttl_s
    )
