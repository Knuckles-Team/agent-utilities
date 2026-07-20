#!/usr/bin/python
from __future__ import annotations

"""``:AgentTask`` claim backend resolution — the unified WorkItem state machine (D13, AU-P1-1).

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3b)

**No-Legacy history (report §2 Claims seam / §9 #4).** This module used to be a
three-way backend switch (``kg`` default | ``engine`` | ``workitem``): ``kg``
delegated to a KG-``:AgentLease``-node claim primitive in
``agent_dispatch_worker``, and ``engine`` probed a set of never-shipped
engine-native namespace/method-name guesses
(``_CLAIM_NEXT_CANDIDATES`` — ``tasks``/``scheduler``/``lease`` × ``claim_next``/
``claim``) that the prior audit flagged as dead code, unused by default. Both are
now GONE:

* The ``engine`` backend's entire implementation WAS that namespace-probing dead
  code — there is nothing else to preserve once it is removed.
* The ``kg`` backend's own underlying primitive
  (``agent_dispatch_worker.claim_agent_task``/``CLAIM_TTL_S``) no longer exists —
  the AU-P1-1/"authority-convergence" assimilation rewrote ``agent_dispatch_worker``
  around the engine-native ``WorkItem`` state machine directly
  (``execute_work_item_turn``/``claim_orchestrator_work_item``), retiring the
  KG-``:AgentLease``-only dispatch path this backend called into. Keeping a
  ``kg`` option selectable — even a fail-loud one — would be exactly the "left-
  behind legacy reference" No-Legacy forbids, so it is deleted, not shimmed.

:func:`claim_agent_task` therefore has ONE live backend —
:func:`~agent_utilities.orchestration.work_item.claim_agent_task_via_work_item`
(the SAME CAS/lease/fencing primitives the removed backends used, now with
atomic dependency release, bounded-retry-then-DLQ, and idempotent result commit
layered on top; see ``work_item.py``'s module docstring for the full
migrated-vs-shimmed accounting). :func:`resolve_claim_backend` still validates
``AGENT_CLAIM_BACKEND`` against the known-backend set (fail-safe, never silently
disable claiming on a typo'd env var) rather than being inlined away, so a future
second backend has a real seam to land in — but today resolving anything other
than ``workitem`` is unreachable in practice.
"""

import logging
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_CLAIM_BACKEND_WORKITEM",
    "resolve_claim_backend",
    "claim_agent_task",
]

#: AU-P1-1: the unified WorkItem state machine — the sole claim backend.
AGENT_CLAIM_BACKEND_WORKITEM = "workitem"
_BACKENDS = {AGENT_CLAIM_BACKEND_WORKITEM}


def resolve_claim_backend(explicit: str | None = None) -> str:
    """Resolve the active claim backend: ``explicit`` wins, else ``AGENT_CLAIM_BACKEND``.

    Unrecognized values fall back to the safe default (``workitem``, the only
    backend) rather than raising — a typo'd env var must never silently disable
    claiming.
    """
    backend = explicit or setting(
        "AGENT_CLAIM_BACKEND", default=AGENT_CLAIM_BACKEND_WORKITEM
    )
    backend = str(backend or AGENT_CLAIM_BACKEND_WORKITEM).strip().lower()
    if backend not in _BACKENDS:
        logger.warning(
            "engine_claim: unknown AGENT_CLAIM_BACKEND %r — defaulting to %r",
            backend,
            AGENT_CLAIM_BACKEND_WORKITEM,
        )
        return AGENT_CLAIM_BACKEND_WORKITEM
    return backend


def claim_agent_task(
    engine: Any,
    task_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float | None = None,
    backend: str | None = None,
) -> dict[str, object] | None:
    """Claim one durable ``:AgentTask`` through the WorkItem state machine.

    ``backend`` overrides ``AGENT_CLAIM_BACKEND`` for this call (mainly for
    tests); production callers leave it unset. ``claim_ttl_s`` defaults to
    :data:`~agent_utilities.orchestration.work_item.DEFAULT_LEASE_TTL_S` when
    omitted. See the module docstring for why this is a single-backend
    resolver rather than the three-way switch it used to be.
    """
    import time

    from agent_utilities.orchestration.work_item import (
        DEFAULT_LEASE_TTL_S,
        claim_agent_task_via_work_item,
    )

    token = token or _worker_token()
    now = now if now is not None else time.time()
    resolve_claim_backend(backend)  # validates/logs an unrecognized override
    ttl = DEFAULT_LEASE_TTL_S if claim_ttl_s is None else claim_ttl_s
    return claim_agent_task_via_work_item(
        engine, task_id, token=token, now=now, claim_ttl_s=ttl
    )


def _worker_token() -> str:
    from agent_utilities.orchestration.agent_dispatch_worker import worker_token

    return worker_token()
