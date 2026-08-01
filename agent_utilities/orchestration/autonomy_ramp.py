#!/usr/bin/python
from __future__ import annotations

"""Earned-autonomy ramp for the governance gate (CONCEPT:AU-OS.governance.autonomy-change-proposer).

Closes the trust half of the operating loop: an action class an actor has performed
*verifiably correctly* enough times graduates from ``approval_required`` ("ask") to
``auto_notify`` ("allow, but tell me") — "200 P3s closed correctly → close P3s
without approval." The trust signal is a per-(actor, action-kind) success ledger
(:TrustScore nodes, in-memory fallback), fed by the action-outcome loop (AHE-3.62)
and consulted by :meth:`ActionPolicy.classify`.

**Safe by default.** Graduation only ever happens for action kinds an operator has
explicitly opted into the policy's ``ramp_eligible`` allowlist — an empty allowlist
(the default) means the ramp mechanism is live but changes nothing, so the security
fence is never silently loosened. It is a one-rung graduation (ask→auto_notify, never
to silent auto) and never touches ``forbidden``.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MIN_SAMPLES = 20
DEFAULT_THRESHOLD = 0.9


def trust_key(actor_id: str, action_kind: str) -> str:
    return f"trust:{actor_id or 'anon'}:{action_kind or '*'}"


def assess_trust(
    total: int,
    successes: int,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Pure ramp predicate: enough samples AND a high enough success rate."""
    if total < min_samples:
        return False
    return (successes / total) >= threshold


# ── persistence (KG :TrustScore node; degrades to an in-memory dict) ───────────
_MEM: dict[str, tuple[int, int]] = {}  # key -> (total, successes)


def _read(backend: Any, key: str) -> tuple[int, int]:
    # D-DST-6: record_trust() updates _MEM unconditionally BEFORE its guarded
    # KG persist attempt (below), but this read prefers the backend whenever
    # it answers without raising. If a persist ever failed, the backend total
    # is frozen at its old value -- and without this guard, every SUBSEQUENT
    # record_trust() call would silently overwrite _MEM's fresher value with
    # that stale backend row, making every outcome recorded during the outage
    # (success or failure) invisible on every later read. `total` only ever
    # grows (record_trust increments it by exactly 1 per call), so a backend
    # total behind _MEM's means a prior persist failed -- never let that stale
    # row silently swallow unpersisted outcomes.
    mem_total, mem_successes = _MEM.get(key, (0, 0))
    if backend is not None and hasattr(backend, "execute"):
        try:
            rows = backend.execute(
                "MATCH (t:TrustScore {id: $id}) RETURN t.total AS total, "
                "t.successes AS successes",
                {"id": key},
            )
            for r in rows or []:
                b_total = int(r.get("total") or 0)
                b_successes = int(r.get("successes") or 0)
                if b_total >= mem_total:
                    return b_total, b_successes
                return mem_total, mem_successes
        except Exception:  # pragma: no cover - read best-effort
            pass
    return mem_total, mem_successes


def record_trust(
    backend: Any, actor_id: str, action_kind: str, success: bool
) -> tuple[int, int]:
    """Record one verified outcome for (actor, action_kind); returns (total, successes)."""
    key = trust_key(actor_id, action_kind)
    total, successes = _read(backend, key)
    total += 1
    successes += 1 if success else 0
    _MEM[key] = (total, successes)
    if backend is not None and hasattr(backend, "add_node"):
        try:
            backend.add_node(
                key,
                node_type="TrustScore",
                actor_id=actor_id,
                action_kind=action_kind,
                total=total,
                successes=successes,
            )
        except Exception as exc:  # pragma: no cover  # noqa: BLE001 — persist is best-effort; _read() now falls back to _MEM whenever the backend's total lags (monotonic-total guard above), so a failed persist here only delays cross-process durability, it can no longer make clears_ramp() silently miss unpersisted successes/failures
            logger.debug("trust persist failed: %s", exc)
    return total, successes


def clears_ramp(
    backend: Any,
    actor_id: str,
    action_kind: str,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """True when (actor, action_kind) has earned graduation past the gate."""
    total, successes = _read(backend, trust_key(actor_id, action_kind))
    return assess_trust(total, successes, min_samples=min_samples, threshold=threshold)


def reset_trust() -> None:
    """Clear the in-memory trust cache (tests)."""
    _MEM.clear()
