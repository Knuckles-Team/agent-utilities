#!/usr/bin/python
from __future__ import annotations

"""WorkItem claim pacing — the Python-side cooperative half of engine backpressure.

CONCEPT:AU-ORCH.scheduling.claim-pacing-backpressure (W2.9 — cluster-wide backpressure
unification, generalizing EG-044/W2.4's engine-native QoS lanes).

THE SEAM (why this module exists)
----------------------------------
The engine's admission gate (``epistemic-graph``'s baseline ``EPISTEMIC_GRAPH_MAX_INFLIGHT``
cap, and — once a build carries W2.4 — the opt-in per-class ``QosScheduler``,
``src/server/qos.rs``) is the sole **authority**: it decides who is admitted and who is
shed, and it reserves headroom so ``Interactive > Orch > Hydration > Ingest`` under
contention. Until this module, that authority and :mod:`agent_utilities.orchestration.
work_item`'s claim loop were **disjoint layers**: a shed ``BUSY: …`` response from the
engine was just another transient error to whatever generic exception handling the
caller happened to have (or didn't) — nothing told the NEXT claim attempt to slow down,
so a claim loop kept re-issuing the identical request at full speed, wasting engine
cycles on requests it had *just* told the client it would shed (the "hammering" this
module stops), and doing so while completely blind to *which class* was contending.

**Unified model:** the engine's lanes remain the sole admission authority — this module
never second-guesses an admission decision or grants extra capacity — but the Python
claim loop now behaves as a **cooperative participant**: it remembers, per
:class:`~agent_utilities.core.resource_priority.PriorityClass`, that class C was just
shed, and stops attempting NEW claims of class C for a computed window (exponential
backoff with a cap, jittered) instead of re-hitting the wire immediately. The window
shrinks to nothing the moment the engine actually admits (or cleanly answers "nothing to
claim" for) that class again — so recovery is exactly as fast as the engine's own signal,
never slower, and never a fixed guess. Pacing is **per class**, not global: an
``Interactive``-class claim loop is never paced by an ``Ingest`` flood's backoff (they key
off completely disjoint state) — this is the client-side mirror of the engine's own
per-class ceilings, so a background-ingestion consumer backing off can never look, from
the outside, like an interactive one going quiet too.

Detection is on the WIRE MESSAGE, not a dedicated exception type (re-verified on HEAD,
2026-07-24): :meth:`epistemic_graph.client.EpistemicGraphClient._send` raises a plain
``RuntimeError(err_msg)`` for every non-OK response — there is no ``EngineBusyError``/
``ResourceExhaustedError`` class today, unlike the dedicated ``ResultTooLargeError``/
``StaleRouteError`` the same method DOES special-case. Every shed, whether from the
always-on baseline in-flight cap or the opt-in QoS scheduler's four reasons (Quota /
FairShare / Backpressure / RateLimited — ``qos.rs::QosReject::busy_message()``), is
wire-identical: a ``RuntimeError`` whose message starts with the literal ``"BUSY:"``
prefix (``src/server/transport.rs:691``, ``qos.rs:298-306``). :func:`is_busy_shed`
detects on that prefix, so pacing engages against ANY engine version — a pre-W2.4 engine
(no per-class QoS, just the blunt baseline cap) benefits exactly as much as a W2.4 one;
this module does not require or read back the engine's own class attribution, only the
CALLER's own ambient :class:`PriorityClass` (the same contextvar
:meth:`agent_utilities.knowledge_graph.core.session.GraphSession.engine_verified_context`
now forwards as the wire ``priority`` claim once an engine build carries W2.4 — see that
method's docstring for the W2.4-2 deploy-ordering constraint).

Wired in at the ONE choke point every claim path already shares (native by default, no
caller changes anywhere): :func:`agent_utilities.orchestration.work_item.claim_specific`
and :func:`~agent_utilities.orchestration.work_item.claim_next` are the sole two
"claiming" entry points into the engine-native ``claim_work_item`` verb — AgentTask
bridge claims, orchestrator-work-item claims, ingest-task claims, loop claims, and any
future claim caller all route through exactly one of those two functions, so pacing them
covers the whole system with zero per-caller wiring.
"""

import logging
import threading
import time as _time_module
from typing import Any

from agent_utilities.core.resource_priority import PriorityClass, current_priority
from agent_utilities.orchestration.resilience import ResiliencePolicy, compute_backoff

logger = logging.getLogger(__name__)

__all__ = [
    "ClaimPaced",
    "DEFAULT_CLAIM_PACING_POLICY",
    "is_busy_shed",
    "pending_pace_seconds",
    "raise_if_paced",
    "record_claim_shed",
    "record_claim_admitted",
    "claim_pacing_snapshot",
    "reset_claim_pacing",
]

#: Untagged ambient context resolves to ORCHESTRATION for pacing purposes too —
#: the SAME "no claim ⇒ high, never starved" default the engine's own
#: ``QosClass::from_priority_claim`` uses (qos.rs) and
#: :mod:`agent_utilities.core.resource_priority` uses internally, so an
#: unclassified claim loop shares pacing state with explicit ORCHESTRATION
#: rather than opening a third, untracked bucket.
_DEFAULT_CLASS = PriorityClass.ORCHESTRATION

#: Claim-pacing policy AS DATA (ADR-5 division: the engine owns admission
#: authority; this policy only shapes how eagerly Python retries claiming
#: after being told to back off). Deliberately a much shorter base delay than
#: :data:`agent_utilities.orchestration.work_item.DEFAULT_BACKOFF_BASE_S`
#: (30s, a WorkItem's own post-failure retry-after backoff, a different
#: concept): a claim-pacing window governs "how soon may I even ATTEMPT to
#: claim again", so it should react in the sub-second-to-low-second range,
#: not the tens-of-seconds range a failed unit of WORK waits before retry.
#: Tune by replacing this policy instance — the pacing LOGIC below never
#: hardcodes a number.
DEFAULT_CLAIM_PACING_POLICY = ResiliencePolicy(
    backoff_base_s=0.25,
    backoff_factor=2.0,
    max_backoff_s=30.0,
    backoff_strategy="exponential",
    jitter=True,
    jitter_strategy="proportional",
    name="claim_pacing",
)


class ClaimPaced(RuntimeError):
    """Raised by ``work_item.claim_specific``/``claim_next`` when this claim's
    class is inside an active client-side backoff window from a recently
    observed engine shed — the engine is NOT contacted for this attempt (that
    is the entire point: stop hammering a lane that just said BUSY).

    The message is ``BUSY:``-prefixed on purpose: whether a caller's request
    was shed by the engine just now, or preempted client-side because the
    engine said so moments ago, it should read — and be handled — identically
    by any existing generic error handling (:func:`is_busy_shed` recognizes
    both). No caller needs new exception-handling code for this class to
    behave correctly; at worst an un-adapted caller sees the same kind of
    failure it already tolerated, just faster and less frequently.
    """


class _ClassPacingState:
    """Mutable per-class pacing state. Not a dataclass — genuinely mutable,
    thread-shared counters, the DATA/behavior split ADR-5 draws around."""

    __slots__ = ("consecutive_sheds", "paced_until")

    def __init__(self) -> None:
        self.consecutive_sheds = 0
        self.paced_until = 0.0


_lock = threading.Lock()
_state: dict[PriorityClass, _ClassPacingState] = {}


def _effective_class(priority: PriorityClass | None) -> PriorityClass:
    if priority is not None:
        return priority
    ambient = current_priority()
    return ambient if ambient is not None else _DEFAULT_CLASS


def _state_for(cls: PriorityClass) -> _ClassPacingState:
    with _lock:
        state = _state.get(cls)
        if state is None:
            state = _ClassPacingState()
            _state[cls] = state
        return state


def _now(now: float | None) -> float:
    return now if now is not None else _time_module.monotonic()


def is_busy_shed(exc: BaseException) -> bool:
    """True iff ``exc`` is the engine's backpressure shed signal (``BUSY: …``).

    Matches on the wire message prefix, not a dedicated exception type — see
    the module docstring for why (no ``EngineBusyError`` exists in the client
    today). Deliberately narrow: a :class:`WorkItemBackendUnavailable` /
    :class:`NativeWorkItemRequired` (missing native support — a structural
    problem, not overload) or any other ``RuntimeError`` does NOT match, so
    pacing never engages for an error that has nothing to do with load.
    """
    return isinstance(exc, RuntimeError) and str(exc).startswith("BUSY:")


def pending_pace_seconds(
    priority: PriorityClass | None = None, *, now: float | None = None
) -> float:
    """Seconds remaining in ``priority``'s (or the ambient class's) active
    backoff window; ``0.0`` when not paced. Pure read — never mutates state."""
    cls = _effective_class(priority)
    state = _state_for(cls)
    with _lock:
        remaining = state.paced_until - _now(now)
    return max(0.0, remaining)


def raise_if_paced(
    priority: PriorityClass | None = None, *, now: float | None = None
) -> None:
    """Raise :class:`ClaimPaced` without touching the engine if ``priority``
    (or the ambient class) is still inside its backoff window from a prior
    shed. A no-op otherwise. Called BEFORE every native claim attempt."""
    remaining = pending_pace_seconds(priority, now=now)
    if remaining <= 0.0:
        return
    cls = _effective_class(priority)
    raise ClaimPaced(
        f"BUSY: client-side claim pacing active for class {cls.value!r}, "
        f"retry after {remaining:.3f}s (engine not contacted)"
    )


def record_claim_shed(
    priority: PriorityClass | None = None,
    *,
    policy: ResiliencePolicy = DEFAULT_CLAIM_PACING_POLICY,
    rng: Any = None,
    now: float | None = None,
) -> float:
    """Record a REAL engine shed for ``priority`` (or the ambient class) and
    compute the next backoff window (exponential with cap + jitter, reusing
    :func:`agent_utilities.orchestration.resilience.compute_backoff` so the
    growth math is the SAME curve every other resilience policy in this
    codebase uses). Returns the computed delay. Only ever called after the
    engine itself answered BUSY — never from :func:`raise_if_paced`'s
    preemptive path, so re-hitting an already-known-paced class does not
    compound the backoff beyond what the engine has actually signalled."""
    cls = _effective_class(priority)
    state = _state_for(cls)
    with _lock:
        state.consecutive_sheds += 1
        attempt = state.consecutive_sheds
        delay = compute_backoff(attempt, policy, rng=rng)
        state.paced_until = _now(now) + delay
    logger.info(
        "[claim_pacing] class=%s shed #%d -> backing off %.3fs (cap=%.1fs)",
        cls.value,
        attempt,
        delay,
        policy.max_backoff_s,
    )
    return delay


def record_claim_admitted(
    priority: PriorityClass | None = None, *, now: float | None = None
) -> None:
    """Record a NON-shed engine response for ``priority`` (or the ambient
    class) — a positive claim or a legitimate negative ("nothing to claim")
    both count, since either proves the engine is answering this class
    normally. Resets the class's backoff to zero: recovery is exactly as fast
    as the engine's own next real answer, never a fixed guess."""
    cls = _effective_class(priority)
    state = _state_for(cls)
    with _lock:
        had_backoff = state.consecutive_sheds > 0
        sheds = state.consecutive_sheds
        state.consecutive_sheds = 0
        state.paced_until = 0.0
    if had_backoff:
        logger.info(
            "[claim_pacing] class=%s recovered after %d shed(s); backoff cleared",
            cls.value,
            sheds,
        )


def claim_pacing_snapshot(*, now: float | None = None) -> dict[str, dict[str, float]]:
    """Observability/test snapshot of every class with live pacing state,
    keyed by :class:`PriorityClass` wire value."""
    at = _now(now)
    with _lock:
        return {
            cls.value: {
                "consecutive_sheds": float(state.consecutive_sheds),
                "paced_remaining_s": max(0.0, state.paced_until - at),
            }
            for cls, state in _state.items()
        }


def reset_claim_pacing() -> None:
    """Drop all per-class pacing state (test isolation / process restart)."""
    with _lock:
        _state.clear()
