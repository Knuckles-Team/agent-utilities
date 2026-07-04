"""Unified resource-priority class + priority-aware shared-LLM admission.

CONCEPT:AU-ORCH.scheduling.resource-priority-edict (the priority class + carrier) / CONCEPT:AU-ORCH.scheduling.also-fold-vllm-scheduler (the LLM
admission gate) / CONCEPT:AU-KG.compute.priority-class-propagation (cross-component propagation).

THE EDICT (the operator's law)
------------------------------
Ingestion and orchestration share the SAME LLM (the qwen vLLM generator), the
SAME epistemic-graph, and the SAME agent-utilities runtime — but they must never
bottleneck each other. Interactive / orchestration work (a live Claude or
end-user query, a skill / workflow execution, cron-driven orchestration) is
ALWAYS prioritised over background ingestion of documents / codebases / research
papers, dynamically, with NO blocking: background work yields to higher-priority
work when it is actively contending, and uses the spare capacity when it is not
(dynamic scaling, never starved to zero). The ONE explicit exception is initial
**skill + MCP hydration** — foundational bootstrap, NOT deprioritised.

This module is the single source of truth for that priority and the client-side
enforcement on the shared LLM. It deliberately does NOT re-implement the two
reserved lanes that already exist elsewhere — it is the third edge of the same
triangle, keyed off the SAME :class:`PriorityClass` currency:

* the **host worker** reserved interactive lane —
  :class:`agent_utilities.knowledge_graph.core.worker_scheduler.AdmissionPolicy`
  (CONCEPT:AU-KG.compute.interactive-lane-floor) reserves a worker floor that non-interactive lanes can never
  claim, so an interactive task always lands a worker slot.
* the **engine** reserved read lane (EG-044, in epistemic-graph) keeps a read
  slot for interactive reads under a saturating ingest write-storm.
* the **LLM** reserved admission (this module) keeps generator capacity for
  interactive/orchestration/hydration calls under a saturating enrichment fan-out.

All three are the same shape — a reserved floor the lowest class may never spend —
so one interactive request gets a worker slot **and** an engine read **and** an
LLM slot ahead of background ingestion, end to end.

Contention map
--------------
* **qwen vLLM generator** — SHARED by orchestration generation AND ingestion
  enrichment. This is the gate that matters; admission is enforced per generator
  model key here.
* **bge-m3 embeddings** — a SEPARATE endpoint, so it gets a SEPARATE gate key and
  never contends with the generator. (Embedding fan-out is gated only against
  other embedding fan-out; see ``model="embedding"`` in the enrichment path.)
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
from collections.abc import AsyncIterator, Iterator
from enum import Enum

__all__ = [
    "PriorityClass",
    "PRIORITY_HEADER",
    "HYDRATION_TASK_TYPES",
    "current_priority",
    "set_priority",
    "priority_scope",
    "bind_priority",
    "priority_for_lane",
    "priority_for_task_type",
    "vllm_priority",
    "vllm_priority_extra_body",
    "priority_carrier",
    "PriorityModelGate",
    "get_priority_gate",
    "reset_priority_gates",
    "priority_slot",
    "priority_slot_sync",
]


class PriorityClass(Enum):
    """The ONE priority class shared across LLM / host worker / engine (ORCH-1.98).

    Lower :attr:`rank` = higher priority (admitted first, never yields). The four
    classes — and nothing else — are the single vocabulary every reserved lane
    keys off, so a request's class chosen at its entry point governs its worker
    slot, its engine read, and its LLM call identically.
    """

    #: A live Claude / end-user request (an MCP / REST interactive call). Highest.
    INTERACTIVE = "interactive"
    #: A skill / workflow execution (incl. cron-driven orchestration). High.
    ORCHESTRATION = "orchestration"
    #: Initial skill + MCP-server hydration — foundational bootstrap. The explicit
    #: exception: HIGH, NOT deprioritised (ingestion of the *toolset itself*).
    HYDRATION = "hydration"
    #: Documents, codebases, research-paper ingest, enrichment. Lowest — yields to
    #: all of the above, uses only spare capacity.
    BACKGROUND_INGESTION = "background_ingestion"

    @property
    def rank(self) -> int:
        """Numeric precedence (lower = higher priority)."""
        return _RANKS[self]

    @property
    def is_background(self) -> bool:
        """True only for :attr:`BACKGROUND_INGESTION` — the one class that yields."""
        return self is PriorityClass.BACKGROUND_INGESTION

    @property
    def is_interactive_floor(self) -> bool:
        """True for the classes that may claim a reserved floor slot.

        Everything that is NOT background ingestion: interactive, orchestration,
        AND hydration (the explicit non-deprioritised exception). Mirrors the host
        scheduler's ``INTERACTIVE_LANES`` reservation and the engine read lane.
        """
        return not self.is_background


# The explicit edict exception: initial skill + MCP hydration is FOUNDATIONAL — the
# toolset must load before anything can orchestrate — so these task types are HIGH
# (HYDRATION), NOT the BACKGROUND_INGESTION their ingestion lane would otherwise
# imply. The lane still governs worker-slot fairness; only the LLM priority differs.
HYDRATION_TASK_TYPES: frozenset[str] = frozenset({"skill_workflows"})

# Hydration shares orchestration's rank: high, never below background — the edict's
# "NOT deprioritised" exception expressed as a rank, not a special case.
_RANKS: dict[PriorityClass, int] = {
    PriorityClass.INTERACTIVE: 0,
    PriorityClass.ORCHESTRATION: 1,
    PriorityClass.HYDRATION: 1,
    PriorityClass.BACKGROUND_INGESTION: 3,
}

#: Carrier header so the priority rides outbound to the engine (EG-044 read lane)
#: alongside the trace/correlation headers — the cross-process leg of KG-2.293.
PRIORITY_HEADER = "x-resource-priority"


# --- the propagation carrier (a contextvar) ---------------------------------
# Default is None — an UNTAGGED call is treated as high (never throttled), so this
# is purely additive: only an explicitly BACKGROUND_INGESTION-scoped call yields.
_priority: contextvars.ContextVar[PriorityClass | None] = contextvars.ContextVar(
    "_resource_priority", default=None
)


def current_priority() -> PriorityClass | None:
    """The priority class in effect for the current task/request, if tagged."""
    return _priority.get()


def _effective(priority: PriorityClass | None) -> PriorityClass:
    """An untagged context resolves to ORCHESTRATION-level (high, never yields)."""
    return priority if priority is not None else PriorityClass.ORCHESTRATION


def set_priority(priority: PriorityClass | None) -> contextvars.Token:
    """Set the ambient priority; returns a token for :func:`contextvars.Token` reset."""
    return _priority.set(priority)


@contextlib.contextmanager
def priority_scope(priority: PriorityClass | None) -> Iterator[None]:
    """Bind a priority for the duration of a ``with`` block (the entry-point wrap).

    Entry points tag their work once and everything underneath — the LLM admission
    gate, an outbound engine call's header, a nested fan-out — inherits it:

    * an MCP/REST interactive call  → ``priority_scope(PriorityClass.INTERACTIVE)``
    * ``graph_orchestrate`` execute → ``priority_scope(PriorityClass.ORCHESTRATION)``
    * a codebase/document ingest    → ``priority_scope(PriorityClass.BACKGROUND_INGESTION)``
    * the skill/MCP hydration path  → ``priority_scope(PriorityClass.HYDRATION)``
    """
    token = _priority.set(priority)
    try:
        yield
    finally:
        _priority.reset(token)


def bind_priority(value: PriorityClass | str | None) -> None:
    """Restore a priority from a carrier value (header/task-metadata, KG-2.293)."""
    if value is None or isinstance(value, PriorityClass):
        _priority.set(value)
        return
    try:
        _priority.set(PriorityClass(str(value)))
    except ValueError:
        _priority.set(None)


def priority_carrier() -> dict[str, str]:
    """The active priority as a flat header carrier for an outbound call (or empty)."""
    p = _priority.get()
    return {PRIORITY_HEADER: p.value} if p is not None else {}


# --- lane / task-type → priority (reuse the host lane taxonomy) --------------
def priority_for_lane(lane: str | None) -> PriorityClass:
    """Map a host scheduler *lane* to its priority class (KG-2.293).

    Keyed off the SAME ``INTERACTIVE_LANES`` set the worker AdmissionPolicy uses
    (CONCEPT:AU-KG.compute.interactive-lane-floor), so the LLM admission and the worker reservation agree on
    what "interactive" means without a second source of truth.
    """
    from agent_utilities.knowledge_graph.core.task_lanes import INTERACTIVE_LANES

    if lane in INTERACTIVE_LANES:
        return PriorityClass.INTERACTIVE
    return PriorityClass.BACKGROUND_INGESTION


def priority_for_task_type(task_type: str | None) -> PriorityClass:
    """Map a queue *task type* to its priority class (KG-2.293).

    Resolves the type's lane via the canonical ``lane_for_task_type`` and defers to
    :func:`priority_for_lane`, so the LLM gate and the worker pool classify the
    exact same task identically — EXCEPT the foundational hydration task types
    (:data:`HYDRATION_TASK_TYPES`), which are HIGH regardless of their lane.
    """
    from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type

    if task_type in HYDRATION_TASK_TYPES:
        return PriorityClass.HYDRATION
    return priority_for_lane(lane_for_task_type(task_type))


# --- the server-side hint (vLLM request `priority` field) --------------------
def vllm_priority(priority: PriorityClass | None = None) -> int:
    """Translate a class to a vLLM scheduler ``priority`` value (lower = sooner).

    vLLM's priority scheduling admits the *lowest* priority value first, matching
    our rank convention, so we hand it the rank directly. The server uses it only
    when started with ``--scheduling-policy priority``; otherwise it is ignored —
    the client-side gate (this module) is the always-on enforcement.
    """
    return _effective(priority if priority is not None else current_priority()).rank


def vllm_priority_extra_body(
    priority: PriorityClass | None = None,
) -> dict[str, int]:
    """``extra_body`` fragment carrying the vLLM ``priority`` field for a request."""
    return {"priority": vllm_priority(priority)}


# --- the priority-aware shared-LLM admission gate (ORCH-1.99) -----------------
def _auto_reserve(capacity: int) -> int:
    """Reserved high-priority headroom for a gate of ``capacity`` permits.

    Auto-sized (Native-by-default): ``round(capacity * fraction)`` floored at 1 for
    any real pool, clamped to ``capacity - 1`` so background always makes *some*
    progress. A degenerate single-permit gate reserves 0 (that one permit serves
    everything) — exactly mirroring the worker scheduler's ``interactive_floor``.
    Override with ``KG_LLM_PRIORITY_RESERVE``.
    """
    if capacity <= 1:
        return 0
    from agent_utilities.core._env import setting

    raw = setting("KG_LLM_PRIORITY_RESERVE", None)
    if raw not in (None, ""):
        try:
            return max(0, min(int(raw), capacity - 1))
        except (TypeError, ValueError):
            pass
    try:
        frac = float(setting("KG_LLM_PRIORITY_RESERVE_FRACTION", 0.34))
    except (TypeError, ValueError):
        frac = 0.34
    return max(1, min(round(capacity * frac), capacity - 1))


class PriorityModelGate:
    """A capacity gate that admits high-priority LLM calls ahead of background.

    Enforces, for one model endpoint:

    1. **Hard capacity** — at most ``capacity`` calls in flight (subsumes the plain
       per-model semaphore; same width).
    2. **Reserved headroom** — background ingestion may occupy at most
       ``capacity - reserve`` permits, ALWAYS. So ``reserve`` permits are kept free
       for an interactive/orchestration/hydration call to land *immediately*, even
       under a saturating background fan-out (the non-blocking guarantee).
    3. **Active-contention yield** — while any higher-priority call is *waiting*
       (a burst exceeding the reserve), background admission is refused outright so
       the high-priority backlog drains first. Background is only ever throttled
       *while interactive is actively contending*; otherwise it scales up into the
       reserved-minus headroom (dynamic scaling, never starved to zero).

    Symmetric async (:meth:`acquire`/:meth:`release`) and sync
    (:meth:`acquire_sync`/:meth:`release_sync`) faces share one logic core and one
    lock, so an async orchestration call and a sync enrichment call contend on the
    same gate. Pure in-process bookkeeping — cheap, deadlock-free, unit-testable.
    """

    def __init__(self, capacity: int, reserve: int | None = None):
        self.capacity = max(1, int(capacity))
        self.reserve = (
            _auto_reserve(self.capacity)
            if reserve is None
            else max(0, min(int(reserve), self.capacity - 1))
        )
        self._active = 0
        self._high_waiters = 0
        self._mutex = threading.Lock()
        self._sync_cond = threading.Condition(self._mutex)
        self._async_cond = asyncio.Condition()

    # -- the testable decision core (caller holds nothing; pure arithmetic) ---
    def _can_admit(self, is_high: bool) -> bool:
        if self._active >= self.capacity:
            return False
        if is_high:
            return True
        # Background: yield entirely while a higher-priority call is contending …
        if self._high_waiters > 0:
            return False
        # … else use spare capacity up to the reserved-minus headroom.
        return self._active < (self.capacity - self.reserve)

    # -- async face ----------------------------------------------------------
    async def acquire(self, priority: PriorityClass) -> None:
        is_high = priority.is_interactive_floor
        async with self._async_cond:
            if is_high:
                self._high_waiters += 1
            try:
                while True:
                    with self._mutex:
                        if self._can_admit(is_high):
                            self._active += 1
                            break
                    await self._async_cond.wait()
            finally:
                if is_high:
                    self._high_waiters -= 1

    async def release(self) -> None:
        with self._mutex:
            self._active = max(0, self._active - 1)
        async with self._async_cond:
            self._async_cond.notify_all()
        with self._sync_cond:
            self._sync_cond.notify_all()

    # -- sync face -----------------------------------------------------------
    def acquire_sync(self, priority: PriorityClass) -> None:
        is_high = priority.is_interactive_floor
        with self._sync_cond:
            if is_high:
                self._high_waiters += 1
            try:
                while not self._can_admit(is_high):
                    self._sync_cond.wait()
                self._active += 1
            finally:
                if is_high:
                    self._high_waiters -= 1

    def release_sync(self) -> None:
        with self._sync_cond:
            self._active = max(0, self._active - 1)
            self._sync_cond.notify_all()

    # -- introspection (tests / observability) -------------------------------
    @property
    def active(self) -> int:
        with self._mutex:
            return self._active


# Cached per (model_key, capacity) so a capacity change yields a fresh gate. Unlike
# an asyncio.Semaphore the gate is loop-agnostic (its async Condition lazily binds),
# so one gate spans the async and sync faces for the same model.
_gate_lock = threading.Lock()
_gates: dict[tuple[str, int], PriorityModelGate] = {}


def _model_key(model: str | None) -> str:
    return (model or "__default__").strip().lower() or "__default__"


def get_priority_gate(
    model: str | None = None, capacity: int | None = None
) -> PriorityModelGate:
    """The shared priority gate for ``model`` sized to ``capacity`` (ORCH-1.99).

    Capacity defaults to the model's resolved parallel-call capacity (the same
    value the plain per-model semaphore uses), so the gate is a drop-in priority
    upgrade of that semaphore at the identical width.
    """
    if capacity is None:
        from agent_utilities.core.model_concurrency import resolve_capacity

        capacity = resolve_capacity(model)
    cap = max(1, int(capacity))
    k = (_model_key(model), cap)
    with _gate_lock:
        gate = _gates.get(k)
        if gate is None:
            gate = PriorityModelGate(cap)
            _gates[k] = gate
        return gate


def reset_priority_gates() -> None:
    """Drop all cached gates (test isolation / config reload)."""
    with _gate_lock:
        _gates.clear()


@contextlib.asynccontextmanager
async def priority_slot(
    model: str | None = None,
    *,
    capacity: int | None = None,
    priority: PriorityClass | None = None,
) -> AsyncIterator[None]:
    """Async ``with`` gate around one LLM call, classed by the ambient priority."""
    prio = _effective(priority if priority is not None else current_priority())
    gate = get_priority_gate(model, capacity)
    await gate.acquire(prio)
    try:
        yield
    finally:
        await gate.release()


@contextlib.contextmanager
def priority_slot_sync(
    model: str | None = None,
    *,
    capacity: int | None = None,
    priority: PriorityClass | None = None,
) -> Iterator[None]:
    """Sync ``with`` gate around one LLM call, classed by the ambient priority."""
    prio = _effective(priority if priority is not None else current_priority())
    gate = get_priority_gate(model, capacity)
    gate.acquire_sync(prio)
    try:
        yield
    finally:
        gate.release_sync()
