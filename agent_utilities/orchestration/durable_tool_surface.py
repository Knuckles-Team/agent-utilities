#!/usr/bin/python
from __future__ import annotations

"""The unified durable-execution tool surface (DE2,
CONCEPT:AU-ORCH.execution.durable-routing-trait).

`durable-execution-native.md` Part 3: one small, documented mental model — the
literal analog of a restate SDK author's ``ctx.*`` — regardless of which of the
four backends actually serves a call. The routing decision itself is
`eg-durable`'s `CallShape`/`WorkShape`/`DurableBackendKind` contract (landed at
DE0, `crates/eg-durable/src/route.rs`); THIS module is its one AU-side caller
for the two backends genuinely reachable from Python today:

* ``durable_run`` — ``CallShape::Run`` -> ``WorkShape::AgentLoopContinuation``
  -> ``DurableBackendKind::PythonDurableRun`` -> :class:`DurableRun`. Real.
* ``durable_sleep`` — ``CallShape::Sleep`` -> ``WorkShape::AsyncCheckpointedWork``
  -> (routed at DE0 to ``DurableBackendKind::Jobs``, i.e. ``eg-jobs``'s
  cron/interval triggers). ``eg-jobs`` has no Python-callable submit/query
  surface today either (see below), so this module backs ``durable_sleep``
  with :meth:`DurableRun.sleep_until` instead — a **documented, honest
  substitution**, not DE0's own routing table: it delivers the same
  externally-observed contract (a durable deadline bound to one in-flight
  execution) on the one backend Python can actually reach, poll-based rather
  than ``eg-jobs``'s tick-driven triggers. Callers relying on ``eg-jobs``'s own
  cron/interval semantics should keep using that backend directly.

``durable_call`` (``CallShape::Call`` -> ``WorkShape::CrossStoreAtomicStep`` ->
``DurableBackendKind::MutationStoreSaga``) and ``durable_state_get``/
``durable_state_set`` (``CallShape::StateGet``/``StateSet`` ->
``WorkShape::KeyedSingleWriterState`` -> ``DurableBackendKind::Statechart``)
are **NOT backed** — raising :class:`DurableCallNotBacked` rather than silently
degrading to the wrong backend or reporting a false-positive success. Both
target backends (`eg-mutation-store`'s saga coordinator, `eg-statechart`'s
keyed OCC state) are real and durable, but neither has a Python-reachable
wire-protocol surface today: `protocols/epistemic_operations/_generated.py`
carries an ``AnalyticsJob`` DTO but no live query/mutate method for it, and no
DTO at all yet for a statechart instance or a saga (`Method::Statechart{...}`,
`Method::Saga{...}` do not exist). Closing this is new engine wire-protocol
work, out of scope for this lane — see `docs/durable-execution.md`'s parity
sweep for the tracked gap. Failing loud here (never a bare no-op that hides
the gap) is the same "fail closed" discipline every other gate in this repo
uses for a degraded read.
"""

import time
from typing import Any, NoReturn, TypedDict

from agent_utilities.orchestration.durable_execution import DurableRun

__all__ = [
    "DurableCallNotBacked",
    "DurableRunHandle",
    "DurableCheckpointResult",
    "DurableSleepResult",
    "durable_run",
    "durable_checkpoint",
    "durable_sleep",
    "durable_state_get",
    "durable_state_set",
    "durable_call",
    "durable_status",
]


class DurableCallNotBacked(RuntimeError):
    """A DE0 ``CallShape`` whose routed backend has no Python-reachable surface
    yet. Never silently degrades — see this module's docstring."""


class DurableRunHandle(TypedDict):
    """Return shape of :func:`durable_run` — a handle, never the run itself."""

    session_id: str
    run_id: str
    resumed: bool
    definition_version: str | None


class DurableCheckpointResult(TypedDict):
    """Return shape of :func:`durable_checkpoint`."""

    session_id: str
    run_id: str
    step_label: str
    result: Any


class DurableSleepResult(TypedDict):
    """Return shape of :func:`durable_sleep`."""

    session_id: str
    run_id: str
    name: str
    wake_at_ms: int
    fired: bool
    now_ms: int


def durable_run(
    engine: Any | None,
    session_id: str,
    *,
    definition_version: str | None = None,
) -> DurableRunHandle:
    """Start-or-resume a durable run; return a handle (never the run itself).

    Mirrors restate's ``ctx``-scoped invocation identity: the caller gets back
    ``run_id``/``resumed``/``definition_version`` and then drives the run's
    actual steps in code via :meth:`DurableRun.step`/``auto_step`` — exactly
    as a restate handler calls ``ctx.run(...)`` internally rather than over
    the wire per step. This tool is how an external caller (an agent, an
    operator) OBTAINS or CHECKS a run handle, not how steps execute.
    """
    run = DurableRun(session_id, engine=engine, definition_version=definition_version)
    return DurableRunHandle(
        session_id=session_id,
        run_id=run.run_id,
        resumed=run.resumed,
        definition_version=run.definition_version,
    )


def durable_checkpoint(
    engine: Any | None,
    session_id: str,
    step_label: str,
    result: Any = None,
    *,
    definition_version: str | None = None,
) -> DurableCheckpointResult:
    """Durably record that ``step_label`` happened, auto-sequenced (DE3).

    For a caller whose step's WORK already happened outside this call (an
    external event, a value already computed) and only the durable, resumable
    RECORD of it is missing — the pure-marker analog of restate's ``ctx.run``.
    Uses :meth:`DurableRun.auto_step` rather than :meth:`DurableRun.step`
    directly, so the caller never hand-picks a unique step name.

    Boundary caveat, honestly stated (see
    ``test_durable_checkpoint_is_idempotent_across_stateless_calls``): THIS
    function reconstructs a fresh :class:`DurableRun` every call — the
    MCP/stateless-tool-call shape — so ``auto_step``'s in-memory per-run
    sequence counter resets each time. Two calls with the SAME
    ``session_id``+``step_label`` therefore resolve to the SAME auto-sequenced
    step and the second call REPLAYS the first-recorded ``result`` rather than
    recording a new one — exactly `DurableRun.step`'s own idempotency-key
    contract. Auto-sequencing's actual value ("no hand-picked name needed for
    N calls") is realized when :meth:`DurableRun.auto_step` is used directly
    on ONE long-lived ``DurableRun`` inside a loop (in-process, restate's
    ``ctx.run`` shape) — not by calling this stateless wrapper repeatedly with
    a repeated label. A caller that needs N distinct durable records over N
    stateless calls should vary ``step_label`` itself (e.g. include an
    external event id).
    """
    run = DurableRun(session_id, engine=engine, definition_version=definition_version)
    recorded = run.auto_step(name=step_label)(lambda: result)()
    return DurableCheckpointResult(
        session_id=session_id,
        run_id=run.run_id,
        step_label=step_label,
        result=recorded,
    )


def durable_sleep(
    engine: Any | None,
    session_id: str,
    name: str,
    wake_at_ms: int,
    *,
    definition_version: str | None = None,
) -> DurableSleepResult:
    """Durable timer bound to one run's one named step — see module docstring
    for why this backs ``CallShape::Sleep`` with :class:`DurableRun` rather
    than ``eg-jobs``."""
    run = DurableRun(session_id, engine=engine, definition_version=definition_version)
    fired = run.sleep_until(name, int(wake_at_ms))
    return DurableSleepResult(
        session_id=session_id,
        run_id=run.run_id,
        name=name,
        wake_at_ms=int(wake_at_ms),
        fired=fired,
        now_ms=int(time.time() * 1000),
    )


def durable_state_get(engine: Any | None, session_id: str, key: str) -> NoReturn:
    raise DurableCallNotBacked(
        "durable_state_get routes to eg-statechart (CallShape::StateGet -> "
        "WorkShape::KeyedSingleWriterState -> DurableBackendKind::Statechart, "
        "eg-durable route.rs), which has no Python-reachable wire-protocol "
        "surface today. See docs/durable-execution.md's parity sweep."
    )


def durable_state_set(
    engine: Any | None, session_id: str, key: str, value: Any
) -> NoReturn:
    raise DurableCallNotBacked(
        "durable_state_set routes to eg-statechart (CallShape::StateSet -> "
        "WorkShape::KeyedSingleWriterState -> DurableBackendKind::Statechart, "
        "eg-durable route.rs), which has no Python-reachable wire-protocol "
        "surface today. See docs/durable-execution.md's parity sweep."
    )


def durable_call(
    engine: Any | None,
    session_id: str,
    name: str,
    *,
    idempotency_key: str | None = None,
) -> NoReturn:
    raise DurableCallNotBacked(
        "durable_call routes to eg-mutation-store's saga coordinator "
        "(CallShape::Call -> WorkShape::CrossStoreAtomicStep -> "
        "DurableBackendKind::MutationStoreSaga, eg-durable route.rs), which "
        "has no Python-reachable wire-protocol surface today (no "
        "Method::Saga{...} DTO in protocols/epistemic_operations). See "
        "docs/durable-execution.md's parity sweep."
    )


def durable_status(
    engine: Any,
    *,
    kinds: tuple[str, ...] | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """ "What is durably in flight, waiting on what" across the whole
    `:DurableExecutionUnit` family (DE1) -- currently populated for
    `:DurableRun` only; see `durable_execution_kg.DURABLE_EXECUTION_UNIT_LABELS`
    for the other three reserved labels awaiting a writer.
    """
    from agent_utilities.knowledge_graph.durable_execution_kg import (
        DURABLE_EXECUTION_UNIT_LABELS,
        list_durable_execution_units,
    )

    return list_durable_execution_units(
        engine,
        kinds=kinds or DURABLE_EXECUTION_UNIT_LABELS,
        status=status,
        limit=limit,
    )
