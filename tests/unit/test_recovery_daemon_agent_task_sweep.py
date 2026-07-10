#!/usr/bin/python
from __future__ import annotations

"""RecoveryDaemon's :AgentTask dependency-firing wiring (C3/Phase 3a).

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3a)

``RecoveryDaemon.stabilize()`` reuses the SAME
``fleet_reconciler.fire_ready_agent_tasks`` sweep the leader-only
``FleetReconciler.reconcile()`` tick calls (see
``tests/unit/test_agent_task_dag.py`` for the firing logic itself) — no
dependency-checking logic is duplicated between the two tick callers. This
covers only the wiring: called when the scheduler has an engine, skipped
(never raises) when it does not.

@pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")
"""

from types import SimpleNamespace

import pytest

from agent_utilities.orchestration.recovery_daemon import RecoveryDaemon

pytestmark = pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")


def _scheduler(engine=None):
    return SimpleNamespace(_processes={}, engine=engine)


@pytest.mark.asyncio
async def test_stabilize_sweeps_agent_task_dependencies_when_engine_present(
    monkeypatch,
):
    calls: list[object] = []
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_reconciler.fire_ready_agent_tasks",
        lambda engine: calls.append(engine) or [],
    )
    engine = object()
    daemon = RecoveryDaemon(_scheduler(engine=engine))
    recovered = await daemon.stabilize()
    assert recovered == 0  # no processes to recover; the sweep is separate
    assert calls == [engine]


@pytest.mark.asyncio
async def test_stabilize_skips_sweep_without_an_engine(monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_reconciler.fire_ready_agent_tasks",
        lambda engine: calls.append(engine) or [],
    )
    daemon = RecoveryDaemon(_scheduler(engine=None))
    await daemon.stabilize()
    assert calls == []


@pytest.mark.asyncio
async def test_stabilize_tolerates_sweep_failure(monkeypatch):
    """A broken sweep must not blow up the recovery tick (never load-bearing)."""

    def _boom(engine):
        raise RuntimeError("engine unreachable")

    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_reconciler.fire_ready_agent_tasks", _boom
    )
    daemon = RecoveryDaemon(_scheduler(engine=object()))
    recovered = await daemon.stabilize()
    assert recovered == 0
