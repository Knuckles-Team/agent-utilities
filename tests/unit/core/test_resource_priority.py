"""Tests for the unified resource-priority class + shared-LLM admission gate.

CONCEPT:ORCH-1.98 / ORCH-1.99 / KG-2.293. Proves the edict's four guarantees:

1. an INTERACTIVE/ORCHESTRATION call is admitted AHEAD of a saturating
   BACKGROUND_INGESTION enrichment fan-out on the shared model;
2. background ingestion uses the SPARE LLM capacity when nothing higher contends
   (dynamic scaling — not starved to zero);
3. HYDRATION is NOT deprioritised (admitted like interactive under saturation);
4. the priority propagates from an entry point to the LLM gate / worker lane /
   engine carrier, keyed off ONE shared class.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent_utilities.core import resource_priority as rp
from agent_utilities.core.resource_priority import (
    PriorityClass,
    PriorityModelGate,
    current_priority,
    priority_for_lane,
    priority_for_task_type,
    priority_scope,
)


# --- the priority class: single source of truth ------------------------------
def test_priority_class_ranks_interactive_highest_background_lowest():
    assert PriorityClass.INTERACTIVE.rank < PriorityClass.ORCHESTRATION.rank
    # Hydration is the explicit exception: high, never below background.
    assert PriorityClass.HYDRATION.rank == PriorityClass.ORCHESTRATION.rank
    assert PriorityClass.HYDRATION.rank < PriorityClass.BACKGROUND_INGESTION.rank
    assert PriorityClass.BACKGROUND_INGESTION.is_background
    for p in (
        PriorityClass.INTERACTIVE,
        PriorityClass.ORCHESTRATION,
        PriorityClass.HYDRATION,
    ):
        assert p.is_interactive_floor and not p.is_background


# --- (4) propagation: entry point → class, keyed off the SAME lane taxonomy ---
def test_lane_and_task_type_map_to_shared_class():
    # The host worker AdmissionPolicy reserves INTERACTIVE_LANES (KG-2.289); the LLM
    # gate keys off the SAME set, so both agree on what "interactive" means.
    from agent_utilities.knowledge_graph.core.task_lanes import INTERACTIVE_LANES

    for lane in INTERACTIVE_LANES:
        assert priority_for_lane(lane) is PriorityClass.INTERACTIVE
    assert priority_for_task_type("conversation") is PriorityClass.INTERACTIVE
    assert priority_for_task_type("codebase") is PriorityClass.BACKGROUND_INGESTION
    assert priority_for_task_type("document") is PriorityClass.BACKGROUND_INGESTION
    # (3) Foundational hydration overrides its background lane → NOT deprioritised.
    assert priority_for_task_type("skill_workflows") is PriorityClass.HYDRATION


def test_priority_scope_sets_and_restores_contextvar():
    assert current_priority() is None
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        assert current_priority() is PriorityClass.BACKGROUND_INGESTION
        with priority_scope(PriorityClass.INTERACTIVE):
            assert current_priority() is PriorityClass.INTERACTIVE
        assert current_priority() is PriorityClass.BACKGROUND_INGESTION
    assert current_priority() is None


def test_priority_propagates_through_correlation_carrier():
    """KG-2.293: the class rides the carrier so a child/engine inherits it."""
    from agent_utilities.observability import correlation

    with priority_scope(PriorityClass.INTERACTIVE):
        carrier = correlation.current_carrier()
    assert carrier.get(rp.PRIORITY_HEADER) == PriorityClass.INTERACTIVE.value
    # bind_carrier restores it in the (simulated child) context.
    assert current_priority() is None
    with correlation.bind_carrier(carrier):
        assert current_priority() is PriorityClass.INTERACTIVE
    assert current_priority() is None


# --- the gate decision core (deterministic, no threads) ----------------------
def test_gate_reserves_headroom_for_high_priority():
    gate = PriorityModelGate(capacity=4, reserve=1)
    # background may fill only capacity - reserve = 3.
    gate._active = 3  # noqa: SLF001 — exercising the decision core directly
    assert gate._can_admit(is_high=False) is False  # noqa: SLF001
    # but a high-priority call can still take the reserved 4th permit.
    assert gate._can_admit(is_high=True) is True  # noqa: SLF001
    gate._active = 4  # noqa: SLF001 — full → even high waits for a release
    assert gate._can_admit(is_high=True) is False  # noqa: SLF001


def test_gate_background_yields_while_high_waits():
    gate = PriorityModelGate(capacity=4, reserve=1)
    gate._active = 0  # noqa: SLF001
    gate._high_waiters = 1  # noqa: SLF001 — a high-priority call is contending
    # Even with permits free, background defers entirely so the high backlog drains.
    assert gate._can_admit(is_high=False) is False  # noqa: SLF001
    gate._high_waiters = 0  # noqa: SLF001
    assert gate._can_admit(is_high=False) is True  # noqa: SLF001


def test_single_permit_gate_serves_everything():
    # Degenerate pool: reserve 0, the one permit serves all classes.
    gate = PriorityModelGate(capacity=1)
    assert gate.reserve == 0
    assert gate._can_admit(is_high=False) is True  # noqa: SLF001


# --- (1) interactive jumps ahead of a SATURATING background fan-out -----------
@pytest.mark.asyncio
async def test_interactive_admitted_ahead_of_saturating_background():
    rp.reset_priority_gates()
    cap = 2
    gate = rp.get_priority_gate("qwen-test-1", capacity=cap)
    assert gate.reserve >= 1

    release = asyncio.Event()
    bg_running = 0
    bg_peak = 0
    lock = asyncio.Lock()

    async def background():
        nonlocal bg_running, bg_peak
        await gate.acquire(PriorityClass.BACKGROUND_INGESTION)
        async with lock:
            bg_running += 1
            bg_peak = max(bg_peak, bg_running)
        try:
            await release.wait()  # hold the permit (saturate)
        finally:
            async with lock:
                bg_running -= 1
            await gate.release()

    # Fire many background tasks that try to saturate the model.
    bg_tasks = [asyncio.create_task(background()) for _ in range(8)]
    await asyncio.sleep(0.05)
    # Background can NEVER hold more than cap - reserve permits, leaving headroom.
    assert bg_peak <= cap - gate.reserve

    # An interactive call lands immediately into the reserved headroom, even though
    # background is actively saturating the model.
    admitted = asyncio.Event()

    async def interactive():
        await gate.acquire(PriorityClass.INTERACTIVE)
        admitted.set()
        await gate.release()

    itask = asyncio.create_task(interactive())
    await asyncio.wait_for(admitted.wait(), timeout=1.0)
    assert admitted.is_set()

    release.set()
    await asyncio.gather(itask, *bg_tasks)
    rp.reset_priority_gates()


# --- (2) background uses SPARE capacity when nothing higher contends ----------
@pytest.mark.asyncio
async def test_background_uses_spare_capacity_when_idle():
    rp.reset_priority_gates()
    cap = 4
    gate = rp.get_priority_gate("qwen-test-2", capacity=cap)
    expected = cap - gate.reserve
    assert expected >= 2

    release = asyncio.Event()
    running = 0
    peak = 0
    lock = asyncio.Lock()

    async def background():
        nonlocal running, peak
        await gate.acquire(PriorityClass.BACKGROUND_INGESTION)
        async with lock:
            running += 1
            peak = max(peak, running)
        try:
            await release.wait()
        finally:
            async with lock:
                running -= 1
            await gate.release()

    tasks = [asyncio.create_task(background()) for _ in range(8)]
    await asyncio.sleep(0.05)
    # Dynamic scaling: background fans out to the full reserved-minus headroom,
    # NOT throttled to one — it is using the spare capacity.
    assert peak == expected
    release.set()
    await asyncio.gather(*tasks)
    rp.reset_priority_gates()


# --- (3) HYDRATION is NOT deprioritised --------------------------------------
@pytest.mark.asyncio
async def test_hydration_not_deprioritised_under_saturation():
    rp.reset_priority_gates()
    cap = 2
    gate = rp.get_priority_gate("qwen-test-3", capacity=cap)
    release = asyncio.Event()

    async def background():
        await gate.acquire(PriorityClass.BACKGROUND_INGESTION)
        try:
            await release.wait()
        finally:
            await gate.release()

    bg_tasks = [asyncio.create_task(background()) for _ in range(8)]
    await asyncio.sleep(0.05)

    admitted = asyncio.Event()

    async def hydration():
        # Treated like interactive — lands in the reserved headroom immediately.
        await gate.acquire(PriorityClass.HYDRATION)
        admitted.set()
        await gate.release()

    htask = asyncio.create_task(hydration())
    await asyncio.wait_for(admitted.wait(), timeout=1.0)
    assert admitted.is_set()
    release.set()
    await asyncio.gather(htask, *bg_tasks)
    rp.reset_priority_gates()


# --- the sync face contends on the SAME gate (enrichment is often sync) -------
def test_sync_gate_reserves_headroom_for_high():
    gate = PriorityModelGate(capacity=2, reserve=1)
    held = threading.Event()
    release = threading.Event()

    def background():
        gate.acquire_sync(PriorityClass.BACKGROUND_INGESTION)
        held.set()
        release.wait(2.0)
        gate.release_sync()

    t = threading.Thread(target=background, daemon=True)
    t.start()
    assert held.wait(1.0)
    # background holds 1 (cap-reserve); a high call still has the reserved permit.
    start = time.monotonic()
    gate.acquire_sync(PriorityClass.INTERACTIVE)
    assert time.monotonic() - start < 1.0  # admitted immediately, did not block
    gate.release_sync()
    release.set()
    t.join(2.0)
