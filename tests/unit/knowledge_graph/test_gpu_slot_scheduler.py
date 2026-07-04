"""Unit tests for the single-GPU-slot scheduler (CONCEPT:AU-KG.compute.code-intelligence-tools).

Exercises the preempt / backfill / held / restart-reconcile state machine with a
cooperative runner that checkpoints between items and resumes from where it left
off — no GPU required.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler import (
    GpuSlotScheduler,
    InMemoryCheckpointStore,
    Job,
    JobState,
)


def _make_runner(
    total: int, *, gate: asyncio.Event | None = None, log: list | None = None
):
    """A cooperative runner that processes ``total`` items, resuming from the
    job's checkpoint, yielding the slot when asked to pause."""

    async def _run(job: Job, sched: GpuSlotScheduler) -> None:
        done = int(job.checkpoint.get("done", 0))
        while done < total:
            if gate is not None:
                await gate.wait()
            if sched.should_pause(job.job_id):
                await sched.checkpoint(job.job_id, {"done": done})
                return  # yield the slot cooperatively
            await asyncio.sleep(0)  # simulate a unit of work
            done += 1
            if log is not None:
                log.append((job.job_id, done))
            await sched.checkpoint(job.job_id, {"done": done})
        job.state = JobState.DONE

    return _run


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


@pytest.mark.asyncio
async def test_single_job_runs_to_done() -> None:
    sched = GpuSlotScheduler()
    await sched.start(_make_runner(3))
    await sched.submit("j1")
    await _wait_until(
        lambda: sched.get("j1") and sched.get("j1").state == JobState.DONE
    )
    assert sched.get("j1").checkpoint["done"] == 3
    await sched.stop()


@pytest.mark.asyncio
async def test_preempt_backfill_resume_from_checkpoint() -> None:
    log: list = []
    gate = asyncio.Event()
    gate.set()
    sched = GpuSlotScheduler(preempt_foreground=True)
    await sched.start(_make_runner(6, gate=gate, log=log))

    await sched.submit("A")
    await _wait_until(lambda: any(j == "A" for j, _ in log))  # A is running
    # Preempt A with a fresh foreground job B.
    await sched.submit("B")
    # B runs to completion; A was checkpointed mid-flight then backfilled.
    await _wait_until(
        lambda: sched.get("B") and sched.get("B").state == JobState.DONE, timeout=3.0
    )
    await _wait_until(
        lambda: sched.get("A") and sched.get("A").state == JobState.DONE, timeout=3.0
    )
    # A resumed from its checkpoint: exactly 6 increments, never restarted from 0.
    a_steps = [d for j, d in log if j == "A"]
    assert a_steps == sorted(set(a_steps))  # strictly increasing, no repeats
    assert a_steps[-1] == 6
    await sched.stop()


@pytest.mark.asyncio
async def test_training_job_preempted_by_foreground_inference() -> None:
    """A long ``kind="training"`` job yields the slot to interactive inference and
    resumes from its checkpoint — the background tier (CONCEPT:AU-AHE.trainer.join-inference) needs no
    ``preempt_foreground`` flag."""
    log: list = []
    gate = asyncio.Event()
    gate.set()
    sched = GpuSlotScheduler()  # preempt_foreground stays False
    await sched.start(_make_runner(6, gate=gate, log=log))

    await sched.submit("train", kind="training")
    await _wait_until(lambda: any(j == "train" for j, _ in log))
    # Foreground inference (default kind) preempts the background training run.
    await sched.submit("infer")
    await _wait_until(
        lambda: sched.get("infer") and sched.get("infer").state == JobState.DONE,
        timeout=3.0,
    )
    await _wait_until(
        lambda: sched.get("train") and sched.get("train").state == JobState.DONE,
        timeout=3.0,
    )
    train_steps = [d for j, d in log if j == "train"]
    assert train_steps == sorted(set(train_steps))  # resumed, never restarted
    assert train_steps[-1] == 6
    await sched.stop()


@pytest.mark.asyncio
async def test_held_job_is_not_backfilled() -> None:
    sched = GpuSlotScheduler()
    await sched.start(_make_runner(3))
    # Submit a job, immediately hold it before it can finish is racy; instead
    # submit, let it finish, then verify a manually-held paused job stays put.
    j = await sched.submit("h1")
    j.state = JobState.PAUSED
    sched._jobs["h1"].state = JobState.PAUSED  # simulate a preempted job
    await sched.hold("h1")
    assert sched.get("h1").state == JobState.HELD
    # Give the worker a chance: held job must NOT be auto-backfilled to running.
    await asyncio.sleep(0.1)
    assert sched.get("h1").state == JobState.HELD
    await sched.stop()


@pytest.mark.asyncio
async def test_restart_reconcile_marks_midflight_paused() -> None:
    store = InMemoryCheckpointStore()
    # Simulate a prior process that died with a RUNNING job persisted.
    store.save(
        Job(job_id="r1", state=JobState.RUNNING, checkpoint={"done": 2}, submitted=1.0)
    )
    sched = GpuSlotScheduler(store=store, auto_backfill=False)
    await sched.start(_make_runner(5))
    reconciled = sched.get("r1")
    assert reconciled is not None
    assert reconciled.state == JobState.PAUSED
    assert reconciled.preempted is True
    assert reconciled.checkpoint["done"] == 2  # checkpoint survived
    await sched.stop()
