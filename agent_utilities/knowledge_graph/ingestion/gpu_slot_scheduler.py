"""Single-GPU-slot inference scheduler (CONCEPT:AU-KG.compute.code-intelligence-tools).

The durable Kafka ingest queue (KG-2.55–2.57) fans work out *across hosts*; this
scheduler governs the **one** scarce resource a host actually contends on — its
single GPU inference slot (the GB10/GR1080 vLLM endpoint). Only one extraction
job may hold the slot at a time, so a fresh foreground submission must be able to
**preempt** the running job, which is checkpointed and **auto-resumes from where
it left off** when the slot frees.

Assimilated from the ``knowledge-graph-extractor`` ``jobs.py`` single-slot
scheduler; layered *above* our durable queue (it does not replace it) and made
backend-agnostic via a ``CheckpointStore`` so checkpoints can live graph-natively
(dogfooding the GraphBackend) rather than in loose files.

State machine::

    queued  → waiting for the slot (foreground, FIFO)
    running → on the slot now
    pausing → running job flagged to yield (transient)
    paused  → preemptible idle; auto-backfill may resume it
    held    → user-paused; sticky, NOT auto-backfilled (explicit resume only)
    done    → finished
    failed  → errored

Cooperative: the job runner checks ``scheduler.should_pause(job_id)`` at safe
boundaries (between files/chunks/rounds), persists its checkpoint, and returns;
the scheduler then leaves it ``paused`` (system preempt) or ``held`` (user) and
backfills the next job.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    HELD = "held"
    DONE = "done"
    FAILED = "failed"


# A runner gets (job, scheduler) and drives the work, checking should_pause().
JobRunner = Callable[["Job", "GpuSlotScheduler"], Awaitable[None]]


@dataclass
class Job:
    """A unit of GPU work with a resumable checkpoint."""

    job_id: str
    kind: str = "extract"
    state: JobState = JobState.QUEUED
    submitted: float = 0.0  # monotonic-ish ordering token (set by submitter)
    preempted: bool = False  # restarted/preempted → higher backfill priority
    user_held: bool = False  # sticky user pause
    params: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": str(self.state),
            "preempted": self.preempted,
            "user_held": self.user_held,
            "checkpoint": self.checkpoint,
            "error": self.error,
        }


@runtime_checkable
class CheckpointStore(Protocol):
    """Where job metadata + checkpoints survive a restart."""

    def save(self, job: Job) -> None: ...

    def load_all(self) -> list[Job]: ...


class InMemoryCheckpointStore:
    """Default non-durable store (tests / single-process dev). Production passes
    a graph-native store so checkpoints survive a redeploy."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def save(self, job: Job) -> None:
        # store a copy so external mutation doesn't alias persisted state
        self._jobs[job.job_id] = Job(**{**job.__dict__})

    def load_all(self) -> list[Job]:
        return [Job(**{**j.__dict__}) for j in self._jobs.values()]


class GpuSlotScheduler:
    """Runs at most one job at a time, with preemption + auto-backfill.

    The scheduler owns no GPU client — it sequences *access* to the slot. Start
    it with ``await scheduler.start(runner)``; submit jobs with ``submit``.
    """

    #: Kinds that run in the preemptible **background tier** — they always yield the
    #: slot to a fresh foreground submission and auto-backfill when it frees.
    #: Training jobs (CONCEPT:AU-AHE.trainer.join-inference) join inference ``"auto"`` here, so an
    #: interactive request preempts a long training run, which checkpoints and
    #: resumes from where it left off.
    BACKGROUND_KINDS: tuple[str, ...] = ("auto", "training")

    def __init__(
        self,
        store: CheckpointStore | None = None,
        *,
        auto_backfill: bool = True,
        preempt_foreground: bool = False,
    ) -> None:
        self._store: CheckpointStore = store or InMemoryCheckpointStore()
        self._auto_backfill = auto_backfill
        # If False, a new foreground job only preempts *auto* jobs, never another
        # foreground one (it waits its turn). Matches upstream PREEMPT_FOREGROUND.
        self._preempt_foreground = preempt_foreground
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []  # FIFO of foreground QUEUED ids
        self._current: str | None = None
        self._pause_flags: set[str] = set()
        self._cond = asyncio.Condition()
        self._runner: JobRunner | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._seq = 0  # deterministic submission ordering (no wall clock)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    async def start(self, runner: JobRunner) -> None:
        """Reconcile persisted jobs and start the worker loop (idempotent)."""
        self._runner = runner
        await self._reconcile_on_startup()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._worker_task = None

    async def _reconcile_on_startup(self) -> None:
        """Mid-flight jobs (running/queued/pausing) from a prior process become
        resumable ``paused`` in the preempted tier so they rejoin backfill."""
        async with self._cond:
            for job in self._store.load_all():
                if job.state in (
                    JobState.RUNNING,
                    JobState.QUEUED,
                    JobState.PAUSING,
                ):
                    job.state = JobState.PAUSED
                    job.preempted = True
                    self._store.save(job)
                self._jobs[job.job_id] = job
            self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # submission + control
    # ------------------------------------------------------------------ #

    async def submit(
        self,
        job_id: str,
        *,
        kind: str = "extract",
        params: dict[str, Any] | None = None,
    ) -> Job:
        """Enqueue a foreground job, preempting the running one if allowed."""
        async with self._cond:
            self._seq += 1
            job = Job(
                job_id=job_id,
                kind=kind,
                state=JobState.QUEUED,
                submitted=float(self._seq),
                params=params or {},
            )
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._store.save(job)
            self._preempt_running_locked()
            self._cond.notify_all()
        return job

    async def hold(self, job_id: str) -> None:
        """User-pause a job: sticky ``held``, not auto-backfilled."""
        async with self._cond:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.user_held = True
            if job.state == JobState.RUNNING and self._current == job_id:
                self._pause_flags.add(job_id)
                job.state = JobState.PAUSING
            elif job.state in (JobState.PAUSED, JobState.QUEUED):
                job.state = JobState.HELD
                if job_id in self._queue:
                    self._queue.remove(job_id)
            self._store.save(job)
            self._cond.notify_all()

    async def resume(self, job_id: str) -> None:
        """Resume a held/paused job as foreground work."""
        async with self._cond:
            job = self._jobs.get(job_id)
            if not job or job.state not in (JobState.HELD, JobState.PAUSED):
                return
            job.user_held = False
            job.state = JobState.QUEUED
            if job_id not in self._queue:
                self._queue.append(job_id)
            self._store.save(job)
            self._cond.notify_all()

    def should_pause(self, job_id: str) -> bool:
        """Cooperative check the runner calls at safe boundaries."""
        return job_id in self._pause_flags

    async def checkpoint(self, job_id: str, checkpoint: dict[str, Any]) -> None:
        """Persist a job's resume point (called by the runner)."""
        async with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.checkpoint = dict(checkpoint)
            self._store.save(job)

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return [j.public() for j in self._jobs.values()]

    # ------------------------------------------------------------------ #
    # internals (all called under self._cond)
    # ------------------------------------------------------------------ #

    def _preempt_running_locked(self) -> None:
        jid = self._current
        if not jid:
            return
        cur = self._jobs.get(jid)
        if not cur or cur.state != JobState.RUNNING:
            return
        is_background = cur.kind in self.BACKGROUND_KINDS
        if not is_background and not self._preempt_foreground:
            return  # foreground job runs to its next yield point uninterrupted
        self._pause_flags.add(jid)
        cur.state = JobState.PAUSING
        if not is_background:
            cur.preempted = True
        self._store.save(cur)

    def _next_foreground(self) -> str | None:
        while self._queue:
            jid = self._queue[0]
            job = self._jobs.get(jid)
            if job and job.state == JobState.QUEUED:
                return jid
            self._queue.pop(0)  # drop stale entries
        return None

    def _paused_pool(self) -> list[str]:
        cands = [
            j
            for j in self._jobs.values()
            if j.state == JobState.PAUSED and not j.user_held
        ]
        # preempted (interrupted) jobs resume first, then oldest by submit order
        cands.sort(key=lambda j: (0 if j.preempted else 1, j.submitted))
        return [j.job_id for j in cands]

    def _select_next_locked(self) -> str | None:
        fg = self._next_foreground()
        if fg is not None:
            self._queue.remove(fg)
            return fg
        if self._auto_backfill:
            for jid in self._paused_pool():
                job = self._jobs[jid]
                job.state = JobState.QUEUED
                job.preempted = False
                self._store.save(job)
                return jid
        return None

    async def _worker(self) -> None:
        assert self._runner is not None
        while True:
            async with self._cond:
                job_id = self._select_next_locked()
                while job_id is None:
                    await self._cond.wait()
                    job_id = self._select_next_locked()
                job = self._jobs[job_id]
                job.state = JobState.RUNNING
                self._current = job_id
                self._pause_flags.discard(job_id)
                self._store.save(job)

            try:
                await self._runner(job, self)
            except Exception as e:  # noqa: BLE001 — one bad job never kills the loop
                logger.warning("job %s failed: %s", job_id, e)
                async with self._cond:
                    job.state = JobState.FAILED
                    job.error = str(e)
                    self._store.save(job)

            async with self._cond:
                self._current = None
                paused = job_id in self._pause_flags
                self._pause_flags.discard(job_id)
                if job.state not in (JobState.DONE, JobState.FAILED):
                    if job.user_held:
                        job.state = JobState.HELD
                    elif paused:
                        job.state = JobState.PAUSED
                    else:
                        # runner returned without finishing or pausing → treat as
                        # done to avoid a stuck non-terminal job.
                        job.state = JobState.DONE
                    self._store.save(job)
                self._cond.notify_all()
