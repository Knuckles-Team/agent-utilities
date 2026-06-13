"""Extraction job manager — runs fact extraction through the GPU slot (KG-2.65).

The live consumer of :class:`GpuSlotScheduler`: it submits document-extraction
jobs, runs them one-at-a-time on the single GPU inference slot, checkpoints
per-file progress so a preempted or restart-interrupted job resumes from where
it stopped, and persists facts as graph edges as they are produced.

Checkpoints persist graph-natively (a small ``ExtractionJob`` node) so a redeploy
doesn't lose the queue — dogfooding the GraphBackend rather than loose files.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from ..ingestion.gpu_slot_scheduler import (
    CheckpointStore,
    GpuSlotScheduler,
    Job,
    JobState,
)
from .fact_extractor import (
    ExtractedFact,
    FactDeduper,
    extract_facts,
    facts_to_jsonl,
    persist_facts,
)

logger = logging.getLogger(__name__)

_JOB_NODE_TYPE = "extraction_job"


class GraphCheckpointStore:
    """Persist scheduler jobs as graph nodes so the queue survives a redeploy.

    Stores only the resumable metadata (state + checkpoint), not the facts —
    facts are written as edges incrementally by the runner. Best-effort: if the
    engine read/write is unavailable the scheduler still runs in-process.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def save(self, job: Job) -> None:
        try:
            self._engine.add_node(
                f"extractjob:{job.job_id}",
                _JOB_NODE_TYPE,
                {
                    "job_id": job.job_id,
                    "state": str(job.state),
                    "preempted": job.preempted,
                    "user_held": job.user_held,
                    "checkpoint_json": json.dumps(job.checkpoint),
                    "params_json": json.dumps(job.params),
                    "submitted": job.submitted,
                    "kind": job.kind,
                },
            )
        except Exception as e:  # noqa: BLE001 - persistence is best-effort
            logger.debug("checkpoint save failed for %s: %s", job.job_id, e)

    def load_all(self) -> list[Job]:
        nodes = self._query_job_nodes()
        jobs: list[Job] = []
        for n in nodes:
            try:
                jobs.append(
                    Job(
                        job_id=n["job_id"],
                        kind=n.get("kind", "extract"),
                        state=JobState(n.get("state", "queued")),
                        submitted=float(n.get("submitted", 0.0)),
                        preempted=bool(n.get("preempted", False)),
                        user_held=bool(n.get("user_held", False)),
                        params=json.loads(n.get("params_json", "{}")),
                        checkpoint=json.loads(n.get("checkpoint_json", "{}")),
                    )
                )
            except Exception:  # noqa: BLE001 - skip a corrupt row, don't fail boot
                continue
        return jobs

    def delete(self, job_id: str) -> None:
        # The engine has no hard-delete on this path; leave a tombstone state.
        # (A terminal job's node is harmless; the scheduler ignores DONE/FAILED.)
        return

    def _query_job_nodes(self) -> list[dict[str, Any]]:
        try:
            rows = self._engine.query(f"MATCH (n:{_JOB_NODE_TYPE}) RETURN n", None)
        except Exception:  # noqa: BLE001
            return []
        out: list[dict[str, Any]] = []
        for r in rows or []:
            node = r.get("n") if isinstance(r, dict) else None
            if isinstance(node, dict):
                out.append(node)
        return out


class ExtractionJobManager:
    """Owns the slot scheduler + per-job results for fact extraction."""

    def __init__(self, engine: Any, *, store: CheckpointStore | None = None) -> None:
        self._engine = engine
        self._scheduler = GpuSlotScheduler(
            store=store if store is not None else GraphCheckpointStore(engine),
            preempt_foreground=True,
        )
        # in-process results (facts) keyed by job_id; facts also persist as edges
        self._results: dict[str, list[ExtractedFact]] = {}
        # per-job event history (for SSE replay) + live subscribers
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._started = False

    _EVENT_BUFFER_CAP = 5000

    def _publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Buffer an event (bounded) and fan it out to live SSE subscribers."""
        buf = self._events.setdefault(job_id, [])
        buf.append(event)
        if len(buf) > self._EVENT_BUFFER_CAP:
            del buf[: len(buf) - self._EVENT_BUFFER_CAP]
        for q in self._subscribers.get(job_id, []):
            with contextlib.suppress(Exception):
                q.put_nowait(event)

    async def stream(self, job_id: str):
        """Yield this job's events (buffered history then live) until it ends.

        Mirrors the upstream SSE taxonomy so any frontend renders identically;
        terminates on the synthetic ``job_done`` event the runner emits.
        """
        q: asyncio.Queue = asyncio.Queue()
        # replay buffered history first so a late subscriber misses nothing
        for ev in list(self._events.get(job_id, [])):
            yield ev
            if ev.get("type") == "job_done":
                return
        self._subscribers.setdefault(job_id, []).append(q)
        try:
            while True:
                ev = await q.get()
                yield ev
                if ev.get("type") == "job_done":
                    return
        finally:
            subs = self._subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)

    async def ensure_started(self) -> None:
        if not self._started:
            await self._scheduler.start(self._run_job)
            self._started = True

    async def submit(
        self,
        *,
        text: str = "",
        files: list[dict[str, str]] | None = None,
        rounds: int = 1,
        dedup: bool = True,
        dedup_field: str = "triple",
        dedup_threshold: float = 0.90,
        job_id: str | None = None,
    ) -> str:
        """Submit an extraction job. ``files`` is ``[{name, text}]`` for a corpus;
        ``text`` is a single document. Returns the job id."""
        await self.ensure_started()
        jid = (
            job_id
            or f"ext-{abs(hash((text[:64], len(text), tuple(f['name'] for f in (files or []))))) & 0xFFFFFF:06x}"
        )
        params: dict[str, Any] = {
            "rounds": rounds,
            "dedup": dedup,
            "dedup_field": dedup_field,
            "dedup_threshold": dedup_threshold,
        }
        if files:
            params["files"] = files
        else:
            params["text"] = text
        self._results.setdefault(jid, [])
        await self._scheduler.submit(jid, kind="extract", params=params)
        return jid

    def status(self, job_id: str) -> dict[str, Any] | None:
        job = self._scheduler.get(job_id)
        if job is None:
            return None
        facts = self._results.get(job_id, [])
        unique = sum(1 for f in facts if not f.is_duplicate)
        return {
            **job.public(),
            "total_facts": len(facts),
            "unique_facts": unique,
            "duplicate_facts": len(facts) - unique,
        }

    def jobs(self) -> list[dict[str, Any]]:
        return [self.status(j["job_id"]) or j for j in self._scheduler.list_jobs()]

    def jsonl(self, job_id: str) -> str:
        return facts_to_jsonl(self._results.get(job_id, []))

    async def pause(self, job_id: str) -> None:
        await self._scheduler.hold(job_id)

    async def resume(self, job_id: str) -> None:
        await self._scheduler.resume(job_id)

    # ------------------------------------------------------------------ #
    # the runner (cooperative, resumable)
    # ------------------------------------------------------------------ #

    async def _run_job(self, job: Job, sched: GpuSlotScheduler) -> None:
        try:
            await self._run_job_inner(job, sched)
        except Exception:
            # ensure any SSE subscriber sees a terminal event on failure
            self._publish(job.job_id, {"type": "job_done", "state": "failed"})
            raise

    async def _run_job_inner(self, job: Job, sched: GpuSlotScheduler) -> None:
        p = job.params
        rounds = int(p.get("rounds", 1))
        deduper = (
            FactDeduper(
                field=p.get("dedup_field", "triple"),
                threshold=float(p.get("dedup_threshold", 0.90)),
            )
            if p.get("dedup", True)
            else None
        )
        results = self._results.setdefault(job.job_id, [])
        if deduper is not None:
            deduper.rehydrate(results)  # resume: seed corpus from kept facts

        files = p.get("files")
        if files:
            done_files = set(job.checkpoint.get("done_files", []))
            for idx, f in enumerate(files):
                name = f.get("name", str(idx))
                if name in done_files:
                    continue
                if sched.should_pause(job.job_id):
                    await sched.checkpoint(
                        job.job_id, {"done_files": sorted(done_files)}
                    )
                    return
                self._publish(
                    job.job_id, {"type": "file_start", "file": name, "file_index": idx}
                )
                await self._extract_into(
                    job.job_id, f.get("text", ""), name, rounds, deduper, results
                )
                done_files.add(name)
                self._publish(job.job_id, {"type": "file_end", "file": name})
                await sched.checkpoint(job.job_id, {"done_files": sorted(done_files)})
        else:
            if not job.checkpoint.get("done"):
                await self._extract_into(
                    job.job_id, p.get("text", ""), "", rounds, deduper, results
                )
                await sched.checkpoint(job.job_id, {"done": True})

        job.state = JobState.DONE
        self._publish(job.job_id, {"type": "job_done", "state": str(JobState.DONE)})

    async def _extract_into(
        self,
        job_id: str,
        text: str,
        source_file: str,
        rounds: int,
        deduper: FactDeduper | None,
        results: list[ExtractedFact],
    ) -> None:
        fresh: list[ExtractedFact] = []
        async for ev in extract_facts(
            text,
            rounds=rounds,
            dedup=deduper is not None,
            deduper=deduper,
            source_file=source_file,
        ):
            self._publish(job_id, ev)  # round_start / fact / metrics / round_end / done
            if ev["type"] == "fact":
                fact = ExtractedFact(**ev["fact"])
                results.append(fact)
                if not fact.is_duplicate:
                    fresh.append(fact)
        if fresh:
            persist_facts(EngineStoreAdapter(self._engine), fresh)


class EngineStoreAdapter:
    """Adapt ``IntelligenceGraphEngine`` to the ``add_node(label=)/add_edge``
    store protocol that ``persist_facts`` writes against, so fact persistence
    reuses the tested merge logic on the live engine (KG-2.64)."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def add_node(self, node_id: str, label: str = "", **props: Any) -> None:
        self._engine.add_node(node_id, "entity", {"label": label, **props})

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **props: Any
    ) -> None:
        self._engine.add_edge(source, target, rel_type, **props)
