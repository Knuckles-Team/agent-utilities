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
import hmac
import json
import logging
import re
from typing import Any
from uuid import uuid4

from agent_utilities.security.identifiers import validate_identifier

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

_JOB_NODE_TYPE = validate_identifier("extraction_job", kind="label")
_OWNER_PARAM = "_owner_ref"
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_CORPUS_BYTES = 16 * 1024 * 1024
_MAX_FILES = 256
_MAX_JOBS = 1024
_MAX_SUBSCRIBERS_PER_JOB = 16
_MAX_EVENT_BYTES = 256 * 1024
_OWNER_REF_RE = re.compile(r"pref_[a-z0-9_]+_[0-9a-f]{64}\Z")


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
            from agent_utilities.security.persistence_privacy import (
                persistence_reference,
            )

            params = job.params
            persistent_params: dict[str, Any] = {
                "rounds": max(1, min(10, int(params.get("rounds", 1)))),
                "dedup": bool(params.get("dedup", True)),
                "dedup_field": str(params.get("dedup_field", "triple"))[:32],
                "dedup_threshold": max(
                    0.0, min(1.0, float(params.get("dedup_threshold", 0.9)))
                ),
                _OWNER_PARAM: str(params.get(_OWNER_PARAM, ""))[:256],
                "payload_persisted": False,
            }
            if "files" in params:
                files = params.get("files") or []
                persistent_params.update(
                    {
                        "file_count": len(files),
                        "payload_ref": persistence_reference(
                            "extraction_corpus", files, namespace="extraction-job"
                        ),
                    }
                )
            else:
                text = str(params.get("text", ""))
                persistent_params.update(
                    {
                        "text_character_count": len(text),
                        "payload_ref": persistence_reference(
                            "extraction_document", text, namespace="extraction-job"
                        ),
                    }
                )
            self._engine.add_node(
                f"extractjob:{job.job_id}",
                _JOB_NODE_TYPE,
                {
                    "job_id": job.job_id,
                    "state": str(job.state),
                    "preempted": job.preempted,
                    "user_held": job.user_held,
                    "checkpoint_json": json.dumps(job.checkpoint),
                    "params_json": json.dumps(persistent_params),
                    "submitted": job.submitted,
                    "kind": job.kind,
                },
            )
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.debug(
                "checkpoint save failed (exception_type=%s)", type(exc).__name__
            )

    def load_all(self) -> list[Job]:
        nodes = self._query_job_nodes()
        jobs: list[Job] = []
        for n in nodes:
            try:
                state = JobState(n.get("state", "queued"))
                params = json.loads(n.get("params_json", "{}"))
                if not isinstance(params, dict):
                    continue
                # Document payloads deliberately never enter durable storage.
                # Interrupted jobs therefore become terminal and must be
                # resubmitted instead of silently running with missing input.
                payload_missing = not bool(params.get("payload_persisted", False))
                interrupted = payload_missing and state not in {
                    JobState.DONE,
                    JobState.FAILED,
                }
                if interrupted:
                    state = JobState.FAILED
                jobs.append(
                    Job(
                        job_id=n["job_id"],
                        kind=n.get("kind", "extract"),
                        state=state,
                        submitted=float(n.get("submitted", 0.0)),
                        preempted=bool(n.get("preempted", False)),
                        user_held=bool(n.get("user_held", False)),
                        params=params,
                        checkpoint=json.loads(n.get("checkpoint_json", "{}")),
                        error=(
                            "payload unavailable after restart" if interrupted else ""
                        ),
                    )
                )
            except Exception:  # noqa: BLE001 - skip a corrupt row, don't fail boot
                continue
        return jobs

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
        from agent_utilities.security.persistence_privacy import (
            PersistencePrivacyGuard,
        )

        clean, _report = PersistencePrivacyGuard().sanitize(event)
        if not isinstance(clean, dict):
            clean = {"type": "event_rejected"}
        try:
            encoded_size = len(
                json.dumps(clean, separators=(",", ":"), default=str).encode("utf-8")
            )
        except Exception:
            clean = {"type": "event_rejected"}
            encoded_size = 0
        if encoded_size > _MAX_EVENT_BYTES:
            clean = {"type": "event_rejected", "reason": "event_too_large"}
        buf = self._events.setdefault(job_id, [])
        buf.append(clean)
        if len(buf) > self._EVENT_BUFFER_CAP:
            del buf[: len(buf) - self._EVENT_BUFFER_CAP]
        for q in self._subscribers.get(job_id, []):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(clean)

    def _owned_job(self, job_id: str, owner_ref: str | None) -> Job | None:
        job = self._scheduler.get(job_id)
        if job is None:
            return None
        if owner_ref is None:
            return job
        stored = str(job.params.get(_OWNER_PARAM, ""))
        if not stored or not hmac.compare_digest(stored, owner_ref):
            return None
        return job

    async def stream(self, job_id: str, *, owner_ref: str | None = None):
        """Yield this job's events (buffered history then live) until it ends.

        Mirrors the upstream SSE taxonomy so any frontend renders identically;
        terminates on the synthetic ``job_done`` event the runner emits.
        """
        if self._owned_job(job_id, owner_ref) is None:
            raise KeyError("extraction job not found")
        subscribers = self._subscribers.setdefault(job_id, [])
        if len(subscribers) >= _MAX_SUBSCRIBERS_PER_JOB:
            raise RuntimeError("extraction subscriber limit reached")
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        # replay buffered history first so a late subscriber misses nothing
        for ev in list(self._events.get(job_id, [])):
            yield ev
            if ev.get("type") == "job_done":
                return
        subscribers.append(q)
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
        owner_ref: str | None = None,
    ) -> str:
        """Submit an extraction job. ``files`` is ``[{name, text}]`` for a corpus;
        ``text`` is a single document. Returns the job id."""
        await self.ensure_started()
        if len(self._scheduler.list_jobs()) >= _MAX_JOBS:
            raise RuntimeError("extraction job limit reached")
        from agent_utilities.security.persistence_privacy import (
            PersistencePrivacyGuard,
            persistence_reference,
        )

        guard = PersistencePrivacyGuard()
        if owner_ref is not None and not _OWNER_REF_RE.fullmatch(str(owner_ref)):
            raise ValueError("owner_ref must be an opaque persistence reference")
        if files is not None:
            if not isinstance(files, list) or len(files) > _MAX_FILES:
                raise ValueError("extraction corpus exceeds its file-count boundary")
            clean_files: list[dict[str, str]] = []
            total_bytes = 0
            for index, file in enumerate(files):
                if not isinstance(file, dict):
                    raise ValueError("extraction file entries must be objects")
                raw_text = file.get("text", "")
                if not isinstance(raw_text, str):
                    raise ValueError("extraction file content must be text")
                total_bytes += len(raw_text.encode("utf-8"))
                if total_bytes > _MAX_CORPUS_BYTES:
                    raise ValueError("extraction corpus exceeds its size boundary")
                raw_name = str(file.get("name", index))[:1024]
                clean_text, _ = guard.sanitize_text(raw_text)
                clean_files.append(
                    {
                        "name": persistence_reference(
                            "source", raw_name, namespace="extraction-corpus"
                        ),
                        "text": clean_text,
                    }
                )
            files = clean_files
            text = ""
        else:
            if (
                not isinstance(text, str)
                or len(text.encode("utf-8")) > _MAX_DOCUMENT_BYTES
            ):
                raise ValueError("extraction document exceeds its size boundary")
            text = guard.sanitize_text(text)[0]

        jid = (
            f"ext-{uuid4().hex}"
            if not job_id
            else persistence_reference(
                "extraction_job", str(job_id)[:1024], namespace="extraction-job"
            )
        )
        if dedup_field not in {
            "triple",
            "title",
            "description",
            "title+desc",
            "triple+title",
            "all",
        }:
            raise ValueError("unsupported extraction deduplication field")
        params: dict[str, Any] = {
            "rounds": max(1, min(10, int(rounds))),
            "dedup": dedup,
            "dedup_field": dedup_field,
            "dedup_threshold": max(0.0, min(1.0, float(dedup_threshold))),
            _OWNER_PARAM: str(owner_ref or "")[:256],
        }
        if files:
            params["files"] = files
        else:
            params["text"] = text
        self._results.setdefault(jid, [])
        await self._scheduler.submit(jid, kind="extract", params=params)
        return jid

    def status(
        self, job_id: str, *, owner_ref: str | None = None
    ) -> dict[str, Any] | None:
        job = self._owned_job(job_id, owner_ref)
        if job is None:
            return None
        facts = self._results.get(job_id, [])
        unique = sum(1 for f in facts if not f.is_duplicate)
        public = job.public()
        if public.get("error"):
            public["error"] = "extraction job failed"
        return {
            **public,
            "total_facts": len(facts),
            "unique_facts": unique,
            "duplicate_facts": len(facts) - unique,
        }

    def jobs(self, *, owner_ref: str | None = None) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for candidate in self._scheduler.list_jobs():
            status = self.status(candidate["job_id"], owner_ref=owner_ref)
            if status is not None:
                jobs.append(status)
        return jobs

    def jsonl(self, job_id: str, *, owner_ref: str | None = None) -> str:
        if self._owned_job(job_id, owner_ref) is None:
            raise KeyError("extraction job not found")
        return facts_to_jsonl(self._results.get(job_id, []))

    async def pause(self, job_id: str, *, owner_ref: str | None = None) -> None:
        if self._owned_job(job_id, owner_ref) is None:
            raise KeyError("extraction job not found")
        await self._scheduler.hold(job_id)

    async def resume(self, job_id: str, *, owner_ref: str | None = None) -> None:
        if self._owned_job(job_id, owner_ref) is None:
            raise KeyError("extraction job not found")
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
