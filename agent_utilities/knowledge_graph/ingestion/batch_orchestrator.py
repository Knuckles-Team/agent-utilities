#!/usr/bin/python
from __future__ import annotations

"""Enterprise-scale repository batch ingestion (CONCEPT:KG-2.19 / KG-2.49).

Fans out deep code ingestion for tens of thousands of repos into the KG's durable
task queue without oversubscribing the engine. The companion remote enumeration
(GitLab/GitHub listing + shallow clone) lives in the ``repository-manager`` agent;
this module owns the KG-side fan-out, idempotency, and backpressure.

Design (Phase-4 scale-out of ``.specify/specs/ecosystem-evolution``):

  - **Bulk prefilter** via :class:`DeltaManifest` — one ``load_for_graph`` query
    returns ``{clone_path: head_sha}`` so unchanged repos (HEAD unmoved) are
    skipped in memory, no per-repo round-trip. This is the resumability spine: a
    crashed run resumes by re-reading the manifest and re-skipping recorded repos.
  - **Backpressure** — never dumps all N tasks at once; maintains a target
    in-flight depth (``KG_INGEST_INFLIGHT``) by polling the ``:Task`` queue, so
    the auto-scaling workers and the Rust engine stay within their op budget.
  - **Idempotent submit** — ``engine.submit_task`` already dedupes in-flight by
    target; the manifest is recorded at submit time keyed by ``head_sha`` so a
    re-run with the same HEAD is a no-op.

The same path serves incremental re-ingest: a push webhook resolves to a single
:class:`RepoRef` whose ``head_sha`` moved, and ``submit_batch`` re-queues exactly
that one repo.
"""

from dataclasses import dataclass, field
from typing import Any

from agent_utilities.core.config import setting

from .manifest import DeltaManifest

_CATEGORY = "codebase"
_DEFAULT_INFLIGHT = 40


@dataclass
class RepoRef:
    """A repository to (re-)ingest, as produced by the VCS enumerator.

    Attributes:
        vcs: ``gitlab`` | ``github``.
        full_path: Group/namespace path or ``org/name`` (the stable repo key).
        clone_path: Local working-tree path of the (shallow) clone — the ingest
            ``source_uri``.
        head_sha: HEAD commit sha — the content hash for idempotency.
        clone_url / web_url / default_branch / last_activity_at / archived:
            enumeration metadata (provenance / coarse change pre-filter).
    """

    vcs: str
    full_path: str
    clone_path: str
    head_sha: str
    clone_url: str = ""
    web_url: str = ""
    default_branch: str = ""
    last_activity_at: str = ""
    archived: bool = False

    def provenance(self, run_id: str = "") -> dict[str, Any]:
        """Provenance dict stamped onto the ingest task / produced nodes."""
        return {
            "vcs": self.vcs,
            "full_path": self.full_path,
            "head_sha": self.head_sha,
            "web_url": self.web_url,
            "default_branch": self.default_branch,
            "run_id": run_id,
        }


@dataclass
class BatchProgress:
    """Outcome of a ``submit_batch`` call."""

    enumerated: int = 0
    skipped_unchanged: int = 0
    skipped_archived: int = 0
    submitted: int = 0
    deferred_backpressure: int = 0
    job_ids: list[str] = field(default_factory=list)


def _inflight_count(engine: Any) -> int | None:
    """In-flight ingest depth (``None`` if not queryable).

    Prefers the engine's uniform :meth:`ingest_queue_depth` (CONCEPT:KG-2.57):
    the selected queue backend's not-yet-claimed backlog (Kafka = ``kg-ingest``
    consumer-group lag; SQLite/Postgres = row count) PLUS pending/running
    ``:Task`` nodes — so backpressure sees work the graph poll alone would miss
    (e.g. unconsumed Kafka messages). Falls back to the :Task count for engines
    without the method.
    """
    depth_fn = getattr(engine, "ingest_queue_depth", None)
    if callable(depth_fn):
        try:
            return int(depth_fn())
        except Exception:  # noqa: BLE001 — fall through to the :Task count
            pass
    q = getattr(engine, "query_cypher", None)
    if not callable(q):
        return None
    try:
        rows = q(
            "MATCH (t:Task) WHERE t.status IN ['pending','running'] "
            "RETURN count(t) AS c"
        )
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return 0
    row = rows[0]
    if isinstance(row, dict):
        return int(row.get("c", 0) or 0)
    if isinstance(row, list | tuple):
        return int(row[0]) if row else 0
    try:
        return int(row)
    except (TypeError, ValueError):
        return 0


class RepoBatchIngestor:
    """Fan repos into the KG durable queue, idempotent + backpressured."""

    def __init__(
        self,
        engine: Any,
        *,
        graph_name: str = "default",
        manifest: DeltaManifest | None = None,
        inflight_target: int | None = None,
    ) -> None:
        self.engine = engine
        self.graph_name = graph_name
        self.manifest = manifest or DeltaManifest(
            backend=getattr(engine, "backend", None)
        )
        env = setting("KG_INGEST_INFLIGHT")
        self.inflight_target = inflight_target or (
            int(env) if env and env.isdigit() else _DEFAULT_INFLIGHT
        )

    def submit_batch(
        self,
        repo_refs: list[RepoRef],
        *,
        run_id: str = "",
        include_archived: bool = False,
    ) -> BatchProgress:
        """Submit deep-ingest tasks for the changed repos, respecting backpressure.

        Skips archived repos and repos whose ``head_sha`` matches the manifest
        (unchanged). Submits up to the in-flight target, deferring the rest (a
        later call resumes them — the manifest makes already-submitted repos
        skip). Returns a :class:`BatchProgress`.
        """
        progress = BatchProgress(enumerated=len(repo_refs))
        known = self.manifest.load_for_graph(self.graph_name, _CATEGORY)
        submit = getattr(self.engine, "submit_task", None)
        if not callable(submit):
            return progress

        inflight = _inflight_count(self.engine)
        for ref in repo_refs:
            if ref.archived and not include_archived:
                progress.skipped_archived += 1
                continue
            if known.get(ref.clone_path) == ref.head_sha and ref.head_sha:
                progress.skipped_unchanged += 1
                continue
            # Backpressure: stop submitting once the queue hits the target depth;
            # remaining repos resume on the next call (manifest skips done ones).
            if inflight is not None and inflight >= self.inflight_target:
                progress.deferred_backpressure += 1
                continue
            try:
                job_id = submit(
                    ref.clone_path,
                    True,  # is_codebase
                    ref.provenance(run_id),
                    "codebase",
                )
            except Exception:  # noqa: BLE001 - one failed submit never aborts the batch  # nosec B112
                continue
            progress.submitted += 1
            progress.job_ids.append(job_id)
            if ref.head_sha:
                self.manifest.record(
                    self.graph_name, _CATEGORY, ref.clone_path, ref.head_sha
                )
            if inflight is not None:
                inflight += 1
        return progress

    def status(self) -> dict[str, int]:
        """Return queue counts by task status (``pending``/``running``/...)."""
        q = getattr(self.engine, "query_cypher", None)
        out: dict[str, int] = {}
        if not callable(q):
            return out
        try:
            rows = q("MATCH (t:Task) RETURN t.status AS s, count(t) AS c")
        except Exception:  # noqa: BLE001
            return out
        for row in rows or []:
            if isinstance(row, dict):
                out[str(row.get("s", "unknown"))] = int(row.get("c", 0) or 0)
            elif isinstance(row, list | tuple) and len(row) >= 2:
                out[str(row[0])] = int(row[1])
        return out


__all__ = ["RepoRef", "BatchProgress", "RepoBatchIngestor"]
