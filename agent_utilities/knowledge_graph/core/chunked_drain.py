"""Chunked async drain — one big `source_sync(full)` call, normalized into waves.

CONCEPT:AU-KG.ontology.single-source-full-drain — Chunked async drain (the "controlled waves" baked in)
CONCEPT:AU-KG.compute.connector-declared-page-drainer — Connector-declared page drainer (generalized pagination)

A single ``source_sync(source=X, mode="full")`` on a LARGE corpus (e.g. FreshRSS's
~11k-article backlog) must NOT run synchronously to completion: that would block the
MCP/REST request until the whole corpus is drained (timeout) or force a human/agent to
hand-repeat delta waves. Instead the system normalizes that ONE call into a **stream of
paginated, capacity-guarded batch-tasks**:

* :func:`start_chunked_drain` enqueues the FIRST ``connector_drain`` batch-task and returns
  IMMEDIATELY with a handle (``drain_id`` + how to watch progress) — never the full result.
* each ``connector_drain`` task drains ONE bounded page (``KG_DRAIN_PAGE_SIZE`` items) via the
  connector's own resumable :meth:`PollConnector.poll` cursor, ingests it, then — while the
  cursor still ``has_more`` — **self-continues** by enqueuing the NEXT page-task carrying the
  advanced :class:`ConnectorCheckpoint`. The corpus drains across many tasks until the cursor
  is exhausted.
* the page-tasks ride the existing ``connectors`` lane (CONCEPT:AU-ORCH.execution.two-level-fair-rotation), so each runs under
  the BACKGROUND_INGESTION priority edict (ORCH-1.98/1.99) and the GB10 server-capacity guard
  (ORCH-1.102/1.103) — it can't time out the request and can't OOM the box. Re-draining is
  cheap: the write-layer content-hash delta (KG-2.9) skips unchanged items, and each page-task
  is idempotent + resumable from its carried checkpoint.

**Connector-declared pagination (KG-2.302).** A source opts into chunked drain by registering a
:class:`PageDrainer` — *how to build its connector* and *how to ingest one drained page*. The
generic driver below walks ANY such connector's :meth:`poll` cursor to exhaustion, so the
mechanism is not FreshRSS-specific: any large poll-paginated source can register one.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: The task type the page-tasks carry (mapped to the ``connectors`` lane in task_lanes).
DRAIN_TASK_TYPE = "connector_drain"

#: Hard backstop on the number of page-tasks one drain may chain — a defensive guard
#: against a connector that reports ``has_more`` forever. ~11k items / 100 per page ≈ 110
#: pages, so 5000 is generous headroom. Override with ``KG_DRAIN_MAX_PAGES``.
_DEFAULT_MAX_PAGES = 5000


def _setting(key: str, default: str) -> str:
    from agent_utilities.core.config import setting

    return str(setting(key, default=default) or default)


def _drain_page_size() -> int:
    """Items drained per page-task (bounded so a page completes within the lane timeout)."""
    try:
        return max(1, int(_setting("KG_DRAIN_PAGE_SIZE", "100")))
    except (TypeError, ValueError):
        return 100


def _drain_max_pages() -> int:
    try:
        return max(1, int(_setting("KG_DRAIN_MAX_PAGES", str(_DEFAULT_MAX_PAGES))))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_PAGES


def chunked_drain_enabled() -> bool:
    """Whether the chunked-drain path is active (default on; ``KG_CHUNKED_DRAIN=false`` disables)."""
    from agent_utilities.base_utilities import to_boolean

    return to_boolean(_setting("KG_CHUNKED_DRAIN", "True"))


# ── Connector-declared page drainers (CONCEPT:AU-KG.compute.connector-declared-page-drainer) ──────────────────────


@dataclass(frozen=True)
class PageDrainer:
    """How to drain ONE source page-by-page across batch-tasks (CONCEPT:AU-KG.compute.connector-declared-page-drainer).

    Attributes:
        source: the ``source_sync`` key this drains (e.g. ``"freshrss"``).
        build_connector: ``(engine, mode) -> PollConnector`` — a connector whose
            :meth:`poll` walks the corpus via a resumable cursor. For ``mode="full"`` it
            must walk the ENTIRE backlog (no since-filter); for ``"delta"`` it may bind the
            watermark. The connector is rebuilt per page-task, so ALL pagination state must
            live in the carried :class:`ConnectorCheckpoint` (the connector itself is
            stateless across tasks).
        ingest_page: ``(engine, docs) -> dict`` — ingest one drained page and return its
            counts (folded into the page-task result + the cumulative drain state).
    """

    source: str
    build_connector: Callable[[Any, str], Any]
    ingest_page: Callable[[Any, list[Any]], dict[str, Any]]


_PAGE_DRAINERS: dict[str, PageDrainer] = {}


def register_page_drainer(drainer: PageDrainer) -> None:
    """Register a :class:`PageDrainer` under its ``source`` key (idempotent)."""
    _PAGE_DRAINERS[drainer.source] = drainer


def get_page_drainer(source: str) -> PageDrainer | None:
    return _PAGE_DRAINERS.get((source or "").lower().strip())


def supports_chunked_drain(source: str) -> bool:
    """True when ``source`` has a registered page drainer (so a full sync can chunk)."""
    return (source or "").lower().strip() in _PAGE_DRAINERS


def list_chunked_sources() -> list[str]:
    return sorted(_PAGE_DRAINERS)


# ── Drain-state node (progress, queryable alongside :Task) ───────────────────


def _drain_node_id(drain_id: str) -> str:
    return f"drain:{drain_id}"


def _write_drain_state(
    engine: Any, drain_id: str, source: str, state: dict[str, Any]
) -> None:
    """Upsert the :SourceDrain progress node (best-effort, backend-agnostic JSON blob)."""
    backend = getattr(engine, "backend", None)
    if backend is None:
        return
    try:
        backend.execute(
            "MERGE (n:SourceDrain {id: $id}) "
            "SET n.source = $src, n.status = $status, n.state = $state, n.updated_at = $ts",
            {
                "id": _drain_node_id(drain_id),
                "src": source,
                "status": str(state.get("status") or "draining"),
                "state": json.dumps(state, default=str),
                "ts": datetime.now(UTC).isoformat(),
            },
        )
    except Exception:  # noqa: BLE001 — progress is best-effort; the drain still runs
        logger.debug("drain state write failed for %s", drain_id, exc_info=True)


def _read_drain_state(engine: Any, drain_id: str) -> dict[str, Any] | None:
    backend = getattr(engine, "backend", None)
    if backend is None:
        return None
    try:
        rows = backend.execute(
            "MATCH (n:SourceDrain {id: $id}) RETURN n.state AS s",
            {"id": _drain_node_id(drain_id)},
        )
        for r in rows or []:
            raw = r.get("s") if isinstance(r, dict) else None
            if raw:
                return json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.debug("drain state read failed for %s", drain_id, exc_info=True)
    return None


def _active_drain_for(engine: Any, source: str) -> str | None:
    """The drain_id of an in-flight drain for ``source`` (so we don't start a duplicate)."""
    backend = getattr(engine, "backend", None)
    if backend is None:
        return None
    try:
        rows = backend.execute(
            "MATCH (n:SourceDrain) WHERE n.source = $src AND n.status = 'draining' "
            "RETURN n.id AS id",
            {"src": source},
        )
        for r in rows or []:
            nid = r.get("id") if isinstance(r, dict) else None
            if nid:
                return str(nid).split("drain:", 1)[-1]
    except Exception:  # noqa: BLE001
        logger.debug("active-drain probe failed for %s", source, exc_info=True)
    return None


# ── The driver: enqueue → drain one page → self-continue ─────────────────────


def _enqueue_drain_page(
    engine: Any,
    *,
    source: str,
    mode: str,
    drain_id: str,
    page: int,
    checkpoint_json: str | None,
) -> str:
    """Enqueue ONE ``connector_drain`` page-task (background bucket, dedupe bypassed).

    All page-tasks share ``target=source``, so the target-based dedupe MUST be bypassed
    (``skip_dedupe``); the per-page identity is ``drain_id``+``page``. Carried on the Task:
    ``sync_mode``/``drain_id``/``drain_source``/``drain_page`` top-level (queryable) and the
    serialized checkpoint in the metadata blob.
    """
    return engine.submit_task(
        target_path=source,
        is_codebase=False,
        provenance={
            "sync_mode": mode,
            "drain_id": drain_id,
            "drain_source": source,
            "drain_page": page,
        },
        task_type=DRAIN_TASK_TYPE,
        skip_dedupe=True,
        priority=3,  # background bucket — yields to interactive/orchestration work
        extra_meta={"drain_checkpoint": checkpoint_json, "drain_page": page},
    )


def start_chunked_drain(
    engine: Any, source: str, *, mode: str = "full"
) -> dict[str, Any]:
    """Begin a chunked drain of ``source`` and return a handle IMMEDIATELY (CONCEPT:AU-KG.ontology.single-source-full-drain).

    Enqueues the first ``connector_drain`` page-task and returns ``{drain_id, status, …}``
    without draining inline. If a drain for ``source`` is already in flight, returns its
    handle (idempotent — no duplicate chain).
    """
    source = (source or "").lower().strip()
    drainer = get_page_drainer(source)
    if drainer is None:
        raise KeyError(f"no chunked-drain page drainer registered for {source!r}")

    existing = _active_drain_for(engine, source)
    if existing:
        return {
            "status": "already_draining",
            "source": source,
            "mode": mode,
            "drain_id": existing,
            "note": "a drain for this source is already in flight",
            **_progress_view(engine, existing),
        }

    drain_id = f"{source}-{uuid.uuid4().hex[:8]}"
    _write_drain_state(
        engine,
        drain_id,
        source,
        {
            "status": "draining",
            "source": source,
            "mode": mode,
            "pages_done": 0,
            "items_seen": 0,
            "items_ingested": 0,
            "started_at": datetime.now(UTC).isoformat(),
        },
    )
    first = _enqueue_drain_page(
        engine,
        source=source,
        mode=mode,
        drain_id=drain_id,
        page=0,
        checkpoint_json=None,
    )
    page_size = _drain_page_size()
    logger.info(
        "[KG-2.301] started chunked drain %s for %s (mode=%s, page_size=%d)",
        drain_id,
        source,
        mode,
        page_size,
    )
    return {
        "status": "draining",
        "source": source,
        "mode": mode,
        "drain_id": drain_id,
        "page_size": page_size,
        "first_task": first,
        "watch": {
            "status_tool": f"source_drain (action=status, drain_id={drain_id})",
            "task_query": (
                "MATCH (t:Task) WHERE t.drain_source = "
                f"'{source}' AND t.drain_id = '{drain_id}' "
                "RETURN t.status AS status, count(*) AS n ORDER BY status"
            ),
        },
        "note": (
            "Draining the whole corpus across capacity-guarded background page-tasks; "
            "this returned immediately. Watch via source_drain status or the task_query."
        ),
    }


def run_drain_page(
    engine: Any,
    *,
    source: str,
    mode: str,
    drain_id: str,
    page: int,
    checkpoint_json: str | None,
) -> dict[str, Any]:
    """Drain ONE page, ingest it, and self-continue while the cursor has more (KG-2.301).

    Rebuilds the connector, resumes its :meth:`poll` from the carried checkpoint, ingests the
    batch via the source's :class:`PageDrainer`, and — when ``checkpoint.has_more`` — enqueues
    the next page-task carrying the advanced checkpoint. On exhaustion (or the page backstop)
    marks the drain complete and advances the source watermark so later delta syncs are cheap.
    """
    from agent_utilities.protocols.source_connectors.base import ConnectorCheckpoint

    drainer = get_page_drainer(source)
    if drainer is None:
        return {
            "status": "error",
            "source": source,
            "reason": "no page drainer registered",
        }

    conn = drainer.build_connector(engine, mode)
    checkpoint = (
        ConnectorCheckpoint.from_json(checkpoint_json) if checkpoint_json else None
    )
    batch = conn.poll(checkpoint)
    docs = list(batch.documents or [])
    counts = drainer.ingest_page(engine, docs) if docs else {"items": 0}

    next_cp = batch.checkpoint
    has_more = bool(getattr(next_cp, "has_more", False))
    # Defensive: a cursor that reports has_more but yields nothing must not loop forever.
    exhausted_empty = has_more and not docs
    over_backstop = (page + 1) >= _drain_max_pages()

    prior = _read_drain_state(engine, drain_id) or {}
    pages_done = int(prior.get("pages_done", 0)) + 1
    items_seen = int(prior.get("items_seen", 0)) + len(docs)
    items_ingested = int(prior.get("items_ingested", 0)) + int(
        counts.get("ingested", counts.get("items", 0)) or 0
    )

    next_task: str | None = None
    if has_more and not exhausted_empty and not over_backstop:
        status = "draining"
        next_task = _enqueue_drain_page(
            engine,
            source=source,
            mode=mode,
            drain_id=drain_id,
            page=page + 1,
            checkpoint_json=next_cp.to_json(),
        )
    else:
        status = "completed"
        if over_backstop and has_more:
            status = "stopped_backstop"
        # Advance the source watermark once the full walk exhausts, so later delta syncs
        # only pull what changed (the connector parks the high-water on the final checkpoint).
        watermark = getattr(next_cp, "watermark", None)
        if watermark:
            try:
                from agent_utilities.knowledge_graph.core.source_sync import (
                    _write_watermark,
                )

                _write_watermark(
                    getattr(engine, "backend", None), source, str(watermark)
                )
            except Exception:  # noqa: BLE001 — watermark advance is best-effort
                logger.debug(
                    "drain watermark advance failed for %s", source, exc_info=True
                )

    state = {
        "status": status,
        "source": source,
        "mode": mode,
        "pages_done": pages_done,
        "items_seen": items_seen,
        "items_ingested": items_ingested,
        "has_more": has_more and status == "draining",
        "started_at": prior.get("started_at"),
        "last_page_at": datetime.now(UTC).isoformat(),
    }
    if status != "draining":
        state["completed_at"] = datetime.now(UTC).isoformat()
    _write_drain_state(engine, drain_id, source, state)

    return {
        "status": "ok",
        "drain_id": drain_id,
        "source": source,
        "mode": mode,
        "page": page,
        "page_items": len(docs),
        "page_counts": counts,
        "has_more": has_more and status == "draining",
        "next_task": next_task,
        "drain_status": status,
        "cumulative": {
            "pages_done": pages_done,
            "items_seen": items_seen,
            "items_ingested": items_ingested,
        },
    }


def _progress_view(engine: Any, drain_id: str) -> dict[str, Any]:
    state = _read_drain_state(engine, drain_id) or {}
    return {
        k: state[k]
        for k in ("pages_done", "items_seen", "items_ingested")
        if k in state
    }


def drain_status(engine: Any, drain_id: str) -> dict[str, Any]:
    """Queryable progress for a drain handle (CONCEPT:AU-KG.ontology.single-source-full-drain).

    Returns the cumulative :SourceDrain state plus a live per-status breakdown of the
    chain's ``connector_drain`` :Task nodes, so an operator/agent fires ONE call and
    watches it drain.
    """
    state = _read_drain_state(engine, drain_id)
    tasks: dict[str, int] = {}
    cc = getattr(engine, "_control_cypher", None)
    if callable(cc):
        try:
            rows = cc(
                "MATCH (t:Task) WHERE t.drain_id = $id "
                "RETURN t.status AS status, count(*) AS n",
                {"id": drain_id},
            )
            for r in rows or []:
                if isinstance(r, dict) and r.get("status"):
                    tasks[str(r["status"])] = int(r.get("n", 0) or 0)
        except Exception:  # noqa: BLE001 — task breakdown is best-effort
            logger.debug("drain task breakdown failed for %s", drain_id, exc_info=True)
    if state is None:
        return {"drain_id": drain_id, "status": "unknown", "tasks": tasks}
    return {"drain_id": drain_id, **state, "tasks": tasks}


# ── FreshRSS page drainer (the flagship large corpus, CONCEPT:AU-KG.compute.homelab-rss-reader-as) ───────


def _freshrss_build_connector(engine: Any, mode: str) -> Any:
    """Build the FreshRSS connector for a chunked page-walk.

    ``mode="full"`` binds NO ``newer_than`` so :meth:`poll` walks the ENTIRE backlog via the
    GReader ``continuation`` cursor (all ~11k items); ``"delta"`` binds the persisted watermark
    so it only walks what's new. ``batch_size`` = the page size, so one ``poll`` returns one
    bounded page and parks the continuation on its checkpoint for the next page-task.
    """
    from agent_utilities.protocols.source_connectors.registry import build_connector

    page_size = _drain_page_size()
    config: dict[str, Any] = {
        "preset": "freshrss",
        "batch_size": page_size,
        "max_records": 0,
    }
    if mode != "full":
        from agent_utilities.knowledge_graph.core.source_sync import (
            _as_epoch,
            _read_watermark,
        )

        since = _read_watermark(getattr(engine, "backend", None), "freshrss")
        if since and (epoch := _as_epoch(since)) is not None:
            config["params"] = {"newer_than": epoch}
    return build_connector("mcp_tool", config)


def _freshrss_ingest_page(engine: Any, docs: list[Any]) -> dict[str, Any]:
    """Ingest one drained FreshRSS page through the world-model relevance gate (KG-2.116)."""
    from agent_utilities.automation.worldmodel_pipeline import (
        WorldModelConfig,
        WorldModelPipelineRunner,
    )
    from agent_utilities.base_utilities import to_boolean

    cfg = WorldModelConfig(
        use_novelty=to_boolean(_setting("FRESHRSS_USE_NOVELTY", "False"))
    )
    report = WorldModelPipelineRunner(engine=engine, config=cfg).run_gated_ingest(docs)
    return {
        "items": len(docs),
        "ingested": report.ingested,
        "relevant": report.relevant,
        "marginal": report.marginal,
        "research": report.research,
        "skipped_unchanged": report.skipped,
    }


register_page_drainer(
    PageDrainer(
        source="freshrss",
        build_connector=_freshrss_build_connector,
        ingest_page=_freshrss_ingest_page,
    )
)
