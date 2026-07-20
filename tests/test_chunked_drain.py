"""Chunked async drain — single big source_sync(full) → capacity-guarded waves.

CONCEPT:AU-KG.ontology.single-source-full-drain (chunked drain) / KG-2.302 (connector-declared page drainer).

Proves: (a) a single-source FULL sync ENQUEUES a paginated batch-task and returns a handle
immediately (no inline drain-to-completion); (b) the page-tasks walk the connector cursor to
EXHAUSTION, covering the WHOLE corpus; (c) the page-tasks carry the background priority /
capacity-guard context; (d) a small delta sync stays INLINE.
"""

from __future__ import annotations

import pytest

import agent_utilities.knowledge_graph.core.chunked_drain as cd
import agent_utilities.knowledge_graph.core.source_sync as ss
from agent_utilities.core.resource_priority import (
    PriorityClass,
    priority_for_task_type,
)
from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type
from agent_utilities.protocols.source_connectors.base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    PollConnector,
    SourceDocument,
)

CORPUS_SIZE = 250
PAGE = 100


@pytest.fixture(autouse=True)
def _source_sync_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda _source: {"checked": True, "ok": True},
    )


class _FakeCorpusConnector(PollConnector):
    """A fake source whose 250-item corpus paginates via ``checkpoint.state['offset']``.

    Stateless across instances (rebuilt per page-task) — all position lives in the cursor.
    """

    source_type = "test_corpus"

    def configure(self, **_: object) -> None:  # noqa: D401
        pass

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        offset = int((checkpoint.state.get("offset") if checkpoint else 0) or 0)
        docs = [
            SourceDocument(id=str(i), text=f"item {i}")
            for i in range(offset, min(offset + PAGE, CORPUS_SIZE))
        ]
        new_offset = offset + len(docs)
        has_more = new_offset < CORPUS_SIZE
        cp = ConnectorCheckpoint(
            has_more=has_more,
            watermark=None if has_more else "wm-final",
            state={"offset": new_offset} if has_more else {},
        )
        return CheckpointedBatch(documents=docs, checkpoint=cp)


class _FakeEngine:
    """Records submit_task calls; no real worker — the test drives run_drain_page."""

    def __init__(self) -> None:
        self.backend = object()
        self.tasks: list[dict] = []

    def submit_task(
        self,
        target_path,
        is_codebase,
        provenance,
        task_type=None,
        skip_dedupe=False,
        priority=None,
        extra_meta=None,
        **_,
    ):
        jid = f"job-{len(self.tasks)}"
        self.tasks.append(
            {
                "job_id": jid,
                "target": target_path,
                "task_type": task_type,
                "provenance": dict(provenance or {}),
                "priority": priority,
                "skip_dedupe": skip_dedupe,
                "extra_meta": dict(extra_meta or {}),
            }
        )
        return jid


def _install_fake_drainer(monkeypatch):
    """Register a test PageDrainer + in-memory drain-state store; return the ingested list."""
    ingested: list[str] = []

    def _build(engine, mode):  # noqa: ARG001
        return _FakeCorpusConnector()

    def _ingest(engine, docs):  # noqa: ARG001
        ingested.extend(d.id for d in docs)
        return {"items": len(docs), "ingested": len(docs)}

    monkeypatch.setitem(
        cd._PAGE_DRAINERS,
        "test_corpus",
        cd.PageDrainer(
            source="test_corpus", build_connector=_build, ingest_page=_ingest
        ),
    )

    store: dict[str, dict] = {}
    monkeypatch.setattr(
        cd,
        "_write_drain_state",
        lambda e, did, src, st: store.__setitem__(did, dict(st)),
    )
    monkeypatch.setattr(cd, "_read_drain_state", lambda e, did: store.get(did))
    monkeypatch.setattr(
        cd,
        "_active_drain_for",
        lambda e, src: next(
            (
                d
                for d, s in store.items()
                if s.get("source") == src and s.get("status") == "draining"
            ),
            None,
        ),
    )
    return ingested, store


# ── (a) full sync enqueues a batch-task + returns a handle immediately ────────


def test_full_drain_enqueues_and_returns_handle_without_inline(monkeypatch):
    ingested, _ = _install_fake_drainer(monkeypatch)
    engine = _FakeEngine()

    handle = cd.start_chunked_drain(engine, "test_corpus", mode="full")

    assert handle["status"] == "draining"
    assert handle["drain_id"].startswith("test_corpus-")
    assert handle["first_task"] == "job-0"
    # Returned a HANDLE — did NOT drain inline to completion.
    assert ingested == []
    # Exactly ONE page-task enqueued (the first wave).
    assert len(engine.tasks) == 1
    t = engine.tasks[0]
    assert t["task_type"] == cd.DRAIN_TASK_TYPE
    assert t["target"] == "test_corpus"
    assert t["skip_dedupe"] is True


# ── (b) the page-tasks walk the cursor to exhaustion (whole corpus) ───────────


def test_drain_pages_walk_whole_corpus_to_exhaustion(monkeypatch):
    ingested, store = _install_fake_drainer(monkeypatch)
    watermarks: dict[str, str] = {}
    monkeypatch.setattr(
        ss,
        "_ingest_graph_slice_via_envelope",
        lambda _engine, source, _entities, **kwargs: (
            watermarks.__setitem__(source, kwargs["checkpoint"])
            or {"status": "success"}
        ),
    )

    engine = _FakeEngine()
    handle = cd.start_chunked_drain(engine, "test_corpus", mode="full")
    drain_id = handle["drain_id"]

    # Drive the self-continuing chain exactly as the worker would: each enqueued
    # connector_drain task → run_drain_page → it enqueues the next, until exhausted.
    idx = 0
    pages = 0
    while idx < len(engine.tasks):
        task = engine.tasks[idx]
        idx += 1
        if task["task_type"] != cd.DRAIN_TASK_TYPE:
            continue
        res = cd.run_drain_page(
            engine,
            source=task["provenance"]["drain_source"],
            mode=task["provenance"]["sync_mode"],
            drain_id=task["provenance"]["drain_id"],
            page=task["provenance"]["drain_page"],
            checkpoint_json=task["extra_meta"].get("drain_checkpoint"),
        )
        pages += 1
        if not res["has_more"]:
            assert res["drain_status"] == "completed"

    # Whole corpus drained across the wave of page-tasks (100 + 100 + 50).
    assert sorted(int(x) for x in ingested) == list(range(CORPUS_SIZE))
    assert pages == 3
    assert store[drain_id]["status"] == "completed"
    assert store[drain_id]["items_ingested"] == CORPUS_SIZE
    # Watermark advanced on exhaustion so later delta syncs are cheap.
    assert watermarks.get("test_corpus") == "wm-final"


def test_failed_page_retains_cursor_and_does_not_enqueue_successor(monkeypatch):
    _ingested, _store = _install_fake_drainer(monkeypatch)
    original = cd._PAGE_DRAINERS["test_corpus"]
    monkeypatch.setitem(
        cd._PAGE_DRAINERS,
        "test_corpus",
        cd.PageDrainer(
            source="test_corpus",
            build_connector=original.build_connector,
            ingest_page=lambda _engine, docs: {
                "items": len(docs),
                "ingested": 0,
                "failed": 1,
            },
        ),
    )
    engine = _FakeEngine()
    handle = cd.start_chunked_drain(engine, "test_corpus", mode="full")
    task = engine.tasks[0]

    with pytest.raises(RuntimeError, match="retained its cursor"):
        cd.run_drain_page(
            engine,
            source="test_corpus",
            mode="full",
            drain_id=handle["drain_id"],
            page=0,
            checkpoint_json=task["extra_meta"].get("drain_checkpoint"),
        )

    assert len(engine.tasks) == 1


# ── (c) page-tasks carry background priority + capacity-guard context ─────────


def test_drain_tasks_carry_background_priority_context(monkeypatch):
    _install_fake_drainer(monkeypatch)
    engine = _FakeEngine()
    cd.start_chunked_drain(engine, "test_corpus", mode="full")

    t = engine.tasks[0]
    # Background claim bucket (3) — yields to interactive/orchestration.
    assert t["priority"] == 3
    # The task type routes to the connectors lane → BACKGROUND_INGESTION priority class,
    # which is the same gate that applies the model-server capacity guard (ORCH-1.99/1.102).
    assert lane_for_task_type(cd.DRAIN_TASK_TYPE) == "connectors"
    assert (
        priority_for_task_type(cd.DRAIN_TASK_TYPE) == PriorityClass.BACKGROUND_INGESTION
    )
    # Drain context is carried in WorkItem metadata for progress projection.
    assert t["provenance"]["drain_source"] == "test_corpus"
    assert t["provenance"]["sync_mode"] == "full"
    assert "drain_id" in t["provenance"]


# ── (d) a small delta sync stays INLINE (fast path preserved) ─────────────────


def test_delta_sync_stays_inline(monkeypatch):
    _install_fake_drainer(monkeypatch)
    # Register the test source as a delta handler so sync_source can route it inline.
    sentinel = {"called": False}

    def _inline_handler(engine, *, mode, ids, client):  # noqa: ARG001
        sentinel["called"] = True
        return {"status": "ok", "source": "test_corpus", "mode": mode, "inline": True}

    monkeypatch.setitem(ss._DELTA_HANDLERS, "test_corpus", _inline_handler)
    engine = _FakeEngine()

    res = ss.sync_source(engine, "test_corpus", mode="delta")

    # Connector-specific diagnostics remain available without escaping the
    # canonical EtlResult envelope.
    assert res["details"]["inline"] is True
    assert sentinel["called"] is True
    # Delta did NOT enqueue any chunked page-task.
    assert engine.tasks == []


def test_full_sync_routes_to_chunked_drain(monkeypatch):
    _install_fake_drainer(monkeypatch)
    captured = {}

    def _fake_start(engine, source, *, mode="full"):
        captured["source"] = source
        captured["mode"] = mode
        return {"status": "draining", "drain_id": "x"}

    monkeypatch.setattr(cd, "start_chunked_drain", _fake_start)
    # The inline handler must NOT run when the full path chunks.
    monkeypatch.setitem(
        ss._DELTA_HANDLERS,
        "test_corpus",
        lambda *a, **k: {"status": "ok", "inline": True},
    )
    engine = _FakeEngine()

    res = ss.sync_source(engine, "test_corpus", mode="full")

    assert res["status"] == "draining"
    assert captured == {"source": "test_corpus", "mode": "full"}


# ── freshrss is registered as the flagship large-corpus drainer ───────────────


def test_freshrss_registered_for_chunked_drain():
    assert cd.supports_chunked_drain("freshrss")
    assert "freshrss" in cd.list_chunked_sources()
