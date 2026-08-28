"""Characterization tests for ``TaskManagerMixin._run_background_task``
(CX-AU-01, CCN 106 in ``engine_tasks.py``).

This function is a single ~970-line ``try: if/elif...else: except: finally:``
dispatching on ``task_type`` (and, for one branch, the ``is_codebase`` flag)
across 21 mutually-exclusive branches, each of which does substantial,
independently-imported work (ingestion, sync, scheduling, research...).

The refactor this pins is a pure **extract-branches-into-handler-methods +
dispatch-table** transform: every branch body moves verbatim (after a
mechanical dedent) into its own ``_bg_<name>`` method, and the top-level
function becomes two dict lookups plus the original try/except/finally. Since
each branch's *internal* logic is unchanged (moved byte-for-byte), the
property that actually carries refactor risk is **routing**: for a given
``(task_type, is_codebase)`` pair, does control reach the same branch body as
before? These tests pin that property, plus the two structural invariants a
naive dict-only rewrite would get wrong:

  1. The three leading, independent ``if ... return`` checks (scheduled_job/
     enrichment_backfill, research_paper_fetch, kg_memory) and the first
     seven ``elif`` branches (conversation, content_url, feed_ingest,
     feed_sweep, skill_workflows, diff, deep_analysis) all take priority over
     the ``is_codebase`` catch-all, because they were checked earlier in the
     original chain.
  2. The ``is_codebase`` catch-all itself takes priority over every *later*
     ``elif`` (relevance_sweep, self_tool_surface, connector_sync/
     capability_hydration, connector_drain, fleet_event_triage, deploy_watch,
     synthesize/deep_extract/background_research, cohort_synthesize,
     session_upload) and over the trailing bare ``else`` (document ingest).

Coverage intentionally stops at "which branch executed, with what
externally-visible side effect" rather than re-verifying each branch's full
internal business logic — that logic is not changing, so re-deriving it here
would just be a slower copy of the code under test. ``skill_workflows`` has
no dedicated test below because it already has real regression coverage in
``tests/unit/knowledge_graph/core/test_engine_tasks_skill_workflows_atomic_pairing.py``,
which exercises ``_run_background_task`` end to end for that task_type and is
re-run as part of Rung 1 for this refactor.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

FIXTURE_FILE = Path("/var/tmp/cx/cx-au-01/fixtures/tiny.txt")


class _FakeTaskEngine(TaskManagerMixin):
    """Minimal stand-in exposing only what ``_run_background_task`` and its
    branches touch on ``self`` — same pattern as
    ``test_engine_tasks_skill_workflows_atomic_pairing.py``."""

    def __init__(self, *, metadata: dict | None = None, backend=None) -> None:
        self.updates: list[tuple[str, str, dict]] = []
        self.retries: list[tuple[str, str, dict]] = []
        self.checkpoint_calls = 0
        self.added_nodes: list[tuple[str, str, dict | None]] = []
        self.submitted_tasks: list[dict] = []
        self.query_cypher_calls: list[tuple[str, dict | None]] = []
        self.memory_kwargs: dict | None = None
        self.drained: tuple[str, str] | None = None
        self._metadata = metadata or {}
        self.backend = backend
        self._bulk_active = False
        self._fanout_result = False
        self._claim = {"lease_epoch": 1, "fencing_token": "fx-1"}

    # -- hooks every/most branches touch --
    def _ingest_task_metadata(self, job_id: str) -> dict:
        return dict(self._metadata)

    def _update_task_status(self, job_id: str, status: str, payload: dict) -> None:
        self.updates.append((job_id, status, payload))

    def _fail_or_retry_task(self, job_id: str, error_msg: str, payload: dict) -> None:
        self.retries.append((job_id, error_msg, payload))

    def _checkpoint_db(self) -> None:
        self.checkpoint_calls += 1

    # -- branch-specific hooks (only exercised by the branches that use them) --
    def add_node(
        self, nid: str, label: str, properties: dict | None = None, **_kw
    ) -> None:
        self.added_nodes.append((nid, label, properties))

    def query_cypher(self, query: str, params: dict | None = None):
        self.query_cypher_calls.append((query, params))
        return []

    def submit_task(self, **kwargs) -> None:
        self.submitted_tasks.append(kwargs)

    def execute_deep_analysis(self, query: str, max_depth: int) -> dict:
        return {"status": "success", "discovered_targets": []}

    def _bulk_ingest_active(self, threshold: int = 1) -> bool:
        return self._bulk_active

    def _maybe_fanout_codebase(self, job_id, target, meta) -> bool:
        return self._fanout_result

    def store_memory(self, **kwargs):
        self.memory_kwargs = kwargs
        return "mem-1"

    async def _run_relevance_sweep(self, job_id: str, target_codebase: str) -> dict:
        return {"swept": True}

    def _active_work_item_claim(self, job_id, pop: bool = False):
        return self._claim

    def _drain_session_upload(self, job_id: str, task_type: str) -> None:
        self.drained = (job_id, task_type)


def _ingestion_result(**overrides) -> SimpleNamespace:
    defaults = dict(
        status="success", nodes_created=1, edges_created=0, details={}, error=None
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# 1. scheduled_job / enrichment_backfill (leading independent `if ... return`)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", ["scheduled_job", "enrichment_backfill"])
async def test_scheduled_job_dispatches_and_records_schedule_result(task_type):
    engine = _FakeTaskEngine(metadata={"schedule": "sched-a", "payload": {"x": 1}})
    with (
        patch(
            "agent_utilities.core.schedule_engine.run_scheduled_job",
            return_value={"status": "ok", "duration_s": 2.5},
        ) as run_mock,
        patch(
            "agent_utilities.core.schedule_engine.record_schedule_result"
        ) as record_mock,
    ):
        await engine._run_background_task(
            job_id="job:1", target=Path("x"), is_codebase=False, task_type=task_type
        )

    run_mock.assert_called_once()
    record_mock.assert_called_once_with(
        engine, "sched-a", True, duration_s=2.5, status="ok"
    )
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:1", "completed")
    assert payload["schedule"] == "sched-a"
    assert engine.checkpoint_calls == 1


@pytest.mark.asyncio
async def test_scheduled_job_failure_still_records_failed_status():
    engine = _FakeTaskEngine(metadata={"schedule": "sched-b", "payload": {}})
    with (
        patch(
            "agent_utilities.core.schedule_engine.run_scheduled_job",
            side_effect=RuntimeError("boom"),
        ),
        patch(
            "agent_utilities.core.schedule_engine.record_schedule_result"
        ) as record_mock,
    ):
        await engine._run_background_task(
            job_id="job:2",
            target=Path("x"),
            is_codebase=False,
            task_type="scheduled_job",
        )

    assert record_mock.call_args.args[2] is False  # ok=False
    job_id, status, _ = engine.updates[-1]
    assert (job_id, status) == ("job:2", "failed")
    # The outer except was NOT triggered — this is an app-level "ok=False", not
    # an unhandled exception, so _fail_or_retry_task must not have fired.
    assert engine.retries == []


# ---------------------------------------------------------------------------
# 2. research_paper_fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_paper_fetch_dispatches_to_pipeline_runner():
    engine = _FakeTaskEngine(
        metadata={
            "paper": {
                "id": "p1",
                "title": "T",
                "abstract": "A",
                "authors": [],
                "url": "http://example/x",
                "score": 0.9,
                "domains": ["d"],
            }
        }
    )
    runner_instance = MagicMock()
    runner_instance.ingest_paper_full = AsyncMock(return_value="article-1")
    with patch(
        "agent_utilities.automation.research_pipeline.ResearchPipelineRunner",
        return_value=runner_instance,
    ) as runner_cls:
        await engine._run_background_task(
            job_id="job:3",
            target=Path("p1"),
            is_codebase=False,
            task_type="research_paper_fetch",
        )

    runner_cls.assert_called_once_with(engine=engine)
    runner_instance.ingest_paper_full.assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:3", "completed")
    assert payload["article_id"] == "article-1"
    assert "profile" in payload


# ---------------------------------------------------------------------------
# 3. kg_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kg_memory_dispatches_to_store_memory():
    engine = _FakeTaskEngine(
        metadata={"payload": {"content": "c", "memory_type": "episodic", "name": "n"}}
    )
    await engine._run_background_task(
        job_id="job:4", target=Path("x"), is_codebase=False, task_type="kg_memory"
    )

    assert engine.memory_kwargs is not None
    assert engine.memory_kwargs["content"] == "c"
    assert engine.memory_kwargs["_local"] is True
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status, payload["memory_id"]) == ("job:4", "completed", "mem-1")


# ---------------------------------------------------------------------------
# 4. conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_dispatches_by_source_in_path():
    target = Path("/var/tmp/cx/cx-au-01/fixtures/.claude/projects/p/conv.jsonl")
    engine = _FakeTaskEngine()
    with (
        patch(
            "agent_utilities.knowledge_graph.core.conversation_ingestion.parse_claude_logs",
            return_value=[{"path": str(target)}],
        ) as parse_mock,
        patch(
            "agent_utilities.knowledge_graph.core.conversation_ingestion.ingest_conversations_to_kg",
            return_value={"total_ingested": 1, "total_messages": 3},
        ) as ingest_mock,
    ):
        await engine._run_background_task(
            job_id="job:5", target=target, is_codebase=False, task_type="conversation"
        )

    parse_mock.assert_called_once()
    ingest_mock.assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:5", "completed")
    assert payload["total_ingested"] == 1


# ---------------------------------------------------------------------------
# 5. content_url
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_url_dispatches_to_ingestion_engine():
    engine = _FakeTaskEngine(metadata={"source_url": "https://example.com/a"})
    ing_instance = MagicMock()
    ing_instance.ingest = AsyncMock(
        return_value=_ingestion_result(status="success", nodes_created=2)
    )
    with patch(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        return_value=ing_instance,
    ):
        await engine._run_background_task(
            job_id="job:6",
            target=Path("unused"),
            is_codebase=False,
            task_type="content_url",
        )

    ing_instance.ingest.assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:6", "completed")
    assert payload["target"] == "https://example.com/a"
    assert payload["nodes"] == 2


# ---------------------------------------------------------------------------
# 6. feed_ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_ingest_dispatches_to_document_processor():
    engine = _FakeTaskEngine(
        metadata={
            "feed_doc": {
                "document_id": "doc-1",
                "text": "hello",
                "title": "T",
                "doc_type": "news_article",
                "source": "s",
                "metadata": {},
            }
        }
    )
    proc_instance = MagicMock()
    ing_instance = MagicMock()
    ing_instance.enrich_text = AsyncMock(return_value=None)
    with (
        patch(
            "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
            return_value=proc_instance,
        ) as proc_cls,
        patch(
            "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
            return_value=ing_instance,
        ),
    ):
        await engine._run_background_task(
            job_id="job:7",
            target=Path("unused"),
            is_codebase=False,
            task_type="feed_ingest",
        )

    proc_cls.assert_called_once()
    proc_instance.process.assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:7", "completed")
    assert payload["target"] == "doc-1"


@pytest.mark.asyncio
async def test_feed_ingest_missing_payload_fails_without_dispatch():
    engine = _FakeTaskEngine(metadata={"feed_doc": {}})
    await engine._run_background_task(
        job_id="job:7b",
        target=Path("unused"),
        is_codebase=False,
        task_type="feed_ingest",
    )
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:7b", "failed")
    assert payload["error"] == "no feed_doc payload"


# ---------------------------------------------------------------------------
# 7. feed_sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_sweep_dispatches_to_sync_source():
    engine = _FakeTaskEngine(metadata={"feed_source": "rss", "feed_mode": "delta"})
    with patch(
        "agent_utilities.knowledge_graph.core.source_sync.sync_source",
        return_value={"ok": True},
    ) as sync_mock:
        await engine._run_background_task(
            job_id="job:8",
            target=Path("unused"),
            is_codebase=False,
            task_type="feed_sweep",
        )

    sync_mock.assert_called_once_with(engine, "rss", mode="delta")
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:8", "completed")
    assert payload["target"] == "feed:rss"


# ---------------------------------------------------------------------------
# 8. diff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diff_dispatches_and_embeds_then_adds_node():
    engine = _FakeTaskEngine()
    embed_model = MagicMock()
    embed_model.get_text_embedding = MagicMock(return_value=[0.1, 0.2])
    with patch(
        "agent_utilities.core.embedding_utilities.create_embedding_model",
        return_value=embed_model,
    ):
        await engine._run_background_task(
            job_id="job:9",
            target=Path("not-a-real-file-diff-content"),
            is_codebase=False,
            task_type="diff",
        )

    assert len(engine.added_nodes) == 1
    nid, label, props = engine.added_nodes[0]
    assert label == "DiffEntry"
    assert nid.startswith("diff-")
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:9", "completed")
    assert payload["diffs_added"] == 1


# ---------------------------------------------------------------------------
# 9. deep_analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_analysis_dispatches_flat_when_no_fanout():
    engine = _FakeTaskEngine(metadata={"current_depth": 0, "max_depth": 0})
    await engine._run_background_task(
        job_id="job:10",
        target=Path("concept-x"),
        is_codebase=False,
        task_type="deep_analysis",
    )
    assert engine.submitted_tasks == []
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:10", "completed")
    assert payload["result"]["status"] == "success"


@pytest.mark.asyncio
async def test_deep_analysis_recurses_into_discovered_targets():
    engine = _FakeTaskEngine(metadata={"current_depth": 0, "max_depth": 2})
    engine.execute_deep_analysis = lambda query, max_depth: {
        "status": "success",
        "discovered_targets": ["other-concept"],
    }
    await engine._run_background_task(
        job_id="job:11",
        target=Path("concept-x"),
        is_codebase=False,
        task_type="deep_analysis",
    )
    assert len(engine.submitted_tasks) == 1
    submitted = engine.submitted_tasks[0]
    assert submitted["target_path"] == "other-concept"
    assert submitted["task_type"] == "deep_analysis"
    assert submitted["provenance"]["parent_concept"] == "concept-x"


# ---------------------------------------------------------------------------
# 10. codebase (is_codebase flag OR literal task_type == "codebase")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_codebase_via_is_codebase_flag():
    engine = _FakeTaskEngine()
    ing_instance = MagicMock()
    ing_instance.ingest = AsyncMock(
        return_value=_ingestion_result(
            status="success", nodes_created=5, edges_created=3
        )
    )
    with patch(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        return_value=ing_instance,
    ):
        await engine._run_background_task(
            job_id="job:12",
            target=Path("/repo"),
            is_codebase=True,
            task_type="document",
        )

    ing_instance.ingest.assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:12", "completed")
    assert payload["type"] == "codebase"
    assert payload["nodes_added"] == 5


@pytest.mark.asyncio
async def test_codebase_via_literal_task_type_even_when_is_codebase_false():
    engine = _FakeTaskEngine()
    ing_instance = MagicMock()
    ing_instance.ingest = AsyncMock(return_value=_ingestion_result())
    with patch(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        return_value=ing_instance,
    ):
        await engine._run_background_task(
            job_id="job:13",
            target=Path("/repo"),
            is_codebase=False,
            task_type="codebase",
        )
    ing_instance.ingest.assert_called_once()
    assert engine.updates[-1][2]["type"] == "codebase"


@pytest.mark.asyncio
async def test_codebase_fanout_returns_early_without_completing_status():
    """When ``_maybe_fanout_codebase`` fans a big repo out into shard sub-tasks,
    the ORIGINAL code returns from inside the ``try`` block without ever
    calling ``_update_task_status`` — but ``finally: self._checkpoint_db()``
    still runs. This is the trickiest control-flow case the refactor has to
    preserve: an early ``return`` from inside an extracted handler method
    must still reach the outer dispatcher's ``finally``."""
    engine = _FakeTaskEngine()
    engine._fanout_result = True
    await engine._run_background_task(
        job_id="job:14",
        target=Path("/big-repo"),
        is_codebase=True,
        task_type="document",
    )
    assert engine.updates == []
    assert engine.checkpoint_calls == 1


# ---------------------------------------------------------------------------
# 11. relevance_sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relevance_sweep_dispatches_to_helper():
    engine = _FakeTaskEngine()
    called = {}

    async def _fake_sweep(job_id, target_codebase):
        called["args"] = (job_id, target_codebase)
        return {"swept": 3}

    engine._run_relevance_sweep = _fake_sweep
    await engine._run_background_task(
        job_id="job:15",
        target=Path("/repo"),
        is_codebase=False,
        task_type="relevance_sweep",
    )
    assert called["args"] == ("job:15", "/repo")
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status, payload) == ("job:15", "completed", {"swept": 3})


# ---------------------------------------------------------------------------
# 12. self_tool_surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_tool_surface_dispatches_to_ingestion_engine():
    engine = _FakeTaskEngine()
    ing_instance = MagicMock()
    ing_instance._ingest_self_tools = AsyncMock(
        return_value=_ingestion_result(
            status="success", nodes_created=7, edges_created=4
        )
    )
    with patch(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        return_value=ing_instance,
    ):
        await engine._run_background_task(
            job_id="job:16",
            target=Path("unused"),
            is_codebase=False,
            task_type="self_tool_surface",
        )
    ing_instance._ingest_self_tools.assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:16", "completed")
    assert payload["nodes_added"] == 7


# ---------------------------------------------------------------------------
# 13. connector_sync / capability_hydration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", ["connector_sync", "capability_hydration"])
async def test_connector_sync_dispatches_to_sync_source(task_type):
    engine = _FakeTaskEngine(metadata={"sync_mode": "full"})
    with patch(
        "agent_utilities.knowledge_graph.core.source_sync.sync_source",
        return_value={"synced": 9},
    ) as sync_mock:
        await engine._run_background_task(
            job_id="job:17",
            target=Path("gitlab"),
            is_codebase=False,
            task_type=task_type,
        )
    sync_mock.assert_called_once_with(engine, "gitlab", mode="full")
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:17", "completed")
    assert payload["synced"] == 9


# ---------------------------------------------------------------------------
# 14. connector_drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connector_drain_dispatches_to_run_drain_page():
    engine = _FakeTaskEngine(
        metadata={
            "drain_source": "confluence",
            "sync_mode": "full",
            "drain_id": "d1",
            "drain_page": 2,
            "drain_checkpoint": "{}",
        }
    )
    with patch(
        "agent_utilities.knowledge_graph.core.chunked_drain.run_drain_page",
        return_value={"has_more": False},
    ) as drain_mock:
        await engine._run_background_task(
            job_id="job:18",
            target=Path("unused"),
            is_codebase=False,
            task_type="connector_drain",
        )
    drain_mock.assert_called_once()
    assert drain_mock.call_args.kwargs["page"] == 2
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:18", "completed")
    assert payload["has_more"] is False


# ---------------------------------------------------------------------------
# 15. fleet_event_triage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fleet_event_triage_dispatches_and_ensures_playbooks():
    engine = _FakeTaskEngine()
    with (
        patch(
            "agent_utilities.knowledge_graph.adaptation.remediation_playbooks.ensure_registered"
        ) as ensure_mock,
        patch(
            "agent_utilities.knowledge_graph.adaptation.fleet_event_triage.triage_fleet_event",
            return_value={"severity": "warn"},
        ) as triage_mock,
    ):
        await engine._run_background_task(
            job_id="job:19",
            target=Path("event:1"),
            is_codebase=False,
            task_type="fleet_event_triage",
        )
    ensure_mock.assert_called_once()
    triage_mock.assert_called_once_with(engine, "event:1")
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:19", "completed")
    assert payload["severity"] == "warn"


# ---------------------------------------------------------------------------
# 16. deploy_watch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_watch_dispatches_to_run_deploy_watch():
    engine = _FakeTaskEngine()
    with patch(
        "agent_utilities.orchestration.deploy_watch.run_deploy_watch",
        return_value={"rolled_back": False},
    ) as watch_mock:
        await engine._run_background_task(
            job_id="job:20",
            target=Path("svc-x"),
            is_codebase=False,
            task_type="deploy_watch",
        )
    watch_mock.assert_called_once_with(engine, "svc-x", "job:20")
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:20", "completed")
    assert payload["rolled_back"] is False


# ---------------------------------------------------------------------------
# 17. synthesize / deep_extract / background_research
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_type,attr",
    [
        ("synthesize", "synthesize"),
        ("deep_extract", "deep_extract"),
        ("background_research", "background_research"),
    ],
)
async def test_analyzer_task_dispatches_to_matching_method(task_type, attr):
    engine = _FakeTaskEngine(metadata={"top_k": 7})
    analyzer_instance = MagicMock()
    setattr(analyzer_instance, attr, AsyncMock(return_value={"ok": True}))
    with patch(
        "agent_utilities.analysis.analyzer.GraphAnalyzer",
        return_value=analyzer_instance,
    ):
        await engine._run_background_task(
            job_id="job:21",
            target=Path("query-x"),
            is_codebase=False,
            task_type=task_type,
        )
    getattr(analyzer_instance, attr).assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:21", "completed")
    assert payload["result"] == {"ok": True}


@pytest.mark.asyncio
async def test_analyzer_task_exception_is_caught_locally_not_by_outer_handler():
    """The ``synthesize``/``deep_extract``/``background_research`` branch wraps
    its own call in a LOCAL try/except that calls ``_fail_or_retry_task``
    directly — it must NOT fall through to the outer function-level except
    (which would additionally stamp a ``traceback`` key onto the payload)."""
    engine = _FakeTaskEngine(metadata={"top_k": 1})
    analyzer_instance = MagicMock()
    analyzer_instance.synthesize = AsyncMock(side_effect=ValueError("analyzer boom"))
    with patch(
        "agent_utilities.analysis.analyzer.GraphAnalyzer",
        return_value=analyzer_instance,
    ):
        await engine._run_background_task(
            job_id="job:22",
            target=Path("query-x"),
            is_codebase=False,
            task_type="synthesize",
        )
    assert len(engine.retries) == 1
    job_id, error_msg, payload = engine.retries[0]
    assert job_id == "job:22"
    assert "analyzer boom" in error_msg
    assert "traceback" not in payload  # proves the LOCAL catch fired, not the outer one
    assert engine.checkpoint_calls == 1


# ---------------------------------------------------------------------------
# 18. cohort_synthesize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cohort_synthesize_finalizes_when_ready():
    engine = _FakeTaskEngine(metadata={"cohort_id": "c1", "deadline_unix": 0.0})
    with (
        patch(
            "agent_utilities.knowledge_graph.research.cohort.cohort_ready",
            return_value=(True, {"total": 2, "terminal": 2}),
        ),
        patch(
            "agent_utilities.knowledge_graph.research.cohort.finalize_cohort",
            return_value={"feature_matrix": {"counts": {"n": 4}}},
        ) as finalize_mock,
    ):
        await engine._run_background_task(
            job_id="job:23",
            target=Path("unused"),
            is_codebase=False,
            task_type="cohort_synthesize",
        )
    finalize_mock.assert_called_once_with(engine, "c1")
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:23", "completed")
    assert payload["feature_matrix"] == {"n": 4}


@pytest.mark.asyncio
async def test_cohort_synthesize_defers_when_not_ready():
    engine = _FakeTaskEngine(
        metadata={"cohort_id": "c2", "deadline_unix": 0.0, "work_item_id": "wi-1"}
    )
    with (
        patch(
            "agent_utilities.knowledge_graph.research.cohort.cohort_ready",
            return_value=(False, {"total": 2, "terminal": 1}),
        ),
        patch(
            "agent_utilities.orchestration.work_item.defer_work_item", return_value=True
        ) as defer_mock,
    ):
        await engine._run_background_task(
            job_id="job:24",
            target=Path("unused"),
            is_codebase=False,
            task_type="cohort_synthesize",
        )
    defer_mock.assert_called_once()
    # A deferral is not a completion — no terminal status is recorded here.
    assert engine.updates == []
    assert engine.checkpoint_calls == 1


# ---------------------------------------------------------------------------
# 19. session_upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_upload_dispatches_to_drain_helper():
    engine = _FakeTaskEngine()
    await engine._run_background_task(
        job_id="job:25",
        target=Path("unused"),
        is_codebase=False,
        task_type="session_upload",
    )
    assert engine.drained == ("job:25", "session_upload")
    # The drain helper owns its own status recording; the dispatcher does not
    # call _update_task_status for this branch (matches the original explicit
    # `return` right after the call).
    assert engine.updates == []
    assert engine.checkpoint_calls == 1


# ---------------------------------------------------------------------------
# 20. default / else (document ingest) — reached for ANY unrecognized
#     task_type when is_codebase is False, not just the literal "document".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", ["document", "some-unrecognized-type"])
async def test_unmatched_task_type_falls_back_to_document_ingest(task_type):
    engine = _FakeTaskEngine(backend=MagicMock())
    fake_doc = SimpleNamespace(
        text="hello world", metadata={"file_path": str(FIXTURE_FILE)}
    )
    reader_instance = MagicMock()
    reader_instance.load_data = MagicMock(return_value=[fake_doc])
    embed_model = MagicMock()
    embed_model.get_text_embedding_batch = MagicMock(return_value=[[0.1, 0.2]])

    with (
        patch("llama_index.core.SimpleDirectoryReader", return_value=reader_instance),
        patch(
            "agent_utilities.core.embedding_utilities.create_embedding_model",
            return_value=embed_model,
        ),
    ):
        await engine._run_background_task(
            job_id="job:26", target=FIXTURE_FILE, is_codebase=False, task_type=task_type
        )

    assert len(engine.added_nodes) == 1
    nid, label, props = engine.added_nodes[0]
    assert label == "Article"
    job_id, status, payload = engine.updates[-1]
    assert (job_id, status) == ("job:26", "completed")
    assert payload["type"] == "document"
    assert payload["chunks_added"] == 1
    engine.backend.execute.assert_called_once()  # the stale-row cleanup DELETE


# ---------------------------------------------------------------------------
# 21. priority — the two invariants a naive rewrite would get wrong
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_priority_early_literal_type_beats_is_codebase_flag():
    """``conversation`` is checked BEFORE the ``is_codebase`` catch-all in the
    original chain, so is_codebase=True must not steal it."""
    target = Path("/var/tmp/cx/cx-au-01/fixtures/.claude/projects/p/conv.jsonl")
    engine = _FakeTaskEngine()
    with (
        patch(
            "agent_utilities.knowledge_graph.core.conversation_ingestion.parse_claude_logs",
            return_value=[{"path": str(target)}],
        ) as parse_mock,
        patch(
            "agent_utilities.knowledge_graph.core.conversation_ingestion.ingest_conversations_to_kg",
            return_value={"total_ingested": 1, "total_messages": 1},
        ),
        patch(
            "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine"
        ) as ing_cls,
    ):
        await engine._run_background_task(
            job_id="job:27", target=target, is_codebase=True, task_type="conversation"
        )
    parse_mock.assert_called_once()
    ing_cls.assert_not_called()
    assert engine.updates[-1][2]["type"] == "conversation"


@pytest.mark.asyncio
async def test_priority_explicit_late_literal_type_beats_is_codebase_flag():
    """BUG-CX-051 (fixed): an explicit ``task_type`` that has its own
    dedicated handler in ``_LATE_TASK_HANDLERS`` (e.g. ``relevance_sweep``)
    must route to THAT handler regardless of ``is_codebase`` -- the
    ``is_codebase``/``"codebase"`` branch is the generic catch-all for a task
    with no more specific ``task_type``, not a flag that should silently
    override an explicit, more specific task_type. The previous ordering let
    ``is_codebase=True`` silently misroute ``relevance_sweep`` (and
    ``self_tool_surface``/``connector_drain``/etc.) to ``_bg_codebase``
    purely by if/elif position -- latent because no live caller combined
    both, but any caller that does must not get silently wrong routing.
    """
    engine = _FakeTaskEngine()
    swept = {"called": False}

    async def _fake_sweep(job_id, target_codebase):
        swept["called"] = True
        return {"status": "completed", "type": "relevance_sweep"}

    engine._run_relevance_sweep = _fake_sweep
    ing_instance = MagicMock()
    ing_instance.ingest = AsyncMock(return_value=_ingestion_result())
    with patch(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        return_value=ing_instance,
    ):
        await engine._run_background_task(
            job_id="job:28",
            target=Path("/repo"),
            is_codebase=True,
            task_type="relevance_sweep",
        )
    assert swept["called"] is True, (
        "is_codebase=True silently pre-empted the explicit relevance_sweep "
        "task_type instead of routing to its dedicated handler"
    )
    ing_instance.ingest.assert_not_called()
    assert engine.updates[-1][2]["type"] == "relevance_sweep"


# ---------------------------------------------------------------------------
# 22. the outer except/finally: unhandled exception -> retry, checkpoint always
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unhandled_branch_exception_reaches_outer_fail_or_retry():
    engine = _FakeTaskEngine()

    async def _boom(job_id, target_codebase):
        raise ValueError("relevance sweep exploded")

    engine._run_relevance_sweep = _boom
    await engine._run_background_task(
        job_id="job:29",
        target=Path("/repo"),
        is_codebase=False,
        task_type="relevance_sweep",
    )
    assert len(engine.retries) == 1
    job_id, error_msg, payload = engine.retries[0]
    assert job_id == "job:29"
    assert "relevance sweep exploded" in error_msg
    assert "traceback" in payload  # proves the OUTER except fired
    assert engine.updates == []
    assert engine.checkpoint_calls == 1
