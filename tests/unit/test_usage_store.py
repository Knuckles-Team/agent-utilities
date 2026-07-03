"""UsageStore (SQLite+FTS5 default) round-trip + analytics tests.

CONCEPT:ECO-4.39. Exercises write_bundle, summary/breakdown/tools/activity,
session detail, FTS search, dedup, the sync skip-cache, and the runtime
recorder (plane B).
"""

from __future__ import annotations

import pytest

from agent_utilities.usage.backends.sqlite_fts import SqliteUsageBackend
from agent_utilities.usage.models import (
    ORIGIN_RUNTIME,
    ParsedSessionBundle,
    UsageEvent,
    UsageMessage,
    UsageSession,
    UsageToolCall,
)
from agent_utilities.usage.recorder import UsageRecorder


@pytest.fixture()
def backend(tmp_path):
    b = SqliteUsageBackend(tmp_path / "usage.db")
    b.ensure_schema()
    return b


def _bundle(sid="s1", project="proj-a", agent="claude", model="claude-opus-4-8"):
    return ParsedSessionBundle(
        session=UsageSession(
            id=sid,
            project=project,
            agent=agent,
            started_at="2026-06-10T09:00:00Z",
            message_count=2,
            total_output_tokens=500,
            health_grade="A",
            outcome="success",
        ),
        messages=[
            UsageMessage(
                session_id=sid,
                ordinal=0,
                role="user",
                content="please refactor the authentication module",
            ),
            UsageMessage(
                session_id=sid,
                ordinal=1,
                role="assistant",
                content="done, edited auth.py",
                has_tool_use=True,
                output_tokens=500,
            ),
        ],
        tool_calls=[
            UsageToolCall(
                session_id=sid,
                message_ordinal=1,
                tool_name="Edit",
                category="edit",
                status="ok",
            ),
            UsageToolCall(
                session_id=sid,
                message_ordinal=1,
                tool_name="run_tests",
                category="skill",
                skill_name="automated-test-runner",
                status="error",
            ),
        ],
        usage_events=[
            UsageEvent(
                session_id=sid,
                source="agent",
                model=model,
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=200,
                dedup_key="e1",
            ),
        ],
    )


def test_write_and_summary(backend):
    # Direct write_bundle stores raw rows (no pricing — that's the recorder's job).
    b = backend
    b.write_bundle(_bundle())
    s = b.summary()
    assert s.session_count == 1
    assert s.totals.input_tokens == 1000
    assert s.totals.output_tokens == 500
    assert s.totals.cost_usd == 0.0  # unpriced via the low-level path
    assert s.cache_hit_rate == pytest.approx(200 / 200)  # only cache_read present


def test_recorder_prices_events(backend):
    rec = UsageRecorder(backend)
    assert rec.record_bundle(_bundle()) is True
    detail = backend.session_detail("s1")
    assert detail is not None
    assert detail.usage_events[0].cost_status == "catalog"
    # 1000 in @ $5 + 500 out @ $25 + 200 cache_read @ $0.50 (per Mtok)
    assert detail.usage_events[0].cost_usd == pytest.approx(0.0176, rel=1e-6)


def test_idempotent_reingest(backend):
    backend.write_bundle(_bundle())
    backend.write_bundle(_bundle())  # same id -> replace, not duplicate
    assert backend.summary().session_count == 1
    detail = backend.session_detail("s1")
    assert len(detail.messages) == 2
    assert len(detail.usage_events) == 1


def test_breakdowns(backend):
    backend.write_bundle(_bundle(sid="s1", project="proj-a", model="claude-opus-4-8"))
    backend.write_bundle(_bundle(sid="s2", project="proj-b", model="gpt-5.5"))
    by_model = {b.key: b for b in backend.breakdown("model")}
    assert set(by_model) == {"claude-opus-4-8", "gpt-5.5"}
    by_project = {b.key: b for b in backend.breakdown("project")}
    assert set(by_project) == {"proj-a", "proj-b"}


def test_tool_stats(backend):
    backend.write_bundle(_bundle())
    stats = {t.name: t for t in backend.tool_stats()}
    assert stats["Edit"].success_rate == 1.0
    assert stats["run_tests"].success_rate == 0.0  # status=error
    assert stats["run_tests"].category == "skill"


def test_activity_buckets(backend):
    backend.write_bundle(_bundle())
    cells = backend.activity()
    assert len(cells) == 1
    # 2026-06-10 is a Wednesday (weekday 2), hour 9
    assert cells[0].day_of_week == 2
    assert cells[0].hour == 9


def test_fts_search(backend):
    backend.write_bundle(_bundle())
    hits = backend.search("authentication")
    assert hits and hits[0].session_id == "s1"
    assert "[authentication]" in hits[0].snippet or "authentication" in hits[0].snippet
    assert backend.search("nonexistenttermxyz") == []


def test_skip_cache(backend):
    assert backend.should_sync("/x/a.jsonl", 100, 50) is True
    backend.mark_synced("/x/a.jsonl", 100, 50)
    assert backend.should_sync("/x/a.jsonl", 100, 50) is False
    assert backend.should_sync("/x/a.jsonl", 101, 50) is True  # mtime changed


def test_filters_by_project_and_origin(backend):
    backend.write_bundle(_bundle(sid="s1", project="proj-a"))
    backend.write_bundle(_bundle(sid="s2", project="proj-b"))
    assert backend.summary(project="proj-a").session_count == 1
    # runtime origin
    rec = UsageRecorder(backend)
    rec.record_run(
        run_id="run-1",
        model="claude-opus-4-8",
        token_usage={"input_tokens": 10, "output_tokens": 20},
    )
    assert backend.summary(origin=ORIGIN_RUNTIME).session_count == 1
    assert backend.summary().session_count == 3


def test_record_tool_call_runtime(backend):
    rec = UsageRecorder(backend)
    rec.record_run(
        run_id="run-1",
        model="claude-opus-4-8",
        token_usage={"input_tokens": 10, "output_tokens": 20},
    )
    rec.record_tool_call(
        session_id="run-1", tool_name="graph_query", category="db", status="ok"
    )
    stats = {t.name: t for t in backend.tool_stats(origin=ORIGIN_RUNTIME)}
    # tool_stats filters on session origin; runtime session has the call
    assert "graph_query" in stats
    assert stats["graph_query"].category == "db"
