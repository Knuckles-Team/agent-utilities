"""Runtime instrumentation (plane B) tests (CONCEPT:OS-5.31)."""

from __future__ import annotations

import pytest

from agent_utilities.usage.backends.sqlite_fts import SqliteUsageBackend
from agent_utilities.usage.models import ORIGIN_RUNTIME
from agent_utilities.usage.recorder import UsageRecorder


@pytest.fixture()
def backend(tmp_path):
    b = SqliteUsageBackend(tmp_path / "u.db")
    b.ensure_schema()
    return b


def test_record_run_creates_runtime_session(backend):
    rec = UsageRecorder(backend)
    ok = rec.record_run(
        run_id="run-xyz",
        query="do a thing",
        status="success",
        duration_ms=1234.0,
        model="claude-opus-4-8",
        token_usage={
            "input_tokens": 1000,
            "output_tokens": 200,
            "reasoning_tokens": 50,
        },
    )
    assert ok is True
    s = backend.summary(origin=ORIGIN_RUNTIME)
    assert s.session_count == 1
    assert s.totals.input_tokens == 1000
    assert s.totals.cost_usd > 0  # priced via catalog
    detail = backend.session_detail("run-xyz")
    assert detail.session.origin == ORIGIN_RUNTIME
    assert detail.usage_events[0].reasoning_tokens == 50


def test_record_run_disabled(monkeypatch, backend):
    monkeypatch.setenv("USAGE_TRACKING_ENABLED", "false")
    from agent_utilities.core.config import config

    config.reload()
    rec = UsageRecorder(backend)
    assert rec.record_run(run_id="r2", model="x") is False
    assert backend.summary().session_count == 0
    monkeypatch.setenv("USAGE_TRACKING_ENABLED", "true")
    config.reload()


def test_tool_call_metrics_and_rows(backend):
    rec = UsageRecorder(backend)
    rec.record_run(
        run_id="run-1",
        model="claude-opus-4-8",
        token_usage={"input_tokens": 10, "output_tokens": 5},
    )
    rec.record_tool_call(
        session_id="run-1", tool_name="graph_query", category="db", status="ok"
    )
    rec.record_tool_call(
        session_id="run-1",
        tool_name="run-tests",
        category="skill",
        skill_name="automated-test-runner",
    )
    detail = backend.session_detail("run-1")
    cats = {t.category for t in detail.tool_calls}
    assert cats == {"db", "skill"}
    skills = [t.skill_name for t in detail.tool_calls if t.category == "skill"]
    assert skills == ["automated-test-runner"]


def test_metrics_module_exposes_counters():
    # The three OS-5.31 counters exist and are labelable (no-op or real).
    from agent_utilities.observability import gateway_metrics as gm

    gm.TOOL_CALLS.labels(category="db").inc()
    gm.SKILL_CALLS.labels(skill="x").inc()
    gm.DB_CALLS.labels(store="usage").inc()
