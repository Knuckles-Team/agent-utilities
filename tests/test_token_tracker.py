"""Tests for TokenUsageTracker — 4-Bucket Analytics (CONCEPT:OS-5.1).

@pytest.mark.concept("OS-5.5")
"""

import pytest

from agent_utilities.observability.token_tracker import (
    TokenUsageRecord,
    TokenUsageSummary,
    TokenUsageTracker,
)


@pytest.fixture
def tracker() -> TokenUsageTracker:
    return TokenUsageTracker()


@pytest.fixture
def sample_record() -> TokenUsageRecord:
    return TokenUsageRecord(
        agent_name="test_agent",
        model_name="gpt-4o",
        session_id="sess-1",
        user_id="user-1",
        prompt_tokens=100,
        response_tokens=50,
        thoughts_tokens=20,
        tool_use_tokens=10,
    )


# ---------------------------------------------------------------------------
# TokenUsageRecord model
# ---------------------------------------------------------------------------


class TestTokenUsageRecord:
    def test_auto_total_computation(self):
        r = TokenUsageRecord(
            prompt_tokens=100,
            response_tokens=50,
            thoughts_tokens=20,
            tool_use_tokens=10,
        )
        assert r.total_tokens == 180

    def test_explicit_total_preserved(self):
        r = TokenUsageRecord(
            prompt_tokens=100,
            response_tokens=50,
            total_tokens=999,  # Explicitly set
        )
        assert r.total_tokens == 999

    def test_zero_buckets(self):
        r = TokenUsageRecord()
        assert r.total_tokens == 0

    def test_timestamp_auto(self):
        r = TokenUsageRecord()
        assert r.timestamp > 0


# ---------------------------------------------------------------------------
# TokenUsageSummary
# ---------------------------------------------------------------------------


class TestTokenUsageSummary:
    def test_compute_total(self):
        s = TokenUsageSummary(
            total_prompt_tokens=100,
            total_response_tokens=50,
            total_thoughts_tokens=20,
            total_tool_use_tokens=10,
        )
        total = s.compute_total()
        assert total == 180
        assert s.total_tokens == 180


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_record_basic(self, tracker, sample_record):
        result = tracker.record(sample_record)
        assert result.total_tokens == 180
        assert result.id != ""

    def test_record_generates_id(self, tracker):
        r = TokenUsageRecord(agent_name="a", prompt_tokens=10)
        result = tracker.record(r)
        assert result.id.startswith("token:a:")

    def test_auto_total_on_record(self, tracker):
        r = TokenUsageRecord(
            prompt_tokens=50,
            response_tokens=30,
            thoughts_tokens=10,
            tool_use_tokens=5,
        )
        result = tracker.record(r)
        assert result.total_tokens == 95


# ---------------------------------------------------------------------------
# Session totals
# ---------------------------------------------------------------------------


class TestSessionTotals:
    def test_single_session(self, tracker):
        tracker.record(
            TokenUsageRecord(
                session_id="s1",
                prompt_tokens=100,
                response_tokens=50,
            )
        )
        tracker.record(
            TokenUsageRecord(
                session_id="s1",
                prompt_tokens=200,
                response_tokens=100,
            )
        )
        totals = tracker.get_session_totals("s1")
        assert totals.call_count == 2
        assert totals.total_prompt_tokens == 300
        assert totals.total_response_tokens == 150
        assert totals.total_tokens == 450

    def test_unknown_session(self, tracker):
        totals = tracker.get_session_totals("nonexistent")
        assert totals.call_count == 0
        assert totals.total_tokens == 0

    def test_cross_session_isolation(self, tracker):
        tracker.record(TokenUsageRecord(session_id="s1", prompt_tokens=100))
        tracker.record(TokenUsageRecord(session_id="s2", prompt_tokens=200))
        assert tracker.get_session_totals("s1").total_prompt_tokens == 100
        assert tracker.get_session_totals("s2").total_prompt_tokens == 200


# ---------------------------------------------------------------------------
# Agent breakdown
# ---------------------------------------------------------------------------


class TestAgentBreakdown:
    def test_breakdown_per_bucket(self, tracker):
        tracker.record(
            TokenUsageRecord(
                agent_name="agent_a",
                prompt_tokens=100,
                response_tokens=50,
                thoughts_tokens=20,
                tool_use_tokens=10,
            )
        )
        breakdown = tracker.get_agent_breakdown("agent_a")
        assert breakdown["call_count"] == 1
        assert breakdown["total_prompt_tokens"] == 100
        assert breakdown["total_response_tokens"] == 50
        assert breakdown["total_thoughts_tokens"] == 20
        assert breakdown["total_tool_use_tokens"] == 10

    def test_unknown_agent(self, tracker):
        breakdown = tracker.get_agent_breakdown("nobody")
        assert breakdown["call_count"] == 0


# ---------------------------------------------------------------------------
# Budget alerts
# ---------------------------------------------------------------------------


class TestBudgetAlerts:
    def test_no_alert_under_threshold(self, tracker):
        tracker.record(TokenUsageRecord(prompt_tokens=100))
        alerts = tracker.get_budget_alerts({"prompt": 1000})
        assert len(alerts) == 0

    def test_warning_at_80_percent(self, tracker):
        tracker.record(TokenUsageRecord(prompt_tokens=850, session_id="s1"))
        alerts = tracker.get_budget_alerts(
            {"prompt": 1000},
            session_id="s1",
        )
        assert len(alerts) >= 1
        assert alerts[0].severity == "warning"
        assert alerts[0].percentage >= 80

    def test_critical_at_100_percent(self, tracker):
        tracker.record(TokenUsageRecord(prompt_tokens=1100, session_id="s1"))
        alerts = tracker.get_budget_alerts(
            {"prompt": 1000},
            session_id="s1",
        )
        assert any(a.severity == "critical" for a in alerts)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_structure(self, tracker, sample_record):
        tracker.record(sample_record)
        export = tracker.export_summary()
        assert "overall" in export
        assert "by_agent" in export
        assert export["overall"]["call_count"] == 1
        assert "test_agent" in export["by_agent"]

    def test_export_empty(self, tracker):
        export = tracker.export_summary()
        assert export["overall"]["call_count"] == 0


# ---------------------------------------------------------------------------
# LLM response adapter
# ---------------------------------------------------------------------------


class TestLLMResponseAdapter:
    def test_from_usage_metadata(self, tracker):
        class FakeUsage:
            request_tokens = 100
            response_tokens = 50

        record = tracker.record_from_llm_response(
            FakeUsage(),
            agent_name="a",
            session_id="s",
        )
        assert record is not None
        assert record.prompt_tokens == 100
        assert record.response_tokens == 50

    def test_none_metadata(self, tracker):
        assert tracker.record_from_llm_response(None) is None
