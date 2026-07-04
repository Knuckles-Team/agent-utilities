"""Failure-read surface on the trace backends (CONCEPT:AU-AHE.harness.failure-evolution).

Verifies LangfuseTraceBackend wires the new read methods onto the right
LangfuseApi calls/params, and that FileTraceBackend serves the same surface
from JSON fixtures for offline runs.

@pytest.mark.concept("AU-AHE.harness.failure-evolution")
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.harness.trace_backend import (
    FileTraceBackend,
    LangfuseTraceBackend,
)

pytestmark = pytest.mark.concept("AU-AHE.harness.failure-evolution")


class _FakeApi:
    """Records the kwargs each Langfuse endpoint was called with."""

    def __init__(self):
        self.calls = {}

    def legacy_observations_v1_get_many(self, **kwargs):
        self.calls["legacy_observations_v1_get_many"] = kwargs
        return {
            "data": [{"id": "o1", "traceId": "t1", "name": "loop", "level": "ERROR"}]
        }

    def scores_get_many(self, **kwargs):
        self.calls["scores_get_many"] = kwargs
        return {"data": [{"traceId": "t2", "name": "accuracy", "value": 0.2}]}

    def legacy_metrics_v1_metrics(self, query):
        self.calls["legacy_metrics_v1_metrics"] = query
        return {
            "data": [
                {
                    "name": "loop",
                    "p95_latency": 9000,
                    "sum_totalCost": 1.5,
                    "count_count": 10,
                }
            ]
        }


def _backend_with_fake():
    b = LangfuseTraceBackend()
    b._api = _FakeApi()  # bypass _get_api credential check
    return b


class TestLangfuseFailureReads:
    def test_error_observations_uses_legacy_endpoint_with_level(self):
        b = _backend_with_fake()
        rows = asyncio.run(b.get_error_observations(since="2026-01-01T00:00:00Z"))
        call = b._api.calls["legacy_observations_v1_get_many"]
        assert call["level"] == "ERROR"
        assert call["from_start_time"] == "2026-01-01T00:00:00Z"
        assert rows and rows[0]["traceId"] == "t1"

    def test_low_score_traces_filters_below_threshold(self):
        b = _backend_with_fake()
        rows = asyncio.run(b.get_low_score_traces(max_value=0.5))
        call = b._api.calls["scores_get_many"]
        assert call["operator"] == "<"
        assert call["value"] == 0.5
        assert rows[0]["trace_id"] == "t2"
        assert rows[0]["value"] == 0.2

    def test_cost_latency_anomalies_flags_over_budget(self):
        b = _backend_with_fake()
        rows = asyncio.run(
            b.get_cost_latency_anomalies(p95_latency_ms=1000, p95_cost_usd=1.0)
        )
        # query is a JSON string with the observations view
        q = json.loads(b._api.calls["legacy_metrics_v1_metrics"])
        assert q["view"] == "observations"
        assert q["toTimestamp"]  # both timestamps required by the metrics API
        assert rows and rows[0]["over_latency"] is True
        assert rows[0]["over_cost"] is True
        assert rows[0]["p95_latency_ms"] == 9000.0

    def test_api_failure_degrades_to_empty(self):
        class _Boom:
            def legacy_observations_v1_get_many(self, **k):
                raise RuntimeError("down")

        b = LangfuseTraceBackend()
        b._api = _Boom()
        assert asyncio.run(b.get_error_observations()) == []


class TestFileBackendFixtures:
    def test_reads_failure_fixtures_offline(self, tmp_path):
        (tmp_path / "error_observations.json").write_text(
            json.dumps([{"traceId": "t1", "name": "loop", "statusMessage": "boom"}])
        )
        (tmp_path / "low_scores.json").write_text(
            json.dumps([{"trace_id": "t2", "name": "acc", "value": 0.1}])
        )
        b = FileTraceBackend(trace_dir=str(tmp_path))
        errs = asyncio.run(b.get_error_observations())
        lows = asyncio.run(b.get_low_score_traces(max_value=0.5))
        assert errs[0]["name"] == "loop"
        assert lows[0]["value"] == 0.1

    def test_missing_fixtures_return_empty(self, tmp_path):
        b = FileTraceBackend(trace_dir=str(tmp_path))
        assert asyncio.run(b.get_error_observations()) == []
        assert asyncio.run(b.get_cost_latency_anomalies()) == []
