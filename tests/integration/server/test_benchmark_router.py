"""CONCEPT:AU-AHE.evaluation.longmemeval-validation-harness — LongMemEval-S harness.

Unit-tests the pure scoring helpers and confirms the /benchmark router mounts and serves
health + report endpoints via a minimal FastAPI app (no live engine required).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.server.routers.benchmark import (
    aggregate_report,
    judge_binary,
    normalize_answer,
    router,
)

# ── pure scoring ────────────────────────────────────────────────────────────────


@pytest.mark.concept(id="AU-AHE.evaluation.longmemeval-validation-harness")
def test_normalize_answer_strips_articles_and_punct():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


@pytest.mark.concept(id="AU-AHE.evaluation.longmemeval-validation-harness")
def test_judge_binary_substring_and_numeric():
    assert judge_binary("It is Paris.", "Paris") is True
    assert judge_binary("the monitor cost $40", "40") is True
    assert judge_binary("I don't know", "Tokyo") is False
    assert judge_binary("anything", "") is False  # empty gold never auto-correct


@pytest.mark.concept(id="AU-AHE.evaluation.longmemeval-validation-harness")
def test_aggregate_report_accuracy_and_categories():
    rows = [
        {"correct": True, "question_type": "single-session"},
        {"correct": False, "question_type": "single-session"},
        {"correct": True, "question_type": "multi-session"},
    ]
    rep = aggregate_report(rows)
    assert rep["total"] == 3 and rep["correct"] == 2
    assert abs(rep["accuracy"] - 2 / 3) < 1e-9
    assert rep["by_category"]["single-session"]["accuracy"] == 0.5
    assert rep["by_category"]["multi-session"]["accuracy"] == 1.0


# ── router mounting ─────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.concept(id="AU-AHE.evaluation.longmemeval-validation-harness")
def test_health_endpoint(client: TestClient):
    resp = client.get("/benchmark/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["concept"] == "AU-AHE.evaluation.longmemeval-validation-harness"


@pytest.mark.concept(id="AU-AHE.evaluation.longmemeval-validation-harness")
def test_report_endpoint_empty_run(client: TestClient):
    resp = client.get("/benchmark/report/never-seen")
    assert resp.status_code == 200
    assert resp.json() == {"total": 0, "correct": 0, "accuracy": 0.0, "by_category": {}}
