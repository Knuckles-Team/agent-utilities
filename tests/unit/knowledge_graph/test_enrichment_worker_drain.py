"""EH-557 native WorkItem drain fences and bounds model-backed enrichment."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from agent_utilities.knowledge_graph import ingest_worker
from agent_utilities.knowledge_graph.core import work_durability as wi


def _row(*, units: int = 3) -> dict[str, Any]:
    return {
        "id": "wi:enrich-1",
        "tenant": "tenant-a",
        "kind": "enrichment.classical",
        "queue": "enrichment.classical",
        "payload_ref": "eg:source:blob-1",
        "policy_digest": "a" * 64,
        "catalog_digest": "b" * 64,
        "model_digest": "c" * 64,
        "metadata": {
            "content_digest": "d" * 64,
            "parser_capability_digest": "e" * 64,
            "reserved_compute_units": units,
        },
    }


def _wire(monkeypatch, row: dict[str, Any]):
    claims = [{"work_item_id": "wi:enrich-1", "tenant": "tenant-a"}]
    queues: list[str] = []
    commits: list[dict[str, Any]] = []
    deferred: list[str] = []

    def claim_next(engine, **kwargs):
        queues.append(kwargs["queue"])
        if kwargs["queue"] == "enrichment.classical" and claims:
            return claims.pop()
        return None

    def commit_result(engine, item_id, claim, **kwargs):
        commits.append(kwargs)
        return "committed"

    monkeypatch.setattr(wi, "claim_next", claim_next)
    monkeypatch.setattr(wi, "get_work_item", lambda engine, item_id: row)
    monkeypatch.setattr(wi, "mark_running", lambda *args, **kwargs: True)
    monkeypatch.setattr(wi, "heartbeat", lambda *args, **kwargs: True)
    monkeypatch.setattr(wi, "commit_result", commit_result)
    monkeypatch.setattr(
        wi,
        "defer_work_item",
        lambda engine, item_id, claim, **kwargs: deferred.append(kwargs["reason_ref"]) or True,
    )
    return queues, commits, deferred


def test_bounded_drain_executes_only_claimed_enrichment_queue(monkeypatch) -> None:
    queues, commits, _ = _wire(monkeypatch, _row())
    observed: list[ingest_worker.EnrichmentTask] = []

    def execute(task, renew):
        observed.append(task)
        assert renew()
        return ingest_worker.EnrichmentExecution("eg:result:1", 2)

    result = ingest_worker.drain_enrichment_work_items(
        object(), tenant="tenant-a", token="worker:1", execute=execute,
        max_items=2, max_compute_units=5,
    )
    assert result.claimed == result.completed == 1
    assert result.reserved_compute_units == 3
    assert result.spent_compute_units == 2
    assert observed[0].input_ref == "eg:source:blob-1"
    assert commits == [{"outcome": "succeeded", "result_ref": "eg:result:1"}]
    assert all(queue.startswith("enrichment.") for queue in queues)


def test_drain_defers_over_budget_before_execution(monkeypatch) -> None:
    _, commits, deferred = _wire(monkeypatch, _row(units=9))

    def execute(task, renew):
        pytest.fail("over-budget work must not execute")

    result = ingest_worker.drain_enrichment_work_items(
        object(), tenant="tenant-a", token="worker:1", execute=execute,
        max_items=2, max_compute_units=5,
    )
    assert result.claimed == result.deferred == 1
    assert commits == []
    assert deferred == ["enrichment:drain_budget"]


def test_drain_fails_closed_on_unpinned_admission(monkeypatch) -> None:
    row = _row()
    row["policy_digest"] = None
    _, commits, _ = _wire(monkeypatch, row)

    def execute(task, renew):
        pytest.fail("unverified admission must not execute")

    result = ingest_worker.drain_enrichment_work_items(
        object(), tenant="tenant-a", token="worker:1", execute=execute,
        max_items=1, max_compute_units=5,
    )
    assert result.failed == 1
    assert commits == [{
        "outcome": "failed",
        "error_ref": "enrichment:invalid_admission",
        "retryable": False,
    }]


def test_drain_retries_executor_failure_without_leaking_content(monkeypatch) -> None:
    _, commits, _ = _wire(monkeypatch, _row())

    def execute(task, renew):
        raise RuntimeError("secret source content")

    result = ingest_worker.drain_enrichment_work_items(
        object(), tenant="tenant-a", token="worker:1", execute=execute,
        max_items=1, max_compute_units=5,
    )
    assert result.failed == 1
    assert commits == [{
        "outcome": "failed",
        "error_ref": "enrichment:execution_failed",
        "retryable": True,
    }]
    assert result.reserved_compute_units == 3


def test_worker_loop_honors_stop_before_native_claim(monkeypatch) -> None:
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(
        ingest_worker,
        "drain_enrichment_work_items",
        lambda *args, **kwargs: pytest.fail("stopped worker must not claim"),
    )
    ingest_worker.run_enrichment_worker_loop(
        object(), stop, tenant="tenant-a", token="worker:1",
        execute=lambda task, renew: ingest_worker.EnrichmentExecution("eg:result:1", 1),
    )
