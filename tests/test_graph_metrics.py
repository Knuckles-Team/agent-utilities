"""Measured model-request evidence from the real run loop, without timing sleeps."""

from __future__ import annotations

import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from agent_utilities.graph import RunPlan, RunRequest, run
from agent_utilities.graph.metrics import MAX_REQUEST_SAMPLES, RunMetrics


# spec: AU-BOUNDARY-R043.12
async def test_actual_model_calls_emit_attributed_counts_and_client_time(events):
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[TextPart(f"answer:{calls}")],
            usage=RequestUsage(input_tokens=7, output_tokens=3),
            model_name="fixture-model",
            provider_name="fixture-provider",
            provider_response_id=f"response:{calls}",
            provider_details={"latency_ms": 1, "total_time": 0.1},
        )

    result = await run(
        RunRequest(task="test", run_id="run:metrics"),
        plan=RunPlan(model=FunctionModel(respond), steps=("one", "two")),
    )
    assert result.status == "completed"
    metrics = result.metrics
    assert metrics["coverage"] == "complete"
    assert metrics["observed_model_requests"] == calls == 2
    assert metrics["run_elapsed_ms"] >= 0
    requests = metrics["model_requests"]
    assert [r["sequence"] for r in requests] == [1, 2]
    assert [r["provider_response_id"] for r in requests] == ["response:1", "response:2"]
    for request in requests:
        assert request["source"] == "client_monotonic_clock"
        assert request["client_request_elapsed_ms"] >= 0
        assert request["input_tokens"] == 7
        assert request["output_tokens"] == 3
        assert request["outcome"] == "completed"
        assert request["provider_name"] == "fixture-provider"
        assert request["model_name"] == "function:respond:"
    assert events[-1]["result"]["metrics"] == metrics
    serialized = json.dumps(metrics)
    for absent in ("generation", "tps", "queue", "ttft", "latency_ms", "total_time"):
        assert absent not in serialized


# spec: AU-BOUNDARY-R043.12
async def test_failed_model_call_is_observed_without_invented_usage():
    def fail(messages, info):
        raise RuntimeError("model failed")

    result = await run(
        RunRequest(task="test"), plan=RunPlan(model=FunctionModel(fail), steps=("one",))
    )
    assert result.status == "failed"
    assert result.metrics["observed_model_requests"] == 1
    sample = result.metrics["model_requests"][0]
    assert sample["outcome"] == "failed"
    assert sample["client_request_elapsed_ms"] >= 0
    assert "input_tokens" not in sample
    assert "output_tokens" not in sample
    assert "model failed" not in json.dumps(result.metrics)


# spec: AU-BOUNDARY-R043.12
@pytest.mark.parametrize("bad_count", [-1, True, 1.25, 9_007_199_254_740_992])
async def test_invalid_response_counts_do_not_become_metric_values(bad_count):
    def respond(messages, info):
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=bad_count, output_tokens=2),
        )

    result = await run(
        RunRequest(task="test"),
        plan=RunPlan(model=FunctionModel(respond), steps=("one",)),
    )
    sample = result.metrics["model_requests"][0]
    assert "input_tokens" not in sample
    assert sample["output_tokens"] == 2


# spec: AU-BOUNDARY-R043.12
async def test_missing_model_has_zero_observed_requests():
    result = await run(RunRequest(task="test"))
    assert result.status == "failed"
    assert result.metrics["model_requests"] == []
    assert result.metrics["observed_model_requests"] == 0


# spec: AU-BOUNDARY-R043.12
async def test_zero_elapsed_is_not_used_as_a_generation_denominator(monkeypatch):
    import agent_utilities.graph.metrics as metrics

    monkeypatch.setattr(metrics, "perf_counter_ns", lambda: 0)
    result = await run(
        RunRequest(task="test"),
        plan=RunPlan(
            model=FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart("ok")])
            ),
            steps=("one",),
        ),
    )
    assert result.metrics["model_requests"][0]["client_request_elapsed_ms"] == 0
    assert "tps" not in json.dumps(result.metrics)


# spec: AU-BOUNDARY-R043.12
def test_sample_limit_reports_partial_coverage():
    from types import SimpleNamespace

    metrics = RunMetrics()
    model = SimpleNamespace(model_name="m", system="p")
    for _ in range(MAX_REQUEST_SAMPLES + 1):
        metrics.record(metrics.started_ns, None, model=model)
    snapshot = metrics.snapshot()
    assert snapshot["coverage"] == "partial"
    assert snapshot["observed_model_requests"] == MAX_REQUEST_SAMPLES + 1
    assert len(snapshot["model_requests"]) == MAX_REQUEST_SAMPLES
