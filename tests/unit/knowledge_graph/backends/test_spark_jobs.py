"""Unit tests for ``spark_jobs.py`` (CA-27).

Mocks ``call_tool_once`` at its source module
(``agent_utilities.protocols.source_connectors.connectors.mcp_tool``) --
``spark_jobs._call`` re-imports it fresh on every call, so patching the
attribute there is picked up without needing a real fastmcp/spark-mcp
connection.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.spark_jobs import (
    SPARK_LIST_RUNS_TOOL,
    SPARK_SUBMIT_TOOL,
    SparkJobError,
    SparkJobsClient,
    SparkPresetUnavailableError,
    submission_id_for,
)
from agent_utilities.protocols.source_connectors.connectors import mcp_tool


def _manifest(**overrides):
    manifest = {
        "transform": "agg_daily",
        "kind": "sql",
        "body": "SELECT 1",
        "inputs": [{"table": "lakehouse.analytics.raw", "as_of_version": "111"}],
        "output": {"table": "lakehouse.analytics.agg", "mode": "append"},
    }
    manifest.update(overrides)
    return manifest


def _patch_call_tool_once(monkeypatch, fake):
    async def _fake(**kwargs):
        return await fake(**kwargs)

    monkeypatch.setattr(mcp_tool, "call_tool_once", _fake)


# ---------------------------------------------------------------------------
# submission_id_for -- deterministic idempotency key
# ---------------------------------------------------------------------------


def test_submission_id_is_deterministic_for_identical_manifest():
    m1 = _manifest()
    m2 = _manifest()
    assert submission_id_for(m1) == submission_id_for(m2)


def test_submission_id_differs_for_different_manifest():
    assert submission_id_for(_manifest()) != submission_id_for(_manifest(body="SELECT 2"))


# ---------------------------------------------------------------------------
# submit_transform -- calls the REAL spark-mcp tool, idempotent retry
# ---------------------------------------------------------------------------


def test_submit_transform_calls_real_spark_mcp_tool(monkeypatch):
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return {
            "run_id": "run-1",
            "transform": "agg_daily",
            "status": "succeeded",
            "output_snapshot_id": "222",
            "output_table": "lakehouse.analytics.agg",
            "row_count": 10,
            "error": None,
        }

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    submission = client.submit_transform(_manifest())

    assert len(calls) == 1
    assert calls[0]["tool"] == SPARK_SUBMIT_TOOL
    assert calls[0]["server"] == "spark-mcp"
    assert calls[0]["params_style"] == "args"
    assert submission.run_id == "run-1"
    assert submission.status == "succeeded"
    assert submission.output_snapshot_id == "222"
    assert submission.input_snapshot_ids == ("111",)


def test_submit_transform_retry_is_a_no_op_idempotent(monkeypatch):
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return {
            "run_id": f"run-{len(calls)}",
            "transform": "agg_daily",
            "status": "succeeded",
            "output_snapshot_id": "222",
            "output_table": "lakehouse.analytics.agg",
            "row_count": 10,
            "error": None,
        }

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    manifest = _manifest()

    first = client.submit_transform(manifest)
    second = client.submit_transform(manifest)  # retry, same manifest

    assert len(calls) == 1  # the server was NOT called a second time
    assert first.run_id == second.run_id == "run-1"


def test_submit_transform_explicit_submission_id_dedupes(monkeypatch):
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return {"run_id": "run-x", "transform": "t", "status": "succeeded", "output_snapshot_id": "1"}

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    client.submit_transform(_manifest(), submission_id="fixed-id")
    client.submit_transform(_manifest(body="different but same submission_id"), submission_id="fixed-id")
    assert len(calls) == 1


def test_submit_transform_missing_tool_raises_named_error(monkeypatch):
    async def fake(**kwargs):
        raise mcp_tool.McpToolSourceError("MCP source tool call failed (ToolError)") from RuntimeError(
            "Unknown tool: spark_submit_transform"
        )

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    with pytest.raises(SparkPresetUnavailableError, match=SPARK_SUBMIT_TOOL):
        client.submit_transform(_manifest())


def test_submit_transform_generic_failure_raises_spark_job_error(monkeypatch):
    async def fake(**kwargs):
        raise mcp_tool.McpToolSourceError("MCP source tool call failed (ConnectionError)")

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    with pytest.raises(SparkJobError) as excinfo:
        client.submit_transform(_manifest())
    assert not isinstance(excinfo.value, SparkPresetUnavailableError)


# ---------------------------------------------------------------------------
# poll_status -- resolves by transform name (spark-mcp has no run_id filter)
# ---------------------------------------------------------------------------


def test_poll_status_uses_ledger_transform_name(monkeypatch):
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if kwargs["tool"] == SPARK_SUBMIT_TOOL:
            return {"run_id": "run-1", "transform": "agg_daily", "status": "running", "output_snapshot_id": None}
        assert kwargs["tool"] == SPARK_LIST_RUNS_TOOL
        assert kwargs["params"]["transform"] == "agg_daily"
        return {"runs": [{"run_id": "run-1", "status": "succeeded", "output_snapshot_id": "222"}]}

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    client.submit_transform(_manifest())
    status = client.poll_status("run-1")
    assert status["status"] == "succeeded"
    assert status["output_snapshot_id"] == "222"


def test_poll_status_unknown_run_without_transform_name_fails_loudly():
    client = SparkJobsClient()
    with pytest.raises(SparkJobError, match="transform"):
        client.poll_status("never-submitted-run-id")


def test_poll_status_explicit_transform_name_for_cold_ledger(monkeypatch):
    async def fake(**kwargs):
        return {"runs": [{"run_id": "run-9", "status": "failed", "output_snapshot_id": None}]}

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()  # fresh process, empty ledger
    status = client.poll_status("run-9", transform="agg_daily")
    assert status["status"] == "failed"


# ---------------------------------------------------------------------------
# fence() / build_envelope() -- R5, only for a succeeded run
# ---------------------------------------------------------------------------


def test_fence_refuses_a_failed_submission(monkeypatch):
    async def fake(**kwargs):
        return {"run_id": "run-1", "transform": "agg_daily", "status": "failed", "error": "boom", "output_snapshot_id": None}

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    submission = client.submit_transform(_manifest())
    with pytest.raises(SparkJobError, match="status"):
        client.fence(submission, code_version="agg_daily@v1")


def test_fence_and_build_envelope_carry_all_four_fence_fields(monkeypatch):
    async def fake(**kwargs):
        return {
            "run_id": "run-1",
            "transform": "agg_daily",
            "status": "succeeded",
            "output_snapshot_id": "222",
            "output_table": "lakehouse.analytics.agg",
            "row_count": 10,
        }

    _patch_call_tool_once(monkeypatch, fake)
    client = SparkJobsClient()
    submission = client.submit_transform(_manifest())
    envelope = client.build_envelope(submission, code_version="agg_daily@v1")

    assert envelope.provenance["run_id"] == "run-1"
    assert envelope.provenance["input_snapshot_ids"] == ["111"]
    assert envelope.provenance["code_version"] == "agg_daily@v1"
    assert envelope.confidence == 1.0
    assert envelope.source_object_id == "lakehouse.analytics.agg"
