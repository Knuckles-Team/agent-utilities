"""Tests for the ``run_summary`` transparency surface on ``agent_runner.run_agent``
(CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency).

Covers the pure helpers directly (``_extract_failure_text``, ``_build_run_summary``), the
opt-in envelope contract on ``_render_agent_result``, and an end-to-end run through
``run_agent`` reproducing the reported bug: a github-mcp fleet-gate HTTPS failure must
surface as a ``degraded`` outcome whose ``run_summary.failure`` carries the REAL cause (not
the old hardcoded "delegation produced no usable data" sentinel), with a resolvable
``trace_ref``. Mirrors the mocking convention established in
``tests/unit/test_agent_stack_seams.py``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.orchestration import agent_runner
from agent_utilities.orchestration.agent_runner import (
    _build_run_summary,
    _extract_failure_text,
    _render_agent_result,
)

# ── _extract_failure_text ──────────────────────────────────────────────────────────────


def test_extract_failure_text_prefers_error_field() -> None:
    result = {"error": "RuntimeError: boom", "results": {"output": "unrelated text"}}
    assert _extract_failure_text(result) == "RuntimeError: boom"


def test_extract_failure_text_falls_back_to_metadata_error() -> None:
    result = {"metadata": {"error": "graph terminated at error_recovery"}}
    assert _extract_failure_text(result) == "graph terminated at error_recovery"


def test_extract_failure_text_falls_back_to_results_output() -> None:
    """_fleet_server_failed_result's composed message lives in results.output — this is the
    REAL cause the old hardcoded sentinel used to discard."""
    result = {
        "status": "failed",
        "results": {
            "output": (
                "Delegation to fleet server 'github-mcp' could not produce a "
                "tool-grounded result (RuntimeError: fleet MCP endpoint requires "
                "HTTPS outside loopback)."
            )
        },
        "metadata": {"degraded": True, "outcome": "fleet_server_failed"},
    }
    text = _extract_failure_text(result)
    assert "requires HTTPS outside loopback" in text


def test_extract_failure_text_never_empty_even_with_no_signal() -> None:
    junk_values: tuple[object, ...] = (
        {},
        {"results": {}},
        {"metadata": {}},
        None,
        "",
        42,
    )
    for junk in junk_values:
        text = _extract_failure_text(junk)
        assert isinstance(text, str) and text.strip()


def test_extract_failure_text_bare_string_result() -> None:
    assert _extract_failure_text("plain string output") == "plain string output"


# ── _build_run_summary ─────────────────────────────────────────────────────────────────


def test_build_run_summary_ok_outcome_has_no_failure_key() -> None:
    summary = _build_run_summary(
        route={"agents": [], "servers": ["portainer-mcp"], "why": "x"},
        outcome="ok",
        stage_reached="tool-call: portainer-mcp",
        run_id="run:" + "a" * 32,
        raw_failure=None,
    )
    assert summary["outcome"] == "ok"
    assert "failure" not in summary
    assert summary["route"]["servers"] == ["portainer-mcp"]
    assert summary["stage_reached"] == "tool-call: portainer-mcp"
    assert summary["trace_ref"].startswith("trace:")
    assert summary["execution_mode"] == "other"


def test_build_run_summary_degraded_outcome_carries_translated_failure() -> None:
    summary = _build_run_summary(
        route={"agents": [], "servers": ["github-mcp"], "why": "lexical gate"},
        outcome="degraded",
        stage_reached="tool-call: github-mcp",
        run_id="run:" + "b" * 32,
        raw_failure="RuntimeError: fleet MCP endpoint requires HTTPS outside loopback",
    )
    assert summary["outcome"] == "degraded"
    assert summary["execution_mode"] == "other"
    assert summary["failure"]["category"] == "fleet_https_gate"
    assert (
        "HTTPS" in summary["failure"]["translated"]
        or "TLS" in summary["failure"]["translated"]
    )
    assert summary["failure"]["hint"]
    assert summary["failure"]["raw"]


def test_build_run_summary_trace_ref_is_stable_for_the_same_run_id() -> None:
    a = _build_run_summary(
        route={},
        outcome="ok",
        stage_reached="x",
        run_id="run:" + "c" * 32,
        raw_failure=None,
    )
    b = _build_run_summary(
        route={},
        outcome="ok",
        stage_reached="x",
        run_id="run:" + "c" * 32,
        raw_failure=None,
    )
    assert a["trace_ref"] == b["trace_ref"]


# ── _render_agent_result: run_summary is opt-in and additive ──────────────────────────


def test_render_agent_result_bare_string_contract_unaffected_by_default() -> None:
    """No caller flags set -> the exact bare-string contract, unchanged."""
    out = _render_agent_result("hello", run_id="run:" + "d" * 32, return_mermaid=False)
    assert out == "hello"


def test_render_agent_result_run_summary_forces_the_envelope() -> None:
    summary = {
        "outcome": "degraded",
        "route": {},
        "stage_reached": "x",
        "trace_ref": "trace:y",
    }
    out = _render_agent_result(
        "hello",
        run_id="run:" + "e" * 32,
        return_mermaid=False,
        run_summary=summary,
    )
    payload = json.loads(out)
    assert payload["output"] == "hello"
    assert payload["run_summary"] == summary
    assert "channel_id" not in payload
    assert "mermaid" not in payload


def test_render_agent_result_run_summary_none_keeps_bare_string_even_with_channel_id_absent() -> (
    None
):
    out = _render_agent_result(
        "hello", run_id="run:" + "f" * 32, return_mermaid=False, run_summary=None
    )
    assert out == "hello"


# ── End-to-end through run_agent: the reported github-mcp fleet-gate scenario ──────────


@pytest.mark.asyncio
async def test_run_agent_fleet_gate_failure_produces_a_transparent_run_summary() -> (
    None
):
    """Reproduces the reported bug's ORCH-1.74 branch: a focused-tools delegation to
    github-mcp fails the fleet HTTPS gate. run_agent must surface a `degraded` run_summary
    whose failure.raw/translated carry the REAL cause (not the old hardcoded generic
    sentinel), with a resolvable trace_ref — end to end, through the real dispatch/Step-5
    code (only the KG-engine boundary and the network tool call are mocked)."""
    from types import SimpleNamespace

    # The KG engine boundary is entirely synchronous (every backend/registry call in
    # this branch is offloaded via ``asyncio.to_thread`` / ``_call_without_blocking``,
    # never awaited directly) — an AsyncMock here made every auto-vivified attribute
    # (e.g. FeedbackService.from_engine's ``engine.store`` fallback) a coroutine whose
    # synchronous call site never awaits it, leaking an "AsyncMock ... was never
    # awaited" RuntimeWarning. MagicMock matches the real, synchronous contract.
    fake_engine = MagicMock()
    fake_engine.backend = None

    shape = SimpleNamespace(
        tool_servers=("github-mcp",),
        resolve_agent=False,
        direct_complete=False,
    )

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch(
            "agent_utilities.orchestration.execution_profile.plan_execution_shape",
            return_value=shape,
        ),
        patch.object(
            agent_runner, "_build_execution_config", return_value={"mcp_toolsets": []}
        ),
        patch.object(
            agent_runner,
            "_execute_focused_tools",
            new=AsyncMock(
                side_effect=RuntimeError(
                    "fleet MCP endpoint requires HTTPS outside loopback"
                )
            ),
        ),
        patch.object(agent_runner, "_record_execution_trace") as mock_trace,
        patch.object(agent_runner, "_write_step_credit"),
    ):
        raw = await agent_runner.run_agent(
            agent_name="messaging-assistant",
            task="does my github org have issues/PRs",
            include_run_summary=True,
            run_id="run:" + "1" * 32,
        )

    payload = json.loads(raw)
    assert "could not produce a tool-grounded result" in payload["output"]

    summary = payload["run_summary"]
    assert summary["outcome"] == "degraded"
    assert summary["execution_mode"] == "single_server_agent"
    assert summary["stage_reached"] == "tool-call: github-mcp"
    assert summary["route"]["servers"] == ["github-mcp"]
    assert summary["trace_ref"] == "trace:" + __import__(
        "agent_utilities.security.persistence_privacy",
        fromlist=["persistence_reference"],
    ).persistence_reference("run", "run:" + "1" * 32, namespace="trace")

    failure = summary["failure"]
    assert failure["category"] == "fleet_https_gate"
    assert "HTTPS" in failure["translated"] or "TLS" in failure["translated"]
    assert "requires HTTPS outside loopback" in failure["raw"]

    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — the durable RunTrace's
    # error= must carry the REAL cause too, not the old hardcoded
    # "delegation produced no usable data (degraded)" sentinel.
    _, kwargs = mock_trace.call_args
    assert kwargs["status"] == "degraded"
    assert "requires HTTPS outside loopback" in kwargs["error"]
    assert kwargs["error"] != "delegation produced no usable data (degraded)"


@pytest.mark.asyncio
async def test_explicit_server_pin_cannot_be_rebound_and_requires_tool_provenance() -> (
    None
):
    """A caller pin wins over a task lexical match for another fleet server.

    This is the live-shaped regression: a github-mcp request with a repository-manager
    lexical shape must bind github-mcp, and a fenced pseudo-tool call without an actual
    executor ``tool_calls`` record is a degraded result rather than a success.
    """
    from types import SimpleNamespace

    # The KG engine boundary is entirely synchronous (every backend/registry call in
    # this branch is offloaded via ``asyncio.to_thread`` / ``_call_without_blocking``,
    # never awaited directly) — an AsyncMock here made every auto-vivified attribute
    # (e.g. FeedbackService.from_engine's ``engine.store`` fallback) a coroutine whose
    # synchronous call site never awaits it, leaking an "AsyncMock ... was never
    # awaited" RuntimeWarning. MagicMock matches the real, synchronous contract.
    fake_engine = MagicMock()
    fake_engine.backend = None
    shape = SimpleNamespace(
        tool_servers=("repository-manager-mcp",),
        resolve_agent=False,
        direct_complete=False,
    )

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch(
            "agent_utilities.orchestration.execution_profile.plan_execution_shape",
            return_value=shape,
        ),
        patch.object(
            agent_runner,
            "_resolve_agent_from_kg",
            return_value={"type": "server", "tools": [{"name": "gith__repos"}]},
        ),
        patch.object(
            agent_runner,
            "_build_execution_config",
            return_value={"mcp_toolsets": [object()]},
        ),
        patch.object(
            agent_runner,
            "_execute_single_server",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "results": {
                        "output": '```json\n{"tool": "repository-manager"}\n```'
                    },
                    "tool_calls": [],
                }
            ),
        ) as execute_server,
        patch.object(agent_runner, "_record_execution_trace") as mock_trace,
        patch.object(agent_runner, "_write_step_credit"),
    ):
        raw = await agent_runner.run_agent(
            agent_name="github-mcp",
            task="list repositories",
            allowed_tools=["gith__repos"],
            include_run_summary=True,
            run_id="run:" + "2" * 32,
        )

    payload = json.loads(raw)
    assert payload["run_summary"]["route"]["servers"] == ["github-mcp"]
    assert payload["run_summary"]["outcome"] == "degraded"
    assert payload["run_summary"]["execution_mode"] == "single_server_agent"
    assert "without recorded ToolCall provenance" in payload["output"]
    assert "repository-manager-mcp" not in payload["run_summary"]["route"]["servers"]
    assert execute_server.await_args is not None
    assert execute_server.await_args.kwargs["agent_name"] == "github-mcp"
    _, trace_kwargs = mock_trace.call_args
    assert trace_kwargs["status"] == "degraded"
    assert trace_kwargs["tool_call_count"] == 0


@pytest.mark.asyncio
async def test_run_agent_success_run_summary_has_ok_outcome_and_no_failure() -> None:
    # The KG engine boundary is entirely synchronous (every backend/registry call in
    # this branch is offloaded via ``asyncio.to_thread`` / ``_call_without_blocking``,
    # never awaited directly) — an AsyncMock here made every auto-vivified attribute
    # (e.g. FeedbackService.from_engine's ``engine.store`` fallback) a coroutine whose
    # synchronous call site never awaits it, leaking an "AsyncMock ... was never
    # awaited" RuntimeWarning. MagicMock matches the real, synchronous contract.
    fake_engine = MagicMock()
    fake_engine.backend = None

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner, "_build_execution_config", return_value={"mcp_toolsets": []}
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "results": {"output": "Found 3 running containers."},
                    "metadata": {},
                    "execution_evidence": {
                        "schema_version": "graph-execution-evidence-v1",
                        "topology": "basic",
                        "topology_digest": "sha256:topology",
                        "version_digest": "sha256:version",
                        "runtime_version": "2.21.0",
                        "node_sequence": ["route", "__end__"],
                        "transitions": [],
                        "checkpoint_ids": [],
                        "resume_supported": False,
                    },
                }
            ),
        ),
        patch.object(agent_runner, "_record_execution_trace") as record_trace,
        patch.object(agent_runner, "_write_step_credit"),
    ):
        raw = await agent_runner.run_agent(
            agent_name="some-agent", task="t", include_run_summary=True
        )

    payload = json.loads(raw)
    assert payload["output"] == "Found 3 running containers."
    assert payload["run_summary"]["outcome"] == "ok"
    assert payload["run_summary"]["execution_mode"] == "pydantic_graph"
    assert "failure" not in payload["run_summary"]
    assert (
        record_trace.call_args.kwargs["graph_execution_evidence"]["topology_digest"]
        == "sha256:topology"
    )


@pytest.mark.asyncio
async def test_run_agent_without_include_run_summary_keeps_bare_string_contract() -> (
    None
):
    """The default (existing) contract is untouched for a caller that never opts in."""
    # The KG engine boundary is entirely synchronous (every backend/registry call in
    # this branch is offloaded via ``asyncio.to_thread`` / ``_call_without_blocking``,
    # never awaited directly) — an AsyncMock here made every auto-vivified attribute
    # (e.g. FeedbackService.from_engine's ``engine.store`` fallback) a coroutine whose
    # synchronous call site never awaits it, leaking an "AsyncMock ... was never
    # awaited" RuntimeWarning. MagicMock matches the real, synchronous contract.
    fake_engine = MagicMock()
    fake_engine.backend = None

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner, "_build_execution_config", return_value={"mcp_toolsets": []}
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "results": {"output": "plain answer"},
                }
            ),
        ),
        patch.object(agent_runner, "_record_execution_trace"),
        patch.object(agent_runner, "_write_step_credit"),
    ):
        out = await agent_runner.run_agent(agent_name="some-agent", task="t")

    assert out == "plain answer"  # bare string — not JSON


@pytest.mark.asyncio
async def test_run_agent_cancellation_best_effort_records_a_timeout_trace() -> None:
    """A caller-side wall-clock/reply-budget cancellation must NOT leave zero durable trace
    for the pre-generated run_id: run_agent best-effort records a status="timeout" RunTrace
    before re-raising CancelledError, so a trace_ref handed out before the call resolves to a
    REAL node."""
    import asyncio as _asyncio

    # The KG engine boundary is entirely synchronous (every backend/registry call in
    # this branch is offloaded via ``asyncio.to_thread`` / ``_call_without_blocking``,
    # never awaited directly) — an AsyncMock here made every auto-vivified attribute
    # (e.g. FeedbackService.from_engine's ``engine.store`` fallback) a coroutine whose
    # synchronous call site never awaits it, leaking an "AsyncMock ... was never
    # awaited" RuntimeWarning. MagicMock matches the real, synchronous contract.
    fake_engine = MagicMock()
    fake_engine.backend = None

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner, "_build_execution_config", return_value={"mcp_toolsets": []}
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(side_effect=_asyncio.CancelledError()),
        ),
        patch.object(agent_runner, "_record_execution_trace") as mock_trace,
        patch.object(agent_runner, "_write_step_credit"),
    ):
        pinned_run_id = "run:" + "2" * 32
        with pytest.raises(_asyncio.CancelledError):
            await agent_runner.run_agent(
                agent_name="some-agent", task="t", run_id=pinned_run_id
            )

    # Exactly one best-effort trace write happened, status="timeout", for the SAME run_id.
    assert mock_trace.call_count == 1
    args, kwargs = mock_trace.call_args
    assert args[1] == pinned_run_id  # run_id positional arg
    assert kwargs["status"] == "timeout"
    assert "cancelled" in kwargs["error"].lower()


@pytest.mark.asyncio
async def test_run_agent_cancellation_trace_write_never_blocks_the_reraise() -> None:
    """Even if the best-effort trace write itself raises, CancelledError must still
    propagate cleanly (never swallowed/converted)."""
    import asyncio as _asyncio

    # The KG engine boundary is entirely synchronous (every backend/registry call in
    # this branch is offloaded via ``asyncio.to_thread`` / ``_call_without_blocking``,
    # never awaited directly) — an AsyncMock here made every auto-vivified attribute
    # (e.g. FeedbackService.from_engine's ``engine.store`` fallback) a coroutine whose
    # synchronous call site never awaits it, leaking an "AsyncMock ... was never
    # awaited" RuntimeWarning. MagicMock matches the real, synchronous contract.
    fake_engine = MagicMock()
    fake_engine.backend = None

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner, "_build_execution_config", return_value={"mcp_toolsets": []}
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(side_effect=_asyncio.CancelledError()),
        ),
        patch.object(
            agent_runner,
            "_record_execution_trace",
            side_effect=RuntimeError("KG backend exploded"),
        ),
        patch.object(agent_runner, "_write_step_credit"),
    ):
        with pytest.raises(_asyncio.CancelledError):
            await agent_runner.run_agent(agent_name="some-agent", task="t")
