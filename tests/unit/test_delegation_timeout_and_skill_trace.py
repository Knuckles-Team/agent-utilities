"""Delegation wall-clock timeout (fail-loud) + skill-utilization provenance (F8).

CONCEPT:AU-ORCH.execution.delegation-wall-clock — a blocking fleet tool must not hang the whole
delegation; the single-server loop times out and raises so the caller records a
degraded/failed run. CONCEPT:AU-ORCH.execution.skill-utilization-provenance — the RunTrace records
which skill drove the run and which server it bound, plus a USES_SKILL edge.
"""

from __future__ import annotations

import asyncio

import pytest

import agent_utilities.orchestration.agent_runner as ar
from agent_utilities.observability.trace_ontology import trace_id
from agent_utilities.security.persistence_privacy import persistence_reference

# --- Fix 1: wall-clock timeout ------------------------------------------------


@pytest.mark.asyncio
async def test_single_server_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(ar, "_EXECUTE_AGENT_WALL_CLOCK_S", 0.1)

    class _HangAgent:
        async def run(self, *a, **k):
            await asyncio.sleep(30)  # a bound tool that never returns

    monkeypatch.setattr(
        "agent_utilities.agent.factory.create_agent",
        lambda **k: (_HangAgent(), True),
    )
    config = {"mcp_toolsets": [object()], "agent_model": "m", "provider": "openai"}
    with pytest.raises(RuntimeError, match="wall-clock"):
        await ar._execute_single_server(
            config=config,
            task="list things",
            max_steps=2,
            agent_meta={"type": "server"},
            agent_name="systems-manager-mcp",
        )


# --- Fix 2 (F8): skill-utilization provenance --------------------------------


class _CapturingBackend:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def execute(self, cypher, params):
        self.calls.append((cypher, params))
        return []


class _CapturingEngine:
    def __init__(self):
        self.backend = _CapturingBackend()
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, label, properties=None):
        self.nodes[node_id] = {"label": label, **(properties or {})}

    def link_nodes(self, source, target, rel_type, properties=None):
        self.edges.append((source, target, rel_type))


def test_runtrace_records_skill_and_bound_server_and_edge():
    eng = _CapturingEngine()
    ar._record_execution_trace(
        eng,
        "run:abc",
        "container-manager-kubernetes-operations",
        "list namespaces",
        status="completed",
        skill_used="container-manager-kubernetes-operations",
        bound_server="container-manager-mcp",
        skill_id="resource:skill:container-manager-kubernetes-operations",
    )
    trace = eng.nodes[trace_id("run:abc")]
    assert trace["skill_ref"] == persistence_reference(
        "skill",
        "container-manager-kubernetes-operations",
        namespace="execution-trace",
    )
    assert trace["server_ref"] == persistence_reference(
        "server", "container-manager-mcp", namespace="execution-trace"
    )
    # EXECUTED_ON links to the BOUND server (not srv:<skill>, which doesn't exist)
    exec_on = [edge for edge in eng.edges if edge[2] == "EXECUTED_ON"]
    assert exec_on == [
        (trace_id("run:abc"), "srv:container-manager-mcp", "EXECUTED_ON")
    ]
    # USES_SKILL edge matches the skill by ID (the engine can't match by name in a write)
    uses = [edge for edge in eng.edges if edge[2] == "USES_SKILL"]
    assert uses == [
        (
            trace_id("run:abc"),
            "resource:skill:container-manager-kubernetes-operations",
            "USES_SKILL",
        )
    ]


def test_uses_skill_edge_falls_back_to_skill_prefix_id():
    eng = _CapturingEngine()
    ar._record_execution_trace(
        eng, "run:def", "some-skill", "t", status="completed", skill_used="some-skill"
    )
    uses = [edge for edge in eng.edges if edge[2] == "USES_SKILL"]
    assert uses == [(trace_id("run:def"), "resource:skill:some-skill", "USES_SKILL")]


def test_runtrace_no_skill_edge_for_plain_server_run():
    eng = _CapturingEngine()
    ar._record_execution_trace(
        eng, "run:xyz", "tunnel-manager-mcp", "list hosts", status="completed"
    )
    assert "skill_ref" not in eng.nodes[trace_id("run:xyz")]
    assert not [edge for edge in eng.edges if edge[2] == "USES_SKILL"]
    # EXECUTED_ON falls back to the agent's own server node
    exec_on = [edge for edge in eng.edges if edge[2] == "EXECUTED_ON"]
    assert exec_on == [(trace_id("run:xyz"), "srv:tunnel-manager-mcp", "EXECUTED_ON")]


class _FailingEngine:
    """An engine whose write raises -- reproduces a run that returns `status="ok"`
    to its caller (this function runs after dispatch, on every exit path) while
    its own RunTrace persistence fails (D-DG-7)."""

    def add_node(self, *_args, **_kwargs):
        raise RuntimeError("Graph 'test_9b7c1387b9c7' not found")

    def link_nodes(self, *_args, **_kwargs):
        return None


def test_runtrace_write_failure_is_logged_at_error_not_swallowed_silently(caplog):
    """D-DG-7: a run that reports `status="ok"` with a `trace_ref` whose write
    actually failed was invisible to the reward/evolution flywheel because the
    failure was logged at `debug` -- below every production log level. Assert
    the failure is now logged at `error`, naming both the run id and the exact
    trace id a reader would look for, so a failed provenance write is as
    diagnosable as any other production failure (never raises into the caller
    -- `_record_execution_trace` runs on every exit path of `run_agent`)."""
    import logging

    eng = _FailingEngine()
    with caplog.at_level(
        logging.ERROR, logger="agent_utilities.orchestration.agent_runner"
    ):
        ar._record_execution_trace(  # must not raise
            eng, "run:trace-write-fails", "some-skill", "t", status="ok"
        )

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "expected the failed trace write to log at ERROR"
    message = error_records[0].getMessage()
    assert "run_id='run:trace-write-fails'" in message
    assert trace_id("run:trace-write-fails") in message
    assert "Graph 'test_9b7c1387b9c7' not found" in message
