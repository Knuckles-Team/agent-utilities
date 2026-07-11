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

    def add_node(self, node_id, label, properties=None):
        self.nodes[node_id] = {"label": label, **(properties or {})}


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
    )
    trace = eng.nodes["trace:run:abc"]
    assert trace["skill_used"] == "container-manager-kubernetes-operations"
    assert trace["bound_server"] == "container-manager-mcp"
    # EXECUTED_ON links to the BOUND server (not srv:<skill>, which doesn't exist)
    exec_on = [c for c in eng.backend.calls if "EXECUTED_ON" in c[0]]
    assert exec_on and exec_on[0][1]["sid"] == "srv:container-manager-mcp"
    # and a USES_SKILL edge records which skill drove the run
    uses = [c for c in eng.backend.calls if "USES_SKILL" in c[0]]
    assert uses and uses[0][1]["skill"] == "container-manager-kubernetes-operations"


def test_runtrace_no_skill_edge_for_plain_server_run():
    eng = _CapturingEngine()
    ar._record_execution_trace(
        eng, "run:xyz", "tunnel-manager-mcp", "list hosts", status="completed"
    )
    assert "skill_used" not in eng.nodes["trace:run:xyz"]
    assert not [c for c in eng.backend.calls if "USES_SKILL" in c[0]]
    # EXECUTED_ON falls back to the agent's own server node
    exec_on = [c for c in eng.backend.calls if "EXECUTED_ON" in c[0]]
    assert exec_on and exec_on[0][1]["sid"] == "srv:tunnel-manager-mcp"
