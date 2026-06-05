"""CONCEPT:ORCH-1.32 — KG-Governed Agent Swarm (SWARM-1…7).

Assimilated from Kimi Agent Swarm. These extend the existing ParallelEngine (ORCH-1.8) rather than
reinventing it: the engine already does dependency-ordered parallel waves + synthesis. The tests
exercise the *new* governance/quality deltas — critical-path metric, per-agent schema enforcement,
retry/backoff, heterogeneous model routing, and the planner→execute→verify loop — through the real
engine entry point (`execute`), with LLM calls mocked.
"""

from __future__ import annotations

import asyncio

from agent_utilities.graph.parallel_engine import (
    ParallelEngine,
    enforce_structured_output,
    resolve_model_role,
)
from agent_utilities.models.execution_manifest import (
    AgentExecutionResult,
    AgentSpec,
    ExecutionManifest,
)

# ── SWARM-4: structured-output enforcement (pure) ────────────────────────────────


def test_enforce_structured_output_valid_and_invalid():
    assert enforce_structured_output('{"a": 1}', '{"required": ["a"]}')[0] is True
    assert enforce_structured_output('{"b": 1}', '{"required": ["a"]}')[0] is False
    assert enforce_structured_output("prose not json", "a,b")[0] is False
    # fenced JSON is tolerated
    assert enforce_structured_output('```json\n{"a":1}\n```', "a")[0] is True
    # no schema = always pass (back-compat)
    assert enforce_structured_output("anything", None)[0] is True


# ── SWARM-6: heterogeneous model routing (best-effort) ───────────────────────────


def test_resolve_model_role_returns_str():
    # unresolvable role -> "" (caller falls back); never raises
    assert isinstance(resolve_model_role("definitely-not-a-role"), str)
    assert resolve_model_role("") == ""


# ── SWARM-3: critical-path scheduling ────────────────────────────────────────────


def test_critical_path_is_longest_chain_not_wave_count():
    eng = ParallelEngine()
    specs = [
        AgentSpec(agent_id="a"),
        AgentSpec(agent_id="b", depends_on=["a"]),
        AgentSpec(agent_id="c", depends_on=["b"]),
        AgentSpec(agent_id="d"),  # independent
        AgentSpec(agent_id="e"),  # independent
    ]
    m = ExecutionManifest(agents=specs, execution_mode="wave")
    waves = eng._schedule_waves(m)
    assert [a.agent_id for a in waves[0]] == [
        "a",
        "d",
        "e",
    ]  # first generation parallel
    assert eng._schedule_meta["critical_path_length"] == 3  # a->b->c
    assert eng._schedule_meta["parallelism_ratio"] == round(5 / 3, 2)


def test_critical_path_all_independent_is_one():
    eng = ParallelEngine()
    m = ExecutionManifest(
        agents=[AgentSpec(agent_id=f"x{i}") for i in range(6)], execution_mode="wave"
    )
    eng._schedule_waves(m)
    assert eng._schedule_meta["critical_path_length"] == 1


# ── SWARM-2/5/7: verify loop + retry/backoff + telemetry (engine integration) ────


def test_swarm_execute_verifies_retries_and_reports_telemetry():
    async def _run():
        eng = ParallelEngine()
        calls = {"b": 0}

        async def fake_exec(agent, manifest, graph_deps, wave_results, proc=None):
            if agent.agent_id == "b":  # transient: fail once, then succeed (SWARM-5)
                calls["b"] += 1
                ok = calls["b"] >= 2
                return AgentExecutionResult(
                    agent_id="b",
                    output="ok" if ok else "",
                    success=ok,
                    error="" if ok else "transient",
                )
            return AgentExecutionResult(
                agent_id=agent.agent_id, output=f"out-{agent.agent_id}", success=True
            )

        eng._execute_agent = fake_exec  # type: ignore[method-assign]

        seen = {"c": 0}

        async def fake_judge(output, criteria, query, graph_deps):
            if (
                "addresses c" in criteria.lower() and seen["c"] == 0
            ):  # fail c once -> re-dispatch
                seen["c"] += 1
                return False, "needs detail"
            return True, ""

        eng._judge_against_criteria = fake_judge  # type: ignore[method-assign]

        specs = [
            AgentSpec(agent_id="a", success_criteria="addresses a"),
            AgentSpec(
                agent_id="b",
                depends_on=["a"],
                max_retries=2,
                success_criteria="addresses b",
            ),
            AgentSpec(agent_id="c", depends_on=["b"], success_criteria="addresses c"),
        ]
        m = ExecutionManifest(
            agents=specs,
            query="goal",
            execution_mode="wave",
            metadata={"verify": True, "max_retries": 2},
        )
        return await eng.execute(m), calls

    result, calls = asyncio.run(_run())
    # SWARM-3
    assert result.critical_path_length == 3
    # SWARM-5: b failed once then retried to success
    assert calls["b"] == 2
    # SWARM-2: all three verified; c was re-dispatched once
    assert result.verification == {
        "checked": 3,
        "passed": 3,
        "failed": 0,
        "redispatched": 1,
    }
    # SWARM-7: per-wave telemetry present
    assert len(result.telemetry["waves"]) == result.wave_count
    assert result.telemetry["critical_path_length"] == 3
    assert result.success is True


def test_verify_is_noop_when_not_enabled():
    async def _run():
        eng = ParallelEngine()

        async def fake_exec(agent, manifest, graph_deps, wave_results, proc=None):
            return AgentExecutionResult(
                agent_id=agent.agent_id, output="x", success=True
            )

        eng._execute_agent = fake_exec  # type: ignore[method-assign]
        m = ExecutionManifest(
            agents=[AgentSpec(agent_id="a", success_criteria="must")],
            query="g",
            execution_mode="parallel",
        )
        return await eng.execute(m)

    result = asyncio.run(_run())
    assert result.verification == {}  # verify off by default


# ── SWARM-1: the swarm action is wired into graph_orchestrate ────────────────────


def test_swarm_action_registered_in_graph_orchestrate():
    """Wire-First: the one-shot swarm action must exist on the graph_orchestrate tool body."""
    import inspect

    from agent_utilities.mcp import kg_server

    src = inspect.getsource(kg_server)
    assert 'action == "swarm"' in src
    assert "ParallelEngine(engine=engine).execute(manifest)" in src
    assert 'manifest.metadata["verify"] = True' in src  # governance ON by default
