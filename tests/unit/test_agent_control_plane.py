"""Direct contract tests for AU's verified, typed RLM application operation."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from agent_utilities.api import (
    GraphRlmBenchmarkRequest,
    GraphRlmEvolutionOptions,
    GraphRlmEvolvePromptRequest,
    GraphRlmRunRequest,
    compose_agent_control_plane,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.mcp import kg_server
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, CredentialLease


class _FakeEpistemicGraphClient:
    def __init__(self) -> None:
        self.claims: list[dict[str, object]] = []
        self.active_claims: dict[str, object] | None = None

    @contextmanager
    def use_verified_context(self, claims: dict[str, object]):
        self.claims.append(claims)
        previous = self.active_claims
        self.active_claims = claims
        try:
            yield
        finally:
            self.active_claims = previous


def _session() -> GraphSession:
    actor = ActorContext(
        actor_id="agent:rlm-test",
        actor_type=ActorType.AI_AGENT,
        tenant_id="tenant:test",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="tenant:test",
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="policy:test",
        audience="graph-os",
    )


@pytest.mark.asyncio
async def test_run_uses_instance_bound_verified_context_and_typed_result():
    eg_client = _FakeEpistemicGraphClient()
    session = _session()
    control = compose_agent_control_plane(eg_client, session)

    async def run_rlm(task: str, *, input_text: str):
        assert eg_client.active_claims == eg_client.claims[-1]
        assert eg_client.active_claims["principal"] == "agent:rlm-test"
        assert eg_client.active_claims["policy_version"] == "policy:test"
        assert task == "summarize evidence"
        assert input_text == "evidence text"
        return {
            "ok": True,
            "result": "summary",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total": 5},
            "max_depth": 4,
        }

    with patch("agent_utilities.rlm.runner.run_rlm", new=run_rlm):
        with use_session(session):
            result = await control.graph_rlm(
                GraphRlmRunRequest(
                    task="summarize evidence", input_text="evidence text"
                )
            )

    assert result.action == "run"
    assert result.ok is True
    assert result.result == "summary"
    assert result.max_depth == 4
    assert result.usage["total"] == 5
    assert len(eg_client.claims) == 1


@pytest.mark.asyncio
async def test_run_failure_is_typed_and_does_not_expose_runner_error_text():
    eg_client = _FakeEpistemicGraphClient()
    session = _session()
    control = compose_agent_control_plane(eg_client, session)
    with patch(
        "agent_utilities.rlm.runner.run_rlm",
        new=AsyncMock(
            return_value={
                "ok": False,
                "error": "private endpoint and payload must not escape",
                "failure_class": "host_tool_timeout",
            }
        ),
    ):
        with use_session(session):
            result = await control.graph_rlm(GraphRlmRunRequest(task="test"))

    assert result.ok is False
    assert result.failure_class == "host_tool_timeout"
    assert result.error is not None
    assert result.error.correlation_id.startswith("correlation:")
    assert result.error.detail_ref == result.error.correlation_id
    assert "private endpoint" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_benchmark_uses_typed_options_and_returns_typed_rows(monkeypatch):
    from agent_utilities.rlm import benchmarks

    calls: list[dict[str, object]] = []

    async def run_benchmark(task, *, scales, cases_per_scale):
        calls.append(
            {"task": task, "scales": scales, "cases_per_scale": cases_per_scale}
        )
        from agent_utilities.rlm.benchmarks.base import BenchResult

        return [
            BenchResult(
                task="s_niah",
                complexity="O(n)",
                system="rlm",
                scale=1000,
                accuracy=1.0,
                n=1,
            )
        ]

    monkeypatch.setattr(benchmarks, "list_tasks", lambda: ["s_niah"])
    monkeypatch.setattr(benchmarks, "run_benchmark", run_benchmark)
    monkeypatch.setattr(benchmarks, "render_scoreboard", lambda _rows: "scoreboard")

    session = _session()
    control = compose_agent_control_plane(_FakeEpistemicGraphClient(), session)
    with use_session(session):
        result = await control.graph_rlm(
            GraphRlmBenchmarkRequest(
                action="benchmark",
                task="s_niah",
                options={"scales": [1000], "cases_per_scale": 2},
            )
        )

    assert result.ok is True
    assert result.results[0].task == "s_niah"
    assert result.results[0].scale == 1000
    assert result.results[0].accuracy == 1.0
    assert result.scoreboard == "scoreboard"
    assert calls == [{"task": "s_niah", "scales": [1000], "cases_per_scale": 2}]


@pytest.mark.asyncio
async def test_evolve_prompt_preserves_gepa_and_dynamic_reward_path():
    class _MockMutateResponse:
        output = (
            '{"rationale":"Tightened the answer format.", '
            '"mutated_prompt":"Answer precisely."}'
        )

    async def mock_mutate_run(_self, _prompt, **_kwargs):
        return _MockMutateResponse()

    async def mock_harness_run(self_harness, **inputs):
        query = str(inputs.get("query") or "")
        answer = "Paris" if "France" in query else "4"
        return self_harness.signature(query=query, response=answer)

    async def mock_create_or_merge(_node):
        return {"status": "merged"}

    session = _session()
    control = compose_agent_control_plane(_FakeEpistemicGraphClient(), session)
    with (
        patch("pydantic_ai.Agent.run", new=mock_mutate_run),
        patch("agent_utilities.rlm.predict_rlm.PredictRLM.run", new=mock_harness_run),
        patch(
            "agent_utilities.rlm.gepa.create_or_merge_node",
            new=mock_create_or_merge,
        ),
    ):
        with use_session(session):
            result = await control.graph_rlm(
                GraphRlmEvolvePromptRequest(
                    action="evolve_prompt",
                    options=GraphRlmEvolutionOptions(iterations=1, batch_size=2),
                )
            )

    assert result.ok is True
    assert result.winning_prompt
    assert "accuracy" in result.scores
    assert result.reward_weights == {"accuracy": 1.0}
    assert result.frontier_size >= 1


def test_composition_fails_closed_on_absent_or_unverified_authority():
    with pytest.raises(ValueError, match="client is required"):
        compose_agent_control_plane(None, _session())
    with pytest.raises(TypeError, match="verified context"):
        compose_agent_control_plane(object(), _session())
    with pytest.raises(TypeError, match="verified GraphSession"):
        compose_agent_control_plane(_FakeEpistemicGraphClient(), None)
    with pytest.raises(PermissionError):
        compose_agent_control_plane(
            _FakeEpistemicGraphClient(),
            GraphSession(
                actor=ActorContext(
                    actor_id="agent:unverified", tenant_id="tenant:test"
                ),
                tenant="tenant:test",
                audience="graph-os",
                policy_version="policy:test",
            ),
        )


@pytest.mark.asyncio
async def test_expired_authority_returns_correlated_refusal_without_calling_rlm():
    eg_client = _FakeEpistemicGraphClient()
    actor = ActorContext(
        actor_id="agent:rlm-test",
        actor_type=ActorType.AI_AGENT,
        tenant_id="tenant:test",
        authenticated=True,
        credential_lease=CredentialLease(int(time.time()) + 60),
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant:test",
        policy_version="policy:test",
        audience="graph-os",
    )
    control = compose_agent_control_plane(eg_client, session)
    with use_session(session):
        actor.credential_lease.renew(1)
        result = await control.graph_rlm(GraphRlmRunRequest(task="never invoked"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "permission_denied"
    assert result.error.correlation_id.startswith("correlation:")
    assert eg_client.claims == []


@pytest.mark.asyncio
async def test_verified_but_nonambient_session_is_rejected():
    eg_client = _FakeEpistemicGraphClient()
    session = _session()
    control = compose_agent_control_plane(eg_client, session)

    result = await control.graph_rlm(GraphRlmRunRequest(task="not ambient"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "permission_denied"
    assert eg_client.claims == []


@pytest.mark.asyncio
async def test_typed_options_are_strict_and_bounded():
    with pytest.raises(ValidationError):
        GraphRlmBenchmarkRequest.model_validate(
            {"action": "benchmark", "options": {"scales": [10_000_001]}}
        )
    with pytest.raises(ValidationError):
        GraphRlmEvolutionOptions.model_validate(
            {"iterations": 21, "untrusted_authority": "tenant:other"}
        )


def test_legacy_global_rlm_route_is_absent_and_api_has_no_kg_server_dependency():
    from agent_utilities.api import agent_control_plane

    assert importlib.util.find_spec("agent_utilities.mcp.tools.rlm_tools") is None
    assert "graph_rlm" not in kg_server.REGISTERED_TOOLS
    assert "/graph/rlm" not in kg_server.ACTION_TOOL_ROUTES.values()
    assert "agent_utilities.mcp.kg_server" not in agent_control_plane.__dict__
    assert "agent_utilities.mcp.kg_server" not in inspect.getsource(agent_control_plane)
    assert "agent_utilities.mcp.tools.rlm_tools" not in sys.modules
