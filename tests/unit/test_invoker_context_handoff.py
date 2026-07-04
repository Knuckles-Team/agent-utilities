"""Invoker→spawned-agent curated-context handoff (CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox, MVP Phase 1).

Covers: the budgeted ``### INVOKER CONTEXT`` section helper, the ``GraphState.invoker_context``
field, and that ``run_agent(context=...)`` threads the curated context into the execution
config that seeds ``GraphState`` (and thus the spawn assemblers).

@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
"""

from __future__ import annotations

import pytest

from agent_utilities.graph.executor import invoker_context_section
from agent_utilities.graph.state import GraphState


@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
def test_section_empty_when_no_context():
    state = GraphState(query="q")
    assert state.invoker_context == ""
    assert invoker_context_section(state) == ""


@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
def test_section_rendered_when_present():
    state = GraphState(query="q", invoker_context="The user prefers metric units.")
    section = invoker_context_section(state)
    assert "### INVOKER CONTEXT" in section
    assert "metric units" in section


@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
def test_section_budgeted_to_window():
    # 200k chars of context must be trimmed to fit a 32K-token window fraction (~19.6K chars).
    big = "x" * 200_000
    state = GraphState(query="q", invoker_context=big)
    section = invoker_context_section(state, window_tokens=32768)
    assert "truncated to fit model window" in section
    # well under the full blob, comfortably within budget + header overhead
    assert len(section) < 25_000


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
async def test_run_agent_threads_context_into_config(monkeypatch):
    """run_agent(context=...) must place the curated context on the execution config
    that seeds GraphState (proves the entrypoint→state thread)."""
    from agent_utilities.orchestration import agent_runner

    captured = {}

    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "stub"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_build_execution_config",
        lambda e, n, m, **kw: {"tag_prompts": {}},
    )
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def _fake_execute_graph(*, config, **kwargs):
        captured["invoker_context"] = config.get("invoker_context")
        captured["invoker_budget_tokens"] = config.get("invoker_budget_tokens")
        return {"results": {"output": "ok"}}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    await agent_runner.run_agent(
        agent_name="unregistered-stub",
        task="do it",
        engine=object(),
        context="INVOKER SAYS: use the staging cluster only.",
        budget_tokens=12345,
    )
    assert captured["invoker_context"] == "INVOKER SAYS: use the staging cluster only."
    assert captured["invoker_budget_tokens"] == 12345


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
async def test_run_agent_resolves_context_ref(monkeypatch):
    """context_ref resolves a persisted ContextBlob's content into invoker_context
    (Phase 2 cross-process handoff)."""
    from agent_utilities.orchestration import agent_runner

    captured = {}

    class _FakeEngine:
        backend = None

        def query_cypher(self, cypher, params=None, **kw):
            params = params or {}
            if "ContextBlob" in cypher and params.get("id") == "ctx:abc:1":
                return [{"content": "RESOLVED BLOB CONTENT"}]
            return []

    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "stub"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_build_execution_config",
        lambda e, n, m, **kw: {"tag_prompts": {}},
    )
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def _fake_execute_graph(*, config, **kwargs):
        captured["invoker_context"] = config.get("invoker_context")
        return {"results": {"output": "ok"}}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    await agent_runner.run_agent(
        agent_name="stub", task="t", engine=_FakeEngine(), context_ref="ctx:abc:1"
    )
    assert captured["invoker_context"] == "RESOLVED BLOB CONTENT"


@pytest.mark.concept("AU-ORCH.execution.orchestration-flow-mermaid")
def test_plan_output_type_uses_promptedoutput_for_non_json_models(monkeypatch):
    """FU-2: planner wraps GraphPlan in PromptedOutput when the model can't do native JSON."""
    from pydantic_ai import PromptedOutput

    from agent_utilities.graph import hierarchical_planner as hp

    class _M:
        model_name = "qwen-lite"

    monkeypatch.setattr(
        hp, "get_model_config", lambda mid=None: {"supports_json": False}, raising=False
    )
    # ensure the helper imports our patched get_model_config path:
    import agent_utilities.core.model_factory as mf

    monkeypatch.setattr(
        mf, "get_model_config", lambda mid=None: {"supports_json": False}
    )
    out = hp._plan_output_type(_M())
    assert isinstance(out, PromptedOutput)

    monkeypatch.setattr(
        mf, "get_model_config", lambda mid=None: {"supports_json": True}
    )
    assert hp._plan_output_type(_M()) is hp.GraphPlan


@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
def test_apply_tool_scope_filters_to_allow_list():
    from agent_utilities.graph.executor import apply_tool_scope

    def tool_a():  # noqa: D401
        pass

    def tool_b():
        pass

    class _FakeToolset:
        def __init__(self):
            self.predicate = None

        def filtered(self, func):
            self.predicate = func
            return self  # record the predicate; identity is fine for the test

    # No allow-list → unchanged.
    ts = _FakeToolset()
    tools, toolsets = apply_tool_scope(GraphState(query="q"), [tool_a, tool_b], [ts])
    assert tools == [tool_a, tool_b] and ts.predicate is None

    # Allow-list → function tools filtered by name; toolset gets a filtering predicate.
    state = GraphState(query="q", invoker_allowed_tools=["tool_a", "list_projects"])
    tools, toolsets = apply_tool_scope(state, [tool_a, tool_b], [ts])
    assert tools == [tool_a]
    assert ts.predicate is not None
    # predicate admits allowed names, rejects others
    td_ok = type("TD", (), {"name": "list_projects"})()
    td_no = type("TD", (), {"name": "delete_everything"})()
    assert ts.predicate(None, td_ok) is True
    assert ts.predicate(None, td_no) is False


@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
def test_cred_ref_resolves_to_auth_token_only_on_transient_deps():
    """Phase 4: cred_ref (a reference) resolves to the raw token onto AgentDeps.auth_token via
    the secrets client; the raw secret is never placed on GraphState."""
    from types import SimpleNamespace

    from agent_utilities.graph.executor import agent_deps_from_graph

    secrets = SimpleNamespace(
        get=lambda key: "SECRET-TOKEN-XYZ" if key == "cred:sess1" else None
    )
    deps = SimpleNamespace(
        project_root="",
        knowledge_engine=None,
        mcp_toolsets=[],
        ssl_verify=True,
        provider="openai",
        base_url=None,
        api_key=None,
        request_id="",
        approval_timeout=0.0,
        event_queue=None,
        secrets_client=secrets,
    )
    state = GraphState(query="q", invoker_cred_ref="cred:sess1")
    agent_deps = agent_deps_from_graph(deps, [], state=state)
    assert agent_deps.auth_token == "SECRET-TOKEN-XYZ"
    # The raw secret must NOT be on GraphState (only the reference).
    assert "SECRET-TOKEN-XYZ" not in str(vars(state))
    assert state.invoker_cred_ref == "cred:sess1"

    # No cred_ref → no token.
    agent_deps2 = agent_deps_from_graph(deps, [], state=GraphState(query="q"))
    assert agent_deps2.auth_token is None


@pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")
def test_spawn_usage_limits_enforces_budget():
    from agent_utilities.graph.executor import spawn_usage_limits

    # No budget → request-bounded only, no token cap.
    ul = spawn_usage_limits(GraphState(query="q"))
    assert ul.request_limit is not None
    assert ul.total_tokens_limit is None

    # Budget set → enforced as a hard total-tokens limit.
    ul2 = spawn_usage_limits(GraphState(query="q", invoker_budget_tokens=9000))
    assert ul2.total_tokens_limit == 9000
