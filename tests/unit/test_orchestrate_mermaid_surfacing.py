"""Tests for orchestration flow-diagram surfacing (CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid).

Covers the trickiest part of the feature — ``run_agent`` return shaping: a bare
string for internal callers and one ``output``/``run_id``/``mermaid`` envelope
for public delegation, including early success and failure exits. The additive
``mermaid`` keys on the swarm/compile/execute_workflow MCP handlers are exercised
end-to-end by ``test_workflow_e2e.py``.

@pytest.mark.concept("AU-ORCH.execution.orchestration-flow-mermaid")
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.orchestration import agent_runner


@pytest.fixture
def _patched_run_agent(monkeypatch):
    """Stub out KG resolution + graph execution so only return-shaping is under test."""
    fake_response = {
        "results": {"output": "the answer"},
        "mermaid": "```mermaid\ngraph TD\n  A-->B\n```",
    }

    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "stub"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_build_execution_config",
        lambda e, n, m, **kw: {
            "agent_model": "synthetic-model",
            "selected_model_class": "standard",
        },
    )
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def _fake_execute_graph(**kwargs):
        return dict(fake_response)

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)
    return fake_response


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.execution.orchestration-flow-mermaid")
async def test_run_agent_default_returns_bare_string(_patched_run_agent):
    """AC5: default (return_mermaid=False) preserves the bare-string contract."""
    out = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent",
        task="q",
        engine=object(),
    )
    assert out == "the answer"
    # must not be a JSON wrapper
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.execution.orchestration-flow-mermaid")
async def test_run_agent_return_mermaid_wraps_when_present(_patched_run_agent):
    """AC4: return_mermaid=True yields a JSON wrapper carrying output + mermaid."""
    out = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    payload = json.loads(out)
    assert payload["output"] == "the answer"
    assert "mermaid" in payload["mermaid"]
    assert payload["mermaid"].startswith("```mermaid")


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.execution.rich-result-wrapper")
async def test_run_agent_no_mermaid_still_surfaces_run_id(monkeypatch):
    """ORCH-1.97: return_mermaid=True ALWAYS wraps to carry the ``run_id`` handle
    (even with no diagram) — the trackable handle for a delegated run's RunTrace +
    :ToolCall provenance. Internal callers (return_mermaid=False) keep bare strings.
    """
    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "stub"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_build_execution_config",
        lambda e, n, m, **kw: {
            "agent_model": "synthetic-model",
            "selected_model_class": "standard",
        },
    )
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def _fake_execute_graph(**kwargs):
        return {"results": {"output": "no-diagram"}, "mermaid": None}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    out = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    payload = json.loads(out)
    assert payload["output"] == "no-diagram"
    assert payload["run_id"].startswith("run:")
    assert payload["mermaid"] is None


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.execution.rich-result-wrapper")
async def test_enterprise_early_success_uses_public_envelope(monkeypatch):
    """The enterprise fast path must not bypass the public delegation contract."""
    from agent_utilities.graph import manifest_generators, parallel_engine

    class _ParallelEngine:
        def __init__(self, *, engine):
            self.engine = engine

        async def execute(self, _manifest):
            return "enterprise-result"

    monkeypatch.setattr(
        manifest_generators, "manifest_for_enterprise", lambda _task, _engine: object()
    )
    monkeypatch.setattr(parallel_engine, "ParallelEngine", _ParallelEngine)
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    rich = await agent_runner.run_agent(
        agent_name="enterprise",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    payload = json.loads(rich)
    assert set(payload) == {"output", "run_id", "mermaid"}
    assert payload["output"] == "enterprise-result"
    assert payload["run_id"].startswith("run:")
    assert payload["mermaid"] is None

    bare = await agent_runner.run_agent(
        agent_name="enterprise", task="q", engine=object()
    )
    assert bare == "enterprise-result"


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.execution.rich-result-wrapper")
async def test_service_registry_early_success_uses_public_envelope(monkeypatch):
    """A native capability result retains the same public delegation envelope."""
    from agent_utilities.core.registry.service_adapter import ServiceRegistry

    class _Capability:
        def run(self, _task):
            return "service-result"

    class _Descriptor:
        @staticmethod
        def get_class():
            return _Capability

    registry = ServiceRegistry.instance()
    monkeypatch.setattr(
        registry,
        "get",
        lambda name: _Descriptor() if name == "synthetic-capability" else None,
    )

    rich = await agent_runner.run_agent(
        agent_name="synthetic-capability",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    payload = json.loads(rich)
    assert set(payload) == {"output", "run_id", "mermaid"}
    assert payload["output"] == "service-result"
    assert payload["run_id"].startswith("run:")
    assert payload["mermaid"] is None

    bare = await agent_runner.run_agent(
        agent_name="synthetic-capability", task="q", engine=object()
    )
    assert bare == "service-result"


@pytest.mark.asyncio
@pytest.mark.concept("AU-ORCH.execution.rich-result-wrapper")
async def test_main_execution_failure_uses_public_envelope(
    _patched_run_agent, monkeypatch
):
    """A failed graph execution still returns a trackable public run handle."""
    from agent_utilities.orchestration import execution_profile

    async def _fail_execute_graph(**_kwargs):
        raise RuntimeError("synthetic_failure")

    monkeypatch.setattr(agent_runner, "_execute_graph", _fail_execute_graph)
    monkeypatch.setattr(agent_runner, "_write_step_credit", lambda *a, **k: 0)
    monkeypatch.setattr(execution_profile, "record_shape_outcome", lambda *a, **k: None)

    rich = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    payload = json.loads(rich)
    assert set(payload) == {"output", "run_id", "mermaid"}
    assert payload["output"].startswith("Agent execution failed:")
    assert payload["run_id"].startswith("run:")
    assert payload["mermaid"] is None

    bare = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent", task="q", engine=object()
    )
    assert bare.startswith("Agent execution failed:")
    with pytest.raises(json.JSONDecodeError):
        json.loads(bare)
