"""Isolated contracts for delegated, run-scoped native GraphOS tools."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import UsageLimitExceeded
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_graph import End

from agent_utilities.graph import _router_impl, builder, executor
from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.models import GraphResponse
from agent_utilities.models.graph import ExecutionStep
from agent_utilities.orchestration import agent_runner
from agent_utilities.orchestration import engine as orchestration_engine_module
from agent_utilities.orchestration.engine import AgentOrchestrationEngine
from agent_utilities.security import permissions_kernel, tool_guard


async def _graph_lookup(query: str) -> dict[str, str]:
    """Look up one record in the operational graph."""

    return {"query": query}


def _build_config(
    *, native_toolset: FunctionToolset[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    return builder._build_graph_config(
        graph_nodes={},
        knowledge_engine=object(),
        agent_subject="delegated-query-skill",
        mcp_toolsets=[native_toolset],
        tag_prompts={"query": "Query specialist"},
        tag_env_vars={},
        mcp_url=None,
        mcp_config=None,
        router_model=None,
        agent_model=None,
        router_timeout=None,
        verifier_timeout=None,
        min_confidence=0.5,
        sub_agents=None,
        routing_strategy="hybrid",
        kwargs=kwargs,
    )


def test_graph_builder_requires_identity_for_native_toolset_and_forwards_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FunctionToolset(
        [_graph_lookup],
        id="run-scoped-native-tools",
        metadata={"graphos_native": True},
    )
    kernel = object()
    identity = object()
    exact_capabilities = ("graph_write_record", "graph_lookup")
    captured: dict[str, Any] = {}

    def resolve_permission_context(config: Any, **kwargs: Any) -> Any:
        captured["config"] = config
        captured.update(kwargs)
        return SimpleNamespace(kernel=kernel, identity=identity)

    monkeypatch.setattr(
        permissions_kernel,
        "resolve_permission_context",
        resolve_permission_context,
    )

    config = _build_config(
        native_toolset=native,
        kwargs={"capabilities": exact_capabilities},
    )

    assert captured["required"] is True
    assert captured["agent_subject"] == "delegated-query-skill"
    assert captured["capabilities"] is exact_capabilities
    assert config["permissions_kernel"] is kernel
    assert config["agent_identity"] is identity


class _SyntheticStream:
    usage = None

    async def __aenter__(self) -> _SyntheticStream:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def stream_text(self, *, delta: bool):
        assert delta is True
        if False:
            yield "unreachable"

    async def get_output(self) -> str:
        return "native execution completed"

    async def all_messages(self) -> list[Any]:
        return []


class _SyntheticAgent:
    def run_stream(self, *_args: Any, **_kwargs: Any) -> _SyntheticStream:
        return _SyntheticStream()


@pytest.mark.asyncio
async def test_specialist_retains_native_toolset_and_guards_before_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FunctionToolset(
        [_graph_lookup],
        id="server-that-does-not-match-query-specialist",
        metadata={"graphos_native": True},
    )
    kernel = object()
    identity = object()
    knowledge_engine = object()
    state = GraphState(
        query="Use the native graph lookup",
        invoker_allowed_tools=["_graph_lookup"],
    )
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[native],
        router_model=None,
        agent_model=None,
        permissions_kernel=kernel,
        agent_identity=identity,
        knowledge_engine=knowledge_engine,
    )
    ctx = SimpleNamespace(state=state, deps=deps, inputs=None)
    events: list[tuple[str, Any]] = []
    scoped_toolset = object()

    class GuardedToolset:
        def filtered(self, predicate: Any) -> object:
            events.append(("allowlist", self))
            assert predicate(None, SimpleNamespace(name="_graph_lookup")) is True
            assert predicate(None, SimpleNamespace(name="other_tool")) is False
            return scoped_toolset

    guarded = GuardedToolset()

    def apply_identity_policy(
        toolsets: list[Any],
        *,
        permissions_kernel: Any,
        agent_identity: Any,
        engine: Any,
    ) -> list[Any]:
        events.append(("identity", toolsets[0]))
        assert toolsets == [native]
        assert permissions_kernel is kernel
        assert agent_identity is identity
        assert engine is knowledge_engine
        return [guarded]

    def reject_domain_filter(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a run-scoped native toolset reached domain/server tag filtering")

    captured_agent: dict[str, Any] = {}

    def create_agent(**kwargs: Any) -> _SyntheticAgent:
        captured_agent.update(kwargs)
        events.append(("construct", tuple(kwargs["toolsets"])))
        return _SyntheticAgent()

    monkeypatch.setattr(tool_guard, "flag_mcp_tool_definitions", apply_identity_policy)
    monkeypatch.setattr(tool_guard, "apply_tool_guard_approvals", lambda _agent: None)
    monkeypatch.setattr(executor, "filter_tools_by_tag", reject_domain_filter)
    monkeypatch.setattr(
        executor,
        "get_discovery_registry",
        lambda: SimpleNamespace(agents=[]),
    )
    monkeypatch.setattr(executor, "load_specialized_prompts", lambda _name: "prompt")
    monkeypatch.setattr(
        executor,
        "_get_domain_tools",
        AsyncMock(return_value=([], [])),
    )
    monkeypatch.setattr(
        executor, "pick_specialist_model", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(executor, "create_context_agent", create_agent)
    monkeypatch.setattr(
        executor, "agent_deps_from_graph", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(executor, "spawn_usage_limits", lambda _state: None)
    monkeypatch.setattr(executor, "on_enter_specialist", AsyncMock())
    monkeypatch.setattr(executor, "on_exit_specialist", AsyncMock())
    monkeypatch.setattr(executor, "emit_graph_event", lambda *_args, **_kwargs: None)

    await executor._execute_specialized_step(ctx, "query-specialist")

    assert events[:3] == [
        ("identity", native),
        ("allowlist", guarded),
        ("construct", (scoped_toolset,)),
    ]
    assert captured_agent["permissions_kernel"] is kernel
    assert captured_agent["agent_identity"] is identity
    assert captured_agent["permission_engine"] is knowledge_engine
    assert captured_agent["toolsets"] == [scoped_toolset]


@pytest.mark.asyncio
async def test_execute_graph_forwards_agent_name_and_exact_invoker_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    exact_allowed_tools = ["graph_write_record", "graph_lookup"]
    graph = object()

    def create_graph_agent(**kwargs: Any) -> tuple[object, dict[str, Any]]:
        captured["builder"] = kwargs
        return graph, {"built": True}

    class SyntheticEngine:
        async def execute_graph(self, **kwargs: Any) -> dict[str, Any]:
            captured["execution"] = kwargs
            return {"status": "completed"}

    monkeypatch.setattr(builder, "create_graph_agent", create_graph_agent)
    monkeypatch.setattr(
        "agent_utilities.orchestration.engine.AgentOrchestrationEngine",
        SyntheticEngine,
    )

    result = await agent_runner._execute_graph(
        config={
            "tag_prompts": {"query": "Query specialist"},
            "mcp_toolsets": [],
            "invoker_allowed_tools": exact_allowed_tools,
            "execution_shape": SimpleNamespace(direct_complete=False),
        },
        query="Execute a delegated native lookup",
        run_id="synthetic-run",
        max_steps=3,
        agent_meta={"type": "skill"},
        agent_name="query-skill",
    )

    assert result == {"status": "completed"}
    assert captured["builder"]["name"] == "query-skill"
    assert captured["builder"]["capabilities"] == tuple(exact_allowed_tools)
    assert captured["execution"]["graph"] is graph


@pytest.mark.asyncio
async def test_router_direct_dispatch_reads_authority_from_graph_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FunctionToolset(
        [_graph_lookup],
        id="run-scoped-native-tools",
        metadata={"graphos_native": True},
    )
    kernel = object()
    identity = object()
    knowledge_engine = object()
    state = GraphState(
        query="Read the delegated record",
        invoker_allowed_tools=["_graph_lookup"],
    )
    deps = GraphDeps(
        tag_prompts={"query": "Query specialist"},
        tag_env_vars={},
        mcp_toolsets=[native],
        agent_model=None,
        execution_shape=SimpleNamespace(run_discovery=False),
        permissions_kernel=kernel,
        agent_identity=identity,
        knowledge_engine=knowledge_engine,
    )
    ctx = SimpleNamespace(state=state, deps=deps, inputs=None)
    agent = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(output="delegated result"))
    )
    constructed: dict[str, Any] = {}
    adapted: dict[str, Any] = {}
    adapted_deps = object()

    def create_agent(**kwargs: Any) -> Any:
        constructed.update(kwargs)
        return agent

    monkeypatch.setattr(_router_impl, "setting", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(_router_impl, "_emit_node_lifecycle", lambda *_args: None)
    monkeypatch.setattr(
        _router_impl, "emit_graph_event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(_router_impl, "create_context_agent", create_agent)
    monkeypatch.setattr(_router_impl, "spawn_usage_limits", lambda _state: object())

    def adapt_graph_deps(
        graph_deps: GraphDeps,
        toolsets: list[Any],
        *,
        state: GraphState,
    ) -> object:
        assert graph_deps is deps
        assert state is ctx.state
        adapted["toolsets"] = toolsets
        return adapted_deps

    monkeypatch.setattr(executor, "agent_deps_from_graph", adapt_graph_deps)

    result = await _router_impl.router_step(ctx)

    assert isinstance(result, End)
    assert result.data.results == {"output": "delegated result"}
    assert result.data.metadata["direct_dispatch"] is True
    assert constructed["permissions_kernel"] is kernel
    assert constructed["agent_identity"] is identity
    assert constructed["permission_engine"] is knowledge_engine
    assert len(constructed["toolsets"]) == 1
    assert isinstance(constructed["toolsets"][0].wrapped, ApprovalRequiredToolset)
    assert adapted["toolsets"] == constructed["toolsets"]
    agent.run.assert_awaited_once()
    assert agent.run.await_args.kwargs["deps"] is adapted_deps
    assert not hasattr(state, "permissions_kernel")
    assert not hasattr(state, "agent_identity")
    assert not hasattr(state, "knowledge_engine")


@pytest.mark.asyncio
async def test_router_direct_dispatch_permission_denial_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FunctionToolset(
        [_graph_lookup],
        id="denied-native-tools",
        metadata={"graphos_native": True},
    )
    kernel = SimpleNamespace(authorize_tool=lambda *_args, **_kwargs: "deny")
    state = GraphState(
        query="Attempt a denied delegated lookup",
        invoker_allowed_tools=["_graph_lookup"],
    )
    deps = GraphDeps(
        tag_prompts={"query": "Query specialist"},
        tag_env_vars={},
        mcp_toolsets=[native],
        agent_model=None,
        execution_shape=SimpleNamespace(run_discovery=False),
        permissions_kernel=kernel,
        agent_identity=object(),
        knowledge_engine=object(),
    )
    ctx = SimpleNamespace(state=state, deps=deps, inputs=None)

    class DeniedAgent:
        async def run(self, *_args: Any, **_kwargs: Any) -> None:
            scoped = constructed["toolsets"][0]
            approval = scoped.wrapped
            assert isinstance(approval, ApprovalRequiredToolset)
            approval.approval_required_func(
                SimpleNamespace(),
                SimpleNamespace(name="_graph_lookup", metadata={}),
                {"query": "denied"},
            )
            pytest.fail("the denied native tool unexpectedly executed")

    constructed: dict[str, Any] = {}

    def create_agent(**kwargs: Any) -> DeniedAgent:
        constructed.update(kwargs)
        return DeniedAgent()

    monkeypatch.setattr(_router_impl, "setting", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(_router_impl, "_emit_node_lifecycle", lambda *_args: None)
    monkeypatch.setattr(
        _router_impl, "emit_graph_event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(_router_impl, "create_context_agent", create_agent)
    monkeypatch.setattr(_router_impl, "spawn_usage_limits", lambda _state: object())
    monkeypatch.setattr(
        executor, "agent_deps_from_graph", lambda *_args, **_kwargs: object()
    )

    with pytest.raises(PermissionError, match="denied by identity policy"):
        await _router_impl.router_step(ctx)


@pytest.mark.asyncio
async def test_router_direct_dispatch_never_falls_back_past_usage_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FunctionToolset(
        [_graph_lookup],
        id="budgeted-native-tools",
        metadata={"graphos_native": True},
    )
    state = GraphState(
        query="Read one bounded delegated record",
        invoker_allowed_tools=["_graph_lookup"],
    )
    deps = GraphDeps(
        tag_prompts={"budgeted-native-tools": "Bounded synthetic skill"},
        tag_env_vars={},
        mcp_toolsets=[native],
        agent_model=None,
        execution_shape=SimpleNamespace(run_discovery=False),
        permissions_kernel=object(),
        agent_identity=object(),
    )
    ctx = SimpleNamespace(state=state, deps=deps, inputs=None)
    run = AsyncMock(side_effect=UsageLimitExceeded("synthetic budget exhausted"))

    monkeypatch.setattr(_router_impl, "setting", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(_router_impl, "_emit_node_lifecycle", lambda *_args: None)
    monkeypatch.setattr(
        _router_impl, "emit_graph_event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        _router_impl,
        "create_context_agent",
        lambda **_kwargs: SimpleNamespace(run=run),
    )
    monkeypatch.setattr(_router_impl, "spawn_usage_limits", lambda _state: object())
    monkeypatch.setattr(
        executor, "agent_deps_from_graph", lambda *_args, **_kwargs: object()
    )

    with pytest.raises(UsageLimitExceeded, match="synthetic budget exhausted"):
        await _router_impl.router_step(ctx)

    run.assert_awaited_once()


class _SyntheticMcpFallbackStream:
    usage = None

    async def __aenter__(self) -> _SyntheticMcpFallbackStream:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def stream_text(self, *, delta: bool):
        assert delta is True
        if False:
            yield "unreachable"

    async def get_output(self) -> str:
        return "fallback completed"

    def all_messages(self) -> list[Any]:
        return []


class _SyntheticMcpFallbackAgent:
    def run_stream(self, *_args: Any, **_kwargs: Any) -> _SyntheticMcpFallbackStream:
        return _SyntheticMcpFallbackStream()


@pytest.mark.asyncio
async def test_mcp_server_fallback_preserves_approval_before_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FunctionToolset(
        [_graph_lookup],
        id="fallback-native-tools",
        metadata={"graphos_native": True},
    )
    calls: list[tuple[str, str]] = []

    class ApprovalKernel:
        def authorize_tool(
            self,
            _identity: Any,
            tool_name: str,
            *,
            required_capability: str | None,
        ) -> str:
            assert required_capability is None
            calls.append(("authorize", tool_name))
            return "require_approval"

    state = GraphState(
        query="Use the fallback native lookup",
        invoker_allowed_tools=["_graph_lookup"],
    )
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[native],
        agent_model=None,
        permissions_kernel=ApprovalKernel(),
        agent_identity=object(),
    )
    ctx = SimpleNamespace(
        state=state,
        deps=deps,
        inputs="fallback-native-tools",
    )
    constructed: dict[str, Any] = {}

    def create_agent(**kwargs: Any) -> _SyntheticMcpFallbackAgent:
        constructed.update(kwargs)
        return _SyntheticMcpFallbackAgent()

    monkeypatch.setattr(
        _router_impl,
        "get_discovery_registry",
        lambda: SimpleNamespace(agents=[]),
    )
    monkeypatch.setattr(
        _router_impl, "emit_graph_event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(_router_impl, "create_context_agent", create_agent)

    result = await _router_impl.mcp_server_step(ctx)

    assert result == "execution_joiner"
    assert len(constructed["toolsets"]) == 1
    scoped = constructed["toolsets"][0]
    approval = scoped.wrapped
    assert isinstance(approval, ApprovalRequiredToolset)
    assert (
        approval.approval_required_func(
            SimpleNamespace(),
            SimpleNamespace(name="_graph_lookup", metadata={}),
            {"query": "approval"},
        )
        is True
    )
    assert calls == [("authorize", "_graph_lookup")]


class _SyntheticDynamicStream:
    async def __aenter__(self) -> _SyntheticDynamicStream:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def get_output(self) -> str:
        return "dynamic delegated result"


class _SyntheticDynamicAgent:
    def __init__(self) -> None:
        self.run_stream_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def run_stream(self, *args: Any, **kwargs: Any) -> _SyntheticDynamicStream:
        self.run_stream_calls.append((args, kwargs))
        return _SyntheticDynamicStream()


@pytest.mark.asyncio
async def test_dynamic_expert_fallback_guards_native_toolset_before_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FunctionToolset(
        [_graph_lookup],
        id="native-does-not-match-dynamic-node",
        metadata={"graphos_native": True},
    )
    kernel = object()
    identity = object()

    class KnowledgeEngine:
        def query_cypher(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            return []

    knowledge_engine = KnowledgeEngine()
    state = GraphState(
        query="Use the delegated native lookup",
        invoker_allowed_tools=["_graph_lookup"],
    )
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[native],
        agent_model=None,
        verifier_timeout=1,
        permissions_kernel=kernel,
        agent_identity=identity,
        knowledge_engine=knowledge_engine,
    )
    ctx = SimpleNamespace(
        state=state,
        deps=deps,
        inputs=ExecutionStep(
            id="planner-emitted-dynamic-node",
            description="Perform the delegated lookup",
        ),
    )
    events: list[tuple[str, Any]] = []
    scoped_toolset = object()

    class GuardedToolset:
        def filtered(self, predicate: Any) -> object:
            events.append(("allowlist", self))
            assert predicate(None, SimpleNamespace(name="_graph_lookup")) is True
            assert predicate(None, SimpleNamespace(name="other_tool")) is False
            return scoped_toolset

    guarded = GuardedToolset()

    def identity_guard(
        toolsets: list[Any],
        *,
        permissions_kernel: Any,
        agent_identity: Any,
        engine: Any,
    ) -> list[Any]:
        events.append(("identity", toolsets[0]))
        assert toolsets == [native]
        assert permissions_kernel is kernel
        assert agent_identity is identity
        assert engine is knowledge_engine
        return [guarded]

    class Validator:
        def validate_pre(self, *_args: Any) -> bool:
            return True

        def validate_post(self, *_args: Any) -> bool:
            return True

    class Locker:
        def fork_state(self, *_args: Any) -> None:
            return None

        def update_branch_state(self, *_args: Any) -> None:
            return None

        def merge_state(self, *_args: Any) -> bool:
            return True

    dynamic_agent = _SyntheticDynamicAgent()
    constructed: dict[str, Any] = {}
    adapted_deps = object()

    def create_agent(**kwargs: Any) -> _SyntheticDynamicAgent:
        constructed.update(kwargs)
        events.append(("construct", tuple(kwargs["toolsets"])))
        return dynamic_agent

    from agent_utilities.harness import contract_validator, distributed_state_manager

    monkeypatch.setattr(
        contract_validator.ContractValidator,
        "instance",
        classmethod(lambda _cls: Validator()),
    )
    monkeypatch.setattr(distributed_state_manager, "BranchMergeStateLocker", Locker)
    monkeypatch.setattr(
        executor,
        "_get_domain_tools",
        AsyncMock(return_value=([], [])),
    )
    monkeypatch.setattr(tool_guard, "flag_mcp_tool_definitions", identity_guard)
    monkeypatch.setattr(_router_impl, "create_context_agent", create_agent)
    monkeypatch.setattr(_router_impl, "spawn_usage_limits", lambda _state: object())

    def adapt_graph_deps(
        graph_deps: GraphDeps,
        toolsets: list[Any],
        *,
        state: GraphState,
    ) -> object:
        assert graph_deps is deps
        assert toolsets == [scoped_toolset]
        assert state is ctx.state
        return adapted_deps

    monkeypatch.setattr(executor, "agent_deps_from_graph", adapt_graph_deps)

    result = await _router_impl.expert_executor_step(ctx)

    assert result == "execution_joiner"
    assert events[:3] == [
        ("identity", native),
        ("allowlist", guarded),
        ("construct", (scoped_toolset,)),
    ]
    assert constructed["permissions_kernel"] is kernel
    assert constructed["agent_identity"] is identity
    assert constructed["permission_engine"] is knowledge_engine
    assert constructed["toolsets"] == [scoped_toolset]
    assert dynamic_agent.run_stream_calls[0][1]["deps"] is adapted_deps
    assert state.results_registry["planner-emitted-dynamic-node"] == (
        "dynamic delegated result"
    )


def test_graph_nodes_never_read_runtime_authority_from_graph_state() -> None:
    graph_root = Path(__file__).parents[3] / "agent_utilities" / "graph"
    forbidden = re.compile(
        r"\bctx\.state\.(?:permissions_kernel|agent_identity|knowledge_engine)\b"
    )
    violations = [
        f"{path.relative_to(graph_root)}:{line_number}"
        for path in graph_root.rglob("*.py")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if forbidden.search(line)
    ]

    assert violations == []


class _CaptureRunGraph:
    def __init__(self, mode: str, captures: dict[str, GraphDeps]) -> None:
        self._mode = mode
        self._captures = captures

    async def run(self, *, state: GraphState, deps: GraphDeps) -> GraphResponse:
        self._captures[self._mode] = deps
        assert not hasattr(state, "knowledge_engine")
        return GraphResponse(status="completed", results={"output": "done"})


class _EmptyGraphRun:
    async def __aenter__(self) -> _EmptyGraphRun:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    def __aiter__(self) -> _EmptyGraphRun:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _CaptureIterGraph:
    def __init__(self, captures: dict[str, GraphDeps]) -> None:
        self._captures = captures

    def iter(self, *, state: GraphState, deps: GraphDeps) -> _EmptyGraphRun:
        self._captures["iter"] = deps
        assert not hasattr(state, "knowledge_engine")
        return _EmptyGraphRun()


@pytest.mark.asyncio
async def test_all_graph_execution_modes_forward_configured_knowledge_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_engine = object()
    captures: dict[str, GraphDeps] = {}
    config = {
        "tag_prompts": {},
        "tag_env_vars": {},
        "mcp_toolsets": [],
        "router_model": None,
        "agent_model": None,
        "knowledge_engine": knowledge_engine,
    }

    monkeypatch.setattr(
        orchestration_engine_module,
        "create_model",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestration_engine_module,
        "get_discovery_registry",
        lambda: SimpleNamespace(agents=[]),
    )
    monkeypatch.setattr(orchestration_engine_module, "tracer", None)

    from agent_utilities.core.registry import service_adapter

    monkeypatch.setattr(
        service_adapter.ServiceRegistry,
        "instance",
        classmethod(lambda _cls: SimpleNamespace(initialize=lambda: 0)),
    )

    engine = AgentOrchestrationEngine(engine=knowledge_engine)
    execute_result = await engine.execute_graph(
        _CaptureRunGraph("execute", captures),
        config,
        query="execute",
        run_id="execute-run",
        streamdown=False,
    )
    stream_events = [
        event
        async for event in engine.stream_graph(
            _CaptureRunGraph("stream", captures),
            config,
            query="stream",
            run_id="stream-run",
        )
    ]
    iter_events = [
        event
        async for event in engine.iter_graph(
            _CaptureIterGraph(captures),
            config,
            query="iter",
            run_id="iter-run",
        )
    ]

    assert execute_result["status"] == "completed"
    assert stream_events[-1].startswith("data: ")
    assert isinstance(iter_events, list)
    assert set(captures) == {"execute", "stream", "iter"}
    assert all(
        graph_deps.knowledge_engine is knowledge_engine
        for graph_deps in captures.values()
    )
