"""CONCEPT:AU-ORCH.execution.inject-signal-board-observations"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.graph.state import GraphState
from agent_utilities.models import GraphResponse
from agent_utilities.orchestration import AgentOrchestrationEngine as runner


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.run = AsyncMock()
    return graph


@pytest.mark.asyncio
async def test_run_graph_basic(mock_graph):
    # Mock graph.run result - it should return a GraphResponse
    mock_result = GraphResponse(status="completed", results={"output": "final answer"})
    mock_graph.run.return_value = mock_result

    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None
    config = {"deps": deps}

    response = await runner().execute_graph(mock_graph, config, query="hello")

    assert response["status"] == "completed"
    assert response["results"]["output"] == "final answer"
    assert response["metadata"]["execution_mode"] == "pydantic_graph"
    assert response["execution_evidence"]["node_sequence"] == []
    assert response["execution_evidence"]["resume_supported"] is False
    mock_graph.run.assert_called_once()


@pytest.mark.asyncio
async def test_actual_graph_run_uses_a_pydantic_graph_span(
    mock_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the real ``Graph.run`` call is represented as ``pydantic_graph.run``."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from agent_utilities.orchestration import engine as engine_module

    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        engine_module, "tracer", provider.get_tracer("test-pydantic-graph-span")
    )

    async def capture_active_span(*, state, deps):
        span = otel_trace.get_current_span()
        assert span.name == "pydantic_graph.run"
        return GraphResponse(status="completed", results={"output": "done"})

    mock_graph.run.side_effect = capture_active_span
    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None

    response = await runner().execute_graph(
        mock_graph, {"deps": deps}, query="hello", run_id="synthetic-run"
    )

    assert response["metadata"]["execution_mode"] == "pydantic_graph"
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["pydantic_graph.run"]
    assert spans[0].attributes["agent_utilities.execution.mode"] == "pydantic_graph"


@pytest.mark.asyncio
async def test_run_graph_exception(mock_graph):
    mock_graph.run.side_effect = Exception("test error")

    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    config = {"deps": deps}

    response = await runner().execute_graph(mock_graph, config, query="hello")

    assert response["status"] == "error"
    assert "test error" in response["error"]


@pytest.mark.asyncio
async def test_run_graph_exception_preserves_completed_tool_provenance(mock_graph):
    """A failure after a tool call must not erase the completed side effect."""

    async def tool_then_fail(*, state, deps):
        del deps
        state.tool_calls.append(
            {
                "tool_name": "servicenow_get_incidents",
                "args": '{"limit": 1}',
                "result": "one record",
                "error": "",
            }
        )
        raise Exception("synthetic budget exhausted")

    mock_graph.run.side_effect = tool_then_fail
    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None

    response = await runner().execute_graph(
        mock_graph,
        {"deps": deps},
        query="read one incident",
    )

    assert response["status"] == "error"
    assert response["tool_calls"] == [
        {
            "tool_name": "servicenow_get_incidents",
            "args": '{"limit": 1}',
            "result": "one record",
            "error": "",
        }
    ]


@pytest.mark.asyncio
async def test_run_graph_terminal_error_recovery_dict_surfaces_as_failed(mock_graph):
    """CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — a terminal
    ``error_recovery_step`` ``End({"error": ..., "results": {...}})`` (retries exhausted /
    a terminal policy violation) is a REAL failure, not a completed answer. Before this fix
    it fell through to the catch-all, which stringified the whole dict into
    ``results.output`` under ``status="completed"`` — presenting a raw Python-dict-repr as if
    it were a normal reply (the "some sort of failure" black box). It must now surface as a
    failed GraphResponse with the real cause in `error` and a coherent `results.output`."""
    mock_graph.run.return_value = {
        "error": "Execution budget exceeded: max node transitions.",
        "results": {"specialist_0": "partial finding"},
    }

    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None
    config = {"deps": deps}

    response = await runner().execute_graph(mock_graph, config, query="hello")

    assert response["status"] == "failed"
    assert "Execution budget exceeded" in response["error"]
    assert "Execution budget exceeded" in response["results"]["output"]
    assert response["metadata"]["degraded"] is True
    assert response["metadata"]["outcome"] == "graph_terminal_error"
    # Never the raw Python-dict-repr of the whole terminal payload.
    assert "'error':" not in response["results"]["output"]


@pytest.mark.asyncio
async def test_execute_graph_constructs_permission_context_on_deps(mock_graph):
    kernel = object()
    identity = object()
    captured = {}

    async def capture_construction(*, state, deps):
        captured["state"] = state
        captured["deps"] = deps
        return GraphResponse(status="completed", results={"output": "done"})

    mock_graph.run.side_effect = capture_construction
    config = {
        "tag_prompts": {},
        "tag_env_vars": {},
        "mcp_toolsets": [],
        "router_model": None,
        "agent_model": None,
        "permissions_kernel": kernel,
        "agent_identity": identity,
        "response_format": "json",
    }

    with (
        patch(
            "agent_utilities.orchestration.engine.get_discovery_registry"
        ) as mock_registry,
        patch("agent_utilities.orchestration.engine.create_model", return_value=None),
        patch(
            "agent_utilities.orchestration.engine.GraphState", wraps=GraphState
        ) as state_factory,
        patch(
            "agent_utilities.core.registry.service_adapter.ServiceRegistry.instance"
        ) as service_registry,
    ):
        mock_registry.return_value.agents = []
        service_registry.return_value.initialize.return_value = 0
        response = await runner().execute_graph(
            mock_graph,
            config,
            query="synthetic request",
            run_id="synthetic-run",
            streamdown=False,
        )

    assert response["status"] == "completed"
    assert state_factory.call_count == 1
    assert isinstance(captured["state"], GraphState)
    assert captured["state"].session_id == "synthetic-run"
    assert captured["deps"].permissions_kernel is kernel
    assert captured["deps"].agent_identity is identity
    assert captured["deps"].response_format == "json"
    assert not hasattr(captured["state"], "permissions_kernel")
    assert not hasattr(captured["state"], "agent_identity")


@pytest.mark.asyncio
async def test_run_graph_stream_basic(mock_graph):
    # Mock graph.run_stream
    mock_deps = MagicMock()
    mock_deps.mcp_toolsets = []
    mock_deps.tag_prompts = {}
    mock_deps.event_queue = None
    mock_deps.request_id = "test-run"

    config = {"deps": mock_deps, "router_model": "test", "agent_model": "test"}

    # Mock graph.run for the background task
    mock_graph.run = AsyncMock(
        return_value=GraphResponse(status="completed", results={"output": "bg done"})
    )

    class MockStreamedRun:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> bool:
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "_yielded"):
                raise StopAsyncIteration
            self._yielded = True
            # Return a serializable object
            mock_chunk = MagicMock()
            mock_chunk.timestamp = 123456789
            # pydantic-graph chunks often have a .data or similar.
            # In runner.py it seems it just yields the chunk serialized if it's an event.
            return {"type": "chunk", "content": "hello"}

        async def result(self):
            return GraphResponse(status="completed", results={"output": "stream done"})

    mock_graph.run_stream.return_value = MockStreamedRun()

    with (
        patch(
            "agent_utilities.orchestration.engine.get_discovery_registry"
        ) as mock_reg,
        patch("agent_utilities.orchestration.engine.create_model") as mock_model,
    ):
        mock_reg.return_value.agents = []
        mock_model.return_value = MagicMock()

        # We need to test run_graph_stream generator
        stream = runner().stream_graph(mock_graph, config, query="hello")

        results = []
        async for chunk in stream:
            results.append(chunk)

    assert len(results) > 0
    # The runner might not include "stream done" in the SSE stream directly
    # if it's only in the final holder, but let's check what it yields.
    # It yields events from the queue.
    assert any("graph_start" in str(r) for r in results)
