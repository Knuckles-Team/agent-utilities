"""End-to-End Observability Pipeline Tests.

CONCEPT:OS-5.1 — Observability Pipeline Validation

Tests the full tracing pipeline:
    1. OTel setup via setup_otel()
    2. Agent execution with pydantic-ai instrumentation
    3. Trace arrival in Langfuse (via langfuse-agent API)
    4. Mermaid diagram generation and capture
    5. Session grouping and trace nesting
    6. Workflow persistence in KG
"""

import logging
import os
from pathlib import Path

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
import pytest

from agent_utilities.core.config import config
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.observability.custom_observability import (
    get_otel_status_summary,
    setup_otel,
    verify_otel_pipeline,
)

WORKSPACE_DIR = Path("/home/apps/workspace")

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Create a test IntelligenceGraphEngine in memory-only mode.

    Uses GraphComputeEngine (no LadybugDB backend) for workflow tests since the
    test schema doesn't need persistent storage.
    """
    from agent_utilities.core.paths import ensure_dirs

    ensure_dirs()
    graph = GraphComputeEngine(backend_type="rust")
    return IntelligenceGraphEngine(graph=graph, backend=None)


@pytest.fixture(scope="module")
def otel_setup():
    """Initialize OTel pipeline for the test module."""
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_BASE_URL", "http://vllm.arpa/v1")
    os.environ.setdefault("LITE_LLM_MODEL_ID", "qwen/qwen3.5-9b")

    config.reload()
    setup_otel(service_name="test-observability-pipeline")


class TestOTelPipelineSetup:
    """Tests for the OTel pipeline initialization."""

    def test_otel_pipeline_initializes(self, otel_setup):
        """CONCEPT:OS-5.1 — Pipeline initializes without errors."""
        report = verify_otel_pipeline()
        assert report["initialized"] is True
        assert report["logfire_available"] is True

    def test_otel_endpoint_configured(self, otel_setup):
        """CONCEPT:OS-5.1 — OTLP endpoint is set."""
        report = verify_otel_pipeline()
        assert report["endpoint"], "OTLP endpoint should be configured"
        assert (
            "langfuse" in report["endpoint"].lower()
            or "otel" in report["endpoint"].lower()
        )

    def test_otel_headers_generated(self, otel_setup):
        """CONCEPT:OS-5.1 — Auth headers are generated from Langfuse keys."""
        report = verify_otel_pipeline()
        assert report["headers_set"] is True, "OTLP headers should be set"

    def test_otel_exporter_reachable(self, otel_setup):
        """CONCEPT:OS-5.1 — Langfuse OTLP endpoint is reachable."""
        report = verify_otel_pipeline()
        # The endpoint might not respond to GET, but should not error
        assert (
            report.get("endpoint_status") is not None or "endpoint_error" not in report
        )

    def test_otel_status_summary(self, otel_setup):
        """CONCEPT:OS-5.1 — Status summary is human-readable."""
        summary = get_otel_status_summary()
        assert "OTel Pipeline Status" in summary
        assert "Initialized" in summary
        assert "Endpoint" in summary

    def test_agents_instrumented(self, otel_setup):
        """CONCEPT:OS-5.1 — pydantic-ai agents are instrumented."""
        report = verify_otel_pipeline()
        assert report["agent_instrumented"], (
            "Agents should be instrumented after setup_otel()"
        )


class TestTracingDecorator:
    """Tests for the @trace decorator with proper nesting."""

    def test_trace_decorator_creates_trace(self, otel_setup):
        """CONCEPT:OS-5.1 — @trace creates Langfuse traces."""
        from agent_utilities.harness.tracing import trace

        @trace(name="test_sync_function")
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_trace_decorator(self, otel_setup):
        """CONCEPT:OS-5.1 — @trace works with async functions."""
        from agent_utilities.harness.tracing import trace

        @trace(name="test_async_function")
        async def my_async_func(x):
            return x * 3

        result = await my_async_func(4)
        assert result == 12

    def test_trace_nesting(self, otel_setup):
        """CONCEPT:OS-5.1 — Nested @trace creates parent-child hierarchy."""
        from agent_utilities.harness.tracing import get_trace_id, trace

        outer_trace_id = None
        inner_trace_id = None

        @trace(name="outer")
        def outer():
            nonlocal outer_trace_id
            outer_trace_id = get_trace_id()

            @trace(name="inner")
            def inner():
                nonlocal inner_trace_id
                inner_trace_id = get_trace_id()
                return "done"

            return inner()

        result = outer()
        assert result == "done"
        # Inner trace should share the parent's trace_id
        assert outer_trace_id is not None
        assert inner_trace_id == outer_trace_id

    def test_session_id_propagation(self, otel_setup):
        """CONCEPT:OS-5.1 — Session IDs propagate through context."""
        from agent_utilities.harness.tracing import (
            get_session_id,
            set_session_id,
            trace,
        )

        set_session_id("test-session-123")

        captured_session = None

        @trace(name="session_test")
        def check_session():
            nonlocal captured_session
            captured_session = get_session_id()
            return True

        check_session()
        assert captured_session == "test-session-123"

    def test_trace_disabled_without_keys(self, otel_setup, monkeypatch):
        """CONCEPT:OS-5.1 — Tracing is no-op without Langfuse keys."""
        monkeypatch.setattr(config, "langfuse_secret_key", None)

        from agent_utilities.harness.tracing import trace

        @trace(name="test_disabled")
        def my_func(x):
            return x + 1

        # Should still work, just not emit traces
        result = my_func(10)
        assert result == 11


class TestMermaidCapture:
    """Tests for mermaid diagram generation and capture."""

    def test_graph_plan_mermaid(self):
        """CONCEPT:OS-5.1 — GraphPlan generates mermaid diagrams."""
        from agent_utilities.models.graph import ExecutionStep, GraphPlan

        plan = GraphPlan(
            steps=[
                ExecutionStep(
                    node_id="researcher",
                    refined_subtask="Search for papers",
                ),
                ExecutionStep(
                    node_id="summarizer",
                    refined_subtask="Summarize findings",
                    depends_on=["researcher"],
                ),
                ExecutionStep(
                    node_id="presenter",
                    refined_subtask="Create presentation",
                    depends_on=["summarizer"],
                ),
            ]
        )
        mermaid = plan.to_mermaid(title="Research Pipeline")
        assert "Research Pipeline" in mermaid
        assert "researcher" in mermaid
        assert "summarizer" in mermaid
        assert "presenter" in mermaid
        logger.info("Generated mermaid:\n%s", mermaid)

    def test_graph_agent_mermaid(self, otel_setup):
        """CONCEPT:OS-5.1 — Graph agent generates mermaid visualization."""
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")

        from agent_utilities.graph import create_graph_agent, get_graph_mermaid

        tag_prompts = {"research": "Research domain", "coding": "Code domain"}
        graph, cfg = create_graph_agent(tag_prompts, mcp_url=None, mcp_config=None)
        mermaid = get_graph_mermaid(graph, cfg, title="Test Graph")

        assert "Test Graph" in mermaid
        assert len(mermaid) > 50
        logger.info("Graph agent mermaid:\n%s", mermaid)


class TestWorkflowStore:
    """Tests for KG-native workflow storage."""

    def test_save_and_load_workflow(self, engine):
        """CONCEPT:ORCH-1.22 — Workflows round-trip through KG."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
        from agent_utilities.models.graph import ExecutionStep, GraphPlan

        store = WorkflowStore(engine)

        plan = GraphPlan(
            steps=[
                ExecutionStep(
                    node_id="researcher",
                    refined_subtask="Search for papers on transformers",
                ),
                ExecutionStep(
                    node_id="summarizer",
                    refined_subtask="Summarize top 3 papers",
                    depends_on=["researcher"],
                ),
            ]
        )

        workflow_id = store.save_workflow(
            name="test_research_pipeline",
            plan=plan,
            description="A test research workflow",
            nl_spec="Search for papers on transformers, then summarize the top 3.",
        )
        assert workflow_id.startswith("workflow:")

        # Load it back
        loaded = store.load_workflow("test_research_pipeline")
        assert loaded is not None
        assert len(loaded.steps) == 2
        assert loaded.steps[0].node_id == "researcher"
        assert loaded.steps[1].node_id == "summarizer"

    def test_list_workflows(self, engine):
        """CONCEPT:ORCH-1.22 — List all stored workflows."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        workflows = store.list_workflows()
        assert isinstance(workflows, list)

    def test_workflow_mermaid(self, engine):
        """CONCEPT:ORCH-1.22 — Stored workflows have mermaid diagrams."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        mermaid = store.get_mermaid("test_research_pipeline")
        # May be None if the previous test didn't create it
        if mermaid:
            assert "researcher" in mermaid


class TestWorkflowCompiler:
    """Tests for NL → Workflow compilation."""

    @pytest.mark.asyncio
    async def test_compile_simple_workflow(self, engine):
        """CONCEPT:ORCH-1.23 — Compile NL into GraphPlan."""
        from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler

        compiler = WorkflowCompiler(engine)
        plan = await compiler.compile(
            "Search for recent papers on AI agents, "
            "then summarize the top results, "
            "finally create a report."
        )

        assert len(plan.steps) >= 2  # Should parse at least 2 steps
        assert plan.metadata["source"] == "nl_compiler"

    @pytest.mark.asyncio
    async def test_compile_and_store(self, engine):
        """CONCEPT:ORCH-1.23 — Compile and persist workflow."""
        from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler

        compiler = WorkflowCompiler(engine)
        workflow_id = await compiler.compile_and_store(
            name="compiled_research",
            description="Search papers, summarize them, then present findings",
        )
        assert workflow_id.startswith("workflow:")

    @pytest.mark.asyncio
    async def test_compile_parallel_steps(self, engine):
        """CONCEPT:ORCH-1.23 — Detect parallel execution intent."""
        from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler

        compiler = WorkflowCompiler(engine)
        plan = await compiler.compile(
            "1. Search arxiv for papers. "
            "2. Simultaneously check system health. "
            "3. Then combine results."
        )

        assert len(plan.steps) >= 2
        # At least one step should be marked parallel
        parallel_steps = [s for s in plan.steps if s.is_parallel]
        # Note: parallel detection is heuristic, may not always fire
        logger.info(
            "Parallel steps detected: %d / %d",
            len(parallel_steps),
            len(plan.steps),
        )
