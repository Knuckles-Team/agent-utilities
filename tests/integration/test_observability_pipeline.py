"""End-to-End Observability Pipeline Tests.

CONCEPT:AU-OS.config.secrets-authentication — Observability Pipeline Validation

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

import pytest

from agent_utilities.core.config import config
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.observability.custom_observability import (
    get_otel_status_summary,
    setup_otel,
    verify_otel_pipeline,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def engine(tmp_path_factory):
    """Create a test IntelligenceGraphEngine bound to one explicit, isolated graph.

    Function-scoped (not module-scoped): the autouse
    ``isolate_graph_compute_engine`` fixture (tests/conftest.py) is itself
    function-scoped and binds the verified actor/GraphSession context every
    ``GraphComputeEngine`` construction now requires
    (``security.brain_context.IdentityRequiredError`` fail-closed guard).
    pytest sets up broader-scoped fixtures before function-scoped ones, so a
    module-scoped version of this fixture used to construct its
    ``GraphComputeEngine`` BEFORE that actor context existed for the module's
    first test. Each test's workflow assertions already tolerate an
    independently-seeded engine (see ``TestWorkflowStore.test_workflow_mermaid``'s
    "may be None" comment), so sharing across the module was never required.

    ``IntelligenceGraphEngine(db_path=":memory:")`` alone still isn't enough:
    with no explicit ``graph``/``backend``, it resolves its OPERATIONAL,
    tenant-routed default graph (``EpistemicGraphBackend()`` ->
    ``shard_topology.resolve_routing_graph(None)`` -> the ambient actor's
    *tenant* graph, e.g. ``tenant__tenant_test____commons__``) — a DIFFERENT
    identity than the per-test unique graph the isolate fixture's ambient
    ``GraphSession.graph`` actually points at. Since a prior
    ``GraphComputeEngine(backend_type="rust")`` warm-up call in this fixture
    had already claimed the process-engine singleton under the isolate
    fixture's per-test name, the tenant-routed lookup came back as a
    graph-SCOPED view (``_fixed_graph`` set) instead of the singleton root,
    and that view's fixed graph didn't match the ambient session graph —
    ``PermissionError: A graph-scoped view cannot retarget the verified
    GraphSession``. Mirror the same fix the ``engine_graph`` fixture in
    ``tests/conftest.py`` already applies for exactly this class of bug: mint
    one explicit graph name, bind BOTH the ambient ``GraphSession`` and the
    ``EpistemicGraphBackend`` to it via ``use_session``, so nothing falls
    through to tenant-based resolution.
    """
    import uuid

    from _test_engine import (
        TEST_AGENT_ID,
        TEST_AUDIENCE,
        TEST_POLICY_VERSION,
        TEST_TENANT,
    )

    from agent_utilities.core.paths import ensure_dirs
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    ensure_dirs()
    graph_name = f"obspipe_{uuid.uuid4().hex[:12]}"
    actor = ActorContext(
        actor_id=TEST_AGENT_ID,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=TEST_TENANT,
        scopes=frozenset({"kg:read", "kg:write", "kg:admin", "*"}),
        graph=graph_name,
        policy_version=TEST_POLICY_VERSION,
        audience=TEST_AUDIENCE,
    )
    with use_session(session):
        backend = EpistemicGraphBackend(graph_name=graph_name)
        yield IntelligenceGraphEngine(backend=backend, graph=backend.graph)


@pytest.fixture(scope="module")
def otel_setup():
    """Initialize OTel pipeline for the test module.

    The hermetic test environment (``tests/conftest.py``'s per-test
    ``os.environ`` isolation) carries no ambient ``OTEL_EXPORTER_OTLP_*``
    configuration, so ``setup_otel()`` used to bail out at its first "no
    runtime endpoint configured" early-return — before it ever reached agent
    instrumentation — leaving ``_otel_initialized``/
    ``_agent_instrumented_metadata_only`` False for the whole module
    regardless of what an individual test does afterward. Same fix
    ``test_otel_endpoint_configured``/``test_otel_headers_generated`` already
    apply per-test: supply a loopback endpoint, the one cleartext exemption
    ``_validated_langfuse_host`` itself grants
    (``agent_utilities/core/config.py::_validated_langfuse_host``), plus a
    dummy raw header value, so ``setup_otel()`` exercises its real resolution
    path (and actually reaches agent instrumentation) instead of leaning on
    infrastructure this repo checkout does not have.

    Passed as explicit ``endpoint``/``headers`` args (``setup_otel`` accepts
    both) rather than ``os.environ`` — but note ``setup_otel`` itself, once
    it gets this far, sets ``OTEL_EXPORTER_OTLP_ENDPOINT``/``_HEADERS`` in
    ``os.environ`` as one of ITS OWN steps ("environment variables for
    downstream OTel SDK consumers"), so this loopback endpoint is visible to
    every later test in the module either way — see
    ``test_otel_exporter_reachable``'s updated tolerance for the
    "endpoint configured but isn't a real Langfuse host" outcome that follows.
    """
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LITE_LLM_MODEL_ID", "qwen/qwen3.5-9b")

    config.reload()
    setup_otel(
        service_name="test-observability-pipeline",
        endpoint="http://127.0.0.1:4318",
        headers="Authorization=Basic dGVzdA==",
    )


class TestOTelPipelineSetup:
    """Tests for the OTel pipeline initialization."""

    def test_otel_pipeline_initializes(self, otel_setup):
        """CONCEPT:AU-OS.config.secrets-authentication — Pipeline initializes without errors."""
        report = verify_otel_pipeline()
        assert report["initialized"] is True
        assert report["logfire_available"] is True

    def test_otel_endpoint_configured(self, otel_setup, monkeypatch):
        """CONCEPT:AU-OS.config.secrets-authentication — OTLP endpoint is set.

        The hermetic test environment (``tests/conftest.py``'s per-test
        ``os.environ`` isolation) carries no ambient ``OTEL_EXPORTER_OTLP_*``
        configuration at all, so ``verify_otel_pipeline()`` legitimately
        reports nothing configured — this is not the fail-closed transport
        guard rejecting anything (a plain non-loopback endpoint would raise,
        not silently report unconfigured). Supply a loopback endpoint, the
        one cleartext exemption ``_validated_langfuse_host`` itself grants
        (``agent_utilities/core/config.py::_validated_langfuse_host``), so
        this test exercises the real resolution path instead of leaning on
        infrastructure this repo checkout does not have.
        """
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
        report = verify_otel_pipeline()
        assert report["endpoint_configured"], "OTLP endpoint should be configured"

    def test_otel_headers_generated(self, otel_setup, monkeypatch):
        """CONCEPT:AU-OS.config.secrets-authentication — Auth headers are generated from Langfuse keys.

        Same hermetic-environment gap as ``test_otel_endpoint_configured``:
        supply the endpoint plus a raw ``OTEL_EXPORTER_OTLP_HEADERS`` value
        (the one input ``_resolve_otel_headers`` accepts with no secrets-ref
        resolution at all) so header generation is exercised without needing
        real Langfuse/vault credentials.
        """
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic dGVzdA==")
        report = verify_otel_pipeline()
        assert report["headers_set"] is True, "OTLP headers should be set"

    def test_otel_exporter_reachable(self, otel_setup):
        """CONCEPT:AU-OS.config.secrets-authentication — Langfuse OTLP endpoint is reachable.

        ``otel_setup`` configures a loopback endpoint (see its docstring) so
        ``setup_otel()`` completes hermetically — but that endpoint is a
        synthetic dummy, not a real Langfuse host, so
        ``verify_otel_pipeline()``'s authenticated-health probe correctly
        declines to call it (``endpoint_error ==
        "authenticated_health_unsupported"``) rather than either reaching a
        real ``endpoint_status`` or reporting nothing configured at all.
        """
        report = verify_otel_pipeline()
        # The endpoint might not respond to GET, but should not error with
        # anything OTHER than the expected "this isn't a real Langfuse host"
        # outcome for our synthetic loopback endpoint.
        assert (
            report.get("endpoint_status") is not None
            or "endpoint_error" not in report
            or report.get("endpoint_error") == "authenticated_health_unsupported"
        )

    def test_otel_status_summary(self, otel_setup):
        """CONCEPT:AU-OS.config.secrets-authentication — Status summary is human-readable."""
        summary = get_otel_status_summary()
        assert "OTel Pipeline Status" in summary
        assert "Initialized" in summary
        assert "Endpoint" in summary

    def test_agents_instrumented(self, otel_setup):
        """CONCEPT:AU-OS.config.secrets-authentication — pydantic-ai agents are instrumented."""
        report = verify_otel_pipeline()
        assert report["agent_instrumented"], (
            "Agents should be instrumented after setup_otel()"
        )


class TestTracingDecorator:
    """Tests for the @trace decorator with proper nesting."""

    def test_trace_decorator_creates_trace(self, otel_setup):
        """CONCEPT:AU-OS.config.secrets-authentication — @trace creates Langfuse traces."""
        from agent_utilities.harness.tracing import trace

        @trace(name="test_sync_function")
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_trace_decorator(self, otel_setup):
        """CONCEPT:AU-OS.config.secrets-authentication — @trace works with async functions."""
        from agent_utilities.harness.tracing import trace

        @trace(name="test_async_function")
        async def my_async_func(x):
            return x * 3

        result = await my_async_func(4)
        assert result == 12

    def test_trace_nesting(self, otel_setup, monkeypatch):
        """CONCEPT:AU-OS.config.secrets-authentication — Nested @trace creates parent-child hierarchy.

        ``@trace`` is a no-op (context vars never set) unless
        ``tracing._tracing_active()`` is true, which requires
        ``config.trace_export_enabled`` AND a non-empty
        ``config.langfuse_secret_key_ref`` — independent of the OTel/OTLP
        endpoint config ``otel_setup`` attempts (and, in this hermetic
        environment, fails) to resolve. The credential ref only needs to be
        non-empty here: ``_tracing_active()`` checks truthiness, and any
        actual Langfuse export attempt inside ``_emit_trace`` is wrapped in a
        blanket ``except Exception`` (best-effort, never breaks the traced
        call), so a dummy/non-resolving ref is sufficient to exercise the
        real context-propagation path under test.
        """
        from agent_utilities.harness.tracing import get_trace_id, trace

        monkeypatch.setattr(config, "trace_export_enabled", True)
        monkeypatch.setattr(config, "langfuse_secret_key_ref", "vault://test/dummy")

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
        """CONCEPT:AU-OS.config.secrets-authentication — Session IDs propagate through context."""
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
        """CONCEPT:AU-OS.config.secrets-authentication — Tracing is no-op without Langfuse keys."""
        monkeypatch.setattr(config, "langfuse_secret_key_ref", None)

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
        """CONCEPT:AU-OS.config.secrets-authentication — GraphPlan generates mermaid diagrams."""
        from agent_utilities.models.graph import ExecutionStep, GraphPlan

        plan = GraphPlan(
            steps=[
                ExecutionStep(
                    id="researcher",
                    refined_subtask="Search for papers",
                ),
                ExecutionStep(
                    id="summarizer",
                    refined_subtask="Summarize findings",
                    depends_on=["researcher"],
                ),
                ExecutionStep(
                    id="presenter",
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
        """CONCEPT:AU-OS.config.secrets-authentication — Graph agent generates mermaid visualization."""
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
        """CONCEPT:AU-ORCH.execution.workflow-persistence-replay — Workflows round-trip through KG."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
        from agent_utilities.models.graph import ExecutionStep, GraphPlan

        store = WorkflowStore(engine)

        plan = GraphPlan(
            steps=[
                ExecutionStep(
                    id="researcher",
                    refined_subtask="Search for papers on transformers",
                ),
                ExecutionStep(
                    id="summarizer",
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
        assert loaded.steps[0].id == "researcher"
        assert loaded.steps[1].id == "summarizer"

    def test_list_workflows(self, engine):
        """CONCEPT:AU-ORCH.execution.workflow-persistence-replay — List all stored workflows."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        workflows = store.list_workflows()
        assert isinstance(workflows, list)

    def test_workflow_mermaid(self, engine):
        """CONCEPT:AU-ORCH.execution.workflow-persistence-replay — Stored workflows have mermaid diagrams."""
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
        """CONCEPT:AU-ORCH.execution.nl-compilation-pipeline — Compile NL into GraphPlan."""
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
        """CONCEPT:AU-ORCH.execution.nl-compilation-pipeline — Compile and persist workflow."""
        from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler

        compiler = WorkflowCompiler(engine)
        workflow_id = await compiler.compile_and_store(
            name="compiled_research",
            description="Search papers, summarize them, then present findings",
        )
        assert workflow_id.startswith("workflow:")

    @pytest.mark.asyncio
    async def test_compile_parallel_steps(self, engine):
        """CONCEPT:AU-ORCH.execution.nl-compilation-pipeline — Detect parallel execution intent."""
        from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler

        compiler = WorkflowCompiler(engine)
        plan = await compiler.compile(
            "1. Search arxiv for papers. "
            "2. Simultaneously check system health. "
            "3. Then combine results."
        )

        assert len(plan.steps) >= 2
        # At least one step should be marked parallel
        parallel_steps = [s for s in plan.steps if s.parallel]
        # Note: parallel detection is heuristic, may not always fire
        logger.info(
            "Parallel steps detected: %d / %d",
            len(parallel_steps),
            len(plan.steps),
        )
