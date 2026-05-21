"""Unified MCP Orchestration & Workflow Integration Tests.

CONCEPT:ORCH-1.24 — Workflow Lifecycle Testing

Tests the full workflow lifecycle:
    1. YAML catalog loading and validation
    2. GraphPlan conversion and mermaid generation
    3. KG registration with auto-versioning
    4. Live agent execution with full Langfuse tracing
    5. Multi-step pipeline execution (parallel + sequential)
    6. Workflow discovery and round-trip persistence

All live tests use real LLM (LM Studio) and real MCP servers.
Structural tests can run without any external services.
"""

import json
import logging
import os
from pathlib import Path

import networkx as nx
import pytest

from agent_utilities.core.paths import ensure_dirs
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.observability.custom_observability import setup_otel

WORKSPACE_DIR = Path("/home/apps/workspace")
USER_MCP_CONFIG = Path("/home/genius/.gemini/antigravity/mcp_config.json")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Create a test IntelligenceGraphEngine in memory-only mode.

    Uses NetworkX (no LadybugDB) for fast, isolated workflow tests.
    """
    ensure_dirs()
    graph = nx.MultiDiGraph()
    return IntelligenceGraphEngine(graph=graph, backend=None)


@pytest.fixture(autouse=True)
def allow_live_models(request):
    """Allow live models if the test is marked as live."""
    if "live" in request.keywords:
        original = os.environ.get("AGENT_UTILITIES_TESTING")
        os.environ.pop("AGENT_UTILITIES_TESTING", None)
        yield
        if original is not None:
            os.environ["AGENT_UTILITIES_TESTING"] = original
    else:
        yield


@pytest.fixture(scope="module")
def otel_setup():
    """Initialize OTel pipeline for the test module.

    Ensures ENABLE_OTEL=true, pops OTEL_SDK_DISABLED, reloads config,
    and passes Langfuse OTLP keys directly to setup_otel().
    """
    os.environ["ENABLE_OTEL"] = "true"
    os.environ.pop("OTEL_SDK_DISABLED", None)
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_BASE_URL", "http://10.0.0.18:1234/v1")
    os.environ.setdefault("LITE_LLM_MODEL_ID", "qwen/qwen3.5-9b")

    # Extract Langfuse variables from user's mcp_config.json
    mcp_config_path = (
        USER_MCP_CONFIG
        if USER_MCP_CONFIG.exists()
        else Path(__file__).resolve().parents[2]
        / "docs"
        / "examples"
        / "example_mcp_config.json"
    )
    if mcp_config_path.exists():
        with open(mcp_config_path) as f:
            data = json.load(f)
            servers = data.get("mcpServers", {})
            if "langfuse-mcp" in servers:
                env_vars = servers["langfuse-mcp"].get("env", {})
                for k, v in env_vars.items():
                    if k.startswith("LANGFUSE_"):
                        os.environ.setdefault(k, str(v))

    from agent_utilities.core.config import config

    config.reload()

    # Pass keys explicitly to bypass stale module-level defaults
    setup_otel(
        service_name="test-workflow-orchestration",
        endpoint=config.otel_exporter_otlp_endpoint,
        public_key=config.otel_exporter_otlp_public_key or config.langfuse_public_key,
        secret_key=config.otel_exporter_otlp_secret_key or config.langfuse_secret_key,
    )

    logger.info(
        "OTel initialized: endpoint=%s, public_key=%s",
        config.otel_exporter_otlp_endpoint,
        (config.otel_exporter_otlp_public_key or config.langfuse_public_key or "")[:20],
    )

    return mcp_config_path


@pytest.fixture(scope="module")
def catalog():
    """Load the built-in workflow catalog."""
    from agent_utilities.workflows.catalog import WorkflowCatalog

    return WorkflowCatalog.load()


# ═══════════════════════════════════════════════════════════════════════
# Structural Tests — No LLM Required
# ═══════════════════════════════════════════════════════════════════════


class TestWorkflowCatalogStructure:
    """Tests for catalog loading, parsing, and conversion."""

    def test_catalog_loads_from_yaml(self, catalog):
        """CONCEPT:ORCH-1.24 — YAML catalog loads successfully."""
        assert len(catalog.scenarios) > 0
        logger.info("Loaded %d scenarios from catalog", len(catalog.scenarios))
        for s in catalog.scenarios:
            logger.info("  [%s] %s — %d steps", s.domain, s.name, len(s.steps))

    def test_every_scenario_has_required_fields(self, catalog):
        """CONCEPT:ORCH-1.24 — All scenarios have required fields."""
        for s in catalog.scenarios:
            assert s.name, f"Scenario missing name"
            assert s.description, f"Scenario '{s.name}' missing description"
            assert s.domain, f"Scenario '{s.name}' missing domain"
            assert len(s.steps) > 0, f"Scenario '{s.name}' has no steps"
            assert len(s.tags) > 0, f"Scenario '{s.name}' has no tags"
            assert len(s.requires) > 0, f"Scenario '{s.name}' has no requires"

    def test_every_step_has_agent_and_task(self, catalog):
        """CONCEPT:ORCH-1.24 — All steps specify agent and task."""
        for s in catalog.scenarios:
            for i, step in enumerate(s.steps):
                assert step.agent, f"Step {i} in '{s.name}' missing agent"
                assert step.task, f"Step {i} in '{s.name}' missing task"

    def test_dependencies_are_valid(self, catalog):
        """CONCEPT:ORCH-1.24 — Step dependencies reference valid indices."""
        for s in catalog.scenarios:
            for i, step in enumerate(s.steps):
                for dep in step.depends_on:
                    assert 0 <= dep < len(s.steps), (
                        f"Step {i} in '{s.name}' depends_on[{dep}] "
                        f"out of range [0, {len(s.steps)})"
                    )
                    assert dep != i, f"Step {i} in '{s.name}' depends on itself"

    def test_catalog_to_graph_plans(self, catalog):
        """CONCEPT:ORCH-1.24 — All scenarios convert to valid GraphPlans."""
        plans = catalog.to_graph_plans()
        assert len(plans) == len(catalog.scenarios)

        for name, plan in plans.items():
            assert len(plan.steps) > 0, f"Plan '{name}' has no steps"
            assert plan.metadata.get("source") == "workflow_catalog"
            assert plan.metadata.get("scenario_name") == name
            logger.info("Plan '%s': %d steps", name, len(plan.steps))

    def test_mermaid_generation_for_all_plans(self, catalog):
        """CONCEPT:ORCH-1.24 — All plans generate valid mermaid diagrams."""
        plans = catalog.to_graph_plans()
        for name, plan in plans.items():
            mermaid = plan.to_mermaid(title=f"Workflow: {name}")
            assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()
            assert name in mermaid
            logger.info("Mermaid for '%s':\n%s", name, mermaid)

    def test_catalog_filter_by_tag(self, catalog):
        """CONCEPT:ORCH-1.24 — Tag-based filtering works."""
        docker_workflows = catalog.filter_by_tag("docker")
        assert len(docker_workflows) > 0
        for w in docker_workflows:
            assert "docker" in w.tags

    def test_catalog_filter_by_domain(self, catalog):
        """CONCEPT:ORCH-1.24 — Domain-based filtering works."""
        infra_workflows = catalog.filter_by_domain("infrastructure")
        for w in infra_workflows:
            assert w.domain == "infrastructure"

    def test_catalog_get_by_name(self, catalog):
        """CONCEPT:ORCH-1.24 — Name lookup returns the correct scenario."""
        scenario = catalog.get("container_health_check")
        assert scenario is not None
        assert scenario.name == "container_health_check"
        assert scenario.domain == "infrastructure"

    def test_catalog_export_json(self, catalog, tmp_path):
        """CONCEPT:ORCH-1.24 — JSON export produces valid output."""
        export_path = tmp_path / "workflows.json"
        result_path = catalog.export_json(export_path)
        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["workflow_count"] == len(catalog.scenarios)
        assert len(data["workflows"]) == len(catalog.scenarios)

    def test_catalog_export_yaml(self, catalog, tmp_path):
        """CONCEPT:ORCH-1.24 — YAML export produces valid output."""
        export_path = tmp_path / "workflows.yaml"
        result_path = catalog.export_yaml(export_path)
        assert result_path.exists()

        import yaml

        with open(result_path) as f:
            data = yaml.safe_load(f)

        assert len(data["workflows"]) == len(catalog.scenarios)

    def test_catalog_summary(self, catalog):
        """CONCEPT:ORCH-1.24 — Summary is human-readable."""
        summary = catalog.summary()
        assert "Workflow Catalog" in summary
        assert "container_health_check" in summary
        logger.info("Catalog summary:\n%s", summary)


class TestWorkflowKGPersistence:
    """Tests for KG registration and round-trip persistence."""

    def test_register_catalog_in_kg(self, catalog, engine):
        """CONCEPT:ORCH-1.24 — All catalog workflows persist to KG."""
        workflow_ids = catalog.register_in_kg(engine)
        assert len(workflow_ids) == len(catalog.scenarios)
        for wid in workflow_ids:
            assert wid.startswith("workflow:")
        logger.info("Registered %d workflows: %s", len(workflow_ids), workflow_ids)

    def test_workflow_round_trip(self, catalog, engine):
        """CONCEPT:ORCH-1.22 — Workflows survive save→load round-trip."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)

        # Load the container_health_check we just registered
        loaded = store.load_workflow("container_health_check")
        assert loaded is not None
        original = catalog.get("container_health_check")
        assert loaded is not None and original is not None
        assert len(loaded.steps) == len(original.steps)

        # Verify step node_ids match
        for orig_step, loaded_step in zip(original.steps, loaded.steps):
            assert loaded_step.node_id == orig_step.agent

    def test_list_workflows_from_kg(self, engine):
        """CONCEPT:ORCH-1.22 — List stored workflows from KG."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        workflows = store.list_workflows()
        assert isinstance(workflows, list)
        # We registered all catalog workflows above
        logger.info("KG contains %d workflows", len(workflows))

    def test_auto_version_increment(self, catalog, engine):
        """CONCEPT:ORCH-1.24 — Re-registration increments version."""
        # Register again — should auto-increment version
        workflow_ids = catalog.register_in_kg(engine)
        assert len(workflow_ids) == len(catalog.scenarios)
        logger.info("Re-registered with auto-increment: %s", workflow_ids)

    def test_workflow_discovery_by_domain(self, engine):
        """CONCEPT:ORCH-1.24 — Find workflows by NX graph traversal."""
        # Search for infrastructure workflows via NX graph
        infra_workflows = []
        for nid, data in engine.graph.nodes(data=True):
            if data.get("type") == "WorkflowDefinition" and "infrastructure" in str(
                data.get("metadata_json", "")
            ):
                infra_workflows.append(data.get("name", nid))

        logger.info("Infrastructure workflows in NX: %s", infra_workflows)
        assert len(infra_workflows) > 0


class TestWorkflowRunner:
    """Tests for the WorkflowRunner execution engine."""

    def test_runner_wave_builder(self, catalog):
        """CONCEPT:ORCH-1.24 — Execution waves respect dependencies."""
        from agent_utilities.workflows.runner import WorkflowRunner

        runner = WorkflowRunner()
        scenario = catalog.get("container_health_check")
        assert scenario is not None

        plan = scenario.to_graph_plan()
        waves = runner._build_execution_waves(plan)

        assert len(waves) >= 1
        logger.info("Waves for container_health_check: %s", waves)

        # First wave should contain steps with no dependencies
        for step_idx in waves[0]:
            step = plan.steps[step_idx]
            assert len(step.depends_on) == 0 or step_idx == 0

    def test_runner_wave_parallel_detection(self, catalog):
        """CONCEPT:ORCH-1.24 — Parallel steps end up in same wave."""
        from agent_utilities.workflows.runner import WorkflowRunner

        runner = WorkflowRunner()
        scenario = catalog.get("full_ecosystem_health")
        assert scenario is not None

        plan = scenario.to_graph_plan()
        waves = runner._build_execution_waves(plan)

        # The ecosystem health check has 4 parallel steps
        # They should all be in wave 0
        assert len(waves[0]) >= 2
        logger.info(
            "Full ecosystem health waves: %s (wave 0 has %d parallel steps)",
            waves,
            len(waves[0]),
        )


# ═══════════════════════════════════════════════════════════════════════
# Live Tests — Requires LM Studio + MCP Servers
# ═══════════════════════════════════════════════════════════════════════


# Legacy TEST_CASES for backward compatibility — these are a subset of
# the catalog scenarios, kept for quick single-agent validation.
TEST_CASES = [
    (
        "repository-manager-mcp",
        "Can you use the rm_workspace tool to list the available actions for the workspace?",
        ["list", "setup"],
    ),
    (
        "scholarx-mcp",
        "Can you use the sx_info tool to list the categories?",
        ["categories"],
    ),
    (
        "container-manager-mcp",
        "Can you list all docker images, list all running containers, get the logs for one of the running containers, show the volumes, and show the networks using your tools?",
        ["network", "volume"],
    ),
    (
        "audio-transcriber",
        "Can you describe the capabilities of the transcribe_audio tool?",
        ["transcribe"],
    ),
    (
        "tunnel-manager",
        "Can you list the active tunnels from the inventory using your tools?",
        ["tunnel"],
    ),
    (
        "systems-manager",
        "Can you get the system memory and CPU stats?",
        ["memory", "cpu"],
    ),
    (
        "data-science-mcp",
        "Can you describe the iris dataset using the describe_dataset tool?",
        ["dataset"],
    ),
    (
        "langfuse-mcp",
        "Can you check the langfuse health endpoint or list current projects/datasets using your tools?",
        ["health", "project", "dataset"],
    ),
]


@pytest.fixture(scope="module")
def live_engine(tmp_path_factory, otel_setup):
    """Create a test engine with LadybugDB for live tests."""
    from agent_utilities.knowledge_graph.backends import create_backend

    ensure_dirs()
    db_dir = tmp_path_factory.mktemp("ladybug_db")
    db_path = db_dir / "ladybug.db"
    os.environ["GRAPH_DB_PATH"] = str(db_path)

    backend = create_backend(backend_type="ladybug", db_path=str(db_path))
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=backend)
    return engine


@pytest.fixture(scope="module", autouse=False)
async def ingest_mcp_config(live_engine, otel_setup):
    """Ingest MCP config into the live engine."""
    mcp_config_path = otel_setup
    logger.info("Ingesting MCP Config: %s", mcp_config_path)
    if Path(mcp_config_path).exists():
        await live_engine.ingest_agent_toolkit([str(mcp_config_path)])


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("server, task, expected_keywords", TEST_CASES)
async def test_single_agent_execution(
    live_engine,
    otel_setup,
    ingest_mcp_config,
    server,
    task,
    expected_keywords,
    caplog,
):
    """CONCEPT:ORCH-1.21 — Single agent execution with full tracing."""
    from agent_utilities.orchestration.agent_runner import run_agent

    caplog.set_level(logging.INFO)

    logger.info("--- Starting Execution for: %s ---", server)
    logger.info("Task: %s", task)

    result = await run_agent(
        agent_name=server,
        task=task,
        max_steps=5,
        engine=live_engine,
    )

    logger.info("--- Final Result for %s ---", server)
    logger.info(result)

    assert result is not None, f"Agent execution for {server} returned None"

    result_lower = str(result).lower()
    assert "agent execution failed" not in result_lower, (
        f"Execution completely failed for {server}"
    )

    # Soft keyword validation
    found_any = any(kw in result_lower for kw in expected_keywords)
    if expected_keywords and not found_any:
        logger.warning(
            "None of expected keywords %s found in output for %s",
            expected_keywords,
            server,
        )

    assert len(caplog.records) > 0, "No logs were captured during execution."


# ═══════════════════════════════════════════════════════════════════════
# Live Workflow Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════


LIVE_WORKFLOW_SCENARIOS = [
    "workspace_inventory",
    "capability_discovery",
    "system_observability_sweep",
]


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name", LIVE_WORKFLOW_SCENARIOS)
async def test_live_workflow_execution(
    catalog,
    live_engine,
    otel_setup,
    ingest_mcp_config,
    workflow_name,
    caplog,
):
    """CONCEPT:ORCH-1.24 — End-to-end workflow execution with tracing."""
    from agent_utilities.workflows.runner import WorkflowRunner

    caplog.set_level(logging.INFO)

    scenario = catalog.get(workflow_name)
    assert scenario is not None, f"Scenario '{workflow_name}' not found in catalog"

    # Register workflow in KG first
    catalog_subset = type(catalog)(scenarios=[scenario])
    catalog_subset.register_in_kg(live_engine)

    # Execute
    runner = WorkflowRunner(max_steps_per_agent=5)
    plan = scenario.to_graph_plan()
    result = await runner.execute(
        plan=plan,
        engine=live_engine,
        workflow_name=workflow_name,
    )

    logger.info("--- Workflow Result ---\n%s", result.summary())
    logger.info("--- Mermaid ---\n%s", result.mermaid)

    assert result.status in ("completed", "partial")
    assert result.completed_steps >= 1
    assert result.total_duration_ms > 0
    assert result.mermaid  # Non-empty mermaid

    # Log individual step outputs for observability
    for sr in result.step_results:
        logger.info(
            "[Step %d] %s → %s (%.0fms): %.200s",
            sr.step_index,
            sr.node_id,
            sr.status,
            sr.duration_ms,
            sr.output,
        )
