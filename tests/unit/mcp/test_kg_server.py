"""Test suite for the Knowledge Graph MCP Server tools (graph-os).

CONCEPT:AU-ECO.mcp.fastmcp-middleware — KG MCP Server & Execution

Tests use the synthesized graph-os tool names: graph_query, graph_search,
graph_write, graph_ingest, graph_analyze, graph_orchestrate, graph_configure.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class MockMCP:
    def __init__(self):
        self.funcs = {}

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.funcs[func.__name__] = func
            return func

        return decorator

    def custom_route(self, *args, **kwargs):
        # Liveness/health endpoints registered via FastMCP's custom_route; record
        # them like tools so _build_server runs without a real ASGI app.
        def decorator(func):
            self.funcs[func.__name__] = func
            return func

        return decorator


@pytest.fixture
def server_tools():
    mock_mcp = MockMCP()
    build_engine = MagicMock()
    build_engine.backend = MagicMock()
    build_engine.backend.read_only = False
    with patch(
        "agent_utilities.mcp.server_factory.create_mcp_server",
        return_value=(None, mock_mcp, []),
    ):
        with patch(
            "agent_utilities.mcp.kg_server._get_engine", return_value=build_engine
        ):
            from agent_utilities.mcp.kg_server import _build_server

            _build_server()
    return mock_mcp.funcs


@pytest.fixture
def mock_engine():
    with patch("agent_utilities.mcp.kg_server._get_engine") as mock_get_engine:
        engine = MagicMock()
        engine.get_node.return_value = None
        engine.query_cypher.return_value = []
        engine.submit_task.return_value = "job-mock123"
        engine.get_task_status.return_value = None
        engine.list_tasks.return_value = {
            "running": [],
            "pending": [],
            "completed": [],
            "failed": [],
        }
        engine.clear_completed_tasks.return_value = {
            "status": "success",
            "cleared": 1,
            "remaining": 0,
        }
        mock_get_engine.return_value = engine
        yield engine


@pytest.mark.asyncio
async def test_graphos_health_is_liveness_always_200_with_real_report(server_tools):
    """``/health`` is LIVENESS: always 200, and the body is the real shared
    health report (CONCEPT:AU-OS.deployment.liveness-vs-readiness-split) — no
    longer the unconditional ``{"status": "ok"}`` stub that never checked
    anything. Still non-fingerprinting: no raw endpoint/hostname strings, only
    counts/booleans/resolved-mode/platform-id detail.
    """
    response = await server_tools["health_check"](MagicMock())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = json.loads(response.body)
    assert body["status"] in ("healthy", "unhealthy")
    names = {c["name"] for c in body["checks"]}
    assert "engine" in names
    for check in body["checks"]:
        assert check["status"] in ("ok", "unhealthy", "not_configured")
        assert isinstance(check["latency_ms"], (int, float))


@pytest.mark.asyncio
async def test_graphos_health_ready_reflects_status_in_http_code(server_tools):
    """``/health/ready`` is READINESS: the same report, 200/503 mirrors it
    (CONCEPT:AU-OS.deployment.liveness-vs-readiness-split)."""
    response = await server_tools["readiness_check"](MagicMock())

    body = json.loads(response.body)
    assert response.status_code == (200 if body["status"] == "healthy" else 503)


# ── graph_ingest: ingestion ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_ingest_single_codebase(mock_engine, server_tools):
    """Test graph_ingest queues a single codebase (action=ingest)."""
    graph_ingest = server_tools["graph_ingest"]
    # A real codebase directory has no SKILL.md, so ContentType.classify must
    # resolve it to CODEBASE (→ async submit_task), not SKILL. Mock a directory
    # whose ``/ "SKILL.md"`` probe reports missing.
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.__truediv__") as mock_div:
            mock_joined = MagicMock()
            mock_joined.exists.return_value = False
            mock_div.return_value = mock_joined
            res_str = await graph_ingest(
                target_path="/fake/codebase",
                agent_id="test_agent",
                max_depth=3,
                action="ingest",
                job_id="",
                corpus_name="",
                base_path="",
                description="",
            )

    assert "job-mock123" in res_str
    _, kwargs = mock_engine.submit_task.call_args
    assert kwargs["task_type"] == "codebase"


@pytest.mark.asyncio
async def test_graph_ingest_bulk_json_array(mock_engine, server_tools):
    """Test graph_ingest parses and queues a JSON array of targets."""
    graph_ingest = server_tools["graph_ingest"]
    targets = ["/fake/repo1", "/fake/repo2"]
    with patch("pathlib.Path.exists", return_value=True):
        res_str = await graph_ingest(
            target_path=json.dumps(targets),
            agent_id="test_agent",
            max_depth=3,
            action="ingest",
            job_id="",
            corpus_name="",
            base_path="",
            description="",
        )

    assert "Submitted 2 jobs" in res_str


@pytest.mark.asyncio
async def test_graph_ingest_bulk_comma_separated(mock_engine, server_tools):
    """Test graph_ingest parses and queues a comma-separated string."""
    graph_ingest = server_tools["graph_ingest"]
    targets_str = "/fake/repo1,/fake/repo2,/fake/repo3"
    with patch("pathlib.Path.exists", side_effect=[False, True, True, True]):
        res_str = await graph_ingest(
            target_path=targets_str,
            agent_id="test_agent",
            max_depth=3,
            action="ingest",
            job_id="",
            corpus_name="",
            base_path="",
            description="",
        )

    assert "Submitted 3 jobs" in res_str


@pytest.mark.asyncio
async def test_graph_ingest_document_is_async_not_blocking(mock_engine, server_tools):
    """A .pdf/.md document must enqueue an async job, never run inline.

    Live-path guard for the param-minimization fix: a document path is
    auto-classified and routed through ``engine.submit_task`` (the durable
    queue) with ``task_type='document'`` — it must NOT call the synchronous
    ``IngestionEngine`` (the old footgun that blocked the caller for minutes).
    """
    graph_ingest = server_tools["graph_ingest"]
    res_str = await graph_ingest(
        target_path="/papers/2402.03300.pdf",
        agent_id="test_agent",
        action="ingest",
    )

    assert "job-mock123" in res_str
    mock_engine.submit_task.assert_called_once()
    _, kwargs = mock_engine.submit_task.call_args
    assert kwargs["task_type"] == "document"
    assert kwargs["is_codebase"] is False


@pytest.mark.asyncio
async def test_graph_ingest_explicit_document_content_type_still_async(
    mock_engine, server_tools
):
    """Even an explicit content_type='document' override must stay async.

    The whole point of the fix: passing content_type can no longer force the
    blocking synchronous path for a heavy (document/codebase) category.
    """
    graph_ingest = server_tools["graph_ingest"]
    res_str = await graph_ingest(
        target_path="/papers/some_paper.pdf",
        action="ingest",
        content_type="document",
    )

    assert "job-mock123" in res_str
    mock_engine.submit_task.assert_called_once()
    _, kwargs = mock_engine.submit_task.call_args
    assert kwargs["task_type"] == "document"


# ── graph_ingest: job management ─────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_ingest_jobs_list(mock_engine, server_tools):
    """Test listing jobs via graph_ingest action=jobs."""
    graph_ingest = server_tools["graph_ingest"]
    mock_engine.list_tasks.return_value = {
        "running": [{"job_id": "job-1", "target": "source-a"}],
        "pending": [],
        "completed": [],
        "failed": [],
    }

    res_str = await graph_ingest(
        target_path=".",
        action="jobs",
        max_depth=3,
        agent_id="",
        job_id="",
        corpus_name="",
        base_path="",
        description="",
    )
    assert "job-1" in res_str


@pytest.mark.asyncio
async def test_graph_ingest_job_status(mock_engine, server_tools):
    """Test getting status of a specific job via graph_ingest action=job_status."""
    graph_ingest = server_tools["graph_ingest"]
    mock_engine.get_task_status.return_value = {
        "status": "running",
        "metadata": {"target": "source-a"},
    }

    res_str = await graph_ingest(
        target_path=".",
        action="job_status",
        job_id="job-1",
        max_depth=3,
        agent_id="",
        corpus_name="",
        base_path="",
        description="",
    )
    assert "running" in res_str


@pytest.mark.asyncio
async def test_graph_ingest_job_status_not_found(mock_engine, server_tools):
    """Test job_status returns not found for missing job."""
    graph_ingest = server_tools["graph_ingest"]
    mock_engine.query_cypher.return_value = []

    res_str = await graph_ingest(
        target_path=".",
        action="job_status",
        job_id="job-99",
        max_depth=3,
        agent_id="",
        corpus_name="",
        base_path="",
        description="",
    )
    assert "not found" in res_str


# ── graph_query ──────────────────────────────────────────────────────


def test_graph_query_basic(mock_engine, server_tools):
    """Test graph_query returns the current typed EvidenceBundle contract."""
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    graph_query = server_tools["graph_query"]
    mock_engine.query_cypher.return_value = [{"type": "MemoryNode", "count": 50}]

    with patch(
        "agent_utilities.mcp.kg_server._resolve_read_engines",
        return_value=([("default", mock_engine)], {}, False),
    ):
        result = graph_query(
            cypher="MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count",
            params="{}",
            scope="local",
            reference_id="",
        )

    assert isinstance(result, EvidenceBundle)
    assert result.reasoning_trace[-1]["payload"]["rows"][0]["count"] == 50


def test_graph_query_surfaces_backend_write_rejection_as_typed_error(
    mock_engine, server_tools
):
    """The authoritative read-only backend rejects writes without leaking detail."""
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    graph_query = server_tools["graph_query"]
    mock_engine.query_cypher.side_effect = PermissionError("read-only query required")

    with patch(
        "agent_utilities.mcp.kg_server._resolve_read_engines",
        return_value=([("default", mock_engine)], {}, False),
    ):
        result = graph_query(
            cypher="CREATE (n:Test) RETURN n",
            params="{}",
            scope="local",
            reference_id="",
        )

    assert isinstance(result, EvidenceBundle)
    assert result.claims[0]["status"] == "failed"
    assert result.claims[0]["error"]["code"] == "operation_failed"
    assert result.next_actions


# ── graph_analyze: SAI factory specialize action (AHE-3.29) ──────────────


class _FakeSpecializeEngine:
    """Serves WorldModelTransition rows + records add_node (SaiFactoryCycle)."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, str, dict]] = []

    def query_cypher(self, _cypher: str, *_a, **_k):
        return [
            {
                "state": f"cell_{i:02d}",
                "action": act,
                "next_state": f"cell_{i:02d}_{act}",
            }
            for i in range(8)
            for act in ("north", "south", "east", "west")
        ]

    def add_node(self, node_id: str, label: str, properties=None):
        self.nodes.append((node_id, label, properties or {}))


@pytest.mark.asyncio
async def test_graph_analyze_specialize_action_is_gateway_reachable(
    server_tools, monkeypatch
):
    """The SAI factory is reachable through the focused evaluation tool."""
    from agent_utilities.mcp import kg_server
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    assert "graph_evaluate" in kg_server.REGISTERED_TOOLS
    eng = _FakeSpecializeEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: eng)

    out = await kg_server._execute_tool("graph_evaluate", action="specialize")

    assert isinstance(out, EvidenceBundle)
    assert out.claims[0]["status"] == "ok"
    assert "final_specialist_reward" in out.claims[0]
    assert out.claims[0]["transitions"] == 32
    # a queryable SaiFactoryCycle node was persisted by the live run
    assert any(label == "SaiFactoryCycle" for _, label, _ in eng.nodes)


@pytest.mark.asyncio
async def test_graph_analyze_specialize_noops_without_history(
    server_tools, monkeypatch
):
    from agent_utilities.mcp import kg_server
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    class _Empty:
        def query_cypher(self, *_a, **_k):
            return []

    monkeypatch.setattr(kg_server, "_get_engine", lambda: _Empty())
    out = await kg_server._execute_tool("graph_evaluate", action="specialize")
    assert isinstance(out, EvidenceBundle)
    assert out.claims[0]["status"] == "noop"
