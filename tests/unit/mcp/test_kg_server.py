"""Test suite for the Knowledge Graph MCP Server tools (graph-os).

CONCEPT:ECO-4.2 — KG MCP Server & Execution

Tests use the consolidated graph-os tool names: graph_query, graph_search,
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


# ── graph_ingest: ingestion ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_ingest_single_codebase(mock_engine, server_tools):
    """Test graph_ingest queues a single codebase (action=ingest)."""
    graph_ingest = server_tools["graph_ingest"]
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.iterdir", return_value=[MagicMock(name=".git")]):
                with patch("pathlib.Path.__truediv__") as mock_div:
                    mock_joined = MagicMock()
                    mock_joined.exists.return_value = True
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


# ── graph_ingest: job management ─────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_ingest_jobs_list(mock_engine, server_tools):
    """Test listing jobs via graph_ingest action=jobs."""
    graph_ingest = server_tools["graph_ingest"]
    mock_engine.query_cypher.return_value = [
        {"id": "job-1", "status": "running", "meta": json.dumps({"target": "/a"})},
    ]

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
    mock_engine.query_cypher.return_value = [
        {"status": "running", "meta": json.dumps({"target": "/a"})}
    ]

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
    """Test graph_query executes a Cypher query."""
    graph_query = server_tools["graph_query"]
    mock_engine.query_cypher.return_value = [{"type": "MemoryNode", "count": 50}]

    res_str = graph_query(
        cypher="MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count",
        params="{}",
        scope="local",
        reference_id="",
    )
    res = json.loads(res_str)
    assert isinstance(res, list)
    assert res[0]["count"] == 50


def test_graph_query_blocks_writes(mock_engine, server_tools):
    """Test graph_query blocks write operations."""
    graph_query = server_tools["graph_query"]

    res_str = graph_query(
        cypher="CREATE (n:Test) RETURN n", params="{}", scope="local", reference_id=""
    )
    res = json.loads(res_str)
    assert "error" in res
    assert "CREATE" in res["error"]
