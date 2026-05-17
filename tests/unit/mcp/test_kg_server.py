"""Test suite for the Knowledge Graph MCP Server tools."""
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
    with patch("agent_utilities.mcp.server_factory.create_mcp_server", return_value=(None, mock_mcp, [])):
        with patch("agent_utilities.mcp.kg_server._get_engine", return_value=build_engine):
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
        engine.list_tasks.return_value = {"running": [], "pending": [], "completed": [], "failed": []}
        engine.clear_completed_tasks.return_value = {"status": "success", "cleared": 1, "remaining": 0}
        mock_get_engine.return_value = engine
        yield engine

@pytest.mark.asyncio
async def test_kg_ingest_single_codebase(mock_engine, server_tools):
    """Test kg_ingest queues a single codebase successfully."""
    kg_ingest = server_tools["kg_ingest"]
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=True):
            # Mock it being a codebase (has .git)
            with patch("pathlib.Path.iterdir", return_value=[MagicMock(name=".git")]):
                with patch("pathlib.Path.__truediv__") as mock_div:
                    mock_joined = MagicMock()
                    mock_joined.exists.return_value = True
                    mock_div.return_value = mock_joined
                    res_str = await kg_ingest("/fake/codebase", agent_id="test_agent")

    res = json.loads(res_str)
    assert res["status"] == "queuing_in_background"


@pytest.mark.asyncio
async def test_kg_ingest_bulk_json_array(mock_engine, server_tools):
    """Test kg_ingest parses and queues a JSON array of targets."""
    kg_ingest = server_tools["kg_ingest"]
    targets = ["/fake/repo1", "/fake/repo2"]
    with patch("pathlib.Path.exists", return_value=True):
        res_str = await kg_ingest(json.dumps(targets), agent_id="test_agent")

    res = json.loads(res_str)
    assert res["status"] == "queuing_in_background"


@pytest.mark.asyncio
async def test_kg_ingest_bulk_comma_separated(mock_engine, server_tools):
    """Test kg_ingest parses and queues a comma-separated string of targets."""
    kg_ingest = server_tools["kg_ingest"]
    targets_str = "/fake/repo1,/fake/repo2,/fake/repo3"
    with patch("pathlib.Path.exists", side_effect=[False, True, True, True]):
        # first false is for the entire string existence check
        res_str = await kg_ingest(targets_str, agent_id="test_agent")

    res = json.loads(res_str)
    assert res["status"] == "queuing_in_background"

def test_kg_jobs_list(mock_engine, server_tools):
    """Test listing all jobs."""
    kg_jobs = server_tools["kg_jobs"]
    mock_engine.list_tasks.return_value = {
        "pending": [{"job_id": "job-1", "target": "a"}],
        "completed": [{"job_id": "job-2", "target": "b"}]
    }

    res_str = kg_jobs(action="list")
    res = json.loads(res_str)
    assert "pending" in res
    assert len(res["pending"]) == 1
    assert res["pending"][0]["job_id"] == "job-1"


def test_kg_jobs_status(mock_engine, server_tools):
    """Test getting status of a specific job."""
    kg_jobs = server_tools["kg_jobs"]
    mock_engine.get_task_status.return_value = {
        "status": "running",
        "metadata": {"target": "a", "type": "codebase", "submitted_at": "now"}
    }

    res_str = kg_jobs(action="status", job_id="job-1")
    res = json.loads(res_str)
    assert res["status"] == "running"

    mock_engine.get_task_status.return_value = None
    res_str_not_found = kg_jobs(action="status", job_id="job-99")
    res_not_found = json.loads(res_str_not_found)
    assert "error" in res_not_found


def test_kg_jobs_remove(mock_engine, server_tools):
    """Test removing a completed job."""
    kg_jobs = server_tools["kg_jobs"]
    mock_engine.get_task_status.return_value = {"status": "completed", "metadata": {}}

    res_str = kg_jobs(action="remove", job_id="job-completed")
    res = json.loads(res_str)
    assert res["status"] == "success"
    mock_engine.remove_task.assert_called_once()

    mock_engine.get_task_status.return_value = {"status": "running", "metadata": {}}
    res_str_fail = kg_jobs(action="remove", job_id="job-running")
    res_fail = json.loads(res_str_fail)
    assert "error" in res_fail["results"][0]


def test_kg_jobs_clear(mock_engine, server_tools):
    """Test clearing completed/failed jobs."""
    kg_jobs = server_tools["kg_jobs"]
    res_str = kg_jobs(action="clear")
    res = json.loads(res_str)
    assert res["status"] == "success"
    mock_engine.clear_completed_tasks.assert_called_once()


def test_kg_inspect_stats(mock_engine, server_tools):
    """Test getting graph statistics."""
    kg_inspect = server_tools["kg_inspect"]
    mock_engine.query_cypher.side_effect = [
        [{"type": "MemoryNode", "count": 50}],  # node_types
        [{"total": 100}],  # total_nodes
        [{"total": 250}],  # total_edges
    ]

    res_str = kg_inspect(view="stats")
    res = json.loads(res_str)

    assert res["total_nodes"] == 100
    assert res["total_edges"] == 250
