import pytest
import json
from unittest.mock import AsyncMock, patch
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agent_utilities.mcp.kg_server import (
    # Tools/Toggle
    get_tools_endpoint,
    toggle_tool_endpoint,
    
    # Bilateral base endpoints
    graph_query_endpoint,
    graph_search_endpoint,
    graph_write_endpoint,
    graph_ingest_endpoint,
    graph_analyze_endpoint,
    graph_orchestrate_endpoint,
    graph_configure_endpoint,

    # Query
    graph_query_federated_endpoint,

    # Search
    graph_search_hybrid_endpoint,
    graph_search_concept_endpoint,
    graph_search_analogy_endpoint,
    graph_search_memory_endpoint,
    graph_search_discover_endpoint,
    graph_search_dci_endpoint,

    # Write
    graph_write_node_endpoint,
    graph_write_delete_node_endpoint,
    graph_write_edge_endpoint,
    graph_write_delete_edge_endpoint,
    graph_write_external_endpoint,
    graph_write_bulk_endpoint,
    graph_write_memory_endpoint,
    graph_write_memory_recall_endpoint,
    graph_write_chat_endpoint,
    graph_write_sdd_endpoint,
    graph_write_execution_endpoint,

    # Ingest
    graph_ingest_submit_endpoint,
    graph_ingest_corpus_endpoint,
    graph_ingest_jobs_endpoint,
    graph_ingest_job_status_endpoint,
    graph_ingest_rebuild_indexes_endpoint,
    graph_ingest_observe_endpoint,
    graph_ingest_materialize_endpoint,
    graph_ingest_sync_endpoint,
    graph_ingest_reflect_endpoint,
    graph_ingest_agent_toolkit_endpoint,
    graph_ingest_knowledge_pack_endpoint,

    # Analyze
    graph_analyze_synthesize_endpoint,
    graph_analyze_deep_extract_endpoint,
    graph_analyze_background_research_endpoint,
    graph_analyze_relevance_sweep_endpoint,
    graph_analyze_blast_radius_endpoint,
    graph_analyze_inspect_endpoint,
    graph_analyze_context_endpoint,
    graph_analyze_evaluate_alpha_endpoint,
    graph_analyze_evaluate_endpoint,
    graph_analyze_evolve_model_endpoint,
    graph_analyze_forecast_endpoint,
    graph_analyze_causal_endpoint,
    graph_analyze_invariant_endpoint,
    graph_analyze_security_scan_endpoint,

    # Orchestrate
    graph_orchestrate_dispatch_endpoint,
    graph_orchestrate_status_endpoint,
    graph_orchestrate_request_approval_endpoint,
    graph_orchestrate_grant_approval_endpoint,
    graph_orchestrate_execute_agent_endpoint,
    graph_orchestrate_consensus_endpoint,
    graph_orchestrate_start_debate_endpoint,
    graph_orchestrate_submit_risk_veto_endpoint,
    graph_orchestrate_list_cron_jobs_endpoint,
    graph_orchestrate_trigger_cron_job_endpoint,
    graph_orchestrate_compile_workflow_endpoint,
    graph_orchestrate_list_workflows_endpoint,
    graph_orchestrate_execute_workflow_endpoint,
    graph_orchestrate_dispatch_workflow_endpoint,
    graph_orchestrate_workflow_status_endpoint,
    graph_orchestrate_export_workflow_endpoint,

    # Configure
    graph_configure_secret_endpoint,
    graph_configure_register_mcp_endpoint,
    graph_configure_install_hooks_endpoint,
    graph_configure_uninstall_hooks_endpoint,
    graph_configure_doctor_endpoint,
)

@pytest.fixture
def test_app():
    app = Starlette()
    
    # Mount base and Tools endpoints
    app.add_route("/tools", get_tools_endpoint, methods=["GET"])
    app.add_route("/tools/toggle", toggle_tool_endpoint, methods=["POST"])
    
    app.add_route("/graph/query", graph_query_endpoint, methods=["POST"])
    app.add_route("/graph/search", graph_search_endpoint, methods=["POST"])
    app.add_route("/graph/write", graph_write_endpoint, methods=["POST"])
    app.add_route("/graph/ingest", graph_ingest_endpoint, methods=["POST"])
    app.add_route("/graph/analyze", graph_analyze_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate", graph_orchestrate_endpoint, methods=["POST"])
    app.add_route("/graph/configure", graph_configure_endpoint, methods=["POST"])

    # Granular Graph Query endpoints
    app.add_route("/graph/query/federated", graph_query_federated_endpoint, methods=["POST"])

    # Granular Graph Search endpoints
    app.add_route("/graph/search/hybrid", graph_search_hybrid_endpoint, methods=["POST"])
    app.add_route("/graph/search/concept", graph_search_concept_endpoint, methods=["POST"])
    app.add_route("/graph/search/analogy", graph_search_analogy_endpoint, methods=["POST"])
    app.add_route("/graph/search/memory", graph_search_memory_endpoint, methods=["POST"])
    app.add_route("/graph/search/discover", graph_search_discover_endpoint, methods=["POST"])
    app.add_route("/graph/search/dci", graph_search_dci_endpoint, methods=["POST"])

    # Granular Graph Write endpoints
    app.add_route("/graph/write/node", graph_write_node_endpoint, methods=["POST"])
    app.add_route("/graph/write/node/{node_id}", graph_write_delete_node_endpoint, methods=["DELETE"])
    app.add_route("/graph/write/edge", graph_write_edge_endpoint, methods=["POST"])
    app.add_route("/graph/write/edge", graph_write_delete_edge_endpoint, methods=["DELETE"])
    app.add_route("/graph/write/external", graph_write_external_endpoint, methods=["POST"])
    app.add_route("/graph/write/bulk", graph_write_bulk_endpoint, methods=["POST"])
    app.add_route("/graph/write/memory", graph_write_memory_endpoint, methods=["POST"])
    app.add_route("/graph/write/memory/recall", graph_write_memory_recall_endpoint, methods=["POST"])
    app.add_route("/graph/write/chat", graph_write_chat_endpoint, methods=["POST"])
    app.add_route("/graph/write/sdd", graph_write_sdd_endpoint, methods=["POST"])
    app.add_route("/graph/write/execution", graph_write_execution_endpoint, methods=["POST"])

    # Granular Graph Ingest endpoints
    app.add_route("/graph/ingest/submit", graph_ingest_submit_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/corpus", graph_ingest_corpus_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/jobs", graph_ingest_jobs_endpoint, methods=["GET"])
    app.add_route("/graph/ingest/job/{job_id}", graph_ingest_job_status_endpoint, methods=["GET"])
    app.add_route("/graph/ingest/rebuild-indexes", graph_ingest_rebuild_indexes_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/observe", graph_ingest_observe_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/materialize", graph_ingest_materialize_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/sync", graph_ingest_sync_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/reflect", graph_ingest_reflect_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/agent-toolkit", graph_ingest_agent_toolkit_endpoint, methods=["POST"])
    app.add_route("/graph/ingest/knowledge-pack", graph_ingest_knowledge_pack_endpoint, methods=["POST"])

    # Granular Graph Analyze endpoints
    app.add_route("/graph/analyze/synthesize", graph_analyze_synthesize_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/deep-extract", graph_analyze_deep_extract_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/background-research", graph_analyze_background_research_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/relevance-sweep", graph_analyze_relevance_sweep_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/blast-radius", graph_analyze_blast_radius_endpoint, methods=["GET"])
    app.add_route("/graph/analyze/inspect", graph_analyze_inspect_endpoint, methods=["GET"])
    app.add_route("/graph/analyze/context", graph_analyze_context_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/evaluate-alpha", graph_analyze_evaluate_alpha_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/evaluate", graph_analyze_evaluate_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/evolve-model", graph_analyze_evolve_model_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/forecast", graph_analyze_forecast_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/causal", graph_analyze_causal_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/invariant", graph_analyze_invariant_endpoint, methods=["POST"])
    app.add_route("/graph/analyze/security-scan", graph_analyze_security_scan_endpoint, methods=["POST"])

    # Granular Graph Orchestrate endpoints
    app.add_route("/graph/orchestrate/dispatch", graph_orchestrate_dispatch_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/job/{job_id}", graph_orchestrate_status_endpoint, methods=["GET"])
    app.add_route("/graph/orchestrate/request-approval", graph_orchestrate_request_approval_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/grant-approval", graph_orchestrate_grant_approval_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/execute-agent", graph_orchestrate_execute_agent_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/consensus", graph_orchestrate_consensus_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/start-debate", graph_orchestrate_start_debate_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/submit-risk-veto", graph_orchestrate_submit_risk_veto_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/cron-jobs", graph_orchestrate_list_cron_jobs_endpoint, methods=["GET"])
    app.add_route("/graph/orchestrate/trigger-cron-job", graph_orchestrate_trigger_cron_job_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/compile-workflow", graph_orchestrate_compile_workflow_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/workflows", graph_orchestrate_list_workflows_endpoint, methods=["GET"])
    app.add_route("/graph/orchestrate/execute-workflow", graph_orchestrate_execute_workflow_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/dispatch-workflow", graph_orchestrate_dispatch_workflow_endpoint, methods=["POST"])
    app.add_route("/graph/orchestrate/workflow-status/{job_id}", graph_orchestrate_workflow_status_endpoint, methods=["GET"])
    app.add_route("/graph/orchestrate/export-workflow", graph_orchestrate_export_workflow_endpoint, methods=["POST"])

    # Granular Graph Configure endpoints
    app.add_route("/graph/configure/secret", graph_configure_secret_endpoint, methods=["POST"])
    app.add_route("/graph/configure/register-mcp", graph_configure_register_mcp_endpoint, methods=["POST"])
    app.add_route("/graph/configure/install-hooks", graph_configure_install_hooks_endpoint, methods=["POST"])
    app.add_route("/graph/configure/uninstall-hooks", graph_configure_uninstall_hooks_endpoint, methods=["POST"])
    app.add_route("/graph/configure/doctor", graph_configure_doctor_endpoint, methods=["POST"])

    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_base_endpoints(mock_execute_tool, client):
    # Setup mock return value
    mock_execute_tool.return_value = {"status": "mocked_success"}

    # 1. POST /graph/query
    res = client.post("/graph/query", json={"cypher": "MATCH (n) RETURN n"})
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "mocked_success"}}
    mock_execute_tool.assert_called_with("graph_query", cypher="MATCH (n) RETURN n")

    # 2. POST /graph/search
    res = client.post("/graph/search", json={"query": "test query"})
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "mocked_success"}}
    mock_execute_tool.assert_called_with("graph_search", query="test query")

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_granular_query_endpoints(mock_execute_tool, client):
    mock_execute_tool.return_value = {"status": "mocked_success"}

    # POST /graph/query/federated
    res = client.post("/graph/query/federated", json={
        "cypher": "MATCH (n) RETURN n",
        "params": {"id": 123},
        "reference_id": "ref-456",
        "scope": "federated"
    })
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "mocked_success"}}
    mock_execute_tool.assert_called_with(
        "graph_query",
        cypher="MATCH (n) RETURN n",
        params='{"id": 123}',
        reference_id="ref-456",
        scope="federated"
    )

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_granular_search_endpoints(mock_execute_tool, client):
    mock_execute_tool.return_value = [{"id": "node-1"}]

    modes = ["hybrid", "concept", "analogy", "memory", "discover", "dci"]
    for mode in modes:
        res = client.post(f"/graph/search/{mode}", json={"query": "agentic workflow", "top_k": 5})
        assert res.status_code == 200
        assert res.json() == {"status": "success", "result": [{"id": "node-1"}]}
        if mode == "discover":
            mock_execute_tool.assert_called_with(
                "graph_search",
                query="agentic workflow",
                mode="discover"
            )
        else:
            mock_execute_tool.assert_called_with(
                "graph_search",
                query="agentic workflow",
                mode=mode,
                top_k=5
            )

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_granular_write_endpoints(mock_execute_tool, client):
    mock_execute_tool.return_value = {"status": "write_ok"}

    # 1. POST /graph/write/node
    res = client.post("/graph/write/node", json={
        "node_id": "agent-1",
        "node_type": "Agent",
        "properties": {"name": "Test Agent"}
    })
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "write_ok"}}
    mock_execute_tool.assert_called_with(
        "graph_write",
        action="add_node",
        node_id="agent-1",
        node_type="Agent",
        properties='{"name": "Test Agent"}'
    )

    # 2. DELETE /graph/write/node/{node_id}
    res = client.delete("/graph/write/node/agent-1")
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "write_ok"}}
    mock_execute_tool.assert_called_with(
        "graph_write",
        action="delete_node",
        node_id="agent-1"
    )

    # 3. POST /graph/write/edge
    res = client.post("/graph/write/edge", json={
        "source_id": "agent-1",
        "target_id": "skill-1",
        "rel_type": "HAS_SKILL",
        "properties": {"level": "expert"}
    })
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "write_ok"}}
    mock_execute_tool.assert_called_with(
        "graph_write",
        action="add_edge",
        source_id="agent-1",
        target_id="skill-1",
        rel_type="HAS_SKILL",
        properties='{"level": "expert"}'
    )

    # 4. DELETE /graph/write/edge
    res = client.request("DELETE", "/graph/write/edge", json={
        "source_id": "agent-1",
        "target_id": "skill-1",
        "rel_type": "HAS_SKILL"
    })
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "write_ok"}}
    mock_execute_tool.assert_called_with(
        "graph_write",
        action="delete_edge",
        source_id="agent-1",
        target_id="skill-1",
        rel_type="HAS_SKILL"
    )

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_granular_ingest_endpoints(mock_execute_tool, client):
    mock_execute_tool.return_value = {"job_id": "job-100"}

    # 1. POST /graph/ingest/submit
    res = client.post("/graph/ingest/submit", json={
        "target_path": "/workspace/my-repo",
        "max_depth": 5
    })
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"job_id": "job-100"}}
    mock_execute_tool.assert_called_with(
        "graph_ingest",
        action="ingest",
        target_path="/workspace/my-repo",
        max_depth=5,
        agent_id=""
    )

    # 2. GET /graph/ingest/job/{job_id}
    res = client.get("/graph/ingest/job/job-100")
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"job_id": "job-100"}}
    mock_execute_tool.assert_called_with(
        "graph_ingest",
        action="job_status",
        job_id="job-100"
    )

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_granular_analyze_endpoints(mock_execute_tool, client):
    mock_execute_tool.return_value = {"analysis": "complete"}

    # 1. POST /graph/analyze/synthesize
    res = client.post("/graph/analyze/synthesize", json={"query": "security anomalies"})
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"analysis": "complete"}}
    mock_execute_tool.assert_called_with(
        "graph_analyze",
        action="synthesize",
        query="security anomalies",
        top_k=10
    )

    # 2. GET /graph/analyze/blast-radius
    res = client.get("/graph/analyze/blast-radius?node_id=agent-1&depth=3")
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"analysis": "complete"}}
    mock_execute_tool.assert_called_with(
        "graph_analyze",
        action="blast_radius",
        node_id="agent-1",
        depth=3
    )

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_granular_orchestrate_endpoints(mock_execute_tool, client):
    mock_execute_tool.return_value = {"status": "orchestrated"}

    # 1. POST /graph/orchestrate/dispatch
    res = client.post("/graph/orchestrate/dispatch", json={
        "task": "Perform workspace audit",
        "dependencies": ["job-1", "job-2"]
    })
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "orchestrated"}}
    mock_execute_tool.assert_called_with(
        "graph_orchestrate",
        action="dispatch",
        task="Perform workspace audit",
        dependencies='["job-1", "job-2"]'
    )

    # 2. GET /graph/orchestrate/job/dispatch-1
    res = client.get("/graph/orchestrate/job/dispatch-1")
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"status": "orchestrated"}}
    mock_execute_tool.assert_called_with(
        "graph_orchestrate",
        action="status",
        job_id="dispatch-1"
    )

@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_granular_configure_endpoints(mock_execute_tool, client):
    mock_execute_tool.return_value = {"vault": "updated"}

    # POST /graph/configure/secret
    res = client.post("/graph/configure/secret", json={
        "config_key": "github_pat",
        "config_value": "ghp_123"
    })
    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"vault": "updated"}}
    mock_execute_tool.assert_called_with(
        "graph_configure",
        action="set_secret",
        config_key="github_pat",
        config_value="ghp_123"
    )
