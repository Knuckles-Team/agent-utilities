import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from agent_utilities.graph.builder import initialize_graph_from_workspace
from agent_utilities.graph.runner import run_graph
from agent_utilities.models import GraphPlan, ExecutionStep, GraphResponse
from agent_utilities.graph.graph_models import ValidationResult
from pydantic_ai import Agent

class MockStream:
    def __init__(self, data):
        self.data = data
        self.usage_val = MagicMock()
        self.usage_val.__await__ = lambda x: asyncio.sleep(0).__await__()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def get_output(self):
        return self.data
    
    def usage(self):
        return self.usage_val
    
    async def stream_text(self, delta=True):
        yield "chunk"

@pytest.mark.asyncio
async def test_full_graph_lifecycle():
    # 1. Initialize graph
    with patch("agent_utilities.graph.builder.get_discovery_registry") as mock_reg, \
         patch("agent_utilities.discovery.discover_all_specialists", return_value=[]), \
         patch("agent_utilities.mcp_agent_manager.should_sync", return_value=False), \
         patch("agent_utilities.workspace.resolve_mcp_config_path", return_value=None), \
         patch("agent_utilities.discovery.discover_agents", return_value={"researcher": {"description": "Research agent", "type": "prompt"}}):
        
        mock_reg.return_value.agents = []
        graph, config = initialize_graph_from_workspace()

    plan = GraphPlan(
        steps=[ExecutionStep(node_id="researcher", status="pending")],
        metadata={"reasoning": "Test plan"}
    )
    validation_ok = ValidationResult(is_valid=True, score=0.9, feedback="Perfect")
    
    def mock_run_stream(agent_self, prompt, **kwargs):
        if agent_self.output_type == GraphPlan:
            out = plan
            role = "ROUTER"
        elif agent_self.output_type == ValidationResult:
            out = validation_ok
            role = "VERIFIER"
        else:
            out = "Final answer."
            role = "SYNTHESIZER"

        return MockStream(out)

    async def mock_run_call(agent_self, prompt, **kwargs):
        system_prompt = str(agent_self.system_prompt).lower()
        if "policy" in system_prompt or "usage" in system_prompt:
            return MagicMock(output="PASS")
        if "available memories" in str(prompt).lower():
            res = MagicMock()
            res.output = {"selected_memories": []}
            return res
        return MagicMock(output="Research data.")

    with patch.object(Agent, "run", new=mock_run_call), \
         patch.object(Agent, "run_stream", new=mock_run_stream), \
         patch("agent_utilities.graph.runner.create_model", return_value=MagicMock()), \
         patch("agent_utilities.graph.steps.fetch_unified_context", return_value="context"):
        
        result = await run_graph(graph, config, "What is X?")
        assert result["status"] == "completed"

@pytest.mark.asyncio
async def test_graph_parallel_and_fallback():
    # 1. Initialize graph
    with patch("agent_utilities.graph.builder.get_discovery_registry") as mock_reg, \
         patch("agent_utilities.discovery.discover_all_specialists", return_value=[]), \
         patch("agent_utilities.mcp_agent_manager.should_sync", return_value=False), \
         patch("agent_utilities.workspace.resolve_mcp_config_path", return_value=None), \
         patch("agent_utilities.discovery.discover_agents", return_value={"researcher": {"description": "Research agent", "type": "prompt"}}):
        
        mock_reg.return_value.agents = []
        graph, config = initialize_graph_from_workspace()

    plan_a = GraphPlan(
        steps=[
            ExecutionStep(node_id="researcher", is_parallel=True),
            ExecutionStep(node_id="researcher", is_parallel=True)
        ],
        metadata={"reasoning": "Parallel research plan"}
    )
    plan_b = GraphPlan(
        steps=[ExecutionStep(node_id="researcher", is_parallel=False)],
        metadata={"reasoning": "Simple fallback plan"}
    )
    
    validation_fail = ValidationResult(is_valid=False, score=0.2, feedback="Fail")
    validation_ok = ValidationResult(is_valid=True, score=0.9, feedback="Pass")
    
    def mock_run_stream(agent_self, prompt, **kwargs):
        if not hasattr(test_graph_parallel_and_fallback, "verified_count"):
            test_graph_parallel_and_fallback.verified_count = 0
            test_graph_parallel_and_fallback.routed_count = 0

        if agent_self.output_type == GraphPlan:
            out = plan_a if test_graph_parallel_and_fallback.routed_count == 0 else plan_b
            test_graph_parallel_and_fallback.routed_count += 1
        elif agent_self.output_type == ValidationResult:
            out = validation_fail if test_graph_parallel_and_fallback.verified_count == 0 else validation_ok
            test_graph_parallel_and_fallback.verified_count += 1
        else:
            out = "Final answer."

        return MockStream(out)

    async def mock_run_call(agent_self, prompt, **kwargs):
        system_prompt = str(agent_self.system_prompt).lower()
        if "policy" in system_prompt or "usage" in system_prompt:
            return MagicMock(output="PASS")
        if "available memories" in str(prompt).lower():
            res = MagicMock()
            res.output = {"selected_memories": []}
            return res
        
        # Planner uses .run() and expects a RunResult-like object with .data or .output
        # In steps.py line 324: ctx.state.plan = res.data if hasattr(res, "data") else res.output
        if agent_self.output_type == GraphPlan:
            m = MagicMock()
            m.data = plan_b
            m.output = plan_b
            return m
            
        return MagicMock(output="Parallel research data.")

    with patch.object(Agent, "run", new=mock_run_call), \
         patch.object(Agent, "run_stream", new=mock_run_stream), \
         patch("agent_utilities.graph.runner.create_model", return_value=MagicMock()), \
         patch("agent_utilities.graph.steps.fetch_unified_context", return_value="context"):
        
        test_graph_parallel_and_fallback.verified_count = 0
        test_graph_parallel_and_fallback.routed_count = 0
        
        result = await run_graph(graph, config, "Complex query")
        
        assert result["status"] == "completed"
        assert test_graph_parallel_and_fallback.verified_count >= 2
