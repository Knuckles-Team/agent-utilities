import pytest
import asyncio
import os
from typing import Any
from pathlib import Path
from agent_utilities.graph_orchestration import (
    create_graph_agent,
    run_graph,
    GraphState,
    GraphDeps,
    ValidatorNode,
    ValidationResult,
)
from pydantic_graph import BaseNode, End
from pydantic_ai.models.test import TestModel
from dataclasses import dataclass

@dataclass
class MockNode(BaseNode[GraphState, Any, Any]):
    async def run(self, ctx):
        return End("done")

@pytest.mark.asyncio
async def test_file_persistence(tmp_path):
    """Test graph state persistence using pydantic-graph's built-in FileStatePersistence."""
    from pydantic_graph.persistence.file import FileStatePersistence

    json_path = tmp_path / "test_run.json"
    persistence = FileStatePersistence(json_file=json_path)
    state = GraphState(query="test", session_id="session123")

                                                                        
    from pydantic_graph.persistence._utils import set_nodes_type_context
    with set_nodes_type_context([MockNode]):
        persistence.set_types(GraphState, dict)

    node = MockNode()
    await persistence.snapshot_node(state, node)

                  
    history = await persistence.load_all()
    assert len(history) == 1
    assert history[0].state.query == "test"
    assert history[0].state.session_id == "session123"

@pytest.mark.asyncio
async def test_validator_node_llm_logic():
                                
    val_model = TestModel()
    
                                                        
    import pydantic_ai.agent as pydantic_ai_agent
    original_agent_run = pydantic_ai_agent.Agent.run
    
    async def mock_agent_run(self, *args, **kwargs):
                                                                    
        class MockRunRes:
            def __init__(self, output):
                self.output = output
        
                                                   
        if "Failure Case" in state.query:
             return MockRunRes(ValidationResult(is_valid=False, score=0.2, feedback="Too short"))
        return MockRunRes(ValidationResult(is_valid=True, score=0.9, feedback="Great job"))
    
    pydantic_ai_agent.Agent.run = mock_agent_run
    
    tag_prompts = {"test": "Test"}
    state = GraphState(query="What is 2+2?", routed_domain="test")
    state.results["test"] = "4"
    
    class MockDeps:
        def __init__(self, model):
            self.tag_prompts = tag_prompts
            self.provider = "test"
            self.router_model = "test"
            self.base_url = None
            self.api_key = None
            self.enable_llm_validation = True
            self.event_queue = None
            
                                                          
    import agent_utilities.graph_orchestration as go
    original_create_model = go.create_model
    go.create_model = lambda **kwargs: val_model
    
    try:
        node = ValidatorNode()
        class MockCtx:
            def __init__(self):
                self.state = state
                self.deps = MockDeps(val_model)
        
        from pydantic_graph import End
        res = await node.run(MockCtx())
        assert isinstance(res, End)
        
                               
        state.query = "Failure Case: Test"
        state.retry_count = 0
        res = await node.run(MockCtx())
        from agent_utilities.graph_orchestration import DomainNode
        assert isinstance(res, DomainNode)
        assert state.retry_count == 1
        assert state.validation_feedback == "Too short"
    finally:
        go.create_model = original_create_model
        pydantic_ai_agent.Agent.run = original_agent_run

@pytest.mark.asyncio
async def test_parallel_result_merging():
    from agent_utilities.graph_orchestration import ResultMergerNode
    import json
    
    state = GraphState(query="test")
    state.results["domain1"] = '{"key": "value"}'
    state.results["domain2"] = "plain text"
    
    class MockCtx:
        def __init__(self):
            self.state = state
            self.deps = None
            
    node = ResultMergerNode()
    await node.run(MockCtx())
    
    combined = state.results["combined_summary"]
    assert isinstance(combined, dict)
    assert combined["domain1"] == {"key": "value"}
    assert combined["domain2"] == "plain text"
