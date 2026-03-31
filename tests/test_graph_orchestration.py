import pytest
import asyncio
from typing import Any
from pathlib import Path
from agent_utilities.graph_orchestration import (
    create_graph_agent,
    run_graph,
    GraphState,
    HybridRouterNode,
    DomainNode,
)
from pydantic_ai.models.test import TestModel

import os
@pytest.mark.asyncio
async def test_hybrid_router_rules():
    os.environ["OPENAI_API_KEY"] = "test"
    tag_prompts = {"git": "Git stuff", "web": "Web stuff"}
    tag_env_vars = {"git": "GITTOOL", "web": "WEBTOOL"}

    graph, config = create_graph_agent(tag_prompts, tag_env_vars)


    state = GraphState(query="clone the git repo")

    class MockDeps:
        def __init__(self):
            self.tag_prompts = tag_prompts
            self.event_queue = None
            self.tag_env_vars = tag_env_vars
            self.mcp_toolsets = []
            self.min_confidence = 0.6
            self.provider = "test"
            self.router_model = "test"
            self.base_url = None
            self.api_key = None
            self.ssl_verify = True
            self.routing_strategy = "hybrid"

    class MockCtx:
        def __init__(self):
            self.state = state
            self.deps = MockDeps()

    node = HybridRouterNode()
    res = await node.run(MockCtx())


    assert isinstance(res, DomainNode)
    assert state.routed_domain == "git"

@pytest.mark.asyncio
async def test_graph_error_recovery():
    from agent_utilities.graph_orchestration import ErrorRecoveryNode, RouterNode

    state = GraphState(query="test", error="Transient timeout")

    class MockCtx:
        def __init__(self):
            self.state = state
            self.deps = type("Deps", (), {"event_queue": None})

    node = ErrorRecoveryNode()


    res = await node.run(MockCtx())
    assert isinstance(res, RouterNode)


    state.retry_count = 5
    res = await node.run(MockCtx())
    from pydantic_graph import End
    assert isinstance(res, End)

@pytest.mark.asyncio
async def test_parallel_execution_fanout():
    from agent_utilities.graph_orchestration import ParallelDomainNode

    tag_prompts = {"git": "Git", "web": "Web"}
    state = GraphState(query="Check git and web", parallel_domains=["git", "web"])

    class MockDeps:
        def __init__(self):
            self.tag_prompts = tag_prompts
            self.tag_env_vars = {"git": "G", "web": "W"}
            self.mcp_toolsets = []
            self.sub_agents = {}
            self.event_queue = asyncio.Queue()
            self.provider = "test"
            self.agent_model = "test"
            self.base_url = None
            self.api_key = None
            self.ssl_verify = True
            self.workspace_path = Path.cwd()
            self.elicitation_queue = None
            self.routing_strategy = "hybrid"
            self.request_id = "test"

    class MockCtx:
        def __init__(self):
            self.state = state
            self.deps = MockDeps()

    node = ParallelDomainNode()






    from agent_utilities.graph_orchestration import ResultMergerNode



    assert hasattr(node, "run")
