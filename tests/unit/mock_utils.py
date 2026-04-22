import asyncio
from unittest.mock import MagicMock, AsyncMock

class MockModel:
    def __init__(self):
        self.run = AsyncMock()
        self.run_model = AsyncMock()

def mock_create_model(*args, **kwargs):
    return MockModel()

class MockGraph:
    def __init__(self):
        self.run = AsyncMock()

    def mermaid_code(self):
        return "graph TD\n  A --> B"

class MockEngine:
    def __init__(self):
        self.backend = MagicMock()
        self.backend.execute = MagicMock(return_value=[])

    @classmethod
    def get_active(cls):
        return cls()

def get_mock_deps():
    from agent_utilities.graph.state import GraphDeps
    deps = MagicMock(spec=GraphDeps)
    deps.event_queue = asyncio.Queue()
    deps.tag_prompts = {}
    deps.mcp_toolsets = []
    deps.router_model = MockModel()
    deps.agent_model = MockModel()
    return deps
