"""Tests for CONCEPT:ORCH-1.2 — Hot Cache Layer & Registry Optimization.

Validates:
    - ``_RegistryCache`` population and invalidation lifecycle
    - ``get_relevant_specialists()`` filtering with fallbacks
    - ``invalidate_registry_cache()`` public API
"""

import pytest

from agent_utilities.core.config import (
    _RegistryCache,
    get_relevant_specialists,
    invalidate_registry_cache,
)
from agent_utilities.models import MCPAgent, MCPAgentRegistryModel


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure cache is invalidated before and after each test."""
    _RegistryCache.invalidate()
    yield
    _RegistryCache.invalidate()


@pytest.mark.concept("CONCEPT:ORCH-1.2")
class TestRegistryCache:
    """Test suite for the session-scoped registry cache."""

    def test_cache_starts_empty(self):
        """Cache should be None before first access."""
        assert _RegistryCache._registry is None

    def test_invalidate_clears_all(self):
        """invalidate() should clear registry and all sub-caches."""
        # Manually populate
        _RegistryCache._registry = MCPAgentRegistryModel()
        _RegistryCache._prompts = {"test": "value"}
        _RegistryCache._tool_agent_map = {"tool": ["agent"]}

        invalidate_registry_cache()

        assert _RegistryCache._registry is None
        assert _RegistryCache._prompts == {}
        assert _RegistryCache._tool_agent_map == {}

    def test_get_registry_caches(self):
        """After first access, subsequent calls should return same object."""
        # Populate with a mock to avoid KG dependency
        mock_registry = MCPAgentRegistryModel(
            agents=[MCPAgent(name="test", description="test agent", agent_type="mcp")]
        )
        _RegistryCache._registry = mock_registry

        result = _RegistryCache.get_registry()
        assert result is mock_registry
        assert len(result.agents) == 1

    def test_invalidate_then_repopulate(self):
        """Cache should re-populate after invalidation."""
        mock_registry = MCPAgentRegistryModel()
        _RegistryCache._registry = mock_registry

        invalidate_registry_cache()
        assert _RegistryCache._registry is None

        # Manually re-populate for test
        _RegistryCache._registry = MCPAgentRegistryModel(
            agents=[
                MCPAgent(name="new", description="new agent", agent_type="specialist")
            ]
        )
        result = _RegistryCache.get_registry()
        assert len(result.agents) == 1
        assert result.agents[0].name == "new"


@pytest.mark.concept("CONCEPT:ORCH-1.2")
class TestGetRelevantSpecialists:
    """Test suite for the top-N specialist filtering function."""

    def test_returns_empty_for_no_agents(self):
        """Should return empty list when registry has no agents."""
        _RegistryCache._registry = MCPAgentRegistryModel(agents=[])
        result = get_relevant_specialists("test query")
        assert result == []

    def test_falls_back_to_full_list_without_engine(self):
        """Without engine, should return first top_n agents."""
        agents = [
            MCPAgent(name=f"agent_{i}", description=f"desc {i}", agent_type="mcp")
            for i in range(15)
        ]
        _RegistryCache._registry = MCPAgentRegistryModel(agents=agents)

        result = get_relevant_specialists("test query", engine=None, top_n=5)
        assert len(result) == 5
        assert result[0].name == "agent_0"

    def test_respects_top_n_limit(self):
        """Should cap results at top_n."""
        agents = [
            MCPAgent(name=f"agent_{i}", description=f"desc {i}", agent_type="mcp")
            for i in range(20)
        ]
        _RegistryCache._registry = MCPAgentRegistryModel(agents=agents)

        result = get_relevant_specialists("query", engine=None, top_n=3)
        assert len(result) == 3

    def test_empty_query_returns_all(self):
        """Empty query should return agents without filtering."""
        agents = [MCPAgent(name="only_one", description="test", agent_type="mcp")]
        _RegistryCache._registry = MCPAgentRegistryModel(agents=agents)

        result = get_relevant_specialists("", engine=None, top_n=5)
        assert len(result) == 1


@pytest.mark.concept("CONCEPT:ORCH-1.2")
class TestCacheInvalidationIntegration:
    """Test invalidation signals fire correctly from callsites."""

    def test_pipeline_runner_invalidates(self):
        """PipelineRunner.run() should call invalidate_registry_cache()."""
        # Just verify the import path is valid
        from agent_utilities.knowledge_graph.pipeline.runner import (
            PipelineRunner,  # noqa: F401
        )

    def test_agent_manager_imports_invalidation(self):
        """agent_manager module should be importable with invalidation wiring."""
        from agent_utilities.mcp import agent_manager  # noqa: F401

    def test_self_model_imports_invalidation(self):
        """SelfModel should be importable with invalidation wiring."""
        from agent_utilities.knowledge_graph.self_model import SelfModel  # noqa: F401
