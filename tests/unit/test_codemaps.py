import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent_utilities.knowledge_graph.codemaps import CodemapGenerator
from agent_utilities.models.codemap import CodemapArtifact, HierarchicalSection, CodemapNode

@pytest.fixture
def mock_kg():
    kg = MagicMock()
    kg.repo_root = "/mock/root"
    kg.extract_focused_subgraph = AsyncMock()
    kg.store_codemap = AsyncMock()

    # Setup mock subgraph
    subgraph = MagicMock()
    subgraph.nodes = [
        {"id": "node1", "label": "node1", "type": "file", "file": "f1.py", "centrality": 0.5},
        {"id": "node2", "label": "node2", "type": "function", "file": "f1.py", "centrality": 0.8},
    ]
    subgraph.edges = [{"source": "node1", "target": "node2", "type": "contains"}]
    kg.extract_focused_subgraph.return_value = subgraph
    return kg

@pytest.mark.asyncio
async def test_codemap_generator_create(mock_kg):
    # We need to mock create_model and the Agent.run
    with patch("agent_utilities.knowledge_graph.codemaps.create_model") as mock_create_model:
        with patch("agent_utilities.knowledge_graph.codemaps.Agent") as mock_agent_class:
            mock_agent = mock_agent_class.return_value
            mock_agent.run = AsyncMock()

            # Setup mock LLM result
            mock_hierarchy = [
                HierarchicalSection(
                    title="Section 1",
                    nodes=[
                        CodemapNode(id="node1", label="node1", type="file", file="f1.py", line=1, end_line=10, description="D1", importance=0.5),
                        CodemapNode(id="node2", label="node2", type="function", file="f1.py", line=5, end_line=8, description="D2", importance=0.8)
                    ]
                )
            ]
            result = MagicMock()
            result.data = mock_hierarchy
            mock_agent.run.return_value = result

            generator = CodemapGenerator(mock_kg)
            artifact = await generator.create(prompt="test prompt", mode="fast")

            assert isinstance(artifact, CodemapArtifact)
            assert artifact.prompt == "test prompt"
            assert len(artifact.nodes) == 2
            assert len(artifact.hierarchy) == 1
            assert artifact.hierarchy[0].title == "Section 1"

            mock_kg.extract_focused_subgraph.assert_called_once()
            mock_kg.store_codemap.assert_called_once_with(artifact)
